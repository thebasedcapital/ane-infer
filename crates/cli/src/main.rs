//! ane-infer: Hybrid ANE+CPU LLM inference engine
//!
//! Prefills on Apple Neural Engine at ~19 TFLOPS, decodes on CPU/AMX/SME.
//! Beats MLX on TTFT at a fraction of the power draw.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};

use ane_engine::gpu_decode;
use ane_engine::metal_graph::GpuContext;

// Accelerate BLAS (routes to AMX/SME on Apple Silicon)
unsafe extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
}

fn blas_gemv(w: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    unsafe {
        cblas_sgemv(
            101,
            111,
            m as i32,
            n as i32,
            1.0,
            w.as_ptr(),
            n as i32,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "test-ane" => cmd_test_ane(),
        "bench-ane" => cmd_bench_ane(),
        "bench" => cmd_bench(&args[2..]),
        "generate" => cmd_generate(&args[2..]),
        "info" => cmd_info(&args[2..]),
        "--help" | "-h" | "help" => {
            print_usage();
            Ok(())
        }
        _ => {
            // Assume it's --model shorthand
            if args[1] == "--model" || args[1] == "-m" {
                cmd_generate(&args[1..])
            } else {
                eprintln!("unknown command: {}", args[1]);
                print_usage();
                Ok(())
            }
        }
    }
}

fn print_usage() {
    eprintln!(
        "ane-infer — Hybrid ANE+CPU LLM Inference Engine

USAGE:
    ane-infer test-ane                          Verify ANE is accessible (1024x1024 matmul)
    ane-infer bench-ane                         Benchmark ANE throughput (raw matmul)
    ane-infer bench --model <gguf> [OPTIONS]    Full inference benchmark suite
    ane-infer generate --model <gguf> --prompt <text> [OPTIONS]
    ane-infer info --model <gguf>               Print model metadata

OPTIONS:
    --model, -m <path>      Path to GGUF model file
    --prompt, -p <text>     Input prompt
    --max-tokens <n>        Maximum tokens to generate (default: 256)
    --temp <f>              Temperature for sampling (default: 0.7)
    --top-p <f>             Top-p / nucleus sampling (default: 0.9)
"
    );
}

fn cmd_test_ane() -> Result<()> {
    eprintln!("Testing ANE access with 64x64 matmul...");

    // Generate a simple conv MIL: y = W @ x where W is [64, 64, 1, 1], x is [1, 64, 1, 16]
    let mil = mil_gen::mil_gen_conv(64, 64, 16);

    // Build weight blob: identity-like matrix
    let mut w = vec![0.0f32; 64 * 64];
    for i in 0..64 {
        w[i * 64 + i] = 1.0;
    } // identity
    let blob = ane_bridge::build_single_weight_blob(&w);

    let in_bytes = 64 * 16 * 4; // [1, 64, 1, 16] fp32
    let out_bytes = 64 * 16 * 4;

    eprintln!("  Compiling MIL to ANE...");
    let kernel = ane_bridge::AneKernel::compile(&mil, Some(&blob), &[in_bytes], &[out_bytes])
        .context("ANE compilation failed — is SIP disabled or are you running unsigned?")?;

    // Input: identity values in channel-first layout
    let mut input = vec![0f32; 64 * 16];
    for c in 0..64 {
        for s in 0..16 {
            input[c * 16 + s] = (c * 16 + s) as f32 * 0.01;
        }
    }

    eprintln!("  Running on ANE...");
    kernel.write_input_f32(0, &input);
    kernel.eval()?;
    let mut output = vec![0f32; 64 * 16];
    kernel.read_output_f32(0, &mut output);

    // With identity matrix, output should equal input
    let mut max_err = 0f32;
    for i in 0..output.len() {
        let err = (output[i] - input[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    eprintln!("  Max error vs identity: {max_err:.6}");
    if max_err < 0.01 {
        eprintln!("  ANE is working correctly!");
    } else {
        eprintln!("  WARNING: Large error, ANE output may be incorrect");
    }

    Ok(())
}

fn cmd_bench_ane() -> Result<()> {
    eprintln!("Benchmarking ANE throughput...");

    // 1024x1024 matmul, spatial=64
    let dim = 1024;
    let spatial = 64;
    let mil = mil_gen::mil_gen_conv(dim, dim, spatial);

    let mut w = vec![0.01f32; dim * dim];
    for i in 0..dim {
        w[i * dim + i] = 1.0;
    }
    let blob = ane_bridge::build_single_weight_blob(&w);

    let in_bytes = dim * spatial * 4;
    let out_bytes = dim * spatial * 4;

    let kernel = ane_bridge::AneKernel::compile(&mil, Some(&blob), &[in_bytes], &[out_bytes])
        .context("ANE compilation failed")?;

    let input = vec![0.01f32; dim * spatial];
    kernel.write_input_f32(0, &input);

    // Warmup
    for _ in 0..5 {
        kernel.eval()?;
    }

    // Benchmark
    let n_iters = 100;
    let start = Instant::now();
    for _ in 0..n_iters {
        kernel.eval()?;
    }
    let elapsed = start.elapsed();

    let ops_per_eval = 2 * dim * dim * spatial; // matmul FLOPs
    let total_ops = ops_per_eval as f64 * n_iters as f64;
    let tflops = total_ops / elapsed.as_secs_f64() / 1e12;
    let ms_per_eval = elapsed.as_secs_f64() * 1000.0 / n_iters as f64;

    eprintln!("  Matrix: {dim}x{dim}, spatial: {spatial}");
    eprintln!(
        "  {n_iters} iterations in {:.1}ms ({:.3}ms/iter)",
        elapsed.as_secs_f64() * 1000.0,
        ms_per_eval
    );
    eprintln!("  Throughput: {tflops:.2} TFLOPS (FP16)");

    // Multi-procedure test: 2 procedures in one model
    eprintln!("\nTesting multi-procedure model (2 procedures)...");
    let procs = vec![
        ("proc0".into(), dim, dim, spatial, 64u64),
        ("proc1".into(), dim, dim, spatial, 64u64), // same weights, different procedure
    ];
    let multi_mil = mil_gen::mil_gen_multi_procedure(&procs);
    let multi_kernel = ane_bridge::AneKernel::compile(&multi_mil, Some(&blob), &[in_bytes], &[out_bytes])
        .context("Multi-procedure compilation failed")?;

    let n_procs = multi_kernel.num_procedures();
    eprintln!("  Procedures detected: {n_procs}");

    multi_kernel.write_input_f32(0, &input);

    // Test each procedure
    for p in 0..n_procs.min(2) {
        multi_kernel.eval_procedure(p)?;
        let mut out = vec![0f32; dim * spatial];
        multi_kernel.read_output_f32(0, &mut out);
        let max_val = out.iter().cloned().fold(0f32, f32::max);
        eprintln!("  Procedure {p}: ok (max output={max_val:.4})");
    }

    // Benchmark sequential dispatch of both procedures
    for _ in 0..5 {
        multi_kernel.eval_procedure(0)?;
        multi_kernel.eval_procedure(1)?;
    }
    let start2 = Instant::now();
    for _ in 0..n_iters {
        multi_kernel.eval_procedure(0)?;
        multi_kernel.eval_procedure(1)?;
    }
    let elapsed2 = start2.elapsed();
    let ms_per_pair = elapsed2.as_secs_f64() * 1000.0 / n_iters as f64;
    eprintln!("  2-proc sequential: {ms_per_pair:.3}ms/pair ({:.3}ms/proc)", ms_per_pair / 2.0);

    Ok(())
}

/// Full inference benchmark suite — measures TTFT, decode tok/s, memory, and throughput.
fn cmd_bench(args: &[String]) -> Result<()> {
    let mut model_path: Option<PathBuf> = None;
    let mut n_prompt = 128usize;
    let mut n_gen = 128usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                model_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--prompt-tokens" | "-pp" => {
                n_prompt = args[i + 1].parse()?;
                i += 2;
            }
            "--gen-tokens" | "-tg" => {
                n_gen = args[i + 1].parse()?;
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    let model_path = model_path.ok_or_else(|| anyhow::anyhow!("--model is required"))?;

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║           ane-infer Benchmark Suite                         ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();

    // System info
    let model_name = model_path.file_name().unwrap_or_default().to_string_lossy();
    eprintln!("Model: {model_name}");
    eprintln!("Prompt tokens: {n_prompt}");
    eprintln!("Generation tokens: {n_gen}");
    eprintln!();

    // Load model
    eprintln!("Loading model...");
    let load_start = Instant::now();
    let file_data = std::fs::read(&model_path)
        .with_context(|| format!("failed to read {}", model_path.display()))?;
    let file_size = file_data.len();
    let gguf = ane_gguf::GgufFile::parse(&file_data)?;
    let arch = gguf.architecture().unwrap_or("unknown").to_string();

    if arch != "qwen35" {
        anyhow::bail!("bench currently only supports qwen35 architecture, got: {arch}");
    }

    let config = ane_engine::model::Qwen35Config::from_gguf(&gguf)?;
    let c = &config.base;

    // Build tokenizer
    let vocab = gguf
        .tokenizer_tokens()
        .ok_or_else(|| anyhow::anyhow!("no tokenizer"))?;
    let merges = gguf.tokenizer_merges();
    let eos_id = gguf.eos_token_id();
    let tokenizer = ane_engine::tokenizer::BpeTokenizer::from_gguf(vocab, merges, eos_id);

    // Load weights (Q8_0 direct)
    let model = load_qwen35_q8(&gguf, &file_data, &config)?;

    let load_time = load_start.elapsed();
    eprintln!(
        "  Loaded in {:.2}s ({:.0} MB/s)",
        load_time.as_secs_f64(),
        file_size as f64 / 1e6 / load_time.as_secs_f64()
    );
    eprintln!();

    // Tokenize prompt first (needed for ANE kernel compilation size)
    let prompt_text = "The quick brown fox jumps over the lazy dog. ".repeat(20);
    let mut prompt_tokens = tokenizer.encode(&prompt_text);
    prompt_tokens.truncate(n_prompt);
    if prompt_tokens.is_empty() {
        prompt_tokens.push(760);
    }
    let actual_pp = prompt_tokens.len();

    // === Compile ANE prefill kernels ===
    eprintln!("── Compiling ANE prefill kernels (pp{actual_pp}) ──");
    let ane_compile_start = Instant::now();
    let ane_kernels = match ane_engine::ane_prefill::compile_ane_prefill(&model, actual_pp) {
        Ok(k) => {
            eprintln!(
                "  Compiled in {:.1}s ({} layers)",
                ane_compile_start.elapsed().as_secs_f64(),
                k.len()
            );
            Some(k)
        }
        Err(e) => {
            eprintln!("  ANE compilation failed: {e}");
            eprintln!("  Falling back to CPU prefill");
            None
        }
    };
    eprintln!();

    // --- CPU Prefill ---
    eprintln!("── Benchmark: CPU Prefill (pp{n_prompt}) ──");
    let mut cache = ane_engine::deltanet_cache::HybridCache::new(&config);
    let prefill_start = Instant::now();
    let mut last_logits = vec![0.0f32; c.vocab_size];
    for &tok in &prompt_tokens {
        last_logits = qwen35_forward_token(&model, &mut cache, tok)?;
    }
    let prefill_time = prefill_start.elapsed();
    let pp_tok_s = actual_pp as f64 / prefill_time.as_secs_f64();
    eprintln!("  Prompt: {} tokens", actual_pp);
    eprintln!("  Time: {:.1}ms", prefill_time.as_secs_f64() * 1000.0);
    eprintln!("  Speed: {pp_tok_s:.1} tok/s (CPU pp{actual_pp})");
    eprintln!();

    // --- ANE Prefill ---
    if let Some(ref ane_k) = ane_kernels {
        eprintln!("── Benchmark: ANE Prefill (pp{n_prompt}) ──");
        let mut cache_ane = ane_engine::deltanet_cache::HybridCache::new(&config);
        let ane_prefill_start = Instant::now();
        let ane_logits = ane_prefill_tokens(&model, &mut cache_ane, &prompt_tokens, ane_k)?;
        let ane_prefill_time = ane_prefill_start.elapsed();
        let ane_pp_tok_s = actual_pp as f64 / ane_prefill_time.as_secs_f64();
        eprintln!("  Time: {:.1}ms", ane_prefill_time.as_secs_f64() * 1000.0);
        eprintln!("  Speed: {ane_pp_tok_s:.1} tok/s (ANE pp{actual_pp})");
        let speedup = ane_pp_tok_s / pp_tok_s;
        eprintln!("  Speedup vs CPU: {speedup:.1}x");
        // Use ANE logits for decode
        last_logits = ane_logits;
        // Use ANE cache for decode
        cache = cache_ane;
        eprintln!();
    }

    // === Benchmark 2: CPU Decode ===
    eprintln!("── Benchmark: CPU Decode (tg{n_gen}) ──");
    let mut generated = Vec::with_capacity(n_gen);
    let mut next_token = greedy_sample(&last_logits, 0.0);
    generated.push(next_token);

    let decode_start = Instant::now();
    for _ in 1..n_gen {
        last_logits = qwen35_forward_token(&model, &mut cache, next_token)?;
        next_token = greedy_sample(&last_logits, 0.0);
        if tokenizer.is_eos(next_token) {
            break;
        }
        generated.push(next_token);
    }
    let decode_time = decode_start.elapsed();
    let actual_tg = generated.len();
    let tg_tok_s = if actual_tg > 1 {
        (actual_tg - 1) as f64 / decode_time.as_secs_f64()
    } else {
        0.0
    };
    eprintln!("  Tokens: {actual_tg}");
    eprintln!("  Time: {:.1}ms", decode_time.as_secs_f64() * 1000.0);
    eprintln!("  Speed: {tg_tok_s:.1} tok/s (tg{actual_tg})");
    eprintln!();

    // === Benchmark 2b: Metal GPU Decode ===
    eprintln!("── Benchmark: Metal GPU Decode (tg{n_gen}) ──");
    match GpuContext::new() {
        Ok(gpu) => {
            eprintln!("  Uploading model weights to GPU...");
            let gpu_upload_start = Instant::now();
            let gpu_weights = gpu_decode::upload_model_weights(&gpu, &model);
            eprintln!(
                "  Uploaded in {:.1}ms",
                gpu_upload_start.elapsed().as_secs_f64() * 1000.0
            );

            let mut cache_gpu = ane_engine::deltanet_cache::HybridCache::new(&config);
            for &tok in &prompt_tokens {
                let _ = qwen35_forward_token(&model, &mut cache_gpu, tok)?;
            }

            let mut last_logits_gpu = vec![0.0f32; c.vocab_size];
            let first_token = greedy_sample(&last_logits, 0.0);
            last_logits_gpu = gpu_decode::qwen35_forward_token_gpu(
                &model,
                &mut cache_gpu,
                first_token,
                &gpu,
                &gpu_weights,
            )?;

            let mut gen_gpu = vec![first_token];
            let gpu_start = Instant::now();
            for _ in 1..n_gen {
                let next = greedy_sample(&last_logits_gpu, 0.0);
                if tokenizer.is_eos(next) {
                    break;
                }
                gen_gpu.push(next);
                last_logits_gpu = gpu_decode::qwen35_forward_token_gpu(
                    &model,
                    &mut cache_gpu,
                    next,
                    &gpu,
                    &gpu_weights,
                )?;
            }
            let gpu_time = gpu_start.elapsed();
            let gpu_tg = gen_gpu.len();
            let gpu_tok_s = if gpu_tg > 1 {
                (gpu_tg - 1) as f64 / gpu_time.as_secs_f64()
            } else {
                0.0
            };
            eprintln!("  Tokens: {gpu_tg}");
            eprintln!("  Time: {:.1}ms", gpu_time.as_secs_f64() * 1000.0);
            eprintln!("  Speed: {gpu_tok_s:.1} tok/s (GPU Graph tg{gpu_tg})");
            let cpu_speedup = gpu_tok_s / tg_tok_s;
            eprintln!("  Speedup vs CPU: {cpu_speedup:.1}x");
            eprintln!();
        }
        Err(e) => {
            eprintln!("  Metal GPU not available: {e}");
            eprintln!();
        }
    }

    // === Benchmark 3: Per-layer timing ===
    eprintln!("── Benchmark: Per-layer decode ──");
    // Reset cache and time a single token through each layer
    let mut cache2 = ane_engine::deltanet_cache::HybridCache::new(&config);
    // Feed one warmup token
    let _ = qwen35_forward_token(&model, &mut cache2, 760)?;

    let single_start = Instant::now();
    let _ = qwen35_forward_token(&model, &mut cache2, 6511)?;
    let single_time = single_start.elapsed();
    let ms_per_token = single_time.as_secs_f64() * 1000.0;
    let ms_per_layer = ms_per_token / c.n_layers as f64;
    eprintln!("  Single token: {ms_per_token:.1}ms ({ms_per_layer:.2}ms/layer)");
    eprintln!();

    // === Summary ===
    let total_params_b = file_size as f64 / 1e9;
    let output_text = tokenizer.decode(&generated);

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║                    RESULTS SUMMARY                          ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║ Model:        {:<45}║", model_name);
    eprintln!(
        "║ Architecture: {:<45}║",
        format!("{arch} ({}L, {}d)", c.n_layers, c.dim)
    );
    eprintln!(
        "║ Params:       {:<45}║",
        format!("{total_params_b:.1}B (Q8_0)")
    );
    eprintln!("║ Backend:      {:<45}║", "CPU (Accelerate BLAS → AMX)");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║ Load time:    {:<45}║",
        format!("{:.2}s", load_time.as_secs_f64())
    );
    eprintln!(
        "║ pp{:<4}         {:<45}║",
        actual_pp,
        format!("{pp_tok_s:.1} tok/s")
    );
    eprintln!(
        "║ tg{:<4}         {:<45}║",
        actual_tg,
        format!("{tg_tok_s:.1} tok/s")
    );
    eprintln!(
        "║ TTFT:         {:<45}║",
        format!(
            "{:.1}ms (pp{})",
            prefill_time.as_secs_f64() * 1000.0,
            actual_pp
        )
    );
    eprintln!(
        "║ Per-layer:    {:<45}║",
        format!("{ms_per_layer:.2}ms/layer")
    );
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║ Output preview:                                             ║");
    let preview = if output_text.len() > 55 {
        &output_text[..55]
    } else {
        &output_text
    };
    eprintln!("║   {:<57}║", preview.replace('\n', "\\n"));
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // llama.cpp compatible format for easy comparison
    eprintln!();
    eprintln!("llama.cpp compatible format:");
    eprintln!("  llama_perf_sampler_print: n_sample = {actual_tg}");
    eprintln!(
        "  llama_perf_context_print: load time = {:.2} ms",
        load_time.as_secs_f64() * 1000.0
    );
    eprintln!("  llama_perf_context_print: prompt eval time = {:.2} ms / {} tokens ({:.2} ms per token, {pp_tok_s:.2} tokens per second)",
        prefill_time.as_secs_f64() * 1000.0, actual_pp, prefill_time.as_secs_f64() * 1000.0 / actual_pp as f64);
    eprintln!("  llama_perf_context_print:        eval time = {:.2} ms / {} runs ({:.2} ms per token, {tg_tok_s:.2} tokens per second)",
        decode_time.as_secs_f64() * 1000.0, actual_tg, decode_time.as_secs_f64() * 1000.0 / actual_tg.max(1) as f64);
    let total = prefill_time + decode_time;
    eprintln!(
        "  llama_perf_context_print:       total time = {:.2} ms / {} tokens",
        total.as_secs_f64() * 1000.0,
        actual_pp + actual_tg
    );

    Ok(())
}

fn cmd_generate(args: &[String]) -> Result<()> {
    let mut model_path: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut max_tokens = 256usize;
    let mut temperature = 0.7f32;
    let mut _top_p = 0.9f32;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                model_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--prompt" | "-p" => {
                prompt = Some(args[i + 1].clone());
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = args[i + 1].parse()?;
                i += 2;
            }
            "--temp" => {
                temperature = args[i + 1].parse()?;
                i += 2;
            }
            "--top-p" => {
                _top_p = args[i + 1].parse()?;
                i += 2;
            }
            _ => {
                eprintln!("unknown arg: {}", args[i]);
                i += 1;
            }
        }
    }

    let model_path = model_path.ok_or_else(|| anyhow::anyhow!("--model is required"))?;
    let prompt = prompt.ok_or_else(|| anyhow::anyhow!("--prompt is required"))?;

    eprintln!("Loading model from {}", model_path.display());
    let file_data = std::fs::read(&model_path)
        .with_context(|| format!("failed to read {}", model_path.display()))?;

    let gguf = ane_gguf::GgufFile::parse(&file_data)?;
    let arch = gguf.architecture().unwrap_or("unknown").to_string();
    eprintln!("  Architecture: {arch}");
    eprintln!("  Layers: {}", gguf.block_count().unwrap_or(0));
    eprintln!("  Dim: {}", gguf.embedding_length().unwrap_or(0));
    eprintln!("  Tensors: {}", gguf.tensors.len());

    if arch == "qwen35" {
        return cmd_generate_qwen35(&gguf, &file_data, &prompt, max_tokens, temperature);
    }

    // Standard Llama path
    let config = ane_engine::model::ModelConfig::from_gguf(&gguf)?;
    eprintln!(
        "  Config: {}d, {}h, {}kv, {}ff, {}v",
        config.dim, config.n_heads, config.n_kv_heads, config.hidden_dim, config.vocab_size
    );

    eprintln!("Loading weights...");
    let t0 = Instant::now();
    let weights = load_model_weights(&gguf, &file_data, &config)?;
    eprintln!("  Loaded in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    eprintln!("Prompt: \"{prompt}\"");
    let prompt_tokens: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();
    eprintln!("  Tokens (byte-level): {} tokens", prompt_tokens.len());

    let mut engine = ane_engine::InferenceEngine::new(weights);
    match engine.compile_prefill(prompt_tokens.len().max(64)) {
        Ok(()) => {}
        Err(e) => eprintln!("  ANE prefill skipped: {e}"),
    }

    let params = ane_engine::scheduler::SamplingParams {
        temperature,
        top_p: _top_p,
        max_tokens,
    };

    let t0 = Instant::now();
    let tokens = engine.generate(&prompt_tokens, &params)?;
    let elapsed = t0.elapsed();

    let output: String = tokens
        .iter()
        .filter_map(|&t| if t < 256 { Some(t as u8 as char) } else { None })
        .collect();
    println!("{output}");

    let tok_s = tokens.len() as f64 / elapsed.as_secs_f64();
    eprintln!("\n--- Stats ---");
    eprintln!(
        "  Generated: {} tokens in {:.1}ms ({tok_s:.1} tok/s)",
        tokens.len(),
        elapsed.as_secs_f64() * 1000.0
    );
    Ok(())
}

/// Qwen3.5 hybrid DeltaNet+Attention generation.
fn cmd_generate_qwen35(
    gguf: &ane_gguf::GgufFile,
    file_data: &[u8],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<()> {
    use ane_engine::deltanet_cache::HybridCache;
    use ane_engine::model::*;

    let config = Qwen35Config::from_gguf(gguf)?;
    let c = &config.base;
    eprintln!(
        "  Qwen3.5 hybrid: {}d, {}h, {}kv, {}ff, {}v",
        c.dim, c.n_heads, c.n_kv_heads, c.hidden_dim, c.vocab_size
    );
    eprintln!(
        "  DeltaNet: {} state, {} conv, {} groups, interval={}",
        config.ssm_state_size,
        config.ssm_conv_kernel,
        config.ssm_group_count,
        config.full_attention_interval
    );
    let dn_count = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::DeltaNet)
        .count();
    let fa_count = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    eprintln!(
        "  Layer types: {} DeltaNet + {} FullAttention",
        dn_count, fa_count
    );

    // Load weights (Q8_0 — no dequantization, 4x less memory)
    eprintln!("Loading weights (Q8_0 direct)...");
    let t0 = Instant::now();
    let model = load_qwen35_q8(gguf, file_data, &config)?;
    eprintln!("  Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // Initialize cache
    let mut cache = HybridCache::new(&config);

    // Build BPE tokenizer from GGUF metadata
    eprintln!("Building tokenizer...");
    let vocab = gguf
        .tokenizer_tokens()
        .ok_or_else(|| anyhow::anyhow!("no tokenizer in GGUF"))?;
    let merges = gguf.tokenizer_merges();
    let eos_id = gguf.eos_token_id();
    let tokenizer = ane_engine::tokenizer::BpeTokenizer::from_gguf(vocab, merges, eos_id);
    eprintln!(
        "  Vocab: {} tokens, EOS: {:?}",
        tokenizer.vocab.len(),
        eos_id
    );

    // Tokenize prompt
    eprintln!("Prompt: \"{prompt}\"");
    let prompt_tokens = tokenizer.encode(prompt);
    eprintln!(
        "  Encoded: {} tokens → {:?}",
        prompt_tokens.len(),
        &prompt_tokens
    );
    // Verify round-trip
    let decoded_prompt = tokenizer.decode(&prompt_tokens);
    eprintln!("  Decoded: \"{}\"", decoded_prompt);

    // Decode loop (CPU only, one token at a time)
    eprintln!("Generating (CPU decode)...");
    let t0 = Instant::now();

    // Process prompt tokens
    let mut last_logits = vec![0.0f32; c.vocab_size];
    for (ti, &tok) in prompt_tokens.iter().enumerate() {
        eprint!("\r  Prefill: {}/{}", ti + 1, prompt_tokens.len());
        last_logits = qwen35_forward_token(&model, &mut cache, tok)?;
    }
    eprintln!();

    let ttft = t0.elapsed();
    eprintln!("  TTFT: {:.1}ms", ttft.as_secs_f64() * 1000.0);

    // Generate
    let mut generated = Vec::with_capacity(max_tokens);
    let mut next_token = greedy_sample(&last_logits, temperature);
    generated.push(next_token);

    let decode_start = Instant::now();
    for _i in 1..max_tokens {
        last_logits = qwen35_forward_token(&model, &mut cache, next_token)?;
        next_token = greedy_sample(&last_logits, temperature);

        if tokenizer.is_eos(next_token) {
            break;
        }
        generated.push(next_token);

        // Stream output
        let text = tokenizer.decode(&generated);
        eprint!("\r  {}", text);
    }
    let decode_elapsed = decode_start.elapsed();
    eprintln!();

    // Final decoded output
    let output = tokenizer.decode(&generated);
    eprintln!("  Raw IDs: {:?}", &generated[..generated.len().min(20)]);
    println!("\n--- Output ---");
    println!("{}{}", prompt, output);

    let tok_s = if generated.len() > 1 {
        (generated.len() - 1) as f64 / decode_elapsed.as_secs_f64()
    } else {
        0.0
    };
    eprintln!("\n--- Stats ---");
    eprintln!("  TTFT: {:.1}ms", ttft.as_secs_f64() * 1000.0);
    eprintln!("  Generated: {} tokens", generated.len());
    eprintln!("  Decode: {tok_s:.1} tok/s");

    Ok(())
}

/// Forward one token through the Qwen3.5 hybrid model.
fn qwen35_forward_token(
    model: &ane_engine::model::Qwen35ModelWeights,
    cache: &mut ane_engine::deltanet_cache::HybridCache,
    token_id: u32,
) -> Result<Vec<f32>> {
    use ane_engine::deltanet;
    use ane_engine::model::*;

    let c = &model.config.base;
    let dim = c.dim;
    let hidden_dim = c.hidden_dim;
    let eps = c.rms_norm_eps;

    let mut x = vec![0.0f32; dim];
    let tok = token_id as usize;
    if tok < c.vocab_size {
        x.copy_from_slice(&model.embedding[tok * dim..(tok + 1) * dim]);
    }

    for (l, layer) in model.layers.iter().enumerate() {
        let (_layer_type, local_idx) = cache.layer_map[l];

        match layer {
            HybridLayerWeights::DeltaNet(lw) => {
                let mut xnorm = vec![0.0f32; dim];
                rmsnorm_vec(&mut xnorm, &x, &lw.attn_norm, eps);

                let state = &mut cache.deltanet_states[local_idx];
                let attn_out = deltanet::deltanet_decode_step(&xnorm, state, lw, eps);

                for i in 0..dim {
                    x[i] += attn_out[i];
                }

                let mut ffn_in = vec![0.0f32; dim];
                rmsnorm_vec(&mut ffn_in, &x, &lw.post_attn_norm, eps);
                let ffn_out = ffn_forward(&ffn_in, &lw.ffn, dim, hidden_dim);
                for i in 0..dim {
                    x[i] += ffn_out[i];
                }
            }
            HybridLayerWeights::FullAttention(lw) => {
                let mut xnorm = vec![0.0f32; dim];
                rmsnorm_vec(&mut xnorm, &x, &lw.attn_norm, eps);

                let kv_cache = &mut cache.kv_caches[local_idx];
                let attn_out = deltanet::full_attn_decode_step(&xnorm, cache.pos, kv_cache, lw, c);

                for i in 0..dim {
                    x[i] += attn_out[i];
                }

                let mut ffn_in = vec![0.0f32; dim];
                rmsnorm_vec(&mut ffn_in, &x, &lw.post_attn_norm, eps);
                let ffn_out = ffn_forward(&ffn_in, &lw.ffn, dim, hidden_dim);
                for i in 0..dim {
                    x[i] += ffn_out[i];
                }
            }
        }
    }

    let mut final_out = vec![0.0f32; dim];
    rmsnorm_vec(&mut final_out, &x, &model.final_norm, eps);

    let mut logits = vec![0.0f32; c.vocab_size];
    ane_engine::q8_gemv::q8_gemv(&model.lm_head, &final_out, &mut logits);

    cache.advance(1);
    Ok(logits)
}

fn rmsnorm_vec(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let dim = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms * w[i];
    }
}

fn ffn_forward(
    x: &[f32],
    ffn: &ane_engine::model::FfnWeights,
    dim: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    use ane_engine::q8_gemv::q8_gemv;
    use ane_engine::scratch::vec_silu_mul_inplace;

    let mut h1 = vec![0.0f32; hidden_dim];
    let mut h3 = vec![0.0f32; hidden_dim];
    let mut out = vec![0.0f32; dim];

    q8_gemv(&ffn.gate, x, &mut h1);
    q8_gemv(&ffn.up, x, &mut h3);

    vec_silu_mul_inplace(&mut h1, &h3);

    q8_gemv(&ffn.down, &h1, &mut out);
    out
}

fn ffn_forward_scratch(
    x: &[f32],
    ffn: &ane_engine::model::FfnWeights,
    scratch: &mut ane_engine::scratch::ScratchBuffers,
    dim: usize,
    hidden_dim: usize,
) {
    use ane_engine::q8_gemv::q8_gemv;
    use ane_engine::scratch::vec_silu_mul_inplace;

    scratch.ffn_h1.fill(0.0);
    scratch.ffn_h3.fill(0.0);
    scratch.ffn_out.fill(0.0);

    q8_gemv(&ffn.gate, x, &mut scratch.ffn_h1[..hidden_dim]);
    q8_gemv(&ffn.up, x, &mut scratch.ffn_h3[..hidden_dim]);

    vec_silu_mul_inplace(
        &mut scratch.ffn_h1[..hidden_dim],
        &scratch.ffn_h3[..hidden_dim],
    );

    q8_gemv(
        &ffn.down,
        &scratch.ffn_h1[..hidden_dim],
        &mut scratch.ffn_out[..dim],
    );
}

/// ANE-accelerated prefill: batch all tokens through ANE projections per layer,
/// then run recurrence/attention on CPU with pre-computed projections.
fn ane_prefill_tokens(
    model: &ane_engine::model::Qwen35ModelWeights,
    cache: &mut ane_engine::deltanet_cache::HybridCache,
    tokens: &[u32],
    ane_kernels: &[ane_engine::ane_prefill::LayerAneKernels],
) -> Result<Vec<f32>> {
    use ane_engine::ane_prefill::*;
    use ane_engine::model::*;

    let c = &model.config.base;
    let dim = c.dim;
    let hidden_dim = c.hidden_dim;
    let eps = c.rms_norm_eps;
    let seq_len = tokens.len();

    // Embed all tokens: [seq_len, dim]
    let mut x = vec![0.0f32; seq_len * dim];
    for (t, &tok) in tokens.iter().enumerate() {
        let tok = tok as usize;
        if tok < c.vocab_size {
            x[t * dim..(t + 1) * dim].copy_from_slice(&model.embedding[tok * dim..(tok + 1) * dim]);
        }
    }

    // Process each layer
    for (l, (layer, ane_k)) in model.layers.iter().zip(ane_kernels.iter()).enumerate() {
        let (_lt, local_idx) = cache.layer_map[l];

        match (layer, ane_k) {
            (HybridLayerWeights::DeltaNet(lw), LayerAneKernels::DeltaNet(ak)) => {
                // RMSNorm all tokens
                let mut xnorm = vec![0.0f32; seq_len * dim];
                for t in 0..seq_len {
                    rmsnorm_vec(
                        &mut xnorm[t * dim..(t + 1) * dim],
                        &x[t * dim..(t + 1) * dim],
                        &lw.attn_norm,
                        eps,
                    );
                }

                // ANE: batch QKV projection [seq_len, dim] → [seq_len, 3*dim]
                let mut qkv_all = vec![0.0f32; seq_len * dim * 3];
                ak.qkv.forward(&xnorm, &mut qkv_all)?;

                // ANE: batch gate projection [seq_len, dim] → [seq_len, dim]
                let mut gate_all = vec![0.0f32; seq_len * dim];
                ak.gate.forward(&xnorm, &mut gate_all)?;

                // Process each token through recurrence (sequential — can't batch this)
                let state = &mut cache.deltanet_states[local_idx];
                let inner_size = dim * 3;
                let chunk = dim; // inner_size / 3
                let n_heads = state.n_heads;
                let head_dim_k = chunk / n_heads;
                let key_dim = state.key_dim;
                let value_dim = state.value_dim;

                for t in 0..seq_len {
                    let qkv_t = &qkv_all[t * inner_size..(t + 1) * inner_size];
                    let gate_t = &gate_all[t * dim..(t + 1) * dim];

                    // Conv1d
                    state.conv_shift_and_append(qkv_t);
                    let mut conv_out = vec![0.0f32; inner_size];
                    state.conv_apply(&lw.ssm_conv1d, &mut conv_out);
                    for v in conv_out.iter_mut() {
                        *v = *v / (1.0 + (-*v).exp());
                    } // SiLU

                    // Split Q, K, V
                    let q_flat = &conv_out[..chunk];
                    let k_flat = &conv_out[chunk..2 * chunk];
                    let v_flat = &conv_out[2 * chunk..];

                    let mut q = q_flat.to_vec();
                    let mut k = k_flat.to_vec();
                    let scale = 1.0 / (key_dim as f32).sqrt();
                    for h in 0..n_heads {
                        let off = h * head_dim_k;
                        // L2 normalize
                        let norm_sq: f32 = q[off..off + head_dim_k].iter().map(|x| x * x).sum();
                        let inv = 1.0 / (norm_sq + 1e-12).sqrt();
                        for i in 0..head_dim_k {
                            q[off + i] *= inv * scale;
                        }
                        let norm_sq: f32 = k[off..off + head_dim_k].iter().map(|x| x * x).sum();
                        let inv = 1.0 / (norm_sq + 1e-12).sqrt();
                        for i in 0..head_dim_k {
                            k[off + i] *= inv;
                        }
                    }

                    // Beta/alpha from hidden (still needs GEMV — small projections)
                    let hidden_t = &xnorm[t * dim..(t + 1) * dim];
                    let mut beta_raw = vec![0.0f32; n_heads];
                    let mut alpha_raw = vec![0.0f32; n_heads];
                    ane_engine::q8_gemv::q8_gemv(&lw.ssm_beta, hidden_t, &mut beta_raw);
                    ane_engine::q8_gemv::q8_gemv(&lw.ssm_alpha, hidden_t, &mut alpha_raw);

                    let mut beta = vec![0.0f32; n_heads];
                    let mut decay = vec![0.0f32; n_heads];
                    for h in 0..n_heads {
                        beta[h] = 1.0 / (1.0 + (-beta_raw[h]).exp());
                        let g = lw.ssm_a[h] * (1.0 + (alpha_raw[h] + lw.ssm_dt_bias[h]).exp()).ln();
                        decay[h] = g.exp();
                    }

                    // Recurrence per head
                    let v_per_head = chunk / n_heads;
                    let mut output_heads = vec![0.0f32; chunk];
                    for h in 0..n_heads {
                        let k_h = &k[h * head_dim_k..(h + 1) * head_dim_k];
                        let v_h = &v_flat[h * v_per_head..(h + 1) * v_per_head];
                        let q_h = &q[h * head_dim_k..(h + 1) * head_dim_k];
                        let s = state.head_state_mut(h);
                        let kd = key_dim.min(head_dim_k);
                        let vd = value_dim.min(v_per_head);

                        // Decay
                        for i in 0..kd * vd {
                            s[i] *= decay[h];
                        }
                        // Recall
                        let mut sk = vec![0.0f32; vd];
                        for vi in 0..vd {
                            for ki in 0..kd {
                                sk[vi] += s[ki * value_dim + vi] * k_h[ki];
                            }
                        }
                        // Delta
                        let mut delta = vec![0.0f32; vd];
                        for vi in 0..vd {
                            delta[vi] = beta[h] * (v_h[vi] - sk[vi]);
                        }
                        // Update
                        for ki in 0..kd {
                            for vi in 0..vd {
                                s[ki * value_dim + vi] += k_h[ki] * delta[vi];
                            }
                        }
                        // Output
                        for vi in 0..vd {
                            let mut d = 0f32;
                            for ki in 0..kd {
                                d += s[ki * value_dim + vi] * q_h[ki];
                            }
                            if vi < head_dim_k {
                                output_heads[h * head_dim_k + vi] = d;
                            }
                        }
                    }

                    // RMSNormGated + output projection
                    let mut normed = vec![0.0f32; chunk];
                    for h in 0..n_heads {
                        let off = h * head_dim_k;
                        let hslice = &output_heads[off..off + head_dim_k];
                        let nd = head_dim_k.min(lw.ssm_norm.len());
                        let ss: f32 = hslice.iter().map(|v| v * v).sum::<f32>() / nd as f32;
                        let inv = 1.0 / (ss + eps).sqrt();
                        for i in 0..nd {
                            normed[off + i] = hslice[i] * inv * lw.ssm_norm[i];
                        }
                    }
                    for i in 0..chunk.min(dim) {
                        normed[i] *= gate_t[i] / (1.0 + (-gate_t[i]).exp()); // * SiLU(gate)
                    }

                    // SSM out projection via ANE (but per-token, so use Q8 GEMV)
                    let mut attn_out = vec![0.0f32; dim];
                    ane_engine::q8_gemv::q8_gemv(&lw.ssm_out, &normed[..dim], &mut attn_out);

                    // Residual
                    for i in 0..dim {
                        x[t * dim + i] += attn_out[i];
                    }
                }

                // Post-attn norm + FFN (ANE batched)
                let mut ffn_in = vec![0.0f32; seq_len * dim];
                for t in 0..seq_len {
                    rmsnorm_vec(
                        &mut ffn_in[t * dim..(t + 1) * dim],
                        &x[t * dim..(t + 1) * dim],
                        &lw.post_attn_norm,
                        eps,
                    );
                }

                // FUSED FFN: gate+SiLU+up+mul+down = ONE ANE dispatch (6 ops chained)
                let mut ffn_out = vec![0.0f32; seq_len * dim];
                ak.fused_ffn.forward(&ffn_in, &mut ffn_out)?;

                for i in 0..seq_len * dim {
                    x[i] += ffn_out[i];
                }
            }

            (HybridLayerWeights::FullAttention(lw), LayerAneKernels::FullAttention(ak)) => {
                // RMSNorm all tokens
                let mut xnorm = vec![0.0f32; seq_len * dim];
                for t in 0..seq_len {
                    rmsnorm_vec(
                        &mut xnorm[t * dim..(t + 1) * dim],
                        &x[t * dim..(t + 1) * dim],
                        &lw.attn_norm,
                        eps,
                    );
                }

                // For full attention: process token by token (need KV cache updates)
                let kv_cache = &mut cache.kv_caches[local_idx];
                for t in 0..seq_len {
                    let hidden_t = &xnorm[t * dim..(t + 1) * dim];
                    let attn_out = ane_engine::deltanet::full_attn_decode_step(
                        hidden_t,
                        cache.pos + t,
                        kv_cache,
                        lw,
                        c,
                    );
                    for i in 0..dim {
                        x[t * dim + i] += attn_out[i];
                    }
                }

                // Post-attn norm + FFN (ANE batched)
                let mut ffn_in = vec![0.0f32; seq_len * dim];
                for t in 0..seq_len {
                    rmsnorm_vec(
                        &mut ffn_in[t * dim..(t + 1) * dim],
                        &x[t * dim..(t + 1) * dim],
                        &lw.post_attn_norm,
                        eps,
                    );
                }

                let mut ffn_out = vec![0.0f32; seq_len * dim];
                ak.fused_ffn.forward(&ffn_in, &mut ffn_out)?;

                for i in 0..seq_len * dim {
                    x[i] += ffn_out[i];
                }
            }

            _ => anyhow::bail!("layer type mismatch"),
        }
    }

    cache.advance(seq_len);

    // Final norm + logits (last token only)
    let last = &x[(seq_len - 1) * dim..seq_len * dim];
    let mut final_out = vec![0.0f32; dim];
    rmsnorm_vec(&mut final_out, last, &model.final_norm, eps);

    let mut logits = vec![0.0f32; c.vocab_size];
    ane_engine::q8_gemv::q8_gemv(&model.lm_head, &final_out, &mut logits);

    Ok(logits)
}

/// Metal GPU-accelerated forward token — batched command encoding.
/// All GEMV operations per layer are encoded into ONE command buffer.
fn qwen35_forward_token_metal(
    model: &ane_engine::model::Qwen35ModelWeights,
    cache: &mut ane_engine::deltanet_cache::HybridCache,
    token_id: u32,
    metal: &ane_engine::metal_gemv::MetalContext,
) -> Result<Vec<f32>> {
    use ane_engine::model::*;

    let c = &model.config.base;
    let dim = c.dim;
    let hidden_dim = c.hidden_dim;
    let eps = c.rms_norm_eps;

    let mut x = vec![0.0f32; dim];
    let tok = token_id as usize;
    if tok < c.vocab_size {
        x.copy_from_slice(&model.embedding[tok * dim..(tok + 1) * dim]);
    }

    for (l, layer) in model.layers.iter().enumerate() {
        let (_lt, local_idx) = cache.layer_map[l];

        match layer {
            HybridLayerWeights::DeltaNet(lw) => {
                let mut xnorm = vec![0.0f32; dim];
                rmsnorm_vec(&mut xnorm, &x, &lw.attn_norm, eps);

                // BATCH 1: QKV + gate projections — all read from xnorm
                let inner_size = dim * 3;
                let mut qkvz = vec![0.0f32; inner_size];
                let mut gate_out = vec![0.0f32; dim];
                metal.q8_gemv_batch(vec![
                    (&lw.qkv, &xnorm, &mut qkvz),
                    (&lw.attn_gate, &xnorm, &mut gate_out),
                ])?;

                // Beta/alpha (CPU — tiny [2048→16], not worth GPU dispatch)
                let state = &mut cache.deltanet_states[local_idx];
                let n_heads = state.n_heads;
                let mut beta_raw = vec![0.0f32; n_heads];
                let mut alpha_raw = vec![0.0f32; n_heads];
                ane_engine::q8_gemv::q8_gemv(&lw.ssm_beta, &xnorm, &mut beta_raw);
                ane_engine::q8_gemv::q8_gemv(&lw.ssm_alpha, &xnorm, &mut alpha_raw);

                // Conv1d + SiLU (CPU — tiny op)
                state.conv_shift_and_append(&qkvz);
                let mut conv_out = vec![0.0f32; inner_size];
                state.conv_apply(&lw.ssm_conv1d, &mut conv_out);
                for v in conv_out.iter_mut() {
                    *v = *v / (1.0 + (-*v).exp());
                }

                let chunk = inner_size / 3;
                let head_dim_k = chunk / n_heads;
                let key_dim = state.key_dim;
                let value_dim = state.value_dim;

                // Q, K split + L2 norm
                let mut q = conv_out[..chunk].to_vec();
                let mut k = conv_out[chunk..2 * chunk].to_vec();
                let v_flat = &conv_out[2 * chunk..];
                let scale = 1.0 / (key_dim as f32).sqrt();
                for h in 0..n_heads {
                    let off = h * head_dim_k;
                    let nq: f32 = q[off..off + head_dim_k].iter().map(|x| x * x).sum();
                    let inv = 1.0 / (nq + 1e-12).sqrt();
                    for i in 0..head_dim_k {
                        q[off + i] *= inv * scale;
                    }
                    let nk: f32 = k[off..off + head_dim_k].iter().map(|x| x * x).sum();
                    let inv = 1.0 / (nk + 1e-12).sqrt();
                    for i in 0..head_dim_k {
                        k[off + i] *= inv;
                    }
                }

                let mut beta = vec![0.0f32; n_heads];
                let mut decay = vec![0.0f32; n_heads];
                for h in 0..n_heads {
                    beta[h] = 1.0 / (1.0 + (-beta_raw[h]).exp());
                    let g = lw.ssm_a[h] * (1.0 + (alpha_raw[h] + lw.ssm_dt_bias[h]).exp()).ln();
                    decay[h] = g.exp();
                }

                // Recurrence (CPU — sequential, can't GPU this)
                let v_per_head = chunk / n_heads;
                let mut output_heads = vec![0.0f32; chunk];
                for h in 0..n_heads {
                    let k_h = &k[h * head_dim_k..(h + 1) * head_dim_k];
                    let v_h = &v_flat[h * v_per_head..(h + 1) * v_per_head];
                    let q_h = &q[h * head_dim_k..(h + 1) * head_dim_k];
                    let s = state.head_state_mut(h);
                    let kd = key_dim.min(head_dim_k);
                    let vd = value_dim.min(v_per_head);
                    for i in 0..kd * vd {
                        s[i] *= decay[h];
                    }
                    let mut sk = vec![0.0f32; vd];
                    for vi in 0..vd {
                        for ki in 0..kd {
                            sk[vi] += s[ki * value_dim + vi] * k_h[ki];
                        }
                    }
                    let mut delta = vec![0.0f32; vd];
                    for vi in 0..vd {
                        delta[vi] = beta[h] * (v_h[vi] - sk[vi]);
                    }
                    for ki in 0..kd {
                        for vi in 0..vd {
                            s[ki * value_dim + vi] += k_h[ki] * delta[vi];
                        }
                    }
                    sk.fill(0.0);
                    for vi in 0..vd {
                        for ki in 0..kd {
                            sk[vi] += s[ki * value_dim + vi] * q_h[ki];
                        }
                        if vi < head_dim_k {
                            output_heads[h * head_dim_k + vi] = sk[vi];
                        }
                    }
                }

                // RMSNormGated (CPU)
                let mut normed = vec![0.0f32; chunk];
                for h in 0..n_heads {
                    let off = h * head_dim_k;
                    let nd = head_dim_k.min(lw.ssm_norm.len());
                    let ss: f32 = output_heads[off..off + nd]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f32>()
                        / nd as f32;
                    let inv = 1.0 / (ss + eps).sqrt();
                    for i in 0..nd {
                        normed[off + i] = output_heads[off + i] * inv * lw.ssm_norm[i];
                    }
                }

                // gate_out already computed in BATCH 1 above
                for i in 0..chunk.min(dim) {
                    normed[i] *= gate_out[i] / (1.0 + (-gate_out[i]).exp());
                }

                // SSM out on Metal GPU
                let mut attn_out = vec![0.0f32; dim];
                metal.q8_gemv_adhoc(&lw.ssm_out, &normed[..dim], &mut attn_out)?;

                for i in 0..dim {
                    x[i] += attn_out[i];
                }

                // Post-attn norm + FFN
                let mut ffn_in = vec![0.0f32; dim];
                rmsnorm_vec(&mut ffn_in, &x, &lw.post_attn_norm, eps);

                // BATCH 2: FFN gate + up in parallel
                let mut h1 = vec![0.0f32; hidden_dim];
                let mut h3 = vec![0.0f32; hidden_dim];
                metal.q8_gemv_batch(vec![
                    (&lw.ffn.gate, &ffn_in, &mut h1),
                    (&lw.ffn.up, &ffn_in, &mut h3),
                ])?;

                for i in 0..hidden_dim {
                    h1[i] = (h1[i] / (1.0 + (-h1[i]).exp())) * h3[i];
                }

                // BATCH 3: FFN down
                let mut ffn_out = vec![0.0f32; dim];
                metal.q8_gemv_adhoc(&lw.ffn.down, &h1, &mut ffn_out)?;
                for i in 0..dim {
                    x[i] += ffn_out[i];
                }
            }
            HybridLayerWeights::FullAttention(lw) => {
                let mut xnorm = vec![0.0f32; dim];
                rmsnorm_vec(&mut xnorm, &x, &lw.attn_norm, eps);

                // QKV on Metal GPU
                let q_full_dim = lw.wq.m;
                let kv_dim = lw.wk.m;
                let head_dim = c.head_dim;
                let n_heads = c.n_heads;
                let n_kv_heads = c.n_kv_heads;

                // BATCH: QKV projections in parallel
                let mut q_full = vec![0.0f32; q_full_dim];
                let mut kk = vec![0.0f32; kv_dim];
                let mut vv = vec![0.0f32; kv_dim];
                metal.q8_gemv_batch(vec![
                    (&lw.wq, &xnorm, &mut q_full),
                    (&lw.wk, &xnorm, &mut kk),
                    (&lw.wv, &xnorm, &mut vv),
                ])?;

                // Q/gate split, QK norm, RoPE, attention (CPU — sequential)
                let q_only_dim = n_heads * head_dim;
                let mut q = vec![0.0f32; q_only_dim];
                let mut gate = vec![0.0f32; q_only_dim];
                for h in 0..n_heads {
                    let so = h * head_dim * 2;
                    let d = h * head_dim;
                    q[d..d + head_dim].copy_from_slice(&q_full[so..so + head_dim]);
                    gate[d..d + head_dim]
                        .copy_from_slice(&q_full[so + head_dim..so + 2 * head_dim]);
                }

                let qnd = head_dim.min(lw.q_norm.len());
                for h in 0..n_heads {
                    let off = h * head_dim;
                    let ss: f32 = q[off..off + qnd].iter().map(|v| v * v).sum::<f32>() / qnd as f32;
                    let inv = 1.0 / (ss + 1e-6).sqrt();
                    for i in 0..qnd {
                        q[off + i] = q[off + i] * inv * lw.q_norm[i];
                    }
                }
                for h in 0..n_kv_heads {
                    let off = h * head_dim;
                    let nd = head_dim.min(lw.k_norm.len());
                    let ss: f32 = kk[off..off + nd].iter().map(|v| v * v).sum::<f32>() / nd as f32;
                    let inv = 1.0 / (ss + 1e-6).sqrt();
                    for i in 0..nd {
                        kk[off + i] = kk[off + i] * inv * lw.k_norm[i];
                    }
                }

                // RoPE
                for h in 0..n_heads {
                    for i in (0..head_dim).step_by(2) {
                        let f = 1.0 / c.rope_freq_base.powf(i as f32 / head_dim as f32);
                        let v = cache.pos as f32 * f;
                        let (s, co) = v.sin_cos();
                        let o = h * head_dim + i;
                        let q0 = q[o];
                        let q1 = q[o + 1];
                        q[o] = q0 * co - q1 * s;
                        q[o + 1] = q0 * s + q1 * co;
                    }
                }
                for h in 0..n_kv_heads {
                    for i in (0..head_dim).step_by(2) {
                        let f = 1.0 / c.rope_freq_base.powf(i as f32 / head_dim as f32);
                        let v = cache.pos as f32 * f;
                        let (s, co) = v.sin_cos();
                        let o = h * head_dim + i;
                        let k0 = kk[o];
                        let k1 = kk[o + 1];
                        kk[o] = k0 * co - k1 * s;
                        kk[o + 1] = k0 * s + k1 * co;
                    }
                }

                let kv_cache = &mut cache.kv_caches[local_idx];
                kv_cache.write_pos(cache.pos, &kk, &vv);

                let hpk = n_heads / n_kv_heads;
                let sf = cache.pos + 1;
                let mut attn_out = vec![0.0f32; q_only_dim];
                for h in 0..n_heads {
                    let kvh = h / hpk;
                    let kc = kv_cache.key_head(kvh);
                    let vc = kv_cache.value_head(kvh);
                    let mut mx = f32::NEG_INFINITY;
                    let mut scores = vec![0.0f32; sf];
                    for s in 0..sf {
                        let mut d = 0f32;
                        for i in 0..head_dim {
                            d += q[h * head_dim + i] * kc[s * head_dim + i];
                        }
                        scores[s] = d / (head_dim as f32).sqrt();
                        if scores[s] > mx {
                            mx = scores[s];
                        }
                    }
                    let mut sm = 0f32;
                    for s in 0..sf {
                        scores[s] = (scores[s] - mx).exp();
                        sm += scores[s];
                    }
                    for s in 0..sf {
                        scores[s] /= sm;
                    }
                    for i in 0..head_dim {
                        let mut v = 0f32;
                        for s in 0..sf {
                            v += scores[s] * vc[s * head_dim + i];
                        }
                        attn_out[h * head_dim + i] = v;
                    }
                }
                for i in 0..q_only_dim {
                    attn_out[i] *= 1.0 / (1.0 + (-gate[i]).exp());
                }

                // O projection on Metal GPU
                let mut out = vec![0.0f32; dim];
                metal.q8_gemv_adhoc(&lw.wo, &attn_out, &mut out)?;
                for i in 0..dim {
                    x[i] += out[i];
                }

                // Post-attn norm + FFN on Metal GPU
                let mut ffn_in = vec![0.0f32; dim];
                rmsnorm_vec(&mut ffn_in, &x, &lw.post_attn_norm, eps);

                // BATCH: FFN gate + up in parallel
                let mut h1 = vec![0.0f32; hidden_dim];
                let mut h3 = vec![0.0f32; hidden_dim];
                metal.q8_gemv_batch(vec![
                    (&lw.ffn.gate, &ffn_in, &mut h1),
                    (&lw.ffn.up, &ffn_in, &mut h3),
                ])?;
                for i in 0..hidden_dim {
                    h1[i] = (h1[i] / (1.0 + (-h1[i]).exp())) * h3[i];
                }
                let mut fo = vec![0.0f32; dim];
                metal.q8_gemv_adhoc(&lw.ffn.down, &h1, &mut fo)?;
                for i in 0..dim {
                    x[i] += fo[i];
                }
            }
        }
    }

    let mut final_out = vec![0.0f32; dim];
    rmsnorm_vec(&mut final_out, &x, &model.final_norm, eps);
    let mut logits = vec![0.0f32; c.vocab_size];
    metal.q8_gemv_adhoc(&model.lm_head, &final_out, &mut logits)?;

    cache.advance(1);
    Ok(logits)
}

fn greedy_sample(logits: &[f32], temperature: f32) -> u32 {
    if temperature < 0.01 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    } else {
        // Temperature-scaled greedy (no top-p for now)
        let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
        scaled
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }
}

fn cmd_info(args: &[String]) -> Result<()> {
    let model_path = args
        .iter()
        .position(|a| a.as_str() == "--model" || a.as_str() == "-m")
        .map(|i| PathBuf::from(&args[i + 1]))
        .ok_or_else(|| anyhow::anyhow!("--model is required"))?;

    let file_data = std::fs::read(&model_path)?;
    let gguf = ane_gguf::GgufFile::parse(&file_data)?;

    println!("GGUF v{}", gguf.version);
    println!("Architecture: {}", gguf.architecture().unwrap_or("unknown"));
    println!("Embedding dim: {}", gguf.embedding_length().unwrap_or(0));
    println!("Layers: {}", gguf.block_count().unwrap_or(0));
    println!("Heads: {}", gguf.head_count().unwrap_or(0));
    println!("KV Heads: {}", gguf.head_count_kv().unwrap_or(0));
    println!("FFN dim: {}", gguf.feed_forward_length().unwrap_or(0));
    println!("Vocab: {}", gguf.vocab_size().unwrap_or(0));
    println!("RoPE base: {}", gguf.rope_freq_base().unwrap_or(10000.0));
    println!("Tensors: {}", gguf.tensors.len());

    println!("\nTensor list:");
    for t in &gguf.tensors {
        println!(
            "  {} {:?} {:?} ({} bytes)",
            t.name,
            t.typ,
            t.dimensions,
            t.data_size()
        );
    }

    Ok(())
}

fn load_model_weights(
    gguf: &ane_gguf::GgufFile,
    file_data: &[u8],
    config: &ane_engine::model::ModelConfig,
) -> Result<ane_engine::model::ModelWeights> {
    let embedding = ane_gguf::extract_embedding(gguf, file_data)?;
    let final_norm = ane_gguf::extract_final_norm(gguf, file_data)?;
    let lm_head = ane_gguf::extract_lm_head(gguf, file_data)?;
    let lm_head_blob = ane_bridge::build_single_weight_blob(&lm_head);

    let mut layers = Vec::with_capacity(config.n_layers);

    for l in 0..config.n_layers {
        let (qkv_blob, wq, wk, wv) = ane_gguf::extract_qkv_weights(gguf, file_data, l)?;
        let o_proj_blob = ane_gguf::extract_output_proj_weight(gguf, file_data, l)?;
        let wo =
            ane_gguf::extract_tensor_f32(gguf, file_data, &format!("blk.{l}.attn_output.weight"))?;

        let (ffn_up_blob, w1, w3) = ane_gguf::extract_ffn_up_weights(gguf, file_data, l)?;
        let ffn_down_blob = ane_gguf::extract_ffn_down_weight(gguf, file_data, l)?;
        let w2 =
            ane_gguf::extract_tensor_f32(gguf, file_data, &format!("blk.{l}.ffn_down.weight"))?;

        let (attn_norm, ffn_norm) = ane_gguf::extract_layer_norms(gguf, file_data, l)?;

        layers.push(ane_engine::model::LayerWeights {
            qkv_blob,
            o_proj_blob,
            ffn_up_blob,
            ffn_down_blob,
            attn_norm,
            ffn_norm,
            wq,
            wk,
            wv,
            wo,
            w1,
            w3,
            w2,
        });
    }

    Ok(ane_engine::model::ModelWeights {
        config: config.clone(),
        embedding,
        layers,
        final_norm,
        lm_head,
        lm_head_blob,
    })
}

/// Load Q8_0 tensor as Q8Tensor (no dequantization).
fn load_q8(
    gguf: &ane_gguf::GgufFile,
    file_data: &[u8],
    name: &str,
) -> Result<ane_engine::q8_gemv::Q8Tensor> {
    let (data, ne0, ne1) = ane_gguf::extract_tensor_raw(gguf, file_data, name)?;
    Ok(ane_engine::q8_gemv::Q8Tensor::from_raw(data, ne0, ne1))
}

/// Load Qwen3.5 model with Q8_0 weights (no dequantization for projections).
fn load_qwen35_q8(
    gguf: &ane_gguf::GgufFile,
    file_data: &[u8],
    config: &ane_engine::model::Qwen35Config,
) -> Result<ane_engine::model::Qwen35ModelWeights> {
    use ane_engine::model::*;

    let embedding = ane_gguf::extract_embedding(gguf, file_data)?; // FP32 for table lookup
    let final_norm = ane_gguf::extract_final_norm(gguf, file_data)?;
    let lm_head = load_q8(gguf, file_data, "output.weight")
        .or_else(|_| load_q8(gguf, file_data, "token_embd.weight"))?;

    let c = &config.base;
    let mut layers: Vec<HybridLayerWeights> = Vec::with_capacity(c.n_layers);

    for l in 0..c.n_layers {
        eprint!("  Layer {l}/{}...", c.n_layers);
        let layer = if config.layer_types[l] == LayerType::DeltaNet {
            HybridLayerWeights::DeltaNet(DeltaNetLayerWeights {
                attn_norm: ane_gguf::extract_tensor_f32(
                    gguf,
                    file_data,
                    &format!("blk.{l}.attn_norm.weight"),
                )?,
                post_attn_norm: ane_gguf::extract_tensor_f32(
                    gguf,
                    file_data,
                    &format!("blk.{l}.post_attention_norm.weight"),
                )?,
                qkv: load_q8(gguf, file_data, &format!("blk.{l}.attn_qkv.weight"))?,
                attn_gate: load_q8(gguf, file_data, &format!("blk.{l}.attn_gate.weight"))?,
                ssm_a: ane_gguf::extract_tensor_f32(gguf, file_data, &format!("blk.{l}.ssm_a"))?,
                ssm_alpha: load_q8(gguf, file_data, &format!("blk.{l}.ssm_alpha.weight"))?,
                ssm_beta: load_q8(gguf, file_data, &format!("blk.{l}.ssm_beta.weight"))?,
                ssm_conv1d: ane_gguf::extract_tensor_f32(
                    gguf,
                    file_data,
                    &format!("blk.{l}.ssm_conv1d.weight"),
                )?,
                ssm_dt_bias: ane_gguf::extract_tensor_f32(
                    gguf,
                    file_data,
                    &format!("blk.{l}.ssm_dt.bias"),
                )?,
                ssm_norm: ane_gguf::extract_tensor_f32(
                    gguf,
                    file_data,
                    &format!("blk.{l}.ssm_norm.weight"),
                )?,
                ssm_out: load_q8(gguf, file_data, &format!("blk.{l}.ssm_out.weight"))?,
                ffn: FfnWeights {
                    gate: load_q8(gguf, file_data, &format!("blk.{l}.ffn_gate.weight"))?,
                    up: load_q8(gguf, file_data, &format!("blk.{l}.ffn_up.weight"))?,
                    down: load_q8(gguf, file_data, &format!("blk.{l}.ffn_down.weight"))?,
                },
            })
        } else {
            HybridLayerWeights::FullAttention(FullAttnLayerWeights {
                attn_norm: ane_gguf::extract_tensor_f32(
                    gguf,
                    file_data,
                    &format!("blk.{l}.attn_norm.weight"),
                )?,
                post_attn_norm: ane_gguf::extract_tensor_f32(
                    gguf,
                    file_data,
                    &format!("blk.{l}.post_attention_norm.weight"),
                )?,
                wq: load_q8(gguf, file_data, &format!("blk.{l}.attn_q.weight"))?,
                wk: load_q8(gguf, file_data, &format!("blk.{l}.attn_k.weight"))?,
                wv: load_q8(gguf, file_data, &format!("blk.{l}.attn_v.weight"))?,
                wo: load_q8(gguf, file_data, &format!("blk.{l}.attn_output.weight"))?,
                q_norm: ane_gguf::extract_tensor_f32(
                    gguf,
                    file_data,
                    &format!("blk.{l}.attn_q_norm.weight"),
                )?,
                k_norm: ane_gguf::extract_tensor_f32(
                    gguf,
                    file_data,
                    &format!("blk.{l}.attn_k_norm.weight"),
                )?,
                ffn: FfnWeights {
                    gate: load_q8(gguf, file_data, &format!("blk.{l}.ffn_gate.weight"))?,
                    up: load_q8(gguf, file_data, &format!("blk.{l}.ffn_up.weight"))?,
                    down: load_q8(gguf, file_data, &format!("blk.{l}.ffn_down.weight"))?,
                },
            })
        };
        eprintln!(" ok");
        layers.push(layer);
    }

    Ok(Qwen35ModelWeights {
        config: config.clone(),
        embedding,
        layers,
        final_norm,
        lm_head,
    })
}
