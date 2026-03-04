//! ANE Prefill Engine — processes all input tokens in parallel on ANE.
//!
//! Pipeline:
//! 1. Tokenize → embed (CPU table lookup)
//! 2. Per layer: RMSNorm(CPU) → QKV+O_proj(ANE convs) → RoPE(CPU) → attention(CPU) → FFN(ANE) → residual(CPU)
//! 3. Final RMSNorm → logits projection (ANE if vocab fits, else CPU)

use ane_bridge::AneKernel;
use anyhow::Result;
use mil_gen::{
    cpu_rmsnorm, mil_gen_conv, mil_gen_ffn_down, mil_gen_ffn_up, mil_gen_output_proj, mil_gen_qkv,
};

use crate::kv_cache::KvCache;
use crate::model::ModelWeights;

/// Compiled ANE kernels for one transformer layer during prefill.
pub struct PrefillLayerKernels {
    pub qkv: AneKernel,
    pub o_proj: AneKernel,
    pub ffn_up: AneKernel,
    pub ffn_down: AneKernel,
}

/// All compiled prefill kernels.
pub struct PrefillKernels {
    pub layers: Vec<PrefillLayerKernels>,
    pub lm_head: Option<AneKernel>,
    pub seq_len: usize,
}

impl PrefillKernels {
    /// Compile all prefill kernels for a given sequence length.
    /// This is the expensive one-time cost at startup.
    pub fn compile(weights: &ModelWeights, seq_len: usize) -> Result<Self> {
        let cfg = &weights.config;
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim;

        let io_dim = dim * seq_len * 4; // FP32 bytes for [1, dim, 1, seq_len]
        let io_hidden = hidden_dim * seq_len * 4;

        let mut layers = Vec::with_capacity(cfg.n_layers);

        for l in 0..cfg.n_layers {
            let lw = &weights.layers[l];

            // QKV: 1 input [1,dim,1,S], 3 outputs [1,dim,1,S]
            let qkv_mil = mil_gen_qkv(dim, seq_len);
            let qkv = AneKernel::compile(
                &qkv_mil,
                Some(&lw.qkv_blob),
                &[io_dim],
                &[io_dim, io_dim, io_dim],
            )?;

            // O projection: 1 input, 1 output
            let o_mil = mil_gen_output_proj(dim, seq_len);
            let o_proj = AneKernel::compile(&o_mil, Some(&lw.o_proj_blob), &[io_dim], &[io_dim])?;

            // FFN up (gate+up): 1 input [1,dim,1,S], 2 outputs [1,hidden,1,S]
            let up_mil = mil_gen_ffn_up(dim, hidden_dim, seq_len);
            let ffn_up = AneKernel::compile(
                &up_mil,
                Some(&lw.ffn_up_blob),
                &[io_dim],
                &[io_hidden, io_hidden],
            )?;

            // FFN down: 1 input [1,hidden,1,S], 1 output [1,dim,1,S]
            let down_mil = mil_gen_ffn_down(dim, hidden_dim, seq_len);
            let ffn_down =
                AneKernel::compile(&down_mil, Some(&lw.ffn_down_blob), &[io_hidden], &[io_dim])?;

            layers.push(PrefillLayerKernels {
                qkv,
                o_proj,
                ffn_up,
                ffn_down,
            });
        }

        // LM head (classifier): may be too large for ANE with big vocabs
        let lm_head = if cfg.vocab_size <= 32000 {
            let lm_mil = mil_gen_conv(dim, cfg.vocab_size, seq_len);
            let io_vocab = cfg.vocab_size * seq_len * 4;
            AneKernel::compile(&lm_mil, Some(&weights.lm_head_blob), &[io_dim], &[io_vocab]).ok()
        } else {
            None
        };

        Ok(Self {
            layers,
            lm_head,
            seq_len,
        })
    }
}

/// Run prefill: process all tokens, populate KV cache, return logits for last token.
pub fn prefill(
    kernels: &PrefillKernels,
    weights: &ModelWeights,
    kv_cache: &mut KvCache,
    token_ids: &[u32],
) -> Result<Vec<f32>> {
    let cfg = &weights.config;
    let seq_len = token_ids.len();
    let dim = cfg.dim;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let hidden_dim = cfg.hidden_dim;

    assert!(
        seq_len <= kernels.seq_len,
        "prompt too long for compiled kernels"
    );

    // 1. Embed tokens: table lookup → [seq_len, dim]
    let mut x = vec![0f32; seq_len * dim];
    for (t, &tok) in token_ids.iter().enumerate() {
        let src = &weights.embedding[tok as usize * dim..(tok as usize + 1) * dim];
        x[t * dim..(t + 1) * dim].copy_from_slice(src);
    }

    // 2. Per-layer forward pass
    let mut xnorm = vec![0f32; seq_len * dim];
    let mut q_buf = vec![0f32; seq_len * dim];
    let mut k_buf = vec![0f32; seq_len * dim];
    let mut v_buf = vec![0f32; seq_len * dim];
    let mut attn_out = vec![0f32; seq_len * dim];
    let mut ffn_in = vec![0f32; seq_len * dim];
    let mut h1_buf = vec![0f32; seq_len * hidden_dim];
    let mut h3_buf = vec![0f32; seq_len * hidden_dim];
    let mut silu_buf = vec![0f32; seq_len * hidden_dim];

    // Transposed buffers for ANE (channel-first)
    let mut x_t = vec![0f32; seq_len * dim];
    let mut q_t = vec![0f32; seq_len * dim];
    let mut k_t = vec![0f32; seq_len * dim];
    let mut v_t = vec![0f32; seq_len * dim];
    let mut o_t = vec![0f32; seq_len * dim];
    let mut ffn_t = vec![0f32; seq_len * dim];
    let mut h1_t = vec![0f32; seq_len * hidden_dim];
    let mut h3_t = vec![0f32; seq_len * hidden_dim];
    let mut silu_t = vec![0f32; seq_len * hidden_dim];
    let mut ffn_out_t = vec![0f32; seq_len * dim];

    for l in 0..cfg.n_layers {
        let lw = &weights.layers[l];
        let lk = &kernels.layers[l];

        // RMSNorm (CPU)
        cpu_rmsnorm(&mut xnorm, &x, &lw.attn_norm, seq_len, dim);

        // Transpose to channel-first for ANE: [S, D] → [D, S]
        transpose(&xnorm, &mut x_t, seq_len, dim);

        // QKV on ANE
        lk.qkv.write_input_f32(0, &x_t);
        lk.qkv.eval()?;
        lk.qkv.read_output_f32(0, &mut q_t);
        lk.qkv.read_output_f32(1, &mut k_t);
        lk.qkv.read_output_f32(2, &mut v_t);

        // Transpose back: [D, S] → [S, D]
        transpose(&q_t, &mut q_buf, dim, seq_len);
        transpose(&k_t, &mut k_buf, dim, seq_len);
        transpose(&v_t, &mut v_buf, dim, seq_len);

        // RoPE (CPU)
        cpu_rope(
            &mut q_buf,
            &mut k_buf,
            seq_len,
            n_heads,
            head_dim,
            cfg.rope_freq_base,
        );

        // Store K, V in cache
        kv_cache.layers[l].write_range(0, seq_len, &k_buf, &v_buf);

        // Attention (CPU)
        cpu_attention(
            &mut attn_out,
            &q_buf,
            &k_buf,
            &v_buf,
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
        );

        // O projection on ANE
        transpose(&attn_out, &mut o_t, seq_len, dim);
        lk.o_proj.write_input_f32(0, &o_t);
        lk.o_proj.eval()?;
        lk.o_proj.read_output_f32(0, &mut x_t);
        transpose(&x_t, &mut attn_out, dim, seq_len);

        // Residual add
        for i in 0..seq_len * dim {
            x[i] += attn_out[i];
        }

        // FFN RMSNorm (CPU)
        cpu_rmsnorm(&mut ffn_in, &x, &lw.ffn_norm, seq_len, dim);

        // FFN up (gate + up) on ANE
        transpose(&ffn_in, &mut ffn_t, seq_len, dim);
        lk.ffn_up.write_input_f32(0, &ffn_t);
        lk.ffn_up.eval()?;
        lk.ffn_up.read_output_f32(0, &mut h1_t);
        lk.ffn_up.read_output_f32(1, &mut h3_t);
        transpose(&h1_t, &mut h1_buf, hidden_dim, seq_len);
        transpose(&h3_t, &mut h3_buf, hidden_dim, seq_len);

        // SiLU(h1) * h3 (CPU)
        for i in 0..seq_len * hidden_dim {
            let silu = h1_buf[i] / (1.0 + (-h1_buf[i]).exp());
            silu_buf[i] = silu * h3_buf[i];
        }

        // FFN down on ANE
        transpose(&silu_buf, &mut silu_t, seq_len, hidden_dim);
        lk.ffn_down.write_input_f32(0, &silu_t);
        lk.ffn_down.eval()?;
        lk.ffn_down.read_output_f32(0, &mut ffn_out_t);
        let mut ffn_out = vec![0f32; seq_len * dim];
        transpose(&ffn_out_t, &mut ffn_out, dim, seq_len);

        // Residual add
        for i in 0..seq_len * dim {
            x[i] += ffn_out[i];
        }
    }

    // Final RMSNorm
    let mut final_out = vec![0f32; seq_len * dim];
    cpu_rmsnorm(&mut final_out, &x, &weights.final_norm, seq_len, dim);

    // Logits — last token only for generation
    let last_hidden = &final_out[(seq_len - 1) * dim..seq_len * dim];
    let mut logits = vec![0f32; cfg.vocab_size];

    if let Some(ref lm_kernel) = kernels.lm_head {
        // Use ANE for logits
        // For a single token, we transpose the hidden state
        let mut hidden_t = vec![0f32; dim]; // [dim, 1] channel-first
        hidden_t.copy_from_slice(last_hidden);
        lm_kernel.write_input_f32(0, &hidden_t);
        lm_kernel.eval()?;
        lm_kernel.read_output_f32(0, &mut logits);
    } else {
        // CPU fallback for large vocab
        cpu_matmul(
            &weights.lm_head,
            last_hidden,
            &mut logits,
            1,
            dim,
            cfg.vocab_size,
        );
    }

    kv_cache.advance(seq_len);
    Ok(logits)
}

// --- CPU helper functions ---

/// Transpose [rows, cols] → [cols, rows]
fn transpose(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/// RoPE positional encoding
fn cpu_rope(
    q: &mut [f32],
    k: &mut [f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    freq_base: f32,
) {
    let d = n_heads * head_dim;
    for t in 0..seq_len {
        for h in 0..n_heads {
            for i in (0..head_dim).step_by(2) {
                let freq = 1.0 / freq_base.powf(i as f32 / head_dim as f32);
                let val = t as f32 * freq;
                let (sin_v, cos_v) = val.sin_cos();
                let off = t * d + h * head_dim + i;

                let q0 = q[off];
                let q1 = q[off + 1];
                q[off] = q0 * cos_v - q1 * sin_v;
                q[off + 1] = q0 * sin_v + q1 * cos_v;

                let k0 = k[off];
                let k1 = k[off + 1];
                k[off] = k0 * cos_v - k1 * sin_v;
                k[off + 1] = k0 * sin_v + k1 * cos_v;
            }
        }
    }
}

/// Causal self-attention (CPU)
fn cpu_attention(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let d = n_heads * head_dim;
    let kv_d = n_kv_heads * head_dim;
    let heads_per_kv = n_heads / n_kv_heads;

    let mut scores = vec![0f32; seq_len];

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;

        for t in 0..seq_len {
            let mut mx = f32::NEG_INFINITY;
            for s in 0..=t {
                let mut dot = 0f32;
                for i in 0..head_dim {
                    dot += q[t * d + h * head_dim + i] * k[s * kv_d + kv_h * head_dim + i];
                }
                scores[s] = dot * scale;
                if scores[s] > mx {
                    mx = scores[s];
                }
            }

            let mut sm = 0f32;
            for s in 0..=t {
                scores[s] = (scores[s] - mx).exp();
                sm += scores[s];
            }
            for s in 0..=t {
                scores[s] /= sm;
            }

            for i in 0..head_dim {
                let mut val = 0f32;
                for s in 0..=t {
                    val += scores[s] * v[s * kv_d + kv_h * head_dim + i];
                }
                out[t * d + h * head_dim + i] = val;
            }
        }
    }
}

/// CPU matmul: y = W @ x, W[out_dim, in_dim], x[seq_len, in_dim] → y[seq_len, out_dim]
fn cpu_matmul(w: &[f32], x: &[f32], y: &mut [f32], seq_len: usize, in_dim: usize, out_dim: usize) {
    for t in 0..seq_len {
        for i in 0..out_dim {
            let mut sum = 0f32;
            for j in 0..in_dim {
                sum += w[i * in_dim + j] * x[t * in_dim + j];
            }
            y[t * out_dim + i] = sum;
        }
    }
}
