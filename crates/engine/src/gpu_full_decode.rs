//! Full-GPU DeltaNet decode — entire layer in ONE command buffer.
//!
//! Eliminates CPU sync points by running conv1d, recurrence, norms,
//! and all projections on Metal GPU. Only ONE commit+wait per token
//! across ALL 24 layers.

use anyhow::Result;
use metal::*;

use crate::metal_graph::{GpuBuffer, GpuContext};
use crate::gpu_decode::{GpuDeltaNetLayerWeights, GpuModelWeights, GpuLayerWeights};
use crate::model::{Qwen35ModelWeights, HybridLayerWeights};

/// GPU-side scratch buffers for intermediate results within a DeltaNet layer.
/// All allocated once at model load, reused across tokens.
pub struct GpuLayerScratch {
    pub qkv_out: Buffer,      // [inner_size] — QKV projection output
    pub gate_out: Buffer,     // [dim] — gate projection output
    pub conv_out: Buffer,     // [inner_size] — conv1d + SiLU output
    pub q: Buffer,            // [chunk] — normalized Q
    pub k: Buffer,            // [chunk] — normalized K
    pub beta_raw: Buffer,     // [n_heads]
    pub alpha_raw: Buffer,    // [n_heads]
    pub beta: Buffer,         // [n_heads]
    pub decay: Buffer,        // [n_heads]
    pub output_heads: Buffer, // [chunk] — recurrence output
    pub normed: Buffer,       // [chunk] — after RMSNorm + gate
    pub ssm_out: Buffer,      // [dim] — SSM output projection
    pub xnorm: Buffer,        // [dim] — post-attn norm
    pub ffn_h1: Buffer,       // [hidden_dim] — FFN gate output (after SiLU*up)
    pub ffn_out: Buffer,      // [dim] — FFN down output
    pub hidden: Buffer,       // [dim] — residual hidden state
}

impl GpuLayerScratch {
    pub fn new(device: &Device, dim: usize, hidden_dim: usize, inner_size: usize, n_heads: usize) -> Self {
        let alloc = |n: usize| -> Buffer {
            device.new_buffer((n * 4).max(4) as u64, MTLResourceOptions::StorageModeShared)
        };
        let chunk = inner_size / 3;
        Self {
            qkv_out: alloc(inner_size),
            gate_out: alloc(dim),
            conv_out: alloc(inner_size),
            q: alloc(chunk),
            k: alloc(chunk),
            beta_raw: alloc(n_heads),
            alpha_raw: alloc(n_heads),
            beta: alloc(n_heads),
            decay: alloc(n_heads),
            output_heads: alloc(chunk),
            normed: alloc(chunk),
            ssm_out: alloc(dim),
            xnorm: alloc(dim),
            ffn_h1: alloc(hidden_dim),
            ffn_out: alloc(dim),
            hidden: alloc(dim),
        }
    }
}

/// Encode the ENTIRE forward pass for one token into a SINGLE command buffer.
/// Returns logits buffer.
pub fn encode_full_token_gpu(
    gpu: &GpuContext,
    model: &Qwen35ModelWeights,
    gpu_weights: &GpuModelWeights,
    scratch: &GpuLayerScratch,
    token_id: u32,
) -> Result<Vec<f32>> {
    let cfg = &model.config;
    let dim = cfg.base.dim;
    let hidden_dim = cfg.base.hidden_dim;
    let inner_size = dim * 3;
    let n_heads = 16; // DeltaNet heads
    let chunk = inner_size / 3;
    let head_dim_k = chunk / n_heads;
    let eps = cfg.base.rms_norm_eps;

    // Write embedding to hidden buffer
    unsafe {
        let emb = model.embedding.as_ptr().add((token_id as usize) * dim);
        let dst = scratch.hidden.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(emb, dst, dim);
    }

    // Create ONE command buffer for the ENTIRE token
    let cmd_buf = gpu.queue.new_command_buffer();

    for (layer_idx, layer_weights) in gpu_weights.layers.iter().enumerate() {
        match (layer_weights, &model.layers[layer_idx]) {
            (GpuLayerWeights::DeltaNet(gw), HybridLayerWeights::DeltaNet(w)) => {
                let key_dim = cfg.ssm_state_size;
                let value_dim = key_dim;
                let v_per_head = chunk / n_heads;
                let kernel_size = w.ssm_conv1d.len() / inner_size;

                // --- RMSNorm(hidden) → scratch_input for GEMV ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.rmsnorm_simple_pipeline);
                    enc.set_buffer(0, Some(&scratch.hidden), 0);
                    enc.set_buffer(1, Some(&gw.attn_norm_buf), 0);
                    enc.set_buffer(2, Some(&gpu.scratch_input), 0); // output → scratch_input[0]
                    let dim_u32 = dim as u32;
                    let eps_f32 = eps;
                    let dim_buf = gpu.device.new_buffer_with_data(&dim_u32 as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    let eps_buf = gpu.device.new_buffer_with_data(&eps_f32 as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(3, Some(&dim_buf), 0);
                    enc.set_buffer(4, Some(&eps_buf), 0);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
                    enc.end_encoding();
                }

                // --- GEMV: QKV + gate + beta + alpha (5 GEMVs) ---
                // QKV: [dim] → [inner_size]
                encode_gemv(cmd_buf, gpu, &gw.qkv, &gpu.scratch_input, 0, &scratch.qkv_out, 0, dim, inner_size);
                // Gate: [dim] → [dim]
                encode_gemv(cmd_buf, gpu, &gw.attn_gate, &gpu.scratch_input, 0, &scratch.gate_out, 0, dim, dim);
                // Beta: [dim] → [n_heads]
                encode_gemv(cmd_buf, gpu, &gw.ssm_beta, &gpu.scratch_input, 0, &scratch.beta_raw, 0, dim, n_heads);
                // Alpha: [dim] → [n_heads]
                encode_gemv(cmd_buf, gpu, &gw.ssm_alpha, &gpu.scratch_input, 0, &scratch.alpha_raw, 0, dim, n_heads);

                // --- Conv1d + SiLU ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.conv1d_silu_pipeline);
                    enc.set_buffer(0, Some(&gw.conv_state_buf), 0);
                    enc.set_buffer(1, Some(&gw.conv1d_buf), 0);
                    enc.set_buffer(2, Some(&scratch.qkv_out), 0);
                    enc.set_buffer(3, Some(&scratch.conv_out), 0);
                    let is = inner_size as u32;
                    let ks = kernel_size as u32;
                    let is_buf = gpu.device.new_buffer_with_data(&is as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    let ks_buf = gpu.device.new_buffer_with_data(&ks as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(4, Some(&is_buf), 0);
                    enc.set_buffer(5, Some(&ks_buf), 0);
                    let tg = MTLSize::new(((inner_size + 255) / 256) as u64, 1, 1);
                    enc.dispatch_thread_groups(tg, MTLSize::new(256, 1, 1));
                    enc.end_encoding();
                }

                // --- GPU memcpy: conv_out[0..chunk] → q, conv_out[chunk..2*chunk] → k ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.gpu_memcpy_pipeline);
                    enc.set_buffer(0, Some(&scratch.conv_out), 0);
                    enc.set_buffer(1, Some(&scratch.q), 0);
                    let n = chunk as u32;
                    let n_buf = gpu.device.new_buffer_with_data(&n as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(2, Some(&n_buf), 0);
                    enc.dispatch_threads(MTLSize::new(chunk as u64, 1, 1), MTLSize::new(256, 1, 1));
                    enc.end_encoding();
                }
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.gpu_memcpy_pipeline);
                    enc.set_buffer(0, Some(&scratch.conv_out), (chunk * 4) as u64);
                    enc.set_buffer(1, Some(&scratch.k), 0);
                    let n = chunk as u32;
                    let n_buf = gpu.device.new_buffer_with_data(&n as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(2, Some(&n_buf), 0);
                    enc.dispatch_threads(MTLSize::new(chunk as u64, 1, 1), MTLSize::new(256, 1, 1));
                    enc.end_encoding();
                }

                // --- L2 normalize + scale Q and K ---
                {
                    let scale = 1.0f32 / (key_dim as f32).sqrt();
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.l2_norm_scale_pipeline);
                    enc.set_buffer(0, Some(&scratch.q), 0);
                    let hd = head_dim_k as u32;
                    let nh = n_heads as u32;
                    let hd_buf = gpu.device.new_buffer_with_data(&hd as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    let nh_buf = gpu.device.new_buffer_with_data(&nh as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    let sc_buf = gpu.device.new_buffer_with_data(&scale as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(1, Some(&hd_buf), 0);
                    enc.set_buffer(2, Some(&nh_buf), 0);
                    enc.set_buffer(3, Some(&sc_buf), 0);
                    enc.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
                    enc.end_encoding();
                }
                {
                    let scale = 1.0f32; // K doesn't get the 1/sqrt(d) scale
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.l2_norm_scale_pipeline);
                    enc.set_buffer(0, Some(&scratch.k), 0);
                    let hd = head_dim_k as u32;
                    let nh = n_heads as u32;
                    let hd_buf = gpu.device.new_buffer_with_data(&hd as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    let nh_buf = gpu.device.new_buffer_with_data(&nh as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    let sc_buf = gpu.device.new_buffer_with_data(&scale as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(1, Some(&hd_buf), 0);
                    enc.set_buffer(2, Some(&nh_buf), 0);
                    enc.set_buffer(3, Some(&sc_buf), 0);
                    enc.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
                    enc.end_encoding();
                }

                // --- Compute beta/decay ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.compute_beta_decay_pipeline);
                    enc.set_buffer(0, Some(&scratch.beta_raw), 0);
                    enc.set_buffer(1, Some(&scratch.alpha_raw), 0);
                    enc.set_buffer(2, Some(&gw.ssm_a_buf), 0);
                    enc.set_buffer(3, Some(&gw.dt_bias_buf), 0);
                    enc.set_buffer(4, Some(&scratch.beta), 0);
                    enc.set_buffer(5, Some(&scratch.decay), 0);
                    let nh = n_heads as u32;
                    let nh_buf = gpu.device.new_buffer_with_data(&nh as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(6, Some(&nh_buf), 0);
                    enc.dispatch_threads(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
                    enc.end_encoding();
                }

                // --- DeltaNet recurrence (one threadgroup per head) ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.deltanet_recurrence_pipeline);
                    enc.set_buffer(0, Some(&gw.recurrent_state_buf), 0);
                    enc.set_buffer(1, Some(&scratch.q), 0);
                    enc.set_buffer(2, Some(&scratch.k), 0);
                    enc.set_buffer(3, Some(&scratch.conv_out), (2 * chunk * 4) as u64); // v starts at offset 2*chunk
                    enc.set_buffer(4, Some(&scratch.decay), 0);
                    enc.set_buffer(5, Some(&scratch.beta), 0);
                    enc.set_buffer(6, Some(&scratch.output_heads), 0);
                    let params: [u32; 5] = [n_heads as u32, key_dim as u32, value_dim as u32, head_dim_k as u32, v_per_head as u32];
                    let p_buf = gpu.device.new_buffer_with_data(params.as_ptr() as *const _, 20, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(7, Some(&p_buf), 0);
                    enc.set_buffer(8, Some(&p_buf), 4);
                    enc.set_buffer(9, Some(&p_buf), 8);
                    enc.set_buffer(10, Some(&p_buf), 12);
                    enc.set_buffer(11, Some(&p_buf), 16);
                    enc.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
                    enc.end_encoding();
                }

                // --- RMSNorm + gated output ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.rmsnorm_gated_pipeline);
                    enc.set_buffer(0, Some(&scratch.output_heads), 0);
                    enc.set_buffer(1, Some(&gw.ssm_norm_buf), 0);
                    enc.set_buffer(2, Some(&scratch.gate_out), 0); // gate
                    enc.set_buffer(3, Some(&scratch.normed), 0);
                    let td = chunk as u32;
                    let nd = head_dim_k.min(w.ssm_norm.len()) as u32;
                    let nh = n_heads as u32;
                    let has_gate = 1u32;
                    let params: [u32; 4] = [td, nd, nh, has_gate];
                    let eps_f32 = eps;
                    let p_buf = gpu.device.new_buffer_with_data(params.as_ptr() as *const _, 16, MTLResourceOptions::StorageModeShared);
                    let e_buf = gpu.device.new_buffer_with_data(&eps_f32 as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(4, Some(&p_buf), 0);
                    enc.set_buffer(5, Some(&p_buf), 4);
                    enc.set_buffer(6, Some(&p_buf), 8);
                    enc.set_buffer(7, Some(&e_buf), 0);
                    enc.set_buffer(8, Some(&p_buf), 12);
                    enc.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
                    enc.end_encoding();
                }

                // --- GEMV: SSM out [chunk→dim] ---
                encode_gemv(cmd_buf, gpu, &gw.ssm_out, &scratch.normed, 0, &scratch.ssm_out, 0, chunk.min(dim), dim);

                // --- Residual: hidden += ssm_out ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.residual_add_pipeline);
                    enc.set_buffer(0, Some(&scratch.hidden), 0);
                    enc.set_buffer(1, Some(&scratch.ssm_out), 0);
                    let n = dim as u32;
                    let n_buf = gpu.device.new_buffer_with_data(&n as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(2, Some(&n_buf), 0);
                    enc.dispatch_threads(MTLSize::new(dim as u64, 1, 1), MTLSize::new(256, 1, 1));
                    enc.end_encoding();
                }

                // --- Post-attn RMSNorm ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.rmsnorm_simple_pipeline);
                    enc.set_buffer(0, Some(&scratch.hidden), 0);
                    enc.set_buffer(1, Some(&gw.post_attn_norm_buf), 0);
                    enc.set_buffer(2, Some(&scratch.xnorm), 0);
                    let dim_u32 = dim as u32;
                    let d_buf = gpu.device.new_buffer_with_data(&dim_u32 as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    let e_buf = gpu.device.new_buffer_with_data(&eps as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(3, Some(&d_buf), 0);
                    enc.set_buffer(4, Some(&e_buf), 0);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
                    enc.end_encoding();
                }

                // --- FFN: gate + up GEMVs ---
                encode_gemv(cmd_buf, gpu, &gw.ffn_gate, &scratch.xnorm, 0, &scratch.ffn_h1, 0, dim, hidden_dim);
                // Reuse ssm_out buffer for FFN up (it's done being read)
                encode_gemv(cmd_buf, gpu, &gw.ffn_up, &scratch.xnorm, 0, &scratch.ssm_out, 0, dim, hidden_dim);

                // --- SiLU(gate) * up ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.silu_pipeline);
                    enc.set_buffer(0, Some(&scratch.ffn_h1), 0);
                    enc.set_buffer(1, Some(&scratch.ssm_out), 0);
                    let n = hidden_dim as u32;
                    let n_buf = gpu.device.new_buffer_with_data(&n as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(2, Some(&n_buf), 0);
                    enc.dispatch_threads(MTLSize::new(hidden_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
                    enc.end_encoding();
                }

                // --- FFN down GEMV ---
                encode_gemv(cmd_buf, gpu, &gw.ffn_down, &scratch.ffn_h1, 0, &scratch.ffn_out, 0, hidden_dim, dim);

                // --- Residual: hidden += ffn_out ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.residual_add_pipeline);
                    enc.set_buffer(0, Some(&scratch.hidden), 0);
                    enc.set_buffer(1, Some(&scratch.ffn_out), 0);
                    let n = dim as u32;
                    let n_buf = gpu.device.new_buffer_with_data(&n as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
                    enc.set_buffer(2, Some(&n_buf), 0);
                    enc.dispatch_threads(MTLSize::new(dim as u64, 1, 1), MTLSize::new(256, 1, 1));
                    enc.end_encoding();
                }
            }
            (GpuLayerWeights::FullAttention(_gw), HybridLayerWeights::FullAttention(_w)) => {
                // TODO: Full attention on GPU (RoPE, softmax, KV cache)
                // For now skip — FullAttention layers are only 6 of 24
                // They'll produce wrong output but we test DeltaNet layers first
            }
            _ => {}
        }
    }

    // --- Final: LM head GEMV ---
    // RMSNorm hidden → input
    {
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&gpu.rmsnorm_simple_pipeline);
        enc.set_buffer(0, Some(&scratch.hidden), 0);
        let final_norm = &model.final_norm;
        let fn_buf = gpu.device.new_buffer_with_data(
            final_norm.as_ptr() as *const _, (final_norm.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_buffer(1, Some(&fn_buf), 0);
        enc.set_buffer(2, Some(&gpu.scratch_input), 0);
        let dim_u32 = dim as u32;
        let d_buf = gpu.device.new_buffer_with_data(&dim_u32 as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
        let e_buf = gpu.device.new_buffer_with_data(&eps as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
        enc.set_buffer(3, Some(&d_buf), 0);
        enc.set_buffer(4, Some(&e_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
    }

    // LM head GEMV
    encode_gemv(cmd_buf, gpu, &gpu_weights.lm_head, &gpu.scratch_input, 0, &gpu.scratch_output, 0, dim, cfg.base.vocab_size);

    // ONE commit, ONE wait for ENTIRE token
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Read logits
    let mut logits = vec![0.0f32; cfg.base.vocab_size];
    unsafe {
        let ptr = gpu.scratch_output.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr, logits.as_mut_ptr(), logits.len());
    }

    Ok(logits)
}

// Helper: encode a Q8 GEMV into an existing command buffer
fn encode_gemv(
    cmd_buf: &CommandBufferRef,
    gpu: &GpuContext,
    weight: &GpuBuffer,
    input_buf: &Buffer,
    input_offset: usize,
    output_buf: &Buffer,
    output_offset: usize,
    _input_len: usize,
    output_len: usize,
) {
    let m = weight.m as u32;
    let n_blocks = weight.n_blocks;
    let params: [u32; 2] = [n_blocks, m];
    let p_buf = gpu.device.new_buffer_with_data(
        params.as_ptr() as *const _, 8, MTLResourceOptions::StorageModeShared,
    );

    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&gpu.gemv_pipeline);
    enc.set_buffer(0, Some(&weight.buffer), 0);
    enc.set_buffer(1, Some(input_buf), (input_offset * 4) as u64);
    enc.set_buffer(2, Some(output_buf), (output_offset * 4) as u64);
    enc.set_buffer(3, Some(&p_buf), 0);
    enc.set_buffer(4, Some(&p_buf), 4);
    enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(128, 1, 1));
    enc.end_encoding();
}
