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

/// Pre-allocated constant pool for all shader dispatch params.
/// Written once at init, never changes between tokens.
pub struct GpuConstantPool {
    pub buf: Buffer,
}

impl GpuConstantPool {
    // Layout (all u32 unless noted):
    // 0: dim
    // 4: inner_size
    // 8: hidden_dim
    // 12: n_heads
    // 16: head_dim_k (chunk / n_heads)
    // 20: chunk (inner_size / 3)
    // 24: eps (f32)
    // 28: key_dim
    // 32: value_dim
    // 36: v_per_head (chunk / n_heads)
    // 40: kernel_size
    // 44: scale (1/sqrt(key_dim), f32)
    // 48: scale_1 (1.0f32)
    // 52: has_gate (1u32)
    // 56: zero (0u32)
    pub const DIM: usize = 0;
    pub const INNER_SIZE: usize = 4;
    pub const HIDDEN_DIM: usize = 8;
    pub const N_HEADS: usize = 12;
    pub const HEAD_DIM_K: usize = 16;
    pub const CHUNK: usize = 20;
    pub const EPS: usize = 24;
    pub const KEY_DIM: usize = 28;
    pub const VALUE_DIM: usize = 32;
    pub const V_PER_HEAD: usize = 36;
    pub const KERNEL_SIZE: usize = 40;
    pub const SCALE: usize = 44;
    pub const SCALE_1: usize = 48;
    pub const HAS_GATE: usize = 52;
    pub const ZERO: usize = 56;

    pub fn new(device: &Device, dim: usize, hidden_dim: usize, inner_size: usize, n_heads: usize, key_dim: usize, kernel_size: usize, eps: f32) -> Self {
        let buf = device.new_buffer(256, MTLResourceOptions::StorageModeShared);
        let ptr = buf.contents() as *mut u32;
        let chunk = inner_size / 3;
        let head_dim_k = chunk / n_heads;
        let scale = 1.0f32 / (key_dim as f32).sqrt();
        unsafe {
            *ptr.add(0) = dim as u32;
            *ptr.add(1) = inner_size as u32;
            *ptr.add(2) = hidden_dim as u32;
            *ptr.add(3) = n_heads as u32;
            *ptr.add(4) = head_dim_k as u32;
            *ptr.add(5) = chunk as u32;
            *(ptr.add(6) as *mut f32) = eps;
            *ptr.add(7) = key_dim as u32;
            *ptr.add(8) = key_dim as u32; // value_dim = key_dim
            *ptr.add(9) = (chunk / n_heads) as u32;
            *ptr.add(10) = kernel_size as u32;
            *(ptr.add(11) as *mut f32) = scale;
            *(ptr.add(12) as *mut f32) = 1.0f32;
            *ptr.add(13) = 1u32; // has_gate
            *ptr.add(14) = 0u32; // zero
        }
        Self { buf }
    }
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
    cp: &GpuConstantPool,
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

    // Pre-allocate a params pool — one big buffer for ALL small constants
    // Each layer needs ~20 params × 4 bytes = 80 bytes, × 24 layers = 1920 bytes
    // Plus global params. 4KB is plenty.
    let params_pool = gpu.device.new_buffer(4096, MTLResourceOptions::StorageModeShared);
    let pp = params_pool.contents() as *mut u32;

    // Write commonly-used constants at known offsets
    // Offset 0: dim (u32)
    // Offset 4: inner_size (u32)
    // Offset 8: hidden_dim (u32)
    // Offset 12: n_heads (u32)
    // Offset 16: head_dim_k (u32)
    // Offset 20: chunk (u32)
    // Offset 24: eps (f32)
    // Offset 28: key_dim (u32)
    // Offset 32: value_dim (u32)
    // Offset 36: v_per_head (u32)
    // Offset 40: kernel_size (u32)
    // Offset 44...: per-GEMV params (n_blocks, m) pairs
    unsafe {
        *pp.add(0) = dim as u32;
        *pp.add(1) = inner_size as u32;
        *pp.add(2) = hidden_dim as u32;
        *pp.add(3) = n_heads as u32;
        *pp.add(4) = head_dim_k as u32;
        *pp.add(5) = chunk as u32;
        *(pp.add(6) as *mut f32) = eps;
        *pp.add(7) = cfg.ssm_state_size as u32; // key_dim
        *pp.add(8) = cfg.ssm_state_size as u32; // value_dim
        *pp.add(9) = (chunk / n_heads) as u32;  // v_per_head
        // kernel_size written per-layer below
    }

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
                    enc.set_buffer(2, Some(&gpu.scratch_input), 0);
                    enc.set_buffer(3, Some(&cp.buf), GpuConstantPool::DIM as u64);
                    enc.set_buffer(4, Some(&cp.buf), GpuConstantPool::EPS as u64);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
                    enc.end_encoding();
                }

                // --- GEMV: QKV + gate + beta + alpha (5 GEMVs) ---
                // QKV: [dim] → [inner_size]
                encode_gemv_prealloc(cmd_buf, gpu, &gw.qkv, &gpu.scratch_input, 0, &scratch.qkv_out, 0, &gw.qkv_params);
                // Gate: [dim] → [dim]
                encode_gemv_prealloc(cmd_buf, gpu, &gw.attn_gate, &gpu.scratch_input, 0, &scratch.gate_out, 0, &gw.attn_gate_params);
                // Beta: [dim] → [n_heads]
                encode_gemv_prealloc(cmd_buf, gpu, &gw.ssm_beta, &gpu.scratch_input, 0, &scratch.beta_raw, 0, &gw.ssm_beta_params);
                // Alpha: [dim] → [n_heads]
                encode_gemv_prealloc(cmd_buf, gpu, &gw.ssm_alpha, &gpu.scratch_input, 0, &scratch.alpha_raw, 0, &gw.ssm_alpha_params);

                // --- Conv1d + SiLU ---
                // Write kernel_size to pool (may differ per layer)
                unsafe { *(cp.buf.contents() as *mut u32).add(10) = kernel_size as u32; }
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.conv1d_silu_pipeline);
                    enc.set_buffer(0, Some(&gw.conv_state_buf), 0);
                    enc.set_buffer(1, Some(&gw.conv1d_buf), 0);
                    enc.set_buffer(2, Some(&scratch.qkv_out), 0);
                    enc.set_buffer(3, Some(&scratch.conv_out), 0);
                    enc.set_buffer(4, Some(&cp.buf), GpuConstantPool::INNER_SIZE as u64);
                    enc.set_buffer(5, Some(&cp.buf), GpuConstantPool::KERNEL_SIZE as u64);
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
                    enc.set_buffer(2, Some(&cp.buf), GpuConstantPool::CHUNK as u64);
                    enc.dispatch_threads(MTLSize::new(chunk as u64, 1, 1), MTLSize::new(256, 1, 1));
                    enc.end_encoding();
                }
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.gpu_memcpy_pipeline);
                    enc.set_buffer(0, Some(&scratch.conv_out), (chunk * 4) as u64);
                    enc.set_buffer(1, Some(&scratch.k), 0);
                    enc.set_buffer(2, Some(&cp.buf), GpuConstantPool::CHUNK as u64);
                    enc.dispatch_threads(MTLSize::new(chunk as u64, 1, 1), MTLSize::new(256, 1, 1));
                    enc.end_encoding();
                }

                // --- L2 normalize + scale Q and K ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.l2_norm_scale_pipeline);
                    enc.set_buffer(0, Some(&scratch.q), 0);
                    enc.set_buffer(1, Some(&cp.buf), GpuConstantPool::HEAD_DIM_K as u64);
                    enc.set_buffer(2, Some(&cp.buf), GpuConstantPool::N_HEADS as u64);
                    enc.set_buffer(3, Some(&cp.buf), GpuConstantPool::SCALE as u64);
                    enc.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
                    enc.end_encoding();
                }
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.l2_norm_scale_pipeline);
                    enc.set_buffer(0, Some(&scratch.k), 0);
                    enc.set_buffer(1, Some(&cp.buf), GpuConstantPool::HEAD_DIM_K as u64);
                    enc.set_buffer(2, Some(&cp.buf), GpuConstantPool::N_HEADS as u64);
                    enc.set_buffer(3, Some(&cp.buf), GpuConstantPool::SCALE_1 as u64); // K scale = 1.0
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
                    enc.set_buffer(6, Some(&cp.buf), GpuConstantPool::N_HEADS as u64);
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
                    enc.set_buffer(3, Some(&scratch.conv_out), (2 * chunk * 4) as u64);
                    enc.set_buffer(4, Some(&scratch.decay), 0);
                    enc.set_buffer(5, Some(&scratch.beta), 0);
                    enc.set_buffer(6, Some(&scratch.output_heads), 0);
                    enc.set_buffer(7, Some(&cp.buf), GpuConstantPool::N_HEADS as u64);
                    enc.set_buffer(8, Some(&cp.buf), GpuConstantPool::KEY_DIM as u64);
                    enc.set_buffer(9, Some(&cp.buf), GpuConstantPool::VALUE_DIM as u64);
                    enc.set_buffer(10, Some(&cp.buf), GpuConstantPool::HEAD_DIM_K as u64);
                    enc.set_buffer(11, Some(&cp.buf), GpuConstantPool::V_PER_HEAD as u64);
                    enc.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
                    enc.end_encoding();
                }

                // --- RMSNorm + gated output ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.rmsnorm_gated_pipeline);
                    enc.set_buffer(0, Some(&scratch.output_heads), 0);
                    enc.set_buffer(1, Some(&gw.ssm_norm_buf), 0);
                    enc.set_buffer(2, Some(&scratch.gate_out), 0);
                    enc.set_buffer(3, Some(&scratch.normed), 0);
                    enc.set_buffer(4, Some(&cp.buf), GpuConstantPool::CHUNK as u64);
                    enc.set_buffer(5, Some(&cp.buf), GpuConstantPool::HEAD_DIM_K as u64);
                    enc.set_buffer(6, Some(&cp.buf), GpuConstantPool::N_HEADS as u64);
                    enc.set_buffer(7, Some(&cp.buf), GpuConstantPool::EPS as u64);
                    enc.set_buffer(8, Some(&cp.buf), GpuConstantPool::HAS_GATE as u64);
                    enc.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
                    enc.end_encoding();
                }

                // --- GEMV: SSM out [chunk→dim] ---
                encode_gemv_prealloc(cmd_buf, gpu, &gw.ssm_out, &scratch.normed, 0, &scratch.ssm_out, 0, &gw.ssm_out_params);

                // --- Residual: hidden += ssm_out ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.residual_add_pipeline);
                    enc.set_buffer(0, Some(&scratch.hidden), 0);
                    enc.set_buffer(1, Some(&scratch.ssm_out), 0);
                    enc.set_buffer(2, Some(&cp.buf), GpuConstantPool::DIM as u64);
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
                    enc.set_buffer(3, Some(&cp.buf), GpuConstantPool::DIM as u64);
                    enc.set_buffer(4, Some(&cp.buf), GpuConstantPool::EPS as u64);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
                    enc.end_encoding();
                }

                // --- FFN: gate + up GEMVs ---
                encode_gemv_prealloc(cmd_buf, gpu, &gw.ffn_gate, &scratch.xnorm, 0, &scratch.ffn_h1, 0, &gw.ffn_gate_params);
                encode_gemv_prealloc(cmd_buf, gpu, &gw.ffn_up, &scratch.xnorm, 0, &scratch.ssm_out, 0, &gw.ffn_up_params);

                // --- SiLU(gate) * up ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.silu_pipeline);
                    enc.set_buffer(0, Some(&scratch.ffn_h1), 0);
                    enc.set_buffer(1, Some(&scratch.ssm_out), 0);
                    enc.set_buffer(2, Some(&cp.buf), GpuConstantPool::HIDDEN_DIM as u64);
                    enc.dispatch_threads(MTLSize::new(hidden_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
                    enc.end_encoding();
                }

                // --- FFN down GEMV ---
                encode_gemv_prealloc(cmd_buf, gpu, &gw.ffn_down, &scratch.ffn_h1, 0, &scratch.ffn_out, 0, &gw.ffn_down_params);

                // --- Residual: hidden += ffn_out ---
                {
                    let enc = cmd_buf.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&gpu.residual_add_pipeline);
                    enc.set_buffer(0, Some(&scratch.hidden), 0);
                    enc.set_buffer(1, Some(&scratch.ffn_out), 0);
                    enc.set_buffer(2, Some(&cp.buf), GpuConstantPool::DIM as u64);
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
    // RMSNorm hidden → input (final_norm uploaded once as scratch buffer)
    // TODO: pre-upload final_norm at model load to avoid per-token alloc
    {
        let final_norm = &model.final_norm;
        let fn_buf = gpu.device.new_buffer_with_data(
            final_norm.as_ptr() as *const _, (final_norm.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&gpu.rmsnorm_simple_pipeline);
        enc.set_buffer(0, Some(&scratch.hidden), 0);
        enc.set_buffer(1, Some(&fn_buf), 0);
        enc.set_buffer(2, Some(&gpu.scratch_input), 0);
        enc.set_buffer(3, Some(&cp.buf), GpuConstantPool::DIM as u64);
        enc.set_buffer(4, Some(&cp.buf), GpuConstantPool::EPS as u64);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
    }

    // LM head GEMV
    encode_gemv_prealloc(cmd_buf, gpu, &gpu_weights.lm_head, &gpu.scratch_input, 0, &gpu.scratch_output, 0, &gpu_weights.lm_head_params);

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
// params_buf: pre-allocated [n_blocks, m] buffer (from GpuDeltaNetLayerWeights)
fn encode_gemv_prealloc(
    cmd_buf: &CommandBufferRef,
    gpu: &GpuContext,
    weight: &GpuBuffer,
    input_buf: &Buffer,
    input_offset: usize,
    output_buf: &Buffer,
    output_offset: usize,
    params_buf: &Buffer, // pre-allocated at model load, contains [n_blocks, m]
) {
    let m = weight.m as u32;
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&gpu.gemv_pipeline);
    enc.set_buffer(0, Some(&weight.buffer), 0);
    enc.set_buffer(1, Some(input_buf), (input_offset * 4) as u64);
    enc.set_buffer(2, Some(output_buf), (output_offset * 4) as u64);
    enc.set_buffer(3, Some(params_buf), 0);
    enc.set_buffer(4, Some(params_buf), 4);
    enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(128, 1, 1));
    enc.end_encoding();
}
