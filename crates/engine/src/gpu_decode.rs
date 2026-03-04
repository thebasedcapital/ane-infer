//! GPU-accelerated forward token decode for Qwen3.5 DeltaNet hybrid.
//!
//! Uses Metal computational graph to batch GEMVs into single GPU dispatches.

use anyhow::Result;

use crate::deltanet_cache::{DeltaNetLayerState, HybridCache};
use crate::kv_cache::LayerKvCache;
use crate::metal_graph::{GemvOp, GpuBuffer, GpuContext, GpuGraph, SiluMulOp};
use crate::model::{
    DeltaNetLayerWeights, FullAttnLayerWeights, HybridLayerWeights, LayerType, Qwen35Config,
    Qwen35ModelWeights,
};
use crate::q8_gemv::q8_gemv;
use crate::scratch::{vec_scale, vec_silu_inplace, vec_silu_mul_inplace};

#[inline]
fn softplus(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let inv_norm = 1.0 / (norm_sq + 1e-12).sqrt();
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
}

fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let dim = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms * w[i];
    }
}

fn mat_vec_transpose(s: &[f32], x: &[f32], y: &mut [f32], rows: usize, cols: usize, stride: usize) {
    for col in 0..cols {
        let mut dot = 0.0f32;
        for row in 0..rows {
            if row < x.len() && row * stride + col < s.len() {
                dot += s[row * stride + col] * x[row];
            }
        }
        y[col] = dot;
    }
}

fn outer_product_add(
    s: &mut [f32],
    k: &[f32],
    delta: &[f32],
    rows: usize,
    cols: usize,
    stride: usize,
) {
    for row in 0..rows.min(k.len()) {
        for col in 0..cols.min(delta.len()) {
            if row * stride + col < s.len() {
                s[row * stride + col] += k[row] * delta[col];
            }
        }
    }
}

pub struct GpuDeltaNetLayerWeights {
    pub qkv: GpuBuffer,
    pub attn_gate: GpuBuffer,
    pub ssm_out: GpuBuffer,
    pub ffn_gate: GpuBuffer,
    pub ffn_up: GpuBuffer,
    pub ffn_down: GpuBuffer,
}

pub struct GpuFullAttnLayerWeights {
    pub wq: GpuBuffer,
    pub wk: GpuBuffer,
    pub wv: GpuBuffer,
    pub wo: GpuBuffer,
    pub ffn_gate: GpuBuffer,
    pub ffn_up: GpuBuffer,
    pub ffn_down: GpuBuffer,
}

pub enum GpuLayerWeights {
    DeltaNet(GpuDeltaNetLayerWeights),
    FullAttention(GpuFullAttnLayerWeights),
}

pub struct GpuModelWeights {
    pub layers: Vec<GpuLayerWeights>,
    pub lm_head: GpuBuffer,
}

pub fn upload_model_weights(gpu: &GpuContext, model: &Qwen35ModelWeights) -> GpuModelWeights {
    let mut layers = Vec::with_capacity(model.layers.len());

    for lw in &model.layers {
        match lw {
            HybridLayerWeights::DeltaNet(w) => {
                layers.push(GpuLayerWeights::DeltaNet(GpuDeltaNetLayerWeights {
                    qkv: gpu.upload_q8_weights(&w.qkv),
                    attn_gate: gpu.upload_q8_weights(&w.attn_gate),
                    ssm_out: gpu.upload_q8_weights(&w.ssm_out),
                    ffn_gate: gpu.upload_q8_weights(&w.ffn.gate),
                    ffn_up: gpu.upload_q8_weights(&w.ffn.up),
                    ffn_down: gpu.upload_q8_weights(&w.ffn.down),
                }));
            }
            HybridLayerWeights::FullAttention(w) => {
                layers.push(GpuLayerWeights::FullAttention(GpuFullAttnLayerWeights {
                    wq: gpu.upload_q8_weights(&w.wq),
                    wk: gpu.upload_q8_weights(&w.wk),
                    wv: gpu.upload_q8_weights(&w.wv),
                    wo: gpu.upload_q8_weights(&w.wo),
                    ffn_gate: gpu.upload_q8_weights(&w.ffn.gate),
                    ffn_up: gpu.upload_q8_weights(&w.ffn.up),
                    ffn_down: gpu.upload_q8_weights(&w.ffn.down),
                }));
            }
        }
    }

    let lm_head = gpu.upload_q8_weights(&model.lm_head);

    GpuModelWeights { layers, lm_head }
}

struct ScratchLayout {
    input_hidden: usize,
    input_ffn_in: usize,
    output_qkv: usize,
    output_gate: usize,
    output_ssm_out: usize,
    output_ffn_gate: usize,
    output_ffn_up: usize,
    output_ffn_down: usize,
    output_logits: usize,
    dim: usize,
    hidden_dim: usize,
    inner_size: usize,
    vocab_size: usize,
}

impl ScratchLayout {
    fn new(dim: usize, hidden_dim: usize, inner_size: usize, vocab_size: usize) -> Self {
        let input_hidden = 0;
        let input_ffn_in = dim;
        let output_qkv = 0;
        let output_gate = inner_size;
        let output_ssm_out = inner_size + dim;
        let output_ffn_gate = inner_size + 2 * dim;
        let output_ffn_up = inner_size + 2 * dim + hidden_dim;
        let output_ffn_down = inner_size + 2 * dim + 2 * hidden_dim;
        let output_logits = inner_size + 2 * dim + 3 * hidden_dim;

        Self {
            input_hidden,
            input_ffn_in,
            output_qkv,
            output_gate,
            output_ssm_out,
            output_ffn_gate,
            output_ffn_up,
            output_ffn_down,
            output_logits,
            dim,
            hidden_dim,
            inner_size,
            vocab_size,
        }
    }

    fn input_bytes(&self) -> usize {
        (self.input_ffn_in + self.dim) * 4
    }

    fn output_bytes(&self) -> usize {
        (self.output_logits + self.vocab_size) * 4
    }
}

pub fn qwen35_forward_token_gpu(
    model: &Qwen35ModelWeights,
    cache: &mut HybridCache,
    token_id: u32,
    gpu: &GpuContext,
    gpu_weights: &GpuModelWeights,
) -> Result<Vec<f32>> {
    let cfg = &model.config;
    let dim = cfg.base.dim;
    let hidden_dim = cfg.base.hidden_dim;
    let inner_size = dim * 3;
    let eps = cfg.base.rms_norm_eps;
    let vocab_size = cfg.base.vocab_size;

    let layout = ScratchLayout::new(dim, hidden_dim, inner_size, vocab_size);

    let input_ptr = gpu.scratch_input_ptr();
    let output_ptr = gpu.scratch_output_ptr();

    unsafe {
        let emb_ptr = model.embedding.as_ptr().add((token_id as usize) * dim);
        std::ptr::copy_nonoverlapping(emb_ptr, input_ptr.add(layout.input_hidden), dim);
    }

    let mut hidden: Vec<f32> = vec![0.0; dim];
    unsafe {
        std::ptr::copy_nonoverlapping(input_ptr.add(layout.input_hidden), hidden.as_mut_ptr(), dim);
    }

    let mut graph = GpuGraph::new(gpu);
    let params_buf = gpu.create_params_buffer();

    for (layer_idx, (layer_type, layer_cache_idx)) in cache.layer_map.iter().enumerate() {
        let hidden_ptr = unsafe { input_ptr.add(layout.input_hidden) };
        unsafe {
            std::ptr::copy_nonoverlapping(hidden.as_ptr(), hidden_ptr, dim);
        }

        match (
            layer_type,
            &model.layers[layer_idx],
            &gpu_weights.layers[layer_idx],
        ) {
            (
                LayerType::DeltaNet,
                HybridLayerWeights::DeltaNet(w),
                GpuLayerWeights::DeltaNet(gw),
            ) => {
                let state = &mut cache.deltanet_states[*layer_cache_idx];
                let scratch = &mut cache.scratch;

                graph.clear();
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.qkv.clone_buffer(),
                    input_offset: layout.input_hidden,
                    output_offset: layout.output_qkv,
                    input_len: dim,
                    output_len: inner_size,
                });
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.attn_gate.clone_buffer(),
                    input_offset: layout.input_hidden,
                    output_offset: layout.output_gate,
                    input_len: dim,
                    output_len: dim,
                });
                graph.execute_with_params(&params_buf)?;

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output_ptr.add(layout.output_qkv),
                        scratch.qkvz.as_mut_ptr(),
                        inner_size,
                    );
                    std::ptr::copy_nonoverlapping(
                        output_ptr.add(layout.output_gate),
                        scratch.gate_out.as_mut_ptr(),
                        dim,
                    );
                }

                state.conv_shift_and_append(&scratch.qkvz[..inner_size]);
                state.conv_apply(&w.ssm_conv1d, &mut scratch.conv_out[..inner_size]);

                vec_silu_inplace(&mut scratch.conv_out[..inner_size]);

                let chunk = inner_size / 3;
                let (q_flat, k_flat, v_flat) = (
                    &scratch.conv_out[..chunk],
                    &scratch.conv_out[chunk..2 * chunk],
                    &scratch.conv_out[2 * chunk..],
                );

                scratch.q[..chunk].copy_from_slice(q_flat);
                scratch.k[..chunk].copy_from_slice(k_flat);

                let n_heads = state.n_heads;
                let key_dim = state.key_dim;
                let head_dim_k = chunk / n_heads;
                let scale = 1.0 / (key_dim as f32).sqrt();

                for h in 0..n_heads {
                    let off = h * head_dim_k;
                    l2_normalize(&mut scratch.q[off..off + head_dim_k]);
                    l2_normalize(&mut scratch.k[off..off + head_dim_k]);
                    vec_scale(&mut scratch.q[off..off + head_dim_k], scale);
                }

                q8_gemv(&w.ssm_beta, &hidden, &mut scratch.beta_raw[..n_heads]);
                q8_gemv(&w.ssm_alpha, &hidden, &mut scratch.alpha_raw[..n_heads]);

                for h in 0..n_heads {
                    scratch.beta[h] = sigmoid(scratch.beta_raw[h]);
                    let g = w.ssm_a[h] * softplus(scratch.alpha_raw[h] + w.ssm_dt_bias[h]);
                    scratch.decay[h] = g.exp();
                }

                let v_per_head = chunk / n_heads;
                scratch.output_heads[..chunk].fill(0.0);
                let value_dim = state.value_dim;

                for h in 0..n_heads {
                    let k_h = &scratch.k[h * head_dim_k..(h + 1) * head_dim_k];
                    let v_h = &v_flat[h * v_per_head..(h + 1) * v_per_head];
                    let q_h = &scratch.q[h * head_dim_k..(h + 1) * head_dim_k];
                    let s = state.head_state_mut(h);
                    let kd = key_dim.min(head_dim_k);
                    let vd = value_dim.min(v_per_head);

                    vec_scale(&mut s[..kd * vd], scratch.decay[h]);

                    mat_vec_transpose(s, k_h, &mut scratch.sk[..vd], kd, vd, value_dim);

                    for vi in 0..vd {
                        scratch.delta[vi] = scratch.beta[h] * (v_h[vi] - scratch.sk[vi]);
                    }

                    outer_product_add(s, k_h, &scratch.delta[..vd], kd, vd, value_dim);

                    mat_vec_transpose(s, q_h, &mut scratch.sk[..vd], kd, vd, value_dim);

                    for vi in 0..vd.min(head_dim_k) {
                        scratch.output_heads[h * head_dim_k + vi] = scratch.sk[vi];
                    }
                }

                for h in 0..n_heads {
                    let off = h * head_dim_k;
                    let hslice = &scratch.output_heads[off..off + head_dim_k];
                    let norm_dim = head_dim_k.min(w.ssm_norm.len());
                    rmsnorm(
                        &mut scratch.normed[off..off + head_dim_k],
                        hslice,
                        &w.ssm_norm[..norm_dim],
                        eps,
                    );
                }

                vec_silu_mul_inplace(&mut scratch.normed[..chunk], &scratch.gate_out[..chunk]);

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        scratch.normed.as_ptr(),
                        input_ptr.add(layout.input_ffn_in),
                        chunk.min(dim),
                    );
                }

                graph.clear();
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.ssm_out.clone_buffer(),
                    input_offset: layout.input_ffn_in,
                    output_offset: layout.output_ssm_out,
                    input_len: chunk.min(dim),
                    output_len: dim,
                });
                graph.execute_with_params(&params_buf)?;

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output_ptr.add(layout.output_ssm_out),
                        hidden.as_mut_ptr(),
                        dim,
                    );
                }

                rmsnorm(&mut scratch.xnorm[..dim], &hidden, &w.post_attn_norm, eps);

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        scratch.xnorm.as_ptr(),
                        input_ptr.add(layout.input_ffn_in),
                        dim,
                    );
                }

                // Gate + up projections, then fused SiLU(gate)*up on GPU — no CPU roundtrip.
                graph.clear();
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.ffn_gate.clone_buffer(),
                    input_offset: layout.input_ffn_in,
                    output_offset: layout.output_ffn_gate,
                    input_len: dim,
                    output_len: hidden_dim,
                });
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.ffn_up.clone_buffer(),
                    input_offset: layout.input_ffn_in,
                    output_offset: layout.output_ffn_up,
                    input_len: dim,
                    output_len: hidden_dim,
                });
                graph.add_silu_mul(SiluMulOp {
                    gate_offset: layout.output_ffn_gate,
                    up_offset: layout.output_ffn_up,
                    n: hidden_dim,
                });
                graph.execute_with_params(&params_buf)?;

                // Copy GPU-fused result (gate now holds SiLU(gate)*up) directly into input.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output_ptr.add(layout.output_ffn_gate),
                        input_ptr.add(layout.input_ffn_in),
                        hidden_dim,
                    );
                }

                graph.clear();
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.ffn_down.clone_buffer(),
                    input_offset: layout.input_ffn_in,
                    output_offset: layout.output_ffn_down,
                    input_len: hidden_dim,
                    output_len: dim,
                });
                graph.execute_with_params(&params_buf)?;

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output_ptr.add(layout.output_ffn_down),
                        scratch.ffn_out.as_mut_ptr(),
                        dim,
                    );
                }

                for i in 0..dim {
                    hidden[i] += scratch.ffn_out[i];
                }
            }
            (
                LayerType::FullAttention,
                HybridLayerWeights::FullAttention(w),
                GpuLayerWeights::FullAttention(gw),
            ) => {
                let kv_cache = &mut cache.kv_caches[*layer_cache_idx];
                let scratch = &mut cache.scratch;
                let pos = cache.pos;

                graph.clear();
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.wq.clone_buffer(),
                    input_offset: layout.input_hidden,
                    output_offset: layout.output_qkv,
                    input_len: dim,
                    output_len: w.wq.m,
                });
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.wk.clone_buffer(),
                    input_offset: layout.input_hidden,
                    output_offset: layout.output_qkv + w.wq.m,
                    input_len: dim,
                    output_len: w.wk.m,
                });
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.wv.clone_buffer(),
                    input_offset: layout.input_hidden,
                    output_offset: layout.output_qkv + w.wq.m + w.wk.m,
                    input_len: dim,
                    output_len: w.wv.m,
                });
                graph.execute_with_params(&params_buf)?;

                let q_full_dim = w.wq.m;
                let kv_dim = w.wk.m;
                let n_heads = cfg.base.n_heads;
                let n_kv_heads = cfg.base.n_kv_heads;
                let head_dim = cfg.base.head_dim;
                let q_only_dim = n_heads * head_dim;

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output_ptr.add(layout.output_qkv),
                        scratch.q_full.as_mut_ptr(),
                        q_full_dim,
                    );
                    std::ptr::copy_nonoverlapping(
                        output_ptr.add(layout.output_qkv + w.wq.m),
                        scratch.kv_k.as_mut_ptr(),
                        kv_dim,
                    );
                    std::ptr::copy_nonoverlapping(
                        output_ptr.add(layout.output_qkv + w.wq.m + w.wk.m),
                        scratch.kv_v.as_mut_ptr(),
                        kv_dim,
                    );
                }

                for h in 0..n_heads {
                    let src_off = h * head_dim * 2;
                    let dst_off = h * head_dim;
                    scratch.q_only[dst_off..dst_off + head_dim]
                        .copy_from_slice(&scratch.q_full[src_off..src_off + head_dim]);
                    scratch.gate[dst_off..dst_off + head_dim].copy_from_slice(
                        &scratch.q_full[src_off + head_dim..src_off + 2 * head_dim],
                    );
                }

                let q_norm_dim = w.q_norm.len().min(head_dim);
                let k_norm_dim = w.k_norm.len().min(head_dim);

                for h in 0..n_heads {
                    let off = h * head_dim;
                    rmsnorm(
                        &mut scratch.normed[..q_norm_dim],
                        &scratch.q_only[off..off + q_norm_dim],
                        &w.q_norm[..q_norm_dim],
                        eps,
                    );
                    scratch.q_only[off..off + q_norm_dim]
                        .copy_from_slice(&scratch.normed[..q_norm_dim]);
                }
                for h in 0..n_kv_heads {
                    let off = h * head_dim;
                    rmsnorm(
                        &mut scratch.normed[..k_norm_dim],
                        &scratch.kv_k[off..off + k_norm_dim],
                        &w.k_norm[..k_norm_dim],
                        eps,
                    );
                    scratch.kv_k[off..off + k_norm_dim]
                        .copy_from_slice(&scratch.normed[..k_norm_dim]);
                }

                for h in 0..n_heads {
                    for i in (0..head_dim).step_by(2) {
                        let freq = 1.0 / cfg.base.rope_freq_base.powf(i as f32 / head_dim as f32);
                        let val = pos as f32 * freq;
                        let (sin_v, cos_v) = val.sin_cos();
                        let off = h * head_dim + i;
                        let q0 = scratch.q_only[off];
                        let q1 = scratch.q_only[off + 1];
                        scratch.q_only[off] = q0 * cos_v - q1 * sin_v;
                        scratch.q_only[off + 1] = q0 * sin_v + q1 * cos_v;
                    }
                }
                for h in 0..n_kv_heads {
                    for i in (0..head_dim).step_by(2) {
                        let freq = 1.0 / cfg.base.rope_freq_base.powf(i as f32 / head_dim as f32);
                        let val = pos as f32 * freq;
                        let (sin_v, cos_v) = val.sin_cos();
                        let off = h * head_dim + i;
                        let k0 = scratch.kv_k[off];
                        let k1 = scratch.kv_k[off + 1];
                        scratch.kv_k[off] = k0 * cos_v - k1 * sin_v;
                        scratch.kv_k[off + 1] = k0 * sin_v + k1 * cos_v;
                    }
                }

                kv_cache.write_pos(pos, &scratch.kv_k[..kv_dim], &scratch.kv_v[..kv_dim]);

                let heads_per_kv = n_heads / n_kv_heads;
                let seq_so_far = pos + 1;

                scratch.attn_out[..q_only_dim].fill(0.0);

                for h in 0..n_heads {
                    let kv_h = h / heads_per_kv;
                    let k_cache = kv_cache.key_head(kv_h);
                    let v_cache = kv_cache.value_head(kv_h);

                    let mut max_score = f32::NEG_INFINITY;
                    for s in 0..seq_so_far {
                        let mut dot = 0.0f32;
                        for i in 0..head_dim {
                            dot += scratch.q_only[h * head_dim + i] * k_cache[s * head_dim + i];
                        }
                        scratch.scores[s] = dot / (head_dim as f32).sqrt();
                        if scratch.scores[s] > max_score {
                            max_score = scratch.scores[s];
                        }
                    }

                    let mut sm = 0.0f32;
                    for s in 0..seq_so_far {
                        scratch.scores[s] = (scratch.scores[s] - max_score).exp();
                        sm += scratch.scores[s];
                    }
                    for s in 0..seq_so_far {
                        scratch.scores[s] /= sm;
                    }

                    for i in 0..head_dim {
                        let mut val = 0.0f32;
                        for s in 0..seq_so_far {
                            val += scratch.scores[s] * v_cache[s * head_dim + i];
                        }
                        scratch.attn_out[h * head_dim + i] = val;
                    }
                }

                for i in 0..q_only_dim {
                    scratch.attn_out[i] *= sigmoid(scratch.gate[i]);
                }

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        scratch.attn_out.as_ptr(),
                        input_ptr.add(layout.input_ffn_in),
                        q_only_dim,
                    );
                }

                graph.clear();
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.wo.clone_buffer(),
                    input_offset: layout.input_ffn_in,
                    output_offset: layout.output_ssm_out,
                    input_len: q_only_dim,
                    output_len: dim,
                });
                graph.execute_with_params(&params_buf)?;

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output_ptr.add(layout.output_ssm_out),
                        hidden.as_mut_ptr(),
                        dim,
                    );
                }

                rmsnorm(&mut scratch.xnorm[..dim], &hidden, &w.post_attn_norm, eps);

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        scratch.xnorm.as_ptr(),
                        input_ptr.add(layout.input_ffn_in),
                        dim,
                    );
                }

                graph.clear();
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.ffn_gate.clone_buffer(),
                    input_offset: layout.input_ffn_in,
                    output_offset: layout.output_ffn_gate,
                    input_len: dim,
                    output_len: hidden_dim,
                });
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.ffn_up.clone_buffer(),
                    input_offset: layout.input_ffn_in,
                    output_offset: layout.output_ffn_up,
                    input_len: dim,
                    output_len: hidden_dim,
                });
                // Fused SiLU(gate)*up on GPU — appended to same graph batch.
                graph.add_silu_mul(SiluMulOp {
                    gate_offset: layout.output_ffn_gate,
                    up_offset: layout.output_ffn_up,
                    n: hidden_dim,
                });
                graph.execute_with_params(&params_buf)?;

                // GPU-fused result lives in output_ffn_gate; copy directly to input.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output_ptr.add(layout.output_ffn_gate),
                        input_ptr.add(layout.input_ffn_in),
                        hidden_dim,
                    );
                }

                graph.clear();
                graph.add_gemv(GemvOp {
                    weight_buffer: gw.ffn_down.clone_buffer(),
                    input_offset: layout.input_ffn_in,
                    output_offset: layout.output_ffn_down,
                    input_len: hidden_dim,
                    output_len: dim,
                });
                graph.execute_with_params(&params_buf)?;

                let scratch = &mut cache.scratch;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output_ptr.add(layout.output_ffn_down),
                        scratch.ffn_out.as_mut_ptr(),
                        dim,
                    );
                }

                for i in 0..dim {
                    hidden[i] += scratch.ffn_out[i];
                }
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Layer weight type mismatch at layer {}",
                    layer_idx
                ));
            }
        }
    }

    rmsnorm(
        &mut cache.scratch.xnorm[..dim],
        &hidden,
        &model.final_norm,
        eps,
    );

    unsafe {
        std::ptr::copy_nonoverlapping(
            cache.scratch.xnorm.as_ptr(),
            input_ptr.add(layout.input_ffn_in),
            dim,
        );
    }

    graph.clear();
    graph.add_gemv(GemvOp {
        weight_buffer: gpu_weights.lm_head.clone_buffer(),
        input_offset: layout.input_ffn_in,
        output_offset: layout.output_logits,
        input_len: dim,
        output_len: vocab_size,
    });
    graph.execute_with_params(&params_buf)?;

    unsafe {
        std::ptr::copy_nonoverlapping(
            output_ptr.add(layout.output_logits),
            cache.scratch.logits.as_mut_ptr(),
            vocab_size,
        );
    }

    cache.advance(1);

    Ok(cache.scratch.logits.clone())
}

trait CloneBuffer {
    fn clone_buffer(&self) -> GpuBuffer;
}

impl CloneBuffer for GpuBuffer {
    fn clone_buffer(&self) -> GpuBuffer {
        GpuBuffer {
            buffer: self.buffer.clone(),
            m: self.m,
            n: self.n,
            n_blocks: self.n_blocks,
        }
    }
}
