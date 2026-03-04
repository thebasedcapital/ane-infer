//! Gated DeltaNet single-token decode (recurrent mode).
//!
//! Zero-allocation implementation with pre-allocated scratch buffers.

use crate::deltanet_cache::DeltaNetLayerState;
use crate::model::DeltaNetLayerWeights;
use crate::q8_gemv::q8_gemv;
use crate::scratch::{vec_scale, vec_silu_inplace, vec_silu_mul_inplace, ScratchBuffers};

#[inline]
fn softplus(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let inv_norm = 1.0 / (norm_sq + 1e-12).sqrt();
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
}

#[cfg(target_arch = "aarch64")]
fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let dim = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms * w[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let dim = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms * w[i];
    }
}

/// S^T @ x  →  y   (rows=kd, cols=vd, stride=value_dim)
/// Each output element y[col] = sum_row  S[row*stride+col] * x[row]
#[cfg(target_arch = "aarch64")]
fn mat_vec_transpose(s: &[f32], x: &[f32], y: &mut [f32], rows: usize, cols: usize, stride: usize) {
    use std::arch::aarch64::*;
    // Process 4 output columns at a time.
    let col4 = (cols / 4) * 4;
    unsafe {
        // --- vectorised columns ---
        let mut col = 0usize;
        while col < col4 {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);

            // Unroll ki by 4
            let row4 = (rows / 4) * 4;
            let mut row = 0usize;
            while row < row4 {
                // Load 4 consecutive x values
                let x_vec = vld1q_f32(x.as_ptr().add(row));
                // Load S[row+0..3][col..col+3] – 4 separate rows, same 4 cols
                let s0 = vld1q_f32(s.as_ptr().add(row * stride + col));
                let s1 = vld1q_f32(s.as_ptr().add((row + 1) * stride + col));
                let s2 = vld1q_f32(s.as_ptr().add((row + 2) * stride + col));
                let s3 = vld1q_f32(s.as_ptr().add((row + 3) * stride + col));
                // Broadcast each x scalar and fma into accumulators
                acc0 = vfmaq_laneq_f32::<0>(acc0, s0, x_vec);
                acc1 = vfmaq_laneq_f32::<1>(acc1, s1, x_vec);
                acc0 = vfmaq_laneq_f32::<2>(acc0, s2, x_vec);
                acc1 = vfmaq_laneq_f32::<3>(acc1, s3, x_vec);
                row += 4;
            }
            // Remaining rows (scalar)
            let mut total = vaddq_f32(acc0, acc1);
            while row < rows {
                let xv = vdupq_n_f32(x[row]);
                let sv = vld1q_f32(s.as_ptr().add(row * stride + col));
                total = vfmaq_f32(total, sv, xv);
                row += 1;
            }
            // Store 4 outputs
            vst1q_f32(y.as_mut_ptr().add(col), total);
            col += 4;
        }
        // --- scalar tail ---
        while col < cols {
            let mut dot = 0.0f32;
            for row in 0..rows {
                dot += s[row * stride + col] * x[row];
            }
            y[col] = dot;
            col += 1;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
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

/// S += k (outer) delta   row=ki, col=vi
#[cfg(target_arch = "aarch64")]
fn outer_product_add(
    s: &mut [f32],
    k: &[f32],
    delta: &[f32],
    rows: usize,
    cols: usize,
    stride: usize,
) {
    use std::arch::aarch64::*;
    let col4 = (cols / 4) * 4;
    unsafe {
        for row in 0..rows.min(k.len()) {
            let kv = vdupq_n_f32(k[row]);
            let s_row = s.as_mut_ptr().add(row * stride);
            let mut col = 0usize;
            while col < col4 {
                let sv = vld1q_f32(s_row.add(col));
                let dv = vld1q_f32(delta.as_ptr().add(col));
                vst1q_f32(s_row.add(col), vfmaq_f32(sv, kv, dv));
                col += 4;
            }
            while col < cols.min(delta.len()) {
                *s_row.add(col) += k[row] * delta[col];
                col += 1;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
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

/// Decay: S[..kd*vd] *= decay — use NEON for the contiguous slice.
#[cfg(target_arch = "aarch64")]
fn vec_scale_neon(s: &mut [f32], scalar: f32) {
    use std::arch::aarch64::*;
    let n = s.len();
    let n4 = (n / 4) * 4;
    unsafe {
        let sv = vdupq_n_f32(scalar);
        let mut i = 0usize;
        while i < n4 {
            let v = vld1q_f32(s.as_ptr().add(i));
            vst1q_f32(s.as_mut_ptr().add(i), vmulq_f32(v, sv));
            i += 4;
        }
        while i < n {
            s[i] *= scalar;
            i += 1;
        }
    }
}

/// DeltaNet decode step using scratch buffers (zero allocation).
pub fn deltanet_decode_step_scratch(
    hidden: &[f32],
    state: &mut DeltaNetLayerState,
    weights: &DeltaNetLayerWeights,
    eps: f32,
    scratch: &mut ScratchBuffers,
) {
    let inner_size = state.inner_size;
    let n_heads = state.n_heads;
    let key_dim = state.key_dim;
    let value_dim = state.value_dim;
    let chunk = inner_size / 3;
    let head_dim_k = chunk / n_heads;

    q8_gemv(&weights.qkv, hidden, &mut scratch.qkvz[..inner_size]);

    state.conv_shift_and_append(&scratch.qkvz[..inner_size]);
    state.conv_apply(&weights.ssm_conv1d, &mut scratch.conv_out[..inner_size]);

    vec_silu_inplace(&mut scratch.conv_out[..inner_size]);

    let (q_flat, k_flat, v_flat) = (
        &scratch.conv_out[..chunk],
        &scratch.conv_out[chunk..2 * chunk],
        &scratch.conv_out[2 * chunk..],
    );

    scratch.q[..chunk].copy_from_slice(q_flat);
    scratch.k[..chunk].copy_from_slice(k_flat);

    let scale = 1.0 / (key_dim as f32).sqrt();
    for h in 0..n_heads {
        let off = h * head_dim_k;
        l2_normalize(&mut scratch.q[off..off + head_dim_k]);
        l2_normalize(&mut scratch.k[off..off + head_dim_k]);
        vec_scale(&mut scratch.q[off..off + head_dim_k], scale);
    }

    q8_gemv(&weights.ssm_beta, hidden, &mut scratch.beta_raw[..n_heads]);
    q8_gemv(
        &weights.ssm_alpha,
        hidden,
        &mut scratch.alpha_raw[..n_heads],
    );

    for h in 0..n_heads {
        scratch.beta[h] = sigmoid(scratch.beta_raw[h]);
        let g = weights.ssm_a[h] * softplus(scratch.alpha_raw[h] + weights.ssm_dt_bias[h]);
        scratch.decay[h] = g.exp();
    }

    let v_per_head = chunk / n_heads;
    scratch.output_heads[..chunk].fill(0.0);

    for h in 0..n_heads {
        let k_h = &scratch.k[h * head_dim_k..(h + 1) * head_dim_k];
        let v_h = &v_flat[h * v_per_head..(h + 1) * v_per_head];
        let q_h = &scratch.q[h * head_dim_k..(h + 1) * head_dim_k];
        let s = state.head_state_mut(h);
        let kd = key_dim.min(head_dim_k);
        let vd = value_dim.min(v_per_head);

        #[cfg(target_arch = "aarch64")]
        vec_scale_neon(&mut s[..kd * vd], scratch.decay[h]);
        #[cfg(not(target_arch = "aarch64"))]
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

    q8_gemv(&weights.attn_gate, hidden, &mut scratch.gate_out);

    for h in 0..n_heads {
        let off = h * head_dim_k;
        let hslice = &scratch.output_heads[off..off + head_dim_k];
        let norm_dim = head_dim_k.min(weights.ssm_norm.len());
        rmsnorm(
            &mut scratch.normed[off..off + head_dim_k],
            hslice,
            &weights.ssm_norm[..norm_dim],
            eps,
        );
    }

    vec_silu_mul_inplace(&mut scratch.normed[..chunk], &scratch.gate_out[..chunk]);

    q8_gemv(
        &weights.ssm_out,
        &scratch.normed[..chunk],
        &mut scratch.ffn_out,
    );
}

/// Full attention decode step using scratch buffers.
pub fn full_attn_decode_step_scratch(
    hidden: &[f32],
    pos: usize,
    kv_cache: &mut super::kv_cache::LayerKvCache,
    weights: &crate::model::FullAttnLayerWeights,
    config: &crate::model::ModelConfig,
    scratch: &mut ScratchBuffers,
) {
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let q_full_dim = weights.wq.m;
    let kv_dim = weights.wk.m;
    let q_only_dim = n_heads * head_dim;

    q8_gemv(&weights.wq, hidden, &mut scratch.q_full[..q_full_dim]);
    q8_gemv(&weights.wk, hidden, &mut scratch.kv_k[..kv_dim]);
    q8_gemv(&weights.wv, hidden, &mut scratch.kv_v[..kv_dim]);

    for h in 0..n_heads {
        let src_off = h * head_dim * 2;
        let dst_off = h * head_dim;
        scratch.q_only[dst_off..dst_off + head_dim]
            .copy_from_slice(&scratch.q_full[src_off..src_off + head_dim]);
        scratch.gate[dst_off..dst_off + head_dim]
            .copy_from_slice(&scratch.q_full[src_off + head_dim..src_off + 2 * head_dim]);
    }

    let q_norm_dim = weights.q_norm.len().min(head_dim);
    let k_norm_dim = weights.k_norm.len().min(head_dim);

    for h in 0..n_heads {
        let off = h * head_dim;
        rmsnorm(
            &mut scratch.normed[..q_norm_dim],
            &scratch.q_only[off..off + q_norm_dim],
            &weights.q_norm[..q_norm_dim],
            1e-6,
        );
        scratch.q_only[off..off + q_norm_dim].copy_from_slice(&scratch.normed[..q_norm_dim]);
    }
    for h in 0..n_kv_heads {
        let off = h * head_dim;
        rmsnorm(
            &mut scratch.normed[..k_norm_dim],
            &scratch.kv_k[off..off + k_norm_dim],
            &weights.k_norm[..k_norm_dim],
            1e-6,
        );
        scratch.kv_k[off..off + k_norm_dim].copy_from_slice(&scratch.normed[..k_norm_dim]);
    }

    for h in 0..n_heads {
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / config.rope_freq_base.powf(i as f32 / head_dim as f32);
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
            let freq = 1.0 / config.rope_freq_base.powf(i as f32 / head_dim as f32);
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

    q8_gemv(
        &weights.wo,
        &scratch.attn_out[..q_only_dim],
        &mut scratch.ffn_out,
    );
}

pub fn deltanet_decode_step(
    hidden: &[f32],
    state: &mut DeltaNetLayerState,
    weights: &DeltaNetLayerWeights,
    eps: f32,
) -> Vec<f32> {
    let dim = hidden.len();
    let inner_size = state.inner_size;
    let n_heads = state.n_heads;
    let key_dim = state.key_dim;
    let value_dim = state.value_dim;

    let mut qkvz = vec![0.0f32; inner_size];
    q8_gemv(&weights.qkv, hidden, &mut qkvz);

    state.conv_shift_and_append(&qkvz);
    let mut conv_out = vec![0.0f32; inner_size];
    state.conv_apply(&weights.ssm_conv1d, &mut conv_out);

    for v in conv_out.iter_mut() {
        *v = silu(*v);
    }

    let chunk = inner_size / 3;
    let q_flat = &conv_out[..chunk];
    let k_flat = &conv_out[chunk..2 * chunk];
    let v_flat = &conv_out[2 * chunk..];

    let head_dim_k = chunk / n_heads;

    let mut q = q_flat.to_vec();
    let mut k = k_flat.to_vec();
    let scale = 1.0 / (key_dim as f32).sqrt();
    for h in 0..n_heads {
        let off = h * head_dim_k;
        l2_normalize(&mut q[off..off + head_dim_k]);
        l2_normalize(&mut k[off..off + head_dim_k]);
        vec_scale(&mut q[off..off + head_dim_k], scale);
    }

    let mut beta_raw = vec![0.0f32; n_heads];
    let mut alpha_raw = vec![0.0f32; n_heads];
    q8_gemv(&weights.ssm_beta, hidden, &mut beta_raw);
    q8_gemv(&weights.ssm_alpha, hidden, &mut alpha_raw);

    let mut beta = vec![0.0f32; n_heads];
    let mut decay = vec![0.0f32; n_heads];
    for h in 0..n_heads {
        beta[h] = sigmoid(beta_raw[h]);
        let g = weights.ssm_a[h] * softplus(alpha_raw[h] + weights.ssm_dt_bias[h]);
        decay[h] = g.exp();
    }

    let v_per_head = v_flat.len() / n_heads;
    let mut output_heads = vec![0.0f32; chunk];

    for h in 0..n_heads {
        let k_h = &k[h * head_dim_k..(h + 1) * head_dim_k];
        let v_h = &v_flat[h * v_per_head..(h + 1) * v_per_head];
        let q_h = &q[h * head_dim_k..(h + 1) * head_dim_k];
        let s = state.head_state_mut(h);
        let kd = key_dim.min(head_dim_k);
        let vd = value_dim.min(v_per_head);

        #[cfg(target_arch = "aarch64")]
        vec_scale_neon(&mut s[..kd * vd], decay[h]);
        #[cfg(not(target_arch = "aarch64"))]
        vec_scale(&mut s[..kd * vd], decay[h]);

        let mut sk = vec![0.0f32; vd];
        mat_vec_transpose(s, k_h, &mut sk, kd, vd, value_dim);

        let mut delta = vec![0.0f32; vd];
        for vi in 0..vd {
            delta[vi] = beta[h] * (v_h[vi] - sk[vi]);
        }

        outer_product_add(s, k_h, &delta, kd, vd, value_dim);

        mat_vec_transpose(s, q_h, &mut sk, kd, vd, value_dim);
        for vi in 0..vd.min(head_dim_k) {
            output_heads[h * head_dim_k + vi] = sk[vi];
        }
    }

    let mut gate_out = vec![0.0f32; dim];
    q8_gemv(&weights.attn_gate, hidden, &mut gate_out);

    let mut normed = vec![0.0f32; chunk];
    for h in 0..n_heads {
        let off = h * head_dim_k;
        let hslice = &output_heads[off..off + head_dim_k];
        let norm_dim = head_dim_k.min(weights.ssm_norm.len());
        rmsnorm(
            &mut normed[off..off + head_dim_k],
            hslice,
            &weights.ssm_norm[..norm_dim],
            eps,
        );
    }

    // gate_out gets SiLU applied, then multiply into normed
    for i in 0..chunk.min(dim) {
        normed[i] *= gate_out[i] / (1.0 + (-gate_out[i]).exp());
    }

    let mut out = vec![0.0f32; dim];
    q8_gemv(&weights.ssm_out, &normed[..dim.min(chunk)], &mut out);

    out
}

pub fn full_attn_decode_step(
    hidden: &[f32],
    pos: usize,
    kv_cache: &mut super::kv_cache::LayerKvCache,
    weights: &crate::model::FullAttnLayerWeights,
    config: &crate::model::ModelConfig,
) -> Vec<f32> {
    let dim = config.dim;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;

    let q_full_dim = weights.wq.m;
    let kv_dim = weights.wk.m;
    let q_only_dim = n_heads * head_dim;

    let mut q_full = vec![0.0f32; q_full_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];

    q8_gemv(&weights.wq, hidden, &mut q_full);
    q8_gemv(&weights.wk, hidden, &mut k);
    q8_gemv(&weights.wv, hidden, &mut v);

    let mut q = vec![0.0f32; q_only_dim];
    let mut gate = vec![0.0f32; q_only_dim];
    for h in 0..n_heads {
        let src_off = h * head_dim * 2;
        let dst_off = h * head_dim;
        q[dst_off..dst_off + head_dim].copy_from_slice(&q_full[src_off..src_off + head_dim]);
        gate[dst_off..dst_off + head_dim]
            .copy_from_slice(&q_full[src_off + head_dim..src_off + 2 * head_dim]);
    }

    for h in 0..n_heads {
        let off = h * head_dim;
        let norm_d = head_dim.min(weights.q_norm.len());
        let mut normed = vec![0.0f32; head_dim];
        rmsnorm(
            &mut normed[..norm_d],
            &q[off..off + norm_d],
            &weights.q_norm[..norm_d],
            1e-6,
        );
        q[off..off + norm_d].copy_from_slice(&normed[..norm_d]);
    }
    for h in 0..n_kv_heads {
        let off = h * head_dim;
        let norm_d = head_dim.min(weights.k_norm.len());
        let mut normed = vec![0.0f32; head_dim];
        rmsnorm(
            &mut normed[..norm_d],
            &k[off..off + norm_d],
            &weights.k_norm[..norm_d],
            1e-6,
        );
        k[off..off + norm_d].copy_from_slice(&normed[..norm_d]);
    }

    for h in 0..n_heads {
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / config.rope_freq_base.powf(i as f32 / head_dim as f32);
            let val = pos as f32 * freq;
            let (sin_v, cos_v) = val.sin_cos();
            let off = h * head_dim + i;
            let q0 = q[off];
            let q1 = q[off + 1];
            q[off] = q0 * cos_v - q1 * sin_v;
            q[off + 1] = q0 * sin_v + q1 * cos_v;
        }
    }
    for h in 0..n_kv_heads {
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / config.rope_freq_base.powf(i as f32 / head_dim as f32);
            let val = pos as f32 * freq;
            let (sin_v, cos_v) = val.sin_cos();
            let off = h * head_dim + i;
            let k0 = k[off];
            let k1 = k[off + 1];
            k[off] = k0 * cos_v - k1 * sin_v;
            k[off + 1] = k0 * sin_v + k1 * cos_v;
        }
    }

    kv_cache.write_pos(pos, &k, &v);

    let heads_per_kv = n_heads / n_kv_heads;
    let seq_so_far = pos + 1;
    let mut attn_out = vec![0.0f32; q_only_dim];

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        let k_cache = kv_cache.key_head(kv_h);
        let v_cache = kv_cache.value_head(kv_h);

        let mut scores = vec![0.0f32; seq_so_far];
        let mut max_score = f32::NEG_INFINITY;
        for s in 0..seq_so_far {
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += q[h * head_dim + i] * k_cache[s * head_dim + i];
            }
            scores[s] = dot / (head_dim as f32).sqrt();
            if scores[s] > max_score {
                max_score = scores[s];
            }
        }

        let mut sm = 0.0f32;
        for s in 0..seq_so_far {
            scores[s] = (scores[s] - max_score).exp();
            sm += scores[s];
        }
        for s in 0..seq_so_far {
            scores[s] /= sm;
        }

        for i in 0..head_dim {
            let mut val = 0.0f32;
            for s in 0..seq_so_far {
                val += scores[s] * v_cache[s * head_dim + i];
            }
            attn_out[h * head_dim + i] = val;
        }
    }

    for i in 0..q_only_dim {
        attn_out[i] *= sigmoid(gate[i]);
    }

    let mut out = vec![0.0f32; dim];
    q8_gemv(&weights.wo, &attn_out, &mut out);

    out
}
