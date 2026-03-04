//! CPU Decode Engine — single-token autoregressive generation.
//!
//! Uses Accelerate/BLAS for GEMV (routes to AMX/SME on Apple Silicon automatically).
//! Better than ANE for single-token because:
//! - No dispatch overhead (~1ms per ANE kernel call)
//! - Better cache locality for bandwidth-bound GEMV
//! - ~5W vs ~20W for Metal

use crate::kv_cache::KvCache;
use crate::model::ModelWeights;
use anyhow::Result;

// Accelerate BLAS FFI (cblas_sgemv routes to AMX/SME on M-series)
mod blas {
    unsafe extern "C" {
        pub fn cblas_sgemv(
            order: i32, // CblasRowMajor = 101
            trans: i32, // CblasNoTrans = 111, CblasTrans = 112
            m: i32,     // rows of A
            n: i32,     // cols of A
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
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

/// GEMV via Accelerate: y = W @ x, W[m, n], x[n] → y[m]
fn sgemv(w: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    unsafe {
        blas::cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
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

/// RMSNorm for a single vector
fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32]) {
    let dim = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = 1.0 / (ss + 1e-5_f32).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms * w[i];
    }
}

/// SiLU activation
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Decode a single token: takes the previous token embedding through all layers,
/// returns logits for next token prediction.
pub fn decode_token(
    weights: &ModelWeights,
    kv_cache: &mut KvCache,
    token_id: u32,
) -> Result<Vec<f32>> {
    let cfg = &weights.config;
    let dim = cfg.dim;
    let hidden_dim = cfg.hidden_dim;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let pos = kv_cache.pos;

    // Embed token
    let mut x = vec![0f32; dim];
    x.copy_from_slice(&weights.embedding[token_id as usize * dim..(token_id as usize + 1) * dim]);

    let mut xnorm = vec![0f32; dim];
    let mut q = vec![0f32; dim];
    let mut k = vec![0f32; n_kv_heads * head_dim];
    let mut v = vec![0f32; n_kv_heads * head_dim];
    let mut attn_out = vec![0f32; dim];
    let mut o_out = vec![0f32; dim];
    let mut ffn_in = vec![0f32; dim];
    let mut h1 = vec![0f32; hidden_dim];
    let mut h3 = vec![0f32; hidden_dim];
    let mut ffn_out = vec![0f32; dim];

    let kv_dim = n_kv_heads * head_dim;

    for l in 0..cfg.n_layers {
        let lw = &weights.layers[l];

        // RMSNorm
        rmsnorm(&mut xnorm, &x, &lw.attn_norm);

        // QKV projections via Accelerate BLAS
        sgemv(&lw.wq, &xnorm, &mut q, dim, dim);
        sgemv(&lw.wk, &xnorm, &mut k, kv_dim, dim);
        sgemv(&lw.wv, &xnorm, &mut v, kv_dim, dim);

        // RoPE
        for h in 0..n_heads {
            for i in (0..head_dim).step_by(2) {
                let freq = 1.0 / cfg.rope_freq_base.powf(i as f32 / head_dim as f32);
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
                let freq = 1.0 / cfg.rope_freq_base.powf(i as f32 / head_dim as f32);
                let val = pos as f32 * freq;
                let (sin_v, cos_v) = val.sin_cos();
                let off = h * head_dim + i;
                let k0 = k[off];
                let k1 = k[off + 1];
                k[off] = k0 * cos_v - k1 * sin_v;
                k[off + 1] = k0 * sin_v + k1 * cos_v;
            }
        }

        // Store K, V in cache
        kv_cache.layers[l].write_pos(pos, &k, &v);

        // Attention with cached KV
        let heads_per_kv = n_heads / n_kv_heads;
        let seq_so_far = pos + 1;

        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;
            let k_cache = kv_cache.layers[l].key_head(kv_h);
            let v_cache = kv_cache.layers[l].value_head(kv_h);

            // Q·K^T for this head
            let mut max_score = f32::NEG_INFINITY;
            let mut scores = vec![0f32; seq_so_far];
            for s in 0..seq_so_far {
                let mut dot = 0f32;
                for i in 0..head_dim {
                    dot += q[h * head_dim + i] * k_cache[s * head_dim + i];
                }
                scores[s] = dot / (head_dim as f32).sqrt();
                if scores[s] > max_score {
                    max_score = scores[s];
                }
            }

            // Softmax
            let mut sm = 0f32;
            for s in 0..seq_so_far {
                scores[s] = (scores[s] - max_score).exp();
                sm += scores[s];
            }
            for s in 0..seq_so_far {
                scores[s] /= sm;
            }

            // Weighted sum of values
            for i in 0..head_dim {
                let mut val = 0f32;
                for s in 0..seq_so_far {
                    val += scores[s] * v_cache[s * head_dim + i];
                }
                attn_out[h * head_dim + i] = val;
            }
        }

        // Output projection
        sgemv(&lw.wo, &attn_out, &mut o_out, dim, dim);

        // Residual
        for i in 0..dim {
            x[i] += o_out[i];
        }

        // FFN
        rmsnorm(&mut ffn_in, &x, &lw.ffn_norm);
        sgemv(&lw.w1, &ffn_in, &mut h1, hidden_dim, dim);
        sgemv(&lw.w3, &ffn_in, &mut h3, hidden_dim, dim);

        for i in 0..hidden_dim {
            h1[i] = silu(h1[i]) * h3[i];
        }

        sgemv(&lw.w2, &h1, &mut ffn_out, dim, hidden_dim);
        for i in 0..dim {
            x[i] += ffn_out[i];
        }
    }

    // Final norm + logits
    let mut final_out = vec![0f32; dim];
    rmsnorm(&mut final_out, &x, &weights.final_norm);

    let mut logits = vec![0f32; cfg.vocab_size];
    sgemv(
        &weights.lm_head,
        &final_out,
        &mut logits,
        cfg.vocab_size,
        dim,
    );

    kv_cache.advance(1);
    Ok(logits)
}
