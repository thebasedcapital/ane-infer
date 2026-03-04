//! Pre-allocated scratch buffers for zero-allocation inference.
//!
//! Eliminates Vec allocations per-token by reusing persistent buffers.

pub struct ScratchBuffers {
    // Main hidden state buffer (flows through layers)
    pub hidden: Vec<f32>,

    // Work buffer for copies when borrowing is complex
    pub work: Vec<f32>,

    // DeltaNet buffers
    pub qkvz: Vec<f32>,
    pub conv_out: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub output_heads: Vec<f32>,
    pub gate_out: Vec<f32>,
    pub normed: Vec<f32>,
    pub beta_raw: Vec<f32>,
    pub alpha_raw: Vec<f32>,
    pub beta: Vec<f32>,
    pub decay: Vec<f32>,
    pub sk: Vec<f32>,
    pub delta: Vec<f32>,

    // Full attention buffers
    pub q_full: Vec<f32>,
    pub kv_k: Vec<f32>,
    pub kv_v: Vec<f32>,
    pub q_only: Vec<f32>,
    pub gate: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub scores: Vec<f32>,

    // FFN buffers
    pub ffn_h1: Vec<f32>,
    pub ffn_h3: Vec<f32>,
    pub ffn_out: Vec<f32>,

    // Layer norm buffers
    pub xnorm: Vec<f32>,
    pub ffn_in: Vec<f32>,
    pub final_out: Vec<f32>,
    pub logits: Vec<f32>,
}

impl ScratchBuffers {
    pub fn new(
        dim: usize,
        hidden_dim: usize,
        inner_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        key_dim: usize,
        value_dim: usize,
        vocab_size: usize,
        max_seq_len: usize,
        q_full_dim: usize,
        kv_dim: usize,
    ) -> Self {
        let chunk = inner_size / 3;
        Self {
            hidden: vec![0.0; dim],
            work: vec![0.0; dim],

            qkvz: vec![0.0; inner_size],
            conv_out: vec![0.0; inner_size],
            q: vec![0.0; chunk],
            k: vec![0.0; chunk],
            v: vec![0.0; chunk],
            output_heads: vec![0.0; chunk],
            gate_out: vec![0.0; dim],
            normed: vec![0.0; chunk],
            beta_raw: vec![0.0; n_heads],
            alpha_raw: vec![0.0; n_heads],
            beta: vec![0.0; n_heads],
            decay: vec![0.0; n_heads],
            sk: vec![0.0; value_dim],
            delta: vec![0.0; value_dim],

            q_full: vec![0.0; q_full_dim],
            kv_k: vec![0.0; kv_dim],
            kv_v: vec![0.0; kv_dim],
            q_only: vec![0.0; n_heads * head_dim],
            gate: vec![0.0; n_heads * head_dim],
            attn_out: vec![0.0; n_heads * head_dim],
            scores: vec![0.0; max_seq_len],

            ffn_h1: vec![0.0; hidden_dim],
            ffn_h3: vec![0.0; hidden_dim],
            ffn_out: vec![0.0; dim],

            xnorm: vec![0.0; dim],
            ffn_in: vec![0.0; dim],
            final_out: vec![0.0; dim],
            logits: vec![0.0; vocab_size],
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub fn vec_mul_accumulate(y: &mut [f32], a: &[f32], b: &[f32]) {
    use std::arch::aarch64::*;
    let n = y.len().min(a.len()).min(b.len());
    unsafe {
        for i in (0..n).step_by(4) {
            let remaining = n - i;
            if remaining >= 4 {
                let yv = vld1q_f32(y.as_ptr().add(i));
                let av = vld1q_f32(a.as_ptr().add(i));
                let bv = vld1q_f32(b.as_ptr().add(i));
                let res = vfmaq_f32(yv, av, bv);
                vst1q_f32(y.as_mut_ptr().add(i), res);
            } else {
                for j in i..n {
                    y[j] += a[j] * b[j];
                }
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_mul_accumulate(y: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..y.len().min(a.len()).min(b.len()) {
        y[i] += a[i] * b[i];
    }
}

#[cfg(target_arch = "aarch64")]
pub fn vec_scale(y: &mut [f32], scale: f32) {
    use std::arch::aarch64::*;
    unsafe {
        let sv = vdupq_n_f32(scale);
        for i in (0..y.len()).step_by(4) {
            if y.len() - i >= 4 {
                let yv = vld1q_f32(y.as_ptr().add(i));
                vst1q_f32(y.as_mut_ptr().add(i), vmulq_f32(yv, sv));
            } else {
                for j in i..y.len() {
                    y[j] *= scale;
                }
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_scale(y: &mut [f32], scale: f32) {
    for v in y.iter_mut() {
        *v *= scale;
    }
}

#[cfg(target_arch = "aarch64")]
pub fn vec_silu_inplace(x: &mut [f32]) {
    let n = x.len();
    for i in 0..n {
        x[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

#[cfg(target_arch = "aarch64")]
pub fn vec_silu_mul_inplace(a: &mut [f32], b: &[f32]) {
    let n = a.len().min(b.len());
    for i in 0..n {
        a[i] = (a[i] / (1.0 + (-a[i]).exp())) * b[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_silu_mul_inplace(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len().min(b.len()) {
        a[i] = (a[i] / (1.0 + (-a[i]).exp())) * b[i];
    }
}
