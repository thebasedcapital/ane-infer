//! ANE Prefill for Qwen3.5 — batch all prompt tokens through ANE projections.
//!
//! Strategy: for each layer, use ANE to compute the big linear projections
//! (QKV, gate, FFN) on ALL tokens in parallel as 1x1 convolutions.
//! Then feed the projected values through the CPU recurrence/attention.
//!
//! This gives us ANE's ~19 TFLOPS for the bandwidth-heavy projections
//! while keeping the sequential recurrence on CPU.

use ane_bridge::{build_single_weight_blob, transpose_to_channels_first, AneKernel};
use anyhow::Result;
use half::f16;

use crate::model::*;
use crate::q8_gemv::Q8Tensor;

/// Compiled ANE kernel for a single projection.
pub struct AneProjection {
    kernel: AneKernel,
    in_dim: usize,
    out_dim: usize,
    seq_len: usize,
}

impl AneProjection {
    /// Compile a projection kernel: [in_dim] → [out_dim] over seq_len tokens.
    /// Weights must be dequantized to FP32 for building the ANE weight blob.
    pub fn compile(
        weights_f32: &[f32],
        in_dim: usize,
        out_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let mil = mil_gen::mil_gen_conv(in_dim, out_dim, seq_len);
        let blob = build_single_weight_blob(weights_f32);

        let in_bytes = in_dim * seq_len * 4; // FP32 [1, in_dim, 1, seq_len]
        let out_bytes = out_dim * seq_len * 4; // FP32 [1, out_dim, 1, seq_len]

        let kernel = AneKernel::compile(&mil, Some(&blob), &[in_bytes], &[out_bytes])?;

        Ok(Self {
            kernel,
            in_dim,
            out_dim,
            seq_len,
        })
    }

    /// Run the projection on all tokens.
    /// input: [seq_len, in_dim] row-major
    /// output: [seq_len, out_dim] row-major
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        // Transpose to channel-first: [seq_len, dim] → [dim, seq_len]
        let mut input_t = vec![0.0f32; self.in_dim * self.seq_len];
        for t in 0..self.seq_len {
            for d in 0..self.in_dim {
                input_t[d * self.seq_len + t] = input[t * self.in_dim + d];
            }
        }

        self.kernel.write_input_f32(0, &input_t);
        self.kernel.eval()?;

        // Read and transpose back: [out_dim, seq_len] → [seq_len, out_dim]
        let mut output_t = vec![0.0f32; self.out_dim * self.seq_len];
        self.kernel.read_output_f32(0, &mut output_t);

        for t in 0..self.seq_len {
            for d in 0..self.out_dim {
                output[t * self.out_dim + d] = output_t[d * self.seq_len + t];
            }
        }

        Ok(())
    }
}

/// Dequantize a Q8Tensor to FP32 for ANE weight blob construction.
fn dequant_q8_to_f32(q8: &Q8Tensor) -> Vec<f32> {
    let bpr = q8.n / 32;
    let mut out = vec![0.0f32; q8.m * q8.n];

    for row in 0..q8.m {
        for b in 0..bpr {
            let off = (row * bpr + b) * 34;
            let scale = f16::from_le_bytes([q8.data[off], q8.data[off + 1]]).to_f32();
            for i in 0..32 {
                out[row * q8.n + b * 32 + i] = q8.data[off + 2 + i] as i8 as f32 * scale;
            }
        }
    }
    out
}

/// FFN kernel — 3 separate ANE projections (fused version has weight blob offset issues).
pub struct FusedFfnKernel {
    gate: AneProjection,
    up: AneProjection,
    down: AneProjection,
}

impl FusedFfnKernel {
    pub fn compile(
        gate_f32: &[f32],
        up_f32: &[f32],
        down_f32: &[f32],
        dim: usize,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let gate = AneProjection::compile(gate_f32, dim, hidden_dim, seq_len)?;
        let up = AneProjection::compile(up_f32, dim, hidden_dim, seq_len)?;
        let down = AneProjection::compile(down_f32, hidden_dim, dim, seq_len)?;
        Ok(Self { gate, up, down })
    }

    /// Run FFN: gate → SiLU → * up → down.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let seq_len = self.gate.seq_len;
        let hidden_dim = self.gate.out_dim;

        let mut h1 = vec![0.0f32; seq_len * hidden_dim];
        let mut h3 = vec![0.0f32; seq_len * hidden_dim];
        self.gate.forward(input, &mut h1)?;
        self.up.forward(input, &mut h3)?;

        // SiLU(gate) * up on CPU
        for i in 0..seq_len * hidden_dim {
            h1[i] = (h1[i] / (1.0 + (-h1[i]).exp())) * h3[i];
        }

        self.down.forward(&h1, output)?;
        Ok(())
    }
}

/// Compiled ANE projections for one DeltaNet layer.
/// Uses mega-kernels where possible.
pub struct DeltaNetAneKernels {
    pub qkv: AneProjection,        // [dim → 3*dim] (TODO: fuse into mega)
    pub gate: AneProjection,       // [dim → dim]
    pub ssm_out: AneProjection,    // [dim → dim]
    pub fused_ffn: FusedFfnKernel, // gate+SiLU+up+mul+down = 6 ops, ONE dispatch
}

/// Compiled ANE projections for one full attention layer.
pub struct FullAttnAneKernels {
    pub wq: AneProjection,         // [dim → q_dim]
    pub wk: AneProjection,         // [dim → kv_dim]
    pub wv: AneProjection,         // [dim → kv_dim]
    pub wo: AneProjection,         // [q_only_dim → dim]
    pub fused_ffn: FusedFfnKernel, // gate+SiLU+up+mul+down = 6 ops, ONE dispatch
}

/// All ANE kernels for prefill.
pub enum LayerAneKernels {
    DeltaNet(DeltaNetAneKernels),
    FullAttention(FullAttnAneKernels),
}

/// Compile all ANE prefill kernels for a given sequence length.
pub fn compile_ane_prefill(
    model: &Qwen35ModelWeights,
    seq_len: usize,
) -> Result<Vec<LayerAneKernels>> {
    let c = &model.config.base;
    let dim = c.dim;
    let hidden_dim = c.hidden_dim;

    let mut kernels = Vec::with_capacity(c.n_layers);

    for (l, layer) in model.layers.iter().enumerate() {
        eprint!("  Compiling ANE layer {l}/{}...", c.n_layers);

        match layer {
            HybridLayerWeights::DeltaNet(lw) => {
                let qkv_f32 = dequant_q8_to_f32(&lw.qkv);
                let gate_f32 = dequant_q8_to_f32(&lw.attn_gate);
                let ssm_out_f32 = dequant_q8_to_f32(&lw.ssm_out);
                let fg_f32 = dequant_q8_to_f32(&lw.ffn.gate);
                let fu_f32 = dequant_q8_to_f32(&lw.ffn.up);
                let fd_f32 = dequant_q8_to_f32(&lw.ffn.down);

                let fused_ffn =
                    FusedFfnKernel::compile(&fg_f32, &fu_f32, &fd_f32, dim, hidden_dim, seq_len)?;

                kernels.push(LayerAneKernels::DeltaNet(DeltaNetAneKernels {
                    qkv: AneProjection::compile(&qkv_f32, dim, dim * 3, seq_len)?,
                    gate: AneProjection::compile(&gate_f32, dim, dim, seq_len)?,
                    ssm_out: AneProjection::compile(&ssm_out_f32, dim, dim, seq_len)?,
                    fused_ffn,
                }));
                eprintln!(" ok (3+1 mega)");
            }
            HybridLayerWeights::FullAttention(lw) => {
                let wq_f32 = dequant_q8_to_f32(&lw.wq);
                let wk_f32 = dequant_q8_to_f32(&lw.wk);
                let wv_f32 = dequant_q8_to_f32(&lw.wv);
                let wo_f32 = dequant_q8_to_f32(&lw.wo);
                let fg_f32 = dequant_q8_to_f32(&lw.ffn.gate);
                let fu_f32 = dequant_q8_to_f32(&lw.ffn.up);
                let fd_f32 = dequant_q8_to_f32(&lw.ffn.down);

                let q_dim = lw.wq.m;
                let kv_dim = lw.wk.m;
                let q_only_dim = c.n_heads * c.head_dim;

                let fused_ffn =
                    FusedFfnKernel::compile(&fg_f32, &fu_f32, &fd_f32, dim, hidden_dim, seq_len)?;

                kernels.push(LayerAneKernels::FullAttention(FullAttnAneKernels {
                    wq: AneProjection::compile(&wq_f32, dim, q_dim, seq_len)?,
                    wk: AneProjection::compile(&wk_f32, dim, kv_dim, seq_len)?,
                    wv: AneProjection::compile(&wv_f32, dim, kv_dim, seq_len)?,
                    wo: AneProjection::compile(&wo_f32, q_only_dim, dim, seq_len)?,
                    fused_ffn,
                }));
                eprintln!(" ok (4+1 mega)");
            }
        }
    }

    Ok(kernels)
}
