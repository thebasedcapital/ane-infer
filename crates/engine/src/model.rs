//! Model configuration and weight storage

use anyhow::Result;

/// Model configuration parsed from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,
}

impl ModelConfig {
    pub fn from_gguf(gguf: &ane_gguf::GgufFile) -> Result<Self> {
        let dim = gguf
            .embedding_length()
            .ok_or_else(|| anyhow::anyhow!("missing embedding_length"))? as usize;
        let n_heads = gguf
            .head_count()
            .ok_or_else(|| anyhow::anyhow!("missing head_count"))? as usize;
        let n_kv_heads = gguf.head_count_kv().unwrap_or(n_heads as u32) as usize;
        let n_layers = gguf
            .block_count()
            .ok_or_else(|| anyhow::anyhow!("missing block_count"))? as usize;
        let hidden_dim = gguf
            .feed_forward_length()
            .ok_or_else(|| anyhow::anyhow!("missing feed_forward_length"))?
            as usize;
        let vocab_size = gguf.vocab_size().unwrap_or(32000) as usize;
        let rope_freq_base = gguf.rope_freq_base().unwrap_or(10000.0);
        let max_seq_len = gguf.context_length().unwrap_or(2048) as usize;
        let rms_norm_eps = gguf.rms_norm_eps().unwrap_or(1e-6);
        let key_length = gguf.key_length().unwrap_or((dim / n_heads) as u32) as usize;

        Ok(Self {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim: key_length,
            vocab_size,
            max_seq_len: max_seq_len.min(4096), // cap for initial testing
            rope_freq_base,
            rms_norm_eps,
        })
    }
}

/// Qwen3.5 DeltaNet-specific configuration.
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub base: ModelConfig,
    pub full_attention_interval: usize, // e.g. 4 = every 4th layer
    pub ssm_state_size: usize,          // recurrent state dim (128)
    pub ssm_conv_kernel: usize,         // short conv kernel size (4)
    pub ssm_group_count: usize,         // number of DeltaNet heads (16)
    pub ssm_inner_size: usize,          // QKV projection dim
    pub layer_types: Vec<LayerType>,    // per-layer type
}

impl Qwen35Config {
    pub fn from_gguf(gguf: &ane_gguf::GgufFile) -> Result<Self> {
        let base = ModelConfig::from_gguf(gguf)?;
        let interval = gguf.full_attention_interval().unwrap_or(4) as usize;
        let ssm_state_size = gguf.ssm_state_size().unwrap_or(128) as usize;
        let ssm_conv_kernel = gguf.ssm_conv_kernel().unwrap_or(4) as usize;
        let ssm_group_count = gguf.ssm_group_count().unwrap_or(16) as usize;
        let ssm_inner_size = gguf.ssm_inner_size().unwrap_or(base.dim as u32) as usize;

        // Build layer type list
        let mut layer_types = Vec::with_capacity(base.n_layers);
        for l in 0..base.n_layers {
            if gguf.is_deltanet_layer(l) {
                layer_types.push(LayerType::DeltaNet);
            } else {
                layer_types.push(LayerType::FullAttention);
            }
        }

        Ok(Self {
            base,
            full_attention_interval: interval,
            ssm_state_size,
            ssm_conv_kernel,
            ssm_group_count,
            ssm_inner_size,
            layer_types,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    DeltaNet,
    FullAttention,
}

use crate::q8_gemv::Q8Tensor;

/// Shared FFN weights — Q8_0 quantized for decode.
pub struct FfnWeights {
    pub gate: Q8Tensor, // [hidden_dim, dim]
    pub up: Q8Tensor,   // [hidden_dim, dim]
    pub down: Q8Tensor, // [dim, hidden_dim]
}

/// DeltaNet layer weights — large projections stored as Q8_0.
pub struct DeltaNetLayerWeights {
    pub attn_norm: Vec<f32>,      // [dim] — small, keep FP32
    pub post_attn_norm: Vec<f32>, // [dim]
    pub qkv: Q8Tensor,            // [3*dim, dim] fused QKV projection
    pub attn_gate: Q8Tensor,      // [dim, dim] — Z gate
    pub ssm_a: Vec<f32>,          // [n_heads] — tiny
    pub ssm_alpha: Q8Tensor,      // [n_heads, dim] — decay projection
    pub ssm_beta: Q8Tensor,       // [n_heads, dim] — update gate projection
    pub ssm_conv1d: Vec<f32>,     // [conv_kernel, inner_size] — small, keep FP32
    pub ssm_dt_bias: Vec<f32>,    // [n_heads] — tiny
    pub ssm_norm: Vec<f32>,       // [head_dim] — tiny
    pub ssm_out: Q8Tensor,        // [dim, dim] — output projection
    pub ffn: FfnWeights,
}

/// Full attention layer weights — Q8_0 quantized.
pub struct FullAttnLayerWeights {
    pub attn_norm: Vec<f32>,      // [dim]
    pub post_attn_norm: Vec<f32>, // [dim]
    pub wq: Q8Tensor,             // [q_dim, dim]
    pub wk: Q8Tensor,             // [kv_dim, dim]
    pub wv: Q8Tensor,             // [kv_dim, dim]
    pub wo: Q8Tensor,             // [dim, q_dim/2] (only Q half goes through attn)
    pub q_norm: Vec<f32>,         // [head_dim] — tiny
    pub k_norm: Vec<f32>,         // [head_dim]
    pub ffn: FfnWeights,
}

/// A layer is either DeltaNet or full attention.
pub enum HybridLayerWeights {
    DeltaNet(DeltaNetLayerWeights),
    FullAttention(FullAttnLayerWeights),
}

/// Full Qwen3.5 model weights.
pub struct Qwen35ModelWeights {
    pub config: Qwen35Config,
    pub embedding: Vec<f32>, // [vocab_size, dim] — FP32 for table lookup
    pub layers: Vec<HybridLayerWeights>,
    pub final_norm: Vec<f32>, // [dim]
    pub lm_head: Q8Tensor,    // [vocab_size, dim] — Q8 for GEMV
}

// --- Keep old types for backward compat with Llama ---

/// Per-layer compiled kernels and weights (Llama-style).
pub struct LayerWeights {
    pub qkv_blob: Vec<u8>,
    pub o_proj_blob: Vec<u8>,
    pub ffn_up_blob: Vec<u8>,
    pub ffn_down_blob: Vec<u8>,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub w1: Vec<f32>,
    pub w3: Vec<f32>,
    pub w2: Vec<f32>,
}

/// Full model weights (Llama-style).
pub struct ModelWeights {
    pub config: ModelConfig,
    pub embedding: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub lm_head_blob: Vec<u8>,
}
