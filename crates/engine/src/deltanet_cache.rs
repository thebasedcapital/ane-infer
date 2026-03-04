//! DeltaNet state cache — fixed size, doesn't grow with context length.

use crate::scratch::ScratchBuffers;

pub struct DeltaNetLayerState {
    pub conv_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
    pub inner_size: usize,
    pub kernel_size: usize,
    pub n_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl DeltaNetLayerState {
    pub fn new(
        inner_size: usize,
        kernel_size: usize,
        n_heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Self {
        Self {
            conv_state: vec![0.0; inner_size * kernel_size],
            recurrent_state: vec![0.0; n_heads * key_dim * value_dim],
            inner_size,
            kernel_size,
            n_heads,
            key_dim,
            value_dim,
        }
    }

    pub fn conv_shift_and_append(&mut self, input: &[f32]) {
        for ch in 0..self.inner_size {
            let base = ch * self.kernel_size;
            for k in 0..self.kernel_size - 1 {
                self.conv_state[base + k] = self.conv_state[base + k + 1];
            }
            self.conv_state[base + self.kernel_size - 1] = input[ch];
        }
    }

    pub fn conv_apply(&self, weights: &[f32], output: &mut [f32]) {
        for ch in 0..self.inner_size {
            let mut sum = 0.0f32;
            let state_base = ch * self.kernel_size;
            let weight_base = ch * self.kernel_size;
            for k in 0..self.kernel_size {
                sum += self.conv_state[state_base + k] * weights[weight_base + k];
            }
            output[ch] = sum;
        }
    }

    pub fn head_state(&self, h: usize) -> &[f32] {
        let off = h * self.key_dim * self.value_dim;
        &self.recurrent_state[off..off + self.key_dim * self.value_dim]
    }

    pub fn head_state_mut(&mut self, h: usize) -> &mut [f32] {
        let off = h * self.key_dim * self.value_dim;
        &mut self.recurrent_state[off..off + self.key_dim * self.value_dim]
    }
}

pub struct HybridCache {
    pub deltanet_states: Vec<DeltaNetLayerState>,
    pub kv_caches: Vec<super::kv_cache::LayerKvCache>,
    pub layer_map: Vec<(super::model::LayerType, usize)>,
    pub pos: usize,
    pub scratch: ScratchBuffers,
}

impl HybridCache {
    pub fn new(config: &super::model::Qwen35Config) -> Self {
        let mut deltanet_states = Vec::new();
        let mut kv_caches = Vec::new();
        let mut layer_map = Vec::new();

        let inner_size = config.base.dim * 3;
        let key_dim = config.ssm_state_size;
        let value_dim = config.ssm_state_size;
        let c = &config.base;

        for lt in &config.layer_types {
            match lt {
                super::model::LayerType::DeltaNet => {
                    let idx = deltanet_states.len();
                    deltanet_states.push(DeltaNetLayerState::new(
                        inner_size,
                        config.ssm_conv_kernel,
                        config.ssm_group_count,
                        key_dim,
                        value_dim,
                    ));
                    layer_map.push((super::model::LayerType::DeltaNet, idx));
                }
                super::model::LayerType::FullAttention => {
                    let idx = kv_caches.len();
                    kv_caches.push(super::kv_cache::LayerKvCache::new(
                        c.n_kv_heads,
                        c.max_seq_len,
                        c.head_dim,
                    ));
                    layer_map.push((super::model::LayerType::FullAttention, idx));
                }
            }
        }

        let q_full_dim = c.n_heads * c.head_dim * 2;
        let kv_dim = c.n_kv_heads * c.head_dim;

        let scratch = ScratchBuffers::new(
            c.dim,
            c.hidden_dim,
            inner_size,
            config.ssm_group_count,
            c.n_kv_heads,
            c.head_dim,
            key_dim,
            value_dim,
            c.vocab_size,
            c.max_seq_len,
            q_full_dim,
            kv_dim,
        );

        Self {
            deltanet_states,
            kv_caches,
            layer_map,
            pos: 0,
            scratch,
        }
    }

    pub fn advance(&mut self, n: usize) {
        self.pos += n;
    }
}
