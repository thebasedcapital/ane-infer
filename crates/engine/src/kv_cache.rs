//! KV Cache — unified memory buffer for key/value states across layers.
//!
//! On Apple Silicon, both ANE and CPU share the same physical DRAM,
//! so the KV cache written during ANE prefill is directly readable by CPU decode.

/// KV cache for a single layer.
pub struct LayerKvCache {
    /// Key cache: [n_kv_heads, max_seq_len, head_dim]
    pub keys: Vec<f32>,
    /// Value cache: [n_kv_heads, max_seq_len, head_dim]
    pub values: Vec<f32>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl LayerKvCache {
    pub fn new(n_kv_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let size = n_kv_heads * max_seq_len * head_dim;
        Self {
            keys: vec![0.0; size],
            values: vec![0.0; size],
            n_kv_heads,
            head_dim,
            max_seq_len,
        }
    }

    /// Write key/value for a single position.
    /// k, v: [n_kv_heads, head_dim]
    pub fn write_pos(&mut self, pos: usize, k: &[f32], v: &[f32]) {
        for h in 0..self.n_kv_heads {
            let cache_off = h * self.max_seq_len * self.head_dim + pos * self.head_dim;
            let src_off = h * self.head_dim;
            self.keys[cache_off..cache_off + self.head_dim]
                .copy_from_slice(&k[src_off..src_off + self.head_dim]);
            self.values[cache_off..cache_off + self.head_dim]
                .copy_from_slice(&v[src_off..src_off + self.head_dim]);
        }
    }

    /// Write key/value for a range of positions (prefill).
    /// k, v: [n_kv_heads, seq_len, head_dim] — but stored as [seq_len, n_kv_heads * head_dim] row-major
    pub fn write_range(&mut self, start: usize, seq_len: usize, k: &[f32], v: &[f32]) {
        for t in 0..seq_len {
            for h in 0..self.n_kv_heads {
                let cache_off = h * self.max_seq_len * self.head_dim + (start + t) * self.head_dim;
                let src_off = t * self.n_kv_heads * self.head_dim + h * self.head_dim;
                self.keys[cache_off..cache_off + self.head_dim]
                    .copy_from_slice(&k[src_off..src_off + self.head_dim]);
                self.values[cache_off..cache_off + self.head_dim]
                    .copy_from_slice(&v[src_off..src_off + self.head_dim]);
            }
        }
    }

    /// Get key slice for attention: returns pointer to [max_seq_len, head_dim] for head h
    pub fn key_head(&self, h: usize) -> &[f32] {
        let off = h * self.max_seq_len * self.head_dim;
        &self.keys[off..off + self.max_seq_len * self.head_dim]
    }

    pub fn value_head(&self, h: usize) -> &[f32] {
        let off = h * self.max_seq_len * self.head_dim;
        &self.values[off..off + self.max_seq_len * self.head_dim]
    }
}

/// Full KV cache across all layers.
pub struct KvCache {
    pub layers: Vec<LayerKvCache>,
    pub pos: usize, // current position (number of tokens processed)
}

impl KvCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let layers = (0..n_layers)
            .map(|_| LayerKvCache::new(n_kv_heads, max_seq_len, head_dim))
            .collect();
        Self { layers, pos: 0 }
    }

    pub fn advance(&mut self, n: usize) {
        self.pos += n;
    }
}
