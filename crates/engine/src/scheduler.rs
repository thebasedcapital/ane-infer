//! Inference Scheduler — orchestrates prefill (ANE) → decode (CPU) handoff.

use crate::decode::decode_token;
use crate::kv_cache::KvCache;
use crate::model::ModelWeights;
use crate::prefill::{prefill, PrefillKernels};
use anyhow::Result;

/// Sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 256,
        }
    }
}

/// The main inference engine.
pub struct InferenceEngine {
    pub weights: ModelWeights,
    pub prefill_kernels: Option<PrefillKernels>,
}

impl InferenceEngine {
    pub fn new(weights: ModelWeights) -> Self {
        Self {
            weights,
            prefill_kernels: None,
        }
    }

    /// Compile prefill kernels for a given max prompt length.
    pub fn compile_prefill(&mut self, max_seq_len: usize) -> Result<()> {
        let kernels = PrefillKernels::compile(&self.weights, max_seq_len)?;
        self.prefill_kernels = Some(kernels);
        Ok(())
    }

    /// Generate tokens from a prompt (token IDs).
    /// Returns generated token IDs.
    pub fn generate(&self, prompt_tokens: &[u32], params: &SamplingParams) -> Result<Vec<u32>> {
        let cfg = &self.weights.config;
        let mut kv_cache =
            KvCache::new(cfg.n_layers, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim);

        // Phase 1: Prefill on ANE
        let logits = if let Some(ref kernels) = self.prefill_kernels {
            prefill(kernels, &self.weights, &mut kv_cache, prompt_tokens)?
        } else {
            // Fallback: decode tokens one by one (slow but works without ANE)
            let mut last_logits = vec![0f32; cfg.vocab_size];
            for &tok in prompt_tokens {
                last_logits = decode_token(&self.weights, &mut kv_cache, tok)?;
            }
            last_logits
        };

        // Sample first generated token
        let mut generated = Vec::with_capacity(params.max_tokens);
        let mut next_token = sample_token(&logits, params.temperature, params.top_p);
        generated.push(next_token);

        // Phase 2: Decode on CPU
        for _ in 1..params.max_tokens {
            let logits = decode_token(&self.weights, &mut kv_cache, next_token)?;
            next_token = sample_token(&logits, params.temperature, params.top_p);

            // EOS check (common EOS token IDs)
            if next_token == 2 || next_token == 128001 || next_token == 128009 {
                break;
            }

            generated.push(next_token);
        }

        Ok(generated)
    }
}

/// Temperature + top-p sampling.
fn sample_token(logits: &[f32], temperature: f32, top_p: f32) -> u32 {
    if temperature < 1e-6 {
        // Greedy
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scaled.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Top-p filtering
    let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
    sorted_indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let mut cumulative = 0.0f32;
    let mut cutoff_idx = sorted_indices.len();
    for (i, &idx) in sorted_indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Renormalize
    let mut filtered_sum = 0f32;
    for &idx in &sorted_indices[..cutoff_idx] {
        filtered_sum += probs[idx];
    }

    // Random selection (using simple xorshift since we don't need crypto-grade)
    let r = simple_random() * filtered_sum;
    let mut acc = 0f32;
    for &idx in &sorted_indices[..cutoff_idx] {
        acc += probs[idx];
        if acc >= r {
            return idx as u32;
        }
    }

    sorted_indices[0] as u32
}

/// Simple pseudo-random f32 in [0, 1). Not crypto-safe, but fine for sampling.
fn simple_random() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x12345678_9ABCDEF0);
    let mut s = STATE.load(Ordering::Relaxed);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    STATE.store(s, Ordering::Relaxed);
    (s & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
}
