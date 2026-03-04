//! ane-engine: Hybrid ANE+CPU inference engine for LLMs
//!
//! Supports both standard transformer (Llama) and hybrid DeltaNet+Attention (Qwen3.5).

pub mod ane_prefill;
pub mod decode;
pub mod deltanet;
pub mod deltanet_cache;
pub mod gpu_decode;
pub mod kv_cache;
pub mod metal_gemv;
pub mod metal_graph;
pub mod model;
pub mod prefill;
pub mod q8_gemv;
pub mod scheduler;
pub mod scratch;
pub mod tokenizer;

pub use scheduler::InferenceEngine;
