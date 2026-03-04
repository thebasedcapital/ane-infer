//! ane-gguf: GGUF model file parser and weight dequantization
//!
//! Parses GGUF header, metadata, and tensor info.
//! Dequantizes Q4_0 / Q4_K_M weights to FP16 for ANE consumption.

mod dequant;
mod parser;
mod to_ane;

pub use dequant::*;
pub use parser::*;
pub use to_ane::*;
