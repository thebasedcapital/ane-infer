//! Convert GGUF tensors to ANE weight blob format.
//!
//! Reshapes weights from [out, in] to [out, in, 1, 1] for 1x1 conv,
//! converts to FP16, and packages with the ANE weight blob header.

use super::dequant::dequantize_tensor;
use super::parser::{GgmlType, GgufFile};
use anyhow::Result;

/// Extract raw tensor bytes from GGUF without dequantization.
/// Returns (raw_bytes, ne0, ne1, quant_type) where ne0 is fast dim, ne1 is slow dim.
pub fn extract_tensor_raw(
    gguf: &GgufFile,
    file_data: &[u8],
    name: &str,
) -> Result<(Vec<u8>, usize, usize, GgmlType)> {
    let tensor = gguf
        .get_tensor(name)
        .ok_or_else(|| anyhow::anyhow!("tensor not found: {name}"))?;

    let abs_offset = (gguf.tensor_data_offset + tensor.offset) as usize;
    let data_size = tensor.data_size() as usize;
    let data = file_data[abs_offset..abs_offset + data_size].to_vec();

    let ne0 = tensor.dimensions.first().copied().unwrap_or(0) as usize;
    let ne1 = tensor.dimensions.get(1).copied().unwrap_or(1) as usize;
    let quant_type = tensor.typ;

    Ok((data, ne0, ne1, quant_type))
}

/// Extract a tensor from GGUF file data, dequantize to FP32.
pub fn extract_tensor_f32(gguf: &GgufFile, file_data: &[u8], name: &str) -> Result<Vec<f32>> {
    let tensor = gguf
        .get_tensor(name)
        .ok_or_else(|| anyhow::anyhow!("tensor not found: {name}"))?;

    let abs_offset = (gguf.tensor_data_offset + tensor.offset) as usize;
    let data_size = tensor.data_size() as usize;
    let data = &file_data[abs_offset..abs_offset + data_size];
    let n_elements = tensor.n_elements() as usize;

    dequantize_tensor(data, tensor.typ, n_elements)
}

/// Build ANE weight blob from multiple FP32 weight tensors.
/// Each weight is converted to FP16 with the proper chunk header format.
pub fn build_ane_weight_blob(weight_sets: &[&[f32]]) -> Vec<u8> {
    ane_bridge::build_weight_blob(weight_sets)
}

/// Extract and build QKV fused weight blob for a layer.
/// Returns weight blob containing Wq, Wk, Wv as separate chunks.
pub fn extract_qkv_weights(
    gguf: &GgufFile,
    file_data: &[u8],
    layer: usize,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let wq = extract_tensor_f32(gguf, file_data, &format!("blk.{layer}.attn_q.weight"))?;
    let wk = extract_tensor_f32(gguf, file_data, &format!("blk.{layer}.attn_k.weight"))?;
    let wv = extract_tensor_f32(gguf, file_data, &format!("blk.{layer}.attn_v.weight"))?;

    let blob = build_ane_weight_blob(&[&wq, &wk, &wv]);
    Ok((blob, wq, wk, wv))
}

/// Extract FFN up weights (gate_proj + up_proj) for a layer.
pub fn extract_ffn_up_weights(
    gguf: &GgufFile,
    file_data: &[u8],
    layer: usize,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    let w1 = extract_tensor_f32(gguf, file_data, &format!("blk.{layer}.ffn_gate.weight"))?;
    let w3 = extract_tensor_f32(gguf, file_data, &format!("blk.{layer}.ffn_up.weight"))?;

    let blob = build_ane_weight_blob(&[&w1, &w3]);
    Ok((blob, w1, w3))
}

/// Extract FFN down weight (down_proj) for a layer.
pub fn extract_ffn_down_weight(gguf: &GgufFile, file_data: &[u8], layer: usize) -> Result<Vec<u8>> {
    let w2 = extract_tensor_f32(gguf, file_data, &format!("blk.{layer}.ffn_down.weight"))?;
    Ok(build_ane_weight_blob(&[&w2]))
}

/// Extract output projection weight for a layer.
pub fn extract_output_proj_weight(
    gguf: &GgufFile,
    file_data: &[u8],
    layer: usize,
) -> Result<Vec<u8>> {
    let wo = extract_tensor_f32(gguf, file_data, &format!("blk.{layer}.attn_output.weight"))?;
    Ok(build_ane_weight_blob(&[&wo]))
}

/// Extract token embedding table. Returns FP32 [vocab_size, dim].
pub fn extract_embedding(gguf: &GgufFile, file_data: &[u8]) -> Result<Vec<f32>> {
    extract_tensor_f32(gguf, file_data, "token_embd.weight")
}

/// Extract RMSNorm weights for a layer (attention and FFN norms).
pub fn extract_layer_norms(
    gguf: &GgufFile,
    file_data: &[u8],
    layer: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let attn_norm = extract_tensor_f32(gguf, file_data, &format!("blk.{layer}.attn_norm.weight"))?;
    let ffn_norm = extract_tensor_f32(gguf, file_data, &format!("blk.{layer}.ffn_norm.weight"))?;
    Ok((attn_norm, ffn_norm))
}

/// Extract final RMSNorm weight.
pub fn extract_final_norm(gguf: &GgufFile, file_data: &[u8]) -> Result<Vec<f32>> {
    extract_tensor_f32(gguf, file_data, "output_norm.weight")
}

/// Extract LM head (classifier) weight.
pub fn extract_lm_head(gguf: &GgufFile, file_data: &[u8]) -> Result<Vec<f32>> {
    // Try "output.weight" first, fall back to "token_embd.weight" (tied weights)
    extract_tensor_f32(gguf, file_data, "output.weight")
        .or_else(|_| extract_tensor_f32(gguf, file_data, "token_embd.weight"))
}

// --- Qwen3.5 DeltaNet layer extraction ---

// --- Qwen3.5 tensor name helpers ---

/// All DeltaNet tensor names for a layer.
pub fn deltanet_tensor_names(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.attn_norm.weight"),
        format!("blk.{layer}.post_attention_norm.weight"),
        format!("blk.{layer}.attn_qkv.weight"),
        format!("blk.{layer}.attn_gate.weight"),
        format!("blk.{layer}.ssm_a"),
        format!("blk.{layer}.ssm_alpha.weight"),
        format!("blk.{layer}.ssm_beta.weight"),
        format!("blk.{layer}.ssm_conv1d.weight"),
        format!("blk.{layer}.ssm_dt.bias"),
        format!("blk.{layer}.ssm_norm.weight"),
        format!("blk.{layer}.ssm_out.weight"),
        format!("blk.{layer}.ffn_gate.weight"),
        format!("blk.{layer}.ffn_up.weight"),
        format!("blk.{layer}.ffn_down.weight"),
    ]
}

/// All full attention tensor names for a layer.
pub fn full_attn_tensor_names(layer: usize) -> Vec<String> {
    vec![
        format!("blk.{layer}.attn_norm.weight"),
        format!("blk.{layer}.post_attention_norm.weight"),
        format!("blk.{layer}.attn_q.weight"),
        format!("blk.{layer}.attn_k.weight"),
        format!("blk.{layer}.attn_v.weight"),
        format!("blk.{layer}.attn_output.weight"),
        format!("blk.{layer}.attn_q_norm.weight"),
        format!("blk.{layer}.attn_k_norm.weight"),
        format!("blk.{layer}.ffn_gate.weight"),
        format!("blk.{layer}.ffn_up.weight"),
        format!("blk.{layer}.ffn_down.weight"),
    ]
}
