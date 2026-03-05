//! Dequantization: GGML quantized formats → FP32/FP16
//!
//! ANE requires FP16 weights, so we dequantize from Q4_0/Q8_0 etc.

use super::parser::GgmlType;
use anyhow::{bail, Result};
use half::f16;

/// Dequantize Q4_0 block to FP32.
/// Block format: 2 bytes (f16 scale) + 16 bytes (32 × 4-bit values)
/// GGUF layout: elements 0-15 = low nibbles of bytes 0-15, elements 16-31 = high nibbles
pub fn dequant_q4_0_block(block: &[u8], out: &mut [f32]) {
    assert!(block.len() >= 18);
    assert!(out.len() >= 32);

    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

    for i in 0..16 {
        let byte = block[2 + i];
        // Low nibble → element i
        out[i] = ((byte & 0x0F) as i32 - 8) as f32 * scale;
        // High nibble → element i + 16
        out[i + 16] = (((byte >> 4) & 0x0F) as i32 - 8) as f32 * scale;
    }
}

/// Dequantize Q8_0 block to FP32.
/// Block format: 2 bytes (f16 scale) + 32 bytes (32 × i8 values)
pub fn dequant_q8_0_block(block: &[u8], out: &mut [f32]) {
    assert!(block.len() >= 34);
    assert!(out.len() >= 32);

    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

    for i in 0..32 {
        out[i] = block[2 + i] as i8 as f32 * scale;
    }
}

/// Dequantize Q4_K block to FP32.
/// Q4_K_M format: 256 elements per block, 144 bytes per block
/// Layout: 2 bytes (f16 d), 2 bytes (f16 dmin), 12 bytes (scales), 128 bytes (4-bit quants)
pub fn dequant_q4_k_block(block: &[u8], out: &mut [f32]) {
    assert!(block.len() >= 144);
    assert!(out.len() >= 256);

    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

    // Unpack 6-bit scales and mins from the 12-byte scale section
    let scales_bytes = &block[4..16];
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];

    // First 4 scales/mins: 6 bits each, packed in first 8 bytes
    for i in 0..4 {
        sc[i] = scales_bytes[i] & 0x3F;
        mn[i] = scales_bytes[i + 4] & 0x3F;
    }
    // Last 4 scales/mins: low 4 bits from bytes 8-11, high 2 bits from bytes 0-7
    for i in 0..4 {
        let byte = scales_bytes[8 + i];
        sc[4 + i] = (byte & 0x0F) | ((scales_bytes[i] >> 6) << 4);
        mn[4 + i] = (byte >> 4) | ((scales_bytes[i + 4] >> 6) << 4);
    }

    let quants = &block[16..144];

    for j in 0..8 {
        let scale = d * sc[j] as f32;
        let min = dmin * mn[j] as f32;

        for l in 0..16 {
            let idx = j * 16 + l;
            let q_byte = quants[j * 16 + l];
            let lo = (q_byte & 0x0F) as f32;
            let hi = ((q_byte >> 4) & 0x0F) as f32;

            if idx < 128 {
                out[j * 32 + l] = lo * scale - min;
                out[j * 32 + l + 16] = hi * scale - min;
            }
        }
    }
}

/// Dequantize Q6_K block to FP32.
/// Q6_K format: 256 elements per block, 210 bytes per block
/// Layout: ql[128] + qh[64] + scales[16] + d[2]
/// - ql: lower 4 bits of each 6-bit quant (128 bytes)
/// - qh: upper 2 bits of each 6-bit quant (64 bytes)  
/// - scales: int8 scales, one per 16 elements (16 bytes)
/// - d: fp16 super-block scale (2 bytes)
pub fn dequant_q6_k_block(block: &[u8], out: &mut [f32]) {
    assert!(block.len() >= 210);
    assert!(out.len() >= 256);

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales: &[i8] =
        unsafe { std::slice::from_raw_parts(block[192..208].as_ptr() as *const i8, 16) };
    let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

    for n in (0..256).step_by(128) {
        let chunk = n / 128;
        let ql_offset = chunk * 64;
        let qh_offset = chunk * 32;
        let sc_offset = chunk * 8;

        for l in 0..32 {
            let is = l / 16;

            let q1 = ((ql[ql_offset + l] & 0x0F) as i32
                | (((qh[qh_offset + l] >> 0) & 3) as i32) << 4)
                - 32;
            let q2 = ((ql[ql_offset + l + 32] & 0x0F) as i32
                | (((qh[qh_offset + l] >> 2) & 3) as i32) << 4)
                - 32;
            let q3 = ((ql[ql_offset + l] >> 4) as i32
                | (((qh[qh_offset + l] >> 4) & 3) as i32) << 4)
                - 32;
            let q4 = ((ql[ql_offset + l + 32] >> 4) as i32
                | (((qh[qh_offset + l] >> 6) & 3) as i32) << 4)
                - 32;

            out[n + l + 0] = d * scales[sc_offset + is + 0] as f32 * q1 as f32;
            out[n + l + 32] = d * scales[sc_offset + is + 2] as f32 * q2 as f32;
            out[n + l + 64] = d * scales[sc_offset + is + 4] as f32 * q3 as f32;
            out[n + l + 96] = d * scales[sc_offset + is + 6] as f32 * q4 as f32;
        }
    }
}

/// Dequantize a full tensor from GGML quantized format to FP32.
pub fn dequantize_tensor(data: &[u8], typ: GgmlType, n_elements: usize) -> Result<Vec<f32>> {
    let block_size = typ.block_size();
    let bytes_per_block = typ.bytes_per_block();

    if block_size == 0 || bytes_per_block == 0 {
        bail!("unsupported quantization type for dequantization: {typ:?}");
    }

    match typ {
        GgmlType::F32 => {
            let mut out = vec![0f32; n_elements];
            for i in 0..n_elements {
                let offset = i * 4;
                out[i] = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
            }
            Ok(out)
        }
        GgmlType::F16 => {
            let mut out = vec![0f32; n_elements];
            for i in 0..n_elements {
                let offset = i * 2;
                out[i] = f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32();
            }
            Ok(out)
        }
        GgmlType::Q4_0 => {
            let n_blocks = (n_elements + block_size - 1) / block_size;
            let mut out = vec![0f32; n_elements];
            for b in 0..n_blocks {
                let block = &data[b * bytes_per_block..];
                let start = b * block_size;
                let end = (start + block_size).min(n_elements);
                let mut tmp = [0f32; 32];
                dequant_q4_0_block(block, &mut tmp);
                out[start..end].copy_from_slice(&tmp[..end - start]);
            }
            Ok(out)
        }
        GgmlType::Q8_0 => {
            let n_blocks = (n_elements + block_size - 1) / block_size;
            let mut out = vec![0f32; n_elements];
            for b in 0..n_blocks {
                let block = &data[b * bytes_per_block..];
                let start = b * block_size;
                let end = (start + block_size).min(n_elements);
                let mut tmp = [0f32; 32];
                dequant_q8_0_block(block, &mut tmp);
                out[start..end].copy_from_slice(&tmp[..end - start]);
            }
            Ok(out)
        }
        GgmlType::Q4K => {
            let n_blocks = (n_elements + block_size - 1) / block_size;
            let mut out = vec![0f32; n_elements];
            for b in 0..n_blocks {
                let block = &data[b * bytes_per_block..];
                let start = b * block_size;
                let end = (start + block_size).min(n_elements);
                let mut tmp = [0f32; 256];
                dequant_q4_k_block(block, &mut tmp);
                out[start..end].copy_from_slice(&tmp[..end - start]);
            }
            Ok(out)
        }
        GgmlType::Q6K => {
            let n_blocks = (n_elements + block_size - 1) / block_size;
            let mut out = vec![0f32; n_elements];
            for b in 0..n_blocks {
                let block = &data[b * bytes_per_block..];
                let start = b * block_size;
                let end = (start + block_size).min(n_elements);
                let mut tmp = [0f32; 256];
                dequant_q6_k_block(block, &mut tmp);
                out[start..end].copy_from_slice(&tmp[..end - start]);
            }
            Ok(out)
        }
        _ => bail!("dequantization not implemented for {typ:?}"),
    }
}

/// Convert FP32 slice to FP16 bytes (for ANE weight blob).
pub fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; data.len() * 2];
    for (i, &val) in data.iter().enumerate() {
        let h = f16::from_f32(val);
        out[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_dequant() {
        // Create a simple Q4_0 block: scale=1.0, all values = 8 (→ 0 after -8 offset)
        let scale = f16::from_f32(1.0);
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&scale.to_le_bytes());
        // Fill with 0x88 (both nibbles = 8, so value = 0 after -8)
        for i in 2..18 {
            block[i] = 0x88;
        }

        let mut out = [0f32; 32];
        dequant_q4_0_block(&block, &mut out);

        for &v in &out {
            assert!((v - 0.0).abs() < 0.01, "expected ~0.0, got {v}");
        }
    }

    #[test]
    fn test_q8_0_dequant() {
        let scale = f16::from_f32(0.5);
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&scale.to_le_bytes());
        // All values = 2 (as i8)
        for i in 2..34 {
            block[i] = 2;
        }

        let mut out = [0f32; 32];
        dequant_q8_0_block(&block, &mut out);

        for &v in &out {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }
}
