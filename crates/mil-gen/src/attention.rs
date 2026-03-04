//! MIL generation for attention layers (QKV projections + output projection as 1x1 convs)

use crate::{MIL_HEADER, MIL_FOOTER, CONV_PREAMBLE, mil_conv_op};

/// Generate MIL for fused Q/K/V projections.
/// Input: [1, dim, 1, spatial] fp32
/// Outputs: q[1, dim, 1, spatial], k[1, dim, 1, spatial], v[1, dim, 1, spatial] fp32
///
/// Weight blob layout: Wq at offset 64, Wk at offset 64+chunk_size, Wv at offset 64+2*chunk_size
/// where chunk_size = 64 + dim*dim*2
pub fn mil_gen_qkv(dim: usize, spatial: usize) -> String {
    let chunk_size = 64 + dim * dim * 2; // chunk header + FP16 data
    let wq_offset = 64u64; // past global header, first chunk's data starts at global(64) + chunk_header(64) = 128
    // But the BLOBFILE offset points to the chunk data start, which is at:
    // global_header(64) + chunk_0_header_data_offset
    // In maderix's format: offset=64 means "skip 64 bytes from file start" which lands at the chunk header
    // Then the chunk's data_offset field says where the actual data is
    // Actually, looking at maderix: offset = uint64(64) means the FP16 data starts at byte 64+64=128 in the file
    // But the MIL BLOBFILE offset=64 means "64 bytes from start of the weight section"
    // Let me re-read: the offset in BLOBFILE is from the start of the file
    // First weight data is at: 64 (global header) + 64 (chunk header) = 128
    // But maderix uses offset=64 which works because the descriptor wdict sets offset=0 for the data
    // The BLOBFILE offset is relative to the weight blob data as passed in the dict
    // Actually in ane_compile, wdict = {"@model_path/weights/weight.bin": {"offset": 0, "data": weightData}}
    // So BLOBFILE offset=64 means 64 bytes into the weight blob, which is the chunk header start
    // But data is at chunk_header + 64... unless the MIL compiler reads the chunk format itself
    // Looking more carefully at maderix's code: offset=uint64(64) for first weight
    // The weight blob has: [64 global header][64 chunk header][fp16 data]
    // So offset=64 skips the global header, the MIL compiler must understand the chunk format
    // Actually no — looking at the MIL, it says offset=64 and that's passed as a raw byte offset
    // The data at position 64 is the chunk header (DEADBEEF), not the FP16 data
    // The ANE compiler must parse the chunk header to find the actual data
    // This is confirmed by chunk[16] = 128 (absolute data offset)

    // For fused QKV, offsets are: 64, 64+cs, 64+2*cs (matching maderix's mil_gen_qkv)
    let wk_offset = 64 + chunk_size as u64;
    let wv_offset = 64 + 2 * chunk_size as u64;

    let mut s = String::with_capacity(4096);
    s.push_str(MIL_HEADER);
    s.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {spatial}]> x) {{\n"
    ));
    s.push_str(CONV_PREAMBLE);

    // Cast input to fp16
    s.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
    ));

    // Q, K, V convolutions
    s.push_str(&mil_conv_op("Wq", "conv_q", "x16", dim, dim, spatial, wq_offset));
    s.push_str("\n");
    s.push_str(&mil_conv_op("Wk", "conv_k", "x16", dim, dim, spatial, wk_offset));
    s.push_str("\n");
    s.push_str(&mil_conv_op("Wv", "conv_v", "x16", dim, dim, spatial, wv_offset));
    s.push_str("\n");

    // Cast outputs back to fp32
    s.push_str(&format!(
        "        tensor<fp32, [1, {dim}, 1, {spatial}]> q = cast(dtype = to_fp32, x = conv_q)[name = string(\"cast_q\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp32, [1, {dim}, 1, {spatial}]> k = cast(dtype = to_fp32, x = conv_k)[name = string(\"cast_k\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp32, [1, {dim}, 1, {spatial}]> v = cast(dtype = to_fp32, x = conv_v)[name = string(\"cast_v\")];\n"
    ));

    s.push_str("    } -> (q, k, v);\n");
    s.push_str(MIL_FOOTER);
    s
}

/// Generate MIL for output projection (single conv).
/// Input: [1, dim, 1, spatial] fp32
/// Output: [1, dim, 1, spatial] fp32
pub fn mil_gen_output_proj(dim: usize, spatial: usize) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str(MIL_HEADER);
    s.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {spatial}]> x) {{\n"
    ));
    s.push_str(CONV_PREAMBLE);
    s.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
    ));
    s.push_str(&mil_conv_op("Wo", "conv_o", "x16", dim, dim, spatial, 64));
    s.push_str("\n");
    s.push_str(&format!(
        "        tensor<fp32, [1, {dim}, 1, {spatial}]> y = cast(dtype = to_fp32, x = conv_o)[name = string(\"cast_out\")];\n"
    ));
    s.push_str("    } -> (y);\n");
    s.push_str(MIL_FOOTER);
    s
}

/// Generate MIL for a single conv/linear projection.
/// Input: [1, in_ch, 1, spatial] fp32
/// Output: [1, out_ch, 1, spatial] fp32
pub fn mil_gen_conv(in_ch: usize, out_ch: usize, spatial: usize) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str(MIL_HEADER);
    s.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {spatial}]> x) {{\n"
    ));
    s.push_str(CONV_PREAMBLE);
    s.push_str(&format!(
        "        tensor<fp16, [1, {in_ch}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
    ));
    s.push_str(&mil_conv_op("W", "conv", "x16", out_ch, in_ch, spatial, 64));
    s.push_str("\n");
    s.push_str(&format!(
        "        tensor<fp32, [1, {out_ch}, 1, {spatial}]> y = cast(dtype = to_fp32, x = conv)[name = string(\"cast_out\")];\n"
    ));
    s.push_str("    } -> (y);\n");
    s.push_str(MIL_FOOTER);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qkv_mil_generation() {
        let mil = mil_gen_qkv(768, 64);
        assert!(mil.contains("program(1.3)"));
        assert!(mil.contains("conv_q"));
        assert!(mil.contains("conv_k"));
        assert!(mil.contains("conv_v"));
        assert!(mil.contains("-> (q, k, v)"));
        assert!(mil.contains("[768, 768, 1, 1]"));
    }

    #[test]
    fn test_conv_mil_generation() {
        let mil = mil_gen_conv(768, 2048, 64);
        assert!(mil.contains("[2048, 768, 1, 1]"));
        assert!(mil.contains("[1, 2048, 1, 64]"));
    }
}
