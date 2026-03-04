//! Mega-kernel MIL generators — chain 6+ ops into single ANE programs.
//!
//! This is where the real ANE performance comes from.
//! Single-op kernels achieve ~1.66 TFLOPS (8.7% utilization).
//! Chained mega-kernels achieve ~15-19 TFLOPS (80-94% utilization).

use crate::{MIL_HEADER, MIL_FOOTER, CONV_PREAMBLE};

/// Generate a fused FFN mega-kernel: gate_proj + SiLU + up_proj + mul + down_proj
/// All in one MIL program = ONE ANE dispatch.
///
/// Input:  [1, dim, 1, S] fp32
/// Output: [1, dim, 1, S] fp32
///
/// Weight blob layout:
///   Chunk 0: gate_proj [hidden_dim, dim, 1, 1] at offset 64
///   Chunk 1: up_proj   [hidden_dim, dim, 1, 1] at offset 64 + cs_gate
///   Chunk 2: down_proj [dim, hidden_dim, 1, 1] at offset 64 + cs_gate + cs_up
///
/// Operations chained (6 ops):
///   1. cast input fp32 → fp16
///   2. conv(gate_proj, input) → h1 [hidden_dim]
///   3. sigmoid(h1) → sig
///   4. mul(h1, sig) → silu (= SiLU(h1))
///   5. conv(up_proj, input) → h3 [hidden_dim]
///   6. mul(silu, h3) → gated [hidden_dim]
///   7. conv(down_proj, gated) → output [dim]
///   8. cast output fp16 → fp32
pub fn mil_gen_fused_ffn(dim: usize, hidden_dim: usize, spatial: usize) -> String {
    // Weight blob chunk sizes (64-byte chunk header + FP16 data)
    let cs_gate = 64 + hidden_dim * dim * 2; // gate_proj chunk
    let cs_up = 64 + hidden_dim * dim * 2;   // up_proj chunk
    // down_proj chunk starts after gate + up

    let gate_offset = 64u64; // past global header
    let up_offset = 64 + cs_gate as u64;
    let down_offset = 64 + cs_gate as u64 + cs_up as u64;

    let mut s = String::with_capacity(8192);
    s.push_str(MIL_HEADER);
    s.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {spatial}]> x) {{\n"
    ));
    s.push_str(CONV_PREAMBLE);

    // Cast input to fp16
    s.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
    ));

    // gate_proj: [hidden_dim, dim, 1, 1] conv
    s.push_str(&format!(
        "        tensor<fp16, [{hidden_dim}, {dim}, 1, 1]> W_gate = const()[name = string(\"W_gate\"), val = tensor<fp16, [{hidden_dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({gate_offset})))];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> h1 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_gate, x = x16)[name = string(\"conv_gate\")];\n"
    ));

    // SiLU(h1) = h1 * sigmoid(h1)
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> sig = sigmoid(x = h1)[name = string(\"sigmoid\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> silu = mul(x = h1, y = sig)[name = string(\"silu\")];\n"
    ));

    // up_proj: [hidden_dim, dim, 1, 1] conv
    s.push_str(&format!(
        "        tensor<fp16, [{hidden_dim}, {dim}, 1, 1]> W_up = const()[name = string(\"W_up\"), val = tensor<fp16, [{hidden_dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({up_offset})))];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> h3 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_up, x = x16)[name = string(\"conv_up\")];\n"
    ));

    // gated = silu * h3
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> gated = mul(x = silu, y = h3)[name = string(\"gate_mul\")];\n"
    ));

    // down_proj: [dim, hidden_dim, 1, 1] conv
    s.push_str(&format!(
        "        tensor<fp16, [{dim}, {hidden_dim}, 1, 1]> W_down = const()[name = string(\"W_down\"), val = tensor<fp16, [{dim}, {hidden_dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({down_offset})))];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {spatial}]> out16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_down, x = gated)[name = string(\"conv_down\")];\n"
    ));

    // Cast output to fp32
    s.push_str(&format!(
        "        tensor<fp32, [1, {dim}, 1, {spatial}]> y = cast(dtype = to_fp32, x = out16)[name = string(\"cast_out\")];\n"
    ));

    s.push_str("    } -> (y);\n");
    s.push_str(MIL_FOOTER);
    s
}

/// Generate fused dual-projection: two convs from same input, two outputs.
/// Used for DeltaNet gate + ssm_out (both dim→dim from same input).
pub fn mil_gen_fused_dual_proj(in_dim: usize, out_a: usize, out_b: usize, spatial: usize) -> String {
    let cs_a = 64 + out_a * in_dim * 2;
    let a_offset = 64u64;
    let b_offset = 64 + cs_a as u64;

    let mut s = String::with_capacity(4096);
    s.push_str(MIL_HEADER);
    s.push_str(&format!("    func main<ios18>(tensor<fp32, [1, {in_dim}, 1, {spatial}]> x) {{\n"));
    s.push_str(CONV_PREAMBLE);
    s.push_str(&format!("        tensor<fp16, [1, {in_dim}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"));
    s.push_str(&format!("        tensor<fp16, [{out_a}, {in_dim}, 1, 1]> Wa = const()[name = string(\"Wa\"), val = tensor<fp16, [{out_a}, {in_dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({a_offset})))];\n"));
    s.push_str(&format!("        tensor<fp16, [1, {out_a}, 1, {spatial}]> ha = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wa, x = x16)[name = string(\"conv_a\")];\n"));
    s.push_str(&format!("        tensor<fp16, [{out_b}, {in_dim}, 1, 1]> Wb = const()[name = string(\"Wb\"), val = tensor<fp16, [{out_b}, {in_dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({b_offset})))];\n"));
    s.push_str(&format!("        tensor<fp16, [1, {out_b}, 1, {spatial}]> hb = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wb, x = x16)[name = string(\"conv_b\")];\n"));
    s.push_str(&format!("        tensor<fp32, [1, {out_a}, 1, {spatial}]> a = cast(dtype = to_fp32, x = ha)[name = string(\"cast_a\")];\n"));
    s.push_str(&format!("        tensor<fp32, [1, {out_b}, 1, {spatial}]> b = cast(dtype = to_fp32, x = hb)[name = string(\"cast_b\")];\n"));
    s.push_str("    } -> (a, b);\n");
    s.push_str(MIL_FOOTER);
    s
}

/// Generate fused triple-projection: three convs from same input.
/// Used for FullAttn WQ+WK+WV (potentially different output dims).
pub fn mil_gen_fused_triple_proj(in_dim: usize, out_a: usize, out_b: usize, out_c: usize, spatial: usize) -> String {
    let cs_a = 64 + out_a * in_dim * 2;
    let cs_b = 64 + out_b * in_dim * 2;
    let a_offset = 64u64;
    let b_offset = 64 + cs_a as u64;
    let c_offset = 64 + cs_a as u64 + cs_b as u64;

    let mut s = String::with_capacity(4096);
    s.push_str(MIL_HEADER);
    s.push_str(&format!("    func main<ios18>(tensor<fp32, [1, {in_dim}, 1, {spatial}]> x) {{\n"));
    s.push_str(CONV_PREAMBLE);
    s.push_str(&format!("        tensor<fp16, [1, {in_dim}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"));
    s.push_str(&format!("        tensor<fp16, [{out_a}, {in_dim}, 1, 1]> Wa = const()[name = string(\"Wa\"), val = tensor<fp16, [{out_a}, {in_dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({a_offset})))];\n"));
    s.push_str(&format!("        tensor<fp16, [1, {out_a}, 1, {spatial}]> ha = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wa, x = x16)[name = string(\"conv_a\")];\n"));
    s.push_str(&format!("        tensor<fp16, [{out_b}, {in_dim}, 1, 1]> Wb = const()[name = string(\"Wb\"), val = tensor<fp16, [{out_b}, {in_dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({b_offset})))];\n"));
    s.push_str(&format!("        tensor<fp16, [1, {out_b}, 1, {spatial}]> hb = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wb, x = x16)[name = string(\"conv_b\")];\n"));
    s.push_str(&format!("        tensor<fp16, [{out_c}, {in_dim}, 1, 1]> Wc = const()[name = string(\"Wc\"), val = tensor<fp16, [{out_c}, {in_dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({c_offset})))];\n"));
    s.push_str(&format!("        tensor<fp16, [1, {out_c}, 1, {spatial}]> hc = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wc, x = x16)[name = string(\"conv_c\")];\n"));
    s.push_str(&format!("        tensor<fp32, [1, {out_a}, 1, {spatial}]> a = cast(dtype = to_fp32, x = ha)[name = string(\"cast_a\")];\n"));
    s.push_str(&format!("        tensor<fp32, [1, {out_b}, 1, {spatial}]> b = cast(dtype = to_fp32, x = hb)[name = string(\"cast_b\")];\n"));
    s.push_str(&format!("        tensor<fp32, [1, {out_c}, 1, {spatial}]> c = cast(dtype = to_fp32, x = hc)[name = string(\"cast_c\")];\n"));
    s.push_str("    } -> (a, b, c);\n");
    s.push_str(MIL_FOOTER);
    s
}

/// Generate fused gate+SiLU+up+mul kernel (without down_proj to stay under 32MB SRAM).
/// This chains 5 ops: 2 convs + sigmoid + 2 muls.
///
/// Input:  [1, dim, 1, S] fp32
/// Output: [1, hidden_dim, 1, S] fp32 (the gated result, ready for down_proj)
///
/// Weight blob: gate_proj at chunk 0, up_proj at chunk 1
pub fn mil_gen_fused_ffn_gate_up(dim: usize, hidden_dim: usize, spatial: usize) -> String {
    let cs_gate = 64 + hidden_dim * dim * 2;
    let gate_offset = 64u64;
    let up_offset = 64 + cs_gate as u64;

    let mut s = String::with_capacity(4096);
    s.push_str(MIL_HEADER);
    s.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {spatial}]> x) {{\n"
    ));
    s.push_str(CONV_PREAMBLE);

    // Cast input
    s.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
    ));

    // gate_proj conv
    s.push_str(&format!(
        "        tensor<fp16, [{hidden_dim}, {dim}, 1, 1]> W_gate = const()[name = string(\"W_gate\"), val = tensor<fp16, [{hidden_dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({gate_offset})))];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> h1 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_gate, x = x16)[name = string(\"conv_gate\")];\n"
    ));

    // SiLU(h1) = h1 * sigmoid(h1)
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> sig = sigmoid(x = h1)[name = string(\"sigmoid\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> silu = mul(x = h1, y = sig)[name = string(\"silu\")];\n"
    ));

    // up_proj conv
    s.push_str(&format!(
        "        tensor<fp16, [{hidden_dim}, {dim}, 1, 1]> W_up = const()[name = string(\"W_up\"), val = tensor<fp16, [{hidden_dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({up_offset})))];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> h3 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_up, x = x16)[name = string(\"conv_up\")];\n"
    ));

    // gated = silu * h3
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> gated16 = mul(x = silu, y = h3)[name = string(\"gate_mul\")];\n"
    ));

    // Cast output
    s.push_str(&format!(
        "        tensor<fp32, [1, {hidden_dim}, 1, {spatial}]> y = cast(dtype = to_fp32, x = gated16)[name = string(\"cast_out\")];\n"
    ));

    s.push_str("    } -> (y);\n");
    s.push_str(MIL_FOOTER);
    s
}

/// Generate fused QKV projection mega-kernel.
/// 3 parallel convolutions from same input, all in one dispatch.
///
/// Input:  [1, dim, 1, S] fp32
/// Outputs: q[1, dim, 1, S], k[1, dim, 1, S], v[1, dim, 1, S] fp32
///
/// This is 3 convs + 2 casts = 5 ops (was 3 separate kernel dispatches).
pub fn mil_gen_fused_qkv(dim: usize, spatial: usize) -> String {
    let chunk_size = 64 + dim * dim * 2; // per-weight chunk
    let wq_offset = 64u64;
    let wk_offset = 64 + chunk_size as u64;
    let wv_offset = 64 + 2 * chunk_size as u64;

    let mut s = String::with_capacity(4096);
    s.push_str(MIL_HEADER);
    s.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {spatial}]> x) {{\n"
    ));
    s.push_str(CONV_PREAMBLE);

    // Cast input
    s.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
    ));

    // Q conv
    s.push_str(&format!(
        "        tensor<fp16, [{dim}, {dim}, 1, 1]> Wq = const()[name = string(\"Wq\"), val = tensor<fp16, [{dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({wq_offset})))];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {spatial}]> q16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wq, x = x16)[name = string(\"conv_q\")];\n"
    ));

    // K conv
    s.push_str(&format!(
        "        tensor<fp16, [{dim}, {dim}, 1, 1]> Wk = const()[name = string(\"Wk\"), val = tensor<fp16, [{dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({wk_offset})))];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {spatial}]> k16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wk, x = x16)[name = string(\"conv_k\")];\n"
    ));

    // V conv
    s.push_str(&format!(
        "        tensor<fp16, [{dim}, {dim}, 1, 1]> Wv = const()[name = string(\"Wv\"), val = tensor<fp16, [{dim}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({wv_offset})))];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {spatial}]> v16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wv, x = x16)[name = string(\"conv_v\")];\n"
    ));

    // Cast outputs
    s.push_str(&format!(
        "        tensor<fp32, [1, {dim}, 1, {spatial}]> q = cast(dtype = to_fp32, x = q16)[name = string(\"cast_q\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp32, [1, {dim}, 1, {spatial}]> k = cast(dtype = to_fp32, x = k16)[name = string(\"cast_k\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp32, [1, {dim}, 1, {spatial}]> v = cast(dtype = to_fp32, x = v16)[name = string(\"cast_v\")];\n"
    ));

    s.push_str("    } -> (q, k, v);\n");
    s.push_str(MIL_FOOTER);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_ffn_generation() {
        let mil = mil_gen_fused_ffn(2048, 6144, 64);
        assert!(mil.contains("conv_gate"));
        assert!(mil.contains("sigmoid"));
        assert!(mil.contains("silu"));
        assert!(mil.contains("conv_up"));
        assert!(mil.contains("gate_mul"));
        assert!(mil.contains("conv_down"));
        assert!(mil.contains("-> (y)"));
        // Verify 3 weight blobs referenced
        assert!(mil.contains("W_gate"));
        assert!(mil.contains("W_up"));
        assert!(mil.contains("W_down"));
    }

    #[test]
    fn test_fused_qkv_generation() {
        let mil = mil_gen_fused_qkv(2048, 64);
        assert!(mil.contains("conv_q"));
        assert!(mil.contains("conv_k"));
        assert!(mil.contains("conv_v"));
        assert!(mil.contains("-> (q, k, v)"));
    }
}

    #[test]
    fn test_fused_ffn_small() {
        // Test with small dimensions to verify compilation
        let mil = mil_gen_fused_ffn(64, 128, 16);
        assert!(mil.contains("sigmoid"));
        assert!(mil.contains("conv_gate"));
        assert!(mil.contains("conv_down"));
        // Check offsets
        let cs = 64 + 128 * 64 * 2; // 64 + 16384 = 16448
        let up_off = 64 + cs as u64;
        let down_off = 64 + cs as u64 + cs as u64;
        assert!(mil.contains(&format!("offset = uint64(64)")));
        assert!(mil.contains(&format!("offset = uint64({up_off})")));
        assert!(mil.contains(&format!("offset = uint64({down_off})")));
    }

    #[test]
    fn test_fused_gate_up_small() {
        let mil = mil_gen_fused_ffn_gate_up(64, 128, 16);
        assert!(mil.contains("sigmoid"));
        assert!(mil.contains("conv_gate"));
        assert!(mil.contains("conv_up"));
        assert!(!mil.contains("conv_down")); // no down_proj in gate_up version
    }
