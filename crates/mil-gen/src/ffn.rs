//! MIL generation for FFN (feed-forward network) layers
//!
//! Llama-style SwiGLU FFN: output = w2(SiLU(w1(x)) * w3(x))
//! w1 (gate_proj) and w3 (up_proj) are fused into a single MIL program.
//! w2 (down_proj) is a separate kernel.

use crate::{MIL_HEADER, MIL_FOOTER, CONV_PREAMBLE, mil_conv_op};

/// Generate MIL for fused FFN up projections (gate_proj + up_proj).
/// Input: [1, dim, 1, spatial] fp32
/// Outputs: h1[1, hidden_dim, 1, spatial], h3[1, hidden_dim, 1, spatial] fp32
///
/// Weight blob: W1 at chunk 0, W3 at chunk 1
/// chunk_size = 64 + hidden_dim * dim * 2
pub fn mil_gen_ffn_up(dim: usize, hidden_dim: usize, spatial: usize) -> String {
    let chunk_size = 64 + hidden_dim * dim * 2;
    let w1_offset = 64u64;
    let w3_offset = 64 + chunk_size as u64;

    let mut s = String::with_capacity(4096);
    s.push_str(MIL_HEADER);
    s.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {spatial}]> x) {{\n"
    ));
    s.push_str(CONV_PREAMBLE);
    s.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
    ));

    // gate_proj (w1) and up_proj (w3) — parallel convs
    s.push_str(&mil_conv_op("W1", "conv_w1", "x16", hidden_dim, dim, spatial, w1_offset));
    s.push_str("\n");
    s.push_str(&mil_conv_op("W3", "conv_w3", "x16", hidden_dim, dim, spatial, w3_offset));
    s.push_str("\n");

    // Cast back to fp32
    s.push_str(&format!(
        "        tensor<fp32, [1, {hidden_dim}, 1, {spatial}]> out1 = cast(dtype = to_fp32, x = conv_w1)[name = string(\"cast_h1\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp32, [1, {hidden_dim}, 1, {spatial}]> out3 = cast(dtype = to_fp32, x = conv_w3)[name = string(\"cast_h3\")];\n"
    ));

    s.push_str("    } -> (out1, out3);\n");
    s.push_str(MIL_FOOTER);
    s
}

/// Generate MIL for FFN down projection (w2).
/// Input: [1, hidden_dim, 1, spatial] fp32
/// Output: [1, dim, 1, spatial] fp32
pub fn mil_gen_ffn_down(dim: usize, hidden_dim: usize, spatial: usize) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str(MIL_HEADER);
    s.push_str(&format!(
        "    func main<ios18>(tensor<fp32, [1, {hidden_dim}, 1, {spatial}]> x) {{\n"
    ));
    s.push_str(CONV_PREAMBLE);
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
    ));
    s.push_str(&mil_conv_op("W2", "conv_w2", "x16", dim, hidden_dim, spatial, 64));
    s.push_str("\n");
    s.push_str(&format!(
        "        tensor<fp32, [1, {dim}, 1, {spatial}]> y = cast(dtype = to_fp32, x = conv_w2)[name = string(\"cast_out\")];\n"
    ));
    s.push_str("    } -> (y);\n");
    s.push_str(MIL_FOOTER);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_up_mil() {
        let mil = mil_gen_ffn_up(768, 2048, 64);
        assert!(mil.contains("conv_w1"));
        assert!(mil.contains("conv_w3"));
        assert!(mil.contains("-> (out1, out3)"));
        assert!(mil.contains("[2048, 768, 1, 1]"));
    }

    #[test]
    fn test_ffn_down_mil() {
        let mil = mil_gen_ffn_down(768, 2048, 64);
        assert!(mil.contains("conv_w2"));
        assert!(mil.contains("[768, 2048, 1, 1]"));
    }
}
