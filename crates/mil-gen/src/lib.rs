//! mil-gen: MIL program text generator for ANE transformer layers
//!
//! All linear layers expressed as 1x1 convolutions for optimal ANE throughput.
//! Tensor layout: [1, C, 1, S] (batch=1, channels, height=1, spatial=sequence).

mod attention;
mod ffn;
pub mod mega;
mod normalization;

pub use attention::*;
pub use ffn::*;
pub use mega::*;
pub use normalization::*;

/// MIL program header — matches the coremltools version that ANE expects.
pub const MIL_HEADER: &str = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
"#;

/// MIL program footer
pub const MIL_FOOTER: &str = "}\n";

/// Common conv constants preamble (shared by all conv-based ops)
pub const CONV_PREAMBLE: &str = r#"        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
"#;

/// Generate a multi-procedure MIL program with N independent conv projections.
/// Each procedure is a separate function (layer0, layer1, ...) addressable by procedureIndex.
/// All procedures share the same weight blob but reference different offsets.
///
/// `procs`: Vec of (func_name, in_ch, out_ch, spatial, blob_offset)
pub fn mil_gen_multi_procedure(procs: &[(String, usize, usize, usize, u64)]) -> String {
    let mut s = String::with_capacity(4096 * procs.len());
    s.push_str(MIL_HEADER);

    for (i, (name, in_ch, out_ch, spatial, blob_offset)) in procs.iter().enumerate() {
        s.push_str(&format!(
            "    func {name}<ios18>(tensor<fp32, [1, {in_ch}, 1, {spatial}]> x{i}) {{\n"
        ));
        // Each procedure needs its own const names to avoid conflicts
        s.push_str(&format!(
            r#"        string pt_{i} = const()[name = string("pt_{i}"), val = string("valid")];
        tensor<int32, [2]> st_{i} = const()[name = string("st_{i}"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pd_{i} = const()[name = string("pd_{i}"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dl_{i} = const()[name = string("dl_{i}"), val = tensor<int32, [2]>([1, 1])];
        int32 gr_{i} = const()[name = string("gr_{i}"), val = int32(1)];
        string to16_{i} = const()[name = string("to16_{i}"), val = string("fp16")];
        string to32_{i} = const()[name = string("to32_{i}"), val = string("fp32")];
        tensor<fp16, [1, {in_ch}, 1, {spatial}]> x16_{i} = cast(dtype = to16_{i}, x = x{i})[name = string("cin_{i}")];
        tensor<fp16, [{out_ch}, {in_ch}, 1, 1]> W_{i} = const()[name = string("W_{i}"), val = tensor<fp16, [{out_ch}, {in_ch}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({blob_offset})))];
        tensor<fp16, [1, {out_ch}, 1, {spatial}]> y16_{i} = conv(dilations = dl_{i}, groups = gr_{i}, pad = pd_{i}, pad_type = pt_{i}, strides = st_{i}, weight = W_{i}, x = x16_{i})[name = string("conv_{i}")];
        tensor<fp32, [1, {out_ch}, 1, {spatial}]> y{i} = cast(dtype = to32_{i}, x = y16_{i})[name = string("cout_{i}")];
"#
        ));
        s.push_str(&format!("    }} -> (y{i});\n\n"));
    }

    s.push_str(MIL_FOOTER);
    s
}

/// Generate a single conv operation within a MIL function body.
/// `weight_name`: unique name for this weight tensor
/// `conv_name`: unique name for the conv op
/// `input_var`: name of the fp16 input variable
/// `out_ch`, `in_ch`: weight dimensions
/// `spatial`: sequence length
/// `blob_offset`: byte offset in weight.bin for this weight's FP16 data
/// Returns (weight_const_line, conv_line, output_var_name) — all fp16
pub fn mil_conv_op(
    weight_name: &str,
    conv_name: &str,
    input_var: &str,
    out_ch: usize,
    in_ch: usize,
    spatial: usize,
    blob_offset: u64,
) -> String {
    format!(
        r#"        tensor<fp16, [{out_ch}, {in_ch}, 1, 1]> {weight_name} = const()[name = string("{weight_name}"), val = tensor<fp16, [{out_ch}, {in_ch}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({blob_offset})))];
        tensor<fp16, [1, {out_ch}, 1, {spatial}]> {conv_name} = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = {weight_name}, x = {input_var})[name = string("{conv_name}")];"#
    )
}
