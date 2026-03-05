#include <metal_stdlib>
using namespace metal;

#define N_HEADS 8
#define N_KV_HEADS 2
#define HEAD_DIM 256
#define HEADS_PER_KV 4

kernel void rope_apply(
    device float * qk [[buffer(0)]],
    constant uint & pos [[buffer(1)]],
    constant float & freq_base [[buffer(2)]],
    constant uint & n_heads [[buffer(3)]],
    constant uint & head_dim [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total_pairs = n_heads * (head_dim / 2);
    if (tid >= total_pairs) return;

    const uint pair_idx = tid % (head_dim / 2);
    const uint head_idx = tid / (head_dim / 2);

    const float freq = 1.0f / pow(freq_base, float(pair_idx) / float(head_dim / 2));
    const float val = float(pos) * freq;
    const float cos_v = cos(val);
    const float sin_v = sin(val);

    const uint base_idx = head_idx * head_dim + pair_idx * 2;
    const float q0 = qk[base_idx];
    const float q1 = qk[base_idx + 1];
    qk[base_idx] = q0 * cos_v - q1 * sin_v;
    qk[base_idx + 1] = q0 * sin_v + q1 * cos_v;
}

kernel void sdpa_causal(
    device const float * q [[buffer(0)]],
    device const float * k_cache [[buffer(1)]],
    device const float * v_cache [[buffer(2)]],
    device float * attn_out [[buffer(3)]],
    constant uint & seq_len [[buffer(4)]],
    constant uint & n_heads [[buffer(5)]],
    constant uint & n_kv_heads [[buffer(6)]],
    constant uint & head_dim [[buffer(7)]],
    constant float & scale [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint q_head = tg_id;
    if (q_head >= n_heads) return;

    const uint kv_head = q_head / HEADS_PER_KV;
    const uint q_off = q_head * head_dim;
    const uint kv_off = kv_head * head_dim;

    // Online softmax: single pass with running max and correction
    float max_score = -INFINITY;
    float exp_sum = 0.0f;
    float out_local = 0.0f;  // Each thread accumulates one output dimension

    // Each thread handles one output dimension (tiisg = 0..31 maps to d = 0..255)
    const uint d = tiisg;
    if (d >= head_dim) return;

    // Single pass over KV cache
    for (uint s = 0; s < seq_len; s++) {
        // Compute Q·K score
        float score = 0.0f;
        device const float * k_ptr = k_cache + s * n_kv_heads * head_dim + kv_off;
        for (uint dd = 0; dd < head_dim; dd++) {
            score += q[q_off + dd] * k_ptr[dd];
        }
        score *= scale;

        // Online softmax update
        const float new_max = max(max_score, score);
        const float correction = exp(max_score - new_max);

        // Update running stats
        exp_sum = exp_sum * correction + exp(score - new_max);

        // Update output with correction
        const float v_val = v_cache[s * n_kv_heads * head_dim + kv_off + d];
        out_local = out_local * correction + exp(score - new_max) * v_val;

        max_score = new_max;
    }

    // Final normalize
    attn_out[q_off + d] = out_local / exp_sum;
}

kernel void sigmoid_gate(
    device const float * x [[buffer(0)]],
    device const float * gate [[buffer(1)]],
    device float * out [[buffer(2)]],
    constant uint & n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    const float g = gate[tid];
    out[tid] = x[tid] * (1.0f / (1.0f + exp(-g)));
}

kernel void q_gate_split(
    device const float * packed [[buffer(0)]],
    device float * q [[buffer(1)]],
    device float * gate [[buffer(2)]],
    constant uint & n_heads [[buffer(3)]],
    constant uint & head_dim [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = n_heads * head_dim;
    if (tid >= total) return;

    const uint head_idx = tid / head_dim;
    const uint dim_idx = tid % head_dim;

    const uint packed_off = head_idx * (head_dim * 2) + dim_idx;
    q[tid] = packed[packed_off];
    gate[tid] = packed[packed_off + head_dim];
}