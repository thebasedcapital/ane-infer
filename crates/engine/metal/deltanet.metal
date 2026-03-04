#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// DeltaNet recurrence shaders for single-token GPU decode.
// These eliminate CPU sync points between GPU GEMV dispatches.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Conv1d shift + apply + SiLU
// conv_state: [inner_size, kernel_size] — circular buffer, shifts left
// conv_weights: [inner_size, kernel_size]
// input: [inner_size] — new values appended to conv_state
// output: [inner_size] — conv result with SiLU activation
// ---------------------------------------------------------------------------
kernel void conv1d_silu(
    device       float * conv_state    [[buffer(0)]],   // [inner_size * kernel_size]
    device const float * conv_weights  [[buffer(1)]],   // [inner_size * kernel_size]
    device const float * input         [[buffer(2)]],   // [inner_size]
    device       float * output        [[buffer(3)]],   // [inner_size]
    constant uint & inner_size         [[buffer(4)]],
    constant uint & kernel_size        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= inner_size) return;

    // Shift left: state[ch][k] = state[ch][k+1] for k < kernel_size-1
    uint base = tid * kernel_size;
    for (uint k = 0; k < kernel_size - 1; k++) {
        conv_state[base + k] = conv_state[base + k + 1];
    }
    // Append new value
    conv_state[base + kernel_size - 1] = input[tid];

    // Apply conv: dot product of state[ch] with weights[ch]
    float sum = 0.0f;
    for (uint k = 0; k < kernel_size; k++) {
        sum += conv_state[base + k] * conv_weights[tid * kernel_size + k];
    }

    // SiLU activation
    output[tid] = sum / (1.0f + exp(-sum));
}

// ---------------------------------------------------------------------------
// L2 normalize + scale per head
// Operates on [n_heads, head_dim] contiguous vectors.
// Each threadgroup handles one head.
// ---------------------------------------------------------------------------
kernel void l2_normalize_scale(
    device float * vec              [[buffer(0)]],   // [n_heads * head_dim], in-place
    constant uint & head_dim        [[buffer(1)]],
    constant uint & n_heads         [[buffer(2)]],
    constant float & scale          [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid_tg [[thread_position_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]])
{
    if (tg_id >= n_heads) return;
    uint base = tg_id * head_dim;

    // Compute L2 norm using simd reduction
    float norm_sq = 0.0f;
    for (uint i = tid_tg; i < head_dim; i += 32) {
        float v = vec[base + i];
        norm_sq += v * v;
    }
    norm_sq = simd_sum(norm_sq);

    float inv_norm = rsqrt(norm_sq + 1e-12f) * scale;

    // Normalize in-place
    for (uint i = tid_tg; i < head_dim; i += 32) {
        vec[base + i] *= inv_norm;
    }
}

// ---------------------------------------------------------------------------
// DeltaNet recurrence — one head per threadgroup
//
// Per head:
//   1. Decay: S *= decay[h]
//   2. Recall: sk = S^T @ k_h
//   3. Delta: delta = beta[h] * (v_h - sk)
//   4. Update: S += outer(k_h, delta)
//   5. Query: output = S^T @ q_h
//
// state: [n_heads, key_dim, value_dim] — persistent recurrent state
// q, k: [n_heads * head_dim_k] — already normalized+scaled
// v: [n_heads * v_per_head]
// decay, beta: [n_heads] — per-head scalars
// output: [n_heads * head_dim_k] — recurrence output
// ---------------------------------------------------------------------------
kernel void deltanet_recurrence(
    device       float * state     [[buffer(0)]],   // [n_heads * key_dim * value_dim]
    device const float * q         [[buffer(1)]],   // [n_heads * head_dim_k]
    device const float * k         [[buffer(2)]],   // [n_heads * head_dim_k]
    device const float * v         [[buffer(3)]],   // [n_heads * v_per_head]
    device const float * decay     [[buffer(4)]],   // [n_heads]
    device const float * beta      [[buffer(5)]],   // [n_heads]
    device       float * output    [[buffer(6)]],   // [n_heads * head_dim_k]
    constant uint & n_heads        [[buffer(7)]],
    constant uint & key_dim        [[buffer(8)]],
    constant uint & value_dim      [[buffer(9)]],
    constant uint & head_dim_k     [[buffer(10)]],
    constant uint & v_per_head     [[buffer(11)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid_tg [[thread_position_in_threadgroup]])
{
    if (tg_id >= n_heads) return;
    uint h = tg_id;

    uint kd = min(key_dim, head_dim_k);
    uint vd = min(value_dim, v_per_head);

    device float * s = state + h * key_dim * value_dim;
    device const float * q_h = q + h * head_dim_k;
    device const float * k_h = k + h * head_dim_k;
    device const float * v_h = v + h * v_per_head;
    float d = decay[h];
    float b = beta[h];

    // Step 1: Decay state
    for (uint i = tid_tg; i < kd * vd; i += 32) {
        s[i] *= d;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Step 2: Recall — sk[vi] = sum_ki(S[ki,vi] * k_h[ki])
    // Step 3: Delta — delta[vi] = beta * (v_h[vi] - sk[vi])
    // Step 4: Update — S[ki,vi] += k_h[ki] * delta[vi]
    // These must be sequential per-element but we parallelize across vi
    for (uint vi = tid_tg; vi < vd; vi += 32) {
        // Recall
        float sk = 0.0f;
        for (uint ki = 0; ki < kd; ki++) {
            sk += s[ki * value_dim + vi] * k_h[ki];
        }
        // Delta
        float delta = b * (v_h[vi] - sk);
        // Update
        for (uint ki = 0; ki < kd; ki++) {
            s[ki * value_dim + vi] += k_h[ki] * delta;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Step 5: Query — output[vi] = sum_ki(S[ki,vi] * q_h[ki])
    for (uint vi = tid_tg; vi < vd && vi < head_dim_k; vi += 32) {
        float dot = 0.0f;
        for (uint ki = 0; ki < kd; ki++) {
            dot += s[ki * value_dim + vi] * q_h[ki];
        }
        output[h * head_dim_k + vi] = dot;
    }
}

// ---------------------------------------------------------------------------
// Compute beta/decay from raw projections
// beta[h] = sigmoid(beta_raw[h])
// decay[h] = exp(ssm_a[h] * softplus(alpha_raw[h] + dt_bias[h]))
// ---------------------------------------------------------------------------
kernel void compute_beta_decay(
    device const float * beta_raw   [[buffer(0)]],  // [n_heads]
    device const float * alpha_raw  [[buffer(1)]],  // [n_heads]
    device const float * ssm_a      [[buffer(2)]],  // [n_heads]
    device const float * dt_bias    [[buffer(3)]],  // [n_heads]
    device       float * beta_out   [[buffer(4)]],  // [n_heads]
    device       float * decay_out  [[buffer(5)]],  // [n_heads]
    constant uint & n_heads         [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_heads) return;
    beta_out[tid] = 1.0f / (1.0f + exp(-beta_raw[tid]));
    float sp = log(1.0f + exp(alpha_raw[tid] + dt_bias[tid]));
    decay_out[tid] = exp(ssm_a[tid] * sp);
}

// ---------------------------------------------------------------------------
// RMSNorm — per-head, with optional gate multiplication
// output[i] = x[i] * inv_rms * weight[i % norm_dim]
// If gate != nullptr: output[i] *= SiLU(gate[i])
// ---------------------------------------------------------------------------
kernel void rmsnorm_gated(
    device const float * x          [[buffer(0)]],   // [total_dim]
    device const float * weight     [[buffer(1)]],   // [norm_dim]
    device const float * gate       [[buffer(2)]],   // [total_dim] or nullptr
    device       float * output     [[buffer(3)]],   // [total_dim]
    constant uint & total_dim       [[buffer(4)]],
    constant uint & norm_dim        [[buffer(5)]],
    constant uint & n_heads         [[buffer(6)]],
    constant float & eps            [[buffer(7)]],
    constant uint & has_gate        [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid_tg [[thread_position_in_threadgroup]])
{
    if (tg_id >= n_heads) return;
    uint head_dim = total_dim / n_heads;
    uint base = tg_id * head_dim;
    uint nd = min(head_dim, norm_dim);

    // Compute RMS
    float ss = 0.0f;
    for (uint i = tid_tg; i < nd; i += 32) {
        float v = x[base + i];
        ss += v * v;
    }
    ss = simd_sum(ss) / float(nd);
    float inv_rms = rsqrt(ss + eps);

    // Normalize and optionally gate
    for (uint i = tid_tg; i < nd; i += 32) {
        float val = x[base + i] * inv_rms * weight[i];
        if (has_gate != 0) {
            float g = gate[base + i];
            val *= g / (1.0f + exp(-g)); // SiLU(gate)
        }
        output[base + i] = val;
    }
}

// ---------------------------------------------------------------------------
// Simple element-wise operations
// ---------------------------------------------------------------------------

// residual_add: x[i] += y[i]
kernel void residual_add(
    device       float * x  [[buffer(0)]],
    device const float * y  [[buffer(1)]],
    constant uint & n       [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    x[tid] += y[tid];
}

// rmsnorm_simple: out[i] = x[i] * inv_rms * w[i]
// Single threadgroup (128 threads = 4 simdgroups)
kernel void rmsnorm_simple(
    device const float * x      [[buffer(0)]],
    device const float * w      [[buffer(1)]],
    device       float * out    [[buffer(2)]],
    constant uint & dim         [[buffer(3)]],
    constant float & eps        [[buffer(4)]],
    uint tid_tg [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]])
{
    threadgroup float sg_partial[4];

    float local_ss = 0.0f;
    for (uint i = tid_tg; i < dim; i += 128) {
        float v = x[i];
        local_ss += v * v;
    }
    float sg_sum = simd_sum(local_ss);
    if (sg_lane == 0) sg_partial[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_ss;
    if (tid_tg == 0) {
        total_ss = sg_partial[0] + sg_partial[1] + sg_partial[2] + sg_partial[3];
        sg_partial[0] = total_ss; // broadcast
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_ss = sg_partial[0];

    float inv_rms = rsqrt(total_ss / float(dim) + eps);
    for (uint i = tid_tg; i < dim; i += 128) {
        out[i] = x[i] * inv_rms * w[i];
    }
}

// GPU-side memcpy: dst[i] = src[i]
kernel void gpu_memcpy(
    device const float * src [[buffer(0)]],
    device       float * dst [[buffer(1)]],
    constant uint & n        [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    dst[tid] = src[tid];
}

// SiLU in-place: x[i] = x[i] / (1 + exp(-x[i]))
kernel void silu_inplace(
    device float * x     [[buffer(0)]],
    constant uint & n    [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float v = x[tid];
    x[tid] = v / (1.0f + exp(-v));
}
