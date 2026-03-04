#include <metal_stdlib>
using namespace metal;

// Q8_0 block: 2 bytes f16 scale + 32 bytes i8 values = 34 bytes total
struct block_q8_0 {
    half  scale;        // 2 bytes
    char  values[32];   // 32 bytes
};

// ---------------------------------------------------------------------------
// Q8_0 GEMV — threadgroup-parallel version
//
// Grid:  (m)          threadgroups, one per output row
// Block: (128)        threads  = 4 simdgroups x 32 lanes
//
// Strategy (mirrors llama.cpp metal_q8_0):
//   1. All 128 threads cooperatively load the input vector x into threadgroup
//      SRAM (shared_x).  Each thread loads one float at a time with stride 128
//      until n_blocks*32 floats are covered.
//   2. After a barrier, each simdgroup iterates over its own strided slice of
//      blocks for this row, accumulates a local partial sum, then reduces
//      within the simdgroup with simd_sum().
//   3. Lane 0 of each simdgroup writes its partial sum to sg_partial[].
//   4. Thread 0 of the threadgroup sums sg_partial[0..3] and writes y[row].
// ---------------------------------------------------------------------------

#define THREADS_PER_TG  128u
#define N_SIMDGROUPS    4u          // THREADS_PER_TG / 32
#define MAX_SHARED_X    8176u       // 8176*4 + 4*4 = 32720 bytes < 32768 limit

kernel void q8_gemv(
    device const block_q8_0 * W [[buffer(0)]],   // [m rows, n_blocks blocks/row]
    device const float      * x [[buffer(1)]],   // input vector  [n_blocks * 32]
    device       float      * y [[buffer(2)]],   // output vector [m]
    constant uint & n_blocks      [[buffer(3)]],  // blocks per row  (n / 32)
    constant uint & m             [[buffer(4)]],  // number of output rows
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid_tg  [[thread_position_in_threadgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]])
{
    // One threadgroup per output row; bail if out of range.
    if (tg_id >= m) return;

    const uint row        = tg_id;
    const uint n_floats   = n_blocks * 32u;  // total input elements

    // ------------------------------------------------------------------
    // 1. Cooperative load of x into threadgroup shared memory.
    //    Each thread grabs elements at indices tid_tg, tid_tg+128, …
    // ------------------------------------------------------------------
    threadgroup float shared_x[MAX_SHARED_X];

    for (uint i = tid_tg; i < n_floats; i += THREADS_PER_TG) {
        shared_x[i] = x[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------------
    // 2. Each simdgroup handles blocks sg_id, sg_id+4, sg_id+8, …
    //    (stride = N_SIMDGROUPS).  Within a block the 32 q8 values are
    //    split across the 32 lanes: lane k processes element k.
    // ------------------------------------------------------------------
    float simd_acc = 0.0f;

    const uint row_base = row * n_blocks;

    for (uint b = sg_id; b < n_blocks; b += N_SIMDGROUPS) {
        device const block_q8_0 & blk = W[row_base + b];
        const float scale   = float(blk.scale);
        const uint  x_base  = b * 32u;

        // Lane k handles element k of this block (32 lanes, 32 elements).
        float q  = float(blk.values[sg_lane]);
        float xi = shared_x[x_base + sg_lane];
        simd_acc += scale * q * xi;
    }

    // Reduce within simdgroup — each lane's contribution summed to lane 0.
    float sg_sum = simd_sum(simd_acc);

    // ------------------------------------------------------------------
    // 3. Lane 0 of each simdgroup publishes partial sum.
    // ------------------------------------------------------------------
    threadgroup float sg_partial[N_SIMDGROUPS];

    if (sg_lane == 0u) {
        sg_partial[sg_id] = sg_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------------
    // 4. Thread 0 accumulates the 4 simdgroup partials and writes output.
    // ------------------------------------------------------------------
    if (tid_tg == 0u) {
        float row_sum = 0.0f;
        for (uint s = 0u; s < N_SIMDGROUPS; s++) {
            row_sum += sg_partial[s];
        }
        y[row] = row_sum;
    }
}

// ---------------------------------------------------------------------------
// Fused SiLU(gate) * up  — used for FFN gate projection.
// gate[i] = sigmoid(gate[i]) * gate[i] * up[i]   (SiLU activation * up)
// Applied element-wise; one thread per element.
// ---------------------------------------------------------------------------
kernel void silu_mul(
    device       float * gate [[buffer(0)]],   // in/out: SiLU in-place * up
    device const float * up   [[buffer(1)]],
    constant uint & n         [[buffer(2)]],
    uint tid                  [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float g   = gate[tid];
    gate[tid] = (g / (1.0f + exp(-g))) * up[tid];
}
