#include <metal_stdlib>
using namespace metal;

struct block_q8_0 {
    half  d;
    char  qs[32];
};

#define QK8_0 32
#define N_SIMDWIDTH 32
#define N_SIMDGROUPS 4
#define NR0 2
#define NQ 8

kernel void q8_gemv(
    device const block_q8_0 * W [[buffer(0)]],
    device const float      * x [[buffer(1)]],
    device       float      * y [[buffer(2)]],
    constant uint & n_blocks      [[buffer(3)]],
    constant uint & m             [[buffer(4)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int nb = int(n_blocks);
    const int r0 = int(tg_id) * NR0;
    
    if (r0 >= int(m)) return;

    float sumf[NR0] = { 0.0f, 0.0f };

    const short ix = tiisg / (N_SIMDWIDTH / NQ);
    const short il = tiisg % (N_SIMDWIDTH / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const block_q8_0 * w_rows[NR0];
    for (short row = 0; row < NR0; ++row) {
        const int row_idx = r0 + row;
        if (row_idx < int(m)) {
            w_rows[row] = W + row_idx * nb;
        }
    }

    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += N_SIMDGROUPS * NQ) {
        device const float * yb = x + ib * QK8_0 + il * NQ;
        
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; ++row) {
            if (r0 + row >= int(m)) continue;
            
            device const int8_t * qs = (device const int8_t *)(w_rows[row][ib].qs) + il * NQ;
            const float d = float(w_rows[row][ib].d);

            float sumq = 0.0f;
            for (short i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += sumq * d;
        }
    }

    threadgroup float shmem[NR0][N_SIMDGROUPS];
    
    for (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
        if (tiisg == 0) {
            shmem[row][sgitg] = sumf[row];
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        for (short row = 0; row < NR0 && r0 + row < int(m); ++row) {
            float total = 0.0f;
            for (short s = 0; s < N_SIMDGROUPS; ++s) {
                total += shmem[row][s];
            }
            y[r0 + row] = total;
        }
    }
}

kernel void silu_mul(
    device       float * gate [[buffer(0)]],
    device const float * up   [[buffer(1)]],
    constant uint & n         [[buffer(2)]],
    uint tid                  [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float g   = gate[tid];
    gate[tid] = (g / (1.0f + exp(-g))) * up[tid];
}