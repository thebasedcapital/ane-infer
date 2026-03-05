#include <metal_stdlib>
using namespace metal;

struct block_q4_0 {
    half  d;
    uint8_t qs[16];
};

#define QK4_0 32
#define N_SIMDWIDTH 32
#define N_SIMDGROUPS 4
#define NR0 2
#define NQ 8

// GGUF Q4_0: elements 0-15 = low nibbles of bytes 0-15
//            elements 16-31 = high nibbles of bytes 0-15
inline float q4_decode(device const uint8_t* qs, int elem) {
    int byte_idx = elem & 15;         // byte 0-15
    int is_high = (elem >> 4) & 1;    // 0 for elem 0-15, 1 for elem 16-31
    uint8_t byte = qs[byte_idx];
    int nibble = is_high ? (byte >> 4) : (byte & 0x0F);
    return float(nibble) - 8.0f;
}

kernel void q4_gemv(
    device const block_q4_0 * W [[buffer(0)]],
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

    device const block_q4_0 * w_rows[NR0];
    for (short row = 0; row < NR0; ++row) {
        const int row_idx = r0 + row;
        if (row_idx < int(m)) {
            w_rows[row] = W + row_idx * nb;
        }
    }

    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += N_SIMDGROUPS * NQ) {
        device const float * yb = x + ib * QK4_0 + il * NQ;
        
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; ++row) {
            if (r0 + row >= int(m)) continue;
            
            const float d = float(w_rows[row][ib].d);

            float sumq = 0.0f;
            for (short i = 0; i < NQ; ++i) {
                const int elem_in_block = il * NQ + i;
                const float qv = q4_decode(w_rows[row][ib].qs, elem_in_block);
                sumq += qv * yl[i];
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