//! Multi-threaded Q8_0 and Q4_0 quantized GEMV for Apple Silicon.
//!
//! Parallelizes across output rows with rayon, NEON for inner dot product.

use half::f16;
use rayon::prelude::*;

pub struct Q8Tensor {
    pub data: Vec<u8>,
    pub m: usize,
    pub n: usize,
}

pub struct Q4Tensor {
    pub data: Vec<u8>,
    pub m: usize,
    pub n: usize,
}

const Q8_BLOCK: usize = 32;
const Q8_BYTES: usize = 34;

const Q4_BLOCK: usize = 32;
const Q4_BYTES: usize = 18;

impl Q8Tensor {
    pub fn from_raw(data: Vec<u8>, ne0: usize, ne1: usize) -> Self {
        let m = ne1;
        let n = ne0;
        assert_eq!(n % Q8_BLOCK, 0);
        let expected = m * (n / Q8_BLOCK) * Q8_BYTES;
        assert_eq!(
            data.len(),
            expected,
            "Q8 size mismatch: got {}, expected {} (m={m}, n={n})",
            data.len(),
            expected
        );
        Self { data, m, n }
    }
}

impl Q4Tensor {
    pub fn from_raw(data: Vec<u8>, ne0: usize, ne1: usize) -> Self {
        let m = ne1;
        let n = ne0;
        assert_eq!(n % Q4_BLOCK, 0);
        let expected = m * (n / Q4_BLOCK) * Q4_BYTES;
        assert_eq!(
            data.len(),
            expected,
            "Q4 size mismatch: got {}, expected {} (m={m}, n={n})",
            data.len(),
            expected
        );
        Self { data, m, n }
    }
}

#[inline]
fn q4_unpack(byte: u8, offset: usize) -> f32 {
    let nibble = ((byte >> (offset * 4)) & 0x0F) as i32;
    (nibble - 8) as f32
}

pub fn q4_gemv(w: &Q4Tensor, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.n);
    assert_eq!(y.len(), w.m);

    let bpr = w.n / Q4_BLOCK;
    let data = &w.data;

    y.par_iter_mut().enumerate().for_each(|(row, y_out)| {
        *y_out = compute_row_q4(data, x, row, bpr);
    });
}

fn compute_row_q4(data: &[u8], x: &[f32], row: usize, bpr: usize) -> f32 {
    let row_off = row * bpr * Q4_BYTES;
    let mut sum = 0.0f32;

    for b in 0..bpr {
        let off = row_off + b * Q4_BYTES;
        let scale = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        let mut block_sum = 0.0f32;
        for i in 0..16 {
            let packed = data[off + 2 + i];
            let v0 = q4_unpack(packed, 0);
            let v1 = q4_unpack(packed, 1);
            block_sum += v0 * x[b * 32 + i * 2] + v1 * x[b * 32 + i * 2 + 1];
        }
        sum += block_sum * scale;
    }
    sum
}

/// Multi-threaded Q8_0 GEMV: y = W @ x
pub fn q8_gemv(w: &Q8Tensor, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.n);
    assert_eq!(y.len(), w.m);

    let bpr = w.n / Q8_BLOCK;
    let data = &w.data;

    y.par_iter_mut().enumerate().for_each(|(row, y_out)| {
        *y_out = compute_row(data, x, row, bpr);
    });
}

#[cfg(target_arch = "aarch64")]
fn compute_row(data: &[u8], x: &[f32], row: usize, bpr: usize) -> f32 {
    use std::arch::aarch64::*;

    let row_off = row * bpr * Q8_BYTES;
    let mut sum = 0.0f32;

    // Two independent accumulators saturate both M4 FMA pipes.
    unsafe {
        for b in 0..bpr {
            let off = row_off + b * Q8_BYTES;
            let scale = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
            let vals = data.as_ptr().add(off + 2);
            let x_ptr = x.as_ptr().add(b * Q8_BLOCK);

            // acc0 handles elements [0..3], [8..11], [16..19], [24..27]
            // acc1 handles elements [4..7], [12..15], [20..23], [28..31]
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);

            // --- first 16 bytes of weight block ---
            let w8_0 = vld1q_s8(vals as *const i8);
            let w16_lo = vmovl_s8(vget_low_s8(w8_0)); // elements 0..7
            let w16_hi = vmovl_s8(vget_high_s8(w8_0)); // elements 8..15

            acc0 = vfmaq_f32(
                acc0,
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_lo))),
                vld1q_f32(x_ptr),
            );
            acc1 = vfmaq_f32(
                acc1,
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_lo))),
                vld1q_f32(x_ptr.add(4)),
            );
            acc0 = vfmaq_f32(
                acc0,
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_hi))),
                vld1q_f32(x_ptr.add(8)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_hi))),
                vld1q_f32(x_ptr.add(12)),
            );

            // --- second 16 bytes of weight block ---
            let w8_1 = vld1q_s8(vals.add(16) as *const i8);
            let w16_lo2 = vmovl_s8(vget_low_s8(w8_1)); // elements 16..23
            let w16_hi2 = vmovl_s8(vget_high_s8(w8_1)); // elements 24..31

            acc0 = vfmaq_f32(
                acc0,
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_lo2))),
                vld1q_f32(x_ptr.add(16)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_lo2))),
                vld1q_f32(x_ptr.add(20)),
            );
            acc0 = vfmaq_f32(
                acc0,
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_hi2))),
                vld1q_f32(x_ptr.add(24)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_hi2))),
                vld1q_f32(x_ptr.add(28)),
            );

            // Merge the two independent chains before the horizontal add.
            sum += vaddvq_f32(vaddq_f32(acc0, acc1)) * scale;
        }
    }

    sum
}

#[cfg(not(target_arch = "aarch64"))]
fn compute_row(data: &[u8], x: &[f32], row: usize, bpr: usize) -> f32 {
    let row_off = row * bpr * Q8_BYTES;
    let mut sum = 0.0f32;
    for b in 0..bpr {
        let off = row_off + b * Q8_BYTES;
        let scale = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
        let mut block_sum = 0.0f32;
        for i in 0..32 {
            block_sum += data[off + 2 + i] as i8 as f32 * x[b * 32 + i];
        }
        sum += block_sum * scale;
    }
    sum
}

pub fn q8_gemv_scalar(w: &Q8Tensor, x: &[f32], y: &mut [f32]) {
    let bpr = w.n / Q8_BLOCK;
    for row in 0..w.m {
        y[row] = compute_row(&w.data, x, row, bpr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_gemv_identity() {
        let m = 4;
        let n = 32;
        let mut data = vec![0u8; m * Q8_BYTES];
        for row in 0..m {
            let off = row * Q8_BYTES;
            let scale = f16::from_f32(1.0);
            data[off..off + 2].copy_from_slice(&scale.to_le_bytes());
            if row < 32 {
                data[off + 2 + row] = 1i8 as u8;
            }
        }
        let w = Q8Tensor::from_raw(data, n, m);
        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut y = vec![0f32; m];
        q8_gemv(&w, &x, &mut y);
        for i in 0..m {
            assert!(
                (y[i] - i as f32).abs() < 0.01,
                "y[{i}] = {}, expected {i}",
                y[i]
            );
        }
    }

    #[test]
    fn test_q8_gemv_parallel() {
        let m = 256;
        let n = 1024;
        let mut data = vec![0u8; m * (n / Q8_BLOCK) * Q8_BYTES];
        for row in 0..m {
            let bpr = n / Q8_BLOCK;
            for b in 0..bpr {
                let off = row * bpr * Q8_BYTES + b * Q8_BYTES;
                let scale = f16::from_f32(0.5);
                data[off..off + 2].copy_from_slice(&scale.to_le_bytes());
                for i in 0..32 {
                    data[off + 2 + i] = ((row + i) % 256) as i8 as u8;
                }
            }
        }
        let w = Q8Tensor::from_raw(data, n, m);
        let x: Vec<f32> = (0..n).map(|i| (i % 100) as f32).collect();
        let mut y = vec![0f32; m];
        q8_gemv(&w, &x, &mut y);
        let mut y_scalar = vec![0f32; m];
        q8_gemv_scalar(&w, &x, &mut y_scalar);
        for i in 0..m {
            let diff = (y[i] - y_scalar[i]).abs();
            assert!(
                diff < 1e-3,
                "row {i}: parallel={}, scalar={}",
                y[i],
                y_scalar[i]
            );
        }
    }
}
