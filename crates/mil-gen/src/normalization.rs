//! RMSNorm — always runs on CPU (element-wise, not worth ANE dispatch overhead)

/// CPU RMSNorm: out[i] = x[i] * w[i] / sqrt(mean(x^2) + eps)
/// x: [seq_len, dim] row-major
/// w: [dim] weights
/// out: [seq_len, dim] row-major
pub fn cpu_rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], seq_len: usize, dim: usize) {
    for t in 0..seq_len {
        let row = &x[t * dim..(t + 1) * dim];
        let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv_rms = 1.0 / (ss + 1e-5_f32).sqrt();
        let out_row = &mut out[t * dim..(t + 1) * dim];
        for i in 0..dim {
            out_row[i] = row[i] * inv_rms * w[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm() {
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 1x4
        let w = vec![1.0; 4];
        let mut out = vec![0.0; 4];
        cpu_rmsnorm(&mut out, &x, &w, 1, 4);

        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        // out[0] = 1/2.7386 ≈ 0.3651
        assert!((out[0] - 0.3651).abs() < 0.01);
    }
}
