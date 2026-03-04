//! ane-bridge: Safe Rust wrappers for Apple Neural Engine via private framework FFI
//!
//! Provides compile → eval → free lifecycle for MIL programs on ANE hardware.
//! All matmuls should be expressed as 1x1 convolutions for optimal ANE throughput.

use std::ptr;
use std::sync::Once;

use anyhow::{Result, bail};
use half::f16;

// Raw FFI bindings to our C ABI
mod ffi {
    use std::ffi::c_void;

    #[repr(C)]
    pub struct ANEKernel {
        _opaque: [u8; 0],
    }

    unsafe extern "C" {
        pub fn ane_init() -> i32;
        pub fn ane_compile(
            mil_text: *const u8,
            mil_len: usize,
            weight_data: *const u8,
            weight_len: usize,
            n_inputs: i32,
            input_sizes: *const usize,
            n_outputs: i32,
            output_sizes: *const usize,
        ) -> *mut ANEKernel;
        pub fn ane_write_input(k: *mut ANEKernel, idx: i32, data: *const c_void, bytes: usize);
        pub fn ane_read_output(k: *mut ANEKernel, idx: i32, data: *mut c_void, bytes: usize);
        pub fn ane_resize_io(
            k: *mut ANEKernel,
            n_inputs: i32,
            input_sizes: *const usize,
            n_outputs: i32,
            output_sizes: *const usize,
        );
        pub fn ane_eval(k: *mut ANEKernel) -> bool;
        pub fn ane_eval_procedure(k: *mut ANEKernel, proc_idx: i32) -> bool;
        pub fn ane_num_procedures(k: *mut ANEKernel) -> i32;
        pub fn ane_free(k: *mut ANEKernel);
    }
}

static INIT: Once = Once::new();
static mut INIT_OK: bool = false;

/// Initialize the ANE runtime. Called automatically on first compile.
fn ensure_init() -> Result<()> {
    INIT.call_once(|| {
        let rc = unsafe { ffi::ane_init() };
        unsafe { INIT_OK = rc == 0 };
    });
    if unsafe { INIT_OK } {
        Ok(())
    } else {
        bail!("ANE runtime initialization failed — is AppleNeuralEngine.framework available?")
    }
}

/// A compiled ANE kernel ready for evaluation.
/// Owns the underlying IOSurfaces, temp files, and ANE model handle.
pub struct AneKernel {
    raw: *mut ffi::ANEKernel,
    n_inputs: usize,
    n_outputs: usize,
    input_bytes: Vec<usize>,
    output_bytes: Vec<usize>,
}

// ANEKernel is thread-safe: IOSurface I/O is locked, eval serialized by ANE hardware
unsafe impl Send for AneKernel {}

impl AneKernel {
    /// Compile a MIL program with optional baked weights.
    ///
    /// - `mil`: MIL program text (UTF-8)
    /// - `weights`: raw weight blob (64-byte global header + per-chunk headers + FP16 data)
    /// - `input_sizes`: byte sizes for each input IOSurface
    /// - `output_sizes`: byte sizes for each output IOSurface
    pub fn compile(
        mil: &str,
        weights: Option<&[u8]>,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<Self> {
        ensure_init()?;

        let (w_ptr, w_len) = match weights {
            Some(w) => (w.as_ptr(), w.len()),
            None => (ptr::null(), 0),
        };

        let raw = unsafe {
            ffi::ane_compile(
                mil.as_ptr(),
                mil.len(),
                w_ptr,
                w_len,
                input_sizes.len() as i32,
                input_sizes.as_ptr(),
                output_sizes.len() as i32,
                output_sizes.as_ptr(),
            )
        };

        if raw.is_null() {
            bail!("ANE compilation failed — check MIL syntax and weight blob format");
        }

        Ok(Self {
            raw,
            n_inputs: input_sizes.len(),
            n_outputs: output_sizes.len(),
            input_bytes: input_sizes.to_vec(),
            output_bytes: output_sizes.to_vec(),
        })
    }

    /// Write FP32 data to an input tensor, converting layout for ANE.
    /// The caller must ensure `data` has the correct number of elements.
    pub fn write_input_f32(&self, idx: usize, data: &[f32]) {
        assert!(idx < self.n_inputs, "input index out of range");
        let expected = self.input_bytes[idx];
        let bytes = data.len() * 4;
        assert_eq!(bytes, expected, "input size mismatch: got {bytes}, expected {expected}");
        unsafe {
            ffi::ane_write_input(self.raw, idx as i32, data.as_ptr().cast(), bytes);
        }
    }

    /// Write raw bytes to an input IOSurface.
    pub fn write_input_raw(&self, idx: usize, data: &[u8]) {
        assert!(idx < self.n_inputs);
        assert_eq!(data.len(), self.input_bytes[idx]);
        unsafe {
            ffi::ane_write_input(self.raw, idx as i32, data.as_ptr().cast(), data.len());
        }
    }

    /// Read FP32 data from an output tensor.
    pub fn read_output_f32(&self, idx: usize, out: &mut [f32]) {
        assert!(idx < self.n_outputs, "output index out of range");
        let expected = self.output_bytes[idx];
        let bytes = out.len() * 4;
        assert_eq!(bytes, expected, "output size mismatch: got {bytes}, expected {expected}");
        unsafe {
            ffi::ane_read_output(self.raw, idx as i32, out.as_mut_ptr().cast(), bytes);
        }
    }

    /// Read raw bytes from an output IOSurface.
    pub fn read_output_raw(&self, idx: usize, out: &mut [u8]) {
        assert!(idx < self.n_outputs);
        assert_eq!(out.len(), self.output_bytes[idx]);
        unsafe {
            ffi::ane_read_output(self.raw, idx as i32, out.as_mut_ptr().cast(), out.len());
        }
    }

    /// Resize IOSurfaces without recompiling the model.
    pub fn resize_io(&mut self, input_sizes: &[usize], output_sizes: &[usize]) {
        unsafe {
            ffi::ane_resize_io(
                self.raw,
                input_sizes.len() as i32,
                input_sizes.as_ptr(),
                output_sizes.len() as i32,
                output_sizes.as_ptr(),
            );
        }
        self.input_bytes = input_sizes.to_vec();
        self.output_bytes = output_sizes.to_vec();
    }

    /// Execute the kernel on ANE hardware (procedure 0 / single-procedure model).
    pub fn eval(&self) -> Result<()> {
        let ok = unsafe { ffi::ane_eval(self.raw) };
        if ok {
            Ok(())
        } else {
            bail!("ANE evaluation failed")
        }
    }

    /// Execute a specific procedure within a multi-procedure model.
    pub fn eval_procedure(&self, proc_idx: usize) -> Result<()> {
        let ok = unsafe { ffi::ane_eval_procedure(self.raw, proc_idx as i32) };
        if ok {
            Ok(())
        } else {
            bail!("ANE evaluation of procedure {proc_idx} failed")
        }
    }

    /// Get the number of procedures in this model.
    pub fn num_procedures(&self) -> usize {
        unsafe { ffi::ane_num_procedures(self.raw) as usize }
    }
}

impl Drop for AneKernel {
    fn drop(&mut self) {
        unsafe { ffi::ane_free(self.raw) };
    }
}

/// Build an ANE weight blob from FP32 weights.
///
/// Format: 64-byte global header + per-chunk (64-byte chunk header + FP16 data).
/// Each chunk has magic 0xDEADBEEF, version 0x01, data size, data offset.
pub fn build_weight_blob(weight_sets: &[&[f32]]) -> Vec<u8> {
    let mut chunks: Vec<Vec<u8>> = Vec::new();

    for weights in weight_sets {
        let fp16_bytes = weights.len() * 2;
        let mut chunk = vec![0u8; 64 + fp16_bytes];

        // Chunk header: magic
        chunk[0] = 0xEF;
        chunk[1] = 0xBE;
        chunk[2] = 0xAD;
        chunk[3] = 0xDE;
        // Version
        chunk[4] = 0x01;
        // Data size (at offset 8, little-endian u32)
        chunk[8..12].copy_from_slice(&(fp16_bytes as u32).to_le_bytes());
        // Data offset is computed later (absolute from file start)
        // Will be patched below

        // Convert FP32 → FP16
        let fp16_data = &mut chunk[64..];
        for (i, &val) in weights.iter().enumerate() {
            let h = f16::from_f32(val);
            fp16_data[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
        }

        chunks.push(chunk);
    }

    // Build final blob: 64-byte global header + all chunks
    let total: usize = 64 + chunks.iter().map(|c| c.len()).sum::<usize>();
    let mut blob = vec![0u8; total];

    // Global header
    blob[0] = 0x01;
    blob[4] = 0x02;

    // Copy chunks and patch data offsets
    let mut offset = 64usize;
    for chunk in &chunks {
        blob[offset..offset + chunk.len()].copy_from_slice(chunk);
        // Patch the data_offset field (at chunk + 16) to absolute position
        let data_abs = (offset + 64) as u32;
        blob[offset + 16..offset + 20].copy_from_slice(&data_abs.to_le_bytes());
        offset += chunk.len();
    }

    blob
}

/// Convenience: build weight blob for a single weight matrix [out_ch, in_ch].
pub fn build_single_weight_blob(weights: &[f32]) -> Vec<u8> {
    build_weight_blob(&[weights])
}

/// Transpose a row-major [rows, cols] matrix to channel-first [cols, rows] layout
/// for ANE's [1, C, 1, S] tensor format.
pub fn transpose_to_channels_first(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows * cols);
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_blob_format() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix
        let blob = build_single_weight_blob(&w);

        // Global header
        assert_eq!(blob[0], 0x01);
        assert_eq!(blob[4], 0x02);

        // Chunk magic at offset 64
        assert_eq!(&blob[64..68], &[0xEF, 0xBE, 0xAD, 0xDE]);
        // Chunk version
        assert_eq!(blob[68], 0x01);
        // Data size: 4 floats * 2 bytes = 8
        assert_eq!(u32::from_le_bytes([blob[72], blob[73], blob[74], blob[75]]), 8);
        // Data offset: 64 (global) + 64 (chunk header) = 128
        assert_eq!(
            u32::from_le_bytes([blob[80], blob[81], blob[82], blob[83]]),
            128
        );

        // Verify FP16 values at offset 128
        let h1 = f16::from_le_bytes([blob[128], blob[129]]);
        assert!((h1.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_multi_weight_blob() {
        let w1 = vec![1.0f32; 4];
        let w2 = vec![2.0f32; 4];
        let blob = build_weight_blob(&[&w1, &w2]);

        // Two chunks: each 64 + 8 = 72 bytes
        // Total: 64 (global) + 72 + 72 = 208
        assert_eq!(blob.len(), 208);

        // First chunk magic at 64
        assert_eq!(&blob[64..68], &[0xEF, 0xBE, 0xAD, 0xDE]);
        // Second chunk magic at 64 + 72 = 136
        assert_eq!(&blob[136..140], &[0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn test_transpose() {
        // 2x3 matrix: [[1,2,3],[4,5,6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose_to_channels_first(&data, 2, 3);
        // Expected 3x2: [[1,4],[2,5],[3,6]]
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
