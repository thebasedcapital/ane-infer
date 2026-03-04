//! Metal GPU Q8_0 GEMV — persistent buffers, batched command encoding.
//!
//! Key optimizations vs naive approach:
//! 1. Weight buffers created ONCE at model load (no per-dispatch allocation)
//! 2. Input/output buffers are persistent and reused
//! 3. Multiple GEMVs can be encoded into one command buffer before committing

use anyhow::{bail, Result};
use metal::*;
use std::path::Path;

use crate::q8_gemv::Q8Tensor;

/// A weight matrix uploaded to GPU memory once.
pub struct GpuWeightBuffer {
    pub buffer: Buffer,
    pub m: usize,
    pub n: usize,
    pub n_blocks: u32,
}

/// Metal GPU context with persistent buffers.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    gemv_pipeline: ComputePipelineState,
    // Persistent scratch buffers for input/output
    input_buf: Buffer,
    output_buf: Buffer,
    params_buf: Buffer, // [n_blocks, m] packed
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device found"))?;
        let queue = device.new_command_queue();

        // Load shader — try multiple paths
        let library = Self::load_library(&device)?;

        let gemv_fn = library
            .get_function("q8_gemv", None)
            .map_err(|e| anyhow::anyhow!("q8_gemv not found: {e}"))?;
        let gemv_pipeline = device
            .new_compute_pipeline_state_with_function(&gemv_fn)
            .map_err(|e| anyhow::anyhow!("pipeline creation failed: {e}"))?;

        // Pre-allocate max-size scratch buffers
        // Max input: dim=2048 or hidden_dim=6144 → 6144 * 4 = 24KB
        // Max output: hidden_dim=6144 or vocab=248320 → 248320 * 4 ≈ 1MB
        let max_input = 8192 * 4; // 32KB
        let max_output = 256000 * 4; // 1MB
        let input_buf = device.new_buffer(max_input as u64, MTLResourceOptions::StorageModeShared);
        let output_buf =
            device.new_buffer(max_output as u64, MTLResourceOptions::StorageModeShared);
        let params_buf = device.new_buffer(8, MTLResourceOptions::StorageModeShared); // 2x u32

        Ok(Self {
            device,
            queue,
            gemv_pipeline,
            input_buf,
            output_buf,
            params_buf,
        })
    }

    fn load_library(device: &Device) -> Result<Library> {
        let paths = [
            "crates/engine/metal/q8_gemv.metallib",
            "metal/q8_gemv.metallib",
            "q8_gemv.metallib",
        ];
        for p in &paths {
            if Path::new(p).exists() {
                if let Ok(lib) = device.new_library_with_file(p) {
                    return Ok(lib);
                }
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            let p = exe
                .parent()
                .unwrap_or(Path::new("."))
                .join("q8_gemv.metallib");
            if p.exists() {
                if let Ok(lib) = device.new_library_with_file(p) {
                    return Ok(lib);
                }
            }
        }
        bail!("q8_gemv.metallib not found")
    }

    /// Upload Q8 weight matrix to GPU (call once at model load).
    pub fn upload_weights(&self, q8: &Q8Tensor) -> GpuWeightBuffer {
        let buffer = self.device.new_buffer_with_data(
            q8.data.as_ptr() as *const _,
            q8.data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        GpuWeightBuffer {
            buffer,
            m: q8.m,
            n: q8.n,
            n_blocks: (q8.n / 32) as u32,
        }
    }

    /// Q8_0 GEMV on GPU with pre-uploaded weights.
    pub fn q8_gemv(&self, w: &GpuWeightBuffer, x: &[f32], y: &mut [f32]) -> Result<()> {
        let m = w.m as u32;
        let n_blocks = w.n_blocks;

        // Copy input to persistent buffer
        unsafe {
            let ptr = self.input_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(x.as_ptr(), ptr, x.len());
        }

        // Set params
        unsafe {
            let ptr = self.params_buf.contents() as *mut u32;
            *ptr = n_blocks;
            *ptr.add(1) = m;
        }

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.gemv_pipeline);
        encoder.set_buffer(0, Some(&w.buffer), 0);
        encoder.set_buffer(1, Some(&self.input_buf), 0);
        encoder.set_buffer(2, Some(&self.output_buf), 0);
        encoder.set_buffer(3, Some(&self.params_buf), 0);
        encoder.set_buffer(4, Some(&self.params_buf), 4); // m at offset 4

        let threads = MTLSize::new(m as u64, 1, 1);
        let tg_size = MTLSize::new(
            self.gemv_pipeline.thread_execution_width().min(m as u64),
            1,
            1,
        );
        let tg_count = MTLSize::new(m as u64, 1, 1);
        let tg_size = MTLSize::new(128, 1, 1);
        encoder.dispatch_thread_groups(tg_count, tg_size);
        encoder.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read output
        unsafe {
            let ptr = self.output_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, y.as_mut_ptr(), y.len());
        }

        Ok(())
    }

    /// Batched GEMV: encode ALL projections for a layer into ONE command buffer.
    /// This eliminates the per-dispatch overhead that was killing Metal performance.
    /// ops: list of (weight_tensor, input_slice, output_slice)
    pub fn q8_gemv_batch(&self, ops: Vec<(&Q8Tensor, &[f32], &mut [f32])>) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }

        // Create zero-copy weight buffers and input buffers for all ops
        let mut w_bufs = Vec::with_capacity(ops.len());
        let mut x_bufs = Vec::with_capacity(ops.len());
        let mut y_bufs = Vec::with_capacity(ops.len());
        let mut params = Vec::with_capacity(ops.len());

        for (w, x, _y) in &ops {
            let w_buf = self.device.new_buffer_with_bytes_no_copy(
                w.data.as_ptr() as *const _,
                w.data.len() as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let x_buf = self.device.new_buffer_with_data(
                x.as_ptr() as *const _,
                (x.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let y_buf = self
                .device
                .new_buffer((_y.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
            w_bufs.push(w_buf);
            x_bufs.push(x_buf);
            y_bufs.push(y_buf);
            params.push(((w.n / 32) as u32, w.m as u32));
        }

        // Encode ALL operations into ONE command buffer
        let cmd_buf = self.queue.new_command_buffer();

        for (i, (nb, m)) in params.iter().enumerate() {
            let nb_buf = self.device.new_buffer_with_data(
                nb as *const u32 as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            let m_buf = self.device.new_buffer_with_data(
                m as *const u32 as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            );

            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.gemv_pipeline);
            encoder.set_buffer(0, Some(&w_bufs[i]), 0);
            encoder.set_buffer(1, Some(&x_bufs[i]), 0);
            encoder.set_buffer(2, Some(&y_bufs[i]), 0);
            encoder.set_buffer(3, Some(&nb_buf), 0);
            encoder.set_buffer(4, Some(&m_buf), 0);

            let threads = MTLSize::new(*m as u64, 1, 1);
            let tg = MTLSize::new(
                self.gemv_pipeline.thread_execution_width().min(*m as u64),
                1,
                1,
            );
            encoder.dispatch_threads(threads, tg);
            encoder.end_encoding();
        }

        // ONE commit, ONE wait for ALL operations
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read ALL outputs
        for (i, (_, _, y)) in ops.into_iter().enumerate() {
            unsafe {
                let ptr = y_bufs[i].contents() as *const f32;
                std::ptr::copy_nonoverlapping(ptr, y.as_mut_ptr(), y.len());
            }
        }

        Ok(())
    }

    /// GEMV using raw Q8Tensor — zero-copy GPU access via unified memory.
    pub fn q8_gemv_adhoc(&self, w: &Q8Tensor, x: &[f32], y: &mut [f32]) -> Result<()> {
        let m = w.m as u32;
        let n_blocks = (w.n / 32) as u32;

        // Zero-copy: GPU reads directly from CPU memory via unified memory
        let w_buf = self.device.new_buffer_with_bytes_no_copy(
            w.data.as_ptr() as *const _,
            w.data.len() as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        // Copy input to persistent buffer
        unsafe {
            let ptr = self.input_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(x.as_ptr(), ptr, x.len());
        }

        // Set params
        unsafe {
            let ptr = self.params_buf.contents() as *mut u32;
            *ptr = n_blocks;
            *ptr.add(1) = m;
        }

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.gemv_pipeline);
        encoder.set_buffer(0, Some(&w_buf), 0);
        encoder.set_buffer(1, Some(&self.input_buf), 0);
        encoder.set_buffer(2, Some(&self.output_buf), 0);
        encoder.set_buffer(3, Some(&self.params_buf), 0);
        encoder.set_buffer(4, Some(&self.params_buf), 4);

        let threads = MTLSize::new(m as u64, 1, 1);
        let tg_size = MTLSize::new(
            self.gemv_pipeline.thread_execution_width().min(m as u64),
            1,
            1,
        );
        let tg_count = MTLSize::new(m as u64, 1, 1);
        let tg_size = MTLSize::new(128, 1, 1);
        encoder.dispatch_thread_groups(tg_count, tg_size);
        encoder.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read output
        unsafe {
            let ptr = self.output_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, y.as_mut_ptr(), y.len());
        }

        Ok(())
    }
}
