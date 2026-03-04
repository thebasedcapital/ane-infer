//! Metal GPU Computational Graph Engine for LLM inference decode.
//!
//! Key insight: Instead of per-GEMV commit+wait (~0.5ms overhead × 150 dispatches),
//! we encode ALL GEMVs for one token decode into ONE command buffer.
//! This matches llama.cpp's approach for achieving 31 tok/s.

use anyhow::{bail, Result};
use metal::*;
use std::path::Path;

use crate::q8_gemv::Q8Tensor;

/// A weight matrix uploaded to GPU memory (one-time at model load).
pub struct GpuBuffer {
    pub buffer: Buffer,
    pub m: usize,
    pub n: usize,
    pub n_blocks: u32,
}

/// A single GEMV operation in the computational graph.
pub struct GemvOp {
    pub weight_buffer: GpuBuffer,
    pub input_offset: usize,
    pub output_offset: usize,
    pub input_len: usize,
    pub output_len: usize,
}

/// Fused SiLU(gate) * up on the GPU output scratch buffer.
/// Both gate and up are float slices at offsets into scratch_output.
/// Result is written back in-place to gate_offset.
pub struct SiluMulOp {
    /// Offset (in f32 elements) into scratch_output where gate lives (in/out).
    pub gate_offset: usize,
    /// Offset (in f32 elements) into scratch_output where up lives (read-only).
    pub up_offset: usize,
    /// Number of elements.
    pub n: usize,
}

enum GraphOp {
    Gemv(GemvOp),
    SiluMul(SiluMulOp),
}

/// Metal GPU context with persistent resources.
pub struct GpuContext {
    device: Device,
    queue: CommandQueue,
    gemv_pipeline: ComputePipelineState,
    silu_pipeline: ComputePipelineState,
    scratch_input: Buffer,
    scratch_output: Buffer,
    scratch_input_size: usize,
    scratch_output_size: usize,
}

impl GpuContext {
    pub fn new() -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device found"))?;
        let queue = device.new_command_queue();

        let library = Self::load_library(&device)?;

        let gemv_fn = library
            .get_function("q8_gemv", None)
            .map_err(|e| anyhow::anyhow!("q8_gemv not found: {e}"))?;
        let gemv_pipeline = device
            .new_compute_pipeline_state_with_function(&gemv_fn)
            .map_err(|e| anyhow::anyhow!("pipeline creation failed: {e}"))?;

        let silu_fn = library
            .get_function("silu_mul", None)
            .map_err(|e| anyhow::anyhow!("silu_mul not found: {e}"))?;
        let silu_pipeline = device
            .new_compute_pipeline_state_with_function(&silu_fn)
            .map_err(|e| anyhow::anyhow!("silu_mul pipeline creation failed: {e}"))?;

        let scratch_input_size = 128 * 1024;
        let scratch_output_size = 4 * 1024 * 1024;

        let scratch_input = device.new_buffer(
            scratch_input_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scratch_output = device.new_buffer(
            scratch_output_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device,
            queue,
            gemv_pipeline,
            silu_pipeline,
            scratch_input,
            scratch_output,
            scratch_input_size,
            scratch_output_size,
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
    /// Creates a persistent GPU buffer that lives for the model's lifetime.
    pub fn upload_q8_weights(&self, q8: &Q8Tensor) -> GpuBuffer {
        let buffer = self.device.new_buffer_with_data(
            q8.data.as_ptr() as *const _,
            q8.data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        GpuBuffer {
            buffer,
            m: q8.m,
            n: q8.n,
            n_blocks: (q8.n / 32) as u32,
        }
    }

    /// Get scratch buffer pointers for direct CPU access.
    pub fn scratch_input_ptr(&self) -> *mut f32 {
        unsafe { self.scratch_input.contents() as *mut f32 }
    }

    pub fn scratch_output_ptr(&self) -> *const f32 {
        unsafe { self.scratch_output.contents() as *const f32 }
    }

    pub fn scratch_input_size(&self) -> usize {
        self.scratch_input_size
    }

    pub fn scratch_output_size(&self) -> usize {
        self.scratch_output_size
    }

    /// Ensure scratch buffers are large enough.
    pub fn ensure_scratch_capacity(&mut self, input_bytes: usize, output_bytes: usize) {
        if input_bytes > self.scratch_input_size {
            self.scratch_input_size = input_bytes;
            self.scratch_input = self
                .device
                .new_buffer(input_bytes as u64, MTLResourceOptions::StorageModeShared);
        }
        if output_bytes > self.scratch_output_size {
            self.scratch_output_size = output_bytes;
            self.scratch_output = self
                .device
                .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        }
    }
}

/// Computational graph that encodes multiple GEMVs (and fused GPU ops) into one command buffer.
pub struct GpuGraph<'a> {
    ctx: &'a GpuContext,
    ops: Vec<GraphOp>,
}

impl<'a> GpuGraph<'a> {
    pub fn new(ctx: &'a GpuContext) -> Self {
        Self {
            ctx,
            ops: Vec::new(),
        }
    }

    /// Add a GEMV operation to the graph.
    pub fn add_gemv(&mut self, op: GemvOp) {
        self.ops.push(GraphOp::Gemv(op));
    }

    /// Add a GEMV operation from components.
    pub fn add_gemv_raw(
        &mut self,
        weight_buffer: GpuBuffer,
        input_offset: usize,
        output_offset: usize,
        input_len: usize,
        output_len: usize,
    ) {
        self.ops.push(GraphOp::Gemv(GemvOp {
            weight_buffer,
            input_offset,
            output_offset,
            input_len,
            output_len,
        }));
    }

    /// Add a fused SiLU(gate)*up kernel to run entirely on GPU.
    /// gate_offset and up_offset are f32-element offsets into scratch_output.
    /// The result is written back in-place at gate_offset.
    pub fn add_silu_mul(&mut self, op: SiluMulOp) {
        self.ops.push(GraphOp::SiluMul(op));
    }

    /// Clear all operations (reuse the graph for next decode step).
    pub fn clear(&mut self) {
        self.ops.clear();
    }

    /// Execute all GEMVs in ONE command buffer.
    /// This is the key optimization: one commit, one wait for ALL operations.
    pub fn execute(&self) -> Result<()> {
        if self.ops.is_empty() {
            return Ok(());
        }

        let cmd_buf = self.ctx.queue.new_command_buffer();

        for graph_op in &self.ops {
            match graph_op {
                GraphOp::Gemv(op) => {
                    let m = op.weight_buffer.m as u32;
                    let n_blocks = op.weight_buffer.n_blocks;

                    let encoder = cmd_buf.new_compute_command_encoder();

                    encoder.set_compute_pipeline_state(&self.ctx.gemv_pipeline);
                    encoder.set_buffer(0, Some(&op.weight_buffer.buffer), 0);
                    encoder.set_buffer(
                        1,
                        Some(&self.ctx.scratch_input),
                        (op.input_offset * 4) as u64,
                    );
                    encoder.set_buffer(
                        2,
                        Some(&self.ctx.scratch_output),
                        (op.output_offset * 4) as u64,
                    );

                    let n_blocks_buf = self.ctx.device.new_buffer_with_data(
                        &n_blocks as *const u32 as *const _,
                        4,
                        MTLResourceOptions::StorageModeShared,
                    );
                    let m_buf = self.ctx.device.new_buffer_with_data(
                        &m as *const u32 as *const _,
                        4,
                        MTLResourceOptions::StorageModeShared,
                    );

                    encoder.set_buffer(3, Some(&n_blocks_buf), 0);
                    encoder.set_buffer(4, Some(&m_buf), 0);

                    // New shader: one threadgroup per row, 128 threads per threadgroup
                    let tg_count = MTLSize::new(m as u64, 1, 1);
                    let tg_size = MTLSize::new(128, 1, 1);
                    encoder.dispatch_thread_groups(tg_count, tg_size);
                    encoder.end_encoding();
                }
                GraphOp::SiluMul(op) => {
                    let n = op.n as u32;
                    let encoder = cmd_buf.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&self.ctx.silu_pipeline);
                    encoder.set_buffer(
                        0,
                        Some(&self.ctx.scratch_output),
                        (op.gate_offset * 4) as u64,
                    );
                    encoder.set_buffer(
                        1,
                        Some(&self.ctx.scratch_output),
                        (op.up_offset * 4) as u64,
                    );
                    let n_buf = self.ctx.device.new_buffer_with_data(
                        &n as *const u32 as *const _,
                        4,
                        MTLResourceOptions::StorageModeShared,
                    );
                    encoder.set_buffer(2, Some(&n_buf), 0);
                    let threads = MTLSize::new(n as u64, 1, 1);
                    let tg_size = MTLSize::new(
                        self.ctx.silu_pipeline.thread_execution_width().min(n as u64),
                        1,
                        1,
                    );
                    encoder.dispatch_threads(threads, tg_size);
                    encoder.end_encoding();
                }
            }
        }

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        Ok(())
    }

    /// Execute with per-op params buffers.
    /// CRITICAL FIX: Previously shared one params_buf across all ops,
    /// causing later ops to overwrite earlier ops' params before GPU reads them.
    /// Now each op gets its own inline params via setBytes or separate buffer.
    pub fn execute_with_params(&self, _params_buf: &Buffer) -> Result<()> {
        if self.ops.is_empty() {
            return Ok(());
        }

        let cmd_buf = self.ctx.queue.new_command_buffer();

        for graph_op in &self.ops {
            match graph_op {
                GraphOp::Gemv(op) => {
                    let m = op.weight_buffer.m as u32;
                    let n_blocks = op.weight_buffer.n_blocks;

                    // Each op gets its own params buffer — critical for correctness
                    // when multiple GEMVs with different m/n_blocks are in one command buffer
                    let op_params = self.ctx.device.new_buffer_with_data(
                        [n_blocks, m].as_ptr() as *const _,
                        8,
                        MTLResourceOptions::StorageModeShared,
                    );

                    let encoder = cmd_buf.new_compute_command_encoder();

                    encoder.set_compute_pipeline_state(&self.ctx.gemv_pipeline);
                    encoder.set_buffer(0, Some(&op.weight_buffer.buffer), 0);
                    encoder.set_buffer(
                        1,
                        Some(&self.ctx.scratch_input),
                        (op.input_offset * 4) as u64,
                    );
                    encoder.set_buffer(
                        2,
                        Some(&self.ctx.scratch_output),
                        (op.output_offset * 4) as u64,
                    );
                    encoder.set_buffer(3, Some(&op_params), 0);
                    encoder.set_buffer(4, Some(&op_params), 4);

                    let tg_count = MTLSize::new(m as u64, 1, 1);
                    let tg_size = MTLSize::new(128, 1, 1);
                    encoder.dispatch_thread_groups(tg_count, tg_size);
                    encoder.end_encoding();
                }
                GraphOp::SiluMul(op) => {
                    let n = op.n as u32;
                    let n_buf = self.ctx.device.new_buffer_with_data(
                        &n as *const u32 as *const _,
                        4,
                        MTLResourceOptions::StorageModeShared,
                    );

                    let encoder = cmd_buf.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&self.ctx.silu_pipeline);
                    encoder.set_buffer(
                        0,
                        Some(&self.ctx.scratch_output),
                        (op.gate_offset * 4) as u64,
                    );
                    encoder.set_buffer(
                        1,
                        Some(&self.ctx.scratch_output),
                        (op.up_offset * 4) as u64,
                    );
                    encoder.set_buffer(2, Some(&n_buf), 0);
                    let threads = MTLSize::new(n as u64, 1, 1);
                    let tg_size = MTLSize::new(
                        self.ctx.silu_pipeline.thread_execution_width().min(n as u64),
                        1,
                        1,
                    );
                    encoder.dispatch_threads(threads, tg_size);
                    encoder.end_encoding();
                }
            }
        }

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        Ok(())
    }
}

impl GpuContext {
    /// Create a params buffer for reuse across graph executions.
    pub fn create_params_buffer(&self) -> Buffer {
        self.device
            .new_buffer(8, MTLResourceOptions::StorageModeShared)
    }

    /// Convenience method: upload weights and return a GpuBuffer.
    pub fn upload_tensor(&self, tensor: &Q8Tensor) -> GpuBuffer {
        self.upload_q8_weights(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn make_q8_identity(m: usize, n: usize) -> Q8Tensor {
        let bpr = n / 32;
        let mut data = vec![0u8; m * bpr * 34];
        for row in 0..m {
            for b in 0..bpr {
                let off = row * bpr * 34 + b * 34;
                let scale = f16::from_f32(1.0);
                data[off..off + 2].copy_from_slice(&scale.to_le_bytes());
                for i in 0..32 {
                    let idx = b * 32 + i;
                    if idx < n && idx == row {
                        data[off + 2 + i] = 1i8 as u8;
                    }
                }
            }
        }
        Q8Tensor::from_raw(data, n, m)
    }

    #[test]
    fn test_gpu_context_creation() {
        let ctx = GpuContext::new();
        assert!(ctx.is_ok(), "GpuContext creation failed: {:?}", ctx.err());
    }

    #[test]
    fn test_upload_weights() {
        let ctx = GpuContext::new().unwrap();
        let tensor = make_q8_identity(4, 32);
        let gpu_buf = ctx.upload_q8_weights(&tensor);
        assert_eq!(gpu_buf.m, 4);
        assert_eq!(gpu_buf.n, 32);
        assert_eq!(gpu_buf.n_blocks, 1);
    }

    #[test]
    fn test_graph_single_gemv() {
        let ctx = GpuContext::new().unwrap();
        let tensor = make_q8_identity(4, 32);
        let gpu_buf = ctx.upload_q8_weights(&tensor);

        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();

        unsafe {
            let ptr = ctx.scratch_input_ptr();
            std::ptr::copy_nonoverlapping(x.as_ptr(), ptr, 32);
        }

        let mut graph = GpuGraph::new(&ctx);
        graph.add_gemv(GemvOp {
            weight_buffer: gpu_buf,
            input_offset: 0,
            output_offset: 0,
            input_len: 32,
            output_len: 4,
        });

        graph.execute().unwrap();

        let mut y = vec![0f32; 4];
        unsafe {
            let ptr = ctx.scratch_output_ptr();
            std::ptr::copy_nonoverlapping(ptr, y.as_mut_ptr(), 4);
        }

        for i in 0..4 {
            let expected = i as f32;
            let diff = (y[i] - expected).abs();
            assert!(diff < 0.01, "y[{}] = {}, expected {}", i, y[i], expected);
        }
    }
}
