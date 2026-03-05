# ane-infer

**Hybrid ANE+Metal+CPU inference engine for LLMs on Apple Silicon.**

First implementation of Qwen3.5 (Gated DeltaNet + GQA) running natively on Apple Neural Engine via reverse-engineered private APIs. 32 tok/s Metal GPU decode matching llama.cpp, 3.6 TFLOPS fused ANE mega-kernels, built from scratch in Rust + Obj-C + Metal.

Built on the shoulders of [maderix/ANE](https://github.com/maderix/ANE) — the project that cracked open ANE training. We took it further into inference with DeltaNet, Metal GPU shaders, and a complete decode pipeline.

---

## What This Is

A from-scratch LLM inference engine that runs Qwen3.5-2B on three Apple Silicon accelerators simultaneously:

- **Apple Neural Engine (ANE)** — batched prefill via 1x1 convolutions through private `_ANEClient` APIs
- **Metal GPU** — single-token decode with 13 custom compute shaders, ONE command buffer per token
- **CPU (NEON/AMX)** — parallel Q8_0 GEMV via rayon, Accelerate BLAS fallback

No CoreML. No Python. No MLX. Just system frameworks + `objc_msgSend`.

## What This Is Not

- Not faster than llama.cpp (yet). We match their decode speed, not their prefill.
- Not production-ready. Private API usage means it breaks with macOS updates.
- Not a general inference framework. Built specifically for Qwen3.5 DeltaNet hybrid architecture.

---

## Performance

**Qwen3.5-2B Q8_0 on Apple M5 (same chip as llama.cpp benchmarks)**

| Backend | Speed | Power | Notes |
|---------|-------|-------|-------|
| Metal GPU Q8 decode | **32 tok/s** | ~15W | Matches llama.cpp (34.8) |
| Metal GPU Q4 decode | **42 tok/s** | ~15W | Q6K dequant WIP |
| CPU Q8 decode | 23 tok/s | ~5W | Rayon + NEON |
| ANE prefill pp16 | 33 tok/s | ~3W | Fused FFN mega-kernel |
| ANE fused FFN | **3.6 TFLOPS** | ~3W | 3x single-op throughput |

## ANE Reverse Engineering

We went deeper than anyone into Apple's private Neural Engine framework. Key discoveries:

### What We Cracked

| Discovery | Impact |
|-----------|--------|
| `doEvaluateDirectWithModel:` | Bypasses ANE daemon, 10% faster eval |
| Multi-procedure MIL models | N functions in one compiled program, dispatch by `procedureIndex` |
| `prepareChainingWithModel:` **succeeds** | First public success — error 15 was wrong `_ANEIOSurfaceOutputSets` API |
| `_ANEIOSurfaceOutputSets.objectWithstatsSurRef:outputBuffer:` | The correct factory method (not `outputSetsWithBuffers:`) |
| CoreML MLProgram → `MLProgramEngine` → `MLNeuralNetworkEngine` | Confirmed ANE enabled (`isANEPathForbidden=NO`, `modelIsMIL=YES`) |
| Espresso C++ runtime path | CoreML uses Espresso internally, no `_ANEModel` exposed |
| H11ANE IOKit user client type=1,4 | Direct kernel driver access via `IOServiceOpen` |
| `_ANEDaemonConnection` XPC surface | 19 methods including chaining, RT, telemetry |

### ANE Chaining — The Breakthrough

After 7 probe iterations across two sessions, we discovered that `ANEProgramChainingPrepare()` error 15 was **not a firmware limitation** — it was caused by using the wrong `_ANEIOSurfaceOutputSets` factory method.

```
Before: outputSetsWithBuffers:@[buf_out]  → error 15
After:  objectWithstatsSurRef:ioStats outputBuffer:@[buf_out]  → SUCCESS
```

Both `prepareChainingWithModel:` (daemon) and `doPrepareChainingWithModel:` (direct) succeed. `buffersReady` remains blocked — the next frontier.

### Fused Mega-Kernels

Instead of dispatching one ANE kernel per linear projection (1.1 TFLOPS per op), we fuse multiple operations into single MIL programs:

- **Fused FFN**: gate_proj conv → sigmoid → mul → up_proj conv → mul → down_proj conv = **8 ops, ONE dispatch, 3.6 TFLOPS**
- **Fused QKV**: 3 parallel convolutions from same input = 1 dispatch
- **Fused dual projection**: gate + ssm_out in one program

The ANE compiler handles weight blobs >32MB SRAM automatically via DRAM spilling — no manual tiling needed.

---

## Metal GPU Decode

13 custom Metal compute shaders encode the entire DeltaNet + FullAttention forward pass into **one command buffer per token**:

| Shader | Purpose |
|--------|---------|
| `q8_gemv` | Q8_0 GEMV (NR0=2, NQ=8, 4 simdgroups, simd_sum) |
| `q4_gemv` | Q4_0 GEMV (same pattern, nibble unpacking) |
| `deltanet_recurrence` | Full per-head state update (decay/recall/delta/update/query) |
| `conv1d_silu` | Shift + apply + SiLU activation |
| `compute_beta_decay` | sigmoid(beta) + exp(a*softplus(alpha+bias)) |
| `sdpa_causal` | Flash Attention decode (single-pass online softmax) |
| `rope_apply` | Rotary position embeddings |
| `rmsnorm_simple` | 128-thread reduction RMSNorm |
| `rmsnorm_gated` | Per-head RMSNorm with SiLU gate |
| `sigmoid_gate` | Output gating |
| `q_gate_split` | Deinterleave packed Q+gate projection |
| `residual_add` | Element-wise residual connection |
| `silu_mul` | Fused SiLU(gate) * up |

**Zero per-token Metal buffer allocations.** All params pre-allocated at model load.

### The GPU Performance Journey

| Optimization | Speed | Gain |
|---|---|---|
| Starting point (params buffer corruption) | 0.1 tok/s | — |
| Fix shared params buffer | 3.5 tok/s | 35x |
| Single command buffer per token | 5.0 tok/s | 1.4x |
| llama.cpp-style Q8 GEMV shader | 32.6 tok/s | 6.5x |
| NR0=2 threadgroup dispatch fix | 34.7 tok/s | 1.06x |
| FullAttention layers on GPU | 30.0 tok/s | (added 6 layers) |
| Flash SDPA (single-pass softmax) | 42.3 tok/s | +10% |
| **Total improvement** | **0.1 → 42 tok/s** | **420x** |

---

## Architecture

```
                    ┌─────────────┐
                    │  GGUF Model │
                    │  (Q8/Q4_0)  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         ┌────▼────┐  ┌───▼───┐  ┌────▼────┐
         │   ANE   │  │  CPU  │  │  Metal  │
         │ Prefill │  │ NEON  │  │   GPU   │
         │ 33 tk/s │  │ 23t/s │  │  32t/s  │
         └─────────┘  └───────┘  └─────────┘
              │            │            │
              │     ┌──────┴──────┐     │
              │     │ DeltaNet    │     │
              │     │ Recurrence  │     │
              │     │ (sequential)│     │
              │     └─────────────┘     │
              │                         │
              └────────┬────────────────┘
                       │
                  ┌────▼────┐
                  │ Tokenizer│
                  │ (BPE)    │
                  └──────────┘
```

### Qwen3.5-2B Hybrid Architecture
- **24 layers**: 18 DeltaNet (linear attention + SSM recurrence) + 6 Full Attention (GQA)
- **DeltaNet**: O(1) per token, 128-dim recurrent state, conv1d with kernel=4
- **Full Attention**: 8 Q heads, 2 KV heads, head_dim=256, partial RoPE
- **FFN**: SwiGLU, dim=2048 → hidden=6144

---

## Building

```bash
# Prerequisites: Rust, Xcode Command Line Tools
git clone https://github.com/youruser/ane-infer
cd ane-infer

# Compile Metal shaders
cd crates/engine/metal
xcrun -sdk macosx metal -c q8_gemv.metal -o q8_gemv.air
xcrun -sdk macosx metal -c deltanet.metal -o deltanet.air
xcrun -sdk macosx metal -c attention.metal -o attention.air
xcrun -sdk macosx metal -c q4_gemv.metal -o q4_gemv.air
xcrun -sdk macosx metallib q8_gemv.air deltanet.air attention.air q4_gemv.air -o q8_gemv.metallib
cd ../../..

# Build
cargo build --release

# Download model (Q8_0)
# Place at ~/models/Qwen3.5-2B-Q8_0.gguf
```

## Usage

```bash
# Generate text
ane-infer generate -m model.gguf -p "The capital of France is" --max-tokens 256 --temp 0.7

# Full benchmark suite
ane-infer bench -m model.gguf --prompt-tokens 128 --gen-tokens 32

# Test ANE hardware
ane-infer test-ane

# ANE throughput benchmark
ane-infer bench-ane

# Model info
ane-infer info -m model.gguf
```

---

## File Structure

```
crates/
├── ane-bridge/           # ANE private framework FFI
│   ├── objc/
│   │   ├── ane_runtime.m        # _ANEClient, compile/eval/free lifecycle
│   │   ├── ane_runtime.h        # C ABI for Rust FFI
│   │   ├── coreml_probe.m       # CoreML MLProgram reverse engineering
│   │   ├── chaining_e2e.m       # ANE chaining end-to-end test
│   │   ├── iokit_probe.m        # IOKit H11ANE direct access
│   │   └── test_fused_ffn.m     # Fused FFN mega-kernel test
│   └── src/lib.rs               # Safe Rust wrappers (AneKernel, weight blobs)
├── mil-gen/              # MIL program text generation
│   └── src/
│       ├── lib.rs               # MIL header/footer, conv op helper
│       ├── mega.rs              # Fused FFN, dual/triple projections
│       ├── attention.rs         # QKV, output projection
│       └── ffn.rs               # FFN up/down projections
├── engine/               # Core inference engine
│   ├── metal/
│   │   ├── q8_gemv.metal        # Q8_0 GEMV + SiLU (optimized)
│   │   ├── q4_gemv.metal        # Q4_0 GEMV (tiled + simple)
│   │   ├── deltanet.metal       # DeltaNet recurrence shaders (9 kernels)
│   │   └── attention.metal      # RoPE, SDPA, gating (4 kernels)
│   └── src/
│       ├── metal_graph.rs       # GpuContext, GpuGraph, all pipeline states
│       ├── gpu_full_decode.rs   # Full-GPU token decode (ONE cmd buffer)
│       ├── gpu_decode.rs        # GPU weight upload, GpuBuffer types
│       ├── ane_prefill.rs       # ANE batched prefill with mega-kernels
│       ├── deltanet.rs          # CPU DeltaNet recurrence (NEON)
│       ├── q8_gemv.rs           # CPU Q8/Q4 GEMV (rayon parallel)
│       ├── model.rs             # Model weight types, config
│       ├── tokenizer.rs         # GPT-2 BPE tokenizer
│       └── scratch.rs           # Pre-allocated scratch buffers
├── gguf/                 # GGUF file parser
│   └── src/
│       ├── parser.rs            # GGUF v2/v3 parsing
│       ├── to_ane.rs            # Tensor extraction helpers
│       └── dequant.rs           # Q4/Q8/Q6K dequantization
└── cli/                  # CLI binary
    └── src/main.rs              # Commands: generate, bench, test-ane, info
```

---

## Limitations

- **Private APIs**: Uses `_ANEClient`, `_ANEInMemoryModel`, etc. Will break on macOS updates.
- **Q6K dequant**: Partially broken — Q4 models with Q6K embeddings produce degraded output.
- **No speculative decoding**: Same-model speculation doesn't help (draft ~= verify speed). Needs separate tiny draft model.
- **Sequential recurrence**: DeltaNet state update is O(L) per token for prefill. Chunked parallel algorithm (FLA) not yet implemented.
- **FullAttention prefill**: Not yet batched on ANE — only DeltaNet layers use ANE prefill.
- **Single sequence**: No batched inference (batch_size=1 only).

## Acknowledgments

- [maderix/ANE](https://github.com/maderix/ANE) — The breakthrough project that reverse-engineered ANE training. We built on their `_ANEInMemoryModelDescriptor`, weight blob format, and MIL compilation pipeline.
- [hollance/neural-engine](https://github.com/hollance/neural-engine) — Comprehensive ANE documentation.
- [eiln/ane](https://github.com/eiln/ane) — Linux ANE driver reverse engineering.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Metal Q8 GEMV shader patterns, GGUF format, performance reference.
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) — Chunked parallel DeltaNet algorithm reference.
- [metalQwen3](https://github.com/BoltzmannEntropy/metalQwen3) — Metal GPU inference reference for Qwen.

## Disclaimer

This project uses Apple's **private, undocumented frameworks** (`AppleNeuralEngine.framework`). These APIs have no stability guarantee and may change or break with any macOS update. Use at your own risk. Not affiliated with Apple.

## License

MIT
