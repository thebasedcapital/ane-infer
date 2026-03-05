# ane-infer — Handover

> Last updated: 2026-03-04 | 32 commits | https://github.com/thebasedcapital/ane-infer

## Current State: Working, 32 tok/s Q8, 42 tok/s Q4 (buggy)

The engine runs Qwen3.5-2B with correct output on Q8. GPU decode matches llama.cpp. Q4 is faster but Q6K dequant is broken.

## What Works

| Feature | Status | Speed |
|---------|--------|-------|
| GPU Q8 decode (Metal, all 24 layers) | Correct output | 32 tok/s |
| GPU Q4 decode (Metal) | Speed works, output wrong | 42 tok/s |
| CPU Q8 decode (rayon NEON) | Correct output | 23 tok/s |
| ANE prefill (fused FFN mega-kernel) | Correct output | 33 tok/s |
| ANE direct eval | Working | 106 μs/dispatch |
| ANE multi-procedure models | Working | 103 μs/proc |
| ANE chaining prepare | prepareChainingWithModel succeeds | — |
| Flash SDPA (single-pass) | Working | +10% decode |
| GGUF Q8/Q4/Q6K parsing | Q8+Q4 work, Q6K partially broken | — |
| GPT-2 BPE tokenizer | Working | — |
| Zero per-token Metal allocs | Working | — |

## What's Broken

### 1. Q6K Dequantization (BLOCKS Q4)
**File:** `crates/gguf/src/dequant.rs` (Q6K match arm)

The Q4_0 model uses Q6K for the token embedding. Our Q6K dequant produces partially correct output (first token OK, subsequent tokens degrade into Japanese/Korean gibberish). This corrupts the embedding lookup table, making all Q4 inference garbage.

**Fix:** Compare our Q6K implementation line-by-line with llama.cpp's `dequantize_row_q6_K` in `ggml-quants.c`. The Q6K format has complex bit packing: scales in 4-bit groups, high bits separate from low bits, `dmin` offset. Our implementation likely misses the `dmin` subtraction or has wrong bit extraction order.

**Test:** `./target/release/ane-infer bench -m ~/models/Qwen3.5-2B-Q4_0.gguf --prompt-tokens 16 --gen-tokens 8` → check "GPU output" line for English text.

### 2. CPU Q4 GEMV Missing
**File:** `crates/engine/src/q8_gemv.rs`

`compute_row` only handles Q8_0 blocks (34 bytes). Q4_0 blocks (18 bytes) need a separate code path with nibble unpacking. Currently CPU decode on Q4 models reads Q8 format from Q4 data → garbage.

**Fix:** Add Q4_0 branch in `compute_row` matching the Metal Q4 shader logic: low nibbles = elements 0-15, high nibbles = elements 16-31.

### 3. ANE Chaining — buffersReady Blocked
**File:** `crates/ane-bridge/objc/chaining_e2e.m`

`prepareChainingWithModel:` succeeds but `buffersReadyWithModel:` returns NO silently. Tested with: _ANEModel, _ANEInMemoryModel, daemon path, direct path, multi-procedure models, different buffer configs. All fail.

**Likely cause:** The IOSurface mapping step (`mapIOSurfacesWithRequest:cacheInference:error:` on _ANEInMemoryModel) returns error 13. Chaining state machine may require successful IOSurface mapping before buffersReady. Or `_intermediateBufferHandle` needs to be set.

### 4. GPU FullAttention Output Not Verified
The 6 FullAttention layers run on GPU with RoPE, SDPA, KV cache, sigmoid gating. But correctness hasn't been verified token-by-token against CPU reference. The Q8 GPU output is coherent English but may have subtle numerical differences.

### 5. Speculative Decoding Dead End
Self-speculative (FFN-skip draft) doesn't help — draft is only ~2x faster than verify, total is slower than GPU alone. Needs a genuinely tiny draft model (10-100x faster). Options:
- Train a 4-layer DeltaNet drafter from the same embedding
- Use n-gram cache for repetitive text
- Medusa-style parallel heads (but vocab=248K makes heads too large)

## What to Build Next (Priority Order)

### P0: Fix Q6K → Unlock Q4 (42 tok/s correct)
1 day. Compare with llama.cpp `dequantize_row_q6_K`. The Q4 model is already quantized and the GPU shader works — just need correct embedding.

### P1: CPU Q4 NEON GEMV
Half day. Port the Metal Q4 nibble unpacking to NEON intrinsics in `compute_row`. Enables CPU Q4 decode + correct prefill for Q4 models.

### P2: Chunked Parallel DeltaNet Prefill
1-2 weeks. Implement the FLA (Flash Linear Attention) chunked algorithm:
- Split sequence into chunks of 64-128 tokens
- Parallel scan within chunks (Metal GPU or ANE)
- Sequential state update across chunks
- Expected: 33 → 150+ tok/s prefill

Reference: [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention), [DeltaNet Explained Part II](https://sustcsonglin.github.io/blog/2024/deltanet-2/)

### P3: Tiny Draft Model for Speculative Decoding
1-2 weeks. Train a 4-layer DeltaNet model (same embedding, pruned layers) as a fast draft model. Run on ANE (3W, ~100 tok/s for tiny model), verify on GPU (42 tok/s). Expected: 2.5x multiplier → **100+ effective tok/s**.

Reference: [Apple ReDrafter](https://machinelearning.apple.com/research/recurrent-drafter)

### P4: Q3/Q2 Quantization
1 week. Further bandwidth reduction. Q3 = 66% of Q4 bandwidth → ~55 tok/s. Q2 = 50% → ~65 tok/s. Diminishing quality but may be acceptable for draft model.

### P5: Wire GPU Decode into Generate Command
Half day. The `generate` command only uses CPU decode. Add `--gpu` flag to use `encode_full_token_gpu`. Requires GPU-side prefill or CPU prefill → GPU decode handoff with state sync.

## File Map

```
Key files to understand the codebase:

crates/cli/src/main.rs              — Entry point. Commands: generate, bench, test-ane, info.
                                      Benchmark wires CPU prefill → GPU decode.
                                      ~1800 lines, getting large.

crates/engine/src/gpu_full_decode.rs — THE hot path. Full-GPU DeltaNet+FullAttn forward.
                                      One command buffer per token, zero allocs.
                                      ~1100 lines.

crates/engine/src/ane_prefill.rs    — ANE batched prefill with fused mega-kernels.
                                      AneProjection, FusedFfnKernel, compile_ane_prefill.

crates/engine/metal/q8_gemv.metal   — Optimized Q8 GEMV shader (NR0=2, NQ=8, simd_sum).
crates/engine/metal/q4_gemv.metal   — Q4 GEMV (tiled + simple variants).
crates/engine/metal/deltanet.metal  — 9 DeltaNet recurrence shaders.
crates/engine/metal/attention.metal — RoPE, Flash SDPA, sigmoid gate, Q/gate split.

crates/engine/src/metal_graph.rs    — GpuContext with ALL pipeline states loaded.
crates/engine/src/gpu_decode.rs     — Weight upload, GpuBuffer, GpuModelWeights.
crates/engine/src/q8_gemv.rs        — CPU Q8 GEMV (rayon parallel, NEON).
                                      Also has Q4Tensor type but no CPU Q4 kernel.

crates/gguf/src/dequant.rs          — Q4/Q8/Q6K dequantization. Q6K IS THE BUG.
crates/gguf/src/parser.rs           — GGUF v2/v3 parser.
crates/mil-gen/src/mega.rs          — Fused FFN, dual/triple projection MIL gen.

crates/ane-bridge/objc/ane_runtime.m — ANE private API FFI (compile, eval, multi-proc).
docs/ane-internals.md               — All ANE discoveries beyond maderix.
```

## Build & Test

```bash
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

# Test correctness (Q8)
./target/release/ane-infer generate -m ~/models/Qwen3.5-2B-Q8_0.gguf -p "1+1=" --max-tokens 5 --temp 0
# Expected: "1+1=2"

# Benchmark
./target/release/ane-infer bench -m ~/models/Qwen3.5-2B-Q8_0.gguf --prompt-tokens 16 --gen-tokens 8
# Expected: GPU Full ~32 tok/s, CPU ~23 tok/s, ANE pp16 ~33 tok/s

# Test ANE hardware
./target/release/ane-infer test-ane
# Expected: "ANE is working correctly!"

# Q4 (broken output, fast speed)
./target/release/ane-infer bench -m ~/models/Qwen3.5-2B-Q4_0.gguf --prompt-tokens 16 --gen-tokens 8
# Expected: GPU Full ~42 tok/s but GPU output will be gibberish until Q6K is fixed
```

## Models

```
~/models/Qwen3.5-2B-Q8_0.gguf    — 1.9 GB, correct output
~/models/Qwen3.5-2B-Q4_0.gguf    — 1.1 GB, Q6K embedding bug (requantized from Q8)
```

## Key Lessons Learned

1. **Metal buffer params bug** was the #1 GPU blocker — shared params buffer across ops in one command buffer meant later ops overwrote earlier ops' params. Took from 0.1 to 3.5 tok/s just by fixing this.

2. **ANE chaining error 15** was NOT firmware — it was the wrong `_ANEIOSurfaceOutputSets` factory method. The correct one requires a stats IOSurface argument.

3. **Speculative decoding math** doesn't work with same-model. Draft must be 10-100x faster than verify. Self-speculation (layer skipping) only gives ~2x draft speedup, not enough.

4. **Q4 nibble order** in GGUF: elements 0-15 = low nibbles of bytes 0-15, elements 16-31 = high nibbles of bytes 0-15. NOT interleaved per byte.

5. **NR0=2 tiled GEMV** beats 1-thread-per-row on Apple Silicon (38 vs 34 tok/s). Shared input loading via simdgroups amortizes bandwidth.

6. **Flash SDPA** (single-pass online softmax) gives free 10% — eliminates 2 redundant QK score passes.

7. **CoreML MLProgram** uses Espresso C++ runtime internally, completely separate from private ANE framework path. No `_ANEModel` extractable from CoreML.
