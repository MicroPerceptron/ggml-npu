# NPU backend: memory/scaling restructure — decision handoff

**Status:** decision needed. **Audience:** a second opinion who hasn't seen the working session.
**Question in one line:** to run whole models on the NPU without OOM, do we accept ~3× slower
decode (shared programs, weight-as-input) or keep fast-but-non-scaling baked weights?

---

## 1. What this is

`ggml-npu` is a universal NPU backend for **ggml/llama.cpp** — a `no_std` Rust `cdylib`
(`libggml_npu.so`) that ggml `dlopen`s, plus a C++ **OpenVINO shim** that JIT-compiles NPU programs.
Intel Lunar Lake NPU is the first target (via OpenVINO + Level Zero). Goal: offload matmul/FFN
subgraphs from a GGUF model to the NPU (perf-per-watt, free the CPU/iGPU), with CPU fallback. It must
also eventually serve **tensor-parallel** (`--split-mode row`) and **RPC** topologies, which both
want the NPU to behave like a normal ggml device (weights = resident tensor data).

**How dispatch works today:** ggml's scheduler hands us *subgraph splits*. We recognize an offloadable
run of ops (matmul, SwiGLU FFN, RMSNorm, residual), re-quantize each weight, and compile the whole
subgraph as **one OpenVINO program with the weights baked in as compile-time `Constant`s** (the fused
decompress-matmul). One compiled program (blob) per *(weight-set, M)*.

---

## 2. The problem

Full-model offload of **Qwen3-1.7B (Q4_K, 28 layers)** exhausts memory and thrashes the machine to
death. The NPU has **no dedicated VRAM — it shares the 15 GB system RAM** (measured
`NPU_DEVICE_TOTAL_MEM_SIZE = 16.3 GB`). Prefill works and is fast (**pp = 164 tok/s**, ~19× the
sandbox-throttled CPU); **decode OOMs** during compilation.

**Where ~23 GB went for a 1.2 GB model** (measured: RAM + 8 GB swap both full):

| Consumer | ~Size | Why |
|---|---|---|
| Model weights (our NPU buffer) | 1.2 GB | the only "real" cost |
| **226 compiled programs × per-program overhead** | **5–9 GB** | each OV compiled model carries ~14–63 MB of runtime structures *independent of weight size* (measured via RSS slope) × 226 blobs |
| Every blob kept resident **twice** | ~6 GB | we `Vec::leak` the raw blob bytes (3.2 GB) **and** `pfnCreate` loads each into an L0 graph (2nd copy) |
| Prefill (M=32) + decode (M=1) blob sets, both resident | ×~2 | distinct compiled programs |
| OV compile working set | GBs, transient | spikes per cold compile |

**The dominant term is the *number of compiled programs*, not weight bytes.** That's why switching
weights i4-symmetric → u4 (which shrank each blob 24 MB → 8 MB and is correct at ~5.8% vs Q4_K)
**barely moved the needle** — it cut bytes, not program count.

**It's a design pathology, not an NPU limit.** Our design is **O(layers × M-values)** compiled
programs, each with fixed overhead, each double-kept. A 7B (~90 layers) → ~900 blobs → hopeless.
For contrast, OpenVINO GenAI runs 7B INT4 models on this same NPU in a few GB — by compiling the
model as **one program (baked weights, paid once)**, not 226.

---

## 3. What we've measured (the load-bearing facts)

All on the real Lunar Lake NPU, OpenVINO 2026.2.1, decode shape M=1, group-128 asymmetric u4
(`(u4_weight − u4_zero) × f16_scale` — the canonical NNCF/OV shape the NPU compiler keeps compressed).

**Baked `Constant` vs weight-as-`Parameter` (input), same canonical representation:**

| shape (M=1) | BAKED Constant | INPUT (u4) | input penalty |
|---|---|---|---|
| FFN-gate (K2048×N6144) | 4.1 MB blob, **361 µs** | 0.6 MB blob, **1110 µs** | **3.1×** |
| QKV/O (K2048×N2048) | 1.5 MB blob, **140 µs** | 0.3 MB blob, **385 µs** | **2.75×** |

- **Baked** = weight fused into the program → fast, but one 4 MB blob *per weight* → 226 blobs.
- **Input** = weight is data the program reads → program is tiny (0.6 MB), shareable across layers,
  weights stored once — **but ~3× slower**. The NPU compiler **only fuses the dequant for a baked
  `Constant`**; as a runtime input it materializes the f16 weight and spills to RAM (effective weight
  bandwidth drops ~2.5×). The canonical representation does **not** rescue this — tested directly.
- Decode at M=1 is also **dispatch/latency-bound** (~350 µs fixed floor per submit; even baked FFN-gate
  is mostly floor), and only hits ~10 of ~130 GB/s effective — so fusing more ops per program helps the
  floor but not the per-matmul spill.

**Prefill (baked u4) is unaffected and fast** — glue isn't claimed at M>1, so it's plain matmuls.

---

## 4. The fork (pick one)

| | A. Shared programs (weight-input) | B. Baked + hygiene | C. Whole-layer baked |
|---|---|---|---|
| Program count | **~13 (per shape)** | O(layers), fewer than today | ~2/layer |
| Weights | resident once, as data | baked (once, spread across blobs) | baked |
| Decode speed | **~3× slower** (spill) | fast (fused) | fast (fused) |
| Fits 1.7B | yes | yes (w/ decode-only + no double-keep) | yes |
| **Scales to 7B** | **yes** (program count O(1)) | no (~900 blobs) | tight (~180 blobs) |
| Effort | large restructure | small–medium | medium (recognizer spans more) |
| TP / RPC fit | natural (weights = data) | awkward (weights in programs) | awkward |

- **A** is what the team's own analysis targeted ("collapse 226 → ~13 shape-shared programs, store
  weights once"). It's the only option whose resident footprint is ≈ *model size + small fixed
  overhead* regardless of model size. Cost: the measured ~3× decode slowdown.
- **B** keeps today's fast path and just stops the bleeding (decode-only offload, larger fusion, free
  the leaked `Vec`). Fastest to ship, fits ~2B, but the O(layers) overhead wall returns for big models.
- **C** compiles each offloadable layer-half as one baked program. Middle ground; still O(layers).

**What we can't easily do:** OV-GenAI-style **whole-model baked** (fast + fits + scales) needs to
compile the *whole model* as one program. We only get *subgraph splits* from ggml's scheduler, and
**attention core (KQ/softmax/KQV) isn't offloadable yet** (stays on CPU), so the graph is inherently
fragmented into per-layer offload islands. Reaching one-program-per-model would mean intercepting at a
different altitude than ggml's per-op backend interface — a much larger departure.

---

## 5. Constraints a second opinion should weigh

1. **ggml-backend interface**: we receive per-split subgraphs at `graph_compute`, not the whole model.
   Weights arrive as tensor data in our buffer type; the scheduler mediates cross-backend copies.
2. **Attention not yet offloaded** → the decode graph is many small NPU islands interleaved with CPU.
3. **Shared RAM, 15 GB**, no VRAM. The NPU's own MMU (not VT-d, which is identity-mapped for it) maps
   host memory; a bad submission in HW-scheduler mode hard-crashes the box (we run OS-scheduler + TDR
   during dev to make faults recoverable).
4. **Two languages**: `no_std` Rust core + C++ OpenVINO shim. Weight re-quant is host-side (dequant
   the GGUF quant via ggml `to_float`, re-quant to u4/zp/scale).
5. The **canonical u4 representation is settled and correct** (5.8% vs Q4_K on a matmul; requant
   dequant correlates 0.995 with the true weight). The crash bugs (gallocr-aliased caches) are fixed
   and committed. Allocator foundation for "weights resident as `Vec<u8, NpuHostAlloc>`" is in place.

---

## 6. Open questions for the second opinion

1. **Is ~3× slower decode an acceptable price for O(1) program scaling?** (A) is the only path to 7B on
   this box, but 3× is steep. Is a working-but-3×-slower NPU decode still worth it over CPU/iGPU on a
   perf-per-watt basis?
2. **Is the input-weight spill actually unavoidable?** Is there an OV/NPU expression, compile flag, or
   plugin config that gets a *runtime* low-bit weight fused (dequant-in-kernel, no f16 spill) the way
   GPU kernels (Marlin/exllama) do? We've only proven it fused for `Constant`s.
3. **Can we reach OV-GenAI-style whole-model (or whole-layer-batch) baked compilation** without
   abandoning the ggml-backend integration — e.g., recognize and bake a whole transformer layer
   (including a CPU-side attention shim) as one program? What breaks?
4. **Hybrid?** e.g., baked for the big/hot weights (FFN) where fusion matters most, input-shared for the
   rest — or bake per-*shape* with a weights-swap mechanism if OV exposes one.
5. Does the **±3× hold at higher M** (prefill), or is it M=1-specific? (We only measured M=1.)

---

*Appendix — quantified elsewhere in the repo history:* per-model RSS slope (~14 MB weightless / ~63 MB
baked), the 350 µs dispatch floor, the gallocr cache-aliasing crash class, and the OV-GenAI CPU run
(0.5B int4, coherent, 51 tok/s; NPU pipeline blocked by an unrelated `NPU_MAX_TILES` driver bug).
