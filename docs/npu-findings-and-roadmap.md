# NPU backend — findings & roadmap (2026‑07‑10 handoff)

*Supersedes the framing in `docs/restructure-decision.md`. That doc posed a 3‑way "fork"
built on a false premise (see §5.1) and on a benchmark regime that turned out to be the wrong
one (see §5.5). This doc replaces the assumptions with measured data from a real 9B run on the
actual Lunar Lake box.*

**Audience:** a second opinion / future session who hasn't seen the working session.

---

## 0. TL;DR

1. **The NPU is a heterogeneous scheduler citizen — a spillover target — not a whole‑model
   runtime.** This is the settled design intent, and the measurements now back it.
2. **When a model fits on the iGPU, put everything on the iGPU — the NPU adds nothing to tok/s.**
   Measured on the 9B: all‑GPU tg **13.88**, NPU‑spillover tg **9.7** (slower). The NPU's *only*
   tok/s win is as a **spillover target that beats the CPU** (~3.4×) for the layers that don't fit
   on the GPU — and **that scenario is still unproven at scale** (the 9B fits, so we never hit it).
3. **The "20 GB OOM" pathology is real but is mostly fixable bugs/artifacts + two multipliers**
   (i8‑doubling for Q5_K, and M‑value duplication), not an intrinsic NPU limit. The biggest bug
   (double‑keep) is **fixed and committed** (`fb371ba`).
4. **Decode is dispatch‑floor‑bound (~350 µs/submit), not bandwidth‑bound.** So decode tok/s is a
   *dispatch‑count* game; the real lever is **fusing more per dispatch (attention core)**, not
   weight width or host micro‑opts.

---

## 1. What this is

`ggml-npu` is a universal NPU backend for ggml/llama.cpp — a `no_std` Rust `cdylib`
(`libggml_npu.so`) that ggml `dlopen`s, plus a C++ **OpenVINO shim** (`libggml_npu_ovc.so`) that
JIT‑compiles NPU programs. Intel Lunar Lake NPU is the first target (OpenVINO + Level Zero). It
recognizes offloadable op‑subgraphs (matmul, SwiGLU FFN, RMSNorm, residual), re‑quantizes each
weight, and compiles the subgraph as **one OpenVINO program with weights baked in as compile‑time
`Constant`s** (fused decompress‑matmul). One compiled program per *(weight‑set, M)*.

## 2. Design intent (load‑bearing — read first)

The backend was **never** meant to run a whole model on the NPU. It is built to **integrate with the
ggml scheduler** and run in **heterogeneous (CPU + iGPU + NPU) + RPC + tensor‑parallel** topologies.
On a laptop, spreading a model across all the compute is the only way to serve a large model well;
NPU‑only is neither the goal nor a good use of a shared‑RAM device. Consequences:

- The NPU only ever holds the **subgraph splits the scheduler assigns it** — never the whole graph.
- Weights should behave like **normal resident ggml tensor data** (in our NPU‑mapped buffer) so
  `--split-mode row` (TP) and RPC drop in without a representation change.
- Any analysis that assumes "whole model on NPU" is solving a non‑goal.

## 3. What was done this session

- **Double‑keep fix — DONE, committed `fb371ba` on `develop`.** Each compiled program was resident
  twice: `compile.rs` `Vec::leak`'d the native blob **and** `session.rs` `pfnCreate`'d a second copy
  into the L0 graph (keyed by the leaked blob's address). The blob is now a transient compile
  artifact: `compile_{const,subgraph}_blob` return a `Vec<u8>` that `session::build_graph` parses
  into the driver‑owned graph and drops; the built graph is cached semantically (`CONST_GRAPHS` by
  `(weight_ptr,M)`, `SG_GRAPHS` by `sg_signature`), and `LISTS` keys on the graph's own address
  (same per‑program discriminator — the gallocr‑safe binding‑set fold is preserved). Numerically
  transparent; ~3.2 GB recovered at full‑model scale.
- **Uncommitted experiment to revert:** a `GGML_NPU_HEAD` env toggle in `ops/matmul.rs` that raises
  the `n ≤ 65536` cap. It's a **no‑op for placement** (see §4/§5.7) — revert unless we build the
  offload path.

## 4. What we measured (real hardware — LNL, Qwythos‑9B Q5_K_M, iGPU via Vulkan)

All AC power unless noted. `llama-bench`, `-p`/`-n` as shown, `tg` = decode t/s.

| Config | placement | pp | tg |
|---|---|---|---|
| Vulkan‑alone `-ngl 99` | all on iGPU | ~290 (needs reps) | **13.88** |
| Vulkan+NPU `-ngl 31` | 2 layers + MTP on NPU, head on CPU | ~99 | **9.7** |
| (prior) `-ngl 31` NPU vs CPU, same 2 layers | — | — | **3.44× NPU>CPU** |

Characterization facts (measured, load‑bearing):

- **Decode device time ≈ affine: ~350 µs fixed floor + ~0.12 µs/FFN‑dim** (R≈1). Floor decomposes as
  ~59 µs amortizable submit/sync + ~74 µs irreducible per‑graph execute latency + compute. Small
  decode matmul ≈ 217 µs/dispatch; FFN ≈ 350–850 µs.
- **Host copy is <0.5% of decode wall.** Decode is device‑bound; the device is latency/compute‑bound.
- **Weight width is speed‑neutral** (i4/i8/f16 within ~5%) — so low‑bit is a *memory* lever, not a
  speed lever.
- **i8‑for‑Q5_K (big correction):** `choose_bits` picks **i4 only for ≤5 bpw** sources. Q5_K_M is
  ~5.5 bpw → **i8**, doubling the baked‑weight size vs every prior u4‑based estimate. One layer's
  fused FFN program = **298 MB** (M=512, i8); full 9B baked ≈ **~17 GB** → full offload impossible
  by a wide margin. **Rule: Q4_K (→i4) models are ~2× cheaper on the NPU than Q5_K (→i8).**
- **The head (vocab projection, 4096×248320 = 1.02 B params, 11% of the model):** bandwidth‑bound,
  not compute‑bound; NPU reaches only ~10 of ~130 GB/s effective, so extrapolated head ≈ 30–60
  ms/token (NPU) vs ~14 ms (GPU) vs ~128 ms (CPU). Baked as i8 it's a **~1 GB single program** — the
  baked‑weight pathology at its worst. **It belongs on the GPU.**
- **Lesson: benchmark on AC.** Eco mode at 29% battery depressed an earlier run (GPU ref 12.77 →
  13.88 on AC). `-r 1` prefill numbers are unreliable (Vulkan shader warmup) — decode is stable.

## 5. Design considerations (the corrected picture)

**5.1 The false dichotomy (why the prior fork was mis‑framed).** The prior doc treated *program
count* and *weight storage (baked vs input)* as one axis — "baked = fast but 226 programs; input =
few programs but 3× slower." They are **orthogonal.** You reduce program count by **coarsening what
you bake per program**, not by moving weights to runtime inputs. The 3× penalty (weight‑as‑input
spills f16, dequant unfused) is the cost of one option you never need to take.

**5.2 Where memory actually goes** (post double‑keep fix): resident ≈ model weights + Σ(programs) ×
(per‑program L0 overhead + baked weights). The multipliers that hurt: **(a)** M‑value duplication
(a full program set per distinct M — prefill M + decode M=1, and far worse under continuous
batching, §5.5); **(b)** i8‑doubling for Q5_K (§4); **(c)** ~14–63 MB fixed L0 overhead per program.
The prior "226 programs = per‑matmul granularity" premise is **stale** — the recognizer already
fuses QKV and the whole FFN half, so program count is driven by **M‑duplication**, not per‑matmul
explosion. That means the old "step 3: coarsen 226→56" has **little headroom left**, and the old
"step 2: free prefill" is the bigger memory lever *in isolation* but is counterproductive for real
workloads (§6).

**5.3 Decode is a dispatch‑count game.** tok/s ≈ 1 / (dispatches‑per‑token × ~350 µs + compute +
CPU‑attention). Nothing about weight storage or bandwidth moves this. The lever is **fewer, larger
dispatches** → **fusing the attention core** (KQ/softmax/KQV), which also removes the CPU islands
that fragment the decode graph. That's the real tok/s + coverage work item (and it's larger than
anything in the prior fork).

**5.4 The NPU's actual value proposition.** Measured: it is **not** a faster‑than‑GPU decoder. Its
value is **(a) perf‑per‑watt + freeing the CPU/iGPU**, and **(b) a spillover target that beats the
CPU (~3.4×) when the model does not fit on the GPU.** For a model that fits on the iGPU, all‑GPU is
strictly best. This is exactly the heterogeneous‑citizen role from §2 — now measured, not assumed.

**5.5 Continuous batching breaks static‑M compilation (the serving gate).** Under continuous
batching, decode runs at **M = number of active sequences**, varying 1→n_parallel as requests
join/finish. The current design compiles one program per `(weight, M)` (static `ov::Shape{M,K}`,
keyed by M — the blob cache confirms `…-m<M>.blob`). So every distinct batch size spawns its own
program set → a **program explosion far worse than prefill+decode ×2.** Tool‑call/agentic workloads
compound this (prefill and decode interleave every turn). **The gating question for serving:** can
the OV NPU plugin compile a **bucketed or dynamic batch dim** with baked u4/u8 weights, bounding
program count? Unanswered — needs a small OV spike.

**5.6 Model‑fits‑on‑GPU ⇒ all‑GPU wins (measured, §4).** The 6.3 GB 9B fits in the UMA iGPU's share,
so NPU spillover is a net loss for it. The NPU's win case needs a **GPU‑oversized** model.

**5.7 The head won't route to the NPU without new plumbing.** Raising the `n` guard made the NPU
*willing* to claim the head, but `GGML_NPU_HEAD=1 ≡ HEAD=0` — identical tg, identical dispatch
count, **no head blob compiled.** The head weight is never placed in the NPU buffer because our
device exposes **`offload_op: None`** and llama.cpp pins the output tensor. Getting it there needs
an `offload_op` impl or an explicit output‑tensor buffer override — low value, since the head
belongs on the GPU. (Side effect: the ~1 GB‑Constant compile feasibility is still untested.)

## 6. Where the prior "fork" landed

| Prior option | Verdict now |
|---|---|
| **Step 1** — kill double‑keep + hygiene | **DONE** (`fb371ba`), ~3.2 GB, transparent. |
| **Step 2** — free prefill programs after prefill | **Deprioritized.** Needs L0 destruction + breaks the append‑only/lock‑free cache invariant; and it's **counterproductive for tool‑call workloads** where prefill/decode interleave (it'd thrash rebuild/destroy every turn). |
| **Step 3** — coarsen per‑matmul → per‑layer | **Little headroom.** Fusion already collapsed per‑matmul to per‑layer; can't merge across the CPU attention islands. |
| **Option A** — weights as runtime input (shared programs) | **Avoid.** The measured 3× decode penalty; unnecessary (coarsen while keeping weights baked instead). |
| *(new)* **Attention‑core fusion** | The real tok/s + coverage lever; removes CPU islands, cuts dispatches, enables further coarsening. Larger work item (RoPE, softmax, runtime‑operand KQ/KQV matmuls). |

## 7. Next items (prioritized)

1. **Prove or kill the value proposition — GPU‑oversized spillover test.** Constrain GPU memory (or
   use a model too big for the iGPU) so it *doesn't* fit, then measure **NPU‑spillover vs
   CPU‑spillover**. This is the one scenario where the NPU earns its place; everything measured so
   far only shows where it *doesn't* help. **Highest value.**
2. **`choose_bits` → i4 for Q5_K (with a quality check).** Halves the baked footprint everywhere;
   the single biggest memory lever for the baked design. Trade‑off: i4 quality over an FFN (~9% vs
   dequant) — measure before shipping; consider i4 for FFN, i8 for quality‑sensitive weights.
3. **M/dynamic‑shape OV spike (serving gate, §5.5).** Can the NPU plugin bound program count with a
   bucketed/dynamic batch dim + baked low‑bit weights? Memory‑light, no model load. Decides whether
   continuous‑batching serving on this backend is viable before any placement work.
4. **Head:** revert the `GGML_NPU_HEAD` toggle (no‑op for placement). Only build `offload_op` /
   output‑tensor override if #1 shows spillover matters *and* the head is a spill target — low
   priority (head belongs on GPU).
5. **Attention‑core fusion** — the decode tok/s + coverage lever (§5.3). Big; sequence after #1–#3
   clarify whether the perf ceiling is worth chasing on this device.

## 8. Constraints a reader should weigh

- **ggml per‑split interface:** we get per‑split subgraphs at `graph_compute`, not the whole model.
  Weights arrive as tensor data in our buffer type; the scheduler mediates cross‑backend copies.
- **`offload_op: None`** in the device iface, and llama.cpp pins the output tensor → we only run ops
  whose weights land in our buffer at load; we can't pull a CPU‑resident op (e.g. the head) onto the
  NPU without new plumbing (§5.7).
- **No VRAM — NPU shares the 15 GB system RAM** (`NPU_DEVICE_TOTAL_MEM_SIZE = 16.3 GB`); the iGPU
  (UMA) shares it too. Now mitigated by **8 GB zram swap** (systemd‑owned, persists across reboot) —
  the OOM‑kill safety net, and it evicts idle anon to hand RAM back. zram won't make an oversized
  working set fast, only survivable.
- **Machine‑crash risk:** a bad NPU submission (unmapped pointer) can hard‑crash the box via the
  `ivpu` MMU. The gallocr address‑reuse crash class is fixed (binding‑set‑keyed `LISTS`); the 9B
  runs this session were crash‑free. Still: treat NPU runs as machine‑risking, run behind a
  MemAvailable kill‑guard, be reboot‑ready.
- **Two languages:** `no_std` Rust core + C++ OV shim. Weight re‑quant is host‑side (`to_float` →
  requant to u4/u8 + scales/zeros).
- **Attention core (KQ/softmax/KQV) not offloaded** → decode is many small NPU islands interleaved
  with CPU/GPU work.

## 9. Repro / environment

- **Model:** `llama.cpp/models/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q5_K_M.gguf` (qwen35 arch, 9.20 B,
  6.25 GiB, 33 layers + 1 MTP block, vocab 248320).
- **Tool:** `build-dl/bin/llama-bench` (built; `llama-cli`/`llama-server` are NOT — `LLAMA_BUILD_TOOLS=OFF`).
- **Backends (dynamic, `GGML_BACKEND_DL=ON`):** Vulkan auto‑loads from `build-dl/bin/libggml-vulkan.so`;
  NPU via `GGML_BACKEND_PATH=…/ggml-npu/target/release/libggml_npu.so`; both ggml builds are 0.15.3
  (ABI‑compatible).
- **Env for an NPU+Vulkan run:**
  `GGML_BACKEND_PATH=<npu.so>` · `GGML_NPU_OVC=<shim.so>` · `GGML_NPU_PROFILE=1` ·
  `LD_LIBRARY_PATH=build-dl/bin:Scratch:/opt/intel/openvino_2026.2.1/runtime/lib/intel64:/usr/lib64`.
- **Placement knobs:** `-ngl N` (layers on GPU; head offloads to GPU only when ngl > n_layer).
  `-ngl 99` = all‑GPU; `-ngl 31` = spill last 2 layers + MTP (+ head on CPU).
- **Safety:** run NPU benches behind a background MemAvailable kill‑guard (kill the bench at
  <~400 MB rather than let the OOM killer pick a victim). Blob cache at `~/.cache/ggml-npu`
  (`…-m<M>-k<K>-n<N>-b<bits>.blob`; new blobs = the NPU compiled that shape — the ground truth for
  "did it actually run on the NPU").
- **Profiling:** `GGML_NPU_PROFILE=1` → per‑dispatch stage‑in/device/copy‑out + owned‑vs‑bounced +
  affine floor, dumped at exit.
- **Git:** crate on branch `develop`; `fb371ba` = double‑keep fix. `ops/matmul.rs` has an uncommitted
  `GGML_NPU_HEAD` toggle to revert.
