# ggml-npu

A universal NPU backend for [ggml](https://github.com/ggml-org/ggml) / llama.cpp,
targeting Intel (Level Zero), AMD (XRT/XDNA), and Qualcomm (Hexagon/QNN) client NPUs
behind one runtime seam, with CPU fallback for everything it does not claim.

**Status: Intel graph-compiler prototype.** The backend registers as a dynamically-loaded ggml
`NPU` device and currently lowers dense FFN, LFM ShortConv/state writes, RoPE, runtime attention,
FlashAttention, and sparse routed-expert graphs on Intel Lunar Lake. It
re-quantizes model weights, compiles OpenVINO models with low-bit weights baked as Constants,
loads the native artifacts through the Level Zero graph extension, and dispatches scheduler
splits from Rust. Unsupported operations remain on CPU/GPU.

The next architecture target is Liquid AI's LFM family, especially LFM2.5-8B-A1B. That requires
stateful short-convolution, sparse routed-expert, and attention graph coverage; see
[`docs/npu-findings-and-roadmap.md`](docs/npu-findings-and-roadmap.md) for measured constraints.

## Design conventions

- **`no_std` + `alloc`.** No dependency on Rust std; the global allocator forwards to the
  host libc (`malloc`/`posix_memalign`), resolved at `dlopen`. Panics abort via a
  `#[panic_handler]` (no unwinding across the C ABI).
- **`allocator_api` (nightly).** Collections are allocator-parameterized (`A = Global` today)
  so vendors can back device buffers with their own allocators.
- **Atomics over global locks.** The singleton vtable graph is published with a lock-free
  `AtomicPtr` CAS, not a `OnceLock`.
- **Partitioned modules**, organized by association; vendors live under `vendor/<driver>/`.
- **Portable ownership first.** `runtime/` defines device/program/memory/queue/event lifecycles;
  `os/` supplies owned host primitives. Vendor command protocols stay below both interfaces.

## Layout

```
src/lib.rs              no_std crate root: global allocator, panic handler, mod ffi (bindgen), DL entry points
src/backend/mod.rs      singleton vtable graph (AtomicPtr CAS wiring)
src/backend/shell.rs    registry + device + backend(stream) vtables
src/backend/buffer.rs   buffer-type + buffer vtables (host allocation)
src/engine.rs           scheduler split recognition and graph dispatch
src/ops/                matmul, graph-pattern IR, and host requantization
src/hal.rs              legacy five-verb compatibility façade
src/runtime/            portable fallible runtime lifecycle + artifact envelope
src/host/               host contracts, mock provider, and rustix Linux implementation
src/vendor/stub.rs      CPU-echo NpuHal (no hardware)
src/vendor/ivpu/        Intel NPU (Level Zero + OpenVINO); level_zero/ vendored headers   <- active
src/vendor/ivpu/direct/ feature-gated Linux intel_vpu discovery/resource UAPI (no submit)
src/vendor/xdna/        AMD Ryzen AI (XRT + mlir-aie)                                       skeleton
src/vendor/hexagon/     Qualcomm Hexagon (FastRPC/QNN + hexagon-mlir)                       skeleton
tests/registration_check.c   self-contained registration verifier (mimics ggml's DL loader)
build.rs                bindgen driver (locates ggml headers + clang resource dir)
```

## Prerequisites

- **Rust nightly** (pinned by `rust-toolchain.toml`) — required for `allocator_api`.
- **clang** with its resource headers (`build.rs` uses `clang -print-resource-dir`).
- A **llama.cpp checkout** for the ggml headers: sibling `../llama.cpp` by default, or set
  `GGML_DIR=/path/to/llama.cpp`. Stock `ggml-org/llama.cpp` builds (CI) feature-detect missing
  `GGML_BACKEND_DEVICE_TYPE_NPU` / external-memory ABI and keep device typing honest (`ACCEL`
  default; no invented NPU enum). Production hosts that need native NPU scheduling should use
  **MicroPerceptron/llama.cpp** (see issue #13).
- Cross-ISA: `rustup target add aarch64-unknown-linux-gnu` + a cross linker
  (`aarch64-linux-gnu-gcc`).

## Build

```sh
cargo build --release                 # -> target/release/libggml_npu.so
GGML_DIR=/path/to/llama.cpp cargo build --release   # if not a sibling
cargo check --features direct-ivpu    # safe direct Linux foundation; no graph execution
```

The direct runtime boundary and exact prerequisites for a first direct matmul are documented in
[`docs/direct-runtime-foundation.md`](docs/direct-runtime-foundation.md). Direct IVPU
(`src/vendor/ivpu/session.rs`) is the default runtime. Level Zero remains an opt-in
compatibility / differential oracle behind `level-zero-compat`
(`src/vendor/ivpu/session_level_zero.rs`). `direct-ivpu` enables a no-std,
ELF-only Goblin configuration for device artifacts; host shared-library loading continues to use
the platform `dlopen`/`dlsym` boundary.

Host microbenchmarks use feature-gated Criterion, while llama-bench remains the end-to-end pp/tg
framework; see [`docs/benchmarking.md`](docs/benchmarking.md). The io_uring transport decision is
documented in [`docs/io-uring-assessment.md`](docs/io-uring-assessment.md).
Persistent and in-process key semantics are documented in
[`docs/identity.md`](docs/identity.md).

`ggml_*` symbols are left undefined and resolve at `dlopen` from the host; set
`GGML_LIB_DIR=/dir/with/libggml-base.so` to link deterministically instead. The cdylib is
`libggml_npu.so` (underscore); load it by explicit path.

## Verify

```sh
./tests/run_registration_check.sh     # -> REGISTRATION PASS
```

End-to-end, against a host built with `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON`:

```sh
GGML_BACKEND_PATH=$PWD/target/release/libggml_npu.so \
  /path/to/llama.cpp/build/bin/test-backend-ops support -b NPU
```

## Diagnostics

- `GGML_NPU_TRACE=1` prints each assigned scheduler split with op ids, shapes, strides, data
  addresses, and NPU-buffer ownership.
- `GGML_NPU_PROFILE=1` reports stage-in, submit, device/sync, and copy-out time at process exit.
- `GGML_NPU_DEBUG=1` enables failure-path notes.
- `GGML_NPU_BITS=4|8|16` overrides the host-side requant width for quality/performance studies.
- `GGML_NPU_DEVICE_CLASS=npu|igpu|gpu|accel` overrides the scheduler identity. `npu` is the
  default; `igpu` is an experiment that permits whole-layer model placement.
- `GGML_NPU_ATTENTION=full` opts into SoftMax (`max_bias==0`), FlashAttention (`max_bias`/
  softcap `==0`, GQA), and runtime-weight KQ/KQV matmul candidates (capability + `offload_op` pull);
  the default keeps the attention core on GPU while allowing constant-weight/QKV projections,
  RoPE (normal/NEOX), and KV-cache SetRows needed for contiguous prefill QKV regions. Exact
  `full` only — `=1` / presence is not enough. See `docs/coverage-gates.md`.
- `GGML_NPU_MOE=1` opts into experimental MoE glue (Sigmoid/SumRows) and Argsort expert-id
  production with **ggml-visible i32 writeback**. MatMulId stays **unclaimed** — host/Vulkan runs
  routed experts until model-scale evidence. `GGML_NPU_MOE=0` or unset keeps MoE routing
  **unclaimed** (presence of `=0` must not enable). Output-head-scale MatMuls stay unclaimed under
  the expanded constant budget (prefer GPU) unless `GGML_NPU_ALLOW_GIANT_CONSTANTS` is set — see
  [`docs/coverage-gates.md`](docs/coverage-gates.md) (#11).
- `GGML_NPU_MEMORY_LIMIT=<bytes>` sets the shared-memory placement budget reported to ggml.
- `GGML_NPU_ASYNC=1` queues zero-copy NPU-owned splits and synchronizes through backend events;
  any staged foreign binding remains synchronous to protect bounce-buffer lifetimes.
- `GGML_NPU_LFM_SHORTCONV=1` opts into ancestry-based placement of LFM convolution segments while
  leaving sequence gather/write setup and unrelated attention/MoE glue on CPU/GPU.

Run the hermetic claim ⊆ coverage matrix gate with `./tests/run_claim_coverage_check.sh`
(see [`docs/coverage-gates.md`](docs/coverage-gates.md); issue #12). The broader local operation
gate is `./tests/run_coverage_gate.sh`. Set `MODEL` to
also run two fresh-process, timeout- and memory-guarded LFM prefill/decode checks with MoE,
ShortConv, and full attention enabled. The default keeps model placement on CPU (`NGL=0`) and pulls
eligible NPU splits; `NPU_ONLY=1 NPU_DEVICE_CLASS=igpu` remains an explicit experiment. Whole-layer
placement previously reached only 1.84 prompt tok/s versus roughly 53 tok/s for Vulkan+NPU
enumeration, and cold compilation of large routed-expert graphs can exceed the default timeout.

## Roadmap

The backend is moving from bespoke op matching to an independent graph-oriented compiler:
correctness and cache-lifetime gates first, then LFM ShortConv state, sparse MoE feasibility,
heterogeneous attention, broad llama.cpp graph coverage, and asynchronous cost-based placement.
