# ggml-npu

A universal NPU backend for [ggml](https://github.com/ggml-org/ggml) / llama.cpp,
targeting Intel (Level Zero), AMD (XRT/XDNA), and Qualcomm (Hexagon/QNN) client NPUs
behind one runtime seam, with CPU fallback for everything it does not claim.

**Status: Intel prototype, moving toward graph coverage.** The backend registers as a
dynamically-loaded ggml `ACCEL` device and currently accelerates constant-weight matmuls plus
recognized fused subgraphs (SwiGLU FFN, RMSNorm, residual glue) on Intel Lunar Lake. It
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

## Layout

```
src/lib.rs              no_std crate root: global allocator, panic handler, mod ffi (bindgen), DL entry points
src/backend/mod.rs      singleton vtable graph (AtomicPtr CAS wiring)
src/backend/shell.rs    registry + device + backend(stream) vtables
src/backend/buffer.rs   buffer-type + buffer vtables (host allocation)
src/engine.rs           scheduler split recognition and graph dispatch
src/ops/                matmul, graph-pattern IR, and host requantization
src/hal.rs              aspirational universal 5-verb submission seam
src/vendor/stub.rs      CPU-echo NpuHal (no hardware)
src/vendor/ivpu/        Intel NPU (Level Zero + OpenVINO); level_zero/ vendored headers   <- active
src/vendor/xdna/        AMD Ryzen AI (XRT + mlir-aie)                                       skeleton
src/vendor/hexagon/     Qualcomm Hexagon (FastRPC/QNN + hexagon-mlir)                       skeleton
tests/registration_check.c   self-contained registration verifier (mimics ggml's DL loader)
build.rs                bindgen driver (locates ggml headers + clang resource dir)
```

## Prerequisites

- **Rust nightly** (pinned by `rust-toolchain.toml`) — required for `allocator_api`.
- **clang** with its resource headers (`build.rs` uses `clang -print-resource-dir`).
- A **llama.cpp checkout** for the ggml headers: sibling `../llama.cpp` by default, or set
  `GGML_DIR=/path/to/llama.cpp`.
- Cross-ISA: `rustup target add aarch64-unknown-linux-gnu` + a cross linker
  (`aarch64-linux-gnu-gcc`).

## Build

```sh
cargo build --release                 # -> target/release/libggml_npu.so
GGML_DIR=/path/to/llama.cpp cargo build --release   # if not a sibling
```

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
- `GGML_NPU_ATTENTION=full` opts into runtime KQ/softmax/KQV graph candidates; the default keeps
  attention on GPU while allowing constant-weight/QKV projections.
- `GGML_NPU_MOE=1` opts into the experimental baked-expert-bank plus runtime-index path.
- `GGML_NPU_MEMORY_LIMIT=<bytes>` sets the shared-memory placement budget reported to ggml.
- `GGML_NPU_ASYNC=1` queues zero-copy NPU-owned splits and synchronizes through backend events;
  any staged foreign binding remains synchronous to protect bounce-buffer lifetimes.

## Roadmap

The backend is moving from bespoke op matching to an independent graph-oriented compiler:
correctness and cache-lifetime gates first, then LFM ShortConv state, sparse MoE feasibility,
heterogeneous attention, broad llama.cpp graph coverage, and asynchronous cost-based placement.
