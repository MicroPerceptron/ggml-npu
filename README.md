# ggml-npu

A universal NPU backend for [ggml](https://github.com/ggml-org/ggml) / llama.cpp,
targeting Intel (Level Zero), AMD (XRT/XDNA), and Qualcomm (Hexagon/QNN) client NPUs
behind one runtime seam, with CPU fallback for everything it does not claim.

**Status: Phase 0 skeleton + Intel bring-up.** The backend registers as a dynamically-loaded
ggml `ACCEL` device and claims **no** operations (`supports_op == false`), so the scheduler
runs the whole model on CPU — proving registration + the C ABI + the DL entry point. The
Intel NPU stack is validated end-to-end (a fixed-shape MatMul runs on the Lunar Lake NPU and
matches CPU); the Level Zero submit path is next. See
[`../llama.cpp/tmp/NPU-gameplan.md`](../llama.cpp/tmp/NPU-gameplan.md) and
[`../llama.cpp/tmp/NPU.md`](../llama.cpp/tmp/NPU.md).

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
src/hal.rs              NpuHal — the universal 5-verb submission seam
src/vendor/stub.rs      CPU-echo NpuHal (no hardware)
src/vendor/ivpu/        Intel NPU (Level Zero + OpenVINO); level_zero/ vendored headers   <- active
src/vendor/xdna/        AMD Ryzen AI (XRT + mlir-aie)                                       skeleton
src/vendor/hexagon/     Qualcomm Hexagon (FastRPC/QNN + hexagon-mlir)                       skeleton
tests/registration_check.c   self-contained Phase 0 verifier (mimics ggml's DL loader)
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
./tests/run_registration_check.sh     # -> PHASE 0 PASS (ACCEL, api_version 2, supports_op false)
```

End-to-end, against a host built with `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON`:

```sh
GGML_BACKEND_PATH=$PWD/target/release/libggml_npu.so \
  /path/to/llama.cpp/build/bin/test-backend-ops support -b NPU
```

## What's next

Bring up `vendor/ivpu`: bindgen `ze_api.h` + `ze_graph_ext.h`, implement `probe` (zeInit +
enumerate the NPU) and `load`/`submit` over the graph extension, load an OpenVINO-compiled
blob, and wire the first fixed-shape `MUL_MAT` into `graph_compute` — verified vs CPU with
`ggml_backend_compare_graph_backend`.
