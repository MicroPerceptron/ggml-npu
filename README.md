# ggml-npu

An out-of-tree Rust backend connecting [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)
to [virtio-accel](https://github.com/MicroPerceptron/virtio-accel)'s native providers.
Cargo builds the plugin independently; llama.cpp needs no source changes or Rust toolchain.
The bridge calls the provider's Rust `Accelerator` interface directly. It does not use a VM,
virtqueue, wire protocol, or a new C wrapper.

## First integration

This is a correctness baseline, not a full-model NPU implementation or a performance claim.
The active plugin lives in `src/bridge/`. Earlier Intel prototype code, benchmarks, and design
notes remain in the repository for reference but are not compiled by the new Cargo manifest.

- Vulkan is the default provider (`VIRTIO_VULKAN`). A Vulkan 1.3 loader and device are required;
  Mesa lavapipe provides a software device for CI. `GGML_NPU_VULKAN_DEVICE` selects an exact
  name returned by virtio-accel's Vulkan enumeration; otherwise its preferred device is used.
- `--features coreml` additionally exposes `VIRTIO_COREML` on supported Macs. Core ML controls
  CPU/Neural Engine placement; successful execution is not proof of exclusive ANE execution.
- Supported: FP32 vectors/matrices, dense `MUL_MAT`, `ADD`, `MUL`, `TANH`, and `SILU`.
  Binary broadcasting supports singleton dimensions, not ggml's general tiled repetition.
- Input host views may have offsets, padded rows, or transposed strides. Computed views form
  a scheduler boundary; in-place outputs, higher-rank batches, quantization, attention, and
  mutable state are not claimed. Each matrix dimension is at most 8192.
- One ggml split becomes one validated TOSA program. All computed outputs are copied back,
  including intermediate results that ggml may consume after the split. There is a 128-node
  and 128 MiB aggregate boundary-buffer limit.
- Each backend instance owns one provider worker. Calls complete synchronously, so providers
  with thread-local handles need no `Send` implementation. Programs and provider allocations
  are cached for the most recent graph signature; a shape/topology change replaces that entry.
  Tensor addresses and values are not cache keys. Every invocation uploads its current inputs,
  including weights, so this baseline does not provide resident-weight or zero-copy performance.
- Unsupported operations remain on ggml's other backends. A failed assigned split reports
  failure; it is never silently treated as success or automatically replayed on the CPU.

## Build

The integration is pinned to llama.cpp commit
`29c720eaf72fcf8e4e61a56a88b994d92668261d` (ggml backend API 3), and virtio-accel commit
`8b66e79b6d889886fb22b0c9e72d868ff96fc9cb`. Pinning matters: the ggml backend interface is internal.
Use stable Rust, clang/libclang, CMake, and a C/C++ compiler.

```sh
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
git -C ../llama.cpp checkout 29c720eaf72fcf8e4e61a56a88b994d92668261d
cmake -S ../llama.cpp -B ../llama.cpp/build \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON
cmake --build ../llama.cpp/build --target llama-cli -j

export GGML_DIR="$(cd ../llama.cpp && pwd)"
export GGML_LIB_DIR="$GGML_DIR/build/bin"
cargo build --locked --release
```

The Cargo dependencies fetch virtio-accel at the pinned revision; no sibling checkout or vendor
SDK is needed to compile the default Vulkan path. `GGML_LIB_DIR` links the host's `ggml-base`;
keep that same build available to the system loader when loading the plugin. On Linux:

```sh
export LD_LIBRARY_PATH="$GGML_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_BACKEND_PATH="$PWD/target/release/libggml_npu.so"
"$GGML_DIR/build/bin/llama-cli" --list-devices
```

macOS uses `libggml_npu.dylib`. A Homebrew Vulkan loader may need
`DYLD_LIBRARY_PATH="$GGML_LIB_DIR:/opt/homebrew/lib"` (also for Rust unit tests). The plugin registers an auxiliary `ACCEL` device and
pulls supported FP32 matrix multiplication from host buffers; loading it does not make
quantized GGUF models supported. Full model inference and useful offload policies are later work.

## Validation

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-features
bash tests/run_bridge_check.sh
```

The integration harness uses the upstream external-backend loader and compares a
matmul/bias/SiLU/multiply/tanh graph against ggml CPU. It covers updated inputs, new addresses,
program reuse and replacement, padded/offset and transposed input views, plugin-owned host
buffers, unsupported-operation rejection, mixed CPU/bridge scheduling, and teardown/reopening.
A missing selected provider is a test failure, not a skip.

Linux CI pins `VK_DRIVER_FILES` to lavapipe and also checks `llama-cli --list-devices`.
To exercise the same graph translator on Core ML:

```sh
cargo build --locked --release --features coreml
GGML_NPU_TEST_PROVIDER=VIRTIO_COREML \
  bash tests/run_bridge_check.sh "$PWD/target/release/libggml_npu.dylib"
```

`GGML_CPU_MODULE` can override the CPU module path when the host was built with a different
output layout. `TMPDIR` must be writable by Core ML's model compiler on macOS.

## License

[MIT](LICENSE). Third-party components retain their own licenses and notices.
