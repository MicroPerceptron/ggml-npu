# Quant transcode SIMD evaluation

Date: 2026-07-12

Host used for measurements: Intel Core Ultra 5 226V, x86-64, AVX2/FMA, no AVX-512,
Rust 1.99.0-nightly (2026-07-07). Every timed executable ran under `taskset -c 0-1`;
these are the verified 4.5 GHz P-cores, and CPUs 4-7 were not used. Parallel-capable discovery
runs held `flock -s /tmp/ggml-npu-bench-global.lock`. The selected realistic comparison, cycle
counts, full conversion, and artifact benchmark were then repeated while holding the lock
exclusively, so the sequential results below are authoritative. Builds and ordinary tests were
not pinned.

Production `RUSTFLAGS` was unset. The Rust target was the default `x86_64-unknown-linux-gnu`
baseline (`fxsr,sse,sse2,x87`); the Q6_K AVX2 specialization uses per-function
`#[target_feature]` after `cpufeatures` runtime detection. Rust timings used Cargo's optimized
bench/release profiles, and the C++ artifact benchmark used CMake `Release`. CPUs 0-1 used
`intel_pstate`, a 400 MHz-4.5 GHz policy, and the `powersave` governor. Hardware current-frequency
reporting was unavailable; an asserted approximately 2.1 GHz was observed before the runs, and
EPP varied between `balance_performance` and `balance_power` during the measurement window. No
governor or EPP setting was changed. Q4_K and Q6_K have 256 elements per source block.

## Decision

- `portable-simd` is now in the default production feature set. Q4_K / Q6_K packing and fallback
  requant float hot loops use `core::simd` as the written production path; LLVM emits scalar where
  the target has no vectors. On x86-64, Q6_K uses a per-function AVX2 implementation only when the
  no-std-compatible `cpufeatures` detector verifies CPU and OS support.
- Builds with `--no-default-features --features direct-ivpu` use `ScalarKernel` / scalar requant
  helpers. Default builds always select portable SIMD (plus optional AVX2 Q6). No global
  `target-cpu` or `target-feature` flag is required.
- Selection is observable as `scalar`, `portable-simd`, or `portable-simd+avx2-q6` in benchmark
  metadata and once under `GGML_NPU_TRACE`.
- `ScalarKernel` remains the byte-exact reference and the `--no-default-features` path. All
  production and specialized routes are tested against it.
- `pulp` 0.22.3 was evaluated and then removed. Its route was byte-exact, but was no faster than
  scalar on the authoritative sequential realistic workload and would add nine transitive
  packages. No production or benchmark dependency remains.
- GGML externs remain the authoritative fallback decoder, not the selected transcode accelerator.
  Their exported Q4_K/Q6_K conversion functions are scalar and produce F32; SIMD kernels in
  `ggml-cpu` implement Q4_K/Q6_K-by-Q8_K dot products, which is a different operation.

The existing toolchain already requires nightly for `allocator_api` and scalar `f16`. Promoting
`portable_simd` adds another unstable API/code-generation contract: every nightly upgrade must
re-run byte-exact tests, cross-target checks, assembly inspection, and these benchmarks. The
semantic constant descriptor remains architecture-neutral.

## GGML kernel inventory

The base `ggml_get_type_traits` ABI returns a pointer to a process-lifetime static trait table.
Q4_K and Q6_K expose:

- `to_float`: `dequantize_row_q4_K` / `dequantize_row_q6_K`
- `from_float_ref`: `quantize_row_q4_K_ref` / `quantize_row_q6_K_ref`
- block size 256 and storage sizes 144 / 210 bytes

The CPU-only `ggml_get_type_traits_cpu` table adds `from_float` and `vec_dot`. For these two types,
`from_float` is only a checked wrapper around the scalar reference quantizer. The x86 implementation
has AVX and AVX2 Q4_K/Q6_K-by-Q8_K dot products; the ARM implementation has NEON, SVE, and optional
I8MM dot products. Neither architecture has a vectorized exported Q4_K/Q6_K dequantizer,
requantizer, or source-native constant transcoder.

Extern ABI requirements:

- `k` must be a multiple of 256; the C functions assert this.
- Source blocks have at least two-byte C alignment and destination F32 has four-byte alignment.
- `restrict` requires source and destination not to overlap, and both buffers must remain live for
  the duration of the call.
- There is no partial-block tail contract.

Reusing the base decoder preserves GGML numeric formulas, but requires `4 * k` bytes of row scratch
and then a second lossy canonical quantization. Reusing `ggml-cpu` vector-dot code would add a CPU
backend dependency and Q8_K preparation while still not producing the compiler constant format.

## `pulp` fit

The evaluated release was 0.22.3, published 2026-06-20.

- Stable Rust and `no_std` were verified with `--no-default-features --features x86-v3`.
- AArch64 `no_std` cross-check completed. NEON is available through the AArch64 implementation;
  without `std`, ARM feature selection is compile-time, while x86 uses raw CPUID at runtime.
- x86-v3 supplies AVX2 runtime selection. The optional x86-v4/AVX-512 feature also compiled on
  stable, but could not be benchmarked on this non-AVX-512 host.
- Fixed-size block inputs make tails explicit. A 17-block unaligned case produced exact bytes.
- The route added bytemuck, libm, num-complex/num-traits, paste, the WASM feature helper,
  raw-cpuid/bitflags, and reborrow. The dependency was removed after measurement.

## Generated ISA and specialization

AVX-VNNI is the 128/256-bit VEX-encoded integer dot-product extension exposed here as
`avx_vnni`. It is distinct from AVX-512 VNNI, which uses EVEX/AVX-512 state and is absent on this
host.

Assembly wrappers in `benches/quant_asm.rs` establish:

- Baseline x86-64 Q4_K uses SSE2 `movdqu`, `psrlw`, `pand`, `packuswb`, and stores. Baseline Q6_K
  likewise uses XMM shifts, masks, shuffles, and packing; it does not emit YMM instructions.
- The per-function AVX2 Q4_K experiment emits YMM `vpmaddubsw`, `vpackuswb`, and `vpermq`.
  Criterion improved, but the fixed-iteration cycle harness regressed, so it is not selected in
  production.
- The selected Q6_K AVX2 function emits YMM loads, shifts, masks, OR/XOR, and vectorized packing.
  It materially outperformed baseline portable SIMD.
- AArch64 baseline Q4_K emits NEON `ld2`, `sli`, `sri`, and paired Q-register stores. Q6_K emits
  NEON `ushr`, `and`, `ushl`, `shl`, `orr`, `eor`, and `tbl`.
- No `vpdp*`, ZMM, or AVX-512 instructions appear in the isolated route assembly.

Q4_K/Q6_K transcoding contains bit extraction, shifts, masks, rearrangement, and metadata copies,
not integer multiply-dot-accumulate. AVX-VNNI and AVX-512 VNNI are therefore inapplicable, so no
VNNI path was added. There is no unreachable AVX-512 implementation to maintain or accidentally
dispatch.

Reproduction:

```sh
cargo rustc --profile bench --bench quant_asm -- --emit=asm -C codegen-units=1
cargo rustc --target aarch64-unknown-linux-gnu --profile bench --bench quant_asm \
  --no-default-features --features direct-ivpu,portable-simd -- --emit=asm -C codegen-units=1
```

The AArch64 library check succeeds. The assembly file was emitted; final executable linking is not
available on this host because an AArch64 C cross-linker is not installed.

## Benchmark evidence

All SIMD outputs were checked byte-for-byte against scalar before timing. The GGML measurements are
dequantize-to-F32 lower bounds, not equivalent output formats; complete canonical fallback also has
to scan and requantize that F32 row.

Authoritative sequential Criterion run, 1,048,576 elements (`K=4096`, `N=256`), median estimate:

- Q4_K: scalar 34.989 us / 29.969 Gelem/s; portable 24.919 us / 42.079 Gelem/s;
  explicit AVX2 22.926 us / 45.738 Gelem/s; GGML F32 decode 232.89 us / 4.503 Gelem/s.
- Q6_K: scalar 574.49 us / 1.825 Gelem/s; portable 526.01 us / 1.993 Gelem/s;
  explicit AVX2 had a noisy 200.57 us median estimate with a 139.44-245.01 us interval; GGML F32
  decode was 923.46 us / 1.136 Gelem/s. The fixed-iteration result below is authoritative for Q6.

Authoritative sequential `perf stat`, 5,242,880,000 transformed elements:

- Q4_K: scalar 0.086757 cycles/element and 29.830 Gelem/s; portable 0.067422 and
  39.647 Gelem/s; explicit AVX2 0.098044 and 26.718 Gelem/s. Production therefore selects portable
  baseline and measured 0.067176 cycles/element.
- Q6_K: scalar 1.466349 cycles/element and 1.830 Gelem/s; portable 1.346184 and
  1.977 Gelem/s; explicit AVX2 0.350785 and 7.315 Gelem/s. Selected production measured
  0.347148 cycles/element and 7.334 Gelem/s.

Shared-lock, parallel-capable case-coverage run:

- Aligned Q4_K block: scalar 8.595 ns, portable 6.136 ns, explicit AVX2 4.703 ns.
- Aligned Q6_K block: scalar 142.42 ns, portable 133.87 ns, explicit AVX2 30.332 ns.
- Unaligned Q4_K 17-block tail: scalar 146.93 ns, portable 108.20 ns, explicit AVX2 61.674 ns.
- Unaligned Q6_K 17-block tail: scalar 2.3374 us, portable 2.1811 us, explicit AVX2 523.45 ns.

Authoritative sequential optimized full adapter conversion, including allocation and metadata,
for 1,048,576 elements:

- Q4_K: scalar 225.433 us versus selected production 204.136 us (9.45% faster), with
  606,208 persistent bytes.
- Q6_K: scalar 1,085.529 us versus selected production 279.789 us (74.23% faster), with
  860,160 persistent bytes.

The `target-cpu=native` diagnostic was built separately and never used for production. It measured
0.078254 cycles/element / 27.999 Gelem/s for selected Q4_K (worse than baseline) and
0.328804 / 7.829 Gelem/s for selected Q6_K (a modest upper-bound improvement).

For this tensor, canonical asymmetric U4/U8 descriptor input is 1,146,880 bytes. Q4_K native input
is 47.14% smaller than canonical (and 2.78% larger than the original Q4_K file blocks because exact
factorized metadata is separated). Q6_K native input is source-size exact and 25% smaller than
canonical. Scalar has no value scratch; portable/AVX2 use at most 64 bytes of bounded local arrays
per block. Canonical fallback uses one F32 row, 16 KiB at `K=4096`, instead of the old 4 MiB
full-tensor F32 scratch.

The scalar release shared library was 358,704 bytes; default production was 360,544 bytes, an
increase of 1,840 bytes (0.51%).

## Build matrix

Verified with Rust 1.99.0-nightly (2026-07-07), LLVM 22.1.8:

- Default x86-64 baseline: `cargo check --target x86_64-unknown-linux-gnu`.
- Scalar production fallback: `cargo check --no-default-features --features direct-ivpu`.
- AArch64 default production: `cargo check --target aarch64-unknown-linux-gnu`.
- Default tests: 61 passed, one explicit hardware test ignored.
- Scalar fallback tests: 59 passed, one explicit hardware test ignored.

The repository's bare `--no-default-features` configuration predates this work and does not compile
because the Intel session expects either `direct-ivpu` or Level Zero. The supported scalar fallback
therefore explicitly retains `direct-ivpu`.

## Compiler and hardware evidence

The OpenVINO graph inspection test verifies:

- Q4 native: a U4 constant plus scale multiply and affine subtract, with no I8 or F32 weight
  constant.
- Q6 native: an I8 constant plus scale multiply, with no U4, affine offset, or F32 weight constant.

Tiny isolated NPU artifacts compiled and executed with known answers. The authoritative sequential
P-core run used the release C++ benchmark and reported:

- canonical U4: 206.344 ms compile, 41,089-byte artifact, 134.448 us warm p50.
- canonical U8: 75.082 ms compile, 36,993-byte artifact, 132.547 us warm p50.
- Q4_K native: 80.803 ms compile, 69,761-byte artifact, 138.142 us warm p50.
- Q6_K native: 74.926 ms compile, 36,993-byte artifact, 132.972 us warm p50.

The expected outputs (128, 128, 0, and -1408) matched. These tiny kernels are submission-overhead
dominated; they prove artifact acceptance and execution, not performance parity. The graph
inspection proves low-bit/dequant structure, but no compiler diagnostic exposed a fusion decision,
so decompression-matmul fusion is not claimed.
