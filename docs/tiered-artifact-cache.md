# Tiered artifact/program cache

Status: implemented on the isolated `artifact-cache-foundation` worktree based on `505b7c1`.
The direct-runtime and compiler integration is functional; hardware warm/cold validation remains
to be repeated after the active zero-copy work is reconciled.

## Data flow and boundaries

1. The direct runtime checks T0 for a prepared resident program. Entries are byte-accounted,
   atomically pinned while in use, and evicted to a 90% low-water mark. Concurrent preparation of
   the same semantic program is single-flight.
2. A T0 miss invokes the compiler path, which computes a canonical BLAKE3 key from graph structure,
   shape/type/stride metadata, compiler options, and raw constant content before requantization.
3. T1 is checked for a compressed record. T1 stores no uncompressed artifact and uses a 4-way
   count-min frequency sketch plus deterministic frequency/age/key victim ordering. The default
   budget is 256 MiB with a 64 MiB per-entry limit.
4. T2 is checked next. Records are content-addressed by the semantic key, bounded before allocation,
   compatibility checked, decompressed, and BLAKE3 verified. A successful T2 read is admitted to T1.
5. A complete miss elects one compiler per key. Compilation, compression, filesystem operations,
   decompression, ELF parsing, buffer allocation, submission, and waiting all run outside global
   cache spinlocks.
6. The compiled artifact is compressed once, published to T2, considered for T1 admission, and
   returned as the one transient uncompressed staging buffer. The direct loader drops that buffer
   after ELF inspection/linking and device-buffer preparation.

The neutral `artifact_cache` module depends only on `core`/`alloc`, BLAKE3, and provider traits.
Codec, futex, environment, cgroup, and filesystem mechanics live in the Linux host adapter.
The compiler and vendor layers know only the neutral cache API.

The record header is a fixed 352-byte little-endian, pointer-free structure. It includes record and
artifact schemas, the full semantic key, compiler/pipeline/toolchain/target/firmware/runtime
compatibility digests, codec ID/runtime version/options, compressed and uncompressed lengths, an
uncompressed digest, and a digest over the header itself.

T2 uses per-key `flock` files to prevent duplicate publication and a directory-wide `flock` for
cross-process budget accounting. It writes a unique same-directory temporary file, fsyncs it,
atomically renames it, then fsyncs the directory. Reads update recency. Eviction is oldest mtime
first with the filename as a deterministic tie-break. Invalid and incompatible records are removed.

## Codec decision

The production-portable selection remains Zstandard level 1 behind the neutral
`CompressionProvider`. The Linux adapter opens `libzstd.so.1`, resolves every required symbol,
requires a runtime version in `[1.5.0, 2.0.0)`, and records the exact runtime version in each cache
record. Missing library, symbol, or supported version disables T1/T2 and falls back to ordinary
compilation; direct and Level Zero execution remain functional. Deployments therefore need the
`libzstd.so.1` ABI (Fedora package `libzstd`, Debian/Ubuntu package `libzstd1`) for artifact caching.
No codec, SIMD, C ABI, `std`, or libc concept enters the neutral cache contracts.

### Exact candidate builds and acceleration

- `zstd 0.13.3` / `zstd-safe 7.2.4` / `zstd-sys 2.0.16+zstd.1.5.7` was built
  with `default-features = false`: only the transitive `std` marker was active. This intentionally
  omits unrelated legacy-format, dictionary-builder, arrays, and bindgen defaults without disabling
  acceleration. The captured C command was `cc -O3 -ffunction-sections -fdata-sections -fPIC -m64`
  plus hidden visibility and constant merging; there was no `-march=native`. The x86-64 assembly
  source was compiled, and archive inspection found its fast Huffman symbols plus BMI2 `shrx`/`bzhi`
  instructions. On generic GCC/Clang x86 builds, libzstd enables runtime BMI2 dispatch; SSE2 is the
  x86-64 baseline. AArch64 uses compile-time NEON intrinsics. AVX2 outside the BMI2-targeted
  functions is compile-time-only and was not enabled in the production-portable C build.
- `no_asm` is a real x86-64 decompression switch. In the exclusive sequential comparison, assembly
  changed no bytes and no compression throughput, while aggregate decompression rose from
  798.9 to 860.6 MiB/s (+7.7%); on the 116.5 MiB artifact it rose from 915.5 to
  1031.7 MiB/s (+12.7%). Production must not enable `no_asm`.
- `fat-lto` and `thin-lto` are cross-language LTO options, not generic performance toggles. The
  recorded GCC/rust-lld `fat-lto` build failed with unresolved `ZSTD_*` symbols; these options need a
  coordinated Clang/LLVM plugin-LTO toolchain and are not portable defaults. The separate `thin`
  feature is a size-minification mode that forces smaller decoders and disables dynamic BMI2, so it
  was also rejected. `pkg-config` requests a static library; this Fedora installation provides only
  the shared development link, so it is not the package-managed static alternative.
- Fedora `libzstd.so.1.5.7` (RPM `libzstd-1.5.7-5.fc44`, 959,384 bytes) was the system provider.
  Its ELF advertises x86-64 baseline and v3 code and disassembly contains BMI2 instructions.
  Upstream 1.5.7's generic build guards these paths with runtime BMI2 detection. The bundled archive
  was 1,034,876 bytes and increased the otherwise identical benchmark binary by 706,264 bytes.
- `structured-zstd 0.0.48` used `default-features = false`, `std`, `hash`, and the exact portable
  kernel set `sse2,bmi2,avx2,neon,sve,simd128`; AVX-512 VBMI2 was deliberately omitted because the
  crate itself excludes it from defaults. These are stable `std::arch` kernels, not nightly
  `portable_simd`: `std` selects the decode tier at runtime (AVX2 here), while `no_std` selects only
  compile-time target features. Disabling the kernel features produced a scalar decoder, but not a
  wholly scalar encoder: its Huffman writer independently runtime-dispatches BMI2 on x86, and its
  AArch64 histogram has an SVE2-tagged path. At level 1, the AVX2 decoder measured 536.7 versus
  527.9 MiB/s for the decode-scalar build; compression was 299.8 versus 289.6 MiB/s. The small
  deltas do not overcome its lower ratio and throughput than C Zstandard.
- `zrip 0.8.3` used `default-features = false` plus `std,frame,ldm,simd`, matching its useful
  defaults explicitly. In this published version, `simd` affects decode sequence execution only,
  using stable `fearless_simd 0.5.0` runtime dispatch (AVX2/SSE4.2 on x86, NEON on AArch64, and
  compile-time SIMD128 on WebAssembly); the encoder has no explicit SIMD. Its `nightly` feature only
  enables optimization attributes and was not used. The AVX2 build did not outperform the
  decode-scalar build consistently on this corpus and level 1 retained only 1.494x aggregate.
- `lz4_flex 0.13.1` remains a reference, not a candidate default. The portable build explicitly
  enabled safe encode, safe decode, and checked decode. It has no runtime SIMD dispatcher; a small
  copy primitive can use AVX only when enabled at compile time. Disabling the safety/checking
  features improved aggregate compression/decompression from 604.7/1407.7 to
  715.5/1625.0 MiB/s, but that unsafe trusted-input configuration is not suitable for persistent
  cache records and the aggregate ratio remained only 1.052x.

### Reproducible benchmark result

Every codec/artifact pair ran in a fresh child against five existing native Intel artifacts
(68 KiB, 1.56 MiB, 8.04 MiB, 49.6 MiB, and 116.5 MiB). Timed suites inherited
`taskset -c 2-3` on the Lunar Lake 4.5 GHz P-core group and used
`/tmp/ggml-npu-bench-global.lock`; the broad matrix used a shared lock, while selected provider and
assembly comparisons used one exclusive lock and ran sequentially. The `powersave` governor was
left unchanged (observable range 400-4500 MHz). Metadata records affinity and per-sample frequency,
runtime CPU features, versions, exact Cargo features, compiler flags, target features, binary size,
first-operation/cold-main latency, warm medians, ratio, and `VmHWM`.

The portable build used rustc 1.99.0-nightly/LLVM 22.1.8, Cargo 1.99.0-nightly, GCC 16.1.1,
generic `x86_64-unknown-linux-gnu` Rust target features (SSE2 baseline), thin Rust LTO, and no
`RUSTFLAGS`, `CFLAGS`, or `target-cpu=native`. A separate `-C target-cpu=native` result is retained
only as a machine-specific upper-bound observation; it is not deployable configuration and does not
change the separately compiled C libzstd without matching `CFLAGS`.

The authoritative exclusive/sequential level-1 provider comparison measured:

- identical per-file ratios of 25.431x, 1.505x, 1.359x, 1.269x, and 2.200x; total
  184,328,970 source bytes became 103,792,169 bytes (1.776x);
- system provider: 618.7 MiB/s compression and 892.1 MiB/s decompression aggregate;
- bundled provider: 659.6 MiB/s compression and 878.2 MiB/s decompression aggregate;
- on the 116.5 MiB artifact, system first-operation compression/decompression was
  171.9/111.8 ms versus bundled 160.1/110.9 ms; high-water growth was about 287 MiB for both.

The bundled provider's +6.6% aggregate compression and -1.6% decompression difference is not large
enough to justify making a C toolchain and roughly 690 KiB binary growth mandatory. The system
provider is acceptable with the explicit ABI/version/deployment checks above; the bundled provider
remains a viable opt-in for hermetic distributions. Zstandard level 1 remains the recommendation
because it preserved 1.776x aggregate versus LZ4's 1.052x while maintaining promotion throughput
well above the storage/device preparation path.

Machine-readable matrix and authoritative results are in
`benchmarks/cache-codec-results/acceleration-matrix/`; the reproducible runners are
`tools/cache-codec-bench/run-matrix.sh` and `run-zstd-asm-comparison.sh`. Criterion measurements for
compression, verified promotion (decompression plus BLAKE3), and header parsing remain in
`benchmarks/cache-codec-results/criterion-artifact-cache.json`.

The reported `VmHWM` delta includes output vectors and allocator retention, so it is an upper-bound
process measurement rather than isolated codec workspace. No end-to-end model-execution performance
claim is made from these codec-path results.

## Budgets and memory expectation

Defaults are stable after one cgroup-aware initialization:

- T0 high water: 3 GiB unless `GGML_NPU_RESIDENT_MB` overrides it; 90% low water.
- T1: 256 MiB, 64 MiB maximum entry, configurable with `GGML_NPU_RAM_CACHE_MB` and
  `GGML_NPU_RAM_CACHE_ENTRY_MB`.
- T2: 24 GiB, configurable with `GGML_NPU_DISK_CACHE_MB`; 90% low water.

The earlier 12.1 GiB churn came from retaining/unloading large uncompressed artifacts around a
roughly 2.86 GiB resident limit. This design bounds retained compressed host RAM to 256 MiB and
resident device programs to the configured T0 budget, with only one uncompressed artifact active
during a cold promotion. The measured five-file artifact set would occupy about 99 MiB compressed,
not 176 MiB uncompressed. This supports the expectation that artifact retention itself fits the
pp512 budget, but it does not prove that the full pp512 device working set avoids churn: a hardware
run must still record T0 resident bytes, evictions, pins, and staging peaks after zero-copy merges.

## Integration and verification status

Implemented:

- semantic keys before requantization for constant, fused-subgraph, and graph-IR compilation;
- shared T1/T2 compiler path for direct and Level Zero execution;
- pin-aware, byte-budgeted direct T0 with concurrent-load suppression;
- stable cgroup-aware budgets and hysteresis;
- corruption/compatibility/codec rejection, bounded parsing, crash-safe T2 writes, cross-process
  publication, deterministic eviction, and read recency;
- per-tier hits/misses/bytes, bypass/eviction/corruption reasons, compression ratio, and latency
  counters for compile/compress/decompress/disk/promotion/single-flight/rehydration.

Not yet hardware-proven:

- Level Zero keeps its existing resident graph ownership; the new T0 policy applies to direct mode,
  while the neutral compiled-artifact tiers are shared by both modes.
- Compatibility fields currently identify versioned adapter contracts. They do not yet query an
  OpenVINO build ID or live firmware build ID because the existing shim ABI exposes neither.
- Warm/cold NPU smoke tests and the pp512 resident-memory comparison require compatible hardware
  and must be repeated on the reconciled zero-copy tree.
- A bare `--no-default-features` build is not a supported baseline combination: the `505b7c1`
  module selection chooses the direct session when Level Zero is absent while the direct module is
  gated by `direct-ivpu`. Default/direct and no-default/Level-Zero combinations are checked.

## Reconciliation with post-`505b7c1` zero-copy work

The active tree overlaps this work in `src/host/linux/process.rs`, `src/metrics.rs`,
`src/vendor/ivpu/compile.rs`, `src/vendor/ivpu/profile.rs`, and
`src/vendor/ivpu/session.rs`. The last two compiler/runtime files should be merged
semantically, not accepted wholesale:

1. Rebase or apply the saved patch after the zero-copy branch is stable.
2. Preserve zero-copy buffer identity/lifetime changes in compiler keys and T0 bindings.
3. Reapply the pre-requant raw-constant semantic key and the `get_or_compile` wrapper around the
   final compiler entry points.
4. Port the T0 `PinnedProgram`, loading single-flight, reservation accounting, and out-of-lock
   destruction around the final zero-copy `PreparedInference` ownership model.
5. Keep the zero-copy host-memory/cgroup implementation, then add the stable one-time budget
   publication and constrained-available-memory method.
6. Resolve metric enum/profile-counter additions additively.
7. Run formatting, default/direct/Level-Zero checks, cache tests, ignored cross-process tests, then
   cold/warm hardware smoke tests with profiling enabled.
