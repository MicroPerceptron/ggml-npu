# Benchmarking framework

## Host microbenchmarks: Criterion

Criterion is enabled only by the `bench` feature and is absent from default and direct-IVPU
runtime builds:

```sh
cargo bench --features bench --bench hashing
cargo bench --features bench --bench runtime
# Fast smoke sample:
cargo bench --features bench --bench hashing -- --quick
```

`hashing` covers framed identity metadata, Graph IR-sized metadata, 5 MiB constants, and bounded
routed-MoE-style chunk streams. `runtime` covers exact-key encoding/cache lookup, Goblin ELF
parsing, and representative LFM operation-contract metadata. Benchmark-local FNV exists only as a
historical comparison; production identities use BLAKE3-256.

BLAKE3 uses `default-features = false, features = ["pure"]`. On x86/x86_64, `pure` enables the
crate's Rust SSE2, SSE4.1, and AVX2 modules with runtime `cpufeatures` selection and disables
C/assembly plus AVX-512. On little-endian AArch64, `pure` disables the crate's C NEON backend and
falls back to portable Rust. This is an explicit no-C/build-host-coupling tradeoff; change it only
after cross-target Criterion evidence. Do not wrap BLAKE3 compression in portable_simd or pulp.

Criterion output is generated under `target/criterion/`, already excluded by `/target`.

## End-to-end model benchmarks: llama-bench

`benchmarks/run_lfm_baseline.sh` runs llama-bench in a fresh process and captures:

- native llama-bench JSONL statistics;
- environment and source provenance;
- structured process wall/RSS data;
- optional energy;
- timeout, process-group cleanup, and an NPU MemAvailable guard.

All timed full-model and cache/quant microbenchmarks on a shared host use the same exclusive lock:

```sh
flock -x /tmp/ggml-npu-bench-global.lock \
  taskset -c 0-7 env GGML_NPU_BENCH_LOCK=/tmp/ggml-npu-bench-global.lock \
  THREADS=8 ./benchmarks/run_lfm_baseline.sh
```

The harness records the lock path, inherited CPU affinity, start/end load average, memory pressure,
RSS, swap, and device memory. A run that overlaps another benchmark is invalid for comparison even
if it completes; builds and correctness tests do not require the benchmark lock.

llama-bench remains authoritative for pp/tg model measurements. Criterion numbers are host
microbenchmarks and cannot be compared directly with token throughput, latency, energy, or model
quality.

## Guarded hardware benchmarks and checks

`bench-hardware` includes Criterion and `direct-ivpu`, but remains disabled by default. It exposes
ignored query/BO/CMDQ/direct-submit tests and one serial Criterion comparison using the proven
45,192-byte vector-add artifact:

```sh
export GGML_NPU_TEST_ARTIFACT=./artifacts/bounded-vector-add.blob
cargo test --features bench-hardware -- --ignored --test-threads=1 --nocapture
timeout --signal=TERM --kill-after=3s 60s \
  env CARGO_TARGET_DIR=target \
  cargo bench --bench hardware --features bench-hardware -- --noplot
```

The opt-in artifact path has no machine-specific default. `GGML_NPU_BENCH_BACKEND` can similarly
override the repository-relative release backend used by the Criterion fixture.

The hardware benchmark compares the same artifact, inputs, 32-element FP32 output, and two-second
device timeout for direct IVPU and Level Zero. Level Zero uses a reusable fixture: context, native
graph, initialized graph state, bound host buffers, queue, and closed execute command list persist
for warm iterations. Direct likewise retains its staged program, HPI, bindings, command BO, and
explicit queue. The measured scopes are:

- setup/load/bind/queue creation without inference execution;
- cold setup plus one execution;
- warm persistent submit plus wait, excluding output readback;
- warm persistent execution including 128-byte output readback;
- a host-only 128-byte copy control.

Each device scope uses flat Criterion sampling, 20 measured samples, a two-millisecond warm-up, and
a deliberately short requested measurement window. Slow scopes are automatically extended only
enough to collect one iteration per sample; warm scopes collected one or two iterations per sample.
Every device result is checked against the same Level Zero oracle before and after each group.
Errors are fatal and are never retried. Direct user BOs plus the reported preemption allocation
remain below 64 MiB.

### 2026-07-11 Lunar Lake sample

The values below are one bring-up run, with point estimate and 95% confidence interval. Criterion
raw samples, estimates, and reports are under `target/criterion/ivpu_tiny_*`.

- Setup/load/bind: direct mean 4.458 ms `[4.109, 4.762]`, median 4.818 ms
  `[4.404, 4.917]`; Level Zero mean 0.540 ms `[0.502, 0.586]`, median 0.505 ms
  `[0.495, 0.512]`.
- Cold setup plus execute: direct mean 13.857 ms `[13.671, 14.036]`, median 13.885 ms
  `[13.647, 14.182]`; Level Zero mean 19.141 ms `[15.069, 25.505]`, median 17.938 ms
  `[12.955, 19.352]`.
- Warm submit plus wait: direct mean 141.695 us `[137.279, 147.182]`, median 137.795 us
  `[133.848, 141.590]`; Level Zero mean 297.167 us `[196.841, 411.230]`, median 168.255 us
  `[163.188, 194.617]`.
- Warm execute plus readback: direct mean 150.176 us `[145.729, 155.929]`, median 147.816 us
  `[143.059, 150.626]`; Level Zero mean 225.992 us `[155.937, 311.903]`, median 145.771 us
  `[139.417, 154.531]`.
- Host 128-byte copy control: mean 5.952 ns `[5.866, 6.062]`, median 5.854 ns
  `[5.831, 5.911]`.

Level Zero had several severe high outliers, so its mean confidence ranges are wide; the medians
show the warm readback paths effectively tied in this run. This is tiny-artifact latency, not model
throughput, energy efficiency, or evidence about LFM placement.

The steady-state resource gate observed 28 FDs before and after, RSS 160,060 to 160,388 KiB,
process swap 0 KiB, and IVPU memory 68,714,496 bytes before and after. At 100 clock ticks/second,
the measured groups used about 0.08 CPU-seconds user and 1.11 CPU-seconds system. Device busy time
advanced by 45,995 us. Kernel logs gained no reset, MMU, firmware, OOM, or hang entry and the device
returned to D3hot.

## Pareto analysis

`benchmarks/pareto.py` consumes llama-bench `result.jsonl`, `process.json`, optional `energy.json`,
and provenance metadata. It emits JSONL and adds only multi-objective dominance analysis. Legacy
regex parsing of markdown and GNU-time stderr has been removed.
