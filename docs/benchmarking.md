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

llama-bench remains authoritative for pp/tg model measurements. Criterion numbers are host
microbenchmarks and cannot be compared directly with token throughput, latency, energy, or model
quality.

## Guarded hardware checks

The `bench-hardware` feature only exposes ignored query, 4 KiB BO, CMDQ lifecycle, ELF staging, and
pre-submit dry-run tests:

```sh
cargo test --features bench-hardware -- --ignored --test-threads=1 --nocapture
```

These tests do not submit NPU commands. Run specific tests with a finite external timeout; never use
the aggregate command as authorization for future submit tests.

## Pareto analysis

`benchmarks/pareto.py` consumes llama-bench `result.jsonl`, `process.json`, optional `energy.json`,
and provenance metadata. It emits JSONL and adds only multi-objective dominance analysis. Legacy
regex parsing of markdown and GNU-time stderr has been removed.
