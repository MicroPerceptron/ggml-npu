# Benchmark artifacts

`run_lfm_baseline.sh` captures repeatable native CPU, BLAS, Vulkan, and opt-in NPU baselines for an LFM GGUF.
Each run creates a timestamped directory containing:

- native llama-bench `result.jsonl` and diagnostic stderr;
- structured GNU-time `process.json` peak-RSS and wall-time data;
- kernel, memory, model, backend, and timestamp metadata;
- the MemAvailable guard result for NPU runs.

Example:

```sh
./benchmarks/build_cpu_baselines.sh all
MODEL=/path/LFM2.5-8B-A1B-Q4_K_M.gguf MODE=cpu-native ./benchmarks/run_lfm_baseline.sh
MODEL=/path/LFM2.5-8B-A1B-Q4_K_M.gguf MODE=cpu-flexiblas ./benchmarks/run_lfm_baseline.sh
MODEL=/path/LFM2.5-8B-A1B-Q4_K_M.gguf MODE=cpu-mkl ./benchmarks/run_lfm_baseline.sh
MODEL=/path/LFM2.5-8B-A1B-Q4_K_M.gguf MODE=vulkan ./benchmarks/run_lfm_baseline.sh
RUN_NPU=1 MODEL=/path/LFM2.5-8B-A1B-Q4_K_M.gguf MODE=npu NGL=20 \
  ./benchmarks/run_lfm_baseline.sh
RUN_NPU=1 MODEL=/path/LFM2.5-8B-A1B-Q4_K_M.gguf MODE=npu ATTENTION_PLACEMENT=full \
  ./benchmarks/run_lfm_baseline.sh
RUN_NPU=1 MODEL=/path/LFM2.5-8B-A1B-Q4_K_M.gguf MODE=npu FLASH_ATTENTION=on \
  ./benchmarks/run_lfm_baseline.sh
```

`build_cpu_baselines.sh` creates isolated `llama.cpp/build-cpu-{native,openblas,flexiblas,mkl}`
trees and builds only `llama-bench`. A missing development dependency is reported and skipped; the
script never installs packages. OpenBLAS requires `openblas.pc` or `openblas64.pc`, FlexiBLAS
requires `flexiblas_api.pc`, and oneMKL uses the existing oneAPI compilers plus `mkl-sdl.pc`.
`run_lfm_baseline.sh` clears inherited `GGML_BACKEND_PATH` and every `GGML_NPU_*` control used by
this backend before CPU/BLAS runs, so those baselines cannot silently load or schedule the NPU.

Use the Vulkan run for all-GPU attention, the default NPU run for NPU constant-weight/QKV
projections plus GPU attention, and `ATTENTION_PLACEMENT=full` for the experimental runtime
KQ/softmax/KQV path. Trace is an opt-in correctness/placement diagnostic (`NPU_TRACE=1`), not a
performance format. `GGML_NPU_PROFILE` remains available for submit/wait/copy crash forensics.

`FLASH_ATTENTION` accepts `on`, `off`, or `auto` and is recorded in the environment metadata.
Compare flash-attention modes in separate processes: persistent backend/compiler caches can make a
comma-separated multi-configuration `llama-bench` run order-dependent.

NPU execution is never enabled implicitly. Review placement and keep a reboot path available.
NPU runs are terminated after `NPU_TIMEOUT=180s` by default. Timeout, interrupt, and memory-guard
cleanup kill the complete process group. The default MemAvailable floor is 4 GiB after a
wrapper-only guard previously allowed an orphaned process to exhaust the host.
Commit representative result directories when changing lowering, quantization, cache, or scheduling
policy so numerical and performance claims remain externally auditable.

`pareto.py` reads structured JSONL/process/energy files, emits JSONL, and marks configurations that
are not dominated across pp, tg, peak RSS, and energy per generated token. It skips legacy text
runs rather than guessing with regexes.

For Criterion host microbenchmarks and the disabled hardware gate, see
[`docs/benchmarking.md`](../docs/benchmarking.md).
