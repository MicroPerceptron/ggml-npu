# Benchmark artifacts

`run_lfm_baseline.sh` captures repeatable native CPU, BLAS, Vulkan, and opt-in NPU baselines for an LFM GGUF.
Each run creates a timestamped directory containing:

- benchmark stdout and stderr (including NPU trace/profile output);
- `/usr/bin/time -v` peak-RSS and wall-time data;
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
KQ/softmax/KQV path. Compare trace transfer boundaries and profile synchronization time, not only
aggregate tokens/s.

`FLASH_ATTENTION` accepts `on`, `off`, or `auto` and is recorded in the environment metadata.
Compare flash-attention modes in separate processes: persistent backend/compiler caches can make a
comma-separated multi-configuration `llama-bench` run order-dependent.

NPU execution is never enabled implicitly. Review placement and keep a reboot path available.
NPU runs are terminated after `NPU_TIMEOUT=180s` by default (with a ten-second kill grace) and
retain the existing MemAvailable guard; raise the timeout only for a deliberate cold-compile study.
Commit representative result directories when changing lowering, quantization, cache, or scheduling
policy so numerical and performance claims remain externally auditable.

`run_autotune.sh` sweeps quant width, attention partition, and configured GPU/NPU layer boundaries.
`pareto.py` emits CSV and marks configurations that are not dominated across pp, tg, peak RSS, and
energy per generated token. Feed the selected work floor back through
`GGML_NPU_MIN_OFFLOAD_WORK`; retain the raw result directories with any policy change.
