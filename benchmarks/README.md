# Benchmark artifacts

`run_lfm_baseline.sh` captures repeatable CPU, Vulkan, and opt-in NPU baselines for an LFM GGUF.
Each run creates a timestamped directory containing:

- benchmark stdout and stderr (including NPU trace/profile output);
- `/usr/bin/time -v` peak-RSS and wall-time data;
- kernel, memory, model, backend, and timestamp metadata;
- the MemAvailable guard result for NPU runs.

Example:

```sh
MODEL=/path/LFM2.5-8B-A1B-Q4_K_M.gguf MODE=cpu ./benchmarks/run_lfm_baseline.sh
MODEL=/path/LFM2.5-8B-A1B-Q4_K_M.gguf MODE=vulkan ./benchmarks/run_lfm_baseline.sh
RUN_NPU=1 MODEL=/path/LFM2.5-8B-A1B-Q4_K_M.gguf MODE=npu NGL=20 \
  ./benchmarks/run_lfm_baseline.sh
RUN_NPU=1 MODEL=/path/LFM2.5-8B-A1B-Q4_K_M.gguf MODE=npu ATTENTION_PLACEMENT=full \
  ./benchmarks/run_lfm_baseline.sh
```

Use the Vulkan run for all-GPU attention, the default NPU run for NPU constant-weight/QKV
projections plus GPU attention, and `ATTENTION_PLACEMENT=full` for the experimental runtime
KQ/softmax/KQV path. Compare trace transfer boundaries and profile synchronization time, not only
aggregate tokens/s.

NPU execution is never enabled implicitly. Review placement and keep a reboot path available.
Commit representative result directories when changing lowering, quantization, cache, or scheduling
policy so numerical and performance claims remain externally auditable.

`run_autotune.sh` sweeps quant width, attention partition, and configured GPU/NPU layer boundaries.
`pareto.py` emits CSV and marks configurations that are not dominated across pp, tg, peak RSS, and
energy per generated token. Feed the selected work floor back through
`GGML_NPU_MIN_OFFLOAD_WORK`; retain the raw result directories with any policy change.
