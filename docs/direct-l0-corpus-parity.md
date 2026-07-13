# Direct ↔ Level Zero corpus parity

Lab / hermetic gate for the **model-corpus Direct↔L0 binding + numeric parity**
retirement checklist item ([`retire-level-zero-compat.md`](retire-level-zero-compat.md),
GitHub [#63](https://github.com/MicroPerceptron/ggml-npu/issues/63)). Prefer this over a
full locked LFM pp512 when proving artifact-level parity.

## What is compared

Manifest: [`tests/direct_l0_corpus/manifest.csv`](../tests/direct_l0_corpus/manifest.csv).

| Check | Scope | Pass / fail |
| --- | --- | --- |
| Identity | SHA-256 + byte length per artifact | Exact |
| Structural fingerprint (Direct ELF/HPI) | inputs/outputs, logical + allocation bytes, sections/symbols/relocations, resident + placement bytes, link/JIT ops, loader **1.2.2**, NNRT **11.4.10**, CMX slice/barriers | Exact vs golden |
| Binding meta Direct ↔ L0 | Level Zero `pfnGetArgumentProperties` byte sizes vs Direct binding logical bytes | Exact |
| Numeric (FP32 entries only) | Direct submit/readback vs L0 execute | `compare_f32` abs/rel ≤ `1e-5` |

Current corpus (small, representative — not full LFM):

- `vector-add-tiny` — bounded bring-up KAT (`numeric_fp32`)
- `lfm-n1-small`, `lfm-shortconv-e2e`, `lfm-batched-get-rows`, `lfm-n3-slice`, `lfm-n8-slice` —
  LFM / coverage cache slices (`structural` + L0 binding probe)

Blobs are **not** committed (often gitignored lab cache). Resolve via
`GGML_NPU_CORPUS_DIR`, `~/.cache/ggml-npu/`, or `benchmarks/results/lfm-pp512-cache/`.

## How to run

Hermetic (no device; goldens optional if blobs absent):

```sh
export GGML_DIR=/path/to/llama.cpp
cargo test --lib --features direct-ivpu \
  vendor::ivpu::direct::corpus::tests:: \
  vendor::ivpu::direct::oracle::tests::
```

Lab gate (requires Intel NPU + Level Zero + all manifest blobs). Uses `THREADS=6` as the
documented lab default (unused by this artifact gate; kept for bench-machine consistency):

```sh
export GGML_DIR=/path/to/llama.cpp
export GGML_NPU_CORPUS_DIR=/path/to/dir   # contains every manifest artifact name
THREADS=6 ./benchmarks/run_direct_l0_corpus_parity.sh
```

Refresh structural goldens after intentional compiler/artifact churn:

```sh
GGML_NPU_CORPUS_DIR=... GGML_NPU_CORPUS_DUMP=1 \
  cargo test --lib --features direct-ivpu \
  vendor::ivpu::direct::corpus::tests::dump_fingerprints_for_lab_fill \
  -- --nocapture --exact
```

## Gaps (gate still open for L0 delete)

- Full LFM program set / warm-resident accounting under claim-stage is not covered here.
- Non-FP32 I/O slices are binding/structural only until a typed numeric path exists.
- Import/copy / shared-allocation duty-cycle remains under #6 / claim-stage docs.
- Hang / in-flight abort, coverage leftovers (#9/#10/#11), and product default-on knobs
  remain separate delete blockers.
