#!/usr/bin/env bash
# Hermetic claim ⊆ coverage gate (issue #12). No MODEL / pp512 / NPU device required.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MATRIX="$ROOT/tests/coverage_matrix.csv"
[[ -f "$MATRIX" ]] || { echo "missing $MATRIX" >&2; exit 2; }

echo "== claim coverage matrix =="
cargo test --lib coverage:: -- --nocapture

echo "== bench baseline proof classifier =="
python3 -m unittest discover -s "$ROOT/benchmarks" -p 'test_summarize.py' -v

echo "claim coverage gate OK"
