#!/usr/bin/env python3
"""Read structured llama-bench JSONL runs and emit a JSONL Pareto frontier."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(run: Path) -> dict[str, object]:
    result_path = run / "result.jsonl"
    if not result_path.exists():
        raise ValueError("legacy text result (rerun with structured benchmark tooling)")
    results = [
        json.loads(line)
        for line in result_path.read_text().splitlines()
        if line.strip()
    ]
    env = dict(
        line.split("=", 1)
        for line in (run / "environment.txt").read_text(errors="replace").splitlines()
        if "=" in line
    )
    pp_rows = [row for row in results if int(row.get("n_prompt", 0)) > 0 and int(row.get("n_gen", 0)) == 0]
    tg_rows = [row for row in results if int(row.get("n_gen", 0)) > 0 and int(row.get("n_prompt", 0)) == 0]
    pp = float(pp_rows[-1]["avg_ts"]) if pp_rows else 0.0
    tg = float(tg_rows[-1]["avg_ts"]) if tg_rows else 0.0
    process = json.loads((run / "process.json").read_text())
    energy = json.loads((run / "energy.json").read_text()) if (run / "energy.json").exists() else {}
    energy_uj = float(energy.get("energy_uj", 0))
    prompt_tokens = int(env["prompt_tokens"])
    generated_tokens = int(env["generated_tokens"])
    return {
        "run": run.name,
        "mode": env.get("mode", ""),
        "attention": env.get("attention_placement", ""),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "pp_tps": pp,
        "tg_tps": tg,
        "rss_mib": float(process["max_rss_kib"]) / 1024.0,
        "energy_j": energy_uj / 1_000_000.0,
        "joules_per_generated_token": (
            energy_uj / 1_000_000.0 / generated_tokens
            if energy_uj and generated_tokens
            else 0.0
        ),
    }


def dominates(a: dict[str, object], b: dict[str, object]) -> bool:
    # Higher throughput is better; lower RSS and energy/token are better. Missing metrics do not
    # dominate measured ones.
    if (a["prompt_tokens"], a["generated_tokens"]) != (b["prompt_tokens"], b["generated_tokens"]):
        return False
    maximize = ("pp_tps", "tg_tps")
    minimize = ("rss_mib", "joules_per_generated_token")
    comparable = False
    strictly = False
    for key in maximize:
        av, bv = float(a[key]), float(b[key])
        if av and bv:
            comparable = True
            if av < bv:
                return False
            strictly |= av > bv
    for key in minimize:
        av, bv = float(a[key]), float(b[key])
        if av and bv:
            comparable = True
            if av > bv:
                return False
            strictly |= av < bv
    return comparable and strictly


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "results")
    runs = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "environment.txt").exists():
            continue
        try:
            runs.append(load(path))
        except (ValueError, OSError, KeyError) as error:
            print(f"skip {path.name}: {error}", file=sys.stderr)
    runs = [row for row in runs if float(row["pp_tps"]) > 0 and float(row["tg_tps"]) > 0]
    for row in runs:
        row["pareto"] = not any(dominates(other, row) for other in runs if other is not row)
    for row in runs:
        print(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
