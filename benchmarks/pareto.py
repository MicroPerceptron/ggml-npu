#!/usr/bin/env python3
"""Summarize committed benchmark runs and flag the balanced Pareto frontier."""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


def first(pattern: str, text: str, default: float = 0.0) -> float:
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return float(match.group(1)) if match else default


def load(run: Path) -> dict[str, object]:
    out = (run / "stdout.txt").read_text(errors="replace")
    err = (run / "stderr.txt").read_text(errors="replace")
    env = dict(
        line.split("=", 1)
        for line in (run / "environment.txt").read_text(errors="replace").splitlines()
        if "=" in line
    )
    # llama-bench emits pp/tg rows ending in a throughput number; accept both its table and
    # explicit "tokens/s" variants so historical result formats remain analyzable.
    rates = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:±[^|]*)?\|\s*$", out, re.MULTILINE)]
    pp = rates[-2] if len(rates) >= 2 else first(r"pp[^\n]*?([0-9]+(?:\.[0-9]+)?)\s*t/s", out)
    tg = rates[-1] if rates else first(r"tg[^\n]*?([0-9]+(?:\.[0-9]+)?)\s*t/s", out)
    rss_kb = first(r"Maximum resident set size \(kbytes\):\s*(\d+)", err)
    energy_uj = float((run / "energy_uj.txt").read_text()) if (run / "energy_uj.txt").exists() else 0.0
    prompt_tokens = int(env.get("prompt_tokens") or first(r"\s-p\s+(\d+)", err))
    generated_tokens = int(env.get("generated_tokens") or first(r"\s-n\s+(\d+)", err))
    return {
        "run": run.name,
        "mode": env.get("mode", ""),
        "attention": env.get("attention_placement", ""),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "pp_tps": pp,
        "tg_tps": tg,
        "rss_mib": rss_kb / 1024.0,
        "energy_j": energy_uj / 1_000_000.0,
        "joules_per_generated_token": energy_uj / 1_000_000.0 / 128.0 if energy_uj else 0.0,
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
    runs = [load(p) for p in sorted(root.iterdir()) if p.is_dir() and (p / "environment.txt").exists()]
    runs = [row for row in runs if float(row["pp_tps"]) > 0 and float(row["tg_tps"]) > 0]
    for row in runs:
        row["pareto"] = not any(dominates(other, row) for other in runs if other is not row)
    fields = list(runs[0]) if runs else ["run", "pareto"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
