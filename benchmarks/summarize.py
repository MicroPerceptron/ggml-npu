#!/usr/bin/env python3
"""Normalize llama-bench result directories into text and JSON summaries."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PROFILE_DISPATCHES = re.compile(r"\[ggml-npu profile\]\s+dispatches=(\d+)")
DIRECT_RESIDENT = re.compile(r"direct memory: resident\s+([0-9]+(?:\.[0-9]+)?)\s+MiB")
RATE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:±\s*([0-9]+(?:\.[0-9]+)?))?")
WORKLOAD = re.compile(r"^(pp|tg)(\d+)$")
ABSOLUTE_FILE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/)(?:[^/\\\s]+[/\\])+[^/\\\s]+")
PERF_PREFIX = "[ggml-perf] "


def portable_run_path(run: Path) -> str:
    """Use a repository-relative path, or only the run name for external inputs."""
    try:
        return run.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return run.name


def portable_filename(value: object) -> str | None:
    """Discard host-specific parent directories while preserving the artifact name."""
    if value is None:
        return None
    text = str(value).replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1] or None


def portable_diagnostic(value: str | None, private_paths: tuple[object, ...] = ()) -> str | None:
    """Remove absolute input locations from a diagnostic retained in a summary."""
    if value is None:
        return None
    text = value
    for path in private_paths:
        if path:
            text = text.replace(str(path), portable_filename(path) or "<path>")
    return ABSOLUTE_FILE_PATH.sub(
        lambda match: portable_filename(match.group(0)) or "<path>",
        text,
    )


def read_environment(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        key: value
        for line in path.read_text(errors="replace").splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def read_json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, ""
    text = path.read_text(errors="replace")
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, text
    return None, text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def read_perf_events(stderr: str) -> list[dict[str, Any]]:
    events = []
    for line in stderr.splitlines():
        if PERF_PREFIX not in line:
            continue
        payload = line.split(PERF_PREFIX, 1)[1]
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == 1:
            events.append(value)
    return events


def normalize_pass_metrics(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attribute ordered scheduler/NPU events to explicit llama-bench pass boundaries."""
    passes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for event in events:
        kind = event.get("event")
        if event.get("source") == "llama-bench" and kind == "pass_begin":
            current = {
                "phase": event.get("phase"),
                "index": event.get("index"),
                "prompt_tokens": event.get("prompt_tokens"),
                "dispatches": 0,
                "scheduler_splits": 0,
                "scheduler_nodes": 0,
                "resident_hits": 0,
                "cold_misses": 0,
                "rehydrations": 0,
                "evicted_bytes": 0,
                "direct_ns": 0,
                "scheduler_backends": {},
                "scheduler_backend_nodes": {},
                "scheduler_backend_ns": {},
                "scheduler_handoff_ns": 0,
                "program_phase_ns": {},
                "artifact_loads": {},
                "max_artifact_bytes": 0,
                "constant_source_bytes": 0,
                "constant_canonical_bytes": 0,
                "constant_canonicalization_ns": 0,
                "program_node_histogram": {},
                "placement_rejections": {},
                "_split_backends": {},
            }
            continue
        if current is None:
            continue
        if event.get("source") == "ggml-npu" and kind == "dispatch":
            current["dispatches"] += 1
            current["direct_ns"] += sum(
                int(event.get(field, 0) or 0)
                for field in ("stage_in_ns", "submit_ns", "wait_ns", "copy_out_ns")
            )
        elif event.get("source") == "ggml-scheduler" and kind == "scheduler_split_begin":
            current["scheduler_splits"] += 1
            current["scheduler_nodes"] += int(event.get("nodes", 0) or 0)
            backend = str(event.get("backend") or event.get("backend_id") or "unknown")
            split = str(event.get("split", current["scheduler_splits"] - 1))
            current["_split_backends"][split] = backend
            current["scheduler_backends"][backend] = (
                current["scheduler_backends"].get(backend, 0) + 1
            )
            current["scheduler_backend_nodes"][backend] = (
                current["scheduler_backend_nodes"].get(backend, 0)
                + int(event.get("nodes", 0) or 0)
            )
        elif event.get("source") == "ggml-scheduler" and kind == "scheduler_split_end":
            split = str(event.get("split", ""))
            backend = current["_split_backends"].get(
                split, str(event.get("backend_id") or "unknown")
            )
            current["scheduler_backend_ns"][backend] = (
                current["scheduler_backend_ns"].get(backend, 0)
                + int(event.get("elapsed_ns", 0) or 0)
            )
            current["scheduler_handoff_ns"] += int(
                event.get("handoff_begin_ns", 0) or 0
            ) + int(event.get("handoff_end_ns", 0) or 0)
        elif event.get("source") == "ggml-npu" and kind == "program_cache":
            outcome = event.get("outcome")
            if outcome == "resident_hit":
                current["resident_hits"] += 1
            elif outcome == "cold_miss":
                current["cold_misses"] += 1
            elif outcome == "rehydration":
                current["rehydrations"] += 1
        elif event.get("source") == "ggml-npu" and kind == "program_eviction":
            current["evicted_bytes"] += int(event.get("bytes", 0) or 0)
        elif event.get("source") == "ggml-npu" and kind == "program_phase":
            phase = str(event.get("phase") or "unknown")
            current["program_phase_ns"][phase] = (
                current["program_phase_ns"].get(phase, 0)
                + int(event.get("elapsed_ns", 0) or 0)
            )
        elif event.get("source") == "ggml-npu" and kind == "constant_canonicalization":
            current["constant_source_bytes"] += int(event.get("source_bytes", 0) or 0)
            current["constant_canonical_bytes"] += int(
                event.get("canonical_bytes", 0) or 0
            )
            current["constant_canonicalization_ns"] += int(
                event.get("requant_ns", 0) or 0
            )
        elif event.get("source") == "ggml-npu" and kind == "compiler_artifact":
            current["max_artifact_bytes"] = max(
                current["max_artifact_bytes"], int(event.get("elf_bytes", 0) or 0)
            )
        elif event.get("source") == "ggml-npu-compiler" and kind == "artifact_load":
            outcome = str(event.get("outcome") or "unknown")
            current["artifact_loads"][outcome] = (
                current["artifact_loads"].get(outcome, 0) + 1
            )
        elif event.get("source") == "ggml-npu" and kind == "graph_partition":
            nodes = str(event.get("nodes", 0))
            current["program_node_histogram"][nodes] = (
                current["program_node_histogram"].get(nodes, 0) + 1
            )
        elif event.get("source") == "ggml-npu" and kind == "placement_rejection":
            reason = str(event.get("reason") or "unknown")
            current["placement_rejections"][reason] = (
                current["placement_rejections"].get(reason, 0) + 1
            )
        elif event.get("source") == "llama-bench" and kind == "pass_end":
            current["elapsed_ns"] = int(event.get("elapsed_ns", 0) or 0)
            elapsed = current["elapsed_ns"]
            current["unattributed_ns"] = max(0, elapsed - current["direct_ns"])
            current.pop("_split_backends", None)
            passes.append(current)
            current = None
    return passes


def table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def read_legacy_llama_table(path: Path) -> list[dict[str, Any]]:
    """Read llama-bench's own table output without interpreting GNU-time text."""
    if not path.exists():
        return []
    headers: list[str] | None = None
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = table_cells(line)
        if "test" in cells and "t/s" in cells:
            headers = cells
            continue
        if headers is None or len(cells) != len(headers):
            continue
        values = dict(zip(headers, cells))
        match = WORKLOAD.match(values.get("test", ""))
        rate = RATE.fullmatch(values.get("t/s", ""))
        if not match or not rate:
            continue
        kind, tokens = match.groups()
        rows.append(
            {
                "model_type": values.get("model") or None,
                "legacy_backend": values.get("backend") or None,
                "legacy_fa": values.get("fa") or None,
                "n_prompt": int(tokens) if kind == "pp" else 0,
                "n_gen": int(tokens) if kind == "tg" else 0,
                "avg_ts": float(rate.group(1)),
                "stddev_ts": float(rate.group(2)) if rate.group(2) else None,
                "_source_format": "llama-bench-table",
            }
        )
    return rows


def parse_failure(stderr: str, process_text: str, guard: str) -> str | None:
    if guard.strip():
        return guard.strip().splitlines()[-1]
    direct_candidates = []
    general_candidates = []
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("[npu direct]") and any(
            word in stripped
            for word in ("failed", "rejected", "unsupported", "mismatch", "no artifact")
        ):
            direct_candidates.append(stripped)
        elif (
            "llama_bench: error:" in stripped
            or "failed to decode" in stripped
            or "MemAvailable guard" in stripped
        ):
            general_candidates.append(stripped)
    if direct_candidates:
        return direct_candidates[-1]
    if general_candidates:
        return general_candidates[-1]
    for line in process_text.splitlines():
        if line.startswith(("Command exited", "Command terminated")):
            return line
    return None


def execution_path(mode: str, env: dict[str, str], stderr: str) -> dict[str, Any]:
    profile = PROFILE_DISPATCHES.search(stderr)
    profile_dispatches = int(profile.group(1)) if profile else 0
    debug_dispatches = stderr.count("[npu direct] dispatch result=true")
    resident = DIRECT_RESIDENT.search(stderr)
    direct_profile = bool(resident and float(resident.group(1)) > 0)
    direct = debug_dispatches > 0 or (profile_dispatches > 0 and direct_profile)
    level_zero = not direct and bool(re.search(r"level[_ -]?zero", stderr, re.IGNORECASE))

    if mode.startswith("cpu"):
        runtime = mode
    elif mode == "vulkan":
        runtime = "vulkan"
    elif direct:
        runtime = "direct-ivpu"
    elif level_zero:
        runtime = "level-zero"
    elif mode == "npu":
        runtime = "npu-unproven"
    else:
        runtime = mode or "unknown"

    fallback = None
    if mode == "npu" and env.get("npu_only", "0") != "1":
        fallback = "scheduler fallback allowed"
        if env.get("npu_moe", "0") != "1":
            fallback += "; direct MoE disabled"

    proof = []
    if mode.startswith("cpu"):
        proof.append(f"mode={mode}")
    elif mode == "vulkan":
        proof.append("mode=vulkan")
    if profile_dispatches:
        proof.append(f"profile dispatches={profile_dispatches}")
    if debug_dispatches:
        proof.append(f"debug successful dispatches={debug_dispatches}")
    if direct_profile:
        proof.append("direct memory profile")
    if level_zero:
        proof.append("Level Zero diagnostic")

    compiler = None
    if mode == "npu" and env.get("compiler_shim_sha256"):
        compiler = "OpenVINO native-artifact compiler; runtime submission classified separately"
    return {
        "runtime": runtime,
        "proven_direct": direct,
        "direct_dispatches": profile_dispatches or debug_dispatches or None,
        "compiler": compiler,
        "fallback": fallback,
        "proof": proof,
    }


def baseline_eligible(execution: dict[str, Any], status: str = "success") -> bool:
    """NPU baselines require submission proof tags; unproven NPU runs are never baseline-eligible."""
    if status != "success":
        return False
    runtime = execution.get("runtime") or ""
    proof = execution.get("proof") or []
    if runtime.startswith("cpu") or runtime == "vulkan":
        return bool(proof)
    if runtime == "direct-ivpu":
        return bool(execution.get("proven_direct")) and bool(proof)
    if runtime == "level-zero":
        return bool(proof)
    # npu-unproven and unknown runtimes cannot be cited as NPU baselines.
    return False


def workload(row: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    prompt = int(row.get("n_prompt", env.get("prompt_tokens", 0)) or 0)
    generated = int(row.get("n_gen", env.get("generated_tokens", 0)) or 0)
    if prompt and not generated:
        kind, label = "pp", f"pp{prompt}"
    elif generated and not prompt:
        kind, label = "tg", f"tg{generated}"
    else:
        kind, label = "mixed", f"pp{prompt}+tg{generated}"
    flash = row.get("flash_attn")
    if flash is None:
        flash = row.get("legacy_fa", env.get("flash_attention"))
    flash = {
        -1: "auto",
        0: "off",
        1: "on",
        "-1": "auto",
        "0": "off",
        "1": "on",
    }.get(flash, flash)
    return {
        "kind": kind,
        "label": label,
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "repetitions": int(env["repetitions"]) if env.get("repetitions", "").isdigit() else None,
        "threads": int(env["threads"]) if env.get("threads", "").isdigit() else None,
        "flash_attention": flash,
        "attention_placement": env.get("attention_placement") or None,
    }


def normalize_run(run: Path) -> list[dict[str, Any]]:
    env = read_environment(run / "environment.txt")
    stderr = (run / "stderr.txt").read_text(errors="replace") if (run / "stderr.txt").exists() else ""
    guard = (run / "guard.txt").read_text(errors="replace") if (run / "guard.txt").exists() else ""
    process, process_text = read_json_object(run / "process.json")
    resources, _ = read_json_object(run / "resources.json")
    energy, _ = read_json_object(run / "energy.json")
    rows = read_jsonl(run / "result.jsonl")
    source_format = "llama-bench-jsonl"
    if not rows:
        rows = read_legacy_llama_table(run / "stdout.txt")
        source_format = "llama-bench-table" if rows else "none"

    terminated = "Command terminated" in process_text
    exit_status = int(process["exit_status"]) if process and "exit_status" in process else None
    if rows and (exit_status in (None, 0)) and not terminated:
        status = "success"
    elif rows:
        status = "partial"
    elif guard.strip():
        status = "memory-guard"
    elif exit_status == 124:
        status = "timeout"
    elif terminated:
        status = "terminated"
    elif exit_status not in (None, 0):
        status = "failed"
    else:
        status = "incomplete"
    failure = (
        None
        if status == "success"
        else portable_diagnostic(
            parse_failure(stderr, process_text, guard),
            (env.get("model"), run, run.resolve()),
        )
    )

    mode = env.get("mode", "")
    execution = execution_path(mode, env, stderr)
    execution["baseline_eligible"] = baseline_eligible(execution, status)
    perf_passes = normalize_pass_metrics(read_perf_events(stderr))
    common_metrics = {
        "wall_seconds": float(process["wall_seconds"]) if process and "wall_seconds" in process else None,
        "user_seconds": float(process["user_seconds"]) if process and "user_seconds" in process else None,
        "system_seconds": float(process["system_seconds"]) if process and "system_seconds" in process else None,
        "peak_rss_bytes": (
            int(process["max_rss_kib"]) * 1024
            if process and "max_rss_kib" in process
            else int(resources["peak_llama_bench_rss_kib"]) * 1024
            if resources and "peak_llama_bench_rss_kib" in resources
            else None
        ),
        "peak_system_swap_used_bytes": (
            int(resources["peak_swap_used_kib"]) * 1024
            if resources and "peak_swap_used_kib" in resources
            else None
        ),
        "min_mem_available_bytes": (
            int(resources["min_mem_available_kib"]) * 1024
            if resources and "min_mem_available_kib" in resources
            else None
        ),
        "peak_npu_memory_bytes": (
            int(resources["peak_npu_memory_bytes"])
            if resources and "peak_npu_memory_bytes" in resources
            else None
        ),
        "passes": perf_passes,
        "memory_psi_some_delta_us": (
            int(resources["memory_psi_some_delta_us"])
            if resources and "memory_psi_some_delta_us" in resources
            else None
        ),
        "memory_psi_full_delta_us": (
            int(resources["memory_psi_full_delta_us"])
            if resources and "memory_psi_full_delta_us" in resources
            else None
        ),
        "energy_available": bool(energy and energy.get("energy_uj") is not None),
        "energy_joules": float(energy["energy_uj"]) / 1_000_000.0 if energy and "energy_uj" in energy else None,
    }

    if not rows:
        rows = [{}]
    normalized = []
    for row in rows:
        metrics = dict(common_metrics)
        metrics.update(
            {
                "throughput_tps": float(row["avg_ts"]) if row.get("avg_ts") is not None else None,
                "throughput_stddev_tps": (
                    float(row["stddev_ts"]) if row.get("stddev_ts") is not None else None
                ),
                "latency_mean_seconds": (
                    float(row["avg_ns"]) / 1_000_000_000.0 if row.get("avg_ns") is not None else None
                ),
                "latency_stddev_seconds": (
                    float(row["stddev_ns"]) / 1_000_000_000.0
                    if row.get("stddev_ns") is not None
                    else None
                ),
                "sample_count": len(row.get("samples_ns", [])) or None,
            }
        )
        normalized.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_dir": portable_run_path(run),
                "run": run.name,
                "timestamp_utc": env.get("timestamp_utc") or row.get("test_time"),
                "status": status,
                "failure_reason": failure,
                "source_format": row.get("_source_format", source_format),
                "mode": mode,
                "execution": execution,
                "model": {
                    "file": portable_filename(row.get("model_filename", env.get("model"))),
                    "type": row.get("model_type") or None,
                    "size_bytes": int(row["model_size"]) if row.get("model_size") is not None else None,
                    "parameters": int(row["model_n_params"]) if row.get("model_n_params") is not None else None,
                },
                "workload": workload(row, env),
                "metrics": metrics,
                "provenance": {
                    "backend_git_head": env.get("backend_git_head") or None,
                    "backend_sha256": env.get("backend_so_sha256") or None,
                    "compiler_shim_sha256": env.get("compiler_shim_sha256") or None,
                    "llama_bench_sha256": env.get("llama_bench_sha256") or None,
                    "llama_build_commit": row.get("build_commit"),
                    "llama_build_number": row.get("build_number"),
                },
            }
        )
    return normalized


def discover(paths: list[Path]) -> list[Path]:
    runs = []
    for path in paths:
        if (path / "environment.txt").exists():
            runs.append(path)
        elif path.is_dir():
            runs.extend(
                child
                for child in sorted(path.iterdir())
                if child.is_dir() and (child / "environment.txt").exists()
            )
    unique = {str(path.resolve()): path for path in runs}
    return [unique[key] for key in sorted(unique)]


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    gib = value / (1024**3)
    return f"{gib:.2f} GiB"


def format_number(value: float | None, suffix: str = "") -> str:
    return "unavailable" if value is None else f"{value:.3f}{suffix}"


def human_summary(records: list[dict[str, Any]]) -> str:
    lines = [f"Benchmark summary: {len(records)} measurement(s)"]
    for record in records:
        metrics = record["metrics"]
        execution = record["execution"]
        rate = format_number(metrics["throughput_tps"])
        if metrics["throughput_tps"] is not None:
            if metrics["throughput_stddev_tps"] is not None:
                rate += f" ± {metrics['throughput_stddev_tps']:.3f}"
            rate += " tok/s"
        latency = format_number(metrics["latency_mean_seconds"])
        if metrics["latency_mean_seconds"] is not None:
            if metrics["latency_stddev_seconds"] is not None:
                latency += f" ± {metrics['latency_stddev_seconds']:.3f}"
            latency += " s"
        path = execution["runtime"]
        if execution["fallback"]:
            path += f" ({execution['fallback']})"
        proof = "; ".join(execution["proof"]) or "no runtime submission proof"
        eligible = "yes" if execution.get("baseline_eligible") else "no"
        workload_config = record["workload"]
        lines.extend(
            [
                "",
                f"{record['run']} · {record['workload']['label']} · {record['status']}",
                f"  execution: {path}",
                f"  proof: {proof}",
                f"  baseline_eligible: {eligible}",
                *(
                    [f"  compiler: {execution['compiler']}"]
                    if execution["compiler"]
                    else []
                ),
                f"  model: {record['model']['type'] or record['model']['file'] or 'unavailable'}",
                (
                    "  workload: "
                    f"flash_attention={workload_config['flash_attention']}; "
                    f"attention={workload_config['attention_placement'] or 'unavailable'}; "
                    f"repetitions={workload_config['repetitions'] or 'unavailable'}; "
                    f"threads={workload_config['threads'] or 'unavailable'}"
                ),
                f"  throughput: {rate}",
                f"  latency: {latency}",
                (
                    "  resources: RSS "
                    f"{format_bytes(metrics['peak_rss_bytes'])}; system swap "
                    f"{format_bytes(metrics['peak_system_swap_used_bytes'])}; NPU "
                    f"{format_bytes(metrics['peak_npu_memory_bytes'])}; min available "
                    f"{format_bytes(metrics['min_mem_available_bytes'])}"
                ),
                (
                    f"  energy: {metrics['energy_joules']:.3f} J"
                    if metrics["energy_available"]
                    else "  energy: unavailable"
                ),
            ]
        )
        if record["failure_reason"]:
            lines.append(f"  failure: {record['failure_reason']}")
        lines.append(f"  source: {record['artifact_dir']}")
    return "\n".join(lines) + "\n"


def write_or_print(text: str, path: Path | None) -> None:
    if path:
        path.write_text(text)
    else:
        sys.stdout.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="run directories or parent directories containing runs",
    )
    parser.add_argument("--output", type=Path, help="write human-readable summary here")
    parser.add_argument("--json-out", type=Path, help="also write normalized JSON here")
    parser.add_argument(
        "--require-baseline-proof",
        action="store_true",
        help="fail if any successful measurement is not baseline-eligible (rejects npu-unproven)",
    )
    args = parser.parse_args()

    runs = discover(args.paths)
    if not runs:
        parser.error("no benchmark run directories found")
    records = [record for run in runs for record in normalize_run(run)]
    write_or_print(human_summary(records), args.output)
    if args.json_out:
        document = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurements": records,
        }
        args.json_out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    if args.require_baseline_proof:
        bad = [
            record
            for record in records
            if record.get("status") == "success"
            and not record.get("execution", {}).get("baseline_eligible")
        ]
        if bad:
            for record in bad:
                runtime = record.get("execution", {}).get("runtime")
                proof = record.get("execution", {}).get("proof") or []
                print(
                    f"baseline proof required: {record.get('run')} "
                    f"runtime={runtime} proof={proof or ['(none)']}",
                    file=sys.stderr,
                )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
