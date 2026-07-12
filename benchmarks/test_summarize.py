#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import summarize  # noqa: E402


class SummaryTests(unittest.TestCase):
    def make_run(self, root: Path, name: str, env: str) -> Path:
        run = root / name
        run.mkdir()
        (run / "environment.txt").write_text(env)
        return run

    def test_normalizes_proven_direct_run_and_missing_energy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(
                Path(directory),
                "direct",
                "mode=npu\nprompt_tokens=512\ngenerated_tokens=0\nrepetitions=3\n"
                "threads=8\nnpu_only=0\nnpu_moe=0\ncompiler_shim_sha256=abc\n",
            )
            row = {
                "model_type": "fixture",
                "n_prompt": 512,
                "n_gen": 0,
                "avg_ts": 8.25,
                "stddev_ts": 0.15,
                "avg_ns": 62_000_000_000,
                "stddev_ns": 1_000_000_000,
                "samples_ns": [61, 62, 63],
            }
            (run / "result.jsonl").write_text(json.dumps(row) + "\n")
            (run / "process.json").write_text(
                '{"wall_seconds":248.0,"max_rss_kib":1024,"exit_status":0}\n'
            )
            (run / "resources.json").write_text(
                '{"peak_swap_used_kib":2048,"min_mem_available_kib":4096,'
                '"peak_npu_memory_bytes":8192,"memory_psi_some_delta_us":10,'
                '"memory_psi_full_delta_us":5}\n'
            )
            (run / "stderr.txt").write_text(
                "[ggml-npu profile]  dispatches=632\n"
                " direct memory: resident 1 MiB\n"
            )

            record = summarize.normalize_run(run)[0]
            self.assertEqual(record["status"], "success")
            self.assertEqual(record["execution"]["runtime"], "direct-ivpu")
            self.assertEqual(record["execution"]["direct_dispatches"], 632)
            self.assertIn("direct MoE disabled", record["execution"]["fallback"])
            self.assertEqual(record["metrics"]["peak_rss_bytes"], 1024 * 1024)
            self.assertFalse(record["metrics"]["energy_available"])
            self.assertIsNone(record["metrics"]["energy_joules"])

    def test_reports_timeout_failure_without_fabricating_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(
                Path(directory),
                "timeout",
                "mode=npu\nprompt_tokens=16\ngenerated_tokens=0\n",
            )
            (run / "result.jsonl").write_text("")
            (run / "process.json").write_text(
                'Command exited with non-zero status 124\n'
                '{"wall_seconds":300.0,"max_rss_kib":10,"exit_status":124}\n'
            )
            (run / "stderr.txt").write_text(
                "[npu direct] output 0 storage mismatch: required=2 available=1\n"
                "llama_bench: error: failed to run prompt warmup\n"
            )

            record = summarize.normalize_run(run)[0]
            self.assertEqual(record["status"], "timeout")
            self.assertIn("storage mismatch", record["failure_reason"])
            self.assertIsNone(record["metrics"]["throughput_tps"])

    def test_zero_direct_profile_does_not_prove_direct_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(
                Path(directory),
                "unproven",
                "mode=npu\nprompt_tokens=1\ngenerated_tokens=0\n",
            )
            (run / "result.jsonl").write_text('{"n_prompt":1,"n_gen":0,"avg_ts":1}\n')
            (run / "process.json").write_text('{"exit_status":0}\n')
            (run / "stderr.txt").write_text(
                "[ggml-npu profile]  dispatches=2\n"
                " direct memory: resident 0.00 MiB\n"
            )

            record = summarize.normalize_run(run)[0]
            self.assertEqual(record["execution"]["runtime"], "npu-unproven")
            self.assertFalse(record["execution"]["proven_direct"])

    def test_reads_llama_bench_legacy_table_as_explicit_legacy_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(
                Path(directory),
                "vulkan",
                "mode=vulkan\nprompt_tokens=512\ngenerated_tokens=128\nrepetitions=3\n",
            )
            (run / "stdout.txt").write_text(
                "| model | backend | fa | test | t/s |\n"
                "| --- | --- | --: | ---: | ---: |\n"
                "| fixture | Vulkan | 1 | pp512 | 843.51 ± 9.41 |\n"
                "| fixture | Vulkan | 1 | tg128 | 62.32 ± 0.96 |\n"
            )

            records = summarize.normalize_run(run)
            self.assertEqual([record["workload"]["label"] for record in records], ["pp512", "tg128"])
            self.assertEqual(records[0]["source_format"], "llama-bench-table")
            self.assertEqual(records[0]["metrics"]["throughput_tps"], 843.51)
            self.assertIsNone(records[0]["metrics"]["latency_mean_seconds"])
            self.assertIsNone(records[0]["metrics"]["peak_rss_bytes"])


if __name__ == "__main__":
    unittest.main()
