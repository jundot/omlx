#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare Clef GPU and single-ANE inference using the serving engine.

Run from the checkout: python -m scripts.benchmark_decision_ane MODEL_DIRECTORY
Prints a Markdown summary of synthetic requests; writes no benchmark files.
Unload other models first to avoid duplicate model residency.
"""

import argparse
import asyncio
import statistics
import time
from pathlib import Path


def request(repetitions: int) -> dict:
    return {
        "state": {
            "invoice": {"status": "overdue", "total": 1250},
            "message": "Please refund the duplicate payment today.",
            "archive": "Archived entry has no bearing on the current invoice. "
            * repetitions,
        },
        "questions": {
            "status": {
                "type": "choice",
                "instructions": "What is the invoice status?",
                "criteria": {
                    "paid": "Invoice has been paid",
                    "overdue": "Invoice is overdue",
                    "draft": "Invoice is a draft",
                },
            },
            "refund": {
                "type": "noul",
                "instructions": "Does the customer ask for a refund?",
            },
            "size": {
                "type": "score",
                "instructions": "Which amount range contains the invoice total?",
                "criteria": ["Below 1000", "1000 to 2000", "Above 2000"],
            },
        },
    }


def check_answers(result: dict) -> None:
    answers = result["answers"]
    assert answers["status"]["choice"] == "overdue", "Incorrect invoice status"
    assert answers["refund"]["noul"] > 0.5, "Refund intent missed"
    assert 0.5 < answers["size"]["score"] < 1.5, "Incorrect amount range"
    for answer in answers.values():
        if "probabilities" in answer:
            assert abs(sum(answer["probabilities"].values()) - 1) < 0.001


def print_summary(rows: dict) -> None:
    for case in ("short", "long"):
        baseline = rows[("GPU", case)][1]
        print(f"\n### {case.title()} prompt ({rows[('GPU', case)][0]} tokens)\n")
        print("| Path | Median body time | Prompt throughput | Versus GPU |")
        print("|---|---:|---:|---:|")
        for mode in ("GPU", "Single ANE + GPU"):
            tokens, seconds, _, _ = rows[(mode, case)]
            if tokens != rows[("GPU", case)][0]:
                raise ValueError("Cannot compare different token counts")
            print(
                f"| {mode} | {seconds:.4f} s | {tokens / seconds:.1f} tok/s | "
                f"{baseline / seconds:.3f}x |"
            )
    print("\n### Diagnostics\n")
    print(
        "| Mode | Case | Tokens | Median body time (s) | Process GiB | ANE MLP operations |"
    )
    print("|---|---|---:|---:|---:|---:|")
    for (mode, case), (tokens, seconds, memory, operations) in rows.items():
        print(
            f"| {mode} | {case} | {tokens} | {seconds:.4f} | "
            f"{memory:.2f} | {operations} |"
        )


async def benchmark(model: Path, repeats: int) -> None:
    from omlx.custom_kernels.quant_matmul import ane
    from omlx.engine.decision import DecisionEngine
    from omlx.model_settings import ModelSettings
    from omlx.utils.proc_memory import get_phys_footprint

    if not ane.ane_available():
        raise RuntimeError(
            "ANE native runtime unavailable; build with make kernels first"
        )
    rows = {}
    for enabled in (False, True):
        settings = ModelSettings(
            ane_prefill_enabled=enabled,
            ane_prefill_sequence_length=1024,
            ane_prefill_fraction=0.25,
            ane_prefill_max_layers=32,
            ane_prefill_dual_ane=False,
            ane_prefill_gdn=False,
            ane_prefill_cpu_enabled=False,
        )
        engine = DecisionEngine(str(model), model_settings=settings)
        try:
            await engine.start()
            if enabled and not engine.ane_status()["configured"]:
                raise RuntimeError(
                    "ANE requested but no eligible layers were configured"
                )
            for label, repetitions in (("short", 0), ("long", 160)):
                plan = await engine.encode(request(repetitions), truncate=False)
                mode = "Single ANE + GPU" if enabled else "GPU"
                print(f"Running {mode}, {label} prompt...", flush=True)
                if enabled and label == "long":
                    assert len(plan.input_ids) >= 1024, (
                        "Long fixture misses the ANE tile"
                    )
                if not ane.ane_profile_set_enabled(True):
                    raise RuntimeError(
                        "Native execution counters unavailable; rebuild kernels"
                    )
                ane.ane_profile_reset()
                samples = []
                for _ in range(repeats):
                    started = time.perf_counter()
                    result = await engine.systemone(plan)
                    samples.append(time.perf_counter() - started)
                    check_answers(result)
                profile = ane.ane_profile_snapshot()
                operations = int(profile.get("mlp", {}).get("operations", 0))
                if enabled and label == "long" and operations == 0:
                    raise RuntimeError(
                        "Compiled ANE layers never executed; GPU fallback measured"
                    )
                rows[(mode, label)] = (
                    result["input_tokens"],
                    statistics.median(samples[1:]),
                    get_phys_footprint() / 1024**3,
                    operations,
                )
        finally:
            ane.ane_profile_set_enabled(False)
            await engine.stop()
    print_summary(rows)
    print(
        "\nBody time includes backbone, decision head and executor scheduling; excludes load, encoding and HTTP."
    )
    print("Smoke checks passed. Process memory excludes some driver-wired weights.")
    print(
        "GPU occupancy, energy, held-out quality and concurrent serving were not measured."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 (first run is warmup)")
    asyncio.run(benchmark(args.model, args.repeats))


if __name__ == "__main__":
    main()
