#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the bundled task lists for the agentic Intelligence benchmarks.

This script is a maintainer tool, not part of the runtime. It downloads each
Harbor dataset definition (task.toml + Dockerfiles, no images) with the same
pinned Harbor CLI the runtime uses, and writes one sorted JSONL of
``{"id": <Harbor task name>, "category": <str>}`` per suite. The ``id`` is the
exact name Harbor accepts for ``-i`` and reports as ``task_name`` in results.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "omlx" / "eval" / "data"
HARBOR = ["uvx", "--python", "3.13", "--from", "harbor==0.23.0", "harbor"]

SUITES = (
    # (dataset ref, output file, category source)
    ("terminal-bench/terminal-bench@4", "terminalbench_4_tasks.jsonl", "metadata"),
    ("swebench-verified@1.0", "swebench_verified_tasks.jsonl", "repo"),
)


def _task_name(task_dir: Path, config: dict) -> str:
    name = config.get("task", {}).get("name")
    return name if isinstance(name, str) and name else task_dir.name


def _category(name: str, config: dict, source: str) -> str | None:
    if source == "repo":
        # django__django-13741 -> django/django
        return name.rsplit("/", 1)[-1].rsplit("-", 1)[0].replace("__", "/")
    category = config.get("metadata", {}).get("category")
    return category if isinstance(category, str) and category else None


def _requires_gpu(config: dict) -> bool:
    """Tasks needing a GPU (e.g. H100) cannot run on macOS Docker at all.

    Harbor rejects them at job setup, failing the whole job rather than the
    trial, so they are excluded from the bundled list.
    """
    envs = (
        config.get("environment", {}),
        config.get("verifier", {}).get("environment", {}),
    )
    return any((env.get("gpus") or 0) > 0 for env in envs)


def build(dataset: str, output: str, source: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [*HARBOR, "datasets", "download", dataset, "-o", tmp], check=True
        )
        records = []
        skipped = []
        for toml_path in Path(tmp).rglob("task.toml"):
            config = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            name = _task_name(toml_path.parent, config)
            if _requires_gpu(config):
                skipped.append(name)
                continue
            record = {"id": name}
            category = _category(name, config, source)
            if category:
                record["category"] = category
            records.append(record)
    if skipped:
        print(f"{dataset}: excluded {len(skipped)} GPU-only task(s): {sorted(skipped)}")
    records.sort(key=lambda r: r["id"])
    path = OUTPUT_DIR / output
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    print(f"{dataset}: wrote {len(records)} tasks to {path.relative_to(REPO_ROOT)}")
    return len(records)


def main() -> None:
    for dataset, output, source in SUITES:
        build(dataset, output, source)


if __name__ == "__main__":
    main()
