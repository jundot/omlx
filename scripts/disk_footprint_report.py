"""Read-only oMLX data-directory footprint report (design doc §0.2).

Emits the disk-cleanup design doc's §1 table (per-area bytes/counts) plus a
per-day mtime histogram of the SSD cache, as JSON. Nothing is deleted or
moved — every number comes from `stat`/`os.walk`. Run before and after the
Phase 1 reapers to quantify what they actually reclaimed, and keep it
around afterward as a standing regression check ("did the tree grow back
to where it was").

Usage:
    python3 scripts/disk_footprint_report.py [--base-path ~/.omlx]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _dir_stats(path: Path) -> tuple[int, int]:
    """(total_bytes, file_count) for everything under `path`, symlinks excluded."""
    total_bytes = 0
    count = 0
    if not path.is_dir() or path.is_symlink():
        return 0, 0
    for entry in path.rglob("*"):
        try:
            if entry.is_symlink() or not entry.is_file():
                continue
            total_bytes += entry.stat().st_size
            count += 1
        except OSError:
            continue
    return total_bytes, count


def _mtime_histogram(paths: list[Path]) -> dict[str, dict[str, int]]:
    """Per-day {bytes, files} histogram across the given directories' files."""
    by_day: dict[str, dict[str, int]] = defaultdict(lambda: {"bytes": 0, "files": 0})
    for root in paths:
        if not root.is_dir() or root.is_symlink():
            continue
        for entry in root.rglob("*"):
            try:
                if entry.is_symlink() or not entry.is_file():
                    continue
                st = entry.stat()
            except OSError:
                continue
            day = time.strftime("%Y-%m-%d", time.localtime(st.st_mtime))
            by_day[day]["bytes"] += st.st_size
            by_day[day]["files"] += 1
    return dict(sorted(by_day.items()))


def build_report(base_path: Path) -> dict[str, Any]:
    base_path = base_path.expanduser().resolve()
    cache_dir = base_path / "cache"
    models_dir = base_path / "models"
    cluster_dir = base_path / "cluster"
    prompt_cache_ssd_dir = cluster_dir / "runtime" / "prompt-cache-ssd"
    logs_dir = base_path / "logs"

    areas: dict[str, dict[str, int]] = {}

    models_bytes, models_files = _dir_stats(models_dir)
    areas["models"] = {"bytes": models_bytes, "files": models_files}

    main_blocks_bytes = main_blocks_files = 0
    for subdir in "0123456789abcdef":
        b, f = _dir_stats(cache_dir / subdir)
        main_blocks_bytes += b
        main_blocks_files += f
    areas["cache_main_blocks"] = {"bytes": main_blocks_bytes, "files": main_blocks_files}

    sidecar_bytes, sidecar_files = _dir_stats(cache_dir / "_gdn_sidecars")
    areas["cache_gdn_sidecars"] = {"bytes": sidecar_bytes, "files": sidecar_files}

    boundary_bytes, boundary_files = _dir_stats(cache_dir / "_boundary_snapshots")
    areas["cache_boundary_snapshots"] = {"bytes": boundary_bytes, "files": boundary_files}

    response_bytes, response_files = _dir_stats(cache_dir / "response-state")
    areas["cache_response_state"] = {"bytes": response_bytes, "files": response_files}

    vision_bytes, vision_files = _dir_stats(cache_dir / "vision_features")
    areas["cache_vision_features"] = {"bytes": vision_bytes, "files": vision_files}

    prompt_ssd_bytes, prompt_ssd_files = _dir_stats(prompt_cache_ssd_dir)
    areas["cluster_prompt_cache_ssd"] = {"bytes": prompt_ssd_bytes, "files": prompt_ssd_files}

    cluster_total_bytes, cluster_total_files = _dir_stats(cluster_dir)
    areas["cluster_rest"] = {
        "bytes": max(0, cluster_total_bytes - prompt_ssd_bytes),
        "files": max(0, cluster_total_files - prompt_ssd_files),
    }

    logs_bytes, logs_files = _dir_stats(logs_dir)
    areas["logs"] = {"bytes": logs_bytes, "files": logs_files}

    loose_bytes = loose_files = 0
    known_dirs = {"cache", "models", "cluster", "logs", "bin", "__pycache__"}
    if base_path.is_dir():
        for entry in base_path.iterdir():
            if entry.name in known_dirs or entry.is_symlink():
                continue
            if entry.is_file():
                try:
                    loose_bytes += entry.stat().st_size
                    loose_files += 1
                except OSError:
                    pass
            elif entry.is_dir():
                b, f = _dir_stats(entry)
                loose_bytes += b
                loose_files += f
    areas["root_loose_files"] = {"bytes": loose_bytes, "files": loose_files}

    try:
        usage = shutil.disk_usage(base_path if base_path.exists() else base_path.parent)
        volume = {"total": usage.total, "used": usage.used, "free": usage.free}
    except OSError:
        volume = None

    histogram_roots = [
        cache_dir / subdir for subdir in "0123456789abcdef"
    ] + [cache_dir / "_gdn_sidecars"]

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_path": str(base_path),
        "areas": areas,
        "volume": volume,
        "cache_mtime_histogram_by_day": _mtime_histogram(histogram_roots),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("~/.omlx"),
        help="oMLX data directory (default: ~/.omlx)",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    report = build_report(args.base_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
