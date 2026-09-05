#!/usr/bin/env python3
"""Evict file-backed page cache without root by touching anonymous memory.

macOS reclaims clean file-cache pages before compressing/swapping anonymous
pages, so touching a large anonymous buffer pushes the model shards' cached
pages out of the unified buffer cache. Use between benchmark runs for
comparable cold-cache measurements.

Usage:
    python bench/cache_cool.py --gb 28
"""
import argparse
import time

import psutil


def available_gb() -> float:
    return psutil.virtual_memory().available / 1024**3


def main() -> None:
    ap = argparse.ArgumentParser(description="Evict page cache by touching anon memory")
    ap.add_argument("--gb", type=float, default=28.0, help="target anon GB to touch")
    ap.add_argument("--chunk-mb", type=int, default=256)
    ap.add_argument("--hold", type=float, default=4.0, help="seconds to hold touched pages")
    args = ap.parse_args()

    before = available_gb()
    target = min(args.gb * 1024**3, before * 0.72 * 1024**3)
    print(f"cache_cool: available before {before:.1f} GB, touching {target / 1024**3:.1f} GB", flush=True)

    chunk = int(args.chunk_mb * 1024**2)
    window: list[bytearray] = []
    touched = 0
    while touched < target:
        buf = bytearray(chunk)
        for i in range(0, chunk, 4096):
            buf[i] = 1
        window.append(buf)
        touched += chunk
        if len(window) > 2:
            window.pop(0)

    print(f"cache_cool: touched {touched / 1024**3:.1f} GB, holding {args.hold:.0f}s", flush=True)
    time.sleep(args.hold)
    window.clear()
    time.sleep(2.0)

    print(f"cache_cool: available after {available_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
