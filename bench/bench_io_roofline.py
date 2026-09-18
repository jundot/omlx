#!/usr/bin/env python3
"""I/O roofline microbench — reproduces the expert_streaming read path.

Faithful to `shard_bank._read_into`: buffered `os.preadv` into a fresh
np.empty buffer per command (no O_DIRECT/F_NOCACHE — the runtime relies
on page cache for hits, so this measures the miss path).

Anti-cache control: every command targets a uniform-random offset over
the file, never repeating, so reads are cold when file >> RAM. A
`--hot` mode re-reads one small region instead, to expose cache
contamination for comparison.

Arms:
  size sweep : serial reads at each size (random placement)
  seq-vs-rnd : serial reads, sequential vs random placement, same size
  qd sweep   : ThreadPoolExecutor at each queue depth, fixed size

Usage:
  python bench/bench_io_roofline.py FILE [--iters N] [--gb-per-arm G]
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PAGE = 16384  # this box: vm_stat page size


def _preadv(fd: int, off: int, n: int) -> None:
    out = np.empty(n, dtype=np.uint8)  # fresh buffer per read, like _read_into
    got = os.preadv(fd, [memoryview(out)], off)
    if got != n:
        raise OSError(f"short read: {got}/{n}")


def _rand_off(rng: random.Random, span: int, size: int) -> int:
    return (rng.randrange(0, max(1, (span - size) // PAGE)) * PAGE)


def size_sweep(fd: int, span: int, sizes: list[int], iters: int) -> None:
    rng = random.Random(0xC0AC7)
    print(f"\n== serial preadv, random placement (iters={iters}) ==")
    print(f"{'size':>8} {'lat_med_ms':>10} {'lat_p10_ms':>10} {'GB/s':>7}")
    for n in sizes:
        offs = [_rand_off(rng, span, n) for _ in range(iters)]
        lats = []
        for off in offs:  # warm-up not needed per arm; first iter included in p10 anyway
            t0 = time.perf_counter()
            _preadv(fd, off, n)
            lats.append(time.perf_counter() - t0)
        med = statistics.median(lats)
        p10 = sorted(lats)[max(0, iters // 10 - 1)]
        print(f"{n/2**20:>7.2f}M {med*1e3:>10.2f} {p10*1e3:>10.2f} "
              f"{(n/med)/2**30:>7.2f}")


def seq_vs_random(fd: int, span: int, size: int, iters: int) -> None:
    rng = random.Random(0x5E9)
    print(f"\n== serial preadv {size/2**20:.1f}M: sequential vs random ==")
    # sequential: walk contiguous, wrap within a 4GB window far into the file
    base = _rand_off(rng, span, size * iters + PAGE)
    lats = []
    off = base
    for _ in range(iters):
        t0 = time.perf_counter()
        _preadv(fd, off, size)
        lats.append(time.perf_counter() - t0)
        off += size
    med_s = statistics.median(lats)
    lats = []
    for _ in range(iters):
        off = _rand_off(rng, span, size)
        t0 = time.perf_counter()
        _preadv(fd, off, size)
        lats.append(time.perf_counter() - t0)
    med_r = statistics.median(lats)
    print(f"sequential: {med_s*1e3:.2f} ms/cmd  {(size/med_s)/2**30:.2f} GB/s")
    print(f"random    : {med_r*1e3:.2f} ms/cmd  {(size/med_r)/2**30:.2f} GB/s")


def qd_sweep(fd: int, span: int, qds: list[int], size: int, per_worker: int,
             seed: int = 0x0FF5E7) -> None:
    # NOTE: seed must differ per invocation — identical seeds regenerate the
    # same offsets, and re-reading a previously-read region measures page
    # cache (~40 GB/s), not the device.
    rng = random.Random(seed)
    print(f"\n== parallel preadv, {size/2**20:.1f}M cmds, {per_worker} reads/worker ==")
    print(f"{'QD':>4} {'wall_s':>8} {'GB/s':>7} {'cmd_med_ms':>10}")
    for qd in qds:
        offs = [[_rand_off(rng, span, size) for _ in range(per_worker)]
                for _ in range(qd)]
        lats: list[float] = []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=qd) as pool:
            def work(o):
                for x in o:
                    t = time.perf_counter()
                    _preadv(fd, x, size)
                    lats.append(time.perf_counter() - t)
            list(pool.map(work, offs))
        wall = time.perf_counter() - t0
        total = qd * per_worker * size
        print(f"{qd:>4} {wall:>8.2f} {(total/wall)/2**30:>7.2f} "
              f"{statistics.median(lats)*1e3:>10.2f}")


def mech_sweep(path: str, span: int, size: int, iters: int) -> None:
    """Per-mechanism comparison at one command size, random cold offsets.

    Arms replicate the real read paths:
      preadv     — shard_bank._read_into (generic + V4.1 demand path)
      readinto   — legacy V4.1 span mechanism (pre-preadv)
      mmap gather— TensorFile.read(rows=) (V4.1 BASELINE path: page faults,
                   MADV_RANDOM, no readahead)
      dio sync   — dispatch_io, one in-flight command
      dio burst  — N submitted then waited (kernel-managed queue, no threads)
      aio burst  — POSIX aio_read, same submit/wait shape
    """
    import mmap as mmap_mod

    rng = random.Random(int(time.time()))
    # Fresh offsets per arm AND fresh fd per arm: reusing either re-reads
    # warm page cache (inflate) or perturbs dispatch_io (EIO observed after
    # a dup'd buffered reader + mmap touched the same fd).
    def fresh_offs():
        return [(rng.randrange(0, max(1, (span - size) // PAGE)) * PAGE)
                for _ in range(iters)]

    def timed(fn, label, offs):
        lats = []
        for off in offs:
            t0 = time.perf_counter()
            fn(off)
            lats.append(time.perf_counter() - t0)
        med = statistics.median(lats)
        print(f"  {label:<14} {med*1e3:>8.2f} ms/cmd  {(size/med)/2**30:>6.2f} GB/s")
        return med

    print(f"\n== mechanism sweep @ {size/2**20:.1f}M, {iters} cold reads ==")

    fd = os.open(path, os.O_RDONLY)
    offs = fresh_offs()
    timed(lambda o: _preadv(fd, o, size), "preadv", offs)
    os.close(fd)

    offs = fresh_offs()
    with open(path, "rb", buffering=0) as fobj:
        timed(lambda o: (fobj.seek(o), fobj.readinto(np.empty(size, np.uint8))),
              "readinto", offs)

    offs = fresh_offs()
    fd = os.open(path, os.O_RDONLY)
    mm = mmap_mod.mmap(fd, 0, access=mmap_mod.ACCESS_READ)
    mm.madvise(mmap_mod.MADV_RANDOM)
    def _gather(o):
        # page-fault path: slice the mapping then copy, like view[rows]
        np.frombuffer(mm[o:o + size], dtype=np.uint8).copy()
    timed(_gather, "mmap gather", offs)
    mm.close()
    os.close(fd)

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        try:
            from omlx.patches.expert_streaming import dispatch_io as dio
        except ImportError:
            # Bench-only shim (feat/v41-moe-streaming-opts) — not present
            # on every branch; treat its absence like available()=False.
            dio = None
        if dio is not None and dio.available():
            fd = os.open(path, os.O_RDONLY)
            offs = fresh_offs()
            timed(lambda o: dio.read_into(fd, o, np.empty(size, np.uint8)),
                  "dio sync", offs)
            offs = fresh_offs()
            timed(lambda o: dio.read_into(fd, o, np.empty(size, np.uint8),
                                          backend=dio.BACKEND_AIO),
                  "aio sync", offs)

            # burst: submit all, wait all — no worker threads at all
            for name, backend in (("dio burst", dio.BACKEND_DISPATCH_IO),
                                  ("aio burst", dio.BACKEND_AIO)):
                offs = fresh_offs()
                bufs = [np.empty(size, np.uint8) for _ in offs]
                t0 = time.perf_counter()
                hs = [dio.submit(fd, o, b, backend=backend)
                      for o, b in zip(offs, bufs)]
                for h in hs:
                    dio.wait(h)
                wall = time.perf_counter() - t0
                print(f"  {name:<14} {wall/iters*1e3:>8.2f} ms/cmd  "
                      f"{(iters*size/wall)/2**30:>6.2f} GB/s "
                      f"(QD={iters}, kernel-managed)")
            os.close(fd)
        else:
            print("  dio/aio      shim unavailable — skipped")
    except Exception as e:
        print(f"  dio/aio      skipped: {e}")


def hot_check(fd: int, span: int, size: int, iters: int) -> None:
    """Same-offset re-reads: shows what page-cache contamination looks like."""
    rng = random.Random(1)
    off = _rand_off(rng, span, size)
    lats = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _preadv(fd, off, size)
        lats.append(time.perf_counter() - t0)
    med = statistics.median(lats)
    print(f"\n== control: same offset re-read {size/2**20:.1f}M x{iters} ==")
    print(f"{med*1e3:.2f} ms/cmd  {(size/med)/2**30:.2f} GB/s "
          f"(page cache — this is what contamination inflates to)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--iters", type=int, default=48)
    ap.add_argument("--qd-workers", type=int, default=16,
                    help="reads per worker in the QD sweep")
    args = ap.parse_args()

    st = os.stat(args.file)
    span = st.st_size
    print(f"file={args.file} ({span/2**30:.1f} GiB)")

    fd = os.open(args.file, os.O_RDONLY)
    try:
        # warm-up: metadata + first-touch
        _preadv(fd, 0, PAGE)

        sizes = [2**x for x in (18, 19, 20, 21, 22, 23, 24, 25)]  # 256K..32M
        size_sweep(fd, span, sizes, args.iters)
        seq_vs_random(fd, span, 8 * 2**20, args.iters)
        qd_sweep(fd, span, [1, 2, 4, 8, 16, 24, 32], 8 * 2**20, args.qd_workers)
        hot_check(fd, span, 8 * 2**20, min(args.iters, 16))
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
