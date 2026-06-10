#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone P0 measurement harness for the fmlx video engine spec.

Runs Wan2.2 T2V generation profiles under the video venv and measures the
true per-run memory peak via the kernel lifetime-max phys_footprint ledger
(ri_lifetime_max_phys_footprint). Each profile runs in a fresh child process
so the lifetime max is exact for that run: model load + text encoding +
denoise + VAE decode + every sub-poll spike.

Must run under the video venv python (needs mflux). Does NOT import omlx
(see docs/video-generation-engine-spec.md section 4.2: worker venv isolation).

Parent mode (default): spawns one child per profile, samples the child's
phys_footprint every 0.5s, writes per-profile samples + results and a
summary.json.

Child mode (--single): loads the model, generates, saves the mp4, then reads
its OWN lifetime-max ledger and writes a result JSON.

Usage:
  video_p0_measure.py --model DIR --out DIR [--profiles default,steps40,...]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------------------
# phys_footprint via libproc (standalone copy of omlx/utils/proc_memory.py
# layout; this script must not import omlx)
# ---------------------------------------------------------------------------


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


_RUSAGE_INFO_V4 = 4
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_proc_pid_rusage = _libproc.proc_pid_rusage
_proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_proc_pid_rusage.restype = ctypes.c_int


def _rusage(pid: int) -> _RusageInfoV4 | None:
    info = _RusageInfoV4()
    if _proc_pid_rusage(pid, _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return info


def phys_footprint(pid: int) -> int:
    info = _rusage(pid)
    return info.ri_phys_footprint if info else 0


def lifetime_max_phys(pid: int) -> int:
    info = _rusage(pid)
    return info.ri_lifetime_max_phys_footprint if info else 0


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

PROMPT = "A red fox running through a snowy forest at dawn, cinematic, soft light"
SEED = 42

PROFILES: dict[str, dict] = {
    # name: width height frames steps fps (frames must be 4n+1, dims /16).
    # lowram=True mirrors the production worker defaults (mx cache limit
    # 1GB + release denoisers + clear cache per step) -- the numbers that
    # calibrate the shipped lease/predictor. Natural-mode profiles measure
    # the unconstrained envelope.
    "default": dict(width=480, height=272, frames=49, steps=20, fps=16),
    "steps40": dict(width=480, height=272, frames=49, steps=40, fps=16),
    "mid_spatial": dict(width=832, height=480, frames=49, steps=20, fps=16),
    "frames101": dict(width=480, height=272, frames=101, steps=20, fps=16),
    "default_lowram": dict(
        width=480, height=272, frames=49, steps=20, fps=16, lowram=True
    ),
    "mid_spatial_lowram": dict(
        width=832, height=480, frames=49, steps=20, fps=16, lowram=True
    ),
    "frames101_lowram": dict(
        width=480, height=272, frames=101, steps=20, fps=16, lowram=True
    ),
}

GB = 1024**3


# ---------------------------------------------------------------------------
# child mode: run one profile, report own lifetime max
# ---------------------------------------------------------------------------


def run_single(model_dir: str, out_dir: str, name: str) -> int:
    p = PROFILES[name]
    t0 = time.time()
    lowram = bool(p.get("lowram", False))

    def emit(**kw):
        kw["t"] = round(time.time() - t0, 1)
        print(json.dumps(kw), flush=True)

    if lowram:
        import mlx.core as mx

        try:
            mx.set_cache_limit(1 * GB)
        except Exception:
            pass

    emit(phase="loading")
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.wan.variants import Wan2_2_TI2V

    model = Wan2_2_TI2V(
        model_config=ModelConfig.wan2_2_t2v_a14b(), model_path=model_dir
    )
    emit(phase="loaded")

    def cb(ev):
        emit(
            phase=getattr(ev, "phase", "?"),
            step=getattr(ev, "step", 0),
            total_steps=getattr(ev, "total_steps", 0),
        )

    gen_kwargs = dict(
        seed=SEED,
        prompt=PROMPT,
        num_inference_steps=p["steps"],
        height=p["height"],
        width=p["width"],
        num_frames=p["frames"],
        fps=p["fps"],
        progress_callback=cb,
    )
    if lowram:
        gen_kwargs.update(
            release_inactive_denoiser=True,
            release_denoisers_before_decode=True,
            clear_cache_each_step=True,
        )
    video = model.generate_video(**gen_kwargs)
    emit(phase="saving")
    out_mp4 = os.path.join(out_dir, f"{name}.mp4")
    video.save(out_mp4)
    wall = time.time() - t0
    # read own ledger BEFORE exit (proc_pid_rusage fails on a reaped pid)
    result = {
        "profile": name,
        "params": p,
        "wall_seconds": round(wall, 1),
        "lifetime_max_phys_gb": round(lifetime_max_phys(os.getpid()) / GB, 2),
        "final_phys_gb": round(phys_footprint(os.getpid()) / GB, 2),
        "output": out_mp4,
        "output_bytes": os.path.getsize(out_mp4) if os.path.exists(out_mp4) else 0,
        "seed": SEED,
    }
    with open(os.path.join(out_dir, f"{name}.result.json"), "w") as f:
        json.dump(result, f, indent=1)
    emit(phase="done", wall_seconds=result["wall_seconds"])
    return 0


# ---------------------------------------------------------------------------
# parent mode: spawn child per profile, sample its footprint
# ---------------------------------------------------------------------------


def run_parent(model_dir: str, out_dir: str, names: list[str], timeout_s: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    summary = {"profiles": {}, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    for name in names:
        print(f"=== profile {name} ===", flush=True)
        log_path = os.path.join(out_dir, f"{name}.events.jsonl")
        samples_path = os.path.join(out_dir, f"{name}.samples.jsonl")
        child = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--single", name,
             "--model", model_dir, "--out", out_dir],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        stop = threading.Event()
        peak = {"sampled_max": 0, "max_delta_per_sample": 0}

        def sampler():
            last = 0
            with open(samples_path, "w") as sf:
                while not stop.is_set():
                    b = phys_footprint(child.pid)
                    if b:
                        t = round(time.time(), 1)
                        sf.write(json.dumps({"t": t, "gb": round(b / GB, 3)}) + "\n")
                        sf.flush()
                        peak["sampled_max"] = max(peak["sampled_max"], b)
                        if last:
                            peak["max_delta_per_sample"] = max(
                                peak["max_delta_per_sample"], b - last
                            )
                        last = b
                    stop.wait(0.5)

        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        deadline = time.time() + timeout_s
        with open(log_path, "w") as lf:
            for line in child.stdout:  # type: ignore[union-attr]
                lf.write(line)
                lf.flush()
                print(f"  [{name}] {line.rstrip()}", flush=True)
                if time.time() > deadline:
                    child.kill()
                    print(f"  [{name}] TIMEOUT after {timeout_s}s, killed", flush=True)
                    break
        rc = child.wait()
        stop.set()
        th.join(timeout=2)
        entry = {
            "exit_code": rc,
            "sampled_max_gb": round(peak["sampled_max"] / GB, 2),
            "max_delta_per_0p5s_gb": round(peak["max_delta_per_sample"] / GB, 2),
        }
        rpath = os.path.join(out_dir, f"{name}.result.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                entry.update(json.load(f))
        summary["profiles"][name] = entry
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(f"=== {name} done: {json.dumps(entry)} ===", flush=True)
    print("=== ALL DONE ===", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profiles", default="default,steps40,mid_spatial,frames101")
    ap.add_argument("--single", default=None, help="internal: run one profile in-process")
    ap.add_argument("--timeout", type=int, default=10800)
    args = ap.parse_args()
    if args.single:
        return run_single(args.model, args.out, args.single)
    names = [n.strip() for n in args.profiles.split(",") if n.strip()]
    for n in names:
        if n not in PROFILES:
            print(f"unknown profile {n}; known: {list(PROFILES)}", file=sys.stderr)
            return 2
    return run_parent(args.model, args.out, names, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
