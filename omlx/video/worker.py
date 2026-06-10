#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Video generation subprocess worker.

Runs ONE generation job and exits. Spawned by VideoJobManager as:

    <video_venv>/bin/python -I <repo>/omlx/video/worker.py --spec job_spec.json

HARD RULE: this script must not import omlx. It runs under the video venv
(mlx-gen + its deps); only mflux, mlx and the standard library are
available. See docs/video-generation-engine-spec.md section 4.2.

Protocol:
- stdout: one JSON object per line. Phase heartbeats ({"phase": ...}) are
  emitted on every phase transition so silent long phases (42GB weight
  load, torch text encoding, VAE decode) still show liveness; denoise
  steps additionally carry step/total_steps. The manager tracks the last
  line timestamp for stall detection.
- Exit 0 + the output mp4 present and healthy = success. A result manifest
  with timings and the kernel lifetime-max memory peak is written next to
  the output for calibration records.
- Any failure: a failure manifest {code, message, detail} is written at
  spec["manifest_path"] and the exit code is non-zero.

Memory: before loading anything the worker pins its own Metal wired limit
inside the lease (spec 4.4 layer 1) -- overshoot degrades to non-resident
pages or an in-process allocation failure, never wired-sum growth toward
the machine cap.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
import traceback

GB = 1024**3
_T0 = time.time()


def _emit(**kw) -> None:
    kw["t"] = round(time.time() - _T0, 1)
    try:
        print(json.dumps(kw), flush=True)
    except Exception:
        # Never let progress reporting kill the generation (a raising
        # progress callback aborts mlx-gen's denoise loop).
        pass


def _lifetime_max_phys() -> int:
    """Own-process lifetime-max phys_footprint via libproc (best effort).

    rusage_info_v4 layout from sys/resource.h: ri_uuid (16 bytes), then 28
    c_uint64 fields, then ri_lifetime_max_phys_footprint. Standalone copy --
    this script cannot import omlx/utils/proc_memory.py.
    """
    try:
        class _RusageInfoV4(ctypes.Structure):
            _fields_ = (
                [("ri_uuid", ctypes.c_uint8 * 16)]
                + [(f"_u{i}", ctypes.c_uint64) for i in range(28)]
                + [("ri_lifetime_max_phys_footprint", ctypes.c_uint64)]
                + [("_tail", ctypes.c_uint64 * 6)]
            )

        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        fn = libproc.proc_pid_rusage
        fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        info = _RusageInfoV4()
        if fn(os.getpid(), 4, ctypes.byref(info)) != 0:
            return 0
        return int(info.ri_lifetime_max_phys_footprint)
    except Exception:
        return 0


def _write_manifest(path: str, payload: dict) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, path)
    except Exception:
        pass


def run(spec: dict) -> int:
    manifest_path = spec["manifest_path"]
    output_path = spec["output_path"]

    # Layer-1 memory containment: pin our Metal wired limit inside the
    # lease BEFORE any weights load.
    lease = int(spec.get("lease_bytes", 0))
    margin = int(spec.get("wired_margin_bytes", 2 * GB))
    if lease > 0:
        import mlx.core as mx

        limit = max(1 * GB, lease - margin)
        try:
            mx.set_wired_limit(limit)
            _emit(phase="wired_limit_set", limit_gb=round(limit / GB, 1))
        except Exception as e:
            _emit(phase="wired_limit_failed", error=str(e))

    # Low-RAM mode (default ON): release the inactive/high-noise denoiser
    # after the boundary step, free both transformers before VAE decode and
    # clear the MLX cache per step. P0 measurement showed the natural-mode
    # peak at ~49GB even for small profiles; the low-RAM knobs are what the
    # official benchmarks (20.7GB) use. Cost: the model instance is dead
    # after one generation -- irrelevant here, one process per job.
    low_ram = bool(spec.get("low_ram", True))
    if low_ram:
        import mlx.core as mx

        try:
            mx.set_cache_limit(1 * GB)
        except Exception:
            pass

    _emit(phase="loading")
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.wan.variants import Wan2_2_TI2V

    model = Wan2_2_TI2V(
        model_config=ModelConfig.wan2_2_t2v_a14b(),
        model_path=spec["model_dir"],
    )
    _emit(phase="loaded")

    def cb(ev) -> None:
        _emit(
            phase=str(getattr(ev, "phase", "denoise")),
            step=int(getattr(ev, "step", 0) or 0),
            total_steps=int(getattr(ev, "total_steps", 0) or 0),
        )

    kwargs = dict(
        seed=int(spec["seed"]),
        prompt=spec["prompt"],
        num_inference_steps=int(spec["steps"]),
        height=int(spec["height"]),
        width=int(spec["width"]),
        num_frames=int(spec["frames"]),
        fps=int(spec["fps"]),
        progress_callback=cb,
    )
    if low_ram:
        kwargs["release_inactive_denoiser"] = True
        kwargs["release_denoisers_before_decode"] = True
        kwargs["clear_cache_each_step"] = True
    if spec.get("negative_prompt"):
        kwargs["negative_prompt"] = spec["negative_prompt"]
    if spec.get("guidance") is not None:
        kwargs["guidance"] = float(spec["guidance"])
    if spec.get("guidance_2") is not None:
        kwargs["guidance_2"] = float(spec["guidance_2"])

    video = model.generate_video(**kwargs)

    _emit(phase="saving")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video.save(output_path)

    wall = round(time.time() - _T0, 1)
    _write_manifest(
        manifest_path,
        {
            "status": "completed",
            "wall_seconds": wall,
            "lifetime_max_phys_gb": round(_lifetime_max_phys() / GB, 2),
            "output_bytes": (
                os.path.getsize(output_path) if os.path.exists(output_path) else 0
            ),
        },
    )
    _emit(phase="done", wall_seconds=wall)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    with open(args.spec) as f:
        spec = json.load(f)
    try:
        return run(spec)
    except Exception as e:
        _write_manifest(
            spec.get("manifest_path", args.spec + ".manifest.json"),
            {
                "status": "failed",
                "code": "worker_crashed",
                "message": f"{type(e).__name__}: {e}",
                "detail": traceback.format_exc()[-4000:],
            },
        )
        _emit(phase="failed", error=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
