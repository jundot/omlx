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


def _decode_frames(path: str) -> list:
    """Decode every frame of an mp4 into PIL RGB images (PyAV ships with
    mlx-gen). Used to stitch a continuation onto its source video."""
    import av
    import PIL.Image

    frames = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"no video stream in {path}")
        for frame in container.decode(container.streams.video[0]):
            frames.append(PIL.Image.fromarray(frame.to_ndarray(format="rgb24")))
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def _model_config_for(pipeline: str):
    """Map the registry's pipeline kind to mlx-gen's ModelConfig. The
    initializer cross-validates the local dir's model_index.json against
    this config and hard-refuses on mismatch, so a wrong kind cannot load
    the wrong architecture silently."""
    from mflux.models.common.config.model_config import ModelConfig

    if pipeline == "i2v":
        return ModelConfig.wan2_2_i2v_a14b()
    if pipeline == "ti2v":
        return ModelConfig.wan2_2_ti2v_5b()
    return ModelConfig.wan2_2_t2v_a14b()


def _upscale_frames(frames: list, spec: dict) -> list:
    """SeedVR2 per-frame upscale to spec["upscale_resolution"] (target short
    side). The caller must have dropped its Wan model references first --
    the two are never Metal-co-resident inside the lease. A fixed seed
    across frames keeps the single-step diffusion temporally stable."""
    import shutil as _shutil
    import tempfile

    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2

    total = len(frames)
    _emit(phase="upscaling", step=0, total_steps=total)
    upscaler = SeedVR2(
        model_path=spec["upscaler_model_dir"],
        model_config=ModelConfig.seedvr2_3b(),
    )
    target = int(spec["upscale_resolution"])
    seed = int(spec["seed"])
    out = []
    tmp_dir = tempfile.mkdtemp(prefix="fmlx_upscale_")
    try:
        for i, frame in enumerate(frames):
            frame_path = os.path.join(tmp_dir, f"f{i:05d}.png")
            frame.save(frame_path)
            generated = upscaler.generate_image(
                seed=seed, image_path=frame_path, resolution=target
            )
            out.append(generated.image)
            _emit(phase="upscaling", step=i + 1, total_steps=total)
    finally:
        _shutil.rmtree(tmp_dir, ignore_errors=True)
    return out


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

    # Continuations condition on the source video's last frame and stitch
    # the decoded source frames in front of the new segment.
    extend_from = spec.get("extend_from")
    image_path = spec.get("image_path")
    prior_frames: list = []
    if extend_from:
        _emit(phase="extracting_reference")
        prior_frames = _decode_frames(extend_from)
        image_path = os.path.join(
            os.path.dirname(output_path), "extend_reference.png"
        )
        prior_frames[-1].save(image_path)

    _emit(phase="loading")
    from mflux.models.wan.variants import Wan2_2_TI2V

    pipeline = str(spec.get("pipeline") or "t2v")
    model = Wan2_2_TI2V(
        model_config=_model_config_for(pipeline),
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
    if image_path:
        kwargs["image_path"] = image_path
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fps = int(spec["fps"])
    frames = list(video.frames)
    if prior_frames:
        _emit(phase="stitching")
        # I2V reproduces its conditioning image as the first output frame;
        # drop it so the seam does not stutter on a duplicate.
        frames = prior_frames + frames[1:]

    upscale_res = spec.get("upscale_resolution")
    stitched_or_upscaled = bool(prior_frames) or bool(upscale_res)
    if upscale_res:
        from mflux.utils.video_util import VideoUtil

        # Keep the pre-upscale video for future continuations: extending
        # from upscaled frames would force generating the next segment at
        # the upscaled resolution (hours per segment on this hardware).
        raw_path = os.path.join(os.path.dirname(output_path), "output_raw.mp4")
        VideoUtil.save_video(frames=frames, path=raw_path, fps=fps)
        # Release the Wan weights before SeedVR2 loads -- never co-resident
        # inside the lease.
        import gc

        model = None  # noqa: F841 -- drop the only strong reference
        video = None  # noqa: F841
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
        frames = _upscale_frames(frames, spec)

    _emit(phase="saving")
    if stitched_or_upscaled:
        from mflux.utils.video_util import VideoUtil

        VideoUtil.save_video(frames=frames, path=output_path, fps=fps)
    else:
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
            "frames_total": len(frames),
            "extended_from": extend_from or None,
            "upscale_resolution": upscale_res or None,
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
