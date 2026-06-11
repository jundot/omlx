#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Image generation subprocess worker.

Runs ONE generation job (1..n images, same prompt, sequential seeds) and
exits. Spawned by MediaJobManager as:

    <media_venv>/bin/python -I <repo>/omlx/image/worker.py --spec job_spec.json

HARD RULE: this script must not import omlx. It runs under the media venv
(mlx-gen + its deps); only mflux, mlx and the standard library are
available. Same contract as omlx/video/worker.py.

Protocol:
- stdout: one JSON object per line. Phase heartbeats ({"phase": ...}) on
  every transition; denoise lines carry step/total_steps/image_index. The
  manager tracks the last line timestamp for stall detection.
- Exit 0 + a manifest listing the output files = success. The manager
  validates every listed file exists and is non-empty.
- Any failure: a failure manifest {code, message, detail} is written at
  spec["manifest_path"] and the exit code is non-zero.

Model resolution: spec["alias"] is the mflux ModelConfig name recorded at
discovery (model_discovery.read_image_model_kind). It MUST resolve via
ModelConfig.from_name() -- several aliases (qwen-image-edit-2511) have no
static factory method. Image dirs carry no model_index.json for mflux to
cross-validate, so a wrong alias would otherwise fail late at weight
application; _preflight_components fails in seconds instead.

Memory: before loading anything the worker pins its own Metal wired limit
inside the lease (video spec 4.4 layer 1).
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


def _preflight_components(model_dir: str) -> None:
    """Verify the mflux saved-format weights are complete BEFORE loading.

    Image dirs have no model_index.json for the initializer to
    cross-validate, so a partial download or wrong layout would otherwise
    fail minutes in (or worse, at weight application with a confusing
    shape error). Checks every shard listed in the component indexes.
    """
    for component in ("transformer", "text_encoder", "vae"):
        index_path = os.path.join(
            model_dir, component, "model.safetensors.index.json"
        )
        if not os.path.isfile(index_path):
            raise RuntimeError(
                f"model weights incomplete: missing {component} index"
            )
        with open(index_path) as f:
            index = json.load(f)
        shards = set((index.get("weight_map") or {}).values())
        if not shards:
            raise RuntimeError(f"model {component} index lists no shards")
        for shard in shards:
            shard_path = os.path.join(model_dir, component, shard)
            if not os.path.isfile(shard_path) or os.path.getsize(shard_path) == 0:
                raise RuntimeError(
                    f"model weights incomplete: missing {component}/{shard}"
                )


def _load_model(spec: dict):
    """Construct the mflux model class for spec["alias"].

    ModelConfig.from_name() is the only safe resolution path -- several
    aliases (qwen-image-edit-2511) have no static factory method.
    """
    from mflux.models.common.config.model_config import ModelConfig

    alias = str(spec["alias"])
    kwargs: dict = dict(
        model_config=ModelConfig.from_name(alias),
        model_path=spec["model_dir"],
    )
    lora_paths = spec.get("lora_paths") or []
    if lora_paths:
        kwargs["lora_paths"] = [str(p) for p in lora_paths]
        scales = spec.get("lora_scales") or []
        if scales:
            kwargs["lora_scales"] = [float(s) for s in scales]

    if alias.startswith("z-image"):
        from mflux.models.z_image.variants.z_image import ZImage

        return ZImage(**kwargs)
    if "edit" in alias:
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

        return QwenImageEdit(**kwargs)
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    return QwenImage(**kwargs)


def run(spec: dict) -> int:
    manifest_path = spec["manifest_path"]
    output_dir = spec["output_dir"]
    alias = str(spec["alias"])
    pipeline = str(spec.get("pipeline") or "t2i")
    n = max(1, int(spec.get("n") or 1))

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

    # Cap the MLX buffer cache like the video worker's low-RAM mode. We do
    # NOT use mflux's MemorySaver callback: it deletes the text encoders
    # after the first prompt encode, which would break n>1 (the model
    # instance must survive all n generations).
    import mlx.core as mx

    try:
        mx.set_cache_limit(1 * GB)
    except Exception:
        pass

    _preflight_components(spec["model_dir"])

    _emit(phase="loading")
    model = _load_model(spec)
    # VAE decode without tiling materializes multi-GB activation spikes
    # (the SeedVR2 lesson, video worker line ~168); the attribute is
    # duck-typed across mflux model classes.
    try:
        from mflux.models.common.vae.tiling_config import TilingConfig

        model.tiling_config = TilingConfig()
    except Exception:
        pass
    _emit(phase="loaded")

    # Per-step progress: generate_image has NO progress_callback kwarg
    # (that is Wan video only); images use the callback registry.
    current_index = 0

    def cb(ev) -> None:
        _emit(
            phase=str(getattr(ev, "phase", "denoise") or "denoise"),
            step=int(getattr(ev, "step", 0) or 0),
            total_steps=int(getattr(ev, "total_steps", 0) or 0),
            image_index=current_index,
        )

    try:
        model.callbacks.subscribe_progress(cb)
    except Exception as e:
        _emit(phase="progress_subscribe_failed", error=str(e))

    image_paths = [str(p) for p in (spec.get("image_paths") or [])]
    base_kwargs: dict = dict(
        prompt=spec["prompt"],
        num_inference_steps=int(spec["steps"]),
    )
    if spec.get("width") and spec.get("height"):
        base_kwargs["width"] = int(spec["width"])
        base_kwargs["height"] = int(spec["height"])
    # z-image-turbo forces guidance=0 and ignores negative_prompt
    # (supports_guidance=False); passing them is at best a no-op.
    if alias != "z-image-turbo":
        if spec.get("negative_prompt"):
            base_kwargs["negative_prompt"] = spec["negative_prompt"]
        if spec.get("guidance") is not None:
            base_kwargs["guidance"] = float(spec["guidance"])
    if pipeline == "edit":
        # Edit conditions on reference images; width/height default to the
        # first reference's aspect at ~1MP inside mflux (the recommended
        # default -- forced square output degrades qwen-edit quality).
        base_kwargs["image_paths"] = image_paths
    elif image_paths:
        # img2img on a t2i model: single source image + strength.
        base_kwargs["image_path"] = image_paths[0]
        strength = spec.get("image_strength")
        base_kwargs["image_strength"] = (
            float(strength) if strength is not None else 0.6
        )

    os.makedirs(output_dir, exist_ok=True)
    seed = int(spec["seed"])
    outputs: list[str] = []
    out_width = out_height = None
    for i in range(n):
        current_index = i
        generated = model.generate_image(seed=seed + i, **base_kwargs)
        _emit(phase="saving", image_index=i)
        name = f"output-{i}.png"
        generated.save(path=os.path.join(output_dir, name), overwrite=True)
        outputs.append(name)
        if out_width is None:
            try:
                # Report ACTUAL output dimensions: with an input image the
                # default canvas policy re-derives the canvas from the
                # source aspect, so requested width/height can be wrong.
                out_width, out_height = generated.image.size
            except Exception:
                pass
        if n > 1:
            # Keep the cache from compounding across images.
            try:
                mx.clear_cache()
            except Exception:
                pass

    wall = round(time.time() - _T0, 1)
    peak_mlx = 0.0
    try:
        peak_mlx = round(float(mx.get_peak_memory()) / GB, 2)
    except Exception:
        pass
    _write_manifest(
        manifest_path,
        {
            "status": "completed",
            "outputs": outputs,
            "wall_seconds": wall,
            "lifetime_max_phys_gb": round(_lifetime_max_phys() / GB, 2),
            "peak_mlx_gb": peak_mlx,
            "alias": alias,
            "steps": int(spec["steps"]),
            "n": n,
            "width": out_width,
            "height": out_height,
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
