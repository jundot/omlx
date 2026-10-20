# SPDX-License-Identifier: Apache-2.0
"""Standalone, serialized native generation. No server registration."""

import gc
import io
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from omlx.image_bundle import read_json
from omlx.image_inspect import contained, inspect_bundle, tensor_header

from .conditioning import check_cancel, encode_prompt, native_bundle
from .loading import load_transformer
from .vae import Qwen21VAE

SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0)


def initial_noise(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return np.random.default_rng(seed).standard_normal((1, 4096, 64), dtype=np.float32)


def load_vae(bundle):
    root = native_bundle(bundle)
    config = read_json(contained(root, "vae/config.json"))
    expected = read_json(Path(__file__).with_name("vae_config.json"))
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported VAE architecture or latent normalization")
    path = contained(root, "vae/model.safetensors")
    header, _, _, _ = tensor_header(path, False)
    model = Qwen21VAE(config)
    shapes = {k: list(v.shape) for k, v in tree_flatten(model.parameters())}
    if set(header) != set(shapes) or any(
        item["shape"] != shapes[k] or item["dtype"] != "BF16"
        for k, item in header.items()
    ):
        raise ValueError("VAE tensor names/shapes/dtypes do not match native v1")
    model.load_weights(list(mx.load(str(path)).items()), strict=True)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


def release_gpu():
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def generate_png(bundle, prompt, seed, *, cancel=None, progress=None):
    """Return PNG bytes and phase metrics, holding no model objects on return.

    Caller must serialize execution. Request cancellation is cooperative; this
    synchronous function drains GPU work in its finalizer before returning.
    """
    noise = initial_noise(seed)
    root = native_bundle(bundle)
    report = inspect_bundle(root)
    if report["structural_status"] != "valid":
        raise ValueError(report["errors"])
    check_cancel(cancel)
    model = vae = embeddings = prediction = latents = pixels = None
    metrics = {
        "seed": seed,
        "size": "1024x1024",
        "steps": 8,
        "rng": "numpy PCG64",
        "phase_seconds": {},
    }
    started = time.perf_counter()
    mx.reset_peak_memory()

    def stage(name, step=0):
        check_cancel(cancel)
        if progress is not None:
            progress(name, step, 8)

    try:
        stage("encoding")
        t = time.perf_counter()
        embeddings = encode_prompt(root, prompt, cancel=cancel)
        metrics["phase_seconds"]["encoding"] = time.perf_counter() - t
        embeddings = embeddings.astype(mx.bfloat16)
        mx.eval(embeddings)
        release_gpu()
        stage("loading_denoiser")
        t = time.perf_counter()
        model = load_transformer(root, cancel=cancel)
        metrics["phase_seconds"]["denoiser_load"] = time.perf_counter() - t
        latents = mx.array(noise).astype(mx.bfloat16)
        t = time.perf_counter()
        for i in range(8):
            stage("denoising", i)
            prediction = model(latents, embeddings, SIGMAS[i], 64, 64, cancel=cancel)
            latents = latents + prediction * mx.array(
                SIGMAS[i + 1] - SIGMAS[i], dtype=latents.dtype
            )
            mx.eval(latents)
            if not bool(mx.all(mx.isfinite(latents)).item()):
                raise ValueError("non-finite denoising latents")
        metrics["phase_seconds"]["denoising"] = time.perf_counter() - t
        model = prediction = embeddings = None
        release_gpu()
        stage("decoding", 8)
        t = time.perf_counter()
        vae = load_vae(root)
        pixels = vae.decode(
            latents.astype(mx.float32).reshape(1, 64, 64, 64).transpose(0, 3, 1, 2),
            cancel=cancel,
        )
        mx.eval(pixels)
        if pixels.shape != (1, 3, 1024, 1024) or not bool(
            mx.all(mx.isfinite(pixels)).item()
        ):
            raise ValueError("invalid decoded pixels")
        metrics["phase_seconds"]["vae_load_decode"] = time.perf_counter() - t
        check_cancel(cancel)
        array = np.array(mx.clip(pixels * 0.5 + 0.5, 0, 1)[0].transpose(1, 2, 0))
        output = io.BytesIO()
        Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(
            output, format="PNG"
        )
        metrics["mlx_peak_bytes"] = mx.get_peak_memory()
        metrics["total_seconds"] = time.perf_counter() - started
        return output.getvalue(), metrics
    finally:
        model = vae = embeddings = prediction = latents = pixels = None
        release_gpu()
