# SPDX-License-Identifier: Apache-2.0
"""Compare MLX VAE stages against externally generated PyTorch oracle arrays."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from omlx.image.pipeline import load_vae
from omlx.image.vae import VAEOperations
from tools.validate_image_conditioning import compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("reference", type=Path)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    arrays = np.load(args.reference)
    model = load_vae(args.bundle)
    x = mx.array(arrays["latent"])
    x = (
        x * mx.array(model._std)[None, :, None, None]
        + mx.array(model._mean)[None, :, None, None]
    )
    x = VAEOperations.conv(model.post_quant_conv, x, True)
    x = model.decoder.mid_block(VAEOperations.conv(model.decoder.conv_in, x, True))
    mx.eval(x)
    results = {"mid": compare(np.array(x), arrays["mid"])}
    for i, block in enumerate(model.decoder.up_blocks):
        x = block(x)
        mx.eval(x)
        results[f"up_{i}"] = compare(np.array(x), arrays[f"up_{i}"])
    pixels = model.decode(mx.array(arrays["latent"]))
    mx.eval(pixels)
    results["pixels"] = compare(np.array(pixels), arrays["pixels"])
    passed = all(x["passed"] for x in results.values())
    args.report.write_text(
        json.dumps(dict(passed=passed, stages=results), indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
