# SPDX-License-Identifier: Apache-2.0
"""Independent PyTorch/NCHW functional VAE decoder oracle for native weights.

Reference-environment tool only: PyTorch is not an inference dependency. This
executes the equations independently from the MLX classes, transposing exported
OHWI kernels back to OIHW. Source/export inverse-layout parity is a prerequisite.
Small deterministic latents exercise every decoder stage. No image-quality claim.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def reference(bundle, output):
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    cfg = json.loads((bundle / "vae/config.json").read_text())
    raw = load_file(str(bundle / "vae/model.safetensors"))
    weights = {
        k: v.float()
        for k, v in raw.items()
        if k.startswith(("decoder.", "post_quant_conv."))
    }
    del raw

    def conv(x, name):
        weight = weights[name + ".weight"].permute(0, 3, 1, 2).contiguous()
        return functional.conv2d(
            x, weight, weights[name + ".bias"], padding=weight.shape[-1] // 2
        )

    def norm(x, name):
        return (
            functional.normalize(x, p=2, dim=1, eps=1e-12)
            * x.shape[1] ** 0.5
            * weights[name + ".gamma"][None, :, None, None]
        )

    def res(x, name):
        shortcut = (
            conv(x, name + ".conv_shortcut")
            if name + ".conv_shortcut.weight" in weights
            else x
        )
        z = conv(functional.silu(norm(x, name + ".norm1")), name + ".conv1")
        return (
            conv(functional.silu(norm(z, name + ".norm2")), name + ".conv2") + shortcut
        )

    def attention(x, name):
        q, k, v = (
            conv(norm(x, name + ".norm"), name + ".to_qkv").flatten(2).chunk(3, dim=1)
        )
        scores = torch.bmm(q.transpose(1, 2), k) / q.shape[1] ** 0.5
        attended = (
            torch.bmm(scores.softmax(-1), v.transpose(1, 2))
            .transpose(1, 2)
            .reshape(x.shape)
        )
        return conv(attended, name + ".proj") + x

    def duplicate(x, out_channels, ft):
        b, c, h, w = x.shape
        repeated = x.repeat_interleave(out_channels * ft * 4 // c, dim=1)
        return (
            repeated.reshape(b, out_channels, ft, 2, 2, 1, h, w)
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .reshape(b, out_channels, ft, h * 2, w * 2)[:, :, ft - 1]
        )

    latent = np.random.default_rng(123).standard_normal((1, 64, 2, 2), dtype=np.float32)
    x = (
        torch.from_numpy(latent) * torch.tensor(cfg["latents_std"])[None, :, None, None]
        + torch.tensor(cfg["latents_mean"])[None, :, None, None]
    )
    x = conv(x, "post_quant_conv")
    x = conv(x, "decoder.conv_in")
    x = res(x, "decoder.mid_block.resnets.0")
    x = attention(x, "decoder.mid_block.attentions.0")
    x = res(x, "decoder.mid_block.resnets.1")
    records = {"latent": latent, "mid": x.numpy().copy()}
    temporal = list(reversed(cfg["temperal_downsample"]))
    for i, out_dim in enumerate(reversed(cfg["dim_mult"])):
        name = f"decoder.up_blocks.{i}"
        original = x
        for j in range(cfg["num_res_blocks"] + 1):
            x = res(x, f"{name}.resnets.{j}")
        if i < len(cfg["dim_mult"]) - 1:
            shortcut = duplicate(
                original, cfg["decoder_base_dim"] * out_dim, 2 if temporal[i] else 1
            )
            x = (
                conv(
                    functional.interpolate(x, scale_factor=2, mode="nearest"),
                    name + ".upsampler.resample.1",
                )
                + shortcut
            )
        records[f"up_{i}"] = x.numpy().copy()
    x = conv(functional.silu(norm(x, "decoder.norm_out")), "decoder.conv_out")[:, :3]
    records["pixels"] = x.numpy()
    np.savez(output, **records)
    print(
        json.dumps(
            {
                "shape": list(x.shape),
                "finite": bool(torch.isfinite(x).all()),
                "torch": torch.__version__,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("output already exists")
    reference(args.bundle, args.output)
