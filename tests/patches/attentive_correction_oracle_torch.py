"""Torch side of the attentive-correction oracle: run AngelSpec's own module.

Writes every parameter plus the inputs and the reference output to an .npz so the
MLX side can be compared without importing torch and mlx into one process.
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "/tmp/angelspec")
from angelspec.models.draft.dsv41_attentive_correction import DSV41AttentiveCorrection  # noqa: E402

HIDDEN = 32
HEADS = 2
HEAD_DIM = 16
ROPE_DIM = 8
Q_LORA = 16
O_GROUPS = 2
O_LORA = 8
BLOCK = 6
CTX = 5


def _config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE_DIM,
        q_lora_rank=Q_LORA,
        o_groups=O_GROUPS,
        o_lora_rank=O_LORA,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        original_seq_len=0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        attentive_correction_block_keys=True,
        attentive_correction_block_mode="gated",
        attentive_correction_block_kv_in="hidden_embed",
        attentive_correction_backend="torch",
        dsa_kernel=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    module = DSV41AttentiveCorrection(_config()).float().eval()

    # Non-zero weights everywhere: zero-init paths would hide a mismatch.
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("wo_b.weight"):
                parameter.copy_(torch.randn_like(parameter) * 0.1)
            elif parameter.ndim == 1:
                parameter.copy_(torch.randn_like(parameter) * 0.1)
            else:
                parameter.copy_(torch.randn_like(parameter) * 0.1)

    hidden = torch.randn(1, BLOCK, HIDDEN)
    previous = torch.randn(1, BLOCK, HIDDEN)
    # The reference takes already-normalized inputs; the model owns these two
    # norms, so the oracle generates them and records them for the MLX side,
    # which normalizes internally.
    hidden_norm_w = torch.rand(HIDDEN) + 0.5
    embed_norm_w = torch.rand(HIDDEN) + 0.5
    norm_eps = 1e-5

    def rms_norm(x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm_eps) * weight

    hidden_normed = rms_norm(hidden, hidden_norm_w)
    previous_normed = rms_norm(previous, embed_norm_w)
    positions = torch.arange(10, 10 + BLOCK).reshape(1, BLOCK)
    latent = torch.randn(1, CTX, HEAD_DIM)
    indices = torch.tensor([[[0, 1, 2, -1], [1, 2, 3, -1], [2, 3, 4, -1],
                             [3, 4, -1, -1], [0, 2, 4, -1], [0, 1, -1, -1]]])

    with torch.no_grad():
        out, alive = module(hidden_normed, previous_normed, positions, latent, indices, block_size=BLOCK)

    payload = {f"param::{name}": value.detach().numpy() for name, value in module.named_parameters()}
    payload.update(
        hidden=hidden.numpy(),
        previous=previous.numpy(),
        positions=positions.numpy(),
        latent=latent.numpy(),
        indices=indices.numpy(),
        out=out.numpy(),
        hidden_norm_weight=hidden_norm_w.numpy(),
        embed_norm_weight=embed_norm_w.numpy(),
        norm_eps=np.array(norm_eps),
        alive=alive.numpy(),
    )
    np.savez(args.out, **payload)
    print({"wrote": args.out, "params": len(module.state_dict()), "out_shape": list(out.shape)})


if __name__ == "__main__":
    main()
