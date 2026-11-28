"""MLX side of the attentive-correction oracle: compare against AngelSpec output."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from omlx.patches.deepseek_v41_dfly2.dfly2_context import DFly2LayerContext
from omlx.patches.deepseek_v41_dfly2.dfly2_layer import attentive_correction


def _config():
    return SimpleNamespace(
        n_heads=2,
        head_dim=16,
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=8,
        rms_norm_eps=float(1e-5),
        rope_theta=10000.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    args = parser.parse_args()

    data = np.load(args.npz)
    attention = "hidden_correction.attention."
    weights = {
        "hidden_correction.hidden_norm.weight": mx.array(data["hidden_norm_weight"]),
        "hidden_correction.embed_norm.weight": mx.array(data["embed_norm_weight"]),
        # The inherited SwiGLU branch is zeroed so only the attentive part is
        # compared: down_proj zero makes the whole term vanish.
        "hidden_correction.gate_proj.weight": mx.zeros((16, 64)),
        "hidden_correction.up_proj.weight": mx.zeros((16, 64)),
        "hidden_correction.down_proj.weight": mx.zeros((32, 16)),
    }
    for key in data.files:
        if key.startswith("param::"):
            weights[attention + key.removeprefix("param::")] = mx.array(data[key])

    hidden = mx.array(data["hidden"])
    previous = mx.array(data["previous"])
    # The reference lists picks per slot; our context carries one gathered set.
    # Gathering the table per slot makes the two contracts identical.
    table = np.array(data["latent"])[0]
    picks = data["indices"][0]
    reference = data["out"]
    config = _config()

    errors = []
    for slot in range(hidden.shape[1]):
        chosen = picks[slot]
        valid = chosen >= 0
        rows = np.zeros((1, len(chosen), table.shape[-1]), dtype=table.dtype)
        rows[0, valid] = table[chosen[valid]]
        history = DFly2LayerContext(
            indices=mx.array(chosen.astype("int32"))[None],
            valid=mx.array(valid)[None],
            latents=mx.array(rows),
        )
        out = attentive_correction(
            hidden, slot, previous, history, weights, config, start_position=10
        )
        mx.eval(out)
        ours = np.array(out)[0] - np.array(hidden)[0, slot]
        errors.append(np.abs(ours - reference[0, slot]).max())

    print({"max_abs_error": float(max(errors)), "per_slot": [float(e) for e in errors]})


if __name__ == "__main__":
    main()
