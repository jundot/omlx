# SPDX-License-Identifier: Apache-2.0
"""Compare fresh native conditioning against retained independent FP32 outputs.

Reference .npy files must originate from an independently executed source-layout
reference, not from this implementation. This script neither downloads weights
nor overwrites reference data. Run serialized with other GPU validation.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from omlx.image.conditioning import encode_prompt, prompt_tokens

PROMPTS = [
    "A red ceramic teapot on a blue table, soft daylight.",
    'A shop sign with the Chinese text "你好世界", clean typography.',
    "A watercolor landscape with two small boats on a misty lake, mountains behind them, orange sunrise reflected in the water.",
]


def compare(actual, expected):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    relative = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
    cosine = float(np.sum(a * b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    finite = bool(np.isfinite(a).all())
    return dict(
        relative_l2=relative,
        cosine=cosine,
        passed=finite and relative < 0.005 and cosine > 0.99999,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    tokens = np.load(args.references / "tokens.npz")
    reports = []
    for i, prompt in enumerate(PROMPTS):
        ids, drop = prompt_tokens(args.bundle.resolve(), prompt)
        length = int(tokens["valid"][i].sum())
        if ids != tokens["ids"][i, :length].tolist() or drop != int(tokens["drop"]):
            raise ValueError("reference tokenization differs")
        layers = []

        def on_layer(layer, hidden, i=i, length=length, layers=layers):
            reference = np.load(args.references / f"ref_layer_{layer:02}.npy")[
                i : i + 1, :length
            ]
            result = compare(np.array(hidden), reference)
            layers.append(result)
            print(f"prompt {i+1} layer {layer+1}: {result}", flush=True)

        result = encode_prompt(args.bundle, prompt, on_layer=on_layer)
        final = compare(
            np.array(result),
            np.load(args.references / "ref_final.npy")[i : i + 1, drop:length],
        )
        reports.append(dict(layers=layers, final=final))
        mx.synchronize()
        mx.clear_cache()
    passed = all(
        row["final"]["passed"] and all(x["passed"] for x in row["layers"])
        for row in reports
    )
    args.report.write_text(
        json.dumps(dict(passed=passed, prompts=reports), indent=2) + "\n"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
