# SPDX-License-Identifier: Apache-2.0
"""Token-exactness gate for the Qwen3.8-27B native-MTP port.

The challenge's token-fidelity contract: every token an MTP decode emits must
equal the token serial decode would have produced, token for token, from the
same seed. This test loads the merged pinned checkpoint
(~/qwen38-mtp/merged — backbone + head under the mtp. prefix, built by
tools/qwen38_mtp/merge_checkpoint.py) through omlx's Lightning MTP machinery
and compares the greedy MTP trajectory against the serial trajectory over the
challenge's public long-copy gate prompt.

The test is skipped when the merged checkpoint is absent so the suite stays
green in environments that have not staged the ~16 GB artifact. It is a real,
weight-carrying gate when the artifact is present.

Usage:
    Q38_MERGED=/path/to/merged pytest tests/test_qwen38_mtp_token_exact.py -s
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from omlx.patches.mlx_lm_mtp import (
    apply_mlx_lm_mtp_patch,
    set_mtp_active,
    set_mtp_depth,
)

MERGED = os.environ.get(
    "Q38_MERGED", os.path.expanduser("~/qwen38-mtp/merged")
)
PROMPT_FILE = os.environ.get(
    "Q38_PROMPT",
    os.path.expanduser(
        "~/qwen38-mtp/challenge/correctness_prompts/"
        "public_longcopy_gate_english_512.txt"
    ),
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MERGED) or not os.path.exists(PROMPT_FILE),
    reason="Qwen3.8-27B merged checkpoint / challenge prompt not staged",
)

_MAX_TOKENS = int(os.environ.get("Q38_MAX_TOKENS", "64"))


def _decode(model, tokenizer, max_tokens: int):
    from mlx_lm.generate import batch_generate

    with open(PROMPT_FILE, encoding="utf-8") as f:
        prompt = f.read().strip()
    ids = tokenizer.encode(prompt)
    resp = batch_generate(
        model,
        tokenizer,
        prompts=[ids],
        max_tokens=[max_tokens],
        return_token_ids=True,
        verbose=False,
    )
    return resp.token_ids[0]


def _load(mtp: bool, depth: int):
    from mlx_lm import load

    set_mtp_active(mtp)
    if depth is not None:
        set_mtp_depth(depth)
    return load(MERGED)


def test_assets_staged():
    """The merged tree must carry the 15 mtp.* head tensors."""
    import mlx.core as mx

    weights = mx.load(os.path.join(MERGED, "model-00004-of-00004.safetensors"))
    mtp_keys = [k for k in weights if k.startswith("mtp.")]
    assert len(mtp_keys) == 15, f"expected 15 mtp.* tensors, got {len(mtp_keys)}"


def test_mtp_decode_token_exact_vs_serial():
    """MTP depth-2 decode must emit the serial token trajectory exactly."""
    model_s, tokenizer = _load(False, None)
    serial = _decode(model_s, tokenizer, _MAX_TOKENS)

    model_m, _ = _load(True, 2)
    mtp = _decode(model_m, tokenizer, _MAX_TOKENS)

    assert len(mtp) == len(serial), (
        f"length mismatch: serial {len(serial)} vs mtp {len(mtp)}"
    )
    mismatches = [i for i, (a, b) in enumerate(zip(serial, mtp)) if a != b]
    assert not mismatches, (
        f"token divergence at {mismatches[:10]}; serial={serial[:20]} "
        f"mtp={mtp[:20]}"
    )
