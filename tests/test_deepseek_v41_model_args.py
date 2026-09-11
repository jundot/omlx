# SPDX-License-Identifier: Apache-2.0
"""ModelArgs parsing tests against the real V4.1-Flash config.json."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CONFIG = Path("/Volumes/TB5/llm/DeepSeek-V4.1-Flash/config.json")


@pytest.fixture(scope="module")
def raw_config():
    if not CONFIG.exists():
        pytest.skip(f"missing {CONFIG}")
    return json.loads(CONFIG.read_text())


def test_flatten_and_parse_model_args(raw_config):
    from omlx.patches.deepseek_v41 import apply_deepseek_v41_patch
    from omlx.patches.deepseek_v41.utils_patch import flatten_text_config

    apply_deepseek_v41_patch()
    flat = flatten_text_config(raw_config)
    import mlx_lm.models.deepseek_v41 as m

    args = m.ModelArgs.from_dict(flat)
    assert args.model_type.startswith("deepseek_v41")
    assert args.hidden_size == 5120
    assert args.q_lora_rank == 1280
    assert abs(args.rms_norm_eps - 1e-20) < 1e-30
    assert args.n_routed_experts == 384
    assert set(args.compress_ratios) <= {0, 1, 2}
    assert 2 in args.kv_source_layer_ids or 2 in getattr(args, "kv_source_layers", [])
    # candidate source
    assert getattr(args, "candidate_source_layer_id", getattr(args, "candidate_source_layer", -1)) == 20


def test_v4_compress_ratios_rejected_on_v41_parser():
    from omlx.patches.deepseek_v41 import apply_deepseek_v41_patch

    apply_deepseek_v41_patch()
    import mlx_lm.models.deepseek_v41 as m

    with pytest.raises(ValueError):
        m.ModelArgs(
            compress_ratios=[0, 4, 128] + [0] * 37,
            num_hidden_layers=40,
        )
