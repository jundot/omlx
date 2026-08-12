# SPDX-License-Identifier: Apache-2.0
"""Synthetic DeepSeek V4 expert-offload tests — no checkpoint download.

Builds a tiny ``DeepseekV4Model`` (3 layers x 16 routed experts) with
affine 3-bit switch projections — the same ``weight/scales/biases`` field
layout the real oQ-quantized checkpoints use — dumps it to safetensors
with the real checkpoint naming (``model.layers.N.ffn.switch_mlp.*``),
and verifies the extended ``apply_moe_expert_offload``:

* wraps every MoE layer (the DeepSeek V4 native-kernel families were never
  walked by the type-based detection — different SwitchGLU class identity);
* decode and sorted-prefill outputs stay bit-identical to the resident
  model (native block kernels do not activate for group-32 affine, so both
  sides run ``gather_qmm``);
* the expert cache is exercised (misses > 0).

The real 92.8 GB checkpoint (group-64 oQ, native kernels, MTP head) is
covered by the opt-in live acceptance once it is cached locally.
"""

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
from omlx.patches.moe_expert_offload import (
    apply_moe_expert_offload,
    moe_offload_stats,
)

apply_deepseek_v4_patch()

from mlx_lm.models.deepseek_v4 import DeepseekV4Model, ModelArgs  # noqa: E402

_N_LAYERS = 3
_N_EXPERTS = 16


@pytest.fixture
def dsv4_model():
    """Tiny DS4 model with quantized (affine 3-bit) switch projections."""
    config = ModelArgs.from_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 64,
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 64,
            "num_hidden_layers": _N_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "n_shared_experts": 1,
            "n_routed_experts": _N_EXPERTS,
            "num_experts_per_tok": 2,
            "num_hash_layers": 0,
            "q_lora_rank": 4,
            "qk_rope_head_dim": 4,
            "head_dim": 8,
            "o_groups": 2,
            "o_lora_rank": 4,
            "index_n_heads": 2,
            "index_head_dim": 4,
            "index_topk": 2,
            "hc_mult": 4,
            "compress_ratios": [0] * 6,
            "num_nextn_predict_layers": 1,
            "dspark_block_size": 3,
            "dspark_noise_token_id": 63,
            "dspark_target_layer_ids": list(range(_N_LAYERS)),
            "dspark_markov_rank": 4,
        }
    )
    model = DeepseekV4Model(config)
    for block in model.layers:
        smlp = block.ffn.switch_mlp
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(smlp, proj)
            if hasattr(lin, "to_quantized"):  # unquantized SwitchLinear
                setattr(
                    smlp, proj,
                    lin.to_quantized(group_size=32, bits=3, mode="affine"),
                )
    return model


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, p))
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                out.update(_flatten({str(i): item}, p))
        else:
            out[p] = v
    return out


def _dump_checkpoint(model, out_dir: Path) -> int:
    """Save the model with real checkpoint naming (model.layers.N.ffn...)."""
    arrays = {}
    for name, arr in _flatten(model.parameters()).items():
        key = ("model." if not name.startswith("mtp") else "") + name
        np_arr = np.array(arr)
        if np_arr.dtype == object or np_arr.dtype.kind not in "fiub":
            continue
        arrays[key] = np_arr
    from safetensors.numpy import save_file

    save_file(arrays, str(out_dir / "model-00001-of-00001.safetensors"))
    with open(out_dir / "config.json", "w") as f:
        json.dump({"model_type": "deepseek_v4"}, f)
    return len(arrays)


def _moe_output(layer, x, ids):
    out = layer.ffn(x, ids)
    mx.eval(out)
    return np.array(out)


def test_wraps_every_moe_layer(dsv4_model):
    model = dsv4_model
    with tempfile.TemporaryDirectory() as tmp:
        assert _dump_checkpoint(model, Path(tmp)) > 0
        wrapped = apply_moe_expert_offload(model, tmp, resident_fraction=0.25)
        assert wrapped == _N_LAYERS, f"expected {_N_LAYERS}, got {wrapped}"


def test_decode_and_prefill_bit_exact(dsv4_model):
    model = dsv4_model
    twin = copy.deepcopy(model)  # resident reference
    with tempfile.TemporaryDirectory() as tmp:
        _dump_checkpoint(model, Path(tmp))
        assert apply_moe_expert_offload(model, tmp, resident_fraction=0.25) > 0

        mx.random.seed(42)
        # decode-size: 8 tokens x 2 experts = 16 < 64 (no sort path)
        x = mx.random.normal((1, 8, 64))
        ids = mx.array([[3, 7, 11, 15, 2, 9, 5, 13]])
        for i in range(_N_LAYERS):
            a = _moe_output(model.layers[i], x, ids)
            b = _moe_output(twin.layers[i], x, ids)
            assert np.array_equal(a, b), f"layer {i} decode differs"

        # prefill-size: 40 tokens x 2 = 80 >= 64 (sorted gather path)
        x = mx.random.normal((1, 40, 64))
        ids = mx.random.randint(0, 64, (1, 40))
        for i in range(_N_LAYERS):
            a = _moe_output(model.layers[i], x, ids)
            b = _moe_output(twin.layers[i], x, ids)
            assert np.array_equal(a, b), f"layer {i} prefill differs"


def test_cache_exercised(dsv4_model):
    model = dsv4_model
    with tempfile.TemporaryDirectory() as tmp:
        _dump_checkpoint(model, Path(tmp))
        assert apply_moe_expert_offload(model, tmp, resident_fraction=0.25) > 0

        x = mx.random.normal((1, 40, 64))
        ids = mx.random.randint(0, 64, (1, 40))
        for i in range(_N_LAYERS):
            _moe_output(model.layers[i], x, ids)

        stats = moe_offload_stats(model)
        assert stats["layers"] == _N_LAYERS
        assert stats["misses"] > 0, "cache never exercised"


def test_shrink_after_prefill_preserves_outputs(dsv4_model):
    """Page-cache cooperation: post-prefill shrink keeps outputs identical."""
    import os

    from omlx.patches.moe_expert_offload import OffloadSwitchGLU

    model = dsv4_model
    twin = copy.deepcopy(model)
    with tempfile.TemporaryDirectory() as tmp:
        _dump_checkpoint(model, Path(tmp))
        os.environ["OMLX_MOE_OFFLOAD_SHRINK_AFTER_PREFILL"] = "0.5"
        try:
            # 0.75 -> capacity 12 (> the 8 floor), so the 0.5 shrink
            # actually releases slots down to 8
            assert apply_moe_expert_offload(model, tmp, resident_fraction=0.75) > 0
        finally:
            os.environ.pop("OMLX_MOE_OFFLOAD_SHRINK_AFTER_PREFILL", None)

        # prefill-style call with a wide expert working set (20 distinct >
        # capacity 12) forces the chunked path, which triggers the shrink.
        # Chunked prefill re-sorts per chunk, so the contract is the PR's
        # rounding-bounded one (~1e-3), not bit-exactness (decode below is).
        x = mx.random.normal((1, 40, 64))
        wide = mx.array([[i % 16, (i + 1) % 16] for i in range(40)])  # (40, 2)
        a = model.layers[0].ffn.switch_mlp(x, wide)
        b = twin.layers[0].ffn.switch_mlp(x, wide)
        mx.eval(a, b)
        a_np, b_np = np.array(a), np.array(b)
        assert np.allclose(a_np, b_np, rtol=1e-3, atol=1e-3), (
            "prefill changed by shrink beyond kernel rounding"
        )

        # shrink actually released slots (12 -> 8)
        cache = model.layers[0].ffn.switch_mlp.cache
        assert isinstance(model.layers[0].ffn.switch_mlp, OffloadSwitchGLU)
        assert cache.capacity == 8, cache.capacity

        # decode after shrink stays bit-exact
        x = mx.random.normal((1, 8, 64))
        narrow = mx.array([[0, 1]] * 8)
        a = model.layers[0].ffn.switch_mlp(x, narrow)
        b = twin.layers[0].ffn.switch_mlp(x, narrow)
        mx.eval(a, b)
        assert np.array_equal(np.array(a), np.array(b)), "post-shrink decode differs"


def test_pin_hot_survives_eviction(dsv4_model):
    """Learned hot-pin: most-routed experts must survive LRU pressure.

    Exercised at the cache level (direct ensure calls) because the tiny
    model's gate routes to the same experts regardless of input tokens —
    the model-level hammer never forces evictions.
    """
    model = dsv4_model
    with tempfile.TemporaryDirectory() as tmp:
        _dump_checkpoint(model, Path(tmp))
        assert apply_moe_expert_offload(model, tmp, resident_fraction=0.25) > 0
        cache = model.layers[0].ffn.switch_mlp.cache

        # skew usage toward experts 0..3, then pin the top 4
        for _ in range(5):
            cache.ensure(mx.array([0, 1, 2, 3]))
        cache.pin_hot(4)
        pinned = set(cache.pinned)
        assert pinned == {0, 1, 2, 3}, pinned

        # hammer with distinct experts: 4 free slots, then evictions
        for e in (4, 6, 8, 10, 12, 14):
            cache.ensure(mx.array([e]))

        still = pinned & set(cache.slot_of)
        assert still == pinned, f"pinned experts evicted: {pinned - still}"
