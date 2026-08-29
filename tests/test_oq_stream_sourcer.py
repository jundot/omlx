# SPDX-License-Identifier: Apache-2.0
"""Per-layer weight sourcing for the layer-streaming imatrix collector.

The streaming collector never holds the whole model: each MiniMaxDecoderLayer
is built bare, filled straight from the checkpoint through a private sanitize
plan, forwarded, and released. These tests pin the sourcing primitive:

- tensors land in the right block slots, bit-equal to an independent pop
- the strict loader accepts the exact popped key set (dense and MoE)
- a bare block forwards cachelessly with the collector's mask/position_ids
- one MoE layer stays inside the streaming memory budget and leaks nothing

Every test reads the real MiniMax-M3-MXFP8 checkpoint; the module skips when
it is not mounted. test_peak_rss_one_layer is defined first on purpose, so
its memory accounting runs before the module-scoped block fixture exists.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.oq import (
    _forward_layer_result,
    _iter_streamed_layer_blocks,
    _LazyTensorIndex,
    _prepare_layer_inputs,
    _streamed_embed_weight,
    _streamed_layer_items,
    _streamed_source_plan,
    _streamed_text_args,
)

M3_DIR = Path("/Volumes/Scratch/models/MiniMax-M3-MXFP8")
LAYER_PREFIX = "language_model.model.layers."

pytestmark = pytest.mark.skipif(
    not M3_DIR.is_dir(), reason="MiniMax-M3-MXFP8 checkpoint not present"
)


def _bits(a):
    """Numpy view of an mx array's raw bits, for exact equality checks."""
    uint_for_size = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}
    return np.array(a.view(uint_for_size[a.dtype.size]))


@pytest.fixture(scope="module")
def m3_config():
    return json.loads((M3_DIR / "config.json").read_text())


@pytest.fixture(scope="module")
def text_args():
    return _streamed_text_args(M3_DIR)


@pytest.fixture(scope="module")
def sourced_blocks(m3_config):
    """Layer 0 (dense) and layer 3 (first MoE + sparse index) via the generator."""
    blocks = {}
    for layer_idx, block, is_moe in _iter_streamed_layer_blocks(M3_DIR, m3_config):
        if layer_idx in (0, 3):
            blocks[layer_idx] = (block, is_moe)
        del block
        mx.clear_cache()
        if layer_idx >= 3:
            break
    yield blocks
    blocks.clear()
    mx.synchronize()
    mx.clear_cache()


def test_peak_rss_one_layer(m3_config):
    """Sourcing plus forwarding one MoE layer must fit the streaming budget.

    Runs first in the module so no other fixture holds weights: the peak is
    the honest cost of one streamed layer. The active-memory delta at the
    end is the leak tripwire for the collector's release idiom.
    """
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    baseline = mx.get_active_memory()

    block3 = None
    for layer_idx, block, is_moe in _iter_streamed_layer_blocks(M3_DIR, m3_config):
        if layer_idx == 3:
            assert is_moe
            block3 = block
        del block
        mx.clear_cache()
        if layer_idx >= 3:
            break
    assert block3 is not None

    hidden = m3_config["text_config"]["hidden_size"]
    seq = 512
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq).astype(mx.bfloat16)
    position_ids = mx.arange(seq)[None, :]
    mx.random.seed(11)
    for _ in range(4):
        x = (0.1 * mx.random.normal((1, seq, hidden))).astype(mx.bfloat16)
        out, _ = _forward_layer_result(block3, x, mask, position_ids)
        assert out is not None, "no forward signature matched for the MoE block"
        mx.eval(out)
        del out, x

    del block3
    mx.synchronize()
    mx.clear_cache()

    peak = mx.get_peak_memory()
    active_after = mx.get_active_memory()
    print(
        f"\nstreamed layer 3: peak {peak / 1e9:.2f} GB, "
        f"baseline {baseline / 1e9:.2f} GB, after release {active_after / 1e9:.2f} GB"
    )
    assert peak < 30e9, f"peak {peak / 1e9:.2f} GB blows the 30 GB streaming budget"
    assert active_after - baseline < 1e9, (
        f"active memory did not return to baseline: "
        f"{baseline / 1e9:.2f} GB -> {active_after / 1e9:.2f} GB"
    )


def test_sourced_layer_weights_match_ground_truth(m3_config, sourced_blocks):
    """load_weights must route every tensor to the right slot, bit for bit.

    Ground truth is an independent _DiscoveredPlan pop of the same keys:
    same dequant path, separate instance, so a routing bug in the block
    fill cannot cancel out.
    """
    truth = _streamed_source_plan(M3_DIR, m3_config)
    for layer_idx in (0, 3):
        block, _ = sourced_blocks[layer_idx]
        params = dict(tree_flatten(block.parameters()))
        prefix = f"{LAYER_PREFIX}{layer_idx}."
        keys = sorted(k for k in truth if k.startswith(prefix))
        assert keys, f"no plan keys for layer {layer_idx}"
        for key in keys:
            rel = key[len(prefix) :]
            assert rel in params, f"{rel} not a parameter of the bare block"
            expected = truth.pop(key)
            got = params[rel]
            assert got.dtype == expected.dtype, key
            assert got.shape == expected.shape, key
            assert np.array_equal(_bits(got), _bits(expected)), f"{key} differs"
            del expected
            mx.clear_cache()


def test_strict_load_accepts_layer_keyset(text_args, m3_config):
    """The popped key set loads under strict=True for dense and MoE layers.

    Settles the open question about e_score_correction_bias and the
    index_q/k_proj sparse tensors: the checkpoint keys must equal the
    block's parameter tree exactly, with no lenient fallback needed.
    """
    args, layer_cls = text_args
    dp = _streamed_source_plan(M3_DIR, m3_config)
    for layer_idx in (0, 3):
        items = _streamed_layer_items(dp, layer_idx)
        block = layer_cls(args, layer_idx)
        block.load_weights(items, strict=True)
        loaded = {k for k, _ in items}
        expected = {k for k, _ in tree_flatten(block.parameters())}
        assert loaded == expected, (
            f"layer {layer_idx}: missing={sorted(expected - loaded)} "
            f"extra={sorted(loaded - expected)}"
        )
        del block, items
        mx.clear_cache()


def test_bare_block_forwards(text_args, sourced_blocks):
    """A cacheless prefill through a bare block must produce clean output."""
    args, _ = text_args
    mx.random.seed(7)
    x = (0.1 * mx.random.normal((1, 8, args.hidden_size))).astype(mx.bfloat16)
    calib = mx.zeros((1, 8), dtype=mx.int32)

    class _NoModel:
        """Stand-in for the model arg: streaming has no whole model."""

    for layer_idx in (0, 3):
        block, is_moe = sourced_blocks[layer_idx]
        assert is_moe == (layer_idx == 3)
        inputs, masks, position_ids = _prepare_layer_inputs(
            _NoModel(), [block], calib, x
        )
        out, _ = _forward_layer_result(block, inputs, masks[0], position_ids)
        assert out is not None, f"layer {layer_idx}: no forward signature matched"
        mx.eval(out)
        assert out.shape == x.shape
        assert not bool(mx.isnan(out).any()), f"layer {layer_idx}: NaN in output"
        assert not bool(mx.isinf(out).any()), f"layer {layer_idx}: Inf in output"


def test_embed_weight_matches_checkpoint(m3_config):
    """Stage-0 embedding sourcing: bf16, right shape, bit-equal to disk."""
    weight = _streamed_embed_weight(M3_DIR, m3_config)
    text_cfg = m3_config["text_config"]
    assert weight.dtype == mx.bfloat16
    assert weight.shape == (text_cfg["vocab_size"], text_cfg["hidden_size"])

    idx = _LazyTensorIndex(sorted(M3_DIR.glob("*.safetensors")))
    raw = idx["language_model.model.embed_tokens.weight"]
    mx.eval(raw)
    assert np.array_equal(_bits(weight), _bits(raw))
