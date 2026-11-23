# SPDX-License-Identifier: Apache-2.0
"""LiLiCorr drafter wiring: registration, sanitizer, lattice selection, target."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.speculative import dflash as dd
from omlx.speculative.dflash import lilicorr, selection

VOCAB = 64
HIDDEN = 32
TARGET_LAYERS = 6
TARGET_LAYER_IDS = [1, 3, 5]
BOLT_LAYER_IDS = [1, 18, 35, 51, 68, 85]
WINDOW = 12
BLOCK = 4
TOP_K = 4


def _config(target_layer_ids=TARGET_LAYER_IDS, num_target_layers=None, **overrides):
    params = {
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "sliding_window": WINDOW,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "tie_word_embeddings": False,
        "block_size": BLOCK,
        "dflash_config": {
            "block_size": BLOCK,
            "conv_group_size": 8,
            "conv_kernel_size": 2,
            "mask_token_id": VOCAB - 1,
            "target_layer_ids": target_layer_ids,
            "projector_type": "lilicorr",
            "lilicorr_enabled": True,
            "lilicorr_hidden_size": 16,
            "lilicorr_candidate_topk": TOP_K,
            "lilicorr_num_heads": 2,
            "lilicorr_num_layers": 1,
            "lilicorr_mlp_ratio": 2.0,
            "lilicorr_factor_dim": 16,
            "lilicorr_vector_eps": 1e-4,
            "lilicorr_logit_scale": 8.0,
        },
    }
    if num_target_layers is not None:
        params["num_target_layers"] = num_target_layers
    params.update(overrides)
    return lilicorr.LiLiCorrConfig.from_dict(params)


def _target(layer_count=TARGET_LAYERS):
    embed = nn.Embedding(VOCAB, HIDDEN)
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    language = SimpleNamespace(
        config={
            "text_config": {
                "hidden_size": HIDDEN,
                "num_hidden_layers": layer_count,
                "vocab_size": VOCAB,
            }
        },
        model=SimpleNamespace(layers=[object()] * layer_count, embed_tokens=embed),
        lm_head=lm_head,
        rollback_speculative_cache=lambda *args, **kwargs: None,
    )
    mx.eval(embed.parameters(), lm_head.parameters())
    return SimpleNamespace(language_model=language)


def _tiny_drafter(seed=0):
    mx.random.seed(seed)
    model = lilicorr.LiLiCorrDraftModel(_config())
    params = {
        key: mx.random.normal(value.shape) * 0.2
        for key, value in nn.utils.tree_flatten(model.parameters())
    }
    model.load_weights(list(params.items()), strict=False)
    model.bind(_target())
    mx.eval(model.parameters())
    return dd.DFlashDrafter(model, block_size=BLOCK, source_path="tiny")


# -- registration ---------------------------------------------------------


def test_patch_registers_the_lilicorr_drafter():
    import mlx_vlm.speculative.drafters as drafters
    import mlx_vlm.utils as vlm_utils

    assert lilicorr.apply_lilicorr_patch() is True
    assert sys.modules["mlx_vlm.speculative.drafters.lilicorr"] is lilicorr
    assert importlib.import_module("mlx_vlm.speculative.drafters.lilicorr") is (
        lilicorr
    )
    assert drafters.lilicorr is lilicorr
    assert drafters.DRAFTER_KIND_BY_MODEL_TYPE["lilicorr"] == "dflash"
    # The model-type wrapper is idempotent and only claims LiLiCorr payloads.
    wrapped = vlm_utils.get_model_type
    assert lilicorr.apply_lilicorr_patch() is True
    assert vlm_utils.get_model_type is wrapped
    assert vlm_utils.get_model_type(
        {"architectures": ["LiLiCorrDraftModel"], "model_type": "qwen3"}
    ) == "lilicorr"
    assert vlm_utils.get_model_type(
        {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
    ) == "qwen3"


def test_modelopt_fp4_tag_is_translated_only_for_lilicorr_weights():
    from mlx_vlm.quantization import checkpoint

    lilicorr.apply_lilicorr_patch()
    quantization_config = {
        "quant_method": "modelopt_fp4",
        "quant_algo": "W4A16_NVFP4",
        "group_size": 16,
    }
    weights = {
        "lilicorr.slot_embedding": mx.zeros((1, 1, 3, 1, 16), dtype=mx.bfloat16),
        "a.weight": mx.zeros((4, 8), dtype=mx.uint8),
        "a.weight_scale": mx.zeros((4, 1), dtype=mx.uint8),
        "a.weight_scale_2": mx.array(1.0, dtype=mx.float32),
    }
    transformed, quantization = checkpoint._transform_modelopt_nvfp4_weights(
        dict(weights), dict(quantization_config)
    )
    assert quantization == {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    assert transformed["a.weight"].dtype == mx.uint32
    assert transformed["a.scales"].dtype == mx.uint8

    # Another architecture's weights keep the unknown tag, so the stock
    # transform leaves them alone instead of repacking them.
    plain = {
        key: value
        for key, value in weights.items()
        if not key.startswith("lilicorr.")
    }
    untouched, quantization = checkpoint._transform_modelopt_nvfp4_weights(
        plain, dict(quantization_config)
    )
    assert quantization is None
    assert untouched["a.weight"].dtype == mx.uint8


# -- sanitizer ------------------------------------------------------------


def test_sanitize_expands_slot_biases_and_maps_keys():
    model = lilicorr.LiLiCorrDraftModel(_config())
    heads = int(model.config.lilicorr_num_heads)
    slots = BLOCK - 1
    width = int(model.config.lilicorr_hidden_size)
    nodes = slots * TOP_K
    mx.random.seed(1)
    relative = mx.random.normal((heads, 2 * slots + 1))
    same = mx.random.normal((heads,))
    weights = {
        "model.lilicorr.relative_slot_bias": relative,
        "model.lilicorr.same_slot_bias": same,
        "model.lilicorr.slot_embedding": mx.zeros((1, 1, slots, 1, width)),
        "model.lilicorr.rank_embedding": mx.zeros((1, 1, 1, TOP_K, width)),
        "model.lilicorr.layers.0.attn.in_proj_weight": mx.zeros((3 * width, width)),
        "model.lilicorr.layers.0.attn.in_proj_bias": mx.zeros((3 * width,)),
    }
    out = model.sanitize(weights)
    assert out["lilicorr.slot_embedding.weight"].shape == (slots, width)
    assert out["lilicorr.rank_embedding.weight"].shape == (TOP_K, width)
    assert out["lilicorr.layers.0.attn.in_proj.weight"].shape == (3 * width, width)
    assert out["lilicorr.layers.0.attn.in_proj.bias"].shape == (3 * width,)
    assert "model.lilicorr.relative_slot_bias" not in out
    assert "lilicorr.relative_slot_bias" not in out
    assert "lilicorr.same_slot_bias" not in out

    bias = out["lilicorr.attention_bias"]
    assert bias.shape == (heads, nodes, nodes)
    slots_per_node = [node // TOP_K for node in range(nodes)]
    for head in range(heads):
        for query in range(nodes):
            for key in range(nodes):
                offset = slots_per_node[query] - slots_per_node[key] + slots
                expected = relative[head, offset]
                if slots_per_node[query] == slots_per_node[key]:
                    expected = expected + same[head]
                assert mx.allclose(bias[head, query, key], expected)


def test_sanitize_requires_the_slot_biases():
    model = lilicorr.LiLiCorrDraftModel(_config())
    with pytest.raises(ValueError, match="slot attention biases"):
        model.sanitize({"lilicorr.same_slot_bias": mx.zeros((2,))})


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"block_size": 1}, "block_size >= 2"),
        ({"hidden_size": 30}, "num_attention_heads \\* head_dim"),
        ({"lilicorr_candidate_topk": VOCAB + 1}, "fit inside the vocabulary"),
        ({"projector_type": "dflash"}, "projector_type"),
    ],
)
def test_config_rejects_malformed_geometry(params, match):
    dflash = {
        "block_size": BLOCK,
        "conv_group_size": 8,
        "conv_kernel_size": 2,
        "mask_token_id": VOCAB - 1,
        "target_layer_ids": TARGET_LAYER_IDS,
        "lilicorr_hidden_size": 16,
        "lilicorr_candidate_topk": TOP_K,
        "lilicorr_num_heads": 2,
        "lilicorr_num_layers": 1,
    }
    dflash.update(
        {
            key: value
            for key, value in params.items()
            if key not in {"hidden_size"}
        }
    )
    payload = {
        "hidden_size": params.get("hidden_size", HIDDEN),
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": VOCAB,
        "max_position_embeddings": 4096,
        "sliding_window": WINDOW,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "block_size": BLOCK,
        "dflash_config": dflash,
    }
    with pytest.raises((ValueError, TypeError), match=match):
        lilicorr.LiLiCorrConfig.from_dict(payload)


# -- selection ------------------------------------------------------------


class _FixedHead:
    """Scores from a hand-built lattice; records the embeddings it was given."""

    def __init__(self, candidates, scores):
        self.candidates = mx.array(candidates, dtype=mx.int32)
        self.scores = mx.array(scores, dtype=mx.float32)
        self.seen = []

    def score(self, hidden, logits, token_embeddings, anchor_context):
        self.seen.append((hidden, logits, token_embeddings, anchor_context))
        return self.candidates, self.scores


def _lattice():
    candidates = [
        [
            [5, 6, 7, 8],
            [1, 2, 3, 4],
            [9, 10, 11, 12],
        ]
    ]
    scores = [[[[0.0] * TOP_K for _ in range(TOP_K)] for _ in range(3)]]
    # Position 0 reads the anchor row (row 0); later positions read the row of
    # the candidate chosen just before.
    scores[0][0][0] = [0.0, 1.0, 3.0, 2.0]  # picks index 2 -> token 7
    scores[0][1][2] = [4.0, 5.0, 0.5, 0.0]  # picks index 1 -> token 2
    scores[0][2][1] = [0.0, 0.0, 0.0, 6.0]  # picks index 3 -> token 12
    return _FixedHead(candidates, scores)


def test_select_lilicorr_walks_the_correlation_lattice():
    head = _lattice()
    embeddings = nn.Embedding(VOCAB, HIDDEN)
    hidden = mx.zeros((1, 3, HIDDEN))
    logits = mx.zeros((1, 3, VOCAB))
    anchor_context = mx.full((1, HIDDEN), 0.25)
    proposals = selection.select_lilicorr(
        head,
        hidden,
        logits,
        mx.array([9]),
        anchor_context,
        [None],
        embeddings,
    )
    assert len(proposals) == 1
    tokens, accept = proposals[0]
    assert tokens.shape == (1, 3)
    assert tokens.tolist() == [[7, 2, 12]]
    assert accept == []
    # The head scored the draft hidden rows against the newest context row.
    scored_hidden, scored_logits, scored_embeddings, scored_anchor = head.seen[0]
    assert scored_hidden is hidden and scored_logits is logits
    assert scored_embeddings is embeddings
    assert scored_anchor is anchor_context


def test_select_lilicorr_sampled_rows_expose_normalized_sparse_q():
    head = _lattice()
    sampler = SimpleNamespace(temp=1.0, top_p=1.0, top_k=0, min_p=0.0)
    proposals = selection.select_lilicorr(
        head,
        mx.zeros((1, 3, HIDDEN)),
        mx.zeros((1, 3, VOCAB)),
        mx.array([9]),
        mx.zeros((1, HIDDEN)),
        [sampler],
        nn.Embedding(VOCAB, HIDDEN),
    )
    tokens, accept = proposals[0]
    assert tokens.shape == (1, 3)
    assert isinstance(accept, bg.SparseDraftQ)
    assert accept.ids.tolist() == head.candidates[0].tolist()
    assert accept.logq.shape == (3, TOP_K)
    densities = mx.exp(accept.logq)
    assert mx.allclose(mx.sum(densities, axis=-1), mx.ones((3,)))
    assert len(accept) == 3
    # The dense row for a position puts the mass on that position's candidates.
    dense = accept[0]
    assert dense.shape == (VOCAB,)
    picked = {token: float(mx.exp(dense[token])) for token in accept.ids[0].tolist()}
    assert sum(picked.values()) > 0.99


# -- target adapter -------------------------------------------------------


def test_capture_ids_shift_to_layer_inputs():
    assert lilicorr.shift_lilicorr_capture_ids(BOLT_LAYER_IDS) == [
        0,
        17,
        34,
        50,
        67,
        84,
    ]
    assert lilicorr.shift_lilicorr_capture_ids([1]) == [0]
    with pytest.raises(ValueError, match="id 0 has no preceding block output"):
        lilicorr.shift_lilicorr_capture_ids([0, 1])


def test_validates_target_layer_count_and_ids():
    model = lilicorr.LiLiCorrDraftModel(
        _config(target_layer_ids=BOLT_LAYER_IDS, num_target_layers=0)
    )
    # 88 blocks, capture ids up to 85: every id reads a real block output.
    model.bind(_target(layer_count=88))
    with pytest.raises(ValueError, match="fall outside"):
        lilicorr.LiLiCorrDraftModel(
            _config(target_layer_ids=BOLT_LAYER_IDS, num_target_layers=0)
        ).bind(_target(layer_count=5))


def test_target_adapter_marks_the_instance_and_shifts_capture_ids():
    class _ProbeLanguage:
        def __init__(self):
            self.records = []
            self.backbone = SimpleNamespace(layers=[object()])

        def rollback_speculative_cache(self, *args, **kwargs):
            return 0

        def __call__(self, *args, **kwargs):
            self.records.append(dict(kwargs))
            return None

    language = _ProbeLanguage()
    lilicorr.apply_lilicorr_target(language)
    assert language._omlx_lilicorr_target is True
    assert language._omlx_drafter_single_row is True
    language(capture_layer_ids=[1, 18, 35])
    assert language.records[0]["capture_layer_ids"] == [0, 17, 34]
    language(inputs=None)
    assert "capture_layer_ids" not in language.records[1]
    # Loading and attaching both adapt the target; the second call must not
    # stack another wrapper.
    adapted = type(language)
    lilicorr.apply_lilicorr_target(language)
    assert type(language) is adapted
    language(capture_layer_ids=[1])
    assert language.records[-1]["capture_layer_ids"] == [0]
    # The adapter is per instance; the probe class itself stays unmarked.
    assert not hasattr(_ProbeLanguage, "_omlx_lilicorr_capture_inputs")
    assert not hasattr(_ProbeLanguage, "_omlx_lilicorr_target")


def test_attach_drafter_adapts_only_lilicorr_targets():
    class _Host:
        def __init__(self):
            self.backbone = SimpleNamespace(layers=[object()])
            self.calls = []

        def rollback_speculative_cache(self, *args, **kwargs):
            return 0

        def __call__(self, *args, **kwargs):
            self.calls.append(dict(kwargs))
            return None

    host = _Host()
    dd.attach_drafter(host, _tiny_drafter())
    assert host._omlx_lilicorr_target is True
    assert host._omlx_drafter_single_row is True
    assert host._omlx_mtp_decode_enabled is True
    assert host._omlx_mtp_multi_request is False
    host(capture_layer_ids=[1, 18, 35])
    assert host.calls[0]["capture_layer_ids"] == [0, 17, 34]

    stub = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(model_type="dflash2")),
        depth=3,
        target_layer_ids=[1, 3],
    )
    plain = _Host()
    dd.attach_drafter(plain, stub)
    assert not hasattr(plain, "_omlx_lilicorr_target")
    assert plain._omlx_mtp_multi_request is True
    plain(capture_layer_ids=[1, 3])
    assert plain.calls[0]["capture_layer_ids"] == [1, 3]


def test_drafter_reports_the_lilicorr_kind_and_needs_context():
    drafter = _tiny_drafter()
    assert drafter.kind == "lilicorr"
    assert drafter.target_layer_ids == TARGET_LAYER_IDS

    drafter._window_size = 1
    state = SimpleNamespace(
        uid=0, drafts=None, draft_lps=None, draft_accept_lps=None
    )
    with pytest.raises(RuntimeError, match="committed target context"):
        drafter.draft(
            [
                (
                    None,
                    state,
                    [mx.zeros((1, 1, HIDDEN)) for _ in TARGET_LAYER_IDS],
                    mx.array([3], dtype=mx.uint32),
                    None,
                )
            ]
        )
