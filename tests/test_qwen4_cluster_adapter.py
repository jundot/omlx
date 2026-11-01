# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp cluster adapter: registration, capability and stage selection."""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.qwen4_exp_mlx_lm import apply_qwen4_exp_mlx_lm_patch


def _write_safetensors(path, tensors):
    offset = 0
    header = {}
    for name, size in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset)


def test_qwen4_registration_is_a_honored_pipeline_capability(tmp_path):
    from omlx.cluster.pipeline_compat import pipeline_support

    assert apply_qwen4_exp_mlx_lm_patch()
    import mlx_lm.models.qwen4_exp as module

    assert module.SUPPORTS_PIPELINE is True
    assert module.HONORS_PIPELINE_ASSIGNMENT is True
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    assert pipeline_support(tmp_path) == "honored"


def test_qwen4_stage_starts_keep_the_ple_table_on_the_head():
    import mlx_lm.models.qwen4_exp as module

    assert module.pipeline_stage_starts(
        {"text_config": {"num_hidden_layers": 6, "ple_layer_ids": [2]}}
    ) == (2, 3, 4, 5)
    assert module.pipeline_stage_starts({"text_config": {"num_hidden_layers": 6}}) == ()


def test_stage_keys_select_layers_heads_and_mtp_per_rank():
    from omlx.patches.qwen4_exp_mlx_lm.model import _stage_key_needed

    kwargs = dict(start=2, end=4, rank=0, mtp=True, mmap_ple=True)
    assert not _stage_key_needed("vision_tower.patch_embed.weight", **kwargs)
    assert _stage_key_needed(
        "language_model.model.layers.2.mlp.down_proj.weight", **kwargs
    )
    assert not _stage_key_needed(
        "language_model.model.layers.4.mlp.down_proj.weight", **kwargs
    )
    assert _stage_key_needed("language_model.lm_head.weight", **kwargs)
    assert not _stage_key_needed(
        "language_model.lm_head.weight", **{**kwargs, "rank": 1}
    )
    assert _stage_key_needed("mtp.layers.0.self_attn.q_proj.weight", **kwargs)
    assert not _stage_key_needed(
        "mtp.layers.0.self_attn.q_proj.weight", **{**kwargs, "rank": 1}
    )
    assert not _stage_key_needed(
        "mtp.layers.0.self_attn.q_proj.weight", **{**kwargs, "mtp": False}
    )
    ple = (
        "language_model.model.layers.2.ple.ple_embedding."
        "ngram_embedding.shards.0.weight"
    )
    assert not _stage_key_needed(ple, **kwargs)
    assert _stage_key_needed(ple, **{**kwargs, "mmap_ple": False})
    assert _stage_key_needed(
        "language_model.model.embed_tokens.weight",
        **kwargs,
    )


def _install_fake_stage_loader(
    monkeypatch,
    adapter,
    *,
    rank,
    mtp,
    start,
    end,
    has_mtp_head,
):
    """Replace the loader's model construction and weight I/O with fakes."""

    calls = {}
    layer_count = max(end, start) + 2

    class FakePipelineModel:
        def __init__(self):
            self.start_idx = start
            self.end_idx = end
            self.layers = [object() for _ in range(layer_count)]
            self._omlx_mtp_head_prenorm = False

        def pipeline(self, group):
            calls["pipeline_group"] = group

    language_model = SimpleNamespace(
        model=FakePipelineModel(),
        args=SimpleNamespace(vocab_size=42),
        mtp=None,
    )

    class FakeFullModel:
        def __init__(self, args, *, text_only):
            assert text_only is True
            calls["config"] = args
            calls["full_model"] = self
            self.language_model = language_model
            self.mtp = object() if has_mtp_head else None

        def sanitize(self, weights):
            calls["sanitized"] = weights
            return {"sanitized": True}

    class FakeModelConfig:
        @classmethod
        def from_dict(cls, params):
            calls["model_config"] = params
            return cls()

    group = SimpleNamespace(rank=lambda: rank)

    def configure(path, **kwargs):
        calls["runtime"] = (path, kwargs)

    def stage_weights(path, **kwargs):
        calls["stage_weights"] = (path, kwargs)
        return {"raw": 1}

    def load_model_weights(*args, **kwargs):
        calls["load"] = (args, kwargs)

    monkeypatch.setattr(
        adapter,
        "apply_mlx_vlm_qwen4_exp_compat_patch",
        lambda: calls.setdefault("compat", True),
    )
    monkeypatch.setattr(adapter, "is_mtp_active", lambda: mtp)
    monkeypatch.setattr(adapter, "configure_qwen4_exp_runtime", configure)
    monkeypatch.setattr(
        adapter,
        "_read_config",
        lambda _path: {"model_type": "qwen4_exp"},
    )
    monkeypatch.setattr(adapter, "ModelConfig", FakeModelConfig)
    monkeypatch.setattr(adapter, "_FullModel", FakeFullModel)
    monkeypatch.setattr(adapter, "_stage_weights", stage_weights)
    monkeypatch.setattr(adapter, "load_model_weights", load_model_weights)
    return language_model, calls, group


def test_rank_model_delegates_the_mlx_lm_interface():
    from omlx.patches.qwen4_exp_mlx_lm.model import RankModel

    calls = []

    class FakeBackbone:
        pipeline_layers = ["layer"]

        def __call__(self, inputs, cache=None):
            calls.append(("backbone", inputs, cache))
            return "hidden"

    class FakeLanguageModel:
        def __init__(self):
            self.model = FakeBackbone()
            self.args = SimpleNamespace(vocab_size=42)

        def __call__(self, inputs, cache=None, **kwargs):
            calls.append(("language", inputs, cache, kwargs))
            return SimpleNamespace(logits="logits")

        def make_cache(self):
            return ["cache"]

        def mtp_forward(self, *args, **kwargs):
            return ("mtp", args, kwargs)

        def make_mtp_cache(self):
            return ["mtp-cache"]

        def rollback_speculative_cache(self, *args, **kwargs):
            return ("rollback", args, kwargs)

    language_model = FakeLanguageModel()
    rank_model = RankModel(language_model)
    inputs = mx.array([1])

    assert rank_model.model is language_model.model
    assert rank_model.layers == ["layer"]
    assert rank_model.make_cache() == ["cache"]
    assert rank_model.mtp_forward(1, x=2) == ("mtp", (1,), {"x": 2})
    assert rank_model.make_mtp_cache() == ["mtp-cache"]
    assert rank_model.rollback_speculative_cache(3, y=4) == (
        "rollback",
        (3,),
        {"y": 4},
    )
    assert rank_model._omlx_output_vocab_size == 42
    assert rank_model(inputs, cache="c", skip_logits=True) == "hidden"
    delegated = rank_model(inputs, cache="c", foo=1)
    assert delegated.logits == "logits"
    assert calls[-1] == ("language", inputs, "c", {"foo": 1})
    assert rank_model(inputs, cache="c") == "logits"


def test_load_pipeline_stage_configures_a_rank_zero_mtp_head(monkeypatch, tmp_path):
    from omlx.patches.qwen4_exp_mlx_lm import model as adapter

    language_model, calls, group = _install_fake_stage_loader(
        monkeypatch,
        adapter,
        rank=0,
        mtp=True,
        start=2,
        end=6,
        has_mtp_head=True,
    )

    loaded = adapter.load_pipeline_stage(tmp_path, group, engram_ssd_offload=True)

    assert calls["compat"] is True
    assert calls["runtime"] == (
        tmp_path,
        {"mode": "mmap", "mtp_enabled": True},
    )
    assert calls["stage_weights"] == (
        tmp_path,
        {"start": 2, "end": 6, "rank": 0, "mtp": True, "mmap_ple": True},
    )
    args, kwargs = calls["load"]
    assert args[0] is calls["full_model"]
    assert args[1] == {"model_type": "qwen4_exp"}
    assert args[2] == {"sanitized": True}
    assert args[3] == tmp_path
    assert kwargs == {"lazy": False, "strict": False}
    assert calls["pipeline_group"] is group
    assert language_model.mtp is calls["full_model"].mtp
    assert language_model._omlx_mtp_head_prenorm is True
    assert language_model.model._omlx_mtp_head_prenorm is True
    assert language_model.model.layers[:2] == [None, None]
    assert all(layer is not None for layer in language_model.model.layers[2:])
    assert loaded.language_model is language_model
    assert loaded._omlx_mtp_head_prenorm is True


def test_load_pipeline_stage_configures_rank_one_as_mtp_replay(monkeypatch, tmp_path):
    from omlx.patches.qwen4_exp_mlx_lm import model as adapter

    language_model, calls, group = _install_fake_stage_loader(
        monkeypatch,
        adapter,
        rank=1,
        mtp=True,
        start=0,
        end=4,
        has_mtp_head=True,
    )

    loaded = adapter.load_pipeline_stage(tmp_path, group)

    assert calls["runtime"] == (
        tmp_path,
        {"mode": "resident", "mtp_enabled": False},
    )
    assert calls["stage_weights"] == (
        tmp_path,
        {"start": 0, "end": 4, "rank": 1, "mtp": True, "mmap_ple": False},
    )
    assert language_model.mtp is None
    assert language_model._omlx_mtp_decode_enabled is True
    assert language_model._omlx_mtp_verify_only is True
    assert language_model._omlx_mtp_chain is True
    assert getattr(language_model, "_omlx_mtp_head_prenorm", False) is False
    assert all(layer is not None for layer in language_model.model.layers)
    assert loaded._omlx_mtp_head_prenorm is False


def test_load_pipeline_stage_without_mtp_keeps_ple_resident(monkeypatch, tmp_path):
    from omlx.patches.qwen4_exp_mlx_lm import model as adapter

    language_model, calls, group = _install_fake_stage_loader(
        monkeypatch,
        adapter,
        rank=0,
        mtp=False,
        start=0,
        end=4,
        has_mtp_head=False,
    )

    loaded = adapter.load_pipeline_stage(tmp_path, group)

    assert calls["runtime"] == (
        tmp_path,
        {"mode": "resident", "mtp_enabled": False},
    )
    assert calls["stage_weights"] == (
        tmp_path,
        {"start": 0, "end": 4, "rank": 0, "mtp": False, "mmap_ple": False},
    )
    assert not hasattr(language_model, "_omlx_mtp_decode_enabled")
    assert getattr(language_model, "_omlx_mtp_head_prenorm", False) is False
    assert all(layer is not None for layer in language_model.model.layers)
    assert loaded._omlx_mtp_head_prenorm is False


def test_weight_map_reads_a_single_safetensors_file(tmp_path):
    from omlx.patches.qwen4_exp_mlx_lm.model import _weight_map

    _write_safetensors(
        tmp_path / "only.safetensors",
        [("language_model.lm_head.weight", 4)],
    )

    assert _weight_map(tmp_path) == {
        "language_model.lm_head.weight": "only.safetensors"
    }


def test_weight_map_rejects_multiple_files_without_an_index(tmp_path):
    from omlx.patches.qwen4_exp_mlx_lm.model import _weight_map

    _write_safetensors(tmp_path / "a.safetensors", [("a.weight", 4)])
    _write_safetensors(tmp_path / "b.safetensors", [("b.weight", 4)])

    with pytest.raises(ValueError, match="indexed checkpoint"):
        _weight_map(tmp_path)


def test_stage_weights_only_read_shards_that_hold_the_stage(tmp_path):
    from omlx.patches.qwen4_exp_mlx_lm.model import _stage_weights

    _write_safetensors(
        tmp_path / "a.safetensors",
        [
            ("language_model.model.layers.0.mlp.down_proj.weight", 4),
            ("language_model.lm_head.weight", 4),
        ],
    )
    _write_safetensors(
        tmp_path / "b.safetensors",
        [
            ("language_model.model.layers.9.mlp.down_proj.weight", 4),
            ("vision_tower.patch_embed.weight", 4),
            ("mtp.layers.0.self_attn.q_proj.weight", 4),
        ],
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.model.layers.0.mlp.down_proj.weight": "a.safetensors",
                    "language_model.model.layers.0.missing.weight": "a.safetensors",
                    "language_model.lm_head.weight": "a.safetensors",
                    "language_model.model.layers.9.mlp.down_proj.weight": "b.safetensors",
                    "vision_tower.patch_embed.weight": "b.safetensors",
                    "mtp.layers.0.self_attn.q_proj.weight": "b.safetensors",
                }
            }
        )
    )

    weights = _stage_weights(
        tmp_path,
        start=0,
        end=1,
        rank=0,
        mtp=True,
        mmap_ple=False,
    )

    assert set(weights) == {
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.lm_head.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
    }
    for value in weights.values():
        mx.eval(value)
