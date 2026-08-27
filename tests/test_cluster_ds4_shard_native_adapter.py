# SPDX-License-Identifier: Apache-2.0
"""Production wiring for DeepSeek-V4 structure-first rank-local loading."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import mlx_lm.utils as mlx_utils
import pytest

from omlx.cluster.ds4_shard_native_adapter import (
    DS4NativeQualificationError,
    _qualified_native_config,
    try_deepseek_v4_rank_local_load,
)
from omlx.cluster.progressive_loading import (
    _ds4_shard_native_enabled,
    progressive_sharded_load,
)
from omlx.cluster.shard_native_loading import LocalSafetensors


class _Group:
    def __init__(self, rank: int, size: int = 2):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _config() -> dict:
    return {
        "model_type": "deepseek_v4",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "n_shared_experts": 1,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "num_hash_layers": 0,
        "q_lora_rank": 4,
        "qk_rope_head_dim": 4,
        "head_dim": 4,
        "o_groups": 2,
        "o_lora_rank": 4,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 2,
        "hc_mult": 4,
        "compress_ratios": [0],
    }


def _write_checkpoint(root: Path, config: dict) -> None:
    from mlx_lm.models import deepseek_v4

    reference = deepseek_v4.Model(deepseek_v4.ModelArgs.from_dict(config))
    tensors = dict(mlx_utils.tree_flatten(reference.parameters()))
    tensors["embed.weight"] = mx.arange(32 * 8, dtype=mx.float32).reshape(32, 8)
    del tensors["model.embed_tokens.weight"]
    tensors["head.weight"] = mx.arange(32 * 8, dtype=mx.float32).reshape(32, 8)
    del tensors["lm_head.weight"]
    tensors["norm.weight"] = tensors.pop("model.norm.weight")
    tensors["layers.0.attn.attn_sink"] = mx.arange(4, dtype=mx.float32)
    del tensors["model.layers.0.attn.attn_sink"]
    tensors["layers.0.attn.wq_b.weight"] = mx.arange(16 * 4, dtype=mx.float32).reshape(
        16, 4
    )
    del tensors["model.layers.0.attn.wq_b.weight"]
    for expert in range(2):
        rows = (
            mx.arange(8, dtype=mx.float32)[:, None]
            + expert * 100
            + mx.zeros((8, 8), dtype=mx.float32)
        )
        columns = (
            mx.arange(8, dtype=mx.float32)[None]
            + expert * 1000
            + mx.zeros((8, 8), dtype=mx.float32)
        )
        tensors[f"layers.0.ffn.experts.{expert}.w1.weight"] = rows
        tensors[f"layers.0.ffn.experts.{expert}.w2.weight"] = columns
        tensors[f"layers.0.ffn.experts.{expert}.w3.weight"] = rows + 500
    for projection in ("gate_proj", "down_proj", "up_proj"):
        del tensors[f"model.layers.0.ffn.switch_mlp.{projection}.weight"]
    (root / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(root / "model.safetensors"), tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}})
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_structure_first_loads_only_exact_local_ds4_rows(tmp_path, monkeypatch, rank):
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    monkeypatch.delenv("OMLX_TP_SHARD_WEIGHTS", raising=False)
    monkeypatch.delenv("OMLX_TP_NON_MOE_SHARD_WEIGHTS", raising=False)
    monkeypatch.delenv("OMLX_TP_MOE_SHARD_WEIGHTS", raising=False)
    config = _config()
    _write_checkpoint(tmp_path, config)
    progress = []

    result = try_deepseek_v4_rank_local_load(
        tmp_path,
        tmp_path,
        config,
        _Group(rank),
        utils_module=mlx_utils,
        mx_module=mx,
        progress=progress.append,
    )

    assert result is not None
    model, returned_config = result
    assert returned_config == {
        **config,
        "use_native_ratio128_attention": True,
    }
    assert "use_native_ratio128_attention" not in config
    assert model._omlx_shard_native_loading is True
    layer = model.model.layers[0]
    expected_heads = [rank, rank + 2]
    assert layer.attn.attn_sink.tolist() == [float(value) for value in expected_heads]
    assert layer.attn.wq_b.weight[:, 0].tolist() == [
        float(head * 16 + row * 4) for head in expected_heads for row in range(4)
    ]
    row_start = rank * 4
    assert layer.ffn.switch_mlp.gate_proj.weight.shape == (2, 4, 8)
    assert layer.ffn.switch_mlp.gate_proj.weight[:, :, 0].tolist() == [
        [float(expert * 100 + row) for row in range(row_start, row_start + 4)]
        for expert in range(2)
    ]
    assert layer.ffn.switch_mlp.down_proj.weight.shape == (2, 8, 4)
    assert layer.ffn.switch_mlp.down_proj.weight[:, 0].tolist() == [
        [float(expert * 1000 + column) for column in range(row_start, row_start + 4)]
        for expert in range(2)
    ]
    assert model.lm_head.weight.shape == (16, 8)
    assert model.lm_head.weight[:, 0].tolist() == [
        float(row * 8) for row in range(rank * 16, rank * 16 + 16)
    ]
    phases = [event["phase"] for event in progress]
    assert phases.index("tensor_native_qualified") < phases.index(
        "tensor_native_loading"
    )
    assert phases[-1] == "tensor_native_ready"
    ready = progress[-1]
    assert ready["local_bytes"] < ready["source_bytes"]
    assert ready["rank"] == rank


def test_remote_and_incomplete_checkpoints_fall_back_without_a_partial_model(
    tmp_path, monkeypatch
):
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    config = _config()
    _write_checkpoint(tmp_path, config)
    events = []

    assert (
        try_deepseek_v4_rank_local_load(
            "owner/remote-model",
            tmp_path,
            config,
            _Group(0),
            utils_module=mlx_utils,
            mx_module=mx,
            progress=events.append,
        )
        is None
    )
    assert events == []

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    index["weight_map"]["missing.weight"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert (
        try_deepseek_v4_rank_local_load(
            tmp_path,
            tmp_path,
            config,
            _Group(0),
            utils_module=mlx_utils,
            mx_module=mx,
            progress=events.append,
        )
        is None
    )
    assert events[-1]["phase"] == "tensor_native_fallback"
    assert "index/header disagreement" in events[-1]["reason"]


def test_progressive_loader_selects_rank_local_ds4_before_any_lazy_model_load(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OMLX_DSV4_SHARD_NATIVE_LOAD", "1")
    config = _config()
    (tmp_path / "config.json").write_text(json.dumps(config))
    sentinel_model = object()
    calls = []

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel_model, config

    monkeypatch.setattr(
        "omlx.cluster.ds4_shard_native_adapter.try_deepseek_v4_rank_local_load",
        native,
    )
    utils = type(
        "Utils",
        (),
        {
            "_download": staticmethod(lambda repo, allow_patterns=None: tmp_path),
            "load_config": staticmethod(lambda path: config),
            "load_tokenizer": staticmethod(
                lambda path, tokenizer_config, eos_token_ids=None: "tokenizer"
            ),
            "load_model": staticmethod(
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("rank-local DS4 must run before load_model")
                )
            ),
        },
    )

    model, tokenizer = progressive_sharded_load(
        tmp_path,
        tensor_group=_Group(0),
        utils_module=utils,
        mx_module=mx,
    )

    assert model is sentinel_model
    assert tokenizer == "tokenizer"
    assert len(calls) == 1


@pytest.mark.parametrize("value", [None, "0", "off", "unexpected"])
def test_rank_local_loader_is_default_off_and_unknown_values_fail_closed(
    monkeypatch, value
):
    if value is None:
        monkeypatch.delenv("OMLX_DSV4_SHARD_NATIVE_LOAD", raising=False)
    else:
        monkeypatch.setenv("OMLX_DSV4_SHARD_NATIVE_LOAD", value)
    assert _ds4_shard_native_enabled() is False


def test_rank_local_loader_gate_is_propagated_to_every_rank(monkeypatch):
    from omlx.cluster.deployment import _hostfile_envs

    monkeypatch.delenv("OMLX_DSV4_SHARD_NATIVE_LOAD", raising=False)
    assert "OMLX_DSV4_SHARD_NATIVE_LOAD=0" in _hostfile_envs()
    monkeypatch.setenv("OMLX_DSV4_SHARD_NATIVE_LOAD", "1")
    assert "OMLX_DSV4_SHARD_NATIVE_LOAD=1" in _hostfile_envs()


def test_native_config_reuses_production_quantization_and_attention_gates(
    monkeypatch,
):
    config = _config()
    nested = {"quant_method": "mxfp4", "bits": 3}
    config["text_config"] = {"quantization_config": nested}

    normalized = _qualified_native_config(config)

    assert normalized["quantization_config"] is nested
    assert normalized["use_native_ratio128_attention"] is False
    assert "quantization_config" not in config
    assert "use_native_ratio128_attention" not in config

    monkeypatch.setenv("OMLX_DSV4_LMHEAD_Q8", "1")
    with pytest.raises(DS4NativeQualificationError, match="production lazy loader"):
        _qualified_native_config(config)


def test_lmhead_q8_selects_the_unchanged_production_loader(tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_DSV4_LMHEAD_Q8", "1")
    progress = []

    result = try_deepseek_v4_rank_local_load(
        tmp_path,
        tmp_path,
        _config(),
        _Group(0),
        utils_module=mlx_utils,
        mx_module=mx,
        progress=progress.append,
    )

    assert result is None
    assert progress[-1]["phase"] == "tensor_native_fallback"
    assert "OMLX_DSV4_LMHEAD_Q8" in progress[-1]["reason"]


def test_sanitizer_alias_collision_falls_back_before_any_tensor_read(
    tmp_path, monkeypatch
):
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    monkeypatch.delenv("OMLX_DSV4_LMHEAD_Q8", raising=False)
    config = _config()
    _write_checkpoint(tmp_path, config)
    path = tmp_path / "model.safetensors"
    tensors = dict(mx.load(str(path)))
    # mx.load is lazy and these arrays still map ``path``. Materialize every
    # value before save_safetensors replaces that same backing file; otherwise
    # Python 3.13 runners can truncate it before the deferred reads execute.
    mx.eval(*tensors.values())
    tensors["model.embed_tokens.weight"] = tensors["embed.weight"]
    mx.save_safetensors(str(path), tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: path.name for name in tensors}})
    )
    monkeypatch.setattr(
        LocalSafetensors,
        "load_partition",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ambiguous checkpoint must fail before tensor reads")
        ),
    )
    progress = []

    result = try_deepseek_v4_rank_local_load(
        tmp_path,
        tmp_path,
        config,
        _Group(0),
        utils_module=mlx_utils,
        mx_module=mx,
        progress=progress.append,
    )

    assert result is None
    assert progress[-1]["phase"] == "tensor_native_fallback"
    assert "ambiguous DS4 sanitizer aliases" in progress[-1]["reason"]


def test_checkpoint_snapshot_mutation_aborts_before_ready(tmp_path, monkeypatch):
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    monkeypatch.delenv("OMLX_DSV4_LMHEAD_Q8", raising=False)
    config = _config()
    _write_checkpoint(tmp_path, config)
    path = tmp_path / "model.safetensors"
    original = LocalSafetensors.load_partition
    mutated = False

    def load_then_mutate(self, *args, **kwargs):
        nonlocal mutated
        value = original(self, *args, **kwargs)
        if not mutated:
            mutated = True
            stat = path.stat()
            os.utime(
                path,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
            )
        return value

    monkeypatch.setattr(LocalSafetensors, "load_partition", load_then_mutate)
    progress = []

    with pytest.raises(RuntimeError, match="checkpoint changed"):
        try_deepseek_v4_rank_local_load(
            tmp_path,
            tmp_path,
            config,
            _Group(0),
            utils_module=mlx_utils,
            mx_module=mx,
            progress=progress.append,
        )

    assert mutated is True
    assert all(event["phase"] != "tensor_native_ready" for event in progress)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("missing", "model_unfilled"),
        ("extra", "checkpoint_unconsumed"),
    ],
)
def test_strict_coverage_never_publishes_missing_or_extra_sanitizer_keys(
    tmp_path, monkeypatch, mutation, message
):
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    config = _config()
    _write_checkpoint(tmp_path, config)
    path = tmp_path / "model.safetensors"
    tensors = dict(mx.load(str(path)))
    # ``mx.load`` is lazy. Materialize every source tensor before replacing
    # its backing file; otherwise save_safetensors may truncate the file and
    # then try to evaluate an array that still maps the now-empty source.
    mx.eval(*tensors.values())
    if mutation == "missing":
        del tensors["model.layers.0.attn.wq_a.weight"]
    else:
        tensors["unknown.weight"] = mx.zeros((1,), dtype=mx.float32)
    mx.save_safetensors(str(path), tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: path.name for name in tensors}})
    )
    monkeypatch.setattr(
        LocalSafetensors,
        "load_partition",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("coverage mismatch must fail before tensor reads")
        ),
    )
    progress = []

    result = try_deepseek_v4_rank_local_load(
        tmp_path,
        tmp_path,
        config,
        _Group(0),
        utils_module=mlx_utils,
        mx_module=mx,
        progress=progress.append,
    )

    assert result is None
    assert progress[-1]["phase"] == "tensor_native_fallback"
    assert message in progress[-1]["reason"]
    assert all(event["phase"] != "tensor_native_ready" for event in progress)
