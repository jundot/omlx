# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for the lossless DS4 projection bundle."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import mlx.core as mx

from omlx.patches.deepseek_v4 import qkv_prefill_bundle as qkv_bundle


class _Group:
    def __init__(self, size: int):
        self._size = size

    def size(self):
        return self._size


class _Array:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Projection:
    def __init__(self, weight, scales=None, *, mxfp8=False):
        self.values = {"weight": weight}
        if scales is not None:
            self.values["scales"] = scales
        if mxfp8:
            self.group_size = 32
            self.bits = 8
            self.mode = "mxfp8"


def _config():
    return SimpleNamespace(
        model_type="deepseek_v4",
        vocab_size=129280,
        hidden_size=4096,
        moe_intermediate_size=2048,
        num_hidden_layers=43,
        num_attention_heads=64,
        n_routed_experts=256,
        num_experts_per_tok=6,
        max_position_embeddings=1048576,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
    )


def _attention(group=None):
    q_a = _Projection(
        _Array((1024, 1024), mx.uint32),
        _Array((1024, 128), mx.uint8),
        mxfp8=True,
    )
    raw_kv = _Projection(
        _Array((512, 1024), mx.uint32),
        _Array((512, 128), mx.uint8),
        mxfp8=True,
    )
    dense = [
        _Projection(_Array((1024, 4096), mx.bfloat16)),
        _Projection(_Array((1024, 4096), mx.bfloat16)),
        _Projection(_Array((256, 4096), mx.bfloat16)),
        _Projection(_Array((256, 4096), mx.bfloat16)),
    ]
    return SimpleNamespace(
        config=_config(),
        compress_ratio=4,
        training=False,
        sharding_group=group,
        wq_a=q_a,
        wkv=raw_kv,
        compressor=SimpleNamespace(wkv=dense[0], wgate=dense[1]),
        indexer=SimpleNamespace(
            compressor=SimpleNamespace(wkv=dense[2], wgate=dense[3])
        ),
    )


def _enable_contract(monkeypatch):
    monkeypatch.setattr(qkv_bundle, "ENABLED", True)
    monkeypatch.setattr(qkv_bundle, "device_qualified", lambda: True)
    monkeypatch.setattr(qkv_bundle, "is_dspark_verify_armed", lambda: False)
    monkeypatch.setattr(
        qkv_bundle,
        "_value",
        lambda module, name: module.values.get(name),
    )


def test_prefill_contract_is_shared_by_single_node_and_tp2(monkeypatch):
    _enable_contract(monkeypatch)
    x = _Array((1, 1024, 4096), mx.bfloat16)

    single = qkv_bundle.eligible_modules(_attention(), x)
    tp2 = qkv_bundle.eligible_modules(_attention(_Group(2)), x)

    assert single is not None and len(single) == 6
    assert tp2 is not None and len(tp2) == 6
    assert qkv_bundle.eligible_modules(_attention(_Group(3)), x) is None

    monkeypatch.setattr(qkv_bundle, "device_name", lambda: "Apple M3 Ultra")
    local_m2048 = qkv_bundle.eligible_modules(
        _attention(), _Array((1, 2048, 4096), mx.bfloat16)
    )
    assert local_m2048 is not None and len(local_m2048) == 6


def test_prefill_contract_rejects_every_unqualified_boundary(monkeypatch):
    _enable_contract(monkeypatch)
    x = _Array((1, 1024, 4096), mx.bfloat16)
    attn = _attention()

    assert qkv_bundle.eligible_modules(attn, x) is not None
    assert qkv_bundle.eligible_modules(
        attn, _Array((1, 1536, 4096), mx.bfloat16)
    ) is None
    monkeypatch.setattr(qkv_bundle, "device_name", lambda: "Apple M5 Max")
    assert qkv_bundle.eligible_modules(
        attn, _Array((1, 2048, 4096), mx.bfloat16)
    ) is None
    monkeypatch.setattr(qkv_bundle, "device_name", lambda: "Apple M3 Ultra")
    assert (
        qkv_bundle.eligible_modules(attn, _Array((1, 1024, 4096), mx.float16)) is None
    )
    attn.compress_ratio = 128
    assert qkv_bundle.eligible_modules(attn, x) is None
    attn.compress_ratio = 4
    attn.wq_a.values["weight"] = _Array((1024, 1024), mx.int32)
    assert qkv_bundle.eligible_modules(attn, x) is None

    monkeypatch.setattr(qkv_bundle, "device_qualified", lambda: False)
    assert qkv_bundle.eligible_modules(_attention(), x) is None
    monkeypatch.setattr(qkv_bundle, "device_qualified", lambda: True)
    monkeypatch.setattr(qkv_bundle, "ENABLED", False)
    assert qkv_bundle.eligible_modules(_attention(), x) is None


def test_dispatcher_uses_one_shared_seam_and_latches_failure_per_layer(monkeypatch):
    x = object()
    attn = SimpleNamespace(layer_idx=2)
    modules = tuple(object() for _ in range(6))
    banks = object()
    outputs = tuple(object() for _ in range(6))
    monkeypatch.setattr(qkv_bundle, "eligible_modules", lambda owner, value: modules)
    monkeypatch.setattr(qkv_bundle, "canonicalize", lambda owner, value: banks)
    monkeypatch.setattr(qkv_bundle, "project_banks", lambda value, packed: outputs)
    monkeypatch.setattr(qkv_bundle, "_LOGGED", False)

    assert qkv_bundle.prefill_qkv_projection_bundle(attn, x) is outputs

    failed = SimpleNamespace(layer_idx=3)
    monkeypatch.setattr(
        qkv_bundle,
        "canonicalize",
        lambda owner, value: (_ for _ in ()).throw(RuntimeError("pack failed")),
    )
    assert qkv_bundle.prefill_qkv_projection_bundle(failed, x) is None
    assert failed._omlx_qkv_prefill_failed is True
    assert qkv_bundle.prefill_qkv_projection_bundle(failed, x) is None


def test_packer_preserves_checkpoint_formats_and_has_no_requantization_path():
    pack_source = inspect.getsource(qkv_bundle.canonicalize)
    project_source = inspect.getsource(qkv_bundle.project_banks)

    assert "mx.quantize" not in pack_source
    assert "mx.dequantize" not in pack_source
    assert "astype" not in pack_source
    assert "q_a.weight, raw_kv.weight" in pack_source
    assert "banks.qkv_weight[:1024]" in pack_source
    assert "banks.qkv_weight[1024:]" in pack_source
    assert 'mode="mxfp8"' in project_source
    assert project_source.count("mx.split") == 3
    assert "x.shape[1] == 2048" not in project_source
    assert "banks.index_compressor_weight[:256]" in project_source
    assert "banks.index_compressor_weight[256:]" in project_source
