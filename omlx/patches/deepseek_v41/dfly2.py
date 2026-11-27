# SPDX-License-Identifier: MIT
"""DFly2 adapter for DeepSeek-V4.1's generic speculative-generator contract.

Importing this module has no serving-engine side effects.  The target forward
must create a request-local ``DSV41TargetBridge``, call
``capture_attention_slice`` after its indexer produces the target picks, and
set ``model._dfly_cache = bridge`` before the generator invokes
``dspark_forward``.  DFly2 keeps no cache of its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..deepseek_v41_dfly2 import (
    DSV41TargetBridge,
    hidden_correction,
)


@dataclass(frozen=True)
class DFly2Config:
    block_size: int
    mask_token_id: int
    target_kv_layers: tuple[int, ...]
    target_index_layers: tuple[int, ...]
    index_topk: int
    correction_width: int

    @classmethod
    def from_config(cls, config) -> "DFly2Config":
        source = config.get("dfly_config", {}) if isinstance(config, Mapping) else getattr(config, "dfly_config", {})
        if not isinstance(source, Mapping):
            raise ValueError("DFly2 requires a dfly_config mapping")
        kv = tuple(map(int, source.get("target_kv_layers", ())))
        index = tuple(map(int, source.get("target_index_layers", kv)))
        correction = source.get("hidden_correction", {})
        width = int(correction.get("intermediate_size", 0)) if isinstance(correction, Mapping) else 0
        result = cls(int(source.get("block_size", 0)), int(source.get("mask_token_id", -1)), kv, index, int(source.get("index_topk", 0)), width)
        if result.block_size < 2 or result.mask_token_id < 0 or result.index_topk < 1:
            raise ValueError("DFly2 requires block_size >= 2, mask_token_id, and index_topk")
        if not result.target_kv_layers or len(result.target_kv_layers) != len(result.target_index_layers):
            raise ValueError("DFly2 needs paired target_kv_layers and target_index_layers")
        if result.correction_width < 1:
            raise ValueError("DFly2 hidden_correction.intermediate_size is required")
        return result


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps) * self.weight


class DFly2Heads(nn.Module):
    """Officially named DFly2 head parameters, shared by draft stage zero."""

    def __init__(self, c, cfg: DFly2Config):
        super().__init__()
        self.block_size, self.mask_token_id = cfg.block_size, cfg.mask_token_id
        self.mtp_fusion = nn.Module()
        self.mtp_fusion.enorm = _RMSNorm(c.dim, c.norm_eps)
        self.mtp_fusion.hnorm = _RMSNorm(c.dim, c.norm_eps)
        self.mtp_fusion.eh_proj = nn.Linear(c.dim * 2, c.dim, bias=False)
        self.mtp_fusion.gate = mx.zeros((c.dim,))
        self.hidden_correction = nn.Module()
        self.hidden_correction.hidden_norm = _RMSNorm(c.dim, c.norm_eps)
        self.hidden_correction.embed_norm = _RMSNorm(c.dim, c.norm_eps)
        self.hidden_correction.gate_proj = nn.Linear(c.dim * 2, cfg.correction_width, bias=False)
        self.hidden_correction.up_proj = nn.Linear(c.dim * 2, cfg.correction_width, bias=False)
        self.hidden_correction.down_proj = nn.Linear(cfg.correction_width, c.dim, bias=False)
        # The official checkpoint additionally carries attentive correction.  The
        # portable hidden_correction math consumes the dense five tensors above;
        # these named parameters are nevertheless loaded strictly so a staged
        # official checkpoint can never silently omit that trained component.
        attn = self.hidden_correction.attention = nn.Module()
        attn.attn_sink = mx.zeros((c.n_heads,), mx.float32)
        attn.block_gate = _LinearLayout(c.dim, c.dim)
        attn.block_gate_bias = mx.zeros((16,))
        attn.block_rel_bias = mx.zeros((c.n_heads, 16))
        attn.block_sink = mx.zeros((c.n_heads,))
        attn.kv_in = _LinearLayout(c.head_dim, c.dim)
        attn.kv_norm = _RMSNorm(c.head_dim, c.norm_eps)
        attn.q_in = _LinearLayout(c.q_lora_rank, c.dim)
        attn.q_norm = _RMSNorm(c.q_lora_rank, c.norm_eps)
        attn.wo_a = _LinearLayout(c.o_groups * c.o_lora_rank, c.n_heads * c.head_dim // c.o_groups)
        attn.wo_b = _LinearLayout(c.dim, c.o_groups * c.o_lora_rank)
        attn.wq_b = _LinearLayout(c.n_heads * c.head_dim, c.q_lora_rank)
        self.final_norm = _RMSNorm(c.dim, c.norm_eps)


class _LinearLayout(nn.Module):
    def __init__(self, out_dim, in_dim, *, scale=False):
        super().__init__()
        self.weight = mx.zeros((out_dim, in_dim))
        if scale:
            self.scale = mx.ones((out_dim,))


class _ExpertLayout(nn.Module):
    def __init__(self, c, *, scale=True):
        super().__init__()
        self.w1 = _LinearLayout(c.moe_inter_dim, c.dim, scale=scale)
        self.w2 = _LinearLayout(c.dim, c.moe_inter_dim, scale=scale)
        self.w3 = _LinearLayout(c.moe_inter_dim, c.dim, scale=scale)


class DFly2Attention(nn.Module):
    """Official DeepSeek-V4.1 DFly2 attention parameter layout."""

    def __init__(self, c, layer):
        super().__init__()
        self.attn_sink = mx.zeros((c.n_heads,), mx.float32)
        self.wq_a = _LinearLayout(c.q_lora_rank, c.dim)
        self.q_norm = _RMSNorm(c.q_lora_rank, c.norm_eps)
        self.wq_b = _LinearLayout(c.n_heads * c.head_dim, c.q_lora_rank)
        self.wkv = _LinearLayout(c.head_dim, c.dim)
        self.kv_norm = _RMSNorm(c.head_dim, c.norm_eps)
        self.wo_a = _LinearLayout(c.o_groups * c.o_lora_rank, c.n_heads * c.head_dim // c.o_groups)
        self.wo_b = _LinearLayout(c.dim, c.o_groups * c.o_lora_rank)


class _BlockLayout(nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        self.attn = DFly2Attention(c, layer)
        self.attn_norm = _RMSNorm(c.dim, c.norm_eps)
        self.ffn_norm = _RMSNorm(c.dim, c.norm_eps)
        size = (2 + c.hc_mult) * c.hc_mult
        self.hc_attn_fn = mx.zeros((size, c.hc_mult * c.dim))
        self.hc_attn_base = mx.zeros((size,))
        self.hc_attn_scale = mx.ones((3,))
        self.hc_ffn_fn = mx.zeros((size, c.hc_mult * c.dim))
        self.hc_ffn_base = mx.zeros((size,))
        self.hc_ffn_scale = mx.ones((3,))
        self.ffn = nn.Module()
        self.ffn.gate = nn.Module()
        self.ffn.gate.weight = mx.zeros((c.n_routed_experts, c.dim))
        self.ffn.gate.bias = mx.zeros((c.n_routed_experts,))
        self.ffn.gate.bias_vl = mx.zeros((c.n_routed_experts,))
        self.ffn.experts = [_ExpertLayout(c) for _ in range(c.n_routed_experts)]
        self.ffn.shared_experts = _ExpertLayout(c, scale=False)


class DFly2Block(nn.Module):
    """Portable official parameter layout; runtime math remains in deepseek_v41_dfly2."""

    def __init__(self, c, stage: int, cfg: DFly2Config, heads: DFly2Heads | None = None):
        super().__init__()
        self.core = _BlockLayout(c, c.n_layers + stage)
        if stage == 0:
            self.heads = heads if heads is not None else DFly2Heads(c, cfg)


def make_stages(config):
    """Build draft parameter modules.  Official layers are numbered 0..N-1."""
    if isinstance(config, Mapping):
        raise TypeError("make_stages needs a DeepSeek-V4.1 ModelConfig, not raw JSON")
    cfg = DFly2Config.from_config(config)
    heads = DFly2Heads(config, cfg)
    return [DFly2Block(config, index, cfg, heads if index == 0 else None) for index in range(len(cfg.target_kv_layers))]


class DFly2StageCache:
    """Standalone metadata only; generic ``make_mtp_cache`` returns ``[]``."""

    def __init__(self, target_layer, index_layer, index_topk):
        self.target_layer, self.index_layer, self.index_topk = int(target_layer), int(index_layer), int(index_topk)
        self.offset = 0
        self.bridge = DSV41TargetBridge(target_layers=(self.target_layer,), index_topk=self.index_topk)


def make_cache(config):
    cfg = DFly2Config.from_config(config)
    return [DFly2StageCache(kv, index, cfg.index_topk) for kv, index in zip(cfg.target_kv_layers, cfg.target_index_layers)]


def _head_tensors(stage) -> dict[str, mx.array]:
    heads = stage.heads
    return {
        "model.mtp_fusion.enorm.weight": heads.mtp_fusion.enorm.weight,
        "model.mtp_fusion.hnorm.weight": heads.mtp_fusion.hnorm.weight,
        "model.mtp_fusion.eh_proj.weight": heads.mtp_fusion.eh_proj.weight,
        "model.mtp_fusion.gate": heads.mtp_fusion.gate,
        "model.hidden_correction.hidden_norm.weight": heads.hidden_correction.hidden_norm.weight,
        "model.hidden_correction.embed_norm.weight": heads.hidden_correction.embed_norm.weight,
        "model.hidden_correction.gate_proj.weight": heads.hidden_correction.gate_proj.weight,
        "model.hidden_correction.up_proj.weight": heads.hidden_correction.up_proj.weight,
        "model.hidden_correction.down_proj.weight": heads.hidden_correction.down_proj.weight,
        "model.final_norm.weight": heads.final_norm.weight,
    }


def _official_parameter_map(model) -> dict[str, str]:
    result = {}
    for name, _ in tree_flatten(model.parameters()):
        if not name.startswith("mtp."):
            continue
        prefix = "mtp.0.heads."
        if name.startswith(prefix):
            official, destination = "model." + name.removeprefix(prefix), name
        else:
            _mtp, stage, core, suffix = name.split(".", 3)
            if core != "core":
                raise ValueError(f"unsupported DFly2 parameter path: {name}")
            official, destination = f"model.layers.{stage}.{suffix}", name
        if official in result:
            raise ValueError(f"duplicate official DFly2 mapping: {official}")
        result[official] = destination
    if not result:
        raise ValueError("model has no DFly2 stage parameters under mtp")
    return result


def map_official_weights(model, tensors: Mapping[str, mx.array]) -> list[tuple[str, mx.array]]:
    """Strict official name -> MLX parameter mapping; no unknown or missing keys."""
    expected = _official_parameter_map(model)
    unknown, missing = sorted(set(tensors) - set(expected)), sorted(set(expected) - set(tensors))
    if unknown:
        raise KeyError(f"unknown official DFly2 tensor(s): {', '.join(unknown)}")
    if missing:
        raise KeyError(f"missing official DFly2 tensor(s): {', '.join(missing)}")
    parameters = dict(tree_flatten(model.parameters()))
    mapped = []
    for official, destination in expected.items():
        value = tensors[official]
        if value.shape != parameters[destination].shape:
            raise ValueError(f"invalid shape for {official}: {value.shape}, expected {parameters[destination].shape}")
        mapped.append((destination, value))
    return mapped


def load_official_weights(model, tensors: Mapping[str, mx.array]) -> None:
    model.load_weights(map_official_weights(model, tensors), strict=False)


def _runtime(host):
    runtime = getattr(host, "dfly", getattr(host, "_dfly2", None))
    if runtime is None:
        raise RuntimeError("DFly2 adapter is attached but no dfly runtime is installed")
    return runtime


def _runtime_cfg(runtime, host):
    return runtime.cfg if isinstance(getattr(runtime, "cfg", None), DFly2Config) else DFly2Config.from_config(getattr(host, "_config", getattr(host, "args", {})))


def _bridge_extent(bridge, cfg):
    ends = []
    for layer in cfg.target_kv_layers:
        segments = bridge._window[layer]
        if not segments:
            raise RuntimeError("DFly2 target bridge has no captured MLA rows")
        ends.append(max(segment.end for segment in segments))
    if len(set(ends)) != 1:
        raise ValueError("DFly2 target bridge stage offsets differ")
    return ends[0]


def _embed(host, ids):
    fn = getattr(host, "embed", None) or getattr(getattr(host, "model", None), "embed_tokens", None)
    if fn is None:
        raise RuntimeError("DFly2 host has no token embedding")
    return fn(ids)


def _head_weight(host):
    head = getattr(host, "head", None) or getattr(host, "lm_head", None)
    if head is None:
        raise RuntimeError("DFly2 host has no language-model head")
    return head.weight if hasattr(head, "weight") else head


def attach(model_class):
    """Install generic speculative methods on a DeepSeek-V4.1 model class.

    Signatures: ``make_mtp_cache(self)``, ``dspark_append_context(self,
    hidden_rows, cache, **_)``, ``dspark_forward(self, hidden_rows, anchor,
    cache=None, *, draft_length=None)``, and ``dspark_step_logits(self, block,
    index, previous)``.
    """
    def make_mtp_cache(self):
        return []

    def dspark_append_context(self, hidden_rows, cache, **_):
        return None

    def dspark_forward(self, hidden_rows, anchor, cache=None, *, draft_length=None):
        bridge = getattr(self, "_dfly_cache", None)
        if bridge is None:
            raise RuntimeError("DFly2 needs a target forward before drafting")
        if not isinstance(bridge, DSV41TargetBridge):
            raise TypeError("_dfly_cache must be a DSV41TargetBridge")
        runtime, cfg = _runtime(self), _runtime_cfg(_runtime(self), self)
        start = _bridge_extent(bridge, cfg)
        delta = getattr(getattr(self, "_position_delta", None), "value", None)
        if delta is not None:
            start += int(delta)
        histories = [bridge.history(layer=layer, anchor_position=start) for layer in cfg.target_kv_layers]
        width = 1 + max(1, min(int(draft_length or cfg.block_size - 1), cfg.block_size - 1))
        if not hasattr(runtime, "block_hidden"):
            raise RuntimeError("DFly2 runtime must provide block_hidden")
        return runtime.block_hidden(lambda ids: _embed(self, ids), hidden_rows[:, -1:], anchor.reshape(-1)[:1], start, histories, width), None

    def dspark_step_logits(self, block, index, previous):
        runtime = _runtime(self)
        if hasattr(runtime, "correct"):
            hidden = runtime.correct(_embed(self), block[:, index + 1], previous)
        else:
            stage = runtime[0]
            hidden = hidden_correction(block[:, index + 1], _embed(self, previous.reshape(-1, 1))[:, 0].astype(block.dtype), _head_tensors(stage), eps=stage.heads.final_norm.eps)
        from .head import project_logits
        return project_logits(hidden, _head_weight(self))

    model_class.make_mtp_cache = make_mtp_cache
    model_class.dspark_append_context = dspark_append_context
    model_class.dspark_forward = dspark_forward
    model_class.dspark_step_logits = dspark_step_logits
    return model_class


__all__ = ["DFly2Attention", "DFly2Block", "DFly2Config", "DFly2Heads", "DFly2StageCache", "attach", "load_official_weights", "make_cache", "make_stages", "map_official_weights"]
