"""Checkpoint-only loaders used by the native oMLX dSpark providers.

Adapted in part from ARahim3/mlx-dspark (MIT); see THIRD_PARTY_NOTICES.md.

This module intentionally has no target-model loader, model registry, server,
cache, queue, CLI, or benchmark code.  EnginePool supplies the one target model
already loaded by oMLX; these functions only materialize a drafter module whose
tensor layout was selected by the handler registry.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .native_config import DSparkConfig
from .native_model import DSparkDrafter


def _resolve(repo_or_path: str) -> str:
    path = Path(repo_or_path).expanduser()
    if path.is_dir():
        return str(path)
    return snapshot_download(repo_or_path)


def _flatten_params(module) -> list[tuple[str, mx.array]]:
    from mlx.utils import tree_flatten

    return tree_flatten(module.parameters())


def _load_weight_map(path: str) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for shard in glob.glob(os.path.join(path, "*.safetensors")):
        weights.update(mx.load(shard))
    if not weights:
        raise ValueError(f"no safetensors weights found in drafter checkpoint: {path}")
    return weights


def _validate_tensor_keys(
    repo_or_path: str,
    module,
    weights: dict[str, mx.array],
    *,
    format_name: str,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    model_keys = {key for key, _ in _flatten_params(module)}
    checkpoint_keys = set(weights)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if strict and (missing or unexpected):
        details = []
        if missing:
            details.append(f"missing ({len(missing)}): {missing[:8]}")
        if unexpected:
            details.append(f"unexpected ({len(unexpected)}): {unexpected[:8]}")
        raise ValueError(
            f"{repo_or_path}: tensor names do not match {format_name}: "
            + "; ".join(details)
        )
    return missing, unexpected


def load_drafter(
    repo_or_path: str,
    *,
    quantize: bool = False,
    bits: int = 4,
    group_size: int = 64,
    strict: bool = True,
):
    """Load a DeepSpec drafter without mutating its checkpoint precision."""
    del bits, group_size
    if quantize:
        raise ValueError(
            "runtime drafter quantization is disabled; use Prepare dSpark and "
            "load the resulting immutable checkpoint"
        )
    path = _resolve(repo_or_path)
    config = DSparkConfig.from_json(os.path.join(path, "config.json"))
    drafter = DSparkDrafter(config)
    weights = _load_weight_map(path)
    missing, unexpected = _validate_tensor_keys(
        repo_or_path,
        drafter,
        weights,
        format_name="a DeepSpec drafter",
        strict=strict,
    )
    drafter.load_weights(list(weights.items()), strict=not (missing or unexpected))

    if config.offset_rms_norm:
        for _, module in drafter.named_modules():
            if isinstance(module, nn.RMSNorm):
                module.weight = module.weight + 1.0

    mx.eval(drafter.parameters())
    return drafter, config


def load_dflash(
    repo_or_path: str,
    *,
    quantize: bool = False,
    bits: int = 4,
    group_size: int = 64,
    prequantized: tuple[int, int] | None = None,
    target_hidden_size: int | None = None,
):
    """Load a DFlash/Speculators/Higgs drafter from checkpoint metadata."""
    del bits, group_size
    if quantize:
        raise ValueError(
            "runtime drafter quantization is disabled; use Prepare dSpark and "
            "load the resulting immutable checkpoint"
        )

    from .native_dflash_model import (
        DFlashConfig,
        DFlashDraftModel,
        DFlashMarkovDraftModel,
        SpeculatorsDraftModel,
    )

    path = _resolve(repo_or_path)
    with open(os.path.join(path, "config.json")) as config_file:
        raw = json.load(config_file)
    markov_rank = int(raw.get("markov_rank") or 0)
    speculators = bool(
        raw.get("speculators_config")
        or any(
            "dsparkdraftmodel" in str(name).lower()
            for name in raw.get("architectures", ())
        )
    )
    markov_type = str(raw.get("markov_head_type") or "vanilla")
    if markov_rank and markov_type != "vanilla":
        raise ValueError(
            f"{repo_or_path}: unsupported Markov head type {markov_type!r}; "
            "only vanilla is implemented"
        )

    layer = raw.get("transformer_layer_config") if speculators else raw
    layer = layer or raw
    rope = layer.get("rope_parameters") or raw.get("rope_parameters") or {}
    dflash = raw.get("dflash_config") or {}
    block_size = dflash.get("block_size", raw.get("block_size"))
    if block_size is None:
        raise ValueError(f"{repo_or_path}: dSpark drafter config has no block_size")
    layer_types = tuple(
        layer.get("layer_types") or ["full_attention"] * int(layer["num_hidden_layers"])
    )
    config = DFlashConfig(
        hidden_size=layer["hidden_size"],
        num_hidden_layers=layer["num_hidden_layers"],
        num_attention_heads=layer["num_attention_heads"],
        num_key_value_heads=layer["num_key_value_heads"],
        head_dim=layer["head_dim"],
        intermediate_size=layer["intermediate_size"],
        vocab_size=layer["vocab_size"],
        rms_norm_eps=layer["rms_norm_eps"],
        rope_theta=layer.get("rope_theta", rope.get("rope_theta", 1_000_000.0)),
        max_position_embeddings=layer["max_position_embeddings"],
        block_size=int(block_size),
        target_layer_ids=tuple(
            dflash.get("target_layer_ids")
            or raw.get("target_layer_ids")
            or raw.get("aux_hidden_state_layer_ids")
            or ()
        ),
        num_target_layers=int(raw.get("num_target_layers") or 0),
        target_hidden_size=raw.get("target_hidden_size") or target_hidden_size,
        draft_vocab_size=raw.get("draft_vocab_size"),
        markov_rank=markov_rank,
        confidence_head_with_markov=bool(raw.get("confidence_head_with_markov", False)),
        enable_confidence_head=bool(raw.get("enable_confidence_head", False)),
        mask_token_id=dflash.get("mask_token_id", raw.get("mask_token_id", 0)),
        rope_scaling=layer.get("rope_scaling"),
        layer_types=layer_types,
        sliding_window=layer.get("sliding_window"),
        final_logit_softcapping=raw.get("final_logit_softcapping"),
    )
    if speculators:
        drafter = SpeculatorsDraftModel(config)
    elif markov_rank:
        drafter = DFlashMarkovDraftModel(config, markov_rank)
    else:
        drafter = DFlashDraftModel(config)

    # Construct quantized module shapes before binding packed tensors.  This
    # reads immutable checkpoint metadata; it does not quantize BF16 weights.
    if prequantized is not None:
        prepared_bits, prepared_group = prequantized
        if prepared_bits <= 0 or prepared_group <= 0:
            raise ValueError("invalid prepared drafter quantization metadata")
        nn.quantize(
            drafter,
            group_size=prepared_group,
            bits=prepared_bits,
            class_predicate=lambda _path, module: isinstance(
                module, (nn.Linear, nn.Embedding)
            ),
        )

    weights = _load_weight_map(path)
    _validate_tensor_keys(
        repo_or_path,
        drafter,
        weights,
        format_name="a DFlash/Speculators drafter",
    )
    drafter.load_weights(list(weights.items()))
    mx.eval(drafter.parameters())
    return drafter, config
