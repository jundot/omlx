"""Bring an official DFly2 checkpoint up as a live draft runtime.

This is the only place that knows both naming worlds: the checkpoint numbers its
three draft layers from zero and prefixes every tensor with ``model.``, while the
layer math and the heads each take their own view of those names. Everything here
is loading and wiring; no math.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx

from .dfly2_heads import final_norm, hidden_correction, mtp_fusion
from .dfly2_layer import make_draft_layer
from .official_loader import OfficialDFly2Checkpoint, draft_layer_ids, required_prefixes
from .runtime import DFly2Runtime

logger = logging.getLogger(__name__)


def _config_value(config: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    if default is None:
        raise KeyError(f"DFly2 config is missing {names}")
    return default


def load_checkpoint_tensors(checkpoint_dir: str | Path) -> tuple[dict[str, mx.array], dict[str, Any]]:
    """Every tensor the draft needs, under the checkpoint's own names.

    Uses the loader's own metadata path, so a shard that lacks an indexed tensor
    fails here rather than at first forward.
    """
    directory = Path(checkpoint_dir)
    checkpoint = OfficialDFly2Checkpoint.from_metadata(directory / "config.json", directory / "model.safetensors.index.json")
    plan = checkpoint.plan_for_prefixes(
        required_prefixes(layer_ids=draft_layer_ids(checkpoint.config), include_fusion=True)
    )
    loaded = checkpoint.load_available_shards(directory, plan)
    return dict(loaded.tensors), checkpoint.config


def _stage_weights(tensors: Mapping[str, mx.array], layer_id: int) -> dict[str, mx.array]:
    prefix = f"model.layers.{layer_id}."
    return {name.removeprefix(prefix): value for name, value in tensors.items() if name.startswith(prefix)}


def build_runtime(tensors: Mapping[str, mx.array], config: Any, checkpoint_config: Mapping[str, Any]) -> DFly2Runtime:
    """Assemble the runtime from raw official tensors and a model config."""
    eps = float(_config_value(config, "norm_eps", "rms_norm_eps"))
    dfly = checkpoint_config.get("dfly_config") or {}
    mask_token_id = int(dfly.get("mask_token_id", _config_value(config, "mask_token_id", default=128799)))
    layers = [
        make_draft_layer(_stage_weights(tensors, layer_id), config)
        for layer_id in draft_layer_ids(checkpoint_config)
    ]
    return DFly2Runtime(
        layers=layers,
        fusion=lambda anchor, hidden: mtp_fusion(anchor, hidden, tensors, eps=eps),
        final_norm=lambda hidden: final_norm(hidden, tensors, eps=eps),
        correct=lambda hidden, embedding: hidden_correction(hidden, embedding, tensors, eps=eps),
        lm_head=lambda hidden: hidden,  # the engine takes logits from the target head
        mask_token_id=mask_token_id,
    )


def _model_config(model: Any):
    """The text config, wherever the engine hid it.

    A VLM engine hands us its own wrapper, whose config lives one level down on
    the language model it holds; a text engine hands us the model itself.
    """
    candidates = [model]
    for attr in ("language_model", "_language_model", "model"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            candidates.append(inner)
    for candidate in candidates:
        config = getattr(candidate, "_config", None) or getattr(candidate, "args", None)
        if config is not None:
            return config
    return None


def enable_dfly2(model: Any, checkpoint_dir: str | Path):
    """Load the draft into ``model`` and arm the generator methods.

    Returns the runtime; the caller keeps it alive for the model's lifetime. The
    target keeps serving its own weights: nothing here touches the MTP module.
    """
    from ..deepseek_v41.dfly2 import attach

    directory = Path(checkpoint_dir)
    model_config = _model_config(model)
    if model_config is None:
        raise RuntimeError("DFly2 needs the loaded model's config to size its layers")
    tensors, checkpoint_config = load_checkpoint_tensors(directory)
    runtime = build_runtime(tensors, model_config, checkpoint_config)
    model.dfly = runtime
    attach(type(model))
    # The generator only drafts from a host it considers DSpark-active; without
    # this marker the runtime would load and then never be asked for a block.
    for holder in (model, getattr(model, "language_model", None), getattr(model, "_language_model", None)):
        if holder is not None:
            holder._omlx_dspark_decode_enabled = True
            # The cycle log line reads this: "DSpark" alone would be ambiguous
            # between the checkpoint's own head and this replacement drafter.
            holder._omlx_spec_backend_name = "DFly2"
    logger.info(
        "DFly2 draft replaces the checkpoint MTP head: block_size=%d target_kv_layers=%s draft_layers=%d",
        checkpoint_config.get("block_size"),
        checkpoint_config.get("target_kv_layers", checkpoint_config.get("target_kv_index_layers")),
        len(runtime.layers) if hasattr(runtime, "layers") else -1,
    )
    return runtime
