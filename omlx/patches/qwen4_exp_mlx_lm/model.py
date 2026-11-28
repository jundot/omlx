# SPDX-License-Identifier: Apache-2.0
"""``mlx_lm.models.qwen4_exp``: oMLX's Qwen4-Exp for a cluster rank.

The single-node engine serves Qwen4-Exp through mlx-vlm: vision tower, text
backbone and the embedded Lightning MTP head. A pipeline rank holds one text
stage, so this module registers a text-only root whose parameter names match
the checkpoint index and whose forward hands HC residual streams to the next
rank.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.mlx_lm_mtp import is_mtp_active
from omlx.patches.mlx_vlm_qwen4_exp_compat import (
    apply_mlx_vlm_qwen4_exp_compat_patch,
    configure_qwen4_exp_runtime,
)

logger = logging.getLogger(__name__)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp import Model as _FullModel  # noqa: E402
from mlx_vlm.models.qwen4_exp import ModelConfig  # noqa: E402
from mlx_vlm.models.qwen4_exp.qwen4_exp import _NGRAM_STORAGE_RE  # noqa: E402
from mlx_vlm.utils import load_model_weights  # noqa: E402

# Read by the planner. Ranks never load the vision tower; MTP is separate and
# charged to rank zero.
CLUSTER_UNLOADED_WEIGHT_PREFIXES = (
    "vision_tower.",
    "multi_modal_projector.",
    "mtp.",
    "language_model.mtp.",
)
CLUSTER_MTP_WEIGHT_PREFIXES = ("mtp.", "language_model.mtp.")
SUPPORTS_PIPELINE = True
# The stage forward consumes the planner's range through PipelineMixin.pipeline
# (installed by install_unequal_pipeline_plan) and never derives an even split.
HONORS_PIPELINE_ASSIGNMENT = True


def pipeline_stage_starts(config: dict[str, Any]) -> tuple[int, ...]:
    """Cuts that keep the PLE table on the head rank.

    ``ple_layer_ids`` is one-based in the checkpoint; the tensor lives in the
    zero-based layer immediately before it. A cut of at least two keeps layer 1
    on the head, where the adapter loads the table.
    """

    text = config.get("text_config")
    text = text if isinstance(text, dict) else config
    if not text.get("ple_layer_ids"):
        return ()
    layers = int(text.get("num_hidden_layers") or 0)
    return tuple(range(2, layers))


class ModelArgs(ModelConfig):
    """The checkpoint config a rank can load."""

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        return super().from_dict(params)


class RankModel(nn.Module):
    """A loaded stage, presented the way mlx-lm serves a model."""

    _omlx_supports_rank_zero_logits = True

    def __init__(self, language_model: Any) -> None:
        super().__init__()
        self.language_model = language_model
        # The MTP head consumes Qwen4's raw hyper streams; the generic head
        # input helper would otherwise look for ``language_model.model.norm``.
        self._omlx_mtp_head_prenorm = bool(
            getattr(language_model, "_omlx_mtp_head_prenorm", False)
        )

    @property
    def model(self) -> Any:
        return self.language_model.model

    @property
    def layers(self) -> list[Any]:
        return self.language_model.model.pipeline_layers

    def __call__(
        self,
        inputs: mx.array,
        cache: Any = None,
        skip_logits: bool = False,
        **kwargs: Any,
    ) -> Any:
        if skip_logits:
            return self.language_model.model(inputs, cache=cache)
        if kwargs:
            return self.language_model(inputs, cache=cache, **kwargs)
        return self.language_model(inputs, cache=cache).logits

    def make_cache(self) -> list[Any]:
        return self.language_model.make_cache()

    def mtp_forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.language_model.mtp_forward(*args, **kwargs)

    def make_mtp_cache(self) -> list[Any]:
        return self.language_model.make_mtp_cache()

    def rollback_speculative_cache(self, *args: Any, **kwargs: Any) -> Any:
        return self.language_model.rollback_speculative_cache(*args, **kwargs)

    @property
    def _omlx_output_vocab_size(self) -> int:
        return int(self.language_model.args.vocab_size)


def _read_config(model_path: Path) -> dict[str, Any]:
    return json.loads((model_path / "config.json").read_text())


def _weight_map(model_path: Path) -> dict[str, str]:
    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    files = sorted(path.name for path in model_path.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError(
            f"Qwen4-Exp cluster ranks need an indexed checkpoint: {model_path}"
        )
    return {name: files[0] for name in mx.load(str(model_path / files[0]))}


def _layer_index(key: str) -> int | None:
    match = re.search(r"language_model\.model\.layers\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def _stage_key_needed(
    key: str,
    *,
    start: int,
    end: int,
    rank: int,
    mtp: bool,
    mmap_ple: bool,
) -> bool:
    if key.startswith(("vision_tower.", "multi_modal_projector.")):
        return False
    if key.startswith(("mtp.", "language_model.mtp.")):
        return bool(mtp and rank == 0)
    if mmap_ple and _NGRAM_STORAGE_RE.search(key):
        # The virtual PLE reader maps this table from the shard; loading it as
        # a resident array first is exactly the 35.8 GiB the plan moved to SSD.
        return False
    if key.startswith("language_model.lm_head."):
        # Every rank loads the real head. A head left unloaded is
        # ``load_model_weights(strict=False)``'s random initialisation, not a
        # saving: the planner charges the head on every rank already (it is not
        # in ``CLUSTER_UNLOADED_WEIGHT_PREFIXES``), and a rank that samples with
        # random weights returns wrong tokens silently — which is exactly what
        # ``sampling_rank_only=false`` asks it to do.
        return True
    layer = _layer_index(key)
    if layer is not None:
        return start <= layer < end
    return True


def _stage_weights(
    model_path: Path,
    *,
    start: int,
    end: int,
    rank: int,
    mtp: bool,
    mmap_ple: bool,
) -> dict[str, Any]:
    needed: dict[str, list[str]] = {}
    for key, shard in _weight_map(model_path).items():
        if _stage_key_needed(
            key,
            start=start,
            end=end,
            rank=rank,
            mtp=mtp,
            mmap_ple=mmap_ple,
        ):
            needed.setdefault(shard, []).append(key)

    weights: dict[str, Any] = {}
    for shard, keys in needed.items():
        loaded = mx.load(str(model_path / shard))
        for key in keys:
            value = loaded.get(key)
            if value is not None:
                weights[key] = value
    return weights


def load_pipeline_stage(
    model_path: Any,
    group: Any,
    *,
    engram_ssd_offload: bool = False,
) -> RankModel:
    """Load this rank's Qwen4-Exp stage, text only.

    ``engram_ssd_offload`` is the planner's per-rank decision: the rank that
    holds a PLE/Engram table mmaps it, the others keep their (empty) tables
    resident. MTP is configured before construction because the head is built
    with the model.
    """

    apply_mlx_vlm_qwen4_exp_compat_patch()
    path = Path(model_path).expanduser()
    rank = group.rank()
    mtp = bool(is_mtp_active())
    configure_qwen4_exp_runtime(
        path,
        mode="mmap" if engram_ssd_offload else "resident",
        mtp_enabled=mtp and rank == 0,
    )

    config = _read_config(path)
    args = ModelConfig.from_dict(config)
    full = _FullModel(args, text_only=True)
    # The planner's range is installed on PipelineMixin.pipeline by the worker;
    # this call is what registers pole positions, masks and local caches.
    full.language_model.model.pipeline(group)
    if mtp and rank != 0:
        full.language_model._omlx_mtp_decode_enabled = True
        full.language_model._omlx_mtp_verify_only = True
        full.language_model._omlx_mtp_chain = True
    if mtp and rank == 0 and getattr(full, "mtp", None) is not None:
        # The worker's role vote reads the head from the language model.
        full.language_model.mtp = full.mtp
        full.language_model._omlx_mtp_head_prenorm = True
        full.language_model.model._omlx_mtp_head_prenorm = True

    start = full.language_model.model.start_idx
    end = full.language_model.model.end_idx
    weights = _stage_weights(
        path,
        start=start,
        end=end,
        rank=rank,
        mtp=mtp,
        mmap_ple=engram_ssd_offload,
    )
    weights = full.sanitize(weights)
    load_model_weights(
        full,
        config,
        weights,
        path,
        lazy=False,
        strict=False,
    )
    # The loader needs traversable placeholders, the worker's stage validator
    # and PipelineMixin expect the non-owned prefix to be None again.
    if start:
        full.language_model.model.layers[:start] = [None] * start
    logger.info(
        "Qwen4-Exp rank %d loaded layers [%s, %s) (%d weights, MTP=%s, PLE=%s)",
        rank,
        start,
        end,
        len(weights),
        "head" if mtp and rank == 0 else "replay" if mtp else "off",
        "ssd" if engram_ssd_offload else "resident",
    )
    return RankModel(full.language_model)


# ``mlx_lm.utils.load_tokenizer`` resolves the architecture classes before the
# stage loader runs; expose the wrapper under the name mlx-lm expects.
Model = RankModel


__all__ = [
    "CLUSTER_MTP_WEIGHT_PREFIXES",
    "CLUSTER_UNLOADED_WEIGHT_PREFIXES",
    "HONORS_PIPELINE_ASSIGNMENT",
    "Model",
    "ModelArgs",
    "RankModel",
    "SUPPORTS_PIPELINE",
    "load_pipeline_stage",
    "pipeline_stage_starts",
]
