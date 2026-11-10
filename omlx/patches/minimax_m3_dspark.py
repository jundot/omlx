# SPDX-License-Identifier: Apache-2.0
"""Bundled DSpark drafter for MiniMax M3.1.

MiniMax M3.1 ships its DSpark draft model in a ``dspark/`` folder next to the
target weights. The draft is a five-layer DFlash-style block model:

- the target's residual stream after layers ``target_layer_ids`` is
  concatenated, projected by ``fc`` and normalized by ``hidden_norm`` into a
  sliding-window context that every draft layer projects into its own K/V;
- the block ``[anchor, noise x (block_size - 1)]`` attends to that context and
  to itself without a causal mask, through MiniMax-style layers (Gemma norms,
  per-head q/k norms, partial RoPE, SwiGLU-OAI dense FFN);
- every block position produces a proposal through the target ``lm_head``,
  the anchor position included, and a rank-R Markov head adds a bias from
  the previous token before each proposal is sampled.

The draft runs on the batched block-drafter path in
:mod:`omlx.speculative.dflash`; this module provides the model and its
checkpoint loader. The target-side hooks live in the vendored MiniMax
language model.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.rope_utils import initialize_rope
from mlx_vlm.speculative.drafters.dspark.dspark import VanillaMarkov

logger = logging.getLogger(__name__)

DSPARK_DIRNAME = "dspark"
_ARCHITECTURE = "DSparkMiniMaxDraftModel"
_WEIGHT_PREFIX = "language_model.model.dspark."
_MXFP8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}


def _vendored_language():
    """The vendored MiniMax M3 language module (needs the compat namespace)."""
    from .mlx_vlm_minimax_m3_compat import apply_mlx_vlm_minimax_m3_compat_patch

    apply_mlx_vlm_minimax_m3_compat_patch()
    return importlib.import_module("mlx_vlm.models.minimax_m3_vl.language")


def bundled_dspark_path(model_path: str | Path) -> Path | None:
    """``<model>/dspark`` when it holds a MiniMax DSpark draft checkpoint."""
    path = Path(model_path) / DSPARK_DIRNAME
    config_path = path / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    text_config = config.get("text_config") or config
    architectures = set(config.get("architectures") or ()) | set(
        text_config.get("architectures") or ()
    )
    if _ARCHITECTURE not in architectures:
        return None
    if not any(path.glob("*.safetensors")):
        return None
    return path


@dataclass
class MiniMaxDSparkConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    vocab_size: int
    ffn_hidden_size: int
    swiglu_alpha: float
    swiglu_limit: float
    swiglu_beta: float
    # Block inputs: the anchor plus ``block_size - 1`` noise tokens. Every
    # input position yields a proposal, so the draft depth is block_size.
    block_size: int
    mask_token_id: int
    target_layer_ids: list[int]
    sliding_window: int
    markov_rank: int
    sample_from_anchor: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> MiniMaxDSparkConfig:
        text = config.get("text_config") or config
        dspark = text.get("dspark_config") or {}
        head_dim = int(text["head_dim"])
        rotary_dim = text.get("rotary_dim") or int(
            head_dim * float(text.get("partial_rotary_factor", 1.0))
        )
        if str(text.get("dspark_markov_head_type", "vanilla")) != "vanilla":
            raise ValueError("MiniMax DSpark supports only the vanilla Markov head")
        if not text.get("dspark_use_dense_ffn", True):
            raise ValueError("MiniMax DSpark supports only the dense draft FFN")
        layer_types = text.get("layer_types") or []
        if any(layer_type != "sliding_attention" for layer_type in layer_types):
            raise ValueError("MiniMax DSpark expects sliding-window draft layers")
        return cls(
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=head_dim,
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(text["rope_theta"]),
            rotary_dim=int(rotary_dim),
            vocab_size=int(text["vocab_size"]),
            ffn_hidden_size=int(
                text.get("dspark_ffn_hidden_size")
                or dspark.get("ffn_hidden_size")
                or text["dense_intermediate_size"]
            ),
            swiglu_alpha=float(text.get("swiglu_alpha", 1.702)),
            swiglu_limit=float(text.get("swiglu_limit", 7.0)),
            swiglu_beta=float(text.get("swiglu_beta", 1.0)),
            block_size=int(text["dspark_block_size"]),
            mask_token_id=int(
                text.get("dspark_noise_token_id", dspark.get("mask_token_id"))
            ),
            target_layer_ids=[int(i) for i in text["dspark_target_layer_ids"]],
            sliding_window=int(text["sliding_window"]),
            markov_rank=int(text.get("dspark_markov_rank", dspark["markov_rank"])),
        )


class MiniMaxDSparkAttention(nn.Module):
    """MiniMax attention weights in the layout the block drafter drives."""

    def __init__(self, config: MiniMaxDSparkConfig, language):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, hidden, bias=False)
        self.q_norm = language.MiniMaxRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = language.MiniMaxRMSNorm(self.head_dim, config.rms_norm_eps)


class MiniMaxDSparkLayer(nn.Module):
    def __init__(self, config: MiniMaxDSparkConfig, language):
        super().__init__()
        self.self_attn = MiniMaxDSparkAttention(config, language)
        self.mlp = language.MiniMaxMLP(
            config.hidden_size,
            config.ffn_hidden_size,
            config.swiglu_alpha,
            config.swiglu_limit,
            config.swiglu_beta,
        )
        self.input_layernorm = language.MiniMaxRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = language.MiniMaxRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )


class MiniMaxDSparkDraftModel(nn.Module):
    """DSpark draft weights plus the target embeddings and head it borrows."""

    def __init__(self, config: MiniMaxDSparkConfig):
        super().__init__()
        language = _vendored_language()
        self.config = config
        hidden = config.hidden_size
        self.fc = nn.Linear(hidden * len(config.target_layer_ids), hidden, bias=False)
        self.hidden_norm = language.MiniMaxRMSNorm(hidden, config.rms_norm_eps)
        self.layers = [
            MiniMaxDSparkLayer(config, language)
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = language.MiniMaxRMSNorm(hidden, config.rms_norm_eps)
        self.markov_head = VanillaMarkov(config.vocab_size, config.markov_rank)
        self.rope = initialize_rope(
            config.rotary_dim, config.rope_theta, traditional=False
        )
        # Bound methods of the target modules: stored as plain attributes so
        # the target's embeddings and head stay out of this module's tree.
        self._embed_fn = None
        self._lm_head_fn = None

    def bind(self, target_model: Any) -> MiniMaxDSparkDraftModel:
        language_model = getattr(target_model, "language_model", target_model)
        inner = getattr(language_model, "model", language_model)
        embed = getattr(inner, "embed_tokens", None)
        if embed is None:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        head = getattr(language_model, "lm_head", None)
        self._embed_fn = embed.__call__
        self._lm_head_fn = head.__call__ if head is not None else embed.as_linear
        return self

    def _embed_input_tokens(self, tokens: mx.array) -> mx.array:
        return self._embed_fn(tokens)

    def _logits(self, hidden: mx.array) -> mx.array:
        return self._lm_head_fn(hidden)


def sanitize_dspark_weights(weights: dict) -> dict:
    """Map ``language_model.model.dspark.*`` checkpoint keys onto the model.

    MXFP8 projections keep their E4M3 bytes and E8M0 block scales as MLX
    ``mxfp8`` tensors. The confidence head is only used by verify-length
    budgeting, which the batched verify path does not do.
    """
    sanitized = {}
    for key, value in weights.items():
        if not key.startswith(_WEIGHT_PREFIX):
            raise ValueError(f"Unexpected MiniMax DSpark weight: {key}")
        key = key[len(_WEIGHT_PREFIX) :]
        if key.startswith("confidence_head."):
            continue
        key = re.sub(r"^layers\.(\d+)\.decoder_layer\.", r"layers.\1.", key)
        if key.startswith("final_norm."):
            key = "norm." + key[len("final_norm.") :]
        if key.endswith(".weight_scale_inv"):
            key = key[: -len("weight_scale_inv")] + "scales"
        sanitized[key] = value
    for key in [k for k in sanitized if k.endswith(".scales")]:
        weight_key = key[: -len("scales")] + "weight"
        sanitized[weight_key] = sanitized[weight_key].view(mx.uint32)
    return sanitized


def load_minimax_dspark_model(path: str | Path) -> MiniMaxDSparkDraftModel:
    path = Path(path)
    config = MiniMaxDSparkConfig.from_dict(
        json.loads((path / "config.json").read_text())
    )
    model = MiniMaxDSparkDraftModel(config)
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    weights = sanitize_dspark_weights(weights)
    nn.quantize(
        model,
        class_predicate=lambda p, m: (
            dict(_MXFP8)
            if f"{p}.scales" in weights and hasattr(m, "to_quantized")
            else False
        ),
    )
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def wrap_minimax_dspark(
    model: MiniMaxDSparkDraftModel,
    source_path: str,
    depth: int | None = None,
    fixed: bool = False,
):
    """Wrap a bound MiniMax DSpark draft model for the batched verify path.

    ``depth`` caps the drafts a cycle verifies, exactly that many when
    ``fixed``. The block stays the trained one: its rows attend each other
    both ways, so a shorter block would change the leading proposals too.
    """
    from ..speculative.dflash import DFlashDrafter

    # The drafter's DSpark layout is the anchor plus block_size - 2 masks;
    # MiniMax's block_size counts the anchor plus block_size - 1 noise tokens.
    drafter = DFlashDrafter(
        model,
        block_size=model.config.block_size + 1,
        source_path=source_path,
        adaptive=not fixed,
    )
    if depth:
        drafter.depth = max(1, min(drafter.depth, int(depth)))
    drafter.query_windows = True
    drafter.context_copy = True
    return drafter


def load_minimax_dspark_drafter(
    path: str | Path,
    target_model: Any,
    depth: int | None = None,
    fixed: bool = False,
):
    """Load the bundled draft and wrap it for the batched verify path."""
    model = load_minimax_dspark_model(path).bind(target_model)
    drafter = wrap_minimax_dspark(model, str(path), depth, fixed)
    logger.info(
        "MiniMax DSpark drafter loaded: path=%s block=%d depth=%d adaptive=%s "
        "target_layers=%s",
        path,
        drafter.block_size,
        drafter.depth,
        drafter.adaptive,
        drafter.target_layer_ids,
    )
    return drafter
