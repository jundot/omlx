# SPDX-License-Identifier: Apache-2.0
"""LiLiCorr draft head for the DFlash block drafter.

``nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr`` pairs the DFlash
block drafter (five Qwen3-style layers whose attention and MLP are followed by
grouped dynamic causal convolutions) with a *correlation* head that scores the
lattice of candidate tokens. The head never adds unary language-model logits:
proposals come from ``out[p-1] . in[p]`` factor correlations between a
predecessor candidate and the candidates of the next proposal position.

The graph mirrors ``llama.cpp`` commit ``843e7dcc1``,
``src/models/dflash.cpp::build_lilicorr_scores`` tensor for tensor:

* candidate token states come from the raw target token embedding table
  projected by ``lilicorr.token_proj``,
* pass states come from the draft hidden row of the same proposal position
  projected by ``lilicorr.pass_hidden_proj`` and broadcast over the candidates,
* five per-candidate features (log probability, probability, log probability
  relative to the best candidate, normalized rank, rank-0 indicator) are layer
  normalized, mixed by ``lilicorr.feature_mlp`` and added to the token/pass
  states together with the slot and rank embeddings,
* two residual correlation layers with an additive slot-attention bias,
* ``lilicorr.factor_input_proj`` fuses ``[x, anchor, x * anchor]`` before the
  ``out``/``in``/``anchor_out`` factor projections and their L2 normalization.

The target-side adapter exists because LiLiCorr layer ids denote *layer inputs*
while ``NemotronHModel.hidden_sink`` records *layer outputs*: capture id ``i``
is realized by capturing block ``i - 1`` (``[1, 18, 35, 51, 68, 85]`` becomes
``[0, 17, 34, 50, 67, 84]`` for the 88-block Bolt target).

Nothing here is imported by the engine at module scope; ``apply_lilicorr_patch``
and ``apply_lilicorr_target`` are called explicitly by the drafter loader and
the target wrapper.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.speculative.drafters.compatibility import validate_dflash_target
from mlx_vlm.speculative.drafters.dflash2.dflash2 import (
    DFlash2DecoderLayer,
    GroupedDynamicCausalConv,
)
from mlx_vlm.speculative.drafters.qwen3_dflash.config import DFlashConfig
from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel

LILICORR_ARCHITECTURE = "LiLiCorrDraftModel"
LILICORR_MODEL_TYPE = "lilicorr"
LILICORR_DRAFTER_KIND = "dflash"

# Drafters are resolved as ``mlx_vlm.speculative.drafters.<model_type>`` by
# ``mlx_vlm.utils.get_model_and_args``.
_DRAFTER_MODULE = f"mlx_vlm.speculative.drafters.{LILICORR_MODEL_TYPE}"

#: ``ggml_norm(feature, 1.0e-5f)`` in ``build_lilicorr_scores``.
FEATURE_NORM_EPS = 1e-5
#: log probability, probability, relative log probability, rank, rank-0 flag.
FEATURE_COUNT = 5

_SLOT_EMBEDDING_SUFFIX = "lilicorr.slot_embedding"
_RANK_EMBEDDING_SUFFIX = "lilicorr.rank_embedding"
_RELATIVE_BIAS_KEY = "lilicorr.relative_slot_bias"
_SAME_BIAS_KEY = "lilicorr.same_slot_bias"
_ATTENTION_BIAS_KEY = "lilicorr.attention_bias"
_IN_PROJ_WEIGHT = ".attn.in_proj_weight"
_IN_PROJ_BIAS = ".attn.in_proj_bias"

# Set by ``apply_lilicorr_target``; the only marker that enables the capture
# id shift, so untouched targets keep the plain mlx-vlm capture semantics.
_TARGET_MARKER = "_omlx_lilicorr_target"
_ADAPTER_MARKER = "_omlx_lilicorr_capture_inputs"
_SINGLE_ROW_MARKER = "_omlx_drafter_single_row"

# Read by ``mlx_lm_mtp.batch_generator._row_exact_verify``: an armed verify
# forward then keeps one-row decode arithmetic instead of the Qwen sg8/split-K
# verify kernels. Measured on Bolt: the plain arming reroutes the verifier's
# attention q/k/v tuple through ``vk_group_sg8`` (8 rows, K=4096, affine 4-bit
# group 64), which differs from the per-row matvec at bf16 tail-ULP level and
# flipped a 1-ULP greedy tie; row-exact arming is bit-identical to decode.
#
# Narrower armings were measured and rejected. Cause, localized by declining
# route classes in the live engine (fixed prompts, 256 greedy tokens, identity
# against the drafter-off run):
#
#   arming / declines                        tok/s   identical prompts
#   row-exact (shipped)                       25.3    5/5
#   sg8 single-projection declined            34.4    0/5
#   + grouped sg8 declined                    33.5    1/5
#   + lm_head qmm route declined              33.4    1/5
#   + per-linear verify route declined        29.2    0/5
#   lm_head alone on one-row arithmetic       33.3    0/5
#   + sg8 single declined                     33.7    0/5
#   + sg8 single and group declined           32.9    0/5
#   drafter off (reference)                   29.2    -
#
# It is not one op: every multi-row verify projection (sg8 group and single,
# msg/mma/skinny, the per-linear route, and stock multi-row qmm itself) rounds
# differently from serial one-row arithmetic by a few bf16 tail ULP, and on
# Bolt's 1-ULP top-2 ties that flips the greedy token. An isolated 9-row
# verify forward and a 528-token prefill under each arming state are
# bit-identical, so only the live end-to-end gate exposes it. Identity
# therefore requires one-row arithmetic for the whole verify batch, which is
# what this marker enables; the fast variants are not shippable.
_ROW_EXACT_MARKER = "_omlx_mtp_row_exact_verify"


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _config_value(config: Any, name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _is_strictly_increasing_ints(values: Any) -> bool:
    return bool(values) and all(
        isinstance(value, int) and not isinstance(value, bool)
        and (index == 0 or values[index - 1] < value)
        for index, value in enumerate(values)
    )


@dataclass
class LiLiCorrConfig(DFlashConfig):
    """Drafter config for a LiLiCorr checkpoint.

    ``dflash_config`` is flattened on load. ``num_target_layers`` stays ``0``
    unless the checkpoint declares it, so the capture ids are validated against
    the *loaded* target's layer count instead of ``max(target_layer_ids) + 1``
    (Bolt has 88 blocks and captures ids up to 85).
    """

    model_type: str = LILICORR_MODEL_TYPE
    backbone_model_type: str = "qwen3"
    architectures: list[str] = field(default_factory=list)
    hidden_act: str = "silu"
    num_target_layers: int = 0
    conv_kernel_size: int = 2
    conv_group_size: int = 16
    # mlx-vlm 0.8.1's DFlash2DecoderLayer reads this from the config; the
    # LiLiCorr backbone ropes every layer.
    no_rope_layers: list[int] = field(default_factory=list)
    lilicorr_hidden_size: int = 1024
    lilicorr_candidate_topk: int = 8
    lilicorr_num_heads: int = 8
    lilicorr_num_layers: int = 2
    lilicorr_mlp_ratio: float = 2.0
    lilicorr_factor_dim: int = 1024
    lilicorr_vector_eps: float = 1e-4
    lilicorr_logit_scale: float = 8.0

    def __post_init__(self) -> None:
        if not self.layer_types:
            kind = "sliding_attention" if self.sliding_window else "full_attention"
            self.layer_types = [kind] * self.num_hidden_layers
        self.validate()

    # -- derived geometry -------------------------------------------------
    @property
    def lilicorr_slots(self) -> int:
        """Proposal positions per block, i.e. ``block_size - 1``."""
        return int(self.block_size) - 1

    @property
    def lilicorr_nodes(self) -> int:
        """Candidate nodes per block: ``(block_size - 1) * candidate_topk``."""
        return self.lilicorr_slots * int(self.lilicorr_candidate_topk)

    @property
    def lilicorr_mlp_hidden_size(self) -> int:
        return int(round(self.lilicorr_hidden_size * self.lilicorr_mlp_ratio))

    def validate(self) -> None:
        positive = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "block_size",
            "conv_kernel_size",
            "conv_group_size",
            "lilicorr_hidden_size",
            "lilicorr_candidate_topk",
            "lilicorr_num_heads",
            "lilicorr_num_layers",
            "lilicorr_factor_dim",
        )
        for key in positive:
            if not _is_positive_int(getattr(self, key)):
                raise ValueError(f"LiLiCorr requires a positive integer {key}.")
        if self.model_type != LILICORR_MODEL_TYPE:
            raise ValueError(
                "LiLiCorr requires normalized model_type="
                f"{LILICORR_MODEL_TYPE!r}, got {self.model_type!r}."
            )
        if not isinstance(self.num_target_layers, int) or self.num_target_layers < 0:
            raise ValueError(
                "LiLiCorr num_target_layers must be a non-negative integer "
                "(0 leaves the target layer count to load time)."
            )
        if self.hidden_act != "silu":
            raise ValueError("LiLiCorr currently supports only hidden_act='silu'.")
        if not 0 < self.lilicorr_mlp_ratio:
            raise ValueError("LiLiCorr lilicorr_mlp_ratio must be positive.")
        if not 0 < self.lilicorr_vector_eps:
            raise ValueError("LiLiCorr lilicorr_vector_eps must be positive.")
        if not 0 < self.lilicorr_logit_scale:
            raise ValueError("LiLiCorr lilicorr_logit_scale must be positive.")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("LiLiCorr attention heads must be divisible by KV heads.")
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError(
                "LiLiCorr requires num_attention_heads * head_dim == hidden_size."
            )
        if self.hidden_size % self.conv_group_size:
            raise ValueError(
                "LiLiCorr hidden_size must be divisible by conv_group_size."
            )
        if self.lilicorr_hidden_size % self.lilicorr_num_heads:
            raise ValueError(
                "LiLiCorr lilicorr_hidden_size must be divisible by "
                "lilicorr_num_heads."
            )
        if self.block_size < 2:
            raise ValueError("LiLiCorr requires block_size >= 2.")
        if not 0 < self.lilicorr_candidate_topk <= self.vocab_size:
            raise ValueError(
                "LiLiCorr lilicorr_candidate_topk must fit inside the vocabulary."
            )
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("LiLiCorr mask_token_id must be inside the vocabulary.")
        if len(self.layer_types) != self.num_hidden_layers or any(
            layer_type not in {"full_attention", "sliding_attention"}
            for layer_type in self.layer_types
        ):
            raise ValueError(
                "LiLiCorr requires one supported attention type per draft layer."
            )
        if "sliding_attention" in self.layer_types and self.sliding_window is None:
            raise ValueError(
                "LiLiCorr requires sliding_window for sliding_attention layers."
            )
        if (
            not _is_strictly_increasing_ints(self.target_layer_ids)
            or any(layer_id < 1 for layer_id in self.target_layer_ids)
        ):
            raise ValueError(
                "LiLiCorr target_layer_ids must be unique, increasing layer "
                "indices; they denote target layer *inputs*, so id 0 has no "
                "preceding block output to capture."
            )
        if self.num_target_layers and any(
            layer_id > self.num_target_layers for layer_id in self.target_layer_ids
        ):
            raise ValueError(
                "LiLiCorr target_layer_ids must stay inside the declared target "
                f"layer count {self.num_target_layers}."
            )
        if self.runtime_block_size is not None and (
            not isinstance(self.runtime_block_size, int)
            or not 2 <= self.runtime_block_size <= self.block_size
        ):
            raise ValueError(
                "LiLiCorr runtime_block_size must be between 2 and block_size."
            )

    @classmethod
    def from_dict(cls, params: dict) -> "LiLiCorrConfig":
        flat = dict(params or {})
        dflash = flat.pop("dflash_config", None)
        if not isinstance(dflash, Mapping):
            raise TypeError("LiLiCorr requires a dflash_config object.")

        flat["backbone_model_type"] = str(flat.get("model_type") or "qwen3")
        flat["model_type"] = LILICORR_MODEL_TYPE
        for key in (
            "block_size",
            "mask_token_id",
            "target_layer_ids",
            "num_target_layers",
            "runtime_block_size",
            "draft_window_size",
            "final_logit_softcapping",
            "conv_kernel_size",
            "conv_group_size",
            "lilicorr_hidden_size",
            "lilicorr_candidate_topk",
            "lilicorr_num_heads",
            "lilicorr_num_layers",
            "lilicorr_mlp_ratio",
            "lilicorr_factor_dim",
            "lilicorr_vector_eps",
            "lilicorr_logit_scale",
        ):
            if key in dflash:
                flat[key] = dflash[key]
        if "causal" in dflash:
            flat["is_causal"] = bool(dflash["causal"])
        if "attention_sink_bias" in dflash:
            flat["attention_sink_bias"] = bool(dflash["attention_sink_bias"])

        projector = dflash.get("projector_type", dflash.get("projector"))
        if projector is not None and str(projector) != LILICORR_MODEL_TYPE:
            raise ValueError(
                f"LiLiCorr requires projector_type={LILICORR_MODEL_TYPE!r}, "
                f"got {projector!r}."
            )
        if dflash.get("lilicorr_enabled") is False:
            raise ValueError("LiLiCorr checkpoint declares lilicorr_enabled=false.")

        if "use_swa" in dflash and not bool(dflash["use_swa"]):
            layer_types = flat.get("layer_types") or ()
            if any(str(kind) == "sliding_attention" for kind in layer_types):
                raise ValueError(
                    "LiLiCorr use_swa=false contradicts sliding_attention layers."
                )
        swa_window = dflash.get("swa_window_size")
        if swa_window is not None:
            window = flat.get("sliding_window")
            if window is None:
                flat["sliding_window"] = int(swa_window)
            elif int(window) != int(swa_window):
                raise ValueError(
                    "LiLiCorr swa_window_size "
                    f"{swa_window} contradicts sliding_window {window}."
                )

        rope_parameters = flat.pop("rope_parameters", None)
        if isinstance(rope_parameters, Mapping):
            rope_parameters = dict(rope_parameters)
            if "rope_theta" in rope_parameters:
                flat["rope_theta"] = rope_parameters.pop("rope_theta")
            flat["rope_scaling"] = rope_parameters

        signature = {
            name
            for name in cls.__dataclass_fields__
        }
        return cls(**{key: value for key, value in flat.items() if key in signature})

    from_hf_dict = from_dict


def _l2_normalize(value: mx.array, eps: float) -> mx.array:
    """``v / max(||v||_2, eps)`` in float32 (the correlation space)."""
    wide = value.astype(mx.float32)
    norm = mx.sqrt(mx.sum(mx.square(wide), axis=-1, keepdims=True))
    return wide / mx.maximum(norm, eps)


def _expand_slot_bias(
    relative: mx.array,
    same: mx.array,
    heads: int,
    top_k: int,
    slots: int,
) -> mx.array:
    """Expand the per-head slot biases into an additive ``[heads, nodes, nodes]``.

    Node ``n`` is slot ``n // top_k`` and candidate rank ``n % top_k``, matching
    the ``[top_k, slots, blocks]`` candidate lattice of
    ``build_lilicorr_scores``. ``relative_slot_bias[h, q_slot - k_slot + slots]``
    is added for every pair; ``same_slot_bias[h]`` is added again when the
    query and key sit in the same slot.
    """
    nodes = slots * top_k
    if tuple(relative.shape) != (heads, 2 * slots + 1):
        raise ValueError(
            "LiLiCorr relative_slot_bias must be "
            f"{(heads, 2 * slots + 1)}, got {tuple(relative.shape)}."
        )
    if tuple(same.shape) != (heads,):
        raise ValueError(
            f"LiLiCorr same_slot_bias must be {(heads,)}, got {tuple(same.shape)}."
        )
    slots_per_node = mx.arange(nodes) // top_k
    offsets = slots_per_node[:, None] - slots_per_node[None, :] + slots
    bias = mx.take(relative, offsets, axis=1)
    diagonal = (slots_per_node[:, None] == slots_per_node[None, :]).astype(
        bias.dtype
    )
    return bias + same[:, None, None] * diagonal


def _head_embedding(source: Any, ids: mx.array) -> mx.array:
    """Rows of a dense embedding table (module or array) for ``ids``."""
    if isinstance(source, nn.Module):
        return source(ids)
    weight = source
    if isinstance(weight, mx.array) and weight.ndim == 2:
        return weight[ids]
    raise TypeError(
        "LiLiCorr scoring needs the target embedding table; got "
        f"{type(source).__name__}."
    )


class LiLiCorrMLP(nn.Module):
    """Non-gated SiLU stack used by the correlation layers and feature MLP."""

    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int):
        super().__init__()
        self.up_proj = nn.Linear(input_dims, hidden_dims, bias=True)
        self.down_proj = nn.Linear(hidden_dims, output_dims, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.up_proj(x)))


class LiLiCorrAttention(nn.Module):
    """Full attention over the candidate lattice with an additive slot bias."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                "LiLiCorr lilicorr_hidden_size must be divisible by "
                "lilicorr_num_heads."
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, x: mx.array, bias: mx.array) -> mx.array:
        batch, length, width = x.shape
        queries, keys, values = mx.split(self.in_proj(x), 3, axis=-1)

        def heads(tensor: mx.array) -> mx.array:
            split = tensor.reshape(batch, length, self.num_heads, self.head_dim)
            return split.transpose(0, 2, 1, 3)

        attended = mx.fast.scaled_dot_product_attention(
            heads(queries),
            heads(keys),
            heads(values),
            scale=self.scale,
            mask=bias.astype(x.dtype),
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, width)
        return self.out_proj(attended)


class LiLiCorrLayer(nn.Module):
    """One LiLiCorr correlation layer: pre-norm attention then pre-norm MLP."""

    def __init__(self, config: LiLiCorrConfig):
        super().__init__()
        hidden = int(config.lilicorr_hidden_size)
        self.attn_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.mlp_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.attn = LiLiCorrAttention(hidden, int(config.lilicorr_num_heads))
        self.mlp = LiLiCorrMLP(
            hidden, config.lilicorr_mlp_hidden_size, hidden
        )

    def __call__(self, x: mx.array, bias: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x), bias)
        return x + self.mlp(self.mlp_norm(x))


class LiLiCorrHead(nn.Module):
    """Candidate lattice scorer (``lilicorr.*`` weights of the draft checkpoint)."""

    def __init__(self, config: LiLiCorrConfig):
        super().__init__()
        self.config = config
        hidden = int(config.hidden_size)
        width = int(config.lilicorr_hidden_size)
        top_k = int(config.lilicorr_candidate_topk)
        slots = int(config.block_size) - 1
        heads = int(config.lilicorr_num_heads)
        self.top_k = top_k
        self.slots = slots
        self.num_heads = heads
        self.vector_eps = float(config.lilicorr_vector_eps)
        self.logit_scale = float(config.lilicorr_logit_scale)

        self.token_proj = nn.Linear(hidden, width, bias=True)
        self.pass_hidden_proj = nn.Linear(hidden, width, bias=True)
        self.context_proj = nn.Linear(hidden, width, bias=True)
        self.feature_norm = nn.LayerNorm(FEATURE_COUNT, eps=FEATURE_NORM_EPS)
        self.feature_mlp = LiLiCorrMLP(FEATURE_COUNT, width, width)
        self.slot_embedding = nn.Embedding(slots, width)
        self.rank_embedding = nn.Embedding(top_k, width)
        # Derived by ``LiLiCorrDraftModel.sanitize`` from the checkpoint's
        # ``relative_slot_bias`` and ``same_slot_bias`` per-head vectors.
        self.attention_bias = mx.zeros((heads, slots * top_k, slots * top_k))
        self.layers = [
            LiLiCorrLayer(config) for _ in range(int(config.lilicorr_num_layers))
        ]
        self.output_norm = nn.RMSNorm(width, eps=config.rms_norm_eps)
        self.anchor_norm = nn.RMSNorm(width, eps=config.rms_norm_eps)
        self.factor_input_proj = nn.Linear(3 * width, width, bias=True)
        self.out_head = nn.Linear(width, int(config.lilicorr_factor_dim), bias=True)
        self.in_head = nn.Linear(width, int(config.lilicorr_factor_dim), bias=True)
        self.anchor_out_head = nn.Linear(
            width, int(config.lilicorr_factor_dim), bias=True
        )

    # -- scoring ----------------------------------------------------------
    def head_dtype(self) -> mx.Dtype:
        weight = self.out_head.weight
        return weight.dtype if mx.issubdtype(weight.dtype, mx.floating) else mx.bfloat16

    def score(
        self,
        hidden: mx.array,
        logits: mx.array,
        token_embeddings: Any,
        anchor_context: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Score the candidate lattice of a draft block.

        Args:
            hidden: draft hidden rows ``[B, positions, hidden_size]``; only the
                first ``min(positions, block_size - 1)`` proposal positions are
                scored, so a runtime block shorter than the trained one works.
            logits: draft logits ``[B, positions, vocab_size]`` for the same
                positions.
            token_embeddings: the target embedding table (``nn.Embedding`` or a
                dense ``[vocab_size, hidden_size]`` array) used for the
                candidate token states.
            anchor_context: the newest committed projected context row
                ``[B, hidden_size]``.

        Returns:
            ``(candidates, scores)`` with ``candidates`` int32
            ``[B, slots, top_k]`` sorted by descending logit and ``scores``
            float32 ``[B, slots, top_k, top_k]`` where the last two axes are
            ``(predecessor, candidate)`` and already scaled by
            ``lilicorr_logit_scale``. Position 0 has a single predecessor — the
            anchor — in row 0; its remaining rows are zero, matching the
            ``ggml_pad`` layout of ``build_lilicorr_scores``. No unary logits
            are added.
        """
        if hidden.ndim != 3 or logits.ndim != 3:
            raise ValueError(
                "LiLiCorr scoring expects [batch, positions, width] inputs."
            )
        if hidden.shape[0] != logits.shape[0]:
            raise ValueError("LiLiCorr hidden and logits must share a batch size.")
        if hidden.shape[-1] != int(self.config.hidden_size):
            raise ValueError(
                "LiLiCorr draft hidden width mismatch: expected "
                f"{self.config.hidden_size}, got {hidden.shape[-1]}."
            )
        batch = int(hidden.shape[0])
        slots = min(int(hidden.shape[1]), int(logits.shape[1]), self.slots)
        if slots < 1:
            raise ValueError("LiLiCorr scoring needs at least one proposal position.")
        if anchor_context.ndim != 2 or anchor_context.shape != (
            batch,
            int(self.config.hidden_size),
        ):
            raise ValueError(
                "LiLiCorr anchor context must be [batch, hidden_size]; got "
                f"{tuple(anchor_context.shape)}."
            )

        dtype = self.head_dtype()
        top_k = self.top_k
        vocab = int(logits.shape[-1])
        wide_logits = logits[:, :slots].astype(mx.float32)
        probs = mx.softmax(wide_logits, axis=-1)
        if top_k < vocab:
            picked = mx.argpartition(wide_logits, -top_k, axis=-1)[..., -top_k:]
        else:
            picked = mx.broadcast_to(
                mx.arange(vocab)[None, None, :], (batch, slots, vocab)
            )
        picked_logits = mx.take_along_axis(wide_logits, picked, axis=-1)
        order = mx.argsort(-picked_logits, axis=-1)
        candidates = mx.take_along_axis(picked, order, axis=-1).astype(mx.int32)
        candidate_probs = mx.take_along_axis(probs, candidates, axis=-1)
        log_probs = mx.log(mx.maximum(candidate_probs, 1e-30))

        count = candidates.shape[-1]
        rank = mx.arange(count, dtype=mx.float32) / max(count - 1, 1)
        rank_feature = mx.broadcast_to(rank, candidates.shape)
        first_feature = mx.broadcast_to(
            (mx.arange(count) == 0).astype(mx.float32), candidates.shape
        )
        features = mx.stack(
            [
                log_probs,
                candidate_probs,
                log_probs - log_probs[..., :1],
                rank_feature,
                first_feature,
            ],
            axis=-1,
        )
        feature_states = self.feature_mlp(self.feature_norm(features).astype(dtype))

        rows = _head_embedding(token_embeddings, candidates).astype(dtype)
        token_states = self.token_proj(rows)
        pass_states = self.pass_hidden_proj(hidden[:, :slots].astype(dtype))
        pass_states = mx.broadcast_to(pass_states[:, :, None, :], token_states.shape)
        positions = (
            self.slot_embedding(mx.arange(slots))[:, None, :]
            + self.rank_embedding(mx.arange(count))[None, :, :]
        )
        x = token_states + pass_states + feature_states + positions
        x = x.reshape(batch, slots * count, -1)

        anchor_state = self.context_proj(anchor_context.astype(dtype))
        anchor_state = anchor_state.reshape(batch, 1, -1)
        bias = self.attention_bias[:, : x.shape[1], : x.shape[1]]
        for layer in self.layers:
            x = layer(x, bias)

        x = self.output_norm(x)
        anchor_nodes = self.anchor_norm(anchor_state)
        anchor_all = mx.broadcast_to(anchor_nodes, (batch, x.shape[1], x.shape[-1]))
        factors = nn.silu(
            self.factor_input_proj(
                mx.concatenate([x, anchor_all, x * anchor_all], axis=-1)
            )
        )
        out = _l2_normalize(self.out_head(factors), self.vector_eps)
        into = _l2_normalize(self.in_head(factors), self.vector_eps)
        anchor_out = _l2_normalize(
            self.anchor_out_head(anchor_nodes.reshape(batch, -1)), self.vector_eps
        )

        factor_dim = out.shape[-1]
        out = out.reshape(batch, slots, top_k, factor_dim)
        into = into.reshape(batch, slots, top_k, factor_dim)
        # Position 0 has one predecessor — the anchor — which the lattice gives
        # rank 0; later positions correlate out[p-1] with in[p].
        first = mx.sum(into[:, 0] * anchor_out[:, None, :], axis=-1) * self.logit_scale
        first_block = mx.concatenate(
            [first[:, None, :], mx.zeros((batch, top_k - 1, top_k), dtype=mx.float32)],
            axis=1,
        )
        parts = [first_block[:, None, :, :]]
        if slots > 1:
            previous = out[:, :-1].astype(mx.float32)
            current = into[:, 1:].astype(mx.float32)
            correlated = mx.matmul(previous, current.swapaxes(-1, -2))
            parts.append(correlated * self.logit_scale)
        scores = mx.concatenate(parts, axis=1)
        return candidates, scores


class LiLiCorrDraftModel(DFlashDraftModel):
    """DFlash draft stack plus the LiLiCorr correlation head."""

    layer_class = DFlash2DecoderLayer
    # ``block_size`` is the trained ceiling; ``LiLiCorrHead.score`` scores a
    # prefix of the lattice when the adaptive controller verifies fewer rows.

    def __init__(self, config: LiLiCorrConfig):
        super().__init__(config)
        self.lilicorr = LiLiCorrHead(config)

    def validate_target_compatibility(self, target_model: Any) -> None:
        """Check hidden/vocab plus every capture id against the target's blocks."""
        config = self.config
        language_model = getattr(target_model, "language_model", target_model)
        inner = getattr(language_model, "model", language_model)
        layers = getattr(inner, "layers", None) or getattr(
            language_model, "layers", None
        )
        layer_count = len(layers) if layers is not None else None
        declared = int(getattr(config, "num_target_layers", 0) or 0)
        if not layer_count:
            target_config = getattr(language_model, "config", None) or getattr(
                language_model, "args", None
            )
            layer_count = _config_value(target_config, "num_hidden_layers")
        if not layer_count:
            layer_count = declared
        if not layer_count:
            raise ValueError(
                "LiLiCorr cannot determine the target layer count; the capture "
                "ids denote target layer inputs and must be checked."
            )
        layer_count = int(layer_count)
        if declared and declared != layer_count:
            raise ValueError(
                "LiLiCorr target layer-count mismatch: "
                f"declared={declared}, target={layer_count}."
            )
        outside = [
            int(layer_id)
            for layer_id in config.target_layer_ids
            if not 1 <= int(layer_id) <= layer_count
        ]
        if outside:
            raise ValueError(
                "LiLiCorr target_layer_ids "
                f"{outside} fall outside 1..{layer_count}: the ids denote target "
                "layer inputs, so id i reads the output of block i - 1."
            )
        # ``validate_dflash_target`` checks hidden size, vocabulary and cache
        # rollback; give it the resolved layer count so it does not compare
        # against ``max(target_layer_ids) + 1``.
        probe = copy.copy(config)
        probe.num_target_layers = int(layer_count)
        validate_dflash_target(probe, target_model, "LiLiCorr")

    def bind(self, target_model: Any) -> "LiLiCorrDraftModel":
        self.validate_target_compatibility(target_model)
        super().bind(target_model)
        return self

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        """Map the published checkpoint names onto this module's parameters.

        Packed NVFP4 weights (``uint8`` here, ``uint32`` after mlx-vlm's
        ModelOpt transform), their ``.scales`` and their global
        ``.weight_scale_2`` pass through untouched.
        """
        config = self.config
        heads = int(config.lilicorr_num_heads)
        top_k = int(config.lilicorr_candidate_topk)
        slots = int(config.block_size) - 1
        width = int(config.lilicorr_hidden_size)
        relative = None
        same = None
        normalized: dict[str, mx.array] = {}
        for raw_key, value in weights.items():
            key = raw_key[len("model.") :] if raw_key.startswith("model.") else raw_key
            key = key.replace(_IN_PROJ_WEIGHT, ".attn.in_proj.weight").replace(
                _IN_PROJ_BIAS, ".attn.in_proj.bias"
            )
            if key == _SLOT_EMBEDDING_SUFFIX:
                value = _reshape_embedding(key, value, slots, width)
                key = f"{key}.weight"
            elif key == _RANK_EMBEDDING_SUFFIX:
                value = _reshape_embedding(key, value, top_k, width)
                key = f"{key}.weight"
            elif key == _RELATIVE_BIAS_KEY:
                relative = value
                continue
            elif key == _SAME_BIAS_KEY:
                same = value
                continue
            if key in normalized:
                raise ValueError(
                    f"Duplicate LiLiCorr weight key after sanitization: {key}"
                )
            normalized[key] = value
        if relative is None or same is None:
            raise ValueError(
                "LiLiCorr draft checkpoint is missing its slot attention biases: "
                f"expected {_RELATIVE_BIAS_KEY} and {_SAME_BIAS_KEY}, got "
                f"{sorted(weights)[:3]}... ({len(weights)} tensors)."
            )
        normalized[_ATTENTION_BIAS_KEY] = _expand_slot_bias(
            relative, same, heads, top_k, slots
        )
        return normalized


def _reshape_embedding(
    key: str, value: mx.array, rows: int, width: int
) -> mx.array:
    if value.size != rows * width:
        raise ValueError(
            f"LiLiCorr {key} must hold {rows} rows of {width}; got "
            f"{tuple(value.shape)}."
        )
    return value.reshape(rows, width)


Model = LiLiCorrDraftModel
ModelConfig = LiLiCorrConfig


# ---------------------------------------------------------------------------
# Process-wide registration and wrappers
# ---------------------------------------------------------------------------


def _is_lilicorr_checkpoint(weights: Mapping[str, Any]) -> bool:
    for key in weights:
        if key.endswith(_SLOT_EMBEDDING_SUFFIX) or key.endswith(
            _RANK_EMBEDDING_SUFFIX
        ):
            return True
    return False


def apply_lilicorr_patch() -> bool:
    """Register the LiLiCorr drafter with mlx-vlm. Idempotent, process-wide.

    * makes ``mlx_vlm.speculative.drafters.lilicorr`` resolve to this module so
      ``mlx_vlm.utils.get_model_and_args`` can load the checkpoint,
    * maps model type ``lilicorr`` to the ``dflash`` round loop,
    * wraps ``mlx_vlm.utils.get_model_type`` to report ``lilicorr`` only for the
      ``LiLiCorrDraftModel`` architecture,
    * wraps ``mlx_vlm.quantization.checkpoint._transform_modelopt_nvfp4_weights``
      to read ModelOpt's ``modelopt_fp4`` tag as ``modelopt`` for LiLiCorr
      checkpoints only, so packed NVFP4 weights stay packed.

    Every wrapper passes other architectures through unchanged.
    """
    import mlx_vlm.quantization.checkpoint as checkpoint
    import mlx_vlm.speculative.drafters as drafters
    import mlx_vlm.utils as vlm_utils

    module = sys.modules[__name__]
    if sys.modules.get(_DRAFTER_MODULE) is not module:
        sys.modules[_DRAFTER_MODULE] = module
    if getattr(drafters, LILICORR_MODEL_TYPE, None) is not module:
        setattr(drafters, LILICORR_MODEL_TYPE, module)
    drafters.DRAFTER_KIND_BY_MODEL_TYPE.setdefault(
        LILICORR_MODEL_TYPE, LILICORR_DRAFTER_KIND
    )

    current_model_type = vlm_utils.get_model_type
    if not getattr(current_model_type, "_omlx_lilicorr_wrapper", False):

        def _lilicorr_get_model_type(config, *args, **kwargs):
            try:
                architectures = set((config or {}).get("architectures") or ())
            except AttributeError:
                architectures = set(getattr(config, "architectures", None) or ())
            if LILICORR_ARCHITECTURE in architectures:
                return LILICORR_MODEL_TYPE
            return current_model_type(config, *args, **kwargs)

        setattr(_lilicorr_get_model_type, "_omlx_lilicorr_wrapper", True)
        vlm_utils.get_model_type = _lilicorr_get_model_type

    current_transform = checkpoint._transform_modelopt_nvfp4_weights
    if not getattr(current_transform, "_omlx_lilicorr_wrapper", False):

        def _lilicorr_transform(weights, quantization_config, *args, **kwargs):
            if (
                isinstance(quantization_config, Mapping)
                and quantization_config.get("quant_method") == "modelopt_fp4"
                and _is_lilicorr_checkpoint(weights)
            ):
                quantization_config = {
                    **quantization_config,
                    "quant_method": "modelopt",
                }
            return current_transform(weights, quantization_config, *args, **kwargs)

        setattr(_lilicorr_transform, "_omlx_lilicorr_wrapper", True)
        checkpoint._transform_modelopt_nvfp4_weights = _lilicorr_transform

    return True


def shift_lilicorr_capture_ids(layer_ids: Iterable[int]) -> list[int]:
    """Translate LiLiCorr *input* ids into ``hidden_sink`` *output* indices."""
    shifted = []
    for layer_id in layer_ids:
        index = int(layer_id)
        if index < 1:
            raise ValueError(
                "LiLiCorr capture ids denote target layer inputs; id 0 has no "
                "preceding block output to capture."
            )
        shifted.append(index - 1)
    return shifted


def _lilicorr_capture_adapter(cls: type) -> type:
    def __call__(self, *args, **kwargs):
        capture = kwargs.get("capture_layer_ids")
        if capture is not None:
            kwargs["capture_layer_ids"] = shift_lilicorr_capture_ids(capture)
        if kwargs.get("n_confirmed"):
            # A block-drafter verify forward (the chain cycle always passes
            # ``n_confirmed``) has to run through mlx-vlm's native exact
            # verifier: only that route returns the per-layer Mamba states
            # ``rollback_speculative_cache`` replays a partial accept
            # through. The stock body leaves ``gdn_states`` empty, which the
            # batch generator reads as "cache layer rejects chain rollback"
            # and falls back to standard decoding on every step.
            kwargs["speculative_verify"] = True
        return super(adapter, self).__call__(*args, **kwargs)

    adapter = type(
        f"LiLiCorrCaptureAdapter{cls.__name__}",
        (cls,),
        {
            "__call__": __call__,
            "_omlx_lilicorr_capture_inputs": True,
            "__module__": cls.__module__,
            "__qualname__": f"{cls.__qualname__}.LiLiCorrCaptureInputs",
            "__doc__": (
                "LiLiCorr target adapter: capture ids denote layer inputs, so "
                "requested id i reads the output of block i - 1."
            ),
        },
    )
    return adapter


def apply_lilicorr_target(target_model: Any) -> Any:
    """Adapt a Nemotron-H target for LiLiCorr capture semantics.

    Marks *this* target only (never the class globally), shifts every requested
    capture id ``i`` to ``i - 1`` because ``NemotronHModel.hidden_sink`` appends
    block *outputs*, pins ``_omlx_drafter_single_row`` since
    ``rollback_speculative_cache`` needs uniform acceptance across the batch,
    and opts the target into row-exact verify arming: the generic MTP verify
    qmm routing perturbs this target's already-exact verify forward.
    """
    language_model = getattr(target_model, "language_model", target_model)
    if not hasattr(language_model, "rollback_speculative_cache"):
        raise TypeError(
            "LiLiCorr requires a target with speculative cache rollback; "
            f"{type(language_model).__name__} has none."
        )
    backbone = getattr(language_model, "backbone", None)
    if backbone is None or not hasattr(backbone, "layers"):
        raise TypeError(
            "LiLiCorr target adapter expects a Nemotron-H language model with a "
            f"block stack; {type(language_model).__name__} has none."
        )
    if not callable(getattr(language_model, "speculative_verify_dflash_hidden", None)):
        raise TypeError(
            "LiLiCorr verify requires mlx-vlm's native exact verifier route "
            f"(speculative_verify=True); {type(language_model).__name__} has none."
        )
    cls = type(language_model)
    if not getattr(cls, _ADAPTER_MARKER, False):
        language_model.__class__ = _lilicorr_capture_adapter(cls)
    setattr(language_model, _TARGET_MARKER, True)
    setattr(language_model, _SINGLE_ROW_MARKER, True)
    setattr(language_model, _ROW_EXACT_MARKER, True)
    return target_model


__all__ = [
    "DFlash2DecoderLayer",
    "GroupedDynamicCausalConv",
    "LILICORR_ARCHITECTURE",
    "LILICORR_DRAFTER_KIND",
    "LILICORR_MODEL_TYPE",
    "LiLiCorrAttention",
    "LiLiCorrConfig",
    "LiLiCorrDraftModel",
    "LiLiCorrHead",
    "LiLiCorrLayer",
    "LiLiCorrMLP",
    "Model",
    "ModelConfig",
    "apply_lilicorr_patch",
    "apply_lilicorr_target",
    "shift_lilicorr_capture_ids",
]
