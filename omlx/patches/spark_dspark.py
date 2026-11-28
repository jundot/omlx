# SPDX-License-Identifier: Apache-2.0
"""Bundled DSpark drafter for Spark X2.5/X2.6.

Spark ships its DSpark draft model as a ``*DSpark`` checkpoint next to the
target weights. The draft is a five-layer DFlash-style block model whose
draft layers differ from the stock DFlash/dspark template in three ways:

- the draft MLP uses GELU gating (``down(gelu(gate(x)) * up(x))``) instead of
  the stock SiLU/SwiGLU, matching Spark2_5's dense MLP;
- each draft layer gates its attention output with a head-wise sigmoid
  projection (``attn_gate`` / ``g_proj``), matching Spark2_5 attention;
- the block projector normalizes every captured target layer with its own
  RMSNorm (``fc_norm``, one weight vector per target layer) before the shared
  ``fc`` projection and the single ``hidden_norm`` (``enc.output_norm``).

The draft runs on the batched block-drafter path in
:mod:`omlx.speculative.dflash`; this module provides the model and its
checkpoint loader. The target-side hooks live in the mlx_lm Spark model.
"""

from __future__ import annotations

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

_WEIGHT_PREFIX = "language_model.model.dspark."
_MXFP8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}


class _IdentityNorm:
    """Identity norm for Spark draft attention Q/K.

    llama.cpp dflash.cpp has NO attn_q_norm/attn_k_norm tensors, so the draft
    attention uses plain Q/K without RMS normalization. The safetensors do
    carry q_norm/k_norm weights but they are all-1.0 (pure RMS, no scale),
    which measurably degrades acceptance (53.0% -> 36.8%). Keep identity to
    match llama.cpp.
    """

    def __call__(self, x):
        return x


@dataclass
class SparkDSparkConfig:
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
    # Block inputs: the anchor plus ``block_size - 1`` noise tokens. Every
    # input position yields a proposal, so the draft depth is block_size.
    block_size: int
    mask_token_id: int
    target_layer_ids: list[int]
    sliding_window: int
    markov_rank: int
    sample_from_anchor: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> SparkDSparkConfig:
        text = config.get("text_config") or config
        dspark = text.get("dflash_config") or text.get("dspark_config") or {}
        head_dim = int(text["head_dim"])
        rotary_dim = text.get("rotary_dim") or int(
            head_dim * float(text.get("partial_rotary_factor", 1.0))
        )
        if str(text.get("markov_head_type", "vanilla")) != "vanilla":
            raise ValueError("Spark DSpark supports only the vanilla Markov head")
        if str(text.get("hidden_act", "gelu")) != "gelu":
            raise ValueError(
                f"Spark DSpark requires GELU, got {text.get('hidden_act')!r}"
            )
        layer_types = text.get("layer_types") or []
        if layer_types and any(
            layer_type != "sliding_attention" for layer_type in layer_types
        ):
            raise ValueError("Spark DSpark expects sliding-window draft layers")
        target_layer_ids = [int(i) for i in text["target_layer_ids"]]
        return cls(
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=head_dim,
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(text.get("rope_theta", 5000000.0)),
            rotary_dim=int(rotary_dim),
            vocab_size=int(text["vocab_size"]),
            ffn_hidden_size=int(text.get("intermediate_size", 10240)),
            block_size=int(text["block_size"]),
            # The GGUF metadata is authoritative: dflash.mask_token_id=2 for
            # Spark, while the HF config.json top-level mask_token_id=3 is a
            # cosmetic inconsistency (the top-level key is ignored by the
            # official converter). Prefer dflash_config, then GGUF, then top.
            mask_token_id=int(
                dspark.get("mask_token_id")
                or text.get("mask_token_id")
                or 2
            ),
            target_layer_ids=target_layer_ids,
            sliding_window=int(text.get("sliding_window", 512)),
            markov_rank=int(text.get("markov_rank", dspark.get("markov_rank", 256))),
            sample_from_anchor=bool(dspark.get("sample_from_anchor", True)),
        )


class SparkDSparkAttention(nn.Module):
    """Spark draft attention: separate Q/K/V/O plus head-wise sigmoid gate."""

    def __init__(self, config: SparkDSparkConfig):
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
        # Spark gates the attention output per head with a sigmoid projection.
        self.g_proj = nn.Linear(hidden, self.n_heads, bias=False)
        # Identity Q/K norm (see _IdentityNorm docstring); llama.cpp never
        # applies attn_q_norm/attn_k_norm for the draft attention.
        self.q_norm = _IdentityNorm()
        self.k_norm = _IdentityNorm()
        # The drafter's gate_output reads this to pick the activation.
        self.gate_act = "sigmoid"
        # Official llama.cpp dflash.cpp: non-causal draft context (block
        # queries see the main-KV ring plus every block key).
        self.is_causal = False
        self.is_sliding = True
        self.sliding_window = config.sliding_window


class SparkDSparkMLP(nn.Module):
    """Spark dense draft MLP with GELU gating (matches Spark2_5)."""

    def __init__(self, config: SparkDSparkConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.ffn_hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.ffn_hidden_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.ffn_hidden_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class SparkDSparkLayer(nn.Module):
    def __init__(self, config: SparkDSparkConfig):
        super().__init__()
        self.self_attn = SparkDSparkAttention(config)
        self.mlp = SparkDSparkMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class SparkDSparkDraftModel(nn.Module):
    """Spark DSpark draft weights plus the target embeddings/head it borrows."""

    def __init__(self, config: SparkDSparkConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        n_targets = len(config.target_layer_ids)
        self.fc = nn.Linear(hidden * n_targets, hidden, bias=False)
        # Spark's draft checkpoint normalizes each captured target layer
        # before fc (GGUF fc_norm, 5 rows in target_layer_ids order). The
        # block-drafter's project_context applies aux_hidden_norms then fc
        # then hidden_norm — this IS Spark's trained projector.
        self.aux_hidden_norms = [
            nn.RMSNorm(hidden, eps=config.rms_norm_eps) for _ in range(n_targets)
        ]
        self.hidden_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.layers = [SparkDSparkLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.markov_head = VanillaMarkov(config.vocab_size, config.markov_rank)
        self.rope = initialize_rope(
            config.rotary_dim, config.rope_theta, traditional=False
        )
        # Bound methods of the target modules: stored as plain attributes so
        # the target's embeddings and head stay out of this module's tree.
        self._embed_fn = None
        self._lm_head_fn = None

    def bind(self, target_model: Any) -> SparkDSparkDraftModel:
        language_model = getattr(target_model, "language_model", target_model)
        inner = getattr(language_model, "model", language_model)
        embed = getattr(inner, "embedding", None) or getattr(inner, "embed_tokens", None)
        if embed is None:
            raise AttributeError(
                f"Cannot find embedding/embed_tokens in {type(target_model).__name__}"
            )
        head = getattr(language_model, "lm_head", None)
        self._embed_fn = embed.__call__
        self._lm_head_fn = head.__call__ if head is not None else embed.as_linear
        return self

    def _embed_input_tokens(self, tokens: mx.array) -> mx.array:
        return self._embed_fn(tokens)

    def _logits(self, hidden: mx.array) -> mx.array:
        return self._lm_head_fn(hidden)


def sanitize_spark_dspark_weights(weights: dict) -> dict:
    """Map the Spark draft checkpoint keys onto the model.

    The Spark DSpark checkpoint exposes its weights directly (``layers.*``,
    ``norm.*``, ``markov_head.*``), not nested under
    ``language_model.model.dspark.*``. Keys are otherwise kept as-is so
    ``load_weights`` matches the model tree directly.
    """
    sanitized = {}
    for key, value in weights.items():
        if key.startswith(_WEIGHT_PREFIX):
            key = key[len(_WEIGHT_PREFIX) :]
        if key.startswith("confidence_head."):
            continue
        key = re.sub(r"^layers\.(\d+)\.decoder_layer\.", r"layers.\1.", key)
        if key.startswith("final_norm."):
            key = "norm." + key[len("final_norm.") :]
        if key.endswith(".weight_scale_inv"):
            key = key[: -len("weight_scale_inv")] + "scales"
        sanitized[key] = value
    # Quantized tensors store scales in ``.weight_scale_inv`` and their weight
    # bytes as ``.weight``; the MLX QuantizedLinear expects ``.scales`` and
    # ``.weight`` (uint32). Flip the matching weight to uint32 view in the
    # same pass so we don't re-walk the dict.
    for key in [k for k in sanitized if k.endswith(".scales")]:
        weight_key = key[: -len("scales")] + "weight"
        sanitized[weight_key] = sanitized[weight_key].view(mx.uint32)
    return sanitized


def _promote_weights(node):
    """Recursively promote BF16 drafter weights to FP32.

    The official llama.cpp dflash graph computes in FP32; the oMLX drafter
    runs the whole block in the weight dtype (BF16 for this checkpoint),
    which drifts the draft attention/MLP output by ~0.09 at L0 (BF16) vs
    ~0.003 (FP32) when compared to the official values. Promote every
    drafter weight to FP32 so the block forward stays in FP32 and matches
    the reference graph (MLX promotes bfloat16 activations against float32
    weights, keeping the whole matmul in FP32).
    """
    if isinstance(node, mx.array):
        return node.astype(mx.float32) if node.dtype == mx.bfloat16 else node
    if isinstance(node, dict):
        return {k: _promote_weights(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_promote_weights(v) for v in node]
    return node


def load_spark_dspark_model(path: str | Path) -> SparkDSparkDraftModel:
    path = Path(path)
    config = SparkDSparkConfig.from_dict(
        json.loads((path / "config.json").read_text())
    )
    model = SparkDSparkDraftModel(config)
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    weights = sanitize_spark_dspark_weights(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    model.update(_promote_weights(model.parameters()))
    mx.eval(model.parameters())
    return model


def wrap_spark_dspark(
    model: SparkDSparkDraftModel,
    source_path: str,
    depth: int | None = None,
    fixed: bool = False,
):
    """Wrap a bound Spark DSpark draft model for the batched verify path."""
    from ..speculative.dflash import DFlashDrafter

    drafter = DFlashDrafter(
        model,
        block_size=model.config.block_size,
        source_path=source_path,
        adaptive=not fixed,
    )
    if depth:
        drafter.depth = max(1, min(drafter.depth, int(depth)))
    drafter.query_windows = False
    drafter.context_copy = True
    # The VLM engine's prefill path returns ``[captures..., last]`` = n_cap+1
    # layers in ``hidden_states``, but the drafter's ``project_context`` fc is
    # ``Linear(2560*n_targets, 2560)`` and needs exactly ``len(target_layer_ids)``
    # layers. Strip the trailing ``last`` layer here (instance-scoped) instead of
    # editing the scheduler's ``_dflash_seed_prefill``. The helper-prefill path
    # already passes exactly n_cap layers, so the guard below is a no-op there.
    _orig_seed = drafter.seed_request
    _n_cap = len(drafter.target_layer_ids)

    def _spark_seed(request_id, captured, position=None):
        if len(captured) == _n_cap + 1:
            captured = captured[:-1]
        return _orig_seed(request_id, captured, position)

    drafter.seed_request = _spark_seed
    return drafter


def load_spark_dspark_drafter(
    path: str | Path,
    target_model: Any,
    depth: int | None = None,
    fixed: bool = False,
):
    """Load the bundled Spark draft and wrap it for the batched verify path."""
    # The target (mlx_lm Spark2_5) must carry ``capture_layer_ids`` so the
    # batch generator can feed the drafter's target-layer hidden states.
    _patch_spark2_5_capture()
    _patch_gate_output_sigmoid()
    model = load_spark_dspark_model(path).bind(target_model)
    # The drafter is loaded on the mlx-executor thread while decode runs on
    # the mlx-engine thread. MLX matmul auto-synchronizes its operands across
    # per-thread streams, so the packed weights (and the intermediate
    # activations they produce) are read correctly by the engine thread with
    # no explicit materialization: MLX resolves the cross-stream dependency
    # lazily at eval time.
    drafter = wrap_spark_dspark(model, str(path), depth, fixed)
    logger.info(
        "Spark DSpark drafter loaded: path=%s block=%d depth=%d adaptive=%s "
        "target_layers=%s",
        path,
        drafter.block_size,
        drafter.depth,
        drafter.adaptive,
        drafter.target_layer_ids,
    )
    return drafter


def _patch_gate_output_sigmoid() -> bool:
    """Honor Spark's sigmoid per-head attention gate.

    The stock :func:`omlx.speculative.dflash.variants.gate_output` hardcodes
    ``nn.softplus`` for every per-head gate (the Laguna/MiMo contract). Spark
    DFlash (llama.cpp ``dflash.cpp``) uses ``ggml_sigmoid`` instead, so the
    draft attention output diverges unless the Spark adapter is allowed to
    override the activation. This hook monkey-patches ``gate_output`` inside
    the dflash package so a drafter whose attention sets
    ``gate_act="sigmoid"`` (Spark) gets sigmoid while every other model keeps
    the original softplus behavior.

    The patch is idempotent and installed at drafter load time so the
    modification lives entirely inside this adapter module; ``variants.py``
    itself stays at the stock softplus implementation.
    """
    from ..speculative.dflash import variants as _variants

    if getattr(_variants, "_omlx_gate_output_patched", False):
        return True
    _orig_gate_output = _variants.gate_output

    def _gate_output(attn, x, attended):
        gate_proj = getattr(attn, "g_proj", None)
        if gate_proj is None:
            return attended
        if getattr(attn, "gate_act", "softplus") != "sigmoid":
            return _orig_gate_output(attn, x, attended)
        batch, length, _ = attended.shape
        gate = mx.sigmoid(gate_proj(x).astype(mx.float32)).astype(attended.dtype)
        attended = attended.reshape(batch, length, attn.n_heads, -1) * gate[..., None]
        return attended.reshape(batch, length, -1)

    _variants.gate_output = _gate_output
    _variants._omlx_gate_output_patched = True
    logger.info("Spark gate_output sigmoid hook installed")
    return True


def _patch_spark2_5_capture() -> bool:
    """Monkey-patch the mlx_lm Spark2_5 model for drafter layer captures.

    The stock ``Spark2_5Model.__call__`` and ``Model.__call__`` do not accept
    ``capture_layer_ids`` / ``return_hidden`` and return a bare logits array,
    which the batch generator's ``_call_backbone_impl`` cannot consume. The
    patch adds per-layer hidden capture and returns a ``LanguageModelOutput``
    so the block-drafter path sees ``[captures..., last]`` in ``hidden_states``.
    """
    try:
        from mlx_lm.models import spark2_5 as spark_mod
        from mlx_vlm.models.base import LanguageModelOutput
    except Exception as e:  # pragma: no cover
        logger.debug("Spark capture patch unavailable: %s", e)
        return False

    inner = spark_mod.Spark2_5Model
    outer = spark_mod.Model
    _make_mask = getattr(spark_mod, "create_attention_mask", None)

    if getattr(inner, "_omlx_spark_capture_patched", False):
        return True

    _orig_inner = inner.__call__

    def _inner_call(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        capture_layer_ids=None,
        hidden_sink=None,
    ):
        hidden_states = (
            input_embeddings
            if input_embeddings is not None
            else self.embedding(inputs)
        )
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = None
        if self._first_full is not None:
            full_mask = _make_mask(hidden_states, cache[self._first_full])
        sliding_mask = None
        if self._first_sliding is not None:
            sliding_mask = _make_mask(
                hidden_states, cache[self._first_sliding],
                window_size=self.args.sliding_window,
            )
        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for layer_idx, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            mask = sliding_mask if layer.is_sliding else full_mask
            hidden_states = layer(hidden_states, mask, layer_cache)
            if layer_idx in capture_set and hidden_sink is not None:
                # Official set_embeddings_layer_inp(lid) injects at t_layer_inp[lid]
                # (layer INPUT), but GGUF stores 1-based values that index 0-based
                # layers, so GGUF [3,11,19,26,34] -> captures 0-based layer-3 input
                # = layer-2 OUTPUT. We therefore capture the layer OUTPUT here.
                hidden_sink.append(hidden_states)
        return self.norm(hidden_states)

    _orig_outer = outer.__call__

    def _outer_call(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        capture_layer_ids=None,
        return_hidden=False,
        hidden_sink=None,
        **kwargs,
    ):
        if capture_layer_ids is not None or return_hidden:
            requested = sorted(set(capture_layer_ids or []))
            sink = hidden_sink if hidden_sink is not None else []
            out = self.model(
                inputs,
                cache,
                input_embeddings,
                capture_layer_ids=requested,
                hidden_sink=sink,
            )
            if self.args.tie_word_embeddings:
                logits = self.model.embedding.as_linear(out)
            else:
                logits = self.lm_head(out)
            # ``sink`` holds one entry per requested capture, in the same
            # order; ``requested`` is sorted and ``_inner_call`` appends in
            # layer order, so they align directly without a dict lookup.
            return LanguageModelOutput(
                logits=logits,
                hidden_states=list(sink) + [out],
            )
        return _orig_outer(self, inputs, cache, input_embeddings)

    inner.__call__ = _inner_call
    outer.__call__ = _outer_call
    inner._omlx_spark_capture_patched = True

    # Spark is a plain mlx-lm attention model (no GDN/SSM layers, no
    # ``rollback_speculative_cache``, no ``mtp_partial_rollback``). The
    # chain-verify path needs a partial rollback for its offset-trim KV
    # when a draft block is only partially accepted: trim every layer's
    # cache back by ``num_drafts - accepted`` positions. Every Spark cache
    # layer is a stock ``KVCache`` (``is_trimmable`` + ``trim``), so a
    # per-layer trim is sufficient and correct.
    if not hasattr(outer, "mtp_partial_rollback"):
        def _spark_mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
            n = int(num_drafts) - int(accepted)
            if n <= 0:
                return True
            for c in cache:
                if not (hasattr(c, "is_trimmable") and c.is_trimmable()):
                    return False
            for c in cache:
                try:
                    if c.trim(n) != n:
                        return False
                except Exception:
                    return False
            return True

        outer.mtp_partial_rollback = _spark_mtp_partial_rollback
        logger.info("Spark2_5 mtp_partial_rollback hook installed")

    logger.info("Spark2_5 capture patch applied (inner+outer)")
    return True



