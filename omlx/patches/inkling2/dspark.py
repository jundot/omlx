# SPDX-License-Identifier: Apache-2.0
"""Bundled DSpark drafter for Inkling 2 (``mtp.safetensors``).

A five-layer block drafter. Each layer mixes the target's context taps (the
partial block sums entering ``target_layer_ids[:-1]`` and the final-norm
output) with its own softmax weights and projects them to context K/V. The
block attends to that context and to itself in both directions, and the mask
positions propose through the target LM head plus an RNN Markov head.

Every layer, including the full-attention one, sees only the newest
``sliding_window_size - 1`` context positions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

DRAFT_FILENAME = "mtp.safetensors"


def bundled_inkling2_dspark_path(model_path: str | Path) -> Path | None:
    """``<model>/mtp.safetensors`` when the checkpoint declares a DSpark draft."""
    path = Path(model_path)
    weights = path / DRAFT_FILENAME
    try:
        config = json.loads((path / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    mtp = config.get("mtp_config") or {}
    if config.get("model_type") != "inkling2" or not mtp or not weights.is_file():
        return None
    if mtp.get("markov_head_type") != "rnn_hidden":
        return None
    return weights


@dataclass
class Inkling2DSparkConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    local_layer_ids: list
    sliding_window: int
    block_size: int
    markov_rank: int
    target_layer_ids: list
    vocab_size: int
    unpadded_vocab_size: int
    final_logit_softcapping: float
    mask_token_id: int = 0
    sample_from_anchor: bool = False

    @classmethod
    def from_target_config(cls, config: dict) -> Inkling2DSparkConfig:
        text = config["text_config"]
        mtp = config["mtp_config"]
        if mtp.get("q_bias") or mtp.get("o_bias"):
            raise ValueError("Inkling 2 DSpark does not support projection biases")
        return cls(
            hidden_size=int(mtp["hidden_size"]),
            num_hidden_layers=int(mtp["num_hidden_layers"]),
            num_attention_heads=int(mtp["num_attention_heads"]),
            num_key_value_heads=int(mtp["num_key_value_heads"]),
            head_dim=int(mtp["head_dim"]),
            intermediate_size=int(mtp["intermediate_size"]),
            rms_norm_eps=float(mtp.get("rms_norm_eps", 1e-6)),
            rope_theta=float(mtp.get("rope_theta", 1e6)),
            local_layer_ids=list(mtp.get("local_layer_ids") or []),
            sliding_window=int(mtp["sliding_window_size"]),
            block_size=int(mtp["block_size"]),
            markov_rank=int(mtp["markov_rank"]),
            target_layer_ids=[int(i) for i in mtp["target_layer_ids"]],
            vocab_size=int(text["vocab_size"]),
            unpadded_vocab_size=int(
                text.get("unpadded_vocab_size") or text["vocab_size"]
            ),
            final_logit_softcapping=float(text.get("final_logit_softcapping") or 0),
        )


def mix_rmsnorm(context: mx.array, alpha: mx.array, weight: mx.array, eps: float):
    """Softmax-weighted sum of the context taps, RMS-normalized.

    ``context`` is ``[B, n, T * H]`` (taps concatenated in feature order).
    """
    b, n, width = context.shape
    taps = alpha.shape[0]
    x = context.reshape(b, n, taps, width // taps).astype(mx.float32)
    mixed = mx.sum(x * alpha.astype(mx.float32)[:, None], axis=2)
    return mx.fast.rms_norm(mixed, weight.astype(mx.float32), eps).astype(context.dtype)


class DraftAttention(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        h = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        dq = self.n_heads * self.head_dim
        dkv = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(h, dq, bias=False)
        self.k_proj = nn.Linear(h, dkv, bias=False)
        self.v_proj = nn.Linear(h, dkv, bias=False)
        self.o_proj = nn.Linear(dq, h, bias=False)
        self.k_proj_ctx = nn.Linear(h, dkv, bias=False)
        self.v_proj_ctx = nn.Linear(h, dkv, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_ctx = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self._eps = config.rms_norm_eps
        # Bound by the draft model after loading: this layer's tap mixture and
        # its dflare_norm weight.
        self._alpha = None
        self._mix_weight = None

    def project_context_kv(self, context: mx.array):
        mixed = mix_rmsnorm(context, self._alpha, self._mix_weight, self._eps)
        return self.k_proj_ctx(mixed), self.v_proj_ctx(mixed)


class DraftMLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DraftLayer(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        h = config.hidden_size
        self.self_attn = DraftAttention(config)
        self.mlp = DraftMLP(h, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.dflare_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)


class Inkling2MarkovHead(nn.Module):
    """RNN Markov head: per-step bias over the vocabulary.

    The bias is ``lm_head(hidden_out(latent))``; ``lm_head @ hidden_out`` is
    folded once at bind time so a step reads a ``[V, rank]`` matrix instead of
    the full LM head.
    """

    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        r, h = config.markov_rank, config.hidden_size
        self.rank = r
        self.markov_w1 = nn.Embedding(config.vocab_size, r)
        self.rnn_proj = nn.Linear(2 * r + 2 * h, 3 * r, bias=True)
        self.hidden_out = nn.Linear(r, h, bias=True)
        self.dflare_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self._eps = config.rms_norm_eps
        self._softcap = config.final_logit_softcapping
        self._vocab_bias = None
        if config.unpadded_vocab_size < config.vocab_size:
            self._vocab_bias = mx.where(
                mx.arange(config.vocab_size) < config.unpadded_vocab_size,
                0.0,
                -mx.inf,
            ).astype(mx.float32)
        self._fold_weight = None
        self._fold_bias = None
        self._alpha = None
        self._f_rnn: dict[Any, mx.array] = {}
        self._hidden = None
        self._rows_f = None
        self._state = None
        self._step = 0

    def fold(self, lm_head: nn.Module) -> None:
        """Fold the target LM head into the latent projection."""
        if isinstance(lm_head, nn.QuantizedLinear):
            weight = mx.dequantize(
                lm_head.weight,
                lm_head.scales,
                lm_head.get("biases"),
                lm_head.group_size,
                lm_head.bits,
                mode=getattr(lm_head, "mode", "affine"),
            )
        else:
            weight = lm_head.weight
        proj = self.hidden_out.weight.astype(mx.float32)
        bias = self.hidden_out.bias.astype(mx.float32)
        rows, biases = [], []
        for start in range(0, weight.shape[0], 16384):
            head = weight[start : start + 16384].astype(mx.float32)
            rows.append((head @ proj).astype(mx.bfloat16))
            biases.append((head @ bias).astype(mx.bfloat16))
            mx.eval(rows[-1], biases[-1])
        self._fold_weight = mx.concatenate(rows, axis=0)
        self._fold_bias = mx.concatenate(biases, axis=0)
        mx.eval(self._fold_weight, self._fold_bias)

    def prepare_block(self, draft_hidden, context, lengths, uids) -> None:
        """Reset the RNN for a new block; refresh each row's committed taps."""
        rows = []
        for index, (uid, n) in enumerate(zip(uids, lengths)):
            if n:
                last = context[index : index + 1, n - 1 : n]
                self._f_rnn[uid] = mix_rmsnorm(
                    last, self._alpha, self.dflare_norm.weight, self._eps
                )[0]
            rows.append(self._f_rnn[uid])
        self._rows_f = mx.concatenate(rows, axis=0)
        self._hidden = draft_hidden
        self._state = mx.zeros((len(uids), self.rank), dtype=draft_hidden.dtype)
        self._step = 0

    def forget(self, uids) -> None:
        for uid in uids:
            self._f_rnn.pop(uid, None)

    def apply_step_logits(self, base_logits: mx.array, prev_tokens: mx.array):
        state = self._state
        dtype = state.dtype
        inputs = mx.concatenate(
            [
                state,
                self.markov_w1(prev_tokens.reshape(-1)).astype(dtype),
                self._hidden[:, self._step].astype(dtype),
                self._rows_f.astype(dtype),
            ],
            axis=-1,
        )
        proj = self.rnn_proj(inputs).astype(mx.float32)
        r = self.rank
        gate = mx.sigmoid(proj[:, :r]).astype(dtype)
        kept = (gate * state).astype(dtype)
        blend = ((1.0 - gate) * mx.tanh(proj[:, r : 2 * r]).astype(dtype)).astype(dtype)
        self._state = kept + blend
        latent = mx.tanh(proj[:, 2 * r :]).astype(dtype)
        bias = latent @ self._fold_weight.T + self._fold_bias
        self._step += 1
        logits = base_logits.astype(mx.float32) + bias.astype(mx.float32)
        if self._softcap:
            logits = mx.tanh(logits / self._softcap) * self._softcap
        if self._vocab_bias is not None:
            logits = logits + self._vocab_bias
        return logits


class Inkling2DSparkModel(nn.Module):
    # The context rows are the raw target taps; each layer projects its own mix.
    raw_context = True

    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        self.config = config
        self.layers = [DraftLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mask_embed = mx.zeros((config.hidden_size,))
        self.fuse_weight = mx.zeros(
            (config.num_hidden_layers + 1, len(config.target_layer_ids))
        )
        self.markov_head = Inkling2MarkovHead(config)
        self.rope = nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)
        self.context_dtype = mx.bfloat16
        self._embed_fn = None
        self._lm_head_fn = None

    def bind(self, target: nn.Module) -> Inkling2DSparkModel:
        """Share the target's normalized embedding and LM head."""
        text = target.model
        self._embed_fn = lambda tokens: text.embed_norm(text.embed(tokens))
        lm_head = target.lm_head
        self._lm_head_fn = lambda hidden: lm_head(hidden)
        alpha = mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1).astype(
            mx.bfloat16
        )
        for index, layer in enumerate(self.layers):
            layer.self_attn._alpha = alpha[index]
            layer.self_attn._mix_weight = layer.dflare_norm.weight
        self.markov_head._alpha = alpha[-1]
        self.markov_head.fold(lm_head)
        # Generation runs on another thread; nothing lazy may outlive the load.
        lazy = [layer.self_attn._alpha for layer in self.layers]
        lazy.append(self.markov_head._alpha)
        if self.markov_head._vocab_bias is not None:
            lazy.append(self.markov_head._vocab_bias)
        mx.eval(lazy)
        return self

    def _embed_input_tokens(self, tokens: mx.array) -> mx.array:
        return self._embed_fn(tokens)

    def _logits(self, hidden: mx.array) -> mx.array:
        return self._lm_head_fn(hidden)


# Draft layer projections run as 8-bit affine: every cycle reads them all, and
# the target verifies every proposal, so only proposal quality is affected.
_DRAFT_QUANT = {"group_size": 64, "bits": 8}


def load_inkling2_dspark_model(model_path: str | Path) -> Inkling2DSparkModel:
    path = Path(model_path)
    config = Inkling2DSparkConfig.from_target_config(
        json.loads((path / "config.json").read_text())
    )
    model = Inkling2DSparkModel(config)
    weights = mx.load(str(path / DRAFT_FILENAME))
    model.load_weights(list(weights.items()), strict=True)
    nn.quantize(
        model,
        **_DRAFT_QUANT,
        class_predicate=lambda p, m: isinstance(m, nn.Linear)
        and p.startswith("layers."),
    )
    mx.eval(model.parameters())
    return model


def wrap_inkling2_dspark(
    model: Inkling2DSparkModel,
    source_path: str,
    depth: int | None = None,
    fixed: bool = False,
):
    """Wrap a bound draft for the batched verify path.

    The trained block is the anchor plus ``block_size - 1`` masks, and only
    the masks propose, so the drafter block equals ``block_size``.
    """
    from ...speculative.dflash import DFlashDrafter

    drafter = DFlashDrafter(
        model,
        block_size=model.config.block_size,
        source_path=source_path,
        adaptive=not fixed,
        mask_embedding=model.mask_embed,
    )
    if depth:
        drafter.depth = max(1, min(drafter.depth, int(depth)))
    return drafter


def load_inkling2_dspark_drafter(
    model_path: str | Path,
    target_model: Any,
    depth: int | None = None,
    fixed: bool = False,
):
    """Load ``mtp.safetensors`` and wrap it for the batched verify path."""
    model = load_inkling2_dspark_model(model_path).bind(target_model)
    drafter = wrap_inkling2_dspark(model, str(model_path), depth, fixed)
    logger.info(
        "Inkling 2 DSpark drafter loaded: path=%s block=%d depth=%d adaptive=%s "
        "target_layers=%s",
        model_path,
        drafter.block_size,
        drafter.depth,
        drafter.adaptive,
        drafter.target_layer_ids,
    )
    return drafter
