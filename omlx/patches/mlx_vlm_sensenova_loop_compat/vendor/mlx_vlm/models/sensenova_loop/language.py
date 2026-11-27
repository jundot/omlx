"""Shared-weight looped Qwen3.5 decoder with manifold-constrained routing.

Implements the reference computation from sensenova/SenseNova-Loop-9B's
modeling_sensenova_loop.py (Apache-2.0).
"""

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5.language import LanguageModel as Qwen3_5LanguageModel
from ..qwen3_5.language import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5Model,
)
from .config import execution_plan


@partial(mx.compile, shapeless=True)
def sinkhorn(logits, iterations):
    logits = logits.astype(mx.float32)
    for _ in range(iterations):
        logits = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits - mx.logsumexp(logits, axis=-2, keepdims=True)
    return mx.exp(logits)


class MHCRouter(nn.Module):
    def __init__(self, dim, settings):
        super().__init__()
        self.streams = settings["streams"]
        self.iterations = settings["iterations"]
        self.fixed_identity = settings["fixed_identity"]
        self.pre_scale = settings.get("pre_scale", 1.0)
        n = self.streams
        size = 2 * n + (0 if self.fixed_identity else n * n)
        self.weight = mx.zeros((size, n * dim), mx.float32)
        probabilities = mx.array({2: (0.25, 0.75), 4: (0.0625, 0.1875, 0.25, 0.5)}[n])
        bias = [mx.log(probabilities / (1 - probabilities)), mx.zeros(n)]
        if not self.fixed_identity:
            diagonal = math.log((1 - (n - 1) / 16) * 16)
            bias.append((mx.eye(n) * diagonal).reshape(-1))
        self.bias = mx.concatenate(bias)

    def __call__(self, residual):
        flat = residual.reshape(*residual.shape[:-2], -1).astype(mx.float32)
        normalized = mx.fast.rms_norm(flat, None, 1e-6)
        logits = normalized @ self.weight.T + self.bias
        n = self.streams
        pre = mx.sigmoid(logits[..., :n])
        post = 2 * mx.sigmoid(logits[..., n : 2 * n])
        if self.fixed_identity:
            mix = mx.broadcast_to(mx.eye(n), (*flat.shape[:-1], n, n))
        else:
            mix = sinkhorn(
                logits[..., 2 * n :].reshape(*flat.shape[:-1], n, n), self.iterations
            )
        branch = (pre.astype(residual.dtype)[..., None, :] @ residual).squeeze(-2)
        return branch * self.pre_scale, post, mix

    @staticmethod
    def merge(residual, branch, post, mix):
        mixed = mix.astype(residual.dtype) @ residual
        return mixed + post.astype(branch.dtype)[..., None] * branch[..., None, :]


class DecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, args, physical_index, settings):
        nn.Module.__init__(self)
        self.is_linear = args.layer_types[physical_index] == "linear_attention"
        if self.is_linear:
            self.linear_attn = Qwen3_5GatedDeltaNet(args)
        else:
            self.self_attn = Qwen3_5Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = Qwen3_5MLP(args.hidden_size, args.intermediate_size)
        self._mhc_settings = settings
        self._physical_index = physical_index
        if settings is not None:
            self.mhc_attn = MHCRouter(args.hidden_size, settings)
            self.mhc_ffn = MHCRouter(args.hidden_size, settings)

    def __call__(
        self, x, mask=None, cache=None, position_ids=None, position_embeddings=None
    ):
        settings = self._mhc_settings
        if settings is None:
            return super().__call__(x, mask, cache, position_ids, position_embeddings)
        n = settings["streams"]
        if self._physical_index == settings["start"]:
            residual = mx.broadcast_to(x[..., None, :], (*x.shape[:-1], n, x.shape[-1]))
        else:
            residual = x.reshape(*x.shape[:-1], n, -1)
        branch, post, mix = self.mhc_attn(residual)
        branch = self.input_layernorm(branch)
        if self.is_linear:
            branch = self.linear_attn(branch, mask, cache)
        else:
            branch = self.self_attn(
                branch, mask, cache, position_ids, position_embeddings
            )
        residual = MHCRouter.merge(residual, branch, post, mix)
        branch, post, mix = self.mhc_ffn(residual)
        branch = self.mlp(self.post_attention_layernorm(branch))
        residual = MHCRouter.merge(residual, branch, post, mix)
        if self._physical_index == settings["end"] - 1:
            return residual.mean(axis=-2)
        return residual.reshape(*residual.shape[:-2], -1)


class TextModel(Qwen3_5Model):
    def __init__(self, args, config):
        nn.Module.__init__(self)
        self.args = args
        self._execution_plan = execution_plan(config)
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(
                args,
                i,
                (
                    config.mhc
                    if config.mhc is not None
                    and config.mhc["start"] <= i < config.mhc["end"]
                    else None
                ),
            )
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = next(
            i for i, layer in enumerate(self.execution_layers) if not layer.is_linear
        )
        self.ssm_idx = next(
            (i for i, layer in enumerate(self.execution_layers) if layer.is_linear), 0
        )

    @property
    def execution_layers(self):
        # References only: parameters remain registered once under physical layers.
        return [self.layers[i] for i in self._execution_plan]

    def __call__(self, inputs, *args, cache=None, **kwargs):
        if cache is not None and len(cache) != len(self._execution_plan):
            raise ValueError(
                f"SenseNova Loop requires {len(self._execution_plan)} cache entries, one per execution visit"
            )
        return super().__call__(inputs, *args, cache=cache, **kwargs)


class LanguageModel(Qwen3_5LanguageModel):
    def __init__(self, args, config):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = TextModel(args, config)
        self._position_ids = None
        self._rope_deltas = None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, *args, **kwargs):
        # The fused Qwen verifier bypasses decoder residual routing.
        if kwargs.get("speculative_verify", False):
            raise ValueError(
                "SenseNova Loop does not support the Qwen3.5 speculative verifier"
            )
        return super().__call__(*args, **kwargs)

    @property
    def layers(self):
        return self.model.execution_layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def quant_predicate(self):
        return lambda path, _: not any(
            part in path.split(".") for part in ("mhc_attn", "mhc_ffn")
        )

    @property
    def cast_predicate(self):
        base = super().cast_predicate
        return lambda path: base(path) and not any(
            part in path.split(".") for part in ("mhc_attn", "mhc_ffn")
        )
