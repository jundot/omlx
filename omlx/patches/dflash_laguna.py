# SPDX-License-Identifier: Apache-2.0
"""Laguna target and draft adapters for the pinned dflash-mlx runtime.

dflash-mlx 0.1.10 exposes a target-backend registry, but only ships Qwen and
Gemma4 implementations.  Laguna also needs a draft specialization: Poolside's
official checkpoints use per-head gated attention and normalize every captured
target-hidden slice before the shared projection.

The installer below extends dflash-mlx through those existing extension
points.  It is intentionally process-global and idempotent, matching the
runtime's own backend and hook registries.
"""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from dflash_mlx.cache.snapshot import TargetHiddenChunks
from dflash_mlx.engine.target_ops import TargetCapabilities
from dflash_mlx.model import (
    ContextOnlyDraftKVCache,
    DFlashAttention,
    DFlashDecoderLayer,
    DFlashDraftModel,
    DFlashDraftModelArgs,
    FullContextDraftKVCache,
)
from mlx_lm.models import cache as cache_mod
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope

_BACKEND_PATH = "omlx.patches.dflash_laguna:LagunaTargetOps"


def _normalize_target_hidden(
    norm: nn.RMSNorm,
    target_hidden: mx.array | TargetHiddenChunks,
) -> mx.array | TargetHiddenChunks:
    """Normalize sparse target context without materializing trimmed spans."""
    if isinstance(target_hidden, TargetHiddenChunks):
        return TargetHiddenChunks(
            total_len=int(target_hidden.total_len),
            chunks=tuple(norm(chunk) for chunk in target_hidden.chunks),
            spans=target_hidden.spans,
        )
    return norm(target_hidden)


def _model_type(model: Any) -> str:
    args = getattr(model, "args", None)
    if args is None and hasattr(model, "language_model"):
        args = getattr(model.language_model, "args", None)
    value = getattr(args, "model_type", None)
    if value is not None:
        return str(value).lower()
    value = getattr(model, "model_type", None)
    if value is not None:
        return str(value).lower()
    config = getattr(model, "config", None)
    if isinstance(config, dict):
        return str(config.get("model_type", "")).lower()
    return ""


def _trim_recent_cache(cache_entry: Any, n: int) -> int:
    """Remove the newest ``n`` positions, including from a rotating KV ring."""
    n = max(0, int(n))
    offset = int(getattr(cache_entry, "offset", 0) or 0)
    n = min(offset, n)
    if n <= 0:
        return 0

    # RotatingKVCache.trim() cannot rewind an already wrapped ring.  Put the
    # retained values in temporal order before shortening it, as dflash's
    # Gemma4 backend does for speculative rejection.
    if isinstance(cache_entry, cache_mod.RotatingKVCache):
        keys = getattr(cache_entry, "keys", None)
        values = getattr(cache_entry, "values", None)
        if keys is not None and values is not None:
            keys = cache_entry._temporal_order(keys)
            values = cache_entry._temporal_order(values)
            keep_len = max(0, int(keys.shape[2]) - n)
            cache_entry.keys = keys[..., :keep_len, :]
            cache_entry.values = values[..., :keep_len, :]
            cache_entry._idx = keep_len
        else:
            cache_entry._idx = max(0, int(getattr(cache_entry, "_idx", 0) or 0) - n)
        cache_entry.offset = offset - n
        return n

    if hasattr(cache_entry, "trim"):
        return int(cache_entry.trim(n))
    if hasattr(cache_entry, "offset"):
        cache_entry.offset = offset - n
        return n
    return 0


class LagunaTargetOps:
    """dflash-mlx target contract for Laguna's mixed full/SWA decoder."""

    backend_name = "laguna"

    def model_type(self, target_model: Any) -> str:
        return _model_type(target_model)

    def supports_model(self, target_model: Any) -> bool:
        if self.model_type(target_model) != "laguna":
            return False
        try:
            inner = self.text_model(target_model)
        except AttributeError:
            return False
        return (
            hasattr(inner, "layers")
            and hasattr(inner, "embed_tokens")
            and hasattr(inner, "fa_idx")
        )

    def family(self, target_model: Any) -> str:
        del target_model
        return "laguna_swa"

    def capabilities_for(self, target_model: Any) -> TargetCapabilities:
        del target_model
        return TargetCapabilities(
            supports_dflash=True,
            supports_recurrent_rollback=False,
            supports_kv_trim=True,
            supports_prefix_snapshot=True,
            supports_rotating_cache_snapshot=True,
            supports_shared_kv=False,
            supports_target_hidden_capture=True,
            # dflash's specialized verify linears are numerically gated only
            # for its upstream Qwen/Gemma4 backends.  Stock MLX qmm remains
            # correct for Laguna until an equivalent parity suite exists.
            supports_verify_linear=False,
            supports_full_context_draft_layers=False,
            supports_tree_verify=False,
        )

    def supports_tree_cache(self, cache_entries: list[Any]) -> bool:
        del cache_entries
        return False

    def text_wrapper(self, target_model: Any) -> Any:
        language_model = getattr(target_model, "language_model", None)
        if language_model is not None and hasattr(language_model, "model"):
            return language_model
        if hasattr(target_model, "model"):
            return target_model
        raise AttributeError(
            f"Unsupported Laguna model wrapper: {type(target_model)!r}"
        )

    def text_model(self, target_model: Any) -> Any:
        wrapper = self.text_wrapper(target_model)
        inner = getattr(wrapper, "model", None)
        if inner is None:
            raise AttributeError(f"Unsupported Laguna text model: {type(wrapper)!r}")
        return inner

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(
        self, target_model: Any, hidden_states: mx.array
    ) -> mx.array:
        wrapper = self.text_wrapper(target_model)
        if bool(getattr(wrapper.args, "tie_word_embeddings", False)):
            return wrapper.model.embed_tokens.as_linear(hidden_states)
        return wrapper.lm_head(hidden_states)

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool,
        quantize_kv_cache: bool = False,
        target_fa_window: int | None = None,
    ) -> list[Any]:
        del enable_speculative_linear_cache
        if quantize_kv_cache:
            raise ValueError("Laguna target KV quantization is not supported yet")
        if target_fa_window is not None and int(target_fa_window) > 0:
            raise ValueError("Laguna uses its config-defined SWA/full attention cache")
        wrapper = self.text_wrapper(target_model)
        make_cache = getattr(wrapper, "make_cache", None)
        if make_cache is None:
            make_cache = getattr(target_model, "make_cache", None)
        if make_cache is None:
            raise AttributeError("Laguna target must expose make_cache()")
        return make_cache()

    def install_speculative_hooks(self, target_model: Any) -> None:
        # Laguna has no recurrent state.  Stock MLX attention plus explicit KV
        # trimming is sufficient for block verification and rejection.
        self.text_model(target_model)._dflash_speculative_hooks_installed = True

    def forward_with_hidden_capture(
        self,
        target_model: Any,
        *,
        input_ids: mx.array | None = None,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        capture_layer_ids: set[int] | None = None,
        logits_last_only: bool = False,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        inner = self.text_model(target_model)
        h = (
            input_embeddings
            if input_embeddings is not None
            else inner.embed_tokens(input_ids)
        )
        if cache is None:
            cache = [None] * len(inner.layers)
        elif len(cache) != len(inner.layers):
            raise ValueError(
                f"Laguna cache length {len(cache)} != layer count {len(inner.layers)}"
            )

        capture_all = capture_layer_ids is None
        if capture_all:
            captured: list[mx.array] | dict[int, mx.array] = [h]
        else:
            capture_layer_ids = set(capture_layer_ids)
            captured = {0: h} if 0 in capture_layer_ids else {}

        full_mask = create_attention_mask(h, cache[inner.fa_idx])
        sliding_mask = None
        if inner.swa_idx is not None:
            sliding_mask = create_attention_mask(
                h,
                cache[inner.swa_idx],
                window_size=inner.args.sliding_window,
            )

        for layer_idx, (layer, layer_cache) in enumerate(
            zip(inner.layers, cache, strict=True)
        ):
            mask = (
                sliding_mask
                if layer.attention_type == "sliding_attention"
                else full_mask
            )
            h = layer(h, mask, layer_cache)
            capture_key = layer_idx + 1
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h

        normalized = inner.norm(h)
        if logits_last_only and isinstance(captured, dict):
            captured[-1] = normalized
        logits_hidden = normalized[:, -1:, :] if logits_last_only else normalized
        return self.logits_from_hidden(target_model, logits_hidden), captured

    def verify_block(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        capture_layer_ids: set[int] | None = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        if int(verify_ids.shape[1]) <= 0:
            raise ValueError("verify block must contain at least one token")
        return self.forward_with_hidden_capture(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )

    def verify_tree_block(
        self,
        *,
        target_model: Any,
        tree_inputs: Any,
        target_cache: list[Any],
        capture_layer_ids: set[int] | None = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        del target_model, tree_inputs, target_cache, capture_layer_ids
        raise NotImplementedError("Laguna DDTree verification is not implemented")

    def restore_after_tree_acceptance(
        self, cache_entries: list[Any], *, accepted_tree_indices: list[int]
    ) -> int:
        del cache_entries, accepted_tree_indices
        raise NotImplementedError("Laguna DDTree cache commit is not implemented")

    def extract_context_feature(
        self,
        captured_dict: dict[int, mx.array] | list[mx.array],
        target_layer_ids: list[int],
    ) -> mx.array:
        return mx.concatenate(
            [captured_dict[int(layer_id) + 1] for layer_id in target_layer_ids],
            axis=-1,
        )

    def arm_rollback(self, cache_entries: list[Any], *, prefix_len: int) -> None:
        del cache_entries, prefix_len

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int:
        del acceptance_length, drafted_tokens
        started = time.perf_counter_ns()
        trimmed = False
        for cache_entry in cache_entries:
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > int(target_len):
                _trim_recent_cache(cache_entry, offset - int(target_len))
                trimmed = True
        return time.perf_counter_ns() - started if trimmed else 0

    def cleanup_generation_caches(
        self, target_cache: list[Any], draft_cache: list[Any]
    ) -> None:
        draft_cache.clear()
        target_cache.clear()


class LagunaDFlashDraftModelArgs(DFlashDraftModelArgs):
    """Normalize Poolside's Laguna DFlash config for dflash-mlx."""

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> LagunaDFlashDraftModelArgs:
        data = dict(params)
        dflash_config = dict(data.get("dflash_config") or {})
        data.setdefault("block_size", dflash_config.get("block_size"))
        data.setdefault("num_target_layers", dflash_config.get("num_target_layers"))
        data.setdefault("tie_word_embeddings", True)
        base = DFlashDraftModelArgs.from_dict(data)
        obj = cls(
            **{
                name: getattr(base, name)
                for name in DFlashDraftModelArgs.__dataclass_fields__
            }
        )
        obj.gating = data.get("gating", "per-head")
        obj.partial_rotary_factor = data.get("partial_rotary_factor")
        obj.rope_parameters = data.get("rope_parameters")
        obj.architectures = tuple(data.get("architectures") or ())
        obj.draft_vocab_size = int(data.get("draft_vocab_size", obj.vocab_size))

        if not obj.layer_types:
            raise ValueError("Laguna DFlash draft requires layer_types")
        if len(set(obj.layer_types)) != 1:
            raise ValueError("Laguna DFlash draft layers must use one attention type")
        gating = str(obj.gating).replace("_", "-").lower()
        if gating not in ("per-head", "per-element"):
            raise ValueError(f"Unsupported Laguna DFlash gating mode: {obj.gating!r}")
        obj.gating = gating
        return obj


class LagunaDFlashAttention(DFlashAttention):
    """DFlash attention with Laguna RoPE and softplus output gating."""

    def __init__(self, args: LagunaDFlashDraftModelArgs, layer_idx: int):
        super().__init__(args, layer_idx)
        self.gate_per_head = args.gating == "per-head"
        gate_dim = self.n_heads if self.gate_per_head else self.n_heads * self.head_dim
        self.g_proj = nn.Linear(args.hidden_size, gate_dim, bias=False)

        rope_config = dict(args.rope_parameters or args.rope_scaling or {})
        rope_config.setdefault("rope_type", "default")
        partial = float(args.partial_rotary_factor or 1.0)
        rope_config.setdefault("partial_rotary_factor", partial)
        self.rope = initialize_rope(
            int(self.head_dim * partial),
            base=float(rope_config.get("rope_theta", args.rope_theta)),
            traditional=False,
            scaling_config=rope_config,
            max_position_embeddings=args.max_position_embeddings,
        )

    def _apply_gate(self, x: mx.array, output: mx.array) -> mx.array:
        batch, block_len, _ = output.shape
        gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
        if self.gate_per_head:
            shape = output.shape
            output = (
                output.reshape(batch, block_len, self.n_heads, self.head_dim)
                * gate[..., None]
            ).reshape(shape)
        else:
            output = output * gate
        return output

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Any | None = None,
    ) -> mx.array:
        batch, block_len, _ = hidden_states.shape
        ctx_len = int(target_hidden.shape[1])

        queries = self.q_proj(hidden_states)
        queries = self.q_norm(
            queries.reshape(batch, block_len, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)

        context_hidden, context_spans = self._context_segments_for_cache(
            target_hidden, cache
        )
        selected_ctx_len = int(context_hidden.shape[1])
        context_keys = self.k_proj(context_hidden)
        context_keys = self.k_norm(
            context_keys.reshape(batch, selected_ctx_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        context_values = (
            self.v_proj(context_hidden)
            .reshape(batch, selected_ctx_len, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        noise_keys = self.k_proj(hidden_states)
        noise_keys = self.k_norm(
            noise_keys.reshape(batch, block_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        noise_values = (
            self.v_proj(hidden_states)
            .reshape(batch, block_len, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        if isinstance(cache, FullContextDraftKVCache):
            cache_offset = int(cache.offset)
            query_offset = cache_offset + ctx_len
            queries = self.rope(queries, offset=query_offset)
            context_keys, context_values, context_positions = (
                self._rope_context_segments(
                    context_keys,
                    context_values,
                    cache_offset=cache_offset,
                    spans=context_spans,
                )
            )
            noise_keys = self.rope(noise_keys, offset=query_offset)
            cache.append_context(
                context_keys,
                context_values,
                ctx_len,
                positions=context_positions,
                advance_positions=ctx_len,
            )
            keys, values = cache.fetch_with_block(noise_keys, noise_values)
            mask = None
            if self.sliding_window is not None:
                noise_positions = mx.arange(
                    query_offset, query_offset + block_len, dtype=mx.int32
                )
                cached_positions = cache.position_indices()
                key_positions = (
                    noise_positions
                    if cached_positions is None
                    else mx.concatenate([cached_positions, noise_positions], axis=0)
                )
                mask = self._attention_mask(
                    block_len=block_len,
                    query_offset=query_offset,
                    key_len=int(keys.shape[-2]),
                    key_positions=key_positions,
                )
            output = scaled_dot_product_attention(
                queries, keys, values, cache=None, scale=self.scale, mask=mask
            )
        elif isinstance(cache, ContextOnlyDraftKVCache):
            cache_offset = int(cache.offset)
            query_offset = cache_offset + ctx_len
            queries = self.rope(queries, offset=query_offset)
            context_keys, context_values, context_positions = (
                self._rope_context_segments(
                    context_keys,
                    context_values,
                    cache_offset=cache_offset,
                    spans=context_spans,
                )
            )
            noise_keys = self.rope(noise_keys, offset=query_offset)
            cache.append_context(
                context_keys,
                context_values,
                ctx_len,
                positions=context_positions,
                advance_positions=ctx_len,
            )
            cached_keys, cached_values = cache.fetch()
            keys = mx.concatenate([cached_keys, noise_keys], axis=-2)
            values = mx.concatenate([cached_values, noise_values], axis=-2)
            cached_positions = cache.position_indices()
            noise_positions = mx.arange(
                query_offset, query_offset + block_len, dtype=mx.int32
            )
            key_positions = mx.concatenate([cached_positions, noise_positions], axis=0)
            mask = self._attention_mask(
                block_len=block_len,
                query_offset=query_offset,
                key_len=int(keys.shape[-2]),
                key_positions=key_positions,
            )
            output = scaled_dot_product_attention(
                queries, keys, values, cache=None, scale=self.scale, mask=mask
            )
        elif cache is not None:
            cache_offset = int(getattr(cache, "offset", 0) or 0)
            query_offset = cache_offset + ctx_len
            queries = self.rope(queries, offset=query_offset)
            context_keys = self.rope(context_keys, offset=cache_offset)
            noise_keys = self.rope(noise_keys, offset=query_offset)
            keys = mx.concatenate([context_keys, noise_keys], axis=-2)
            values = mx.concatenate([context_values, noise_values], axis=-2)
            keys, values = cache.update_and_fetch(keys, values)
            mask = self._attention_mask(
                block_len=block_len,
                query_offset=query_offset,
                key_len=int(keys.shape[-2]),
            )
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        else:
            queries = self.rope(queries, offset=ctx_len)
            context_keys = self.rope(context_keys, offset=0)
            noise_keys = self.rope(noise_keys, offset=ctx_len)
            keys = mx.concatenate([context_keys, noise_keys], axis=-2)
            values = mx.concatenate([context_values, noise_values], axis=-2)
            mask = self._attention_mask(
                block_len=block_len,
                query_offset=ctx_len,
                key_len=int(keys.shape[-2]),
            )
            output = scaled_dot_product_attention(
                queries, keys, values, cache=None, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(batch, block_len, -1)
        return self.o_proj(self._apply_gate(hidden_states, output))


class LagunaDFlashDecoderLayer(DFlashDecoderLayer):
    def __init__(self, args: LagunaDFlashDraftModelArgs, layer_idx: int):
        nn.Module.__init__(self)
        self.self_attn = LagunaDFlashAttention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array | TargetHiddenChunks,
        cache: Any | None = None,
    ) -> mx.array:
        residual = hidden_states
        attention_input = self.input_layernorm(hidden_states)
        context_input = _normalize_target_hidden(self.input_layernorm, target_hidden)
        hidden_states = self.self_attn(
            attention_input, target_hidden=context_input, cache=cache
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states

    def advance_projected_context_cache(
        self, *, target_hidden: mx.array | TargetHiddenChunks, cache: Any
    ) -> None:
        self.self_attn.append_projected_context_cache(
            target_hidden=_normalize_target_hidden(self.input_layernorm, target_hidden),
            cache=cache,
        )


class LagunaDFlashDraftModel(DFlashDraftModel):
    """MLX representation of Poolside's DFlashLagunaForCausalLM."""

    def __init__(self, args: LagunaDFlashDraftModelArgs):
        nn.Module.__init__(self)
        self.args = args
        self.model_type = "dflash_laguna"
        self.layers = [
            LagunaDFlashDecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.target_layer_ids = list(
            (args.dflash_config or {}).get("target_layer_ids") or ()
        )
        if not self.target_layer_ids:
            raise ValueError("Laguna DFlash draft requires target_layer_ids")
        self.aux_hidden_norms = [
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            for _ in self.target_layer_ids
        ]
        self.fc = nn.Linear(
            len(self.target_layer_ids) * args.hidden_size,
            args.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.block_size = int(args.block_size)
        self.mask_token_id = int(
            (args.dflash_config or {}).get("mask_token_id", 0) or 0
        )
        self.embed_scale = 1.0

    def bind_target_model(self, target_model: Any, *, target_ops: Any) -> None:
        if target_ops.family(target_model) != "laguna_swa":
            raise ValueError("Laguna DFlash draft requires a Laguna target")
        text_model = target_ops.text_model(target_model)
        target_layers = len(text_model.layers)
        if target_layers != int(self.args.num_target_layers):
            raise ValueError(
                "Laguna target/draft layer mismatch: "
                f"{target_layers} != {self.args.num_target_layers}"
            )
        if max(self.target_layer_ids) >= target_layers:
            raise ValueError("Laguna DFlash target_layer_ids exceed target depth")
        target_hidden = int(getattr(text_model.args, "hidden_size", 0) or 0)
        if target_hidden != int(self.args.hidden_size):
            raise ValueError(
                "Laguna target/draft hidden-size mismatch: "
                f"{target_hidden} != {self.args.hidden_size}"
            )
        self.embed_scale = getattr(text_model, "embed_scale", 1.0)

    def project_target_hidden(self, target_hidden: mx.array) -> mx.array:
        expected = len(self.target_layer_ids) * int(self.args.hidden_size)
        if int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "Laguna DFlash target feature width mismatch: "
                f"{target_hidden.shape[-1]} != {expected}"
            )
        slices = mx.split(target_hidden, len(self.target_layer_ids), axis=-1)
        normalized = [
            norm(hidden_slice)
            for norm, hidden_slice in zip(self.aux_hidden_norms, slices, strict=True)
        ]
        return self.hidden_norm(self.fc(mx.concatenate(normalized, axis=-1)))

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Split Poolside's fused QKV tensors into dflash-mlx projections."""
        weights = dict(weights)
        q_dim = int(self.args.num_attention_heads) * int(self.args.head_dim)
        kv_dim = int(self.args.num_key_value_heads) * int(self.args.head_dim)
        total_dim = q_dim + 2 * kv_dim

        # The official BF16 checkpoints use ``qkv_proj.weight``.  MLX
        # quantized linears additionally carry ``scales`` and ``biases``;
        # those tensors are also partitioned by output channel on axis zero.
        for layer_idx in range(len(self.layers)):
            fused_prefix = f"layers.{layer_idx}.self_attn.qkv_proj."
            for suffix in ("weight", "bias", "scales", "biases"):
                fused_key = fused_prefix + suffix
                fused = weights.pop(fused_key, None)
                if fused is None:
                    continue
                if int(fused.shape[0]) != total_dim:
                    raise ValueError(
                        f"Unexpected Laguna {fused_key} output width: "
                        f"{fused.shape[0]} != {total_dim}"
                    )
                query, key, value = mx.split(fused, [q_dim, q_dim + kv_dim], axis=0)
                split_prefix = f"layers.{layer_idx}.self_attn."
                weights[split_prefix + "q_proj." + suffix] = query
                weights[split_prefix + "k_proj." + suffix] = key
                weights[split_prefix + "v_proj." + suffix] = value
        return weights


def _is_laguna_draft_config(config: dict[str, Any]) -> bool:
    architectures = tuple(str(a) for a in (config.get("architectures") or ()))
    return (
        str(config.get("model_type", "")).lower() == "laguna"
        and "DFlashLagunaForCausalLM" in architectures
    )


def install_dflash_laguna_backend() -> bool:
    """Register Laguna target ops and draft classes in dflash-mlx."""
    from dflash_mlx.engine import target_ops
    from dflash_mlx.runtime import loading

    changed = False
    if _BACKEND_PATH not in target_ops.TARGET_BACKENDS:
        target_ops.TARGET_BACKENDS.append(_BACKEND_PATH)
        changed = True

    current = loading._get_dflash_model_classes
    if not getattr(current, "_omlx_laguna_draft", False):

        def get_dflash_model_classes(config: dict[str, Any]):
            resolved = current(config)
            if not _is_laguna_draft_config(config):
                return resolved
            # Respect a future upstream specialization automatically.
            if resolved != (DFlashDraftModel, DFlashDraftModelArgs):
                return resolved
            return LagunaDFlashDraftModel, LagunaDFlashDraftModelArgs

        get_dflash_model_classes._omlx_laguna_draft = True
        get_dflash_model_classes._omlx_original = current
        loading._get_dflash_model_classes = get_dflash_model_classes
        changed = True
    return changed


__all__ = [
    "LagunaDFlashDraftModel",
    "LagunaDFlashDraftModelArgs",
    "LagunaTargetOps",
    "install_dflash_laguna_backend",
]
