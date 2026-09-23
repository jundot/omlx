# SPDX-License-Identifier: Apache-2.0
"""MiMo V2 target and draft adapters for the pinned dflash-mlx runtime."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from dflash_mlx.draft_backend import (
    EagerDraftBackend,
    _astype_if_needed,
    _draft_compute_dtype,
)
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

_BACKEND_PATH = "omlx.patches.dflash_mimo_v2:MiMoV2TargetOps"
_MIMO_MODEL_TYPES = frozenset(("mimo_v2", "mimo_v2_flash"))


def _model_type(model: Any) -> str:
    args = getattr(model, "args", None)
    if args is None and hasattr(model, "language_model"):
        args = getattr(model.language_model, "args", None)
    value = getattr(args, "model_type", None)
    if value is None:
        value = getattr(model, "model_type", None)
    if value is not None:
        return str(value).lower()
    config = getattr(model, "config", None)
    if isinstance(config, dict):
        return str(config.get("model_type", "")).lower()
    return ""


def _trim_recent_cache(cache_entry: Any, n: int) -> int:
    n = max(0, int(n))
    offset = int(getattr(cache_entry, "offset", 0) or 0)
    n = min(offset, n)
    if n <= 0:
        return 0

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
            cache_entry._idx = max(0, int(getattr(cache_entry, "_idx", 0)) - n)
        cache_entry.offset = offset - n
        return n

    if hasattr(cache_entry, "trim"):
        return int(cache_entry.trim(n))
    cache_entry.offset = offset - n
    return n


class MiMoV2TargetOps:
    """dflash-mlx target contract for MiMo's mixed full/SWA decoder."""

    backend_name = "mimo_v2"

    def model_type(self, target_model: Any) -> str:
        return _model_type(target_model)

    def supports_model(self, target_model: Any) -> bool:
        if self.model_type(target_model) not in _MIMO_MODEL_TYPES:
            return False
        try:
            inner = self.text_model(target_model)
        except AttributeError:
            return False
        return hasattr(inner, "layers") and hasattr(inner, "embed_tokens")

    def family(self, target_model: Any) -> str:
        del target_model
        return "mimo_v2_swa"

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
            f"Unsupported MiMo V2 model wrapper: {type(target_model)!r}"
        )

    def text_model(self, target_model: Any) -> Any:
        wrapper = self.text_wrapper(target_model)
        inner = getattr(wrapper, "model", None)
        if inner is None:
            raise AttributeError(f"Unsupported MiMo V2 text model: {type(wrapper)!r}")
        return inner

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(
        self, target_model: Any, hidden_states: mx.array
    ) -> mx.array:
        wrapper = self.text_wrapper(target_model)
        args = getattr(wrapper, "args", None)
        if bool(getattr(args, "tie_word_embeddings", False)):
            return self.text_model(target_model).embed_tokens.as_linear(hidden_states)
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
            raise ValueError("MiMo V2 target KV quantization is not supported yet")
        if target_fa_window is not None and int(target_fa_window) > 0:
            raise ValueError("MiMo V2 uses its config-defined SWA/full attention cache")
        wrapper = self.text_wrapper(target_model)
        make_cache = getattr(wrapper, "make_cache", None)
        if make_cache is None:
            make_cache = getattr(target_model, "make_cache", None)
        if make_cache is None:
            raise AttributeError("MiMo V2 target must expose make_cache()")
        return make_cache()

    def install_speculative_hooks(self, target_model: Any) -> None:
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
                f"MiMo V2 cache length {len(cache)} != layer count {len(inner.layers)}"
            )

        capture_all = capture_layer_ids is None
        if capture_all:
            captured: list[mx.array] | dict[int, mx.array] = [h]
        else:
            capture_layer_ids = set(capture_layer_ids)
            captured = {0: h} if 0 in capture_layer_ids else {}

        full_index = getattr(inner, "ga_idx", None)
        sliding_index = getattr(inner, "swa_idx", None)
        if full_index is None:
            full_index = next(
                index
                for index, layer in enumerate(inner.layers)
                if not layer.is_sliding_window
            )
        if sliding_index is None:
            sliding_index = next(
                index
                for index, layer in enumerate(inner.layers)
                if layer.is_sliding_window
            )
        full_mask = create_attention_mask(h, cache[full_index])
        sliding_mask = create_attention_mask(
            h,
            cache[sliding_index],
            window_size=inner.sliding_window_size,
        )

        for layer_index, (layer, layer_cache) in enumerate(
            zip(inner.layers, cache, strict=True)
        ):
            mask = sliding_mask if layer.is_sliding_window else full_mask
            h = layer(h, mask, cache=layer_cache)
            capture_key = layer_index + 1
            if capture_all:
                captured.append(h)
            elif capture_key in capture_layer_ids:
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
        raise NotImplementedError("MiMo V2 DDTree verification is not implemented")

    def restore_after_tree_acceptance(
        self, cache_entries: list[Any], *, accepted_tree_indices: list[int]
    ) -> int:
        del cache_entries, accepted_tree_indices
        raise NotImplementedError("MiMo V2 DDTree cache commit is not implemented")

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


class MiMoDFlashDraftModelArgs(DFlashDraftModelArgs):
    """Preserve MiMo-specific attention fields discarded by dflash-mlx."""

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> MiMoDFlashDraftModelArgs:
        data = dict(params)
        base = DFlashDraftModelArgs.from_dict(data)
        obj = cls(
            **{
                name: getattr(base, name)
                for name in DFlashDraftModelArgs.__dataclass_fields__
            }
        )
        draft = dict(data.get("dflash_config") or {})
        obj.partial_rotary_factor = float(data.get("partial_rotary_factor", 1.0))
        obj.attention_value_scale = draft.get(
            "attention_value_scale", data.get("attention_value_scale")
        )
        obj.attention_sink_bias = bool(
            draft.get("attention_sink_bias", data.get("attention_sink_bias", False))
        )
        obj.v_head_dim = int(data.get("v_head_dim", obj.head_dim))
        return obj


class MiMoDFlashAttention(DFlashAttention):
    """DFlash attention with MiMo partial RoPE, value scaling, and sinks."""

    def __init__(self, args: MiMoDFlashDraftModelArgs, layer_idx: int):
        super().__init__(args, layer_idx)
        self.v_head_dim = int(args.v_head_dim)
        self.v_scale = args.attention_value_scale
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.v_head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.v_head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.attention_sink_bias = (
            mx.zeros((self.n_heads,)) if args.attention_sink_bias else None
        )
        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def _scale_values(self, values: mx.array) -> mx.array:
        if self.v_scale is None:
            return values
        return values * float(self.v_scale)

    def append_projected_context_cache(
        self,
        *,
        target_hidden: mx.array,
        cache: Any,
    ) -> None:
        if not isinstance(cache, ContextOnlyDraftKVCache):
            raise TypeError("draft context advance requires a DFlash draft KV cache")
        context_length = int(target_hidden.shape[1])
        if context_length <= 0:
            return
        context_hidden, spans = self._context_segments_for_cache(target_hidden, cache)
        selected_length = int(context_hidden.shape[1])
        keys = self.k_norm(
            self.k_proj(context_hidden).reshape(
                target_hidden.shape[0], selected_length, self.n_kv_heads, -1
            )
        ).transpose(0, 2, 1, 3)
        values = self._scale_values(
            self.v_proj(context_hidden)
            .reshape(
                target_hidden.shape[0],
                selected_length,
                self.n_kv_heads,
                self.v_head_dim,
            )
            .transpose(0, 2, 1, 3)
        )
        keys, values, positions = self._rope_context_segments(
            keys,
            values,
            cache_offset=int(cache.offset),
            spans=spans,
        )
        cache.append_context(
            keys,
            values,
            context_length,
            positions=positions,
            advance_positions=context_length,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Any | None = None,
    ) -> mx.array:
        batch, block_length, _ = hidden_states.shape
        context_length = int(target_hidden.shape[1])
        queries = self.q_norm(
            self.q_proj(hidden_states).reshape(
                batch, block_length, self.n_heads, self.head_dim
            )
        ).transpose(0, 2, 1, 3)

        context_hidden, spans = self._context_segments_for_cache(target_hidden, cache)
        selected_length = int(context_hidden.shape[1])
        context_keys = self.k_norm(
            self.k_proj(context_hidden).reshape(
                batch, selected_length, self.n_kv_heads, self.head_dim
            )
        ).transpose(0, 2, 1, 3)
        context_values = self._scale_values(
            self.v_proj(context_hidden)
            .reshape(
                batch,
                selected_length,
                self.n_kv_heads,
                self.v_head_dim,
            )
            .transpose(0, 2, 1, 3)
        )
        noise_keys = self.k_norm(
            self.k_proj(hidden_states).reshape(
                batch, block_length, self.n_kv_heads, self.head_dim
            )
        ).transpose(0, 2, 1, 3)
        noise_values = self._scale_values(
            self.v_proj(hidden_states)
            .reshape(
                batch,
                block_length,
                self.n_kv_heads,
                self.v_head_dim,
            )
            .transpose(0, 2, 1, 3)
        )

        if isinstance(cache, FullContextDraftKVCache):
            attention_cache = None
            cache_offset = int(cache.offset)
            query_offset = cache_offset + context_length
            queries = self.rope(queries, offset=query_offset)
            context_keys, context_values, positions = self._rope_context_segments(
                context_keys,
                context_values,
                cache_offset=cache_offset,
                spans=spans,
            )
            noise_keys = self.rope(noise_keys, offset=query_offset)
            cache.append_context(
                context_keys,
                context_values,
                context_length,
                positions=positions,
                advance_positions=context_length,
            )
            keys, values = cache.fetch_with_block(noise_keys, noise_values)
            cached_positions = cache.position_indices()
            noise_positions = mx.arange(
                query_offset,
                query_offset + block_length,
                dtype=mx.int32,
            )
            key_positions = (
                noise_positions
                if cached_positions is None
                else mx.concatenate([cached_positions, noise_positions], axis=0)
            )
            mask = self._attention_mask(
                block_len=block_length,
                query_offset=query_offset,
                key_len=int(keys.shape[-2]),
                key_positions=key_positions,
            )
        elif isinstance(cache, ContextOnlyDraftKVCache):
            attention_cache = None
            cache_offset = int(cache.offset)
            query_offset = cache_offset + context_length
            queries = self.rope(queries, offset=query_offset)
            context_keys, context_values, positions = self._rope_context_segments(
                context_keys,
                context_values,
                cache_offset=cache_offset,
                spans=spans,
            )
            noise_keys = self.rope(noise_keys, offset=query_offset)
            cache.append_context(
                context_keys,
                context_values,
                context_length,
                positions=positions,
                advance_positions=context_length,
            )
            cached_keys, cached_values = cache.fetch()
            keys = mx.concatenate([cached_keys, noise_keys], axis=-2)
            values = mx.concatenate([cached_values, noise_values], axis=-2)
            noise_positions = mx.arange(
                query_offset,
                query_offset + block_length,
                dtype=mx.int32,
            )
            key_positions = mx.concatenate(
                [cache.position_indices(), noise_positions], axis=0
            )
            mask = self._attention_mask(
                block_len=block_length,
                query_offset=query_offset,
                key_len=int(keys.shape[-2]),
                key_positions=key_positions,
            )
        elif cache is not None:
            attention_cache = cache
            cache_offset = int(getattr(cache, "offset", 0) or 0)
            query_offset = cache_offset + context_length
            queries = self.rope(queries, offset=query_offset)
            context_keys = self.rope(context_keys, offset=cache_offset)
            noise_keys = self.rope(noise_keys, offset=query_offset)
            keys = mx.concatenate([context_keys, noise_keys], axis=-2)
            values = mx.concatenate([context_values, noise_values], axis=-2)
            keys, values = cache.update_and_fetch(keys, values)
            mask = self._attention_mask(
                block_len=block_length,
                query_offset=query_offset,
                key_len=int(keys.shape[-2]),
            )
        else:
            attention_cache = None
            query_offset = context_length
            queries = self.rope(queries, offset=query_offset)
            context_keys = self.rope(context_keys, offset=0)
            noise_keys = self.rope(noise_keys, offset=query_offset)
            keys = mx.concatenate([context_keys, noise_keys], axis=-2)
            values = mx.concatenate([context_values, noise_values], axis=-2)
            mask = self._attention_mask(
                block_len=block_length,
                query_offset=query_offset,
                key_len=int(keys.shape[-2]),
            )

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=attention_cache,
            scale=self.scale,
            mask=mask,
            sinks=self.attention_sink_bias,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            batch, block_length, self.n_heads * self.v_head_dim
        )
        return self.o_proj(output)


class MiMoDFlashDecoderLayer(DFlashDecoderLayer):
    def __init__(self, args: MiMoDFlashDraftModelArgs, layer_idx: int):
        nn.Module.__init__(self)
        self.self_attn = MiMoDFlashAttention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )


class MiMoDFlashDraftModel(DFlashDraftModel):
    layer_class = MiMoDFlashDecoderLayer

    def bind_target_model(self, target_model: Any, *, target_ops: Any) -> None:
        if target_ops.family(target_model) != "mimo_v2_swa":
            raise ValueError("MiMo DFlash draft requires a MiMo V2 target")
        text_model = target_ops.text_model(target_model)
        if len(text_model.layers) != int(self.args.num_target_layers):
            raise ValueError(
                "MiMo target/draft layer mismatch: "
                f"{len(text_model.layers)} != {self.args.num_target_layers}"
            )
        if max(self.target_layer_ids) >= len(text_model.layers):
            raise ValueError("MiMo DFlash target_layer_ids exceed target depth")
        target_hidden = int(
            getattr(target_ops.text_wrapper(target_model).args, "hidden_size", 0)
        )
        if target_hidden != int(self.args.hidden_size):
            raise ValueError(
                "MiMo target/draft hidden-size mismatch: "
                f"{target_hidden} != {self.args.hidden_size}"
            )
        super().bind_target_model(target_model, target_ops=target_ops)


class MiMoEagerDraftBackend(EagerDraftBackend):
    """Use MiMo's trained mask vector instead of the target token embedding."""

    def _draft_block_hidden_logits(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
    ) -> tuple[mx.array, mx.array]:
        block_token_ids = mx.concatenate(
            [staged_first[:1], mask_token_tail[: int(block_len) - 1]], axis=0
        )
        noise_embedding = target_ops.embed_tokens(target_model)(block_token_ids[None])
        mask_embedding = getattr(draft_model, "mask_embedding", None)
        if mask_embedding is None:
            raise ValueError("MiMo DFlash draft is missing mask_embedding.pt")
        use_mask = (block_token_ids == int(draft_model.mask_token_id))[None, :, None]
        noise_embedding = mx.where(
            use_mask,
            mask_embedding.astype(noise_embedding.dtype)[None, None, :],
            noise_embedding,
        )
        draft_dtype = _draft_compute_dtype(draft_model)
        if draft_dtype is not None:
            noise_embedding = _astype_if_needed(noise_embedding, draft_dtype)
            draft_context = _astype_if_needed(draft_context, draft_dtype)
        hidden = draft_model.forward_projected_context(
            noise_embedding=noise_embedding,
            draft_context=draft_context,
            cache=draft_cache,
        )[:, 1:, :]
        return hidden, target_ops.logits_from_hidden(target_model, hidden)


def _is_mimo_draft_config(config: dict[str, Any]) -> bool:
    draft = config.get("dflash_config")
    if not isinstance(draft, dict):
        return False
    architectures = {str(value) for value in (config.get("architectures") or ())}
    return (
        "DFlashDraftModel" in architectures
        and "attention_value_scale" in draft
        and bool(draft.get("attention_sink_bias"))
    )


def _load_mask_embedding(path: Path, hidden_size: int) -> mx.array:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if any(name.startswith(("/", "../")) or "/../" in name for name in names):
            raise ValueError(f"Unsafe MiMo mask archive member in {path}")
        byteorder_names = [name for name in names if name.endswith("/byteorder")]
        if byteorder_names and archive.read(byteorder_names[0]).strip() != b"little":
            raise ValueError("MiMo mask embedding uses an unsupported byte order")
        data_names = [name for name in names if name.endswith("/data/0")]
        if len(data_names) != 1:
            raise ValueError("MiMo mask archive must contain exactly one tensor")
        raw = archive.read(data_names[0])
    expected_bytes = int(hidden_size) * 2
    if len(raw) != expected_bytes:
        raise ValueError(
            f"MiMo mask embedding is {len(raw)} bytes, expected {expected_bytes}"
        )
    values = mx.array(np.frombuffer(raw, dtype=np.uint16).copy()).view(mx.bfloat16)
    return values


def prepare_mimo_draft(
    draft_model: Any,
    draft_meta: dict[str, Any],
    draft_model_path: str | Path,
) -> None:
    if not isinstance(draft_model, MiMoDFlashDraftModel):
        return
    from dflash_mlx.runtime.loading import _resolve_local_model_path

    local_path = _resolve_local_model_path(draft_model_path)
    mask_path = local_path / "mask_embedding.pt"
    if not mask_path.is_file():
        raise FileNotFoundError(f"MiMo DFlash mask embedding not found: {mask_path}")
    draft_model.mask_embedding = _load_mask_embedding(
        mask_path, int(draft_model.args.hidden_size)
    )
    draft_meta["mask_embedding"] = str(mask_path.name)


def draft_backend_for(draft_model: Any) -> EagerDraftBackend:
    if isinstance(draft_model, MiMoDFlashDraftModel):
        return MiMoEagerDraftBackend()
    return EagerDraftBackend()


def resolve_bundled_mimo_draft(
    model_path: str | Path,
    configured_draft: str | None,
) -> str | None:
    if configured_draft:
        return configured_draft
    model_path = Path(model_path)
    try:
        target_config = json.loads(
            (model_path / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if str(target_config.get("model_type", "")).lower() not in _MIMO_MODEL_TYPES:
        return None
    draft_path = model_path / "dflash"
    try:
        draft_config = json.loads(
            (draft_path / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if not _is_mimo_draft_config(draft_config):
        return None
    if not (draft_path / "mask_embedding.pt").is_file():
        return None
    if not any(draft_path.glob("*.safetensors")):
        return None
    return str(draft_path)


def install_dflash_mimo_v2_backend() -> bool:
    """Register MiMo target ops and draft classes in dflash-mlx."""
    from dflash_mlx.engine import target_ops
    from dflash_mlx.runtime import loading

    changed = False
    if _BACKEND_PATH not in target_ops.TARGET_BACKENDS:
        target_ops.TARGET_BACKENDS.append(_BACKEND_PATH)
        changed = True

    current = loading._get_dflash_model_classes
    if not getattr(current, "_omlx_mimo_v2_draft", False):

        def get_dflash_model_classes(config: dict[str, Any]):
            resolved = current(config)
            if not _is_mimo_draft_config(config):
                return resolved
            if resolved != (DFlashDraftModel, DFlashDraftModelArgs):
                return resolved
            return MiMoDFlashDraftModel, MiMoDFlashDraftModelArgs

        get_dflash_model_classes._omlx_mimo_v2_draft = True
        get_dflash_model_classes._omlx_original = current
        loading._get_dflash_model_classes = get_dflash_model_classes
        changed = True
    return changed


__all__ = [
    "MiMoDFlashDraftModel",
    "MiMoDFlashDraftModelArgs",
    "MiMoEagerDraftBackend",
    "MiMoV2TargetOps",
    "draft_backend_for",
    "install_dflash_mimo_v2_backend",
    "prepare_mimo_draft",
    "resolve_bundled_mimo_draft",
]
