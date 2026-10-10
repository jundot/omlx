# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 (``glm5_next``) target adapter for the pinned dflash-mlx runtime.

``incoai/GLM-5.3-Flash-DFlash2`` is a generic :class:`DFlash2DraftModel`
checkpoint that the pinned runtime already loads. What dflash-mlx lacks is
the *target* side: GLM-5.3 is served through mlx-vlm (oMLX's vendored
``glm5_next`` module), keeps ``hc_mult`` hyper-connection residual streams,
and mixes recurrent KDA layers (``ArraysCache``) with DeepSeek sparse
attention layers whose cache is ``CacheList(KVCache, PoolingCache)``.

This module provides:

* :class:`Glm5NextTargetOps` -- the dflash-mlx ``TargetOps`` contract for
  that backbone:

  - hidden-state capture with the MHC streams contracted (mean over the
    stream axis, the same contraction the model applies before its final
    norm) so the drafter sees ``[B, T, hidden_size]`` features;
  - chunked cold prefill inside ``forward_with_hidden_capture`` (the runtime
    only splits prefill for targets that support prefix snapshots);
  - exact rollback of rejected draft tokens: KDA layers replay the accepted
    prefix through the vendored linear-attention forward from a state
    snapshot, DSA layers trim the KV component and undo the pooling
    remainder through the MTP undo log.

* :func:`load_glm5_target_bundle` -- loads the target through mlx-vlm the
  way ``VLMBatchedEngine`` does and returns dflash-mlx's
  ``LoadedTargetBundle``.
* :func:`install_dflash_glm5_backend` / :func:`restore_glm5_dflash_class_patches`
  -- backend registration and the class-hook lifecycle shared with the
  other oMLX DFlash adapters.

The adapter fails closed: prefix snapshots (the dflash L1/L2 cache), DDTree
verification, verify-linear kernels and target KV quantization are refused
rather than silently approximated.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from dflash_mlx.engine.target_ops import TargetCapabilities
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

from ..utils.layer_pipeline import LayerPipeline
from .deepseek_v4.cache_extras import POOLING_UNDO_MAX_TOKENS

logger = logging.getLogger(__name__)

_BACKEND_PATH = "omlx.patches.dflash_glm5:Glm5NextTargetOps"
GLM5_MODEL_TYPES = frozenset(("glm5_next", "glm5_next_text"))
_VERIFY_STATE_ATTR = "_omlx_glm5_verify"
_ORIGINAL_LINEAR_CALLS: dict[type, Any] = {}


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _model_type(model: Any) -> str:
    value = getattr(model, "model_type", None)
    if value is not None:
        return str(value).lower()
    value = _config_value(getattr(model, "config", None), "model_type")
    if value is not None:
        return str(value).lower()
    language_model = getattr(model, "language_model", None)
    args = getattr(language_model, "args", None)
    return str(getattr(args, "model_type", "") or "").lower()


def is_glm5_model_type(model_type: Any) -> bool:
    return str(model_type or "").lower() in GLM5_MODEL_TYPES


def is_glm5_dflash_target(model_path: str | Path | None) -> bool:
    """True when ``model_path/config.json`` declares a GLM-5.3 checkpoint."""
    if not model_path:
        return False
    try:
        config = json.loads(
            (Path(str(model_path)).expanduser() / "config.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError, TypeError):
        return False
    return isinstance(config, dict) and is_glm5_model_type(config.get("model_type"))


def _glm_linear_forward() -> Any:
    """Return the vendored GLM projection helper the model itself uses.

    ``mlx_vlm.models.glm5_next`` only exists once oMLX's compat patch has
    registered the vendored package; the target loader applies it before
    the model is built, so this only registers it for direct callers.
    """
    try:
        from mlx_vlm.models.glm5_next.linear import linear_forward
    except ImportError:
        from .mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch

        apply_mlx_vlm_glm5_next_compat_patch()
        from mlx_vlm.models.glm5_next.linear import linear_forward
    return linear_forward


def _contract_mhc_hidden(hidden: mx.array) -> mx.array:
    """Contract GLM's ``[B, T, hc_mult, H]`` residual streams for DFlash."""
    if hasattr(hidden, "materialize"):
        hidden = hidden.materialize()
    if hidden.ndim == 4:
        return hidden.mean(axis=2)
    if hidden.ndim != 3:
        raise ValueError(f"Unexpected GLM hidden-state rank: {hidden.ndim}")
    return hidden


def validate_glm5_dflash_pair(
    target_model: Any,
    draft_model: Any,
    draft_meta: Any,
) -> None:
    """Reject an unproven GLM target/draft pairing before generation starts."""
    if not is_glm5_model_type(_model_type(target_model)):
        return

    config = draft_meta.get("config", {}) if isinstance(draft_meta, dict) else {}
    architectures = tuple(str(v) for v in (_config_value(config, "architectures") or ()))
    if "DFlash2DraftModel" not in architectures or not bool(
        getattr(draft_model, "is_dflash2", False)
    ):
        raise ValueError("GLM-5.3 DFlash requires a DFlash2DraftModel checkpoint")

    language_model = getattr(target_model, "language_model", None)
    target_args = getattr(language_model, "args", None)
    target_inner = getattr(language_model, "model", None)
    draft_args = getattr(draft_model, "args", None)
    if target_args is None or target_inner is None or draft_args is None:
        raise ValueError("GLM-5.3 DFlash target/draft metadata is incomplete")

    checks = (
        ("hidden_size", int(getattr(target_args, "hidden_size", 0))),
        ("vocab_size", int(getattr(target_args, "vocab_size", 0))),
        ("num_target_layers", len(getattr(target_inner, "layers", ()))),
    )
    for draft_name, target_value in checks:
        draft_value = int(getattr(draft_args, draft_name, 0) or 0)
        if draft_value != target_value:
            raise ValueError(
                f"GLM-5.3 DFlash {draft_name} mismatch: "
                f"draft={draft_value}, target={target_value}"
            )

    target_layer_ids = [int(v) for v in getattr(draft_model, "target_layer_ids", ())]
    if (
        not target_layer_ids
        or target_layer_ids != sorted(set(target_layer_ids))
        or target_layer_ids[0] < 0
        or target_layer_ids[-1] >= len(target_inner.layers)
    ):
        raise ValueError(
            "GLM-5.3 DFlash target_layer_ids must be unique, increasing, and "
            "inside the target layer range"
        )

    if (
        not bool(getattr(target_args, "mhc", False))
        or int(getattr(target_args, "hc_mult", 0) or 0) <= 0
    ):
        raise ValueError("GLM-5.3 DFlash requires the checkpoint's MHC target")


class _Glm5RecurrentRollbackCache(RecurrentRollbackCache):
    """Rollback cache that reports verify windows to the vendored KDA gates.

    The fused GLM KDA prefill only runs on caches that are not speculating.
    """

    @property
    def is_speculating(self) -> bool:
        return bool(self._armed)


def _install_glm5_recurrent_hook(linear_attn: Any) -> None:
    """Retain verify inputs so rejected KDA state can be replayed exactly.

    The target forward itself remains the vendored GLM implementation: the
    wrapper records only the already-computed layer input and delegates all
    arithmetic unchanged, so it cannot drift from the model when the
    projection or recurrence kernels change. dflash-mlx's generic
    innovation-tape replay is not used because GLM's vector-gated recurrence
    is not what that kernel reconstructs.
    """
    cls = type(linear_attn)
    if cls in _ORIGINAL_LINEAR_CALLS:
        return
    original_call = cls.__call__
    _ORIGINAL_LINEAR_CALLS[cls] = original_call

    def speculative_call(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        if not isinstance(cache, RecurrentRollbackCache) or not getattr(
            cache, "_armed", False
        ):
            return original_call(self, inputs, mask=mask, cache=cache)
        output = original_call(self, inputs, mask=mask, cache=cache)
        setattr(cache, _VERIFY_STATE_ATTR, (self, inputs, mask))
        return output

    speculative_call._omlx_dflash_glm5 = True  # type: ignore[attr-defined]
    cls.__call__ = speculative_call


def restore_glm5_dflash_class_patches() -> int:
    """Restore every class-level GLM DFlash hook; returns the count restored."""
    restored = 0
    for cls, original_call in tuple(_ORIGINAL_LINEAR_CALLS.items()):
        cls.__call__ = original_call
        restored += 1
    _ORIGINAL_LINEAR_CALLS.clear()
    return restored


class Glm5NextTargetOps:
    """dflash-mlx target contract for GLM-5.3's KDA/DSA text backbone."""

    backend_name = "glm5_next"
    # Cold prefill is split into chunks of this many tokens inside
    # ``forward_with_hidden_capture``. DFlashEngine aligns it with the
    # runtime's ``prefill_step_size`` after the target is loaded.
    prefill_chunk_size = 2048
    # Same threshold as the vendored model: evaluate layer by layer (one
    # layer in flight) above this width to bound prefill memory.
    pipeline_min_tokens = 256

    def model_type(self, target_model: Any) -> str:
        return _model_type(target_model)

    def supports_model(self, target_model: Any) -> bool:
        if not is_glm5_model_type(self.model_type(target_model)):
            return False
        try:
            inner = self.text_model(target_model)
        except AttributeError:
            return False
        return (
            hasattr(inner, "layers")
            and hasattr(inner, "embed_tokens")
            and hasattr(inner, "fa_idx")
            and hasattr(inner, "ssm_idx")
        )

    def family(self, target_model: Any) -> str:
        del target_model
        return "glm5_next_kda_dsa"

    def capabilities_for(self, target_model: Any) -> TargetCapabilities:
        del target_model
        return TargetCapabilities(
            supports_dflash=True,
            supports_recurrent_rollback=True,
            supports_kv_trim=True,
            # dflash-mlx's snapshot codec only serializes bare KVCache and its
            # own recurrent cache; GLM DSA layers use CacheList(KV, Pooling).
            supports_prefix_snapshot=False,
            supports_rotating_cache_snapshot=False,
            supports_shared_kv=False,
            supports_target_hidden_capture=True,
            # The Qwen verify-qmm kernels have not passed GLM parity gates.
            supports_verify_linear=False,
            supports_full_context_draft_layers=False,
            supports_tree_verify=False,
        )

    def supports_tree_cache(self, cache_entries: list[Any]) -> bool:
        del cache_entries
        return False

    def text_wrapper(self, target_model: Any) -> Any:
        wrapper = getattr(target_model, "language_model", None)
        if wrapper is None or not hasattr(wrapper, "model"):
            raise AttributeError(
                f"Unsupported GLM-5.3 model wrapper: {type(target_model)!r}"
            )
        return wrapper

    def text_model(self, target_model: Any) -> Any:
        return self.text_wrapper(target_model).model

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(
        self, target_model: Any, hidden_states: mx.array
    ) -> mx.array:
        wrapper = self.text_wrapper(target_model)
        if bool(getattr(wrapper.args, "tie_word_embeddings", False)):
            return wrapper.model.embed_tokens.as_linear(hidden_states)
        return _glm_linear_forward()(wrapper.lm_head, hidden_states)

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool,
        quantize_kv_cache: bool = False,
        target_fa_window: int | None = None,
    ) -> list[Any]:
        if not enable_speculative_linear_cache:
            raise ValueError("GLM-5.3 DFlash requires recurrent rollback caches")
        if quantize_kv_cache:
            raise ValueError("GLM-5.3 DFlash target KV quantization is unproven")
        if target_fa_window is not None and int(target_fa_window) > 0:
            raise ValueError("GLM-5.3 DFlash does not support target_fa_window")

        wrapper = self.text_wrapper(target_model)
        caches = list(wrapper.make_cache())
        inner = wrapper.model
        if len(caches) != len(inner.layers):
            raise ValueError("GLM-5.3 target cache/layer count mismatch")
        for index, layer in enumerate(inner.layers):
            if getattr(layer, "is_linear", False):
                conv_kernel = int(layer.self_attn.conv_kernel_size)
                caches[index] = _Glm5RecurrentRollbackCache(
                    size=2, conv_kernel_size=conv_kernel
                )
        return caches

    def install_speculative_hooks(self, target_model: Any) -> None:
        inner = self.text_model(target_model)
        for layer in inner.layers:
            if getattr(layer, "is_linear", False):
                _install_glm5_recurrent_hook(layer.self_attn)

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
            raise ValueError("GLM-5.3 cache/layer count mismatch")

        width = int(h.shape[1])
        chunk = max(1, int(self.prefill_chunk_size or 0))
        if width <= chunk:
            return self._forward_chunk(
                target_model,
                inner,
                h,
                cache,
                capture_layer_ids,
                logits_last_only=logits_last_only,
                want_logits=True,
            )

        # Cold prefill wider than one chunk: the runtime hands GLM the whole
        # prompt in one call because chunked prefill is tied to prefix
        # snapshots there. Split it here so a long prompt never becomes one
        # monolithic 45-layer graph, evaluating each chunk before the next.
        logits_parts: list[mx.array] = []
        captured_parts: list[list[mx.array] | dict[int, mx.array]] = []
        for start in range(0, width, chunk):
            end = min(start + chunk, width)
            final = end >= width
            chunk_logits, chunk_captured = self._forward_chunk(
                target_model,
                inner,
                h[:, start:end],
                cache,
                capture_layer_ids,
                logits_last_only=logits_last_only,
                want_logits=final or not logits_last_only,
            )
            if not final:
                pending = (
                    list(chunk_captured.values())
                    if isinstance(chunk_captured, dict)
                    else list(chunk_captured)
                )
                if chunk_logits is not None:
                    pending.append(chunk_logits)
                mx.eval(*pending)
            if chunk_logits is not None:
                logits_parts.append(chunk_logits)
            captured_parts.append(chunk_captured)

        if isinstance(captured_parts[-1], dict):
            captured: list[mx.array] | dict[int, mx.array] = {}
            for key in captured_parts[-1]:
                if key == -1:
                    # Final-position hidden (set for logits_last_only) is
                    # only meaningful for the last chunk.
                    captured[-1] = captured_parts[-1][-1]
                    continue
                captured[key] = mx.concatenate(
                    [part[key] for part in captured_parts], axis=1
                )
        else:
            captured = [
                mx.concatenate(list(columns), axis=1)
                for columns in zip(*captured_parts, strict=True)
            ]
        logits = (
            logits_parts[-1]
            if logits_last_only
            else mx.concatenate(logits_parts, axis=1)
        )
        return logits, captured

    def _forward_chunk(
        self,
        target_model: Any,
        inner: Any,
        h: mx.array,
        cache: list[Any],
        capture_layer_ids: set[int] | None,
        *,
        logits_last_only: bool,
        want_logits: bool,
    ) -> tuple[mx.array | None, list[mx.array] | dict[int, mx.array]]:
        """One target forward over ``h`` mirroring ``Glm5NextModel.__call__``."""
        from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

        from .mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch

        apply_mlx_vlm_glm5_next_compat_patch()
        from mlx_vlm.models.glm5_next.language import (
            _DECODE_BLOCK,
            _DECODE_EVAL_EVERY,
            _HCDeferred,
        )

        fa_cache = cache[inner.fa_idx]
        fa_mask = create_attention_mask(
            h,
            fa_cache[0] if fa_cache is not None else None,
            return_array=True,
        )
        ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])

        h = mx.contiguous(
            mx.broadcast_to(
                h[:, :, None, :],
                (h.shape[0], h.shape[1], inner.hc_mult, h.shape[2]),
            )
        )

        capture_all = capture_layer_ids is None
        if capture_all:
            captured: list[mx.array] | dict[int, mx.array] = [
                _contract_mhc_hidden(h)
            ]
        else:
            capture_layer_ids = set(capture_layer_ids)
            captured = {0: _contract_mhc_hidden(h)} if 0 in capture_layer_ids else {}

        # Wide (prefill) chunks use the vendored model's layer pipeline, so a
        # 45-layer MoE graph never builds up unevaluated.
        pipeline = (
            LayerPipeline(on_evaluated=mx.clear_cache, lazy_last=True)
            if int(h.shape[1]) >= int(self.pipeline_min_tokens)
            else None
        )
        # Same decode/verify scheduling as Glm5NextModel.__call__. A captured
        # deferred layer is materialized; the carry stays deferred.
        defer = h.shape[:2] == (1, 1)
        eval_every = (
            _DECODE_EVAL_EVERY
            if h.shape[0] == 1 and h.shape[1] <= _DECODE_BLOCK
            else 0
        )
        n_layers = len(inner.layers)
        for layer_index, (layer, layer_cache) in enumerate(
            zip(inner.layers, cache, strict=True)
        ):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            if defer:
                h = layer(
                    h, mask=mask, cache=layer_cache, defer=layer_index + 1 < n_layers
                )
            else:
                h = layer(h, mask=mask, cache=layer_cache)
            if pipeline is not None:
                pipeline.push(h)
            elif (
                eval_every
                and (layer_index + 1) % eval_every == 0
                and layer_index + 1 < n_layers
            ):
                mx.async_eval(h.arrays() if isinstance(h, _HCDeferred) else h)
            capture_key = layer_index + 1
            if capture_all:
                captured.append(_contract_mhc_hidden(h))
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = _contract_mhc_hidden(h)
        if pipeline is not None:
            pipeline.drain()

        if not want_logits:
            return None, captured
        normalized = inner.norm(_contract_mhc_hidden(h))
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
        width = int(verify_ids.shape[1])
        if width <= 0:
            raise ValueError("verify block must contain at least one token")
        if width > POOLING_UNDO_MAX_TOKENS:
            raise ValueError(
                "GLM-5.3 DFlash verify blocks are limited to "
                f"{POOLING_UNDO_MAX_TOKENS} tokens (got {width}); lower "
                "dflash_block_size"
            )
        # PoolingCache only retains its cross-boundary undo record while this
        # thread-local gate is armed; scope it to the verify forward.
        from .mlx_lm_mtp import cache_rollback

        cache_rollback.apply()
        cache_rollback.set_undo_armed(True)
        try:
            return self.forward_with_hidden_capture(
                target_model,
                input_ids=verify_ids,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
        finally:
            cache_rollback.set_undo_armed(False)

    def verify_tree_block(
        self,
        *,
        target_model: Any,
        tree_inputs: Any,
        target_cache: list[Any],
        capture_layer_ids: set[int] | None = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        del target_model, tree_inputs, target_cache, capture_layer_ids
        raise NotImplementedError("GLM-5.3 DFlash tree verification is unproven")

    def restore_after_tree_acceptance(
        self, cache_entries: list[Any], *, accepted_tree_indices: list[int]
    ) -> int:
        del cache_entries, accepted_tree_indices
        raise NotImplementedError("GLM-5.3 DFlash tree verification is unproven")

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
        for cache_entry in cache_entries:
            if isinstance(cache_entry, RecurrentRollbackCache):
                cache_entry.arm_rollback(prefix_len=prefix_len)

    @staticmethod
    def _clear_glm_recurrent_transients(cache_entry: Any) -> None:
        if hasattr(cache_entry, _VERIFY_STATE_ATTR):
            delattr(cache_entry, _VERIFY_STATE_ATTR)
        cache_entry.clear_transients()

    @staticmethod
    def _clear_composite_undo(cache_entry: Any) -> None:
        """Drop accepted verify-block undo arrays retained by PoolingCache."""
        for component in getattr(cache_entry, "caches", ()):
            if hasattr(component, "_undo"):
                component._undo = None
            if hasattr(component, "_undo_chain"):
                component._undo_chain = False

    @classmethod
    def _rollback_glm_recurrent(
        cls,
        cache_entry: RecurrentRollbackCache,
        *,
        accepted_steps: int,
    ) -> None:
        snapshot = getattr(cache_entry, "_snapshot", None)
        verify = getattr(cache_entry, _VERIFY_STATE_ATTR, None)
        if snapshot is None or verify is None:
            cls._clear_glm_recurrent_transients(cache_entry)
            raise RuntimeError("GLM-5.3 recurrent rollback state is missing")

        attention, inputs, mask = verify
        accepted_steps = max(0, min(int(accepted_steps), int(inputs.shape[1])))
        cache_entry.cache = list(snapshot)
        if accepted_steps > 0:
            step_mask = (
                mask[:, :accepted_steps]
                if isinstance(mask, mx.array) and mask.ndim == 2
                else None
            )
            original_call = _ORIGINAL_LINEAR_CALLS.get(type(attention))
            if original_call is None:
                cls._clear_glm_recurrent_transients(cache_entry)
                raise RuntimeError("GLM-5.3 recurrent target hook is not installed")
            # One bounded native call recreates both conv and KDA state using
            # exactly the target implementation and the accepted-width shape.
            original_call(
                attention,
                inputs[:, :accepted_steps],
                mask=step_mask,
                cache=cache_entry,
            )
        cls._clear_glm_recurrent_transients(cache_entry)

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int:
        started = time.perf_counter_ns()
        changed = False
        fully_accepted = acceptance_length == drafted_tokens
        for cache_entry in cache_entries:
            if isinstance(cache_entry, RecurrentRollbackCache):
                if fully_accepted:
                    self._clear_glm_recurrent_transients(cache_entry)
                else:
                    self._rollback_glm_recurrent(
                        cache_entry,
                        accepted_steps=int(acceptance_length) + 1,
                    )
                changed = True
                continue

            # GLM DSA cache entries are CacheList(KVCache, PoolingCache).
            try:
                kv_cache = cache_entry[0]
            except (IndexError, TypeError):
                kv_cache = None
            offset = int(getattr(kv_cache, "offset", 0) or 0)
            trim_count = max(0, offset - int(target_len))
            if trim_count <= 0:
                if fully_accepted:
                    self._clear_composite_undo(cache_entry)
                continue
            trim = getattr(cache_entry, "trim", None)
            if not callable(trim):
                raise RuntimeError(
                    f"GLM-5.3 DFlash cannot trim {type(cache_entry).__name__}"
                )
            trimmed = int(trim(trim_count))
            if trimmed != trim_count:
                raise RuntimeError(
                    "GLM-5.3 DFlash composite-cache rollback failed: "
                    f"requested={trim_count}, trimmed={trimmed}"
                )
            self._clear_composite_undo(cache_entry)
            changed = True
        return time.perf_counter_ns() - started if changed else 0

    def cleanup_generation_caches(
        self, target_cache: list[Any], draft_cache: list[Any]
    ) -> None:
        for entry in target_cache:
            if isinstance(entry, RecurrentRollbackCache):
                self._clear_glm_recurrent_transients(entry)
        draft_cache.clear()
        target_cache.clear()


def load_glm5_target_bundle(
    model_ref: str | Path | None,
    *,
    lazy: bool = False,
    quantize_kv_cache: bool = False,
    verify_config: Any | None = None,
    trust_remote_code: bool = False,
) -> Any:
    """Load GLM-5.3 through mlx-vlm and return dflash-mlx's bundle type.

    dflash-mlx's own ``load_target_bundle`` goes through mlx-lm, which has no
    ``glm5_next`` module. Follow ``VLMBatchedEngine`` instead: register the
    vendored module, run the pre-quantization sanitize scope, prefer oMLX's
    custom quantization loader, construct the model eagerly, and then
    materialize the whole tree on the loader (MLX executor) thread before any
    speculative hook or request can observe it.
    """
    del lazy, verify_config
    if quantize_kv_cache:
        raise ValueError("GLM-5.3 DFlash target KV quantization is unproven")
    if model_ref is None:
        raise ValueError("target model reference is required")

    from dflash_mlx.engine.target_ops import resolve_target_ops
    from dflash_mlx.runtime.loading import LoadedTargetBundle
    from mlx_vlm.utils import load as vlm_load

    from ..engine import vlm as vlm_engine
    from ..utils import model_loading
    from .mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch

    model_path = Path(str(model_ref)).expanduser()
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if not is_glm5_model_type(config.get("model_type")):
        raise ValueError(
            f"{model_path} is not a GLM-5.3 checkpoint "
            f"(model_type={config.get('model_type')!r})"
        )

    apply_mlx_vlm_glm5_next_compat_patch()
    vlm_engine._patch_video_processor_bug()
    vlm_engine._patch_torch_free_image_processor()
    vlm_engine.apply_pixtral_torch_free_patch()
    with vlm_engine._force_qwen4_exp_sanitize_on_load(model_path):
        custom_loaded = model_loading.maybe_load_custom_quantization(
            str(model_path), is_vlm=True
        )
        if custom_loaded is not None:
            model, processor = custom_loaded
        else:
            model, processor = vlm_load(
                str(model_path),
                lazy=False,
                strict=True,
                trust_remote_code=bool(trust_remote_code),
            )
    model_loading.materialize_lazy_state(model)
    target_ops = resolve_target_ops(model)
    target_ops.install_speculative_hooks(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    logger.info(
        "GLM-5.3 DFlash target loaded through mlx-vlm: %s (backend=%s, custom_loader=%s)",
        model_path,
        getattr(target_ops, "backend_name", "?"),
        custom_loaded is not None,
    )
    return LoadedTargetBundle(
        model=model,
        tokenizer=tokenizer,
        meta={
            "resolved_model_ref": str(model_path),
            "config": config,
            "quantize_kv_cache": False,
            "target_family": target_ops.family(model),
            "verify_linear_enabled": False,
            "verify_mode": "disabled-glm5-parity-gate",
        },
        target_ops=target_ops,
    )


def install_dflash_glm5_backend() -> bool:
    """Register the GLM target ops in dflash-mlx's backend registry."""
    from dflash_mlx.engine import target_ops

    if _BACKEND_PATH in target_ops.TARGET_BACKENDS:
        return False
    target_ops.TARGET_BACKENDS.append(_BACKEND_PATH)
    return True


__all__ = [
    "GLM5_MODEL_TYPES",
    "Glm5NextTargetOps",
    "install_dflash_glm5_backend",
    "is_glm5_dflash_target",
    "is_glm5_model_type",
    "load_glm5_target_bundle",
    "restore_glm5_dflash_class_patches",
    "validate_glm5_dflash_pair",
]
