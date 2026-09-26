# SPDX-License-Identifier: Apache-2.0
"""Wrapper that delegates VLM MTP decode to mlx-vlm helpers.

Background
==========

mlx-vlm supports Multi-Token Prediction (MTP) speculative decoding with
external drafter models.  Two drafter families are supported:

- ``gemma4_assistant`` (model_type ``gemma4_assistant``) for Gemma 4 VLMs.
- ``qwen3_5_mtp`` (model_type ``qwen3_5_mtp``) for Qwen 3.5/3.6 models.

Both resolve to ``draft_kind="mtp"`` in mlx-vlm's ``load_drafter()`` and
share the same ``_mtp_rounds`` / ``_mtp_rounds_batch`` round loops in
``mlx_vlm.speculative.utils``.

This module hides the mlx-vlm internal symbols behind a small, typed
interface. Anything that needs to change when mlx-vlm rev's its MTP API
should be contained here.

What this wrapper assumes about callers
=======================================

The caller has already run prefill on the target VLM (with
``return_hidden=True`` and ``return_shared_kv=True``) and holds:

- ``prompt_cache``: list of mlx-lm cache objects post-prefill.
- ``hidden``: last layer hidden state at the final prompt token
  ``[B, 1, H]``.
- ``shared_kv_states``: dict of ``layer_type -> (K, V)`` snapshots.
- ``first_bonus``: token sampled from the post-prefill logits.

The wrapper itself does not touch omlx scheduler state — it only yields
generated tokens.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Generator, List, Optional, Set, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.speculative import common as _vlm_common  # noqa: E402, I001
from mlx_vlm.speculative import load_drafter as _vlm_load_drafter  # noqa: E402
from mlx_vlm.speculative.mtp import _buffer_mtp_target_cache

# The round loops dispatch their target-verify and cache-rollback forwards
# inside ``with mx.stream(generation_stream)``, using mlx-vlm's own
# thread-local stream — a different object from mlx-lm's generation_stream
# and from the per-engine stream. Draining the MTP work means draining this
# one, resolved on the thread that advances the round loop.
# PR #1169 (f96138e) moved the MTP round loop helpers from ``mlx_vlm.generate``
# into ``mlx_vlm.speculative.utils``. Import directly from the new location —
# the symbols are still ``_``-prefixed but this is now their canonical home.
from mlx_vlm.speculative.utils import (  # noqa: E402, SLF001
    _mtp_rounds,
    _mtp_rounds_batch,
)

from ..patches.mlx_vlm_mtp import apply_external_mtp_runtime_patch  # noqa: E402
from ..utils.metal_sync import _sync_and_clear_cache  # noqa: E402
from ..utils.model_loading import materialize_lazy_state  # noqa: E402

_vlm_generation_stream = _vlm_common.generation_stream

logger = logging.getLogger(__name__)


class VLMMTPDrafter:
    """Holds a loaded drafter together with the metadata omlx needs.

    ``model.reset(target)`` is intentionally NOT called here: mlx-vlm's
    ``_mtp_rounds`` / ``_mtp_rounds_batch`` call it themselves at the
    start of every round-loop entry, so adding an extra reset would just
    duplicate the bind step (and could mask a target-model swap).
    """

    def __init__(self, model: nn.Module, draft_kind: str, source_path: str) -> None:
        self.model = model
        self.draft_kind = draft_kind
        self.source_path = source_path


class _VLMAdapterMTPProxy:
    """Expose VLM adapter calls while letting Qwen drafters bind embeddings.

    mlx-vlm's MTP loop calls ``model.language_model`` when that attribute is
    present, which would bypass oMLX's VLM adapter and lose mRoPE position
    handling. Qwen's external drafter, however, needs a ``language_model``
    attribute during ``bind()`` to find ``embed_tokens``. This proxy only
    exposes ``language_model`` while ``draft_model.reset(model)`` runs.
    """

    def __init__(self, adapter: nn.Module, language_model: Any) -> None:
        self._adapter = adapter
        self._language_model = language_model
        self._expose_language_model = False
        self._allow_language_model_fast_paths = not bool(
            getattr(adapter, "_uses_mrope", False)
        )

    def __getattr__(self, name: str) -> Any:
        if name == "language_model":
            if self._expose_language_model:
                return self._language_model
            raise AttributeError(name)
        try:
            return getattr(self._adapter, name)
        except AttributeError:
            if (
                not self._allow_language_model_fast_paths
                and (name == "model" or name.startswith("speculative_"))
            ):
                raise
            return getattr(self._language_model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._adapter(*args, **kwargs)

    # Binders resolve these seams on the class (mock attributes must not pose as
    # capabilities), so __getattr__ delegation is not enough: define and forward.
    def set_step_rope_deltas(self, deltas: Any, uids: Any) -> None:
        setter = getattr(type(self._adapter), "set_step_rope_deltas", None)
        if callable(setter):
            setter(self._adapter, deltas, uids)
        else:
            self._adapter.set_batch_rope_deltas(deltas)

    def mark_text_positions(self, uid: Any) -> None:
        marker = getattr(type(self._adapter), "mark_text_positions", None)
        if callable(marker):
            marker(self._adapter, uid)

    def set_text_prefill_rope_delta(self, delta: Any) -> None:
        setter = getattr(type(self._adapter), "set_text_prefill_rope_delta", None)
        if callable(setter):
            setter(self._adapter, delta)
        else:
            self._adapter.set_batch_rope_deltas(mx.array([delta]))


class _MTPResetBindingProxy:
    """Bind the adapter for reset and preserve the Qwen replay cleanup boundary."""

    def __init__(
        self, drafter: nn.Module, target_proxy: _VLMAdapterMTPProxy | None
    ) -> None:
        self._drafter = drafter
        self._target_proxy = target_proxy
        self.clear_before_replay = False

    def __getattr__(self, name: str) -> Any:
        if name == "accept_verified_tokens" and self.clear_before_replay:
            return self._accept_verified_tokens
        return getattr(self._drafter, name)

    def _accept_verified_tokens(self, *args: Any, **kwargs: Any) -> Any:
        # Release the idle pool before replay and commit replace live cache state.
        _sync_and_clear_cache(_vlm_generation_stream)
        return self._drafter.accept_verified_tokens(*args, **kwargs)

    def reset(self, target_model: Any, *args: Any, **kwargs: Any) -> Any:
        if self._target_proxy is not None and target_model is self._target_proxy:
            self._target_proxy._expose_language_model = True
            try:
                return self._drafter.reset(target_model, *args, **kwargs)
            finally:
                self._target_proxy._expose_language_model = False
        return self._drafter.reset(target_model, *args, **kwargs)


def vlm_mtp_positioned_sampling_available(target_language_model: Any) -> bool:
    """True when mlx-vlm's round loop will see ``speculative_logits_from_hidden``.

    The positioned ``sample_target`` verify path — the application point for
    per-request logits processors on the vlm_mtp path — is only consulted
    when the object mlx-vlm resolves as ``lm`` exposes
    ``speculative_logits_from_hidden``. That object is not the inner
    language model: ``run_vlm_mtp_decode`` wraps adapters in
    ``_VLMAdapterMTPProxy``, which for mRoPE adapters (Qwen VLMs)
    intentionally hides the inner model's ``speculative_*`` fast paths so
    verify keeps mRoPE position handling. Checking the inner model
    directly would report the hook as available while the round loop
    falls back to plain vectorized sampling — silently dropping the
    processors (#2399).

    This helper mirrors the proxy's visibility rules exactly; the
    equivalence is pinned by tests against the real proxy resolution.
    """
    adapter_lm = getattr(target_language_model, "_language_model", None)
    if adapter_lm is None:
        # No adapter: the model itself is handed to the round loop, and
        # mlx-vlm resolves ``lm = model.language_model`` when present.
        inner = getattr(target_language_model, "language_model", None)
        lm = inner if inner is not None else target_language_model
        return hasattr(lm, "speculative_logits_from_hidden")
    # Adapter case: rounds receive _VLMAdapterMTPProxy, which hides
    # ``language_model`` outside drafter reset, so mlx-vlm uses the proxy
    # itself as ``lm``. Attribute lookup tries the adapter first, then
    # falls through to the inner model — unless the adapter uses mRoPE,
    # in which case ``speculative_*`` names are blocked at the proxy.
    if hasattr(target_language_model, "speculative_logits_from_hidden"):
        return True
    if bool(getattr(target_language_model, "_uses_mrope", False)):
        return False
    return hasattr(adapter_lm, "speculative_logits_from_hidden")


def load_vlm_mtp_drafter(path: str) -> Optional[VLMMTPDrafter]:
    """Load an MTP drafter (gemma4_assistant or qwen3_5_mtp); return None
    and log if the artifact is the wrong kind. Soft-fails so a misconfigured
    toggle does not crash model loading."""
    try:
        drafter_model, resolved_kind = _vlm_load_drafter(path, kind=None)
    except Exception as e:
        logger.warning(
            "VLM MTP drafter load failed for %r: %s — toggle will be ignored",
            path,
            e,
        )
        return None

    if resolved_kind != "mtp":
        logger.warning(
            "VLM MTP drafter %r resolved to kind=%r (expected 'mtp') — "
            "toggle will be ignored. Only MTP-kind drafters "
            "(gemma4_assistant, qwen3_5_mtp, etc.) are supported.",
            path,
            resolved_kind,
        )
        return None

    model_type = _read_model_type(drafter_model)

    # Materialize frozen buffers (RoPE freqs, masked_embedding tables, etc.) on
    # the loader thread. mlx-vlm's load_model only materializes parameters via
    # ``mx.eval(model.parameters())`` and leaves siblings lazy; those buffers
    # stay bound to whichever stream is current here. When per-engine
    # scheduler.step() later evaluates draft_block outputs from a different
    # thread, mx.async_eval hits "no Stream(gpu, X) in current thread" because
    # those lazy ops target a stream that does not exist on the inference
    # thread. Same root cause and fix as 9d5bed8 for the main VLM model.
    # Issue #1469.
    materialize_lazy_state(drafter_model)

    logger.info(
        "VLM MTP drafter loaded: path=%s kind=%s model_type=%s",
        path,
        resolved_kind,
        model_type,
    )
    return VLMMTPDrafter(drafter_model, resolved_kind, path)


def _read_model_type(drafter: nn.Module) -> Optional[str]:
    """Best-effort lookup of the drafter's HF model_type."""
    config = getattr(drafter, "config", None)
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("model_type")
    return getattr(config, "model_type", None)


def run_vlm_mtp_decode(
    *,
    target_language_model: nn.Module,
    drafter: VLMMTPDrafter,
    prompt_cache: List[Any],
    hidden: mx.array,
    shared_kv_states: dict,
    first_bonus: Union[int, mx.array],
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: Optional[mx.array] = None,
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    eos_token_ids: Optional[Set[int]] = None,
    stop_check: Optional[Callable[[int, int], bool]] = None,
) -> Generator[Union[int, List[Optional[int]]], None, None]:
    """Stream decoded tokens via mlx-vlm's MTP rounds.

    Yields plain Python ints for single-request decode (``first_bonus`` is
    ``int`` or a B=1 ``mx.array``), or ``List[Optional[int]]`` rows for
    batched decode (B > 1 ``mx.array``). ``None`` slots in the batched
    form mark rows that have finished.

    The wrapper yields ``first_bonus`` as its first value: mlx-vlm's
    ``_mtp_rounds`` / ``_mtp_rounds_batch`` expect the caller to have
    already emitted the bonus token before the round loop starts
    (``emitted = 1`` baked in at the top of both helpers).
    """
    target_for_rounds = target_language_model
    apply_external_mtp_runtime_patch()
    drafter_model = drafter.model
    adapter_lm = getattr(target_language_model, "_language_model", None)
    if adapter_lm is not None:
        target_for_rounds = _VLMAdapterMTPProxy(target_language_model, adapter_lm)
        drafter_model = _MTPResetBindingProxy(drafter.model, target_for_rounds)

    is_batch = isinstance(first_bonus, mx.array) and first_bonus.size > 1
    clear_before_replay = (
        not is_batch and _read_model_type(drafter.model) == "qwen3_5_mtp"
    )
    if clear_before_replay:
        if not isinstance(drafter_model, _MTPResetBindingProxy):
            drafter_model = _MTPResetBindingProxy(drafter.model, None)
        drafter_model.clear_before_replay = True

    if is_batch:
        first_bonus_list = first_bonus.tolist()  # forces eval once
        yield [int(x) for x in first_bonus_list]
        eos_set = set(eos_token_ids) if eos_token_ids else None
        for tokens, _ in _mtp_rounds_batch(
            target_for_rounds,
            drafter_model,
            prompt_cache,
            hidden,
            shared_kv_states,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=token_dtype,
            stop_check=stop_check,
            eos_token_ids=eos_set,
        ):
            # mlx-vlm only calls mx.clear_cache() every 256 tokens (see
            # _mtp_rounds_batch in mlx_vlm/speculative/utils.py). On large
            # targets like Gemma 4 31B the buffer pool balloons between
            # those flushes (issue #1416). Clearing per round bounds it.
            #
            # The round leaves work in flight on two streams at this yield
            # boundary: mlx-vlm async_evals the verify hidden state and the
            # drafter's state arrays. The helper drains the stream it is
            # given plus the current default stream, so passing mlx-vlm's
            # stream covers the verify forwards while the default-stream
            # drain covers the engine stream the scheduler advances this
            # generator under (``with mx.stream(self._stream)``).
            _sync_and_clear_cache(_vlm_generation_stream)
            yield tokens
        return

    if isinstance(first_bonus, mx.array):
        first_bonus_int = int(first_bonus.item())
    else:
        first_bonus_int = int(first_bonus)

    yield first_bonus_int

    _buffer_mtp_target_cache(prompt_cache, drafter_model, draft_block_size)
    rounds = _mtp_rounds(
        target_for_rounds,
        drafter_model,
        prompt_cache,
        hidden,
        shared_kv_states,
        prompt_tokens=prompt_tokens,
        first_bonus=first_bonus_int,
        max_tokens=max_tokens,
        sampler=sampler,
        draft_block_size=draft_block_size,
        token_dtype=token_dtype,
    )
    try:
        for tok, _ in rounds:
            if not clear_before_replay:
                _sync_and_clear_cache(_vlm_generation_stream)
            yield tok
    finally:
        close = getattr(rounds, "close", None)
        if close is not None:
            close()
