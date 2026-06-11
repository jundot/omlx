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

from mlx_vlm.speculative import load_drafter as _vlm_load_drafter

# PR #1169 (f96138e) moved the MTP round loop helpers from ``mlx_vlm.generate``
# into ``mlx_vlm.speculative.utils``. Import directly from the new location —
# the symbols are still ``_``-prefixed but this is now their canonical home.
from mlx_vlm.speculative.utils import _mtp_rounds, _mtp_rounds_batch  # noqa: SLF001

from ..utils.model_loading import materialize_lazy_state

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# mlx-vlm compat patch: Qwen3.5 MoE MTP drafter support
# ---------------------------------------------------------------------------
# mlx-vlm's ``qwen3_5_mtp`` module hard-codes the *dense* ``TextConfig``
# from ``mlx_vlm.models.qwen3_5.config``.  When the MTP drafter is trained
# from a MoE base model (e.g. Qwen3.6-35B-A3B), its ``text_config`` has
# ``model_type="qwen3_5_moe_text"`` and uses ``moe_intermediate_size``
# instead of ``intermediate_size``.  The dense ``TextConfig`` rejects this,
# causing ``TextConfig.__init__() missing 1 required positional argument:
# 'intermediate_size'``.
#
# The error occurs in ``mlx_vlm.utils.update_module_configs`` which calls
# ``model_class.TextConfig.from_dict(text_config_dict)`` — bypassing
# ``Qwen3_5MTPConfig.__post_init__`` entirely.  We fix this by replacing
# the ``TextConfig`` re-export on the ``qwen3_5_mtp`` package with a
# dispatcher that picks the correct config class based on ``model_type``.
# This is safe to call multiple times (idempotent).


def _patch_qwen35_mtp_config_for_moe() -> None:
    """Make ``qwen3_5_mtp`` module accept MoE ``text_config`` dicts.

    Two code paths need patching:
    1. ``update_module_configs`` reads ``model_class.TextConfig`` (package attr).
    2. ``Qwen3_5MTPConfig.__post_init__`` imports ``TextConfig`` directly
       from ``mlx_vlm.models.qwen3_5.config``.

    We patch both: replace the package-level re-export *and* monkey-patch
    ``__post_init__`` to use the correct config class.
    """
    try:
        import mlx_vlm.speculative.drafters.qwen3_5_mtp as mtp_pkg
        from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import (
            Qwen3_5MTPConfig,
        )
        from mlx_vlm.models.qwen3_5.config import (
            TextConfig as DenseTextConfig,
        )
    except ImportError:
        return  # drafter module not available; nothing to patch

    # -- Patch 1: replace package-level TextConfig for update_module_configs --
    class _DispatchingTextConfig(DenseTextConfig):
        """TextConfig subclass that dispatches to MoE config when needed."""

        @classmethod
        def from_dict(cls, params: dict):
            if isinstance(params, dict) and params.get("model_type") == "qwen3_5_moe_text":
                try:
                    from mlx_vlm.models.qwen3_5_moe.config import (
                        TextConfig as MoETextConfig,
                    )

                    return MoETextConfig.from_dict(params)
                except Exception:
                    pass  # fall through to dense
            return DenseTextConfig.from_dict(params)

    mtp_pkg.TextConfig = _DispatchingTextConfig

    # -- Patch 2: fix Qwen3_5MTPConfig.__post_init__ direct import --
    _original_post_init = Qwen3_5MTPConfig.__post_init__

    def _patched_post_init(self):
        raw = getattr(self, "text_config", None)
        if isinstance(raw, dict) and raw.get("model_type") == "qwen3_5_moe_text":
            try:
                from mlx_vlm.models.qwen3_5_moe.config import (
                    TextConfig as MoETextConfig,
                )

                self.text_config = MoETextConfig.from_dict(raw)
                for key in ("mtp_num_hidden_layers", "mtp_use_dedicated_embeddings"):
                    if key in raw:
                        setattr(self.text_config, key, raw[key])
                if self.text_config is not None:
                    self.tie_word_embeddings = bool(
                        self.text_config.tie_word_embeddings
                    )
                return  # skip the original __post_init__
            except Exception:
                pass  # fall through to original
        _original_post_init(self)

    Qwen3_5MTPConfig.__post_init__ = _patched_post_init

    # -- Patch 3: use MoE decoder layer for MoE MTP drafters --
    # ``Qwen3_5MTPDraftModel.__init__`` creates ``Qwen3_5DecoderLayer``
    # (dense MLP) but MoE MTP weights use ``Qwen3_5MoeSparseMoeBlock``.
    # We monkey-patch the module-level ``Qwen3_5DecoderLayer`` reference so
    # the existing ``__init__`` picks up MoE layers when the text_config
    # indicates a MoE architecture.
    import mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp as _mtp_mod

    _orig_dense_layer = _mtp_mod.Qwen3_5DecoderLayer

    def _moe_aware_decoder_layer(args, layer_idx):
        """Dispatch to MoE decoder layer when args indicate a MoE model."""
        if getattr(args, "model_type", "") == "qwen3_5_moe_text":
            try:
                from mlx_vlm.models.qwen3_5_moe.language import (
                    Qwen3_5MoeDecoderLayer,
                )

                return Qwen3_5MoeDecoderLayer(args=args, layer_idx=layer_idx)
            except Exception:
                pass  # fall through to dense
        return _orig_dense_layer(args=args, layer_idx=layer_idx)

    _mtp_mod.Qwen3_5DecoderLayer = _moe_aware_decoder_layer
    logger.debug("Patched qwen3_5_mtp for MoE text_config support")


_patch_qwen35_mtp_config_for_moe()


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
    is_batch = isinstance(first_bonus, mx.array) and first_bonus.size > 1

    if is_batch:
        first_bonus_list = first_bonus.tolist()  # forces eval once
        yield [int(x) for x in first_bonus_list]
        eos_set = set(eos_token_ids) if eos_token_ids else None
        for tokens, _ in _mtp_rounds_batch(
            target_language_model,
            drafter.model,
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
            mx.clear_cache()
            yield tokens
        return

    if isinstance(first_bonus, mx.array):
        first_bonus_int = int(first_bonus.item())
    else:
        first_bonus_int = int(first_bonus)

    yield first_bonus_int

    for tok, _ in _mtp_rounds(
        target_language_model,
        drafter.model,
        prompt_cache,
        hidden,
        shared_kv_states,
        first_bonus=first_bonus_int,
        max_tokens=max_tokens,
        sampler=sampler,
        draft_block_size=draft_block_size,
        token_dtype=token_dtype,
    ):
        mx.clear_cache()
        yield tok
