# SPDX-License-Identifier: Apache-2.0
"""Wrapper that delegates Gemma4-style VLM MTP decode to mlx-vlm helpers.

Background
==========

mlx-vlm 191d7c8 added a Multi-Token Prediction (MTP) speculative decoding
path for Gemma 4 with an external assistant drafter (model_type
``gemma4_assistant``). The core loop lives in
``mlx_vlm.generate._mtp_rounds`` / ``_mtp_rounds_batch`` and operates on
plain ``mx.array`` state plus an ``mlx_lm`` ``KVCache`` list, so omlx can
reuse it without porting the algorithm.

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

import importlib
import logging
from typing import Any, Callable, Generator, List, Optional, Set, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.speculative import load_drafter as _vlm_load_drafter

logger = logging.getLogger(__name__)


# mlx-vlm exposes ``generate`` as a top-level function shadowing the module,
# so importlib is the only reliable way to reach the module-level helpers.
_mlx_vlm_generate = importlib.import_module("mlx_vlm.generate")
_mtp_rounds = _mlx_vlm_generate._mtp_rounds  # noqa: SLF001 — intentional pin
_mtp_rounds_batch = _mlx_vlm_generate._mtp_rounds_batch  # noqa: SLF001


# What model_type strings count as a gemma4 assistant drafter. Kept as a
# tuple so we can extend if upstream adds related drafter kinds later.
GEMMA4_ASSISTANT_MODEL_TYPES: tuple[str, ...] = ("gemma4_assistant",)


class VLMMTPDrafter:
    """Holds a loaded drafter together with the metadata omlx needs.

    Loading is deferred to :meth:`load_for_target` so the engine_pool can
    surface load errors with the target model id in the message.
    """

    def __init__(self, model: nn.Module, draft_kind: str, source_path: str) -> None:
        self.model = model
        self.draft_kind = draft_kind
        self.source_path = source_path
        self._is_bound = False

    def bind_to(self, target_language_model: nn.Module) -> None:
        """Call drafter.reset(target) once so the assistant captures the
        target's embed_tokens reference."""
        if self._is_bound:
            return
        self.model.reset(target_language_model)
        self._is_bound = True


def load_vlm_mtp_drafter(path: str) -> Optional[VLMMTPDrafter]:
    """Load a Gemma4 assistant drafter; return None and log if the artifact
    is the wrong kind. Soft-fails so a misconfigured toggle does not crash
    model loading."""
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
            "toggle will be ignored. Only gemma4_assistant drafters are "
            "supported at this time.",
            path,
            resolved_kind,
        )
        return None

    model_type = _read_model_type(drafter_model)
    if model_type not in GEMMA4_ASSISTANT_MODEL_TYPES:
        logger.warning(
            "VLM MTP drafter %r has model_type=%r (expected one of %s) — "
            "toggle will be ignored.",
            path,
            model_type,
            GEMMA4_ASSISTANT_MODEL_TYPES,
        )
        return None

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
    ``int``), or ``List[Optional[int]]`` rows for batched decode
    (``first_bonus`` is an ``mx.array`` of shape ``[B]``). ``None`` slots
    in the batched form mark rows that have finished.
    """
    drafter.bind_to(target_language_model)

    is_batch = isinstance(first_bonus, mx.array) and first_bonus.size > 1

    if is_batch:
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
            yield tokens
        return

    if isinstance(first_bonus, mx.array):
        first_bonus_int = int(first_bonus.item())
    else:
        first_bonus_int = int(first_bonus)

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
        yield tok
