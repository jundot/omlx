# SPDX-License-Identifier: Apache-2.0
"""Wrapper that delegates Gemma4-style VLM MTP decode to mlx-vlm helpers.

Background
==========

mlx-vlm 191d7c8 added a Multi-Token Prediction (MTP) speculative decoding
path for Gemma 4 with an external assistant drafter (model_type
``gemma4_assistant``). f96138e (PR #1169) then moved the core round loop
out of ``mlx_vlm.generate`` and into ``mlx_vlm.speculative.utils``, where
``_mtp_rounds`` / ``_mtp_rounds_batch`` now live. The functions still
operate on plain ``mx.array`` state plus an ``mlx_lm`` ``KVCache`` list,
so omlx can reuse them without porting the algorithm.

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

logger = logging.getLogger(__name__)


# What model_type strings count as a gemma4 assistant drafter. Kept as a
# tuple so we can extend if upstream adds related drafter kinds later.
GEMMA4_ASSISTANT_MODEL_TYPES: tuple[str, ...] = ("gemma4_assistant",)


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
        yield tok
