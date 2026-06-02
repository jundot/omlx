# SPDX-License-Identifier: Apache-2.0
"""Lifecycle wrap for dflash-mlx's class-level monkey patches.

dflash-mlx patches linear-attention / attention ``__call__`` at the class
level (``cls.__call__ = speculative_call`` etc.) inside its hook installer
functions, and uses class attributes like ``_dflash_speculative_call_installed``
as idempotency guards. Those patches persist for the lifetime of the
Python process — engine teardown does not undo them. Two engines sharing
a Python class then see crossed-over state: a later Native MTP load after
a DFlash session ends up with the dflash hook on ``linear_attn.__call__``
and the MTP draft cycle crashes with
``TypeError: speculative_call() got an unexpected keyword argument 'n_confirmed'``
(issue #1388).

This module wraps each dflash hook installer so oMLX can:
  - capture the pre-dflash ``__call__`` before dflash overwrites it
  - on ``restore_dflash_class_patches()`` (called from ``DFlashEngine.stop()``),
    revert each touched class to that captured state and clear dflash's
    idempotency flag so a subsequent DFlash load can re-arm cleanly

The wrap is idempotent and runs once per process — typically at the
beginning of ``DFlashEngine.start()`` just before ``load_target_bundle``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# cls -> {"call": pre_dflash_call, "flag": dflash_idempotency_attr_name}
_DFLASH_BACKUP: dict[type, dict[str, Any]] = {}


def _wrap_installer(mod: Any, fn_name: str, flag_name: str) -> bool:
    """Wrap ``mod.fn_name`` so each first-time class touch is recorded.

    ``fn_name`` is a dflash hook installer that takes a single
    ``linear_attn``-like argument and rewrites its class's ``__call__``.
    ``flag_name`` is the per-class idempotency attribute that dflash
    sets when its hook is installed — we use it to detect "already
    patched" so we don't double-record a backup.
    """
    if getattr(mod, "_omlx_wrapped_" + fn_name, False):
        return True

    original = getattr(mod, fn_name, None)
    if original is None:
        return False

    def wrapped(module_target: Any) -> Any:
        cls = type(module_target)
        if not getattr(cls, flag_name, False):
            # First time dflash installs on this class — snapshot the
            # current __call__ so restore can put it back unchanged.
            _DFLASH_BACKUP.setdefault(cls, {"call": cls.__call__, "flag": flag_name})
        return original(module_target)

    setattr(mod, fn_name, wrapped)
    setattr(mod, "_omlx_wrapped_" + fn_name, True)
    return True


def install_dflash_lifecycle_wrap() -> bool:
    """Monkey-patch dflash's hook installers to record pre-dflash class state.

    Safe to call repeatedly — each installer is wrapped at most once.
    Returns True if at least one backend's installers were wrapped.
    """
    wrapped_any = False

    try:
        from dflash_mlx.engine import target_qwen_gdn as _qwen_gdn
    except ImportError:
        logger.debug("dflash_mlx.engine.target_qwen_gdn not importable")
    else:
        wrapped_any |= _wrap_installer(
            _qwen_gdn,
            "_install_speculative_linear_cache_hook",
            "_dflash_speculative_call_installed",
        )
        # dflash 0.1.7 renamed the Qwen full-attention installer from
        # ``_install_split_full_attention_hook`` to ``_install_full_attention_gqa_hook``
        # (target_qwen_gdn). Without wrapping the new name the pre-dflash
        # ``Attention.__call__`` is never snapshotted, so a DFlash -> MTP
        # transition leaves dflash's hook on the class and the MTP draft
        # cycle crashes with "[convert] Only length-1 arrays ..." on the
        # per-row batched ``cache.offset`` (issue #1510).
        wrapped_any |= _wrap_installer(
            _qwen_gdn,
            "_install_full_attention_gqa_hook",
            "_dflash_full_attention_gqa_installed",
        )

    try:
        from dflash_mlx.engine import target_gemma4 as _gemma4
    except ImportError:
        logger.debug("dflash_mlx.engine.target_gemma4 not importable")
    else:
        wrapped_any |= _wrap_installer(
            _gemma4,
            "_install_full_attention_gqa_hook",
            "_dflash_full_attention_gqa_installed",
        )

    if wrapped_any:
        logger.debug("dflash lifecycle wrap installed")
    return wrapped_any


def restore_dflash_class_patches() -> None:
    """Revert every dflash-touched class to its pre-dflash ``__call__``.

    Also clears the dflash idempotency flag on each class so a later
    DFlash engine load can re-install its hook freshly. Empties the
    backup table.
    """
    if not _DFLASH_BACKUP:
        return

    restored = 0
    for cls, info in list(_DFLASH_BACKUP.items()):
        try:
            cls.__call__ = info["call"]
        except Exception as exc:
            logger.debug("restore failed for %s: %s", cls, exc)
            continue
        flag = info["flag"]
        if flag in cls.__dict__:
            try:
                delattr(cls, flag)
            except AttributeError:
                pass
        restored += 1

    _DFLASH_BACKUP.clear()
    logger.info("dflash class patches restored on %d class(es)", restored)


def get_backup_classes() -> list[type]:
    """Return classes currently in the backup table — used by tests."""
    return list(_DFLASH_BACKUP.keys())
