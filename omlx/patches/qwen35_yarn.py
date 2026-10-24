# SPDX-License-Identifier: Apache-2.0
"""Static YaRN for Qwen3.5-family MRoPE attention past the native context.

A ``max_context_window`` above the checkpoint's ``max_position_embeddings``
adds a yarn ``rope_parameters`` entry with ``factor = window / native`` when
the config is loaded. A yarn entry already in ``config.json`` is used as is.

mlx-lm builds ``YarnRoPE`` from that entry. mlx-vlm's MRoPE ignores the rope
type, so the attention constructor rewrites its frequencies and scaling here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx_lm.utils as lm_utils
from mlx_lm.models.rope_utils import initialize_rope

logger = logging.getLogger(__name__)

YARN_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe", "qwen4_exp"})

_APPLIED = False
_context_windows: dict[str, int] = {}


def _key(model_path: Any) -> str:
    return str(Path(model_path).expanduser().resolve())


def _is_yarn(rope: dict) -> bool:
    return (rope.get("type") or rope.get("rope_type")) == "yarn"


def yarn_factor(
    model_type: str | None, native: int | None, window: Any
) -> float | None:
    """Factor a context window adds on top of the native one, or None."""
    if model_type not in YARN_MODEL_TYPES or not native or not window:
        return None
    if int(window) <= int(native):
        return None
    return int(window) / int(native)


def set_context_window(model_path: Any, window: int | None) -> None:
    """Record the context window the next load of ``model_path`` serves."""
    if window:
        _context_windows[_key(model_path)] = int(window)
    else:
        _context_windows.pop(_key(model_path), None)


def add_yarn_to_config(config: dict, model_path: Any) -> dict:
    """Add a yarn ``rope_parameters`` entry for a window past the native one."""
    window = _context_windows.get(_key(model_path))
    if not window or not isinstance(config, dict):
        return config
    text = config.get("text_config")
    text = text if isinstance(text, dict) else config
    rope = text.get("rope_parameters")
    native = text.get("max_position_embeddings")
    factor = yarn_factor(config.get("model_type"), native, window)
    if factor is None or not isinstance(rope, dict) or _is_yarn(rope):
        return config
    type_keys = [key for key in ("type", "rope_type") if key in rope] or ["type"]
    text["rope_parameters"] = {
        **rope,
        **dict.fromkeys(type_keys, "yarn"),
        "factor": factor,
        "original_max_position_embeddings": native,
    }
    logger.info(
        "YaRN enabled for %s: factor %.4f (context window %d, native %d)",
        Path(model_path).name,
        factor,
        window,
        native,
    )
    return config


def _apply_yarn_rotary(rotary: Any, rope: dict | None, native: int) -> None:
    if not isinstance(rope, dict) or not _is_yarn(rope):
        return
    yarn = initialize_rope(
        rotary.dim,
        rotary.base,
        False,
        {"original_max_position_embeddings": native, **rope},
        native,
    )
    rotary._inv_freq = 1.0 / yarn._freqs
    rotary.attention_scaling = yarn.mscale
    # The fused apply kernel reads inv_freq only and would drop the scaling.
    rotary.fused_apply = False
    rotary._compiled_apply = None
    rotary.eval_cached_arrays()


def apply_qwen35_yarn_patch() -> None:
    """Install the config hooks and the mlx-vlm attention hook. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return

    # mlx-vlm is imported here, not at module scope: a CUDA cluster worker's
    # venv ships without it, and model_loading imports this module on every
    # load (#4739). mlx-lm models still get YaRN from the config hook alone.
    try:
        import mlx_vlm.utils as vlm_utils
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError:
        vlm_utils = q35 = None
        logger.debug("mlx-vlm not installed; YaRN applies to mlx-lm models only")

    for utils in (lm_utils, vlm_utils):
        if utils is None:
            continue
        original_load_config = utils.load_config

        def load_config(model_path, *args, _original=original_load_config, **kwargs):
            config = _original(model_path, *args, **kwargs)
            return add_yarn_to_config(config, model_path)

        utils.load_config = load_config

    if q35 is None:
        _APPLIED = True
        return

    original_init = q35.Qwen3_5Attention.__init__

    def init_with_yarn(self, args, *rest, **kwargs):
        original_init(self, args, *rest, **kwargs)
        _apply_yarn_rotary(
            self.rotary_emb,
            getattr(args, "rope_parameters", None),
            args.max_position_embeddings,
        )

    q35.Qwen3_5Attention.__init__ = init_with_yarn
    _APPLIED = True
