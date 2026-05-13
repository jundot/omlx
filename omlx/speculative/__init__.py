# SPDX-License-Identifier: Apache-2.0
"""omlx speculative-decoding wrappers.

This package collects integration code that bridges omlx scheduling/cache
infrastructure with upstream speculative-decoding implementations in mlx-lm
and mlx-vlm. Pure helpers (no business logic of their own) so the surface
of internal-API dependencies is easy to audit on each upstream bump.

Upstream compatibility patches (applied at import time):

- ``dflash_mlx.runtime.get_stop_token_ids``: HF GemmaTokenizer's
  ``eos_token_ids`` (plural) attribute returns ``int`` rather than a list.
  Upstream wraps in ``list(...)`` which raises ``TypeError: 'int' object is
  not iterable``. We monkey-patch to coerce int → ``[int]``. Discovered
  during D3 spike6 end-to-end on m5max with Gemma 4. The patch is idempotent
  and only modifies the upstream module attribute at runtime, not the source.
"""

def _patch_dflash_get_stop_token_ids() -> None:
    """Coerce int eos_token_ids to list before upstream wraps in list()."""
    try:
        import dflash_mlx.runtime as _dflash_runtime
    except ImportError:
        return
    _original = _dflash_runtime.get_stop_token_ids
    if getattr(_original, "_omlx_patched", False):
        return
    def _patched(tokenizer):
        eid = getattr(tokenizer, "eos_token_ids", None)
        if isinstance(eid, int):
            return [eid]
        return _original(tokenizer)
    _patched._omlx_patched = True
    _dflash_runtime.get_stop_token_ids = _patched


_patch_dflash_get_stop_token_ids()


def detect_fallback_engine_type(model_name: str) -> str:
    """Return ``"vlm"`` if model has vision/audio capability, else ``"batched"``.

    Uses ``vision_config``/``audio_config`` keys in ``config.json`` as the
    canonical multimodal marker. ``processor_config.json`` alone is NOT a
    reliable signal because text-only mlx-community wrappers may include
    one for tokenizer setup (e.g. some Qwen 3.5 packaging).

    For HF repo IDs (non-path), returns ``"batched"`` — caller should pass
    a resolved local model directory.
    """
    import json
    from pathlib import Path

    p = Path(model_name)
    if not p.is_dir():
        return "batched"
    cfg_path = p / "config.json"
    if not cfg_path.exists():
        return "batched"
    try:
        with cfg_path.open() as f:
            cfg = json.load(f)
        if "vision_config" in cfg or "audio_config" in cfg:
            return "vlm"
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return "batched"
