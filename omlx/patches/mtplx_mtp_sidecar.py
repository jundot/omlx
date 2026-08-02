# SPDX-License-Identifier: Apache-2.0
"""Bind the standalone MTP sidecar for MTPLX / mlx-lm-forge exports.

MTPLX-style exports (e.g. ``*MTPLX-8bit``) keep the MTP head in a separate
``mtp.safetensors`` referenced by ``config.json ->
mlx_lm_extra_tensors.mtp_file`` (default ``mtp.safetensors``). The sidecar
stores *bare* ``mtp.*`` keys (MLX-LM text-model convention).

Why this patch is needed, and why two layers:

Layer 1 — VLM path (the model's natural engine):
    ``mlx_vlm.utils.load_model`` globs ``*.safetensors`` and DOES load the
    sidecar, but its ``mtp.*`` -> ``language_model.mtp.*`` remap only runs
    inside ``sanitize_weights``, which is skipped for ``format: mlx``
    checkpoints (the ``if not is_mlx_format`` guard). MTPLX exports are
    ``format: mlx``, so the remap is skipped and the bare ``mtp.*`` keys
    fail to bind to the ``language_model.mtp`` module under strict loading
    -> "Missing N parameters: language_model.mtp.*".
    Fix: inject the sidecar into ``nn.Module.load_weights`` (the hook that
    fires unconditionally before binding, for both loaders and both
    formats), remapping bare ``mtp.*`` -> ``language_model.mtp.*`` (VLM) or
    ``mtp.*`` (text) and dropping the bare keys so they never collide.

Layer 2 — text / LM path (benchmarks that ``force_lm=True``):
    ``mlx_lm.utils.load_model`` globs ``model*.safetensors`` and NEVER opens
    ``mtp.safetensors``, so the weights dict has no ``mtp.*`` keys. The
    qwen35 MTP patch's ``sanitize`` (which runs *before* load_weights) then
    raises "Lightning MTP is enabled ... missing the mtp.* tensors". We bridge
    this by creating a symlink ``model-mtp.safetensors`` -> ``mtp.safetensors``
    in the model dir so mlx_lm's glob picks the sidecar up natively. The
    symlink is zero-copy, idempotent, and LM Studio already ignores any
    safetensors not listed in the index (same as MTPLX's own layout).

Self-gating: ``_resolve_sidecar`` returns ``None`` unless the checkpoint
actually ships a sidecar, so non-MTPLX models pay only a cheap config read
+ existence check per load.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    import mlx.nn as nn
except Exception:  # pragma: no cover - mlx always present in oMLX runtime
    mx = None
    nn = None

try:
    import mlx_lm.utils as _lm_utils
    import mlx_vlm.utils as _vlm_utils
except Exception:  # pragma: no cover
    _lm_utils = None
    _vlm_utils = None

_LOCAL = threading.local()
_APPLIED = False

_MTP_BARE_PREFIX = "mtp."
_VLM_MTP_PREFIX = "language_model.mtp."

# mlx_lm's weight glob is ``model*.safetensors``; bridge the MTPLX sidecar
# into that convention via this symlink name (zero-copy, reversible).
_TEXT_SIDECAR_LINK = "model-mtp.safetensors"


def _resolve_sidecar(model_path) -> Path | None:
    """Return the MTP sidecar path for *model_path*, or ``None``.

    Honors ``mlx_lm_extra_tensors.mtp_file`` from config.json and falls back
    to the conventional ``mtp.safetensors`` / ``mtp/weights.safetensors``
    locations (MTPLX / mlx-lm forge conventions).
    """
    p = Path(model_path)
    if not p.is_dir():
        return None
    try:
        cfg = json.loads((p / "config.json").read_text())
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        return None
    extra = cfg.get("mlx_lm_extra_tensors") or {}
    candidates: list[str] = []
    if isinstance(extra, dict) and extra.get("mtp_file"):
        candidates.append(extra["mtp_file"])
    candidates.extend(["mtp.safetensors", "mtp/weights.safetensors"])
    for name in candidates:
        sp = Path(name)
        if not sp.is_absolute():
            sp = p / sp
        if sp.exists():
            return sp
    return None


def _target_mtp_prefix(model) -> str | None:
    """Prefix the MTP weights must use to bind onto *model*.

    VLM: the head lives under the language-model submodule
    (``language_model.mtp`` / ``_language_model.mtp``), matching the
    existing mlx_vlm MTP sanitize remap. Text: directly on ``model.mtp``.
    Returns ``None`` when the model has no MTP module (nothing to bind).
    """
    for attr in ("language_model", "_language_model"):
        sub = getattr(model, attr, None)
        if sub is not None and getattr(sub, "mtp", None) is not None:
            return _VLM_MTP_PREFIX
    if getattr(model, "mtp", None) is not None:
        return _MTP_BARE_PREFIX
    return None


def _as_dict(weights):
    """Normalize the weights arg (dict OR list of (name, array)) to a dict."""
    if isinstance(weights, dict):
        return weights
    # mlx-lm / mlx-vlm pass ``list(weights.items())`` to load_weights.
    return dict(weights)


def _merge_sidecar(model, weights, sidecar: Path):
    """Remap the sidecar's bare ``mtp.*`` into *weights* under the model's
    MTP prefix, dropping any bare ``mtp.*`` already present (e.g. from the
    VLM glob), and return the merged weights in the SAME form as input
    (dict or list of (name, array)) so downstream callers are unaffected.
    """
    prefix = _target_mtp_prefix(model)
    if prefix is None:
        return weights
    wdict = _as_dict(weights)
    try:
        sidecar_w = mx.load(str(sidecar))
    except Exception as e:
        logger.debug("MTPLX sidecar load failed (%s): %s", sidecar, e)
        return weights
    new = {k: v for k, v in wdict.items() if not k.startswith(_MTP_BARE_PREFIX)}
    injected = 0
    for k, v in sidecar_w.items():
        if k.startswith(_MTP_BARE_PREFIX):
            new[prefix + k[len(_MTP_BARE_PREFIX):]] = v
            injected += 1
        else:
            new.setdefault(k, v)
    if injected:
        logger.info(
            "MTPLX sidecar: injected %d %s weights from %s",
            injected,
            prefix,
            sidecar,
        )
    # Preserve the caller's expected container type.
    if isinstance(weights, dict):
        return new
    return list(new.items())


def _ensure_text_sidecar_symlink(model_path, sidecar: Path) -> None:
    """Bridge the MTPLX sidecar into mlx_lm's ``model*.safetensors`` glob.

    mlx_lm never opens ``mtp.safetensors`` (its glob is ``model*.safetensors``),
    so the text/LM path would otherwise raise "missing the mtp.* tensors" in
    the qwen35 MTP ``sanitize``. A zero-copy symlink with an mlx-lm-compatible
    name makes the sidecar discoverable natively. Idempotent and safe: it is
    only created when missing and only when it would point at our sidecar.
    """
    p = Path(model_path)
    link = p / _TEXT_SIDECAR_LINK
    target = sidecar.resolve()
    try:
        if link.is_symlink():
            if link.resolve() == target:
                return
            link.unlink()
        elif link.exists():
            # A real file with this name already exists; don't clobber it.
            return
        os.symlink(target, link)
        logger.info(
            "MTPLX sidecar: bridged %s -> %s for mlx_lm text-path discovery",
            link,
            target,
        )
    except Exception as e:
        logger.debug("MTPLX sidecar symlink bridge failed: %s", e)


def _consume_sidecar() -> Path | None:
    s = getattr(_LOCAL, "mtp_sidecar", None)
    _LOCAL.mtp_sidecar = None
    return s


def _wrap_load_model(orig):
    def _wrapped(model_path, *args, **kwargs):
        sidecar = _resolve_sidecar(model_path)
        if sidecar is not None:
            # Bridge for the text/LM path (see module docstring, Layer 2).
            _ensure_text_sidecar_symlink(model_path, sidecar)
            _LOCAL.mtp_sidecar = sidecar
        else:
            _LOCAL.mtp_sidecar = None
        try:
            return orig(model_path, *args, **kwargs)
        finally:
            _LOCAL.mtp_sidecar = None

    return _wrapped


def _wrap_load_weights(orig_method):
    def load_weights(self, weights, strict=True):
        s = _consume_sidecar()
        if s is not None:
            weights = _merge_sidecar(self, weights, s)
        return orig_method(self, weights, strict=strict)

    return load_weights


def apply() -> bool:
    """Install the MTPLX sidecar binding. Idempotent; safe no-op if mlx absent.

    Returns True iff at least one loader was patched.
    """
    global _APPLIED
    if _APPLIED:
        return True
    if mx is None or nn is None:
        return False

    patched = False
    if _lm_utils is not None and getattr(_lm_utils, "load_model", None) is not None:
        _lm_utils.load_model = _wrap_load_model(_lm_utils.load_model)
        patched = True
    if _vlm_utils is not None and getattr(_vlm_utils, "load_model", None) is not None:
        _vlm_utils.load_model = _wrap_load_model(_vlm_utils.load_model)
        patched = True

    if getattr(nn.Module, "load_weights", None) is not None:
        _orig_lw = nn.Module.load_weights
        nn.Module.load_weights = _wrap_load_weights(_orig_lw)
    else:
        patched = False

    _APPLIED = patched
    if patched:
        logger.info(
            "Patched mlx_lm/mlx_vlm load_model + nn.Module.load_weights for "
            "MTPLX MTP sidecar binding"
        )
    return patched
