# SPDX-License-Identifier: Apache-2.0
"""Reroute mlx-audio TTS sampling to omlx's compile-free samplers (#2312).

mlx-audio TTS backends (qwen3_tts, moss_tts, dia, chatterbox, ...) import
``categorical_sampling`` / ``apply_top_k`` / ``apply_top_p`` / ``apply_min_p``
from ``mlx_lm.sample_utils``, where they are decorated with
``@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)``.
Inside the omlx server that decorator stops advancing the global RNG state
after the first call — the same defect that forced omlx's own LLM sampler in
``omlx/utils/sampling.py``. A TTS sampler replaying a frozen RNG draw makes
synthesis deterministic across requests, and when the frozen draw favors the
codec EOS column the talker emits EOS at step 0: every ``/v1/audio/speech``
call then returns zero audio until the whole process restarts, because the
RNG state and the compile cache are process-global and survive a model
reload (issue #2312).

Install happens in ``TTSEngine.start()`` before mlx-audio is imported, so
each backend's module-level ``from mlx_lm.sample_utils import ...`` binds the
compile-free implementations. Backend modules imported earlier (mlx-audio
pulls sibling modules in via package imports) are rebound in place, guarded
by identity against the original compiled objects so backends that define
their own samplers under the same names (moss_tts, moss_tts_nano) are left
untouched. STT backends that build samplers through
``mlx_lm.sample_utils.make_sampler`` pick up the rerouted functions too,
since the sampler closures resolve them through module globals at call time.
"""

from __future__ import annotations

import logging
import sys

import mlx_lm.sample_utils as _sample_utils

from ..utils import sampling as _omlx_sampling

logger = logging.getLogger(__name__)

_PATCHED_NAMES = (
    "categorical_sampling",
    "apply_top_k",
    "apply_top_p",
    "apply_min_p",
)

# Snapshot the compiled originals at import time, before any rebind, so the
# in-place module scan below can tell "imported from sample_utils" apart from
# a backend's own same-named function by identity.
_ORIGINALS = {name: getattr(_sample_utils, name) for name in _PATCHED_NAMES}

_installed = False


def ensure_uncompiled_tts_samplers() -> bool:
    """Rebind mlx-lm's compiled samplers to the omlx compile-free versions.

    Idempotent; safe to call on every TTS engine start. Returns True when
    this call changed at least one binding.
    """
    global _installed

    patched = False
    for name in _PATCHED_NAMES:
        replacement = getattr(_omlx_sampling, name)
        if getattr(_sample_utils, name) is not replacement:
            setattr(_sample_utils, name, replacement)
            patched = True

    # Rebind mlx-audio TTS backend modules that imported the compiled
    # originals before this patch ran. Match by value identity, not attr
    # name: several backends alias the import (higgs_audio_v3 / moss_tts
    # bind ``apply_top_k as _apply_top_k_logprobs``), and identity also
    # protects backends that define their own samplers under these names.
    replacement_by_original = {
        id(orig): getattr(_omlx_sampling, name) for name, orig in _ORIGINALS.items()
    }
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("mlx_audio.tts."):
            continue
        for attr_name, attr_value in list(vars(mod).items()):
            replacement = replacement_by_original.get(id(attr_value))
            if replacement is not None:
                setattr(mod, attr_name, replacement)
                patched = True

    if patched and not _installed:
        logger.info(
            "mlx-audio TTS sampling rerouted to compile-free omlx samplers "
            "(issue #2312)"
        )
    _installed = True
    return patched
