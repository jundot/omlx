# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.mlx_audio_sampling (#2312).

mlx-audio TTS backends import the mx.compile'd samplers from
mlx_lm.sample_utils, so they bypass the compile-free omlx sampler that the
LLM path already uses. The patch rebinds the four affected names on
mlx_lm.sample_utils and on any already-imported mlx_audio.tts modules, so a
TTS engine start reroutes every backend to the RNG-advancing versions.
"""

from __future__ import annotations

import sys
import types

import mlx_lm.sample_utils as sample_utils
import pytest

from omlx.patches import mlx_audio_sampling
from omlx.patches.mlx_audio_sampling import (
    _ORIGINALS,
    _PATCHED_NAMES,
    ensure_uncompiled_tts_samplers,
)
from omlx.utils import sampling as omlx_sampling


@pytest.fixture(autouse=True)
def _restore_sample_utils():
    """Leave mlx_lm.sample_utils exactly as the test found it."""
    before = {name: getattr(sample_utils, name) for name in _PATCHED_NAMES}
    yield
    for name, fn in before.items():
        setattr(sample_utils, name, fn)


def test_rebinds_sample_utils_to_omlx_versions():
    for name in _PATCHED_NAMES:
        setattr(sample_utils, name, _ORIGINALS[name])

    assert ensure_uncompiled_tts_samplers() is True
    for name in _PATCHED_NAMES:
        assert getattr(sample_utils, name) is getattr(omlx_sampling, name)


def test_idempotent_second_call_changes_nothing():
    ensure_uncompiled_tts_samplers()
    assert ensure_uncompiled_tts_samplers() is False


def test_rebinds_already_imported_tts_backend_module():
    """A backend imported before the patch must be rebound in place."""
    mod_name = "mlx_audio.tts.models._omlx_fake_backend"
    fake = types.ModuleType(mod_name)
    for name in _PATCHED_NAMES:
        setattr(fake, name, _ORIGINALS[name])
    sys.modules[mod_name] = fake
    try:
        ensure_uncompiled_tts_samplers()
        for name in _PATCHED_NAMES:
            assert getattr(fake, name) is getattr(omlx_sampling, name)
    finally:
        del sys.modules[mod_name]


def test_rebinds_aliased_imports_in_backend_module():
    """higgs_audio_v3 / moss_tts alias the import (apply_top_k as
    _apply_top_k_logprobs) — the identity scan must catch those too."""
    mod_name = "mlx_audio.tts.models._omlx_fake_alias_backend"
    fake = types.ModuleType(mod_name)
    fake._apply_top_k_logprobs = _ORIGINALS["apply_top_k"]
    fake._apply_top_p_logprobs = _ORIGINALS["apply_top_p"]
    sys.modules[mod_name] = fake
    try:
        ensure_uncompiled_tts_samplers()
        assert fake._apply_top_k_logprobs is omlx_sampling.apply_top_k
        assert fake._apply_top_p_logprobs is omlx_sampling.apply_top_p
    finally:
        del sys.modules[mod_name]


def test_leaves_backend_local_samplers_untouched():
    """moss_tts-style backends define their own apply_* — identity guard
    must keep those bindings as-is."""
    mod_name = "mlx_audio.tts.models._omlx_fake_moss"
    fake = types.ModuleType(mod_name)

    def local_apply_top_k(logits, top_k):
        return logits

    fake.apply_top_k = local_apply_top_k
    sys.modules[mod_name] = fake
    try:
        ensure_uncompiled_tts_samplers()
        assert fake.apply_top_k is local_apply_top_k
    finally:
        del sys.modules[mod_name]


def test_originals_snapshot_covers_all_patched_names():
    """The identity guard depends on the snapshot existing for every name."""
    assert set(_ORIGINALS) == set(_PATCHED_NAMES)
    for name in _PATCHED_NAMES:
        assert callable(_ORIGINALS[name])


def test_installed_flag_survives_manual_unpatch():
    """A later engine start must re-apply the rebind even after something
    restored the compiled originals (e.g. a test or a dependency reload)."""
    ensure_uncompiled_tts_samplers()
    sample_utils.categorical_sampling = _ORIGINALS["categorical_sampling"]
    assert ensure_uncompiled_tts_samplers() is True
    assert sample_utils.categorical_sampling is omlx_sampling.categorical_sampling


def test_module_state_reset():
    """Reset the module _installed flag so repeated pytest runs in one
    process (e.g. pytest-xdist reuse) start from a known state."""
    mlx_audio_sampling._installed = False
    ensure_uncompiled_tts_samplers()
    assert mlx_audio_sampling._installed is True
