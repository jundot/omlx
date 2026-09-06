"""VLM MTP adapter degrades gracefully when the inner model lacks hooks.

Regression test: qwen4_exp Lightning MTP has no ``mtp_clamp_accept`` /
``mtp_partial_rollback`` on its LanguageModel. The adapter used to raise
AttributeError, turning every partial draft rejection into a hard request
failure (batch_generator treats the clamp as an optional hook and the
rollback False as step fallback).
"""

from __future__ import annotations

import pytest


def _adapter_with(inner):
    from omlx.models.vlm import VLMModelAdapter

    adapter = VLMModelAdapter.__new__(VLMModelAdapter)
    adapter._language_model = inner
    return adapter


class _BareInner:
    pass


class _FullInner:
    def mtp_clamp_accept(self, cache, accepted, num_drafts):
        return max(0, accepted - 1)

    def mtp_partial_rollback(self, caches, accepted, num_drafts):
        return True


def test_clamp_missing_inner_returns_accepted():
    adapter = _adapter_with(_BareInner())
    assert adapter.mtp_clamp_accept(object(), 2, 3) == 2


def test_rollback_missing_inner_returns_false():
    adapter = _adapter_with(_BareInner())
    assert adapter.mtp_partial_rollback([], 0, 1) is False


def test_present_hooks_still_delegate():
    adapter = _adapter_with(_FullInner())
    assert adapter.mtp_clamp_accept(object(), 2, 3) == 1
    assert adapter.mtp_partial_rollback([], 0, 1) is True
