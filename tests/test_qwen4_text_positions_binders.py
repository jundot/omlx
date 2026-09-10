# SPDX-License-Identifier: Apache-2.0
"""Scheduler and MTP binders carry the text-only position proof to the VLM adapter."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

import omlx.scheduler as scheduler
from omlx.patches.mlx_lm_mtp import batch_generator


class _Adapter:
    """Records the seams the binders call, like VLMModelAdapter after the text-proof change."""

    def __init__(self):
        self._uses_mrope = True
        self._uid_rope_deltas = {7: 3.0}
        self.marked = []
        self.steps = []
        self.batches = []

    def set_text_prefill_rope_delta(self, delta):
        self.batches.append(("text", float(delta)))

    def mark_text_positions(self, uid):
        self.marked.append(uid)

    def set_step_rope_deltas(self, deltas, uids):
        self.steps.append((deltas.tolist(), list(uids)))

    def set_batch_rope_deltas(self, deltas):
        self.batches.append(("generic", deltas.tolist()))


class _LegacyAdapter:
    """A wrapper without the new seams keeps working through the generic binder."""

    def __init__(self):
        self._uses_mrope = True
        self._uid_rope_deltas = {7: 3.0}
        self.batches = []

    def set_batch_rope_deltas(self, deltas):
        self.batches.append(deltas.tolist())


def test_mark_text_positions_uses_the_batch_uid_of_a_proven_request():
    adapter = _Adapter()
    scheduler._mark_text_positions(adapter, SimpleNamespace(text_positions_proven=True), 0)
    scheduler._mark_text_positions(adapter, SimpleNamespace(), 1)  # never proven
    scheduler._mark_text_positions(adapter, SimpleNamespace(text_positions_proven=False), 2)
    assert adapter.marked == [0]


def test_mark_text_positions_tolerates_wrappers_without_the_seam():
    legacy = _LegacyAdapter()
    scheduler._mark_text_positions(legacy, SimpleNamespace(text_positions_proven=True), 0)
    assert legacy.batches == []


def test_step_binder_prefers_uid_aware_seam_and_falls_back():
    adapter = _Adapter()
    scheduler._bind_step_rope_deltas(adapter, mx.array([3.0]), [7])
    assert adapter.steps == [([3.0], [7])] and adapter.batches == []

    legacy = _LegacyAdapter()
    scheduler._bind_step_rope_deltas(legacy, mx.array([3.0]), [7])
    assert legacy.batches == [[3.0]]


def test_mtp_singleton_binder_passes_the_uid():
    adapter = _Adapter()
    batch_generator._set_singleton_mrope_delta(SimpleNamespace(model=adapter, uids=[7]))
    assert adapter.steps == [([3.0], [7])]

    legacy = _LegacyAdapter()
    batch_generator._set_singleton_mrope_delta(SimpleNamespace(model=legacy, uids=[7]))
    assert legacy.batches == [[3.0]]


def test_mtp_proxy_exposes_uid_aware_seams_for_class_level_lookup():
    """The binders look the seams up on the class, so the MTP proxy must define them, not just delegate."""
    from omlx.speculative.vlm_mtp import _VLMAdapterMTPProxy

    adapter = _Adapter()
    proxy = _VLMAdapterMTPProxy(adapter, language_model=object())

    scheduler._bind_step_rope_deltas(proxy, mx.array([3.0]), [7])
    batch_generator._set_singleton_mrope_delta(SimpleNamespace(model=proxy, uids=[7]))
    assert adapter.steps == [([3.0], [7]), ([3.0], [7])]
    assert adapter.batches == []

    scheduler._mark_text_positions(proxy, SimpleNamespace(text_positions_proven=True), 7)
    assert adapter.marked == [7]


def test_mtp_proxy_without_adapter_seams_falls_back_to_generic_binder():
    from omlx.speculative.vlm_mtp import _VLMAdapterMTPProxy

    legacy = _LegacyAdapter()
    proxy = _VLMAdapterMTPProxy(legacy, language_model=object())
    scheduler._bind_step_rope_deltas(proxy, mx.array([3.0]), [7])
    batch_generator._set_singleton_mrope_delta(SimpleNamespace(model=proxy, uids=[7]))
    assert legacy.batches == [[3.0], [3.0]]
