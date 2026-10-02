# SPDX-License-Identifier: Apache-2.0
"""Cached text requests retain gathered decode eligibility without prefill."""

from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.models.vlm import VLMModelAdapter
from omlx.scheduler import _bind_step_rope_deltas, _mark_text_positions
from tests.test_scheduler_chunked_prefill import _make_request, _make_scheduler
from tests.test_vlm_model_adapter import TestPerRequestMRoPEDecode as _MRoPEFixtures


def _adapter():
    model = _MRoPEFixtures()._make_qwen4_mrope_vlm_model()
    return VLMModelAdapter(model)


@pytest.mark.parametrize("cached_tokens", [40000, 40001])
def test_complete_text_cache_hit_keeps_gathered_decode_and_verify(cached_tokens):
    scheduler = _make_scheduler()
    request = _make_request("cached-text", 40001)
    scheduler.add_request(request)
    request.cached_tokens = cached_tokens
    request.remaining_tokens = [0]
    request.prompt_cache = [MagicMock()]
    adapter = _adapter()
    scheduler.model = adapter

    with (
        patch.object(scheduler, "_prepare_prefix_cache_for_request"),
        patch.object(scheduler, "_validate_cache", return_value=True),
        patch.object(scheduler, "_ensure_batch_generator"),
        patch.object(scheduler, "_do_external_prefill") as prefill,
    ):
        scheduled, rejected = scheduler._schedule_waiting()

    assert scheduled == [request]
    assert not rejected
    prefill.assert_not_called()
    assert request.batch_uid in adapter._uid_text_positions

    cache = MagicMock()
    cache.offset = 40000
    _bind_step_rope_deltas(adapter, mx.array([0.0]), [request.batch_uid])
    for width in (1, 4):
        adapter(mx.zeros((1, width), dtype=mx.int32), cache=[cache])
        positions = adapter._language_model.call_args.kwargs["position_ids"]
        assert positions.shape == (1, width)
        assert positions.tolist() == [list(range(40000, 40000 + width))]

    # A marked request must still use the general position form in a batch.
    cache.offset = mx.array([40000, 40000])
    _bind_step_rope_deltas(adapter, mx.array([0.0, 0.0]), [request.batch_uid, 99])
    adapter(mx.zeros((2, 1), dtype=mx.int32), cache=[cache])
    assert adapter._language_model.call_args.kwargs["position_ids"].shape == (3, 2, 1)
    adapter.unregister_rope_delta(request.batch_uid)
    assert request.batch_uid not in adapter._uid_text_positions


@pytest.mark.parametrize(
    "field,value",
    [
        ("vlm_inputs_embeds", mx.zeros((1, 1, 4))),
        ("vlm_extra_kwargs", {"image_grid_thw": [1, 2, 2]}),
        ("vlm_image_hash", "cached-image"),
        ("vlm_cache_key_ranges", [(0, "cached-image")]),
        ("rope_deltas", 1.0),
        ("cached_tokens", 0),
        ("cached_tokens", 39999),
    ],
)
def test_unproven_media_or_incomplete_cache_does_not_gain_text_eligibility(
    field, value
):
    request = _make_request("unproven", 40001)
    request.cached_tokens = 40000
    setattr(request, field, value)
    adapter = _adapter()
    _mark_text_positions(adapter, request, 7)
    assert not adapter._uid_text_positions
    cache = MagicMock()
    cache.offset = 40000
    _bind_step_rope_deltas(adapter, mx.array([0.0]), [7])
    adapter(mx.zeros((1, 4), dtype=mx.int32), cache=[cache])
    assert adapter._language_model.call_args.kwargs["position_ids"].shape == (3, 1, 4)
