from unittest.mock import patch

import pytest

from omlx.engine.base import BaseNonStreamingEngine


class DummyNonStreamingEngine(BaseNonStreamingEngine):
    @property
    def model_name(self):
        return "dummy"

    async def start(self):
        pass

    async def stop(self):
        pass

    def get_stats(self):
        return {}


def test_non_streaming_activity_tracks_multiple_operations_and_reserved_keys():
    engine = DummyNonStreamingEngine()
    first = engine._begin_activity(
        "embedding",
        detail="Embedding",
        metadata={"started_at": -1, "input_count": 2},
    )
    second = engine._begin_activity("transcribing", detail="Transcribing")

    snapshot = engine.get_activity_snapshot()
    assert snapshot["active_requests"] == 2
    assert len(snapshot["activities"]) == 2
    first_activity = next(a for a in snapshot["activities"] if a["request_id"] == first)
    assert first_activity["input_count"] == 2
    assert first_activity["elapsed_seconds"] >= 0

    engine._end_activity(first)
    engine._end_activity(second)
    assert engine._active_count == 0


def test_non_streaming_activity_double_end_raises():
    engine = DummyNonStreamingEngine()
    activity_id = engine._begin_activity("embedding")
    engine._end_activity(activity_id)

    with pytest.raises(RuntimeError):
        engine._end_activity(activity_id)


@pytest.mark.asyncio
async def test_finish_activity_clears_mlx_cache_unconditionally():
    """`_finish_activity` must clear the Metal pool on every call (#684).

    Previous implementation only cleared when `_active_count == 0`, which
    never triggered under steady concurrent loads (RAG indexing).
    """
    engine = DummyNonStreamingEngine()

    with patch("omlx.engine.base.mx") as mock_mx:
        # Two overlapping activities — neither drains active_count to 0
        # before the other finishes, but both must still clear.
        a = engine._begin_activity("embedding")
        b = engine._begin_activity("embedding")

        await engine._finish_activity(a)
        await engine._finish_activity(b)

        assert mock_mx.synchronize.call_count == 2
        assert mock_mx.clear_cache.call_count == 2
        assert engine._active_count == 0
