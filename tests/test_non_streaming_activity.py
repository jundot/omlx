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

    assert engine._end_activity(first) is False
    assert engine._end_activity(second) is True


def test_non_streaming_activity_double_end_raises():
    engine = DummyNonStreamingEngine()
    activity_id = engine._begin_activity("embedding")
    engine._end_activity(activity_id)

    with pytest.raises(RuntimeError):
        engine._end_activity(activity_id)
