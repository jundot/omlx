import time

from omlx.prefill_progress import PrefillProgressTracker


def test_prefill_progress_resets_elapsed_when_phase_changes(monkeypatch):
    tracker = PrefillProgressTracker()
    times = iter([100.0, 110.0, 112.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(times))

    tracker.update("req", 10, 100, "model", phase="specprefill_scoring")
    tracker.update("req", 0, 50, "model", phase="specprefill_sparse")

    progress = tracker.get_model_progress("model")[0]
    assert progress["phase"] == "specprefill_sparse"
    assert progress["elapsed"] == 2.0


def test_prefill_progress_extra_cannot_clobber_base_fields():
    tracker = PrefillProgressTracker()

    tracker.update(
        "req",
        10,
        100,
        "model",
        phase="prefill",
        extra={
            "processed": 999,
            "phase": "wrong",
            "start_time": -1,
            "selected_tokens": 5,
        },
    )

    progress = tracker.get_model_progress("model")[0]
    assert progress["processed"] == 10
    assert progress["phase"] == "prefill"
    assert progress["selected_tokens"] == 5
