from __future__ import annotations

import json
from queue import Queue
from types import SimpleNamespace

from omlx.cluster.disaggregated_server_worker import (
    _CANCEL,
    _PROGRESS,
    _PhaseCacheMaintenance,
    _PhasePromptCache,
    _PhaseResponseGenerator,
    _ServingRequest,
)


class _Control:
    def __init__(self, progress: tuple[int, int]):
        self.progress = progress
        self.calls = []

    def broadcast_owned_bytes(self, payload, *, source_rank, expected_size):
        self.calls.append((payload, source_rank, expected_size))
        if payload is None:
            return _PROGRESS.pack(*self.progress)
        return payload


class _Context:
    def __init__(self):
        self._should_stop = False

    def stop(self):
        self._should_stop = True


class _Telemetry:
    def __init__(self):
        self.progress = []

    def observe_prefill_progress(self, request_id, **progress):
        self.progress.append((request_id, progress))


def _request(progress):
    return _ServingRequest(
        request_id=7,
        prompt=[1, 2],
        args=SimpleNamespace(),
        context=_Context(),
        state_machine=None,
        output=Queue(),
        progress=progress,
    )


def test_prefill_progress_acknowledges_continue_to_prefill_rank():
    generator = _PhaseResponseGenerator.__new__(_PhaseResponseGenerator)
    generator.control = _Control((2048, 4096))
    generator.prefill_rank = 1
    generator.decode_rank = 0
    generator.telemetry = _Telemetry()
    request = _request(lambda *_: None)

    generator._recv_progress(request)

    assert generator.control.calls[-1] == (_CANCEL.pack(0), 0, _CANCEL.size)
    assert generator.telemetry.progress == [
        (7, {"processed_tokens": 2048, "total_tokens": 4096})
    ]


def test_broken_client_cancels_prefill_at_next_chunk_boundary():
    generator = _PhaseResponseGenerator.__new__(_PhaseResponseGenerator)
    generator.control = _Control((2048, 8192))
    generator.prefill_rank = 1
    generator.decode_rank = 0
    generator.telemetry = _Telemetry()

    def disconnected(*_args):
        raise BrokenPipeError("client disconnected")

    request = _request(disconnected)
    generator._recv_progress(request)

    assert request.context._should_stop is True
    assert generator.control.calls[-1] == (_CANCEL.pack(1), 0, _CANCEL.size)


def test_phase_hot_cache_reuses_exact_cache_and_logits():
    class Cache:
        nbytes = 16

        def is_trimmable(self):
            return False

    manager = _PhasePromptCache(
        model=object(),
        model_key="model-a",
        max_size=2,
        max_bytes=None,
        ssd_directory=None,
        ssd_max_bytes=1024,
        step=2048,
    )
    logits = object()
    manager.insert([1, 2, 3], [Cache()], logits)

    cache, rest, restored_logits, tier = manager.lookup([1, 2, 3])

    assert len(cache) == 1
    assert rest == []
    assert restored_logits is logits
    assert tier == "memory"


def test_phase_hot_cache_keeps_two_alternating_conversations():
    class Cache:
        nbytes = 16

        def is_trimmable(self):
            return False

    manager = _PhasePromptCache(
        model=object(),
        model_key="model-a",
        max_size=2,
        max_bytes=None,
        ssd_directory=None,
        ssd_max_bytes=1024,
        step=2048,
    )
    first_logits = object()
    second_logits = object()
    manager.insert([1, 2, 3], [Cache()], first_logits)
    manager.insert([4, 5, 6], [Cache()], second_logits)

    first = manager.lookup([1, 2, 3])
    second = manager.lookup([4, 5, 6])

    assert first[1:] == ([], first_logits, "memory")
    assert second[1:] == ([], second_logits, "memory")


def test_phase_cache_maintenance_uses_existing_rank_ack_protocol(tmp_path):
    class Cache:
        def __init__(self):
            self.calls = []

        def clear(self, *, hot, ssd):
            self.calls.append((hot, ssd))
            return {"hot_cleared": 2, "ssd_deleted": 3}

    cache = Cache()
    maintenance = _PhaseCacheMaintenance(
        prompt_cache=cache,
        state_dir=str(tmp_path),
        deployment_id="phase-a",
        plan_hash="f" * 64,
        rank=1,
    )
    maintenance.epoch_floor = 0
    maintenance.last_epoch = 0
    maintenance.request_path.write_text(
        json.dumps(
            {
                "epoch": 7,
                "deployment_id": "phase-a",
                "plan_hash": "f" * 64,
                "hot": True,
                "ssd": True,
            }
        ),
        encoding="utf-8",
    )

    maintenance.poll()

    assert cache.calls == [(True, True)]
    assert json.loads(maintenance.ack_path.read_text(encoding="utf-8")) == {
        "epoch": 7,
        "hot_cleared": 2,
        "rank": 1,
        "ssd_deleted": 3,
        "status": "ok",
    }
