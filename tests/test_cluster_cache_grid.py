# SPDX-License-Identifier: Apache-2.0
"""Segment boundaries must not prevent later SSD prefix snapshots."""

from types import SimpleNamespace

import mlx.core as mx
import mlx_lm.server as server
import pytest

from omlx.cluster.prompt_snapshot_cache import SSDPromptSnapshotStore
from omlx.cluster.telemetry import install_server_telemetry


class Model:
    layers = [None]

    def __call__(self, tokens, cache):
        n, length = tokens.shape
        cache[0].update_and_fetch(
            mx.ones((n, 1, length, 2)), mx.ones((n, 1, length, 2))
        )
        return mx.zeros((n, length, 32))

    def __repr__(self):
        return "cache-grid-test-model"


@pytest.mark.parametrize("segments", [[3, 10], [5, 3, 5], [4, 9]])
def test_segmented_prompt_saves_all_grid_boundaries(monkeypatch, tmp_path, segments):
    monkeypatch.setattr(
        mx.distributed, "init", lambda: SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    )
    model = Model()
    tokens = list(range(sum(segments)))
    marker = SimpleNamespace(update=lambda *a, **k: None)
    with install_server_telemetry(
        marker,
        heartbeat_interval=0,
        prefill_step_size=4,
        ssd_cache_dir=str(tmp_path),
        ssd_cache_persistent=True,
    ):
        server.LRUPromptCache().fetch_nearest_cache(model, tokens)
        batch = server.BatchGenerator(
            model,
            max_tokens=1,
            prefill_step_size=4,
            completion_batch_size=1,
            prefill_batch_size=1,
        )
        parts = []
        offset = 0
        for size in segments:
            parts.append(tokens[offset : offset + size])
            offset += size
        batch.insert_segments(segments=[parts], all_tokens=[[]])
        for _ in range(20):
            _, generated = batch.next()
            assert batch.prefill_step_size == 4
            if generated:
                break
        else:
            pytest.fail("generation did not complete")
        batch.close()
    store = SSDPromptSnapshotStore(tmp_path, step=4, persistent=True)
    assert store.present_boundaries(model, tokens) == (12, 8, 4)
    cache = store.load(model, tokens, boundary=12)
    assert cache[0].offset == 12


def test_queued_unaligned_prefix_does_not_shrink_active_prefill(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mx.distributed, "init", lambda: SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    )
    marker = SimpleNamespace(update=lambda *a, **k: None)
    model = Model()
    with install_server_telemetry(
        marker,
        heartbeat_interval=0,
        prefill_step_size=4,
        ssd_cache_dir=str(tmp_path),
        ssd_cache_persistent=True,
    ):
        server.LRUPromptCache().fetch_nearest_cache(model, list(range(13)))
        batch = server.BatchGenerator(
            model,
            max_tokens=1,
            prefill_step_size=4,
            completion_batch_size=1,
            prefill_batch_size=1,
        )
        batch.insert_segments(segments=[[list(range(13))]], all_tokens=[[]])
        batch.insert_segments(segments=[[list(range(3, 13))]], all_tokens=[[0, 1, 2]])
        responses, _ = batch.next()
        assert responses[0].progress[0] == 4
        responses, _ = batch.next()
        assert responses[0].progress[0] == 8
        assert batch.prefill_step_size == 4
        batch.close()


@pytest.mark.parametrize(
    "case", ["completion_capacity", "terminal_token", "freed_capacity"]
)
def test_only_admitted_nonterminal_prompts_limit_step(monkeypatch, tmp_path, case):
    monkeypatch.setattr(
        mx.distributed, "init", lambda: SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    )
    marker = SimpleNamespace(update=lambda *a, **k: None)
    model = Model()
    with install_server_telemetry(
        marker,
        heartbeat_interval=0,
        prefill_step_size=4,
        ssd_cache_dir=str(tmp_path),
        ssd_cache_persistent=True,
    ):
        batch = server.BatchGenerator(
            model,
            max_tokens=1 if case == "freed_capacity" else 10,
            prefill_step_size=4,
            completion_batch_size=2,
            prefill_batch_size=2,
        )
        if case in ("completion_capacity", "freed_capacity"):
            batch.insert_segments(segments=[[[0]]], all_tokens=[[]])
            batch.next()
        else:
            batch.insert_segments(segments=[[[3]]], all_tokens=[[0, 1, 2]])
        uid = batch.insert_segments(segments=[[list(range(13))]], all_tokens=[[]])[0]
        if case in ("completion_capacity", "freed_capacity"):
            batch.insert_segments(
                segments=[[list(range(3, 13))]], all_tokens=[[0, 1, 2]]
            )
        responses, _ = batch.next()
        response = next(response for response in responses if response.uid == uid)
        assert response.progress[0] == (1 if case == "freed_capacity" else 4)
        assert batch.prefill_step_size == 4
        batch.close()
