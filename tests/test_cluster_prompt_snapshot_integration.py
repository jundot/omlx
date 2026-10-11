# SPDX-License-Identifier: Apache-2.0
"""The telemetry patch must capture prompt-cache boundaries to SSD during
prefill and restore the longest prefix on a later miss."""

import struct
from types import SimpleNamespace

import mlx.core as mx
import mlx_lm.server as mlx_server
import pytest
from mlx_lm.models.cache import KVCache

from omlx.cluster.prompt_snapshot_cache import SSDPromptSnapshotStore
from omlx.cluster.telemetry import install_server_telemetry

STEP = 4
MODEL = "model-key"


class _Marker:
    def update(self, phase, **extra):
        return None


def _kv(steps=2):
    cache = KVCache()
    for _ in range(steps):
        k = mx.random.normal((1, 2, 1, 4))
        v = mx.random.normal((1, 2, 1, 4))
        cache.update_and_fetch(k, v)
    return [cache]


def _fake_stream_generate(*_args, **kwargs):
    """Stand in for MLX-LM: fire the progress callback at each prefill step."""

    callback = kwargs.get("prompt_progress_callback")
    total = len(kwargs.get("prompt", []))
    processed = 0
    while processed < total:
        processed = min(processed + STEP, total)
        if callback is not None:
            callback(processed, total)
    return
    yield  # make this a generator, matching stream_generate


def _install(tmp_path, monkeypatch):
    import mlx_lm.server as mlx_server

    monkeypatch.setattr(mlx_server, "stream_generate", _fake_stream_generate)
    return mlx_server, install_server_telemetry(
        _Marker(),
        ssd_cache_dir=str(tmp_path),
        prefill_step_size=STEP,
    )


def test_prefill_boundaries_are_snapshotted_to_ssd(tmp_path, monkeypatch):
    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        cache = mlx_server.LRUPromptCache()
        tokens = list(range(8))  # base 0, boundaries at 4 and 8
        cache.prefetch_nearest_cache(MODEL, tokens)
        list(
            mlx_server.stream_generate(
                model=None,
                prompt=tokens,
                prompt_cache=_kv(),
                prompt_progress_callback=None,
            )
        )
        snapshots = sorted(tmp_path.glob("*.safetensors"))

    assert len(snapshots) == 2  # one at 4 tokens, one at 8


def test_a_later_miss_restores_the_longest_ssd_prefix(tmp_path, monkeypatch):
    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx as telemetry:
        cache = mlx_server.LRUPromptCache()
        first = list(range(8))
        cache.prefetch_nearest_cache(MODEL, first)
        list(
            mlx_server.stream_generate(
                model=None,
                prompt=first,
                prompt_cache=_kv(),
                prompt_progress_callback=None,
            )
        )

        # A new request that shares the first eight tokens misses in memory and
        # is served the boundary-8 snapshot from SSD, leaving only the tail.
        longer = list(range(12))
        fresh = mlx_server.LRUPromptCache()
        restored, rest = fresh.prefetch_nearest_cache(MODEL, longer)
        snapshot = telemetry.snapshot()

    assert restored is not None
    assert rest == [8, 9, 10, 11]
    assert snapshot["cache"]["lookups"] == 2
    assert snapshot["cache"]["hits"] == 1
    assert snapshot["cache"]["tokens_reused"] == 8
    assert snapshot["cache"]["entries"] == 2
    assert snapshot["cache"]["bytes"] > 0


def test_the_fetch_path_alone_carries_the_ssd_tier(tmp_path, monkeypatch):
    """A guardless deployment never calls the preflight lookup; MLX-LM only
    calls fetch_nearest_cache, which must still capture and restore."""

    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        cache = mlx_server.LRUPromptCache()
        first = list(range(8))
        cache.fetch_nearest_cache(MODEL, first)
        list(
            mlx_server.stream_generate(
                model=None,
                prompt=first,
                prompt_cache=_kv(),
                prompt_progress_callback=None,
            )
        )

        fresh = mlx_server.LRUPromptCache()
        restored, rest = fresh.fetch_nearest_cache(MODEL, list(range(12)))

    assert restored is not None
    assert rest == [8, 9, 10, 11]


def test_an_aligned_full_hit_keeps_the_last_token_unprocessed(tmp_path, monkeypatch):
    """The pinned batched server dies inserting a request whose segments were
    all consumed, so a prompt that exactly matches its own snapshot must be
    served from the next boundary down, never with an empty rest."""

    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        cache = mlx_server.LRUPromptCache()
        exact = list(range(8))  # snapshots land at 4 and at 8 == len(prompt)
        cache.fetch_nearest_cache(MODEL, exact)
        list(
            mlx_server.stream_generate(
                model=None,
                prompt=exact,
                prompt_cache=_kv(),
                prompt_progress_callback=None,
            )
        )
        assert len(sorted(tmp_path.glob("*.safetensors"))) == 2

        restored, rest = mlx_server.LRUPromptCache().fetch_nearest_cache(MODEL, exact)

    assert restored is not None
    assert rest == [4, 5, 6, 7]  # the 8-boundary is never offered to itself


def test_a_stock_exact_hit_is_trimmed_to_leave_one_token(tmp_path, monkeypatch):
    """MLX-LM's exact-hit branch returns an empty rest; the wrapped lookup
    must hand the last token back, trimming the hit when the cache allows."""

    from mlx_lm.models.cache import ArraysCache

    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        tokens = list(range(8))
        cache = mlx_server.LRUPromptCache()
        cache.insert_cache(MODEL, tokens, _kv(steps=8))
        hit, rest = cache.fetch_nearest_cache(MODEL, tokens)
        assert hit is not None
        assert rest == [7]
        assert hit[0].offset == 7

        # A cache that cannot trim is dropped instead: full prefill beats a
        # request the server cannot insert.
        recurrent = ArraysCache(size=1)
        recurrent[0] = mx.random.normal((1, 2, 4))
        other = mlx_server.LRUPromptCache()
        other.insert_cache(MODEL, tokens, [recurrent])
        dropped, rest = other.fetch_nearest_cache(MODEL, tokens)

    assert dropped is None
    assert rest == tokens


def test_an_unaligned_base_deposits_no_snapshot(tmp_path, monkeypatch):
    """Only aligned boundaries are reusable, so an off-grid base writes nothing."""

    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        cache = mlx_server.LRUPromptCache()
        full = list(range(10))
        cache.prefetch_nearest_cache(MODEL, full)
        # Pretend three tokens were already cached: base 3 keeps every boundary
        # off the step-4 grid.
        list(
            mlx_server.stream_generate(
                model=None,
                prompt=full[3:],
                prompt_cache=_kv(),
                prompt_progress_callback=None,
            )
        )

    assert sorted(tmp_path.glob("*.safetensors")) == []


def test_the_patch_restores_stream_generate_on_exit(tmp_path, monkeypatch):
    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        assert mlx_server.stream_generate is not _fake_stream_generate
    assert mlx_server.stream_generate is _fake_stream_generate


def test_teardown_removes_the_snapshot_directory(tmp_path, monkeypatch):
    """Snapshots are process-lifetime: nothing may outlive the serving span."""

    mlx_server, ctx = _install(tmp_path, monkeypatch)
    with ctx:
        cache = mlx_server.LRUPromptCache()
        tokens = list(range(8))
        cache.fetch_nearest_cache(MODEL, tokens)
        list(
            mlx_server.stream_generate(
                model=None,
                prompt=tokens,
                prompt_cache=_kv(),
                prompt_progress_callback=None,
            )
        )
        assert sorted(tmp_path.glob("*.safetensors"))
    assert not tmp_path.exists()


def test_persistent_tier_survives_telemetry_teardown(tmp_path, monkeypatch):
    import mlx_lm.server as mlx_server

    monkeypatch.setattr(mlx_server, "stream_generate", _fake_stream_generate)
    with install_server_telemetry(
        _Marker(),
        ssd_cache_dir=str(tmp_path),
        ssd_cache_persistent=True,
        prefill_step_size=STEP,
    ):
        tokens = list(range(8))
        mlx_server.LRUPromptCache().fetch_nearest_cache(MODEL, tokens)
        list(
            mlx_server.stream_generate(
                model=None,
                prompt=tokens,
                prompt_cache=_kv(),
                prompt_progress_callback=None,
            )
        )

    assert (tmp_path / "index.json").is_file()
    assert sorted(tmp_path.glob("*.safetensors"))


class _FakeBaseBatchGenerator:
    """Report prefill progress at each step boundary, like BatchGenerator."""

    def __init__(self, *_args, **_kwargs):
        self._call = 0

    def insert_segments(self, *_args, **_kwargs):
        return [0]

    def remove(self, _uids):
        return None

    def next(self):
        self._call += 1
        total = 3 * STEP
        if self._call <= 3:
            processed = self._call * STEP
            return (
                [
                    SimpleNamespace(
                        uid=0,
                        progress=(processed, total),
                        end_of_prompt=processed == total,
                    )
                ],
                [],
            )
        return ([], [])

    def extract_cache(self, uids):
        return {uid: (_kv(), None) for uid in uids}


def test_batched_prefill_snapshots_at_each_boundary(tmp_path, monkeypatch):
    """The path these models actually use: BatchGenerator, not stream_generate."""

    import mlx_lm.server as mlx_server

    monkeypatch.setattr(mlx_server, "BatchGenerator", _FakeBaseBatchGenerator)
    with install_server_telemetry(
        _Marker(), ssd_cache_dir=str(tmp_path), prefill_step_size=STEP
    ):
        tokens = list(range(3 * STEP))
        # Setting snapshot context is the prompt cache's job on the same thread.
        mlx_server.LRUPromptCache().prefetch_nearest_cache(MODEL, tokens)
        batch = mlx_server.BatchGenerator()
        batch.insert_segments(segments=[[tokens]], all_tokens=[[]])
        while True:
            prompt_responses, gen_responses = batch.next()
            if not prompt_responses and not gen_responses:
                break
        snapshots = sorted(tmp_path.glob("*.safetensors"))

    assert len(snapshots) == 3  # STEP, 2*STEP, 3*STEP


def test_batched_capture_restores_on_a_later_batched_miss(tmp_path, monkeypatch):
    import mlx_lm.server as mlx_server

    monkeypatch.setattr(mlx_server, "BatchGenerator", _FakeBaseBatchGenerator)
    with install_server_telemetry(
        _Marker(), ssd_cache_dir=str(tmp_path), prefill_step_size=STEP
    ):
        first = list(range(3 * STEP))
        mlx_server.LRUPromptCache().prefetch_nearest_cache(MODEL, first)
        batch = mlx_server.BatchGenerator()
        batch.insert_segments(segments=[[first]], all_tokens=[[]])
        while True:
            prompt_responses, gen_responses = batch.next()
            if not prompt_responses and not gen_responses:
                break

        # A fresh request sharing 2*STEP tokens misses in memory and is served
        # the boundary snapshot from SSD.
        longer = list(range(3 * STEP)) + [999, 998]
        fresh = mlx_server.LRUPromptCache()
        restored, rest = fresh.prefetch_nearest_cache(MODEL, longer)

    assert restored is not None
    assert rest == [999, 998]


# Rejected memory hits must not hide usable rank-local SSD snapshots.

TOKENS = list(range(10))


class _Agreement:
    def __init__(self, peer_plans):
        self.peer_plans = iter(peer_plans)
        self.local_plans = []

    def broadcast_owned_bytes(self, payload, *, source_rank, expected_size):
        assert expected_size == 24
        if source_rank == 0:
            self.local_plans.append(struct.unpack("!QQQ", payload))
            return payload
        return struct.pack("!QQQ", *next(self.peer_plans))


def _install_rank0(monkeypatch, tmp_path, memory, rest, peers, *, ssd=True, votes=2):
    if ssd:
        store = SSDPromptSnapshotStore(tmp_path, step=STEP, persistent=True)
        assert store.put(MODEL, TOKENS[:4], _kv(4))
        assert store.put(MODEL, TOKENS[:8], _kv(8))
    monkeypatch.setattr(
        mx.distributed, "init", lambda: SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    )
    collective_calls = []

    def all_sum(value):
        collective_calls.append(value.tolist())
        return value * votes

    monkeypatch.setattr(mx.distributed, "all_sum", all_sum)
    monkeypatch.setattr(
        mlx_server.LRUPromptCache, "fetch_nearest_cache", lambda *args: (memory, rest)
    )
    agreement = _Agreement(peers)
    context = install_server_telemetry(
        _Marker(),
        heartbeat_interval=0,
        control_plane=agreement,
        ssd_cache_dir=str(tmp_path) if ssd else None,
        ssd_cache_persistent=True,
        prefill_step_size=STEP,
    )
    return context, agreement, collective_calls


@pytest.mark.parametrize("local_hit", [True, False])
def test_memory_disagreement_retries_shared_ssd(monkeypatch, tmp_path, local_hit):
    memory, rest = (_kv(4), TOKENS[4:]) if local_hit else (None, TOKENS)
    peer = (0, 10, 0) if local_hit else (4, 6, 0)
    ctx, agreement, calls = _install_rank0(
        monkeypatch, tmp_path, memory, rest, [peer, (8, 2, 0)]
    )
    with ctx as telemetry:
        hit, remaining = mlx_server.LRUPromptCache().fetch_nearest_cache(MODEL, TOKENS)
        assert hit[0].offset == 8
        assert remaining == TOKENS[8:]
        assert telemetry.snapshot()["cache"]["tokens_reused"] == 8
    assert len(agreement.local_plans) == 2
    assert calls == [[1, 1]]


def test_invalid_memory_offset_retries_ssd(monkeypatch, tmp_path):
    ctx, _, calls = _install_rank0(
        monkeypatch, tmp_path, _kv(9), TOKENS[8:], [(8, 2, 1), (8, 2, 0)]
    )
    with ctx:
        hit, rest = mlx_server.LRUPromptCache().fetch_nearest_cache(MODEL, TOKENS)
        assert hit[0].offset == 8
        assert rest == TOKENS[8:]
    assert calls == [[1, 1]]


def test_valid_shared_memory_hit_skips_ssd(monkeypatch, tmp_path):
    ctx, _, calls = _install_rank0(
        monkeypatch, tmp_path, _kv(8), TOKENS[8:], [(8, 2, 0)]
    )
    with ctx:
        hit, rest = mlx_server.LRUPromptCache().fetch_nearest_cache(MODEL, TOKENS)
        assert hit[0].offset == 8
        assert rest == TOKENS[8:]
    assert calls == []


@pytest.mark.parametrize("failure", ["local_load", "peer_load", "no_common_boundary"])
def test_ssd_failure_returns_synchronized_full_prefill(monkeypatch, tmp_path, failure):
    peers = [(8, 2, 1)]
    if failure != "no_common_boundary":
        peers.append((0, 10, 0) if failure == "peer_load" else (8, 2, 0))
    ctx, agreement, _ = _install_rank0(
        monkeypatch,
        tmp_path,
        _kv(9),
        TOKENS[8:],
        peers,
        votes=1 if failure == "no_common_boundary" else 2,
    )
    if failure == "local_load":
        monkeypatch.setattr(SSDPromptSnapshotStore, "load", lambda *args: None)
    with ctx as telemetry:
        hit, rest = mlx_server.LRUPromptCache().fetch_nearest_cache(MODEL, TOKENS)
        assert hit is None
        assert rest == TOKENS
        assert telemetry.snapshot()["cache"]["tokens_reused"] == 0
    assert len(agreement.local_plans) == (1 if failure == "no_common_boundary" else 2)


def test_disabled_ssd_keeps_safe_memory_rejection(monkeypatch, tmp_path):
    ctx, _, calls = _install_rank0(
        monkeypatch, tmp_path, _kv(9), TOKENS[8:], [(8, 2, 1)], ssd=False
    )
    with ctx:
        hit, rest = mlx_server.LRUPromptCache().fetch_nearest_cache(MODEL, TOKENS)
        assert hit is None and rest == TOKENS
    assert calls == []


# Segment boundaries must not prevent later SSD prefix snapshots.


class _GridModel:
    layers = [None]

    def __call__(self, tokens, cache):
        n, length = tokens.shape
        cache[0].update_and_fetch(
            mx.ones((n, 1, length, 2)), mx.ones((n, 1, length, 2))
        )
        return mx.zeros((n, length, 32))

    def __repr__(self):
        return "cache-grid-test-model"


def _single_rank(monkeypatch):
    monkeypatch.setattr(
        mx.distributed, "init", lambda: SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    )


def _grid_telemetry(tmp_path):
    return install_server_telemetry(
        _Marker(),
        heartbeat_interval=0,
        prefill_step_size=STEP,
        ssd_cache_dir=str(tmp_path),
        ssd_cache_persistent=True,
    )


@pytest.mark.parametrize("segments", [[3, 10], [5, 3, 5], [4, 9]])
def test_segmented_prompt_saves_all_grid_boundaries(monkeypatch, tmp_path, segments):
    _single_rank(monkeypatch)
    model = _GridModel()
    tokens = list(range(sum(segments)))
    with _grid_telemetry(tmp_path):
        mlx_server.LRUPromptCache().fetch_nearest_cache(model, tokens)
        batch = mlx_server.BatchGenerator(
            model,
            max_tokens=1,
            prefill_step_size=STEP,
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
            assert batch.prefill_step_size == STEP
            if generated:
                break
        else:
            pytest.fail("generation did not complete")
        batch.close()
    store = SSDPromptSnapshotStore(tmp_path, step=STEP, persistent=True)
    assert store.present_boundaries(model, tokens) == (12, 8, 4)
    cache = store.load(model, tokens, boundary=12)
    assert cache[0].offset == 12


def test_queued_unaligned_prefix_does_not_shrink_active_prefill(monkeypatch, tmp_path):
    _single_rank(monkeypatch)
    model = _GridModel()
    with _grid_telemetry(tmp_path):
        mlx_server.LRUPromptCache().fetch_nearest_cache(model, list(range(13)))
        batch = mlx_server.BatchGenerator(
            model,
            max_tokens=1,
            prefill_step_size=STEP,
            completion_batch_size=1,
            prefill_batch_size=1,
        )
        batch.insert_segments(segments=[[list(range(13))]], all_tokens=[[]])
        batch.insert_segments(segments=[[list(range(3, 13))]], all_tokens=[[0, 1, 2]])
        responses, _ = batch.next()
        assert responses[0].progress[0] == 4
        responses, _ = batch.next()
        assert responses[0].progress[0] == 8
        assert batch.prefill_step_size == STEP
        batch.close()


@pytest.mark.parametrize(
    "case", ["completion_capacity", "terminal_token", "freed_capacity"]
)
def test_only_admitted_nonterminal_prompts_limit_step(monkeypatch, tmp_path, case):
    _single_rank(monkeypatch)
    model = _GridModel()
    with _grid_telemetry(tmp_path):
        batch = mlx_server.BatchGenerator(
            model,
            max_tokens=1 if case == "freed_capacity" else 10,
            prefill_step_size=STEP,
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
        assert batch.prefill_step_size == STEP
        batch.close()
