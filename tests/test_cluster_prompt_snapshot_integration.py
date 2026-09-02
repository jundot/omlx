# SPDX-License-Identifier: Apache-2.0
"""The telemetry patch must capture prompt-cache boundaries to SSD during
prefill and restore the longest prefix on a later miss."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

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
    with ctx:
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

    assert restored is not None
    assert rest == [8, 9, 10, 11]


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


@pytest.mark.parametrize("fail_generation", [False, True])
def test_vision_digest_key_lives_through_request_cache_then_is_cleared(
    monkeypatch,
    fail_generation,
):
    from dataclasses import dataclass

    from omlx.cluster import deepseek_v4_vision_runtime
    from omlx.cluster.deepseek_v4_vision_runtime import (
        install_deepseek_v4_vision_runtime,
    )

    prepared = SimpleNamespace(
        patches=b"pixels",
        n_vit_h=2,
        n_vit_w=2,
        types=(0, 1, 2),
    )
    monkeypatch.setattr(
        deepseek_v4_vision_runtime,
        "prepare_token_ids",
        lambda *_args, **_kwargs: ([1, 2, 3, 4], (prepared,)),
    )
    cache_keys = []
    stream_steps = []

    class Model:
        def __init__(self):
            self.inputs = []

        def set_vision_inputs(self, images):
            self.inputs.append(images)

    class Cache:
        def fetch_nearest_cache(self, model_key, prompt):
            cache_keys.append(("fetch", model_key))
            return None, prompt

        def insert_cache(self, model_key, _prompt, _cache):
            cache_keys.append(("insert", model_key))

    class ResponseGenerator:
        def __init__(self):
            self.model_provider = provider
            self.prompt_cache = Cache()

        def _share_request(self, request):
            return request

        def _tokenize(self, _tokenizer, _request, _args):
            return [99], [[99]], ["assistant"], "normal"

        def _serve_single(self, request):
            _queue, request_payload, request_args = request
            prompt = self._tokenize(
                self.model_provider.tokenizer,
                request_payload,
                request_args,
            )[0]
            cache, rest = self.prompt_cache.fetch_nearest_cache(
                self.model_provider.model_key,
                prompt,
            )
            list(
                server.stream_generate(
                    prompt=rest,
                    prompt_cache=cache,
                    prefill_step_size=2,
                )
            )
            self.prompt_cache.insert_cache(
                self.model_provider.model_key,
                prompt,
                cache,
            )

    def stream_generate(*_args, **kwargs):
        stream_steps.append(kwargs["prefill_step_size"])
        if fail_generation:
            raise RuntimeError("generation failed")
        yield "token"

    server = SimpleNamespace(
        ResponseGenerator=ResponseGenerator,
        stream_generate=stream_generate,
    )
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda _token: 99,
        unk_token_id=-1,
    )
    provider = SimpleNamespace(
        model=Model(),
        model_key=("canonical", None, None),
        tokenizer=tokenizer,
        is_batchable=True,
    )

    @dataclass
    class Request:
        messages: list

    request = Request(
        [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "image"}}],
            }
        ],
    )

    with install_deepseek_v4_vision_runtime(server, provider, config={}, rank=0):
        generator = ResponseGenerator()
        shared_request = generator._share_request(
            (object(), request, SimpleNamespace())
        )
        if fail_generation:
            with pytest.raises(RuntimeError, match="generation failed"):
                generator._serve_single(shared_request)
        else:
            generator._serve_single(shared_request)

        assert provider.model_key == ("canonical", None, None)
        assert provider.model.inputs[-1] is None

    assert stream_steps == [4]
    assert [len(key) for _, key in cache_keys] == ([5] if fail_generation else [5, 5])
    assert {key[-2] for _, key in cache_keys} == {"vision"}
