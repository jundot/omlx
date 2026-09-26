# SPDX-License-Identifier: Apache-2.0
"""A remote prefill's pages cross the link intact and decode exactly like a local prefill."""

from __future__ import annotations

import time

import mlx.core as mx
import pytest
from kv_handoff_fakes import (
    FakeProducer,
    build_model,
    export_rank,
    frames_of,
    greedy,
    kv,
    manifest,
    prefill,
)
from mlx_lm.models.cache import make_prompt_cache
from rdma_loopback import LoopbackLink, PythonWordOps

from omlx.cluster.rdma.mailbox import ClientMailbox
from omlx.remote_prefill import job as job_module
from omlx.remote_prefill.inject import extend_caches, layer_updates
from omlx.remote_prefill.job import PrefillJob
from omlx.remote_prefill.settings import RemotePrefillSettings

PROMPT = [(7 * index + 3) % 250 + 1 for index in range(70)]
END = 60


@pytest.fixture(scope="module")
def model():
    return build_model()


@pytest.fixture
def links():
    made = []

    def make(count=1):
        for _ in range(count):
            made.append(LoopbackLink(request_bytes=64 * 1024, reply_bytes=64 * 1024))
        return made[-count:]

    yield make
    for link in made:
        link.close()


def _settings(links) -> RemotePrefillSettings:
    return RemotePrefillSettings(
        url="http://prefill.invalid:8000",
        model="tiny",
        local_model="tiny-mlx",
        links=tuple(link.name for link in links),
        timeout_s=30,
    )


def _job(links, tokens, export_from=0, calls=None):
    job = PrefillJob(
        mx,
        _settings(links),
        tokens,
        export_from,
        attach=lambda name: ClientMailbox.attach(name, PythonWordOps()),
        requester=lambda *args: calls.append(args) if calls is not None else None,
    )
    job.start()
    job.wait(30)
    return job


def _serve(link, local, first, *, rank=0, ranks=1, rows_per_frame=2, **options):
    layers, pages = export_rank(local, first, END, rank=rank, ranks=ranks)
    frames = frames_of(pages, rows_per_frame)
    body = manifest(PROMPT[:END], first, layers, len(frames), rank=rank, ranks=ranks)
    return FakeProducer(link, body, frames, **options)


def test_decode_from_a_remote_prefill_matches_a_local_prefill(model, links):
    local = prefill(model, PROMPT[:END])
    (link,) = links()
    producer = _serve(link, local, 0, waits=2)
    calls = []
    job = _job([link], PROMPT[:END], calls=calls)
    producer.join()
    assert job.state == "done", job.error
    assert producer.closed and producer.checksum is True and producer.error is None
    (call,) = calls
    assert call[1] == PROMPT[:END] and call[2] == job.handoff.hex() and call[3] == 0
    caches = make_prompt_cache(model)
    extend_caches(mx, caches, layer_updates(mx, job.result, 0, END, mx.bfloat16))
    for mine, theirs in zip(caches, local):
        assert mine.offset == END
        assert all(mx.array_equal(a, b).item() for a, b in zip(kv(mine), kv(theirs)))
    reference = greedy(model, prefill(model, PROMPT[:END]), PROMPT[END:], 8)
    assert greedy(model, caches, PROMPT[END:], 8) == reference


def test_two_ranks_merge_their_heads_in_order(model, links):
    local = prefill(model, PROMPT[:END])
    pair = links(2)
    producers = [
        _serve(link, local, 0, rank=rank, ranks=2) for rank, link in enumerate(pair)
    ]
    job = _job(pair, PROMPT[:END])
    for producer in producers:
        producer.join()
    assert job.state == "done", job.error
    caches = make_prompt_cache(model)
    extend_caches(mx, caches, layer_updates(mx, job.result, 0, END, mx.bfloat16))
    for mine, theirs in zip(caches, local):
        assert all(mx.array_equal(a, b).item() for a, b in zip(kv(mine), kv(theirs)))


def test_a_local_prefix_is_extended_with_the_remote_rest(model, links):
    local = prefill(model, PROMPT[:END])
    (link,) = links()
    producer = _serve(link, local, 32)
    job = _job([link], PROMPT[:END], export_from=32)
    producer.join()
    assert job.state == "done", job.error
    caches = prefill(model, PROMPT[:32])
    extend_caches(mx, caches, layer_updates(mx, job.result, 32, END, mx.bfloat16))
    reference = greedy(model, prefill(model, PROMPT[:END]), PROMPT[END:], 8)
    assert greedy(model, caches, PROMPT[END:], 8) == reference


def test_a_flipped_byte_fails_the_handoff_and_frees_the_pages(model, links):
    local = prefill(model, PROMPT[:END])
    (link,) = links()
    producer = _serve(link, local, 0, flip_frame=1)
    job = _job([link], PROMPT[:END])
    producer.join()
    assert job.state == "failed"
    assert job.error == "frame 1 failed its checksum"
    assert producer.closed


def test_a_refused_handoff_carries_the_producers_reason(model, links):
    local = prefill(model, PROMPT[:END])
    (link,) = links()
    producer = _serve(link, local, 0, refuse="unknown handoff")
    job = _job([link], PROMPT[:END])
    producer.join()
    assert job.state == "failed"
    assert job.error == "the producer refused the handoff: unknown handoff"


def test_pages_of_another_prompt_are_refused(model, links):
    local = prefill(model, PROMPT[:END])
    (link,) = links()
    layers, pages = export_rank(local, 0, END)
    frames = frames_of(pages, 4)
    other = [token + 1 for token in PROMPT[:END]]
    producer = FakeProducer(link, manifest(other, 0, layers, len(frames)), frames)
    job = _job([link], PROMPT[:END])
    producer.join()
    assert job.state == "failed"
    assert job.error == "the producer prefilled a different prompt"


def test_a_link_that_is_down_fails_before_vllm_is_asked(links):
    (link,) = links()
    link.drop()
    calls = []
    job = _job([link], PROMPT[:END], calls=calls)
    assert job.state == "failed" and job.error == f"RDMA link {link.name} is down"
    assert calls == []


def test_a_link_without_a_producer_fails_fast_before_vllm_is_asked(links, monkeypatch):
    monkeypatch.setattr(job_module, "_PRODUCER_CHECK_S", 0.5)
    (link,) = links()
    calls = []
    began = time.monotonic()
    job = _job([link], PROMPT[:END], calls=calls)
    assert job.state == "failed" and calls == []
    assert job.error.startswith(f"no MCDMA connector answered on {link.name}")
    assert job.pause and time.monotonic() - began < 5


def test_an_export_that_never_appears_after_vllm_answers_fails_fast(
    model, links, monkeypatch
):
    monkeypatch.setattr(job_module, "_READY_S", 0.5)
    local = prefill(model, PROMPT[:END])
    (link,) = links()
    # The connector answers, but the export for this prompt never shows up.
    producer = _serve(link, local, 0, waits=1000, idle_s=2)
    began = time.monotonic()
    job = _job([link], PROMPT[:END])
    producer.join()
    assert job.state == "failed" and job.pause
    assert job.error == "the producer never had the handoff ready"
    assert time.monotonic() - began < 5
