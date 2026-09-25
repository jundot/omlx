# SPDX-License-Identifier: Apache-2.0
"""The scheduler holds a long prompt while it prefills remotely, then decodes from the handed-over cache."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from kv_handoff_fakes import (
    FakeProducer,
    build_model,
    export_rank,
    frames_of,
    greedy,
    manifest,
    prefill,
)
from rdma_loopback import LoopbackLink, PythonWordOps

from omlx.cluster.rdma.mailbox import ClientMailbox
from omlx.remote_prefill.receiver import HandoffError
from omlx.remote_prefill.service import RemotePrefill
from omlx.remote_prefill.settings import RemotePrefillSettings

PROMPT = [(5 * index + 11) % 240 + 1 for index in range(70)]
END = 60


@pytest.fixture(scope="module")
def model():
    return build_model()


@pytest.fixture
def link():
    loop = LoopbackLink(request_bytes=64 * 1024, reply_bytes=64 * 1024)
    yield loop
    loop.close()


class _Blocks:
    """A paged cache whose shared prefix is `blocks` blocks long."""

    def __init__(self, blocks):
        self.paged_cache = SimpleNamespace(
            find_shared_prefix=lambda tokens: ([0] * blocks, [None] * blocks, [])
        )


def _scheduler(model, blocks=0):
    deleted = []
    return SimpleNamespace(
        model=model,
        config=SimpleNamespace(model_name="tiny-mlx", paged_cache_block_size=16),
        block_aware_cache=_Blocks(blocks) if blocks else None,
        paged_cache_manager=SimpleNamespace(delete_block_table=deleted.append),
        deleted=deleted,
    )


def _request(request_id="r1", prompt=PROMPT, **changes):
    fields = {
        "request_id": request_id,
        "prompt_token_ids": list(prompt),
        "generation_prompt_start": END,
        "cached_tokens": 0,
        "prompt_cache": None,
        "remaining_tokens": list(prompt),
        "block_table": None,
        "shared_prefix_blocks": 0,
        "vlm_extra_keys_for_cache": None,
        "specprefill_indices": None,
    }
    return SimpleNamespace(**{**fields, **changes})


def _service(scheduler, link, *, min_tokens=16, requester=None):
    settings = RemotePrefillSettings(
        url="http://prefill.invalid:8000",
        model="tiny",
        local_model="tiny-mlx",
        links=(link.name,),
        min_tokens=min_tokens,
        timeout_s=30,
    )
    return RemotePrefill(
        scheduler,
        settings,
        attach=lambda name: ClientMailbox.attach(name, PythonWordOps()),
        requester=requester or (lambda *args: None),
    )


def _held(service, request, timeout_s=30):
    """Poll defer the way scheduler steps do, until the request may be admitted."""
    deadline = time.monotonic() + timeout_s
    while service.defer(request):
        assert time.monotonic() < deadline, "the remote prefill never finished"
        time.sleep(0.01)


def _producer(link, local, first):
    layers, pages = export_rank(local, first, END)
    frames = frames_of(pages, 2)
    return FakeProducer(
        link, manifest(PROMPT[:END], first, layers, len(frames)), frames
    )


def test_a_long_prompt_waits_for_its_remote_prefill_then_decodes_from_it(model, link):
    producer = _producer(link, prefill(model, PROMPT[:END]), 0)
    service = _service(_scheduler(model), link)
    request = _request()
    assert service.defer(request) is True
    _held(service, request)
    producer.join()
    service.inject(request)
    assert request.cached_tokens == END
    assert request.remaining_tokens == PROMPT[END:]
    assert service.last["tokens"] == END and service.last["bytes"] > 0
    reference = greedy(model, prefill(model, PROMPT[:END]), PROMPT[END:], 6)
    assert greedy(model, request.prompt_cache, PROMPT[END:], 6) == reference


def test_only_the_part_the_local_cache_lacks_is_exported(model, link):
    asked = []
    producer = _producer(link, prefill(model, PROMPT[:END]), 32)
    service = _service(
        _scheduler(model, blocks=2), link, requester=lambda *args: asked.append(args[3])
    )
    request = _request()
    _held(service, request)
    producer.join()
    assert asked == [32]
    request.cached_tokens = 32
    request.prompt_cache = prefill(model, PROMPT[:32])
    service.inject(request)
    assert request.cached_tokens == END
    reference = greedy(model, prefill(model, PROMPT[:END]), PROMPT[END:], 6)
    assert greedy(model, request.prompt_cache, PROMPT[END:], 6) == reference


def test_a_local_cache_that_shrank_below_the_handoff_prefills_here(model, link):
    producer = _producer(link, prefill(model, PROMPT[:END]), 32)
    service = _service(_scheduler(model, blocks=2), link)
    request = _request()
    _held(service, request)
    producer.join()
    service.inject(request)
    assert request.cached_tokens == 0 and request.prompt_cache is None


@pytest.mark.parametrize(
    ("changes", "min_tokens"),
    [
        ({}, 100),
        ({"vlm_extra_keys_for_cache": ("image",)}, 16),
        ({"specprefill_indices": [1, 2]}, 16),
    ],
)
def test_short_multimodal_and_specprefill_prompts_prefill_here(
    model, link, changes, min_tokens
):
    service = _service(_scheduler(model), link, min_tokens=min_tokens)
    assert service.defer(_request(**changes)) is False
    assert service.status()["running"] == 0


def test_models_whose_caches_cannot_take_a_handoff_prefill_here(link):
    class Recurrent:
        pass

    model = SimpleNamespace(make_cache=lambda: [Recurrent()])
    service = _service(_scheduler(model), link)
    assert service.defer(_request()) is False
    assert "Recurrent" in service.status()["unsupported"]


def test_repeated_failures_pause_remote_prefill(model, link):
    def refuse(*args):
        raise HandoffError("vLLM answered 500: boom")

    service = _service(_scheduler(model), link, requester=refuse)
    for number in range(3):
        request = _request(f"r{number}")
        _held(service, request)
        service.inject(request)
        assert request.cached_tokens == 0
    status = service.status()
    assert status["paused"] and status["last_error"] == "vLLM answered 500: boom"
    assert service.defer(_request("r9")) is False


def test_a_request_that_leaves_the_queue_drops_its_job(model, link):
    producer = _producer(link, prefill(model, PROMPT[:END]), 0)
    service = _service(_scheduler(model), link)
    request = _request()
    assert service.defer(request)
    service.forget(request.request_id)
    service.inject(request)
    assert request.cached_tokens == 0
    producer.join()
