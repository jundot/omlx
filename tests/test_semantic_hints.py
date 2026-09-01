# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the OoO-Spec semantic sidecar foundation."""

import asyncio
import json
import urllib.request

import httpx
import pytest

from omlx.speculative.semantic_hints import (
    SemanticHintConfig,
    SemanticHintMailbox,
    SemanticHintProvider,
    SemanticHintValidationError,
    SemanticToolCall,
    TargetTemplateMismatchError,
    parse_semantic_hint_response,
    prepare_semantic_hint_candidate,
    render_target_hint,
    start_semantic_hint_mailbox,
    start_semantic_hint_request_context,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]
MESSAGES = [{"role": "user", "content": "What is the weather?"}]


@pytest.fixture(autouse=True)
def _force_mlx_cpu_device(monkeypatch):
    import mlx.core as mx

    previous = mx.default_device()
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.mark.asyncio
async def test_provider_posts_final_tools_and_resolves_name_server_side():
    seen: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"tool_index": 0, "arguments": {"city": "Boston"}}
        )

    provider = SemanticHintProvider(
        "https://sidecar.invalid/hint",
        transport=httpx.MockTransport(handler),
    )

    hint = await provider.request_hint(MESSAGES, TOOLS)

    assert hint.function_name == "weather"
    assert hint.arguments == {"city": "Boston"}
    assert seen["body"] == {"messages": MESSAGES, "tools": TOOLS}


@pytest.mark.asyncio
async def test_provider_disables_environment_proxy_inheritance(monkeypatch):
    seen: dict[str, object] = {}
    original_client = httpx.AsyncClient

    class CapturingClient(original_client):
        def __init__(self, *args, **kwargs):
            seen["trust_env"] = kwargs.get("trust_env")
            super().__init__(*args, **kwargs)

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"tool_index": 0, "arguments": {"city": "Boston"}}
        )

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setattr(httpx, "AsyncClient", CapturingClient)
    provider = SemanticHintProvider(
        "http://127.0.0.1:9876/hint",
        transport=httpx.MockTransport(handler),
    )

    assert (await provider.request_hint(MESSAGES, TOOLS)).function_name == "weather"
    assert seen == {"trust_env": False}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"tool_index": 1, "arguments": {"city": "Boston"}},
        {"tool_index": 0, "arguments": {}},
        {"tool_index": 0, "arguments": {"city": "Boston"}, "token_ids": [1]},
    ],
)
async def test_provider_rejects_bad_index_or_schema_without_echoing_arguments(response):
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    provider = SemanticHintProvider(
        "https://sidecar.invalid/hint",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SemanticHintValidationError) as exc_info:
        await provider.request_hint(MESSAGES, TOOLS)

    assert "Boston" not in str(exc_info.value)


@pytest.mark.parametrize("reference_key", ["$ref", "$dynamicRef", "$recursiveRef"])
def test_recursive_schema_references_are_rejected_without_outbound_get(
    monkeypatch, reference_key
):
    outbound: list[str] = []

    def reject_outbound(url, *args, **kwargs):
        outbound.append(str(url))
        raise AssertionError("schema validation attempted outbound retrieval")

    monkeypatch.setattr(urllib.request, "urlopen", reject_outbound)
    tools = json.loads(json.dumps(TOOLS))
    tools[0]["function"]["parameters"]["properties"]["nested"] = {
        "type": "array",
        "items": {reference_key: "http://169.254.169.254/latest/meta-data/"},
    }

    with pytest.raises(SemanticHintValidationError, match="schema references"):
        parse_semantic_hint_response(
            {"tool_index": 0, "arguments": {"city": "Boston"}},
            tools,
        )

    assert (
        prepare_semantic_hint_candidate(
            MESSAGES,
            tools,
            tools,
            {},
            config=SemanticHintConfig(
                enabled=True,
                endpoint="http://127.0.0.1:9876/hint",
            ),
        )
        is None
    )
    assert outbound == []


@pytest.mark.asyncio
async def test_mailbox_is_nonblocking_and_cancellable():
    started = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    provider = SemanticHintProvider(
        "https://sidecar.invalid/hint",
        transport=httpx.MockTransport(handler),
    )
    mailbox = SemanticHintMailbox.start(provider, MESSAGES, TOOLS)

    await started.wait()
    assert mailbox.pending
    assert mailbox.poll_once() is None

    mailbox.cancel()
    assert await mailbox.wait() is None
    assert mailbox.poll_once() is None


@pytest.mark.asyncio
async def test_mailbox_pending_poll_makes_a_late_result_inert():
    release = asyncio.Event()
    hint = SemanticToolCall(
        tool_index=0,
        function_name="weather",
        arguments={"city": "Boston"},
    )

    async def resolve_late() -> SemanticToolCall:
        await release.wait()
        return hint

    mailbox = SemanticHintMailbox(asyncio.create_task(resolve_late()))
    assert mailbox.poll_once() is None

    release.set()
    assert await mailbox.wait() == hint
    assert mailbox.poll_once() is None


@pytest.mark.asyncio
async def test_mailbox_timeout_fails_closed():
    started = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    provider = SemanticHintProvider(
        "https://sidecar.invalid/hint",
        timeout_s=0.001,
        transport=httpx.MockTransport(handler),
    )
    mailbox = SemanticHintMailbox.start(provider, MESSAGES, TOOLS)

    await started.wait()
    assert await mailbox.wait() is None
    assert mailbox.poll_once() is None


def test_target_suffix_is_rendered_and_encoded_by_the_target_only():
    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(char) for char in text]

    def render_chat(messages, tools) -> str:
        assert tools == TOOLS
        prompt = "<target-prompt>"
        if len(messages) == 1:
            return prompt
        function = messages[-1]["tool_calls"][0]["function"]
        return f'{prompt}<call name="{function["name"]}">{function["arguments"]}</call>'

    rendered = render_target_hint(
        messages=MESSAGES,
        tools=TOOLS,
        hint=SemanticToolCall(
            tool_index=0,
            function_name="weather",
            arguments={"city": "Boston"},
        ),
        render_chat=render_chat,
        tokenizer=FakeTokenizer(),
    )

    assert rendered.suffix_text == '<call name="weather">{"city":"Boston"}</call>'
    assert rendered.prompt_token_ids == [ord(char) for char in "<target-prompt>"]
    assert rendered.suffix_token_ids == [ord(char) for char in rendered.suffix_text]


def test_target_suffix_rejects_a_text_append_that_retokenizes_the_prompt_boundary():
    class BoundaryTokenizer:
        def encode(self, text: str) -> list[int]:
            if text == "base":
                return [10, 11]
            if text == "base suffix":
                return [99, 12]
            raise AssertionError(f"unexpected text: {text}")

    def render_chat(messages, tools) -> str:
        assert tools == TOOLS
        return "base" if len(messages) == 1 else "base suffix"

    with pytest.raises(TargetTemplateMismatchError, match="append-only token suffix"):
        render_target_hint(
            messages=MESSAGES,
            tools=TOOLS,
            hint=SemanticToolCall(
                tool_index=0,
                function_name="weather",
                arguments={"city": "Boston"},
            ),
            render_chat=render_chat,
            tokenizer=BoundaryTokenizer(),
        )


@pytest.mark.asyncio
async def test_disabled_or_completed_mailbox_is_one_shot():
    assert (
        start_semantic_hint_mailbox(
            MESSAGES,
            TOOLS,
            config=SemanticHintConfig(
                enabled=False, endpoint="https://sidecar.invalid"
            ),
        )
        is None
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"tool_index": 0, "arguments": {"city": "Boston"}}
        )

    mailbox = start_semantic_hint_mailbox(
        MESSAGES,
        TOOLS,
        config=SemanticHintConfig(enabled=True, endpoint="https://sidecar.invalid"),
        provider=SemanticHintProvider(
            "https://sidecar.invalid/hint",
            transport=httpx.MockTransport(handler),
        ),
    )
    assert mailbox is not None
    hint = await mailbox.wait()
    assert hint is not None
    assert mailbox.poll_once() == hint
    assert mailbox.poll_once() is None


@pytest.mark.asyncio
async def test_request_context_is_loopback_only_and_immutable():
    messages = [{"role": "user", "content": "original"}]
    tools = json.loads(json.dumps(TOOLS))
    template_options = {"enable_thinking": False}

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"tool_index": 0, "arguments": {"city": "Boston"}}
        )

    remote = prepare_semantic_hint_candidate(
        messages,
        tools,
        tools,
        template_options,
        config=SemanticHintConfig(
            enabled=True,
            endpoint="https://sidecar.example/hint",
        ),
    )
    assert remote is None

    provider = SemanticHintProvider(
        "http://127.0.0.1:9876/hint",
        transport=httpx.MockTransport(handler),
    )
    candidate = prepare_semantic_hint_candidate(
        messages,
        tools,
        tools,
        template_options,
        config=SemanticHintConfig(
            enabled=True,
            endpoint="http://127.0.0.1:9876/hint",
        ),
    )
    assert candidate is not None
    context = start_semantic_hint_request_context(candidate, provider=provider)
    assert context is not None

    messages[0]["content"] = "mutated"
    tools[0]["function"]["name"] = "mutated"
    template_options["enable_thinking"] = True

    assert json.loads(context.messages_json)[0]["content"] == "original"
    assert json.loads(context.template_tools_json)[0]["function"]["name"] == ("weather")
    assert json.loads(context.template_options_json) == {"enable_thinking": False}
    assert await context.mailbox.wait() is not None


@pytest.mark.asyncio
async def test_call_free_candidate_dispatches_from_scheduler_thread():
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"tool_index": 0, "arguments": {"city": "Boston"}}
        )

    config = SemanticHintConfig(
        enabled=True,
        endpoint="http://127.0.0.1:9876/hint",
    )
    candidate = prepare_semantic_hint_candidate(
        MESSAGES,
        TOOLS,
        TOOLS,
        {},
        config=config,
    )
    assert candidate is not None
    assert calls == 0
    provider = SemanticHintProvider(
        config.endpoint,
        transport=httpx.MockTransport(handler),
    )

    context = await asyncio.to_thread(
        start_semantic_hint_request_context,
        candidate,
        provider=provider,
    )

    assert context is not None
    hint = await context.mailbox.wait()
    assert hint is not None
    assert hint.function_name == "weather"
    assert calls == 1


def test_public_batch_generator_requires_replay_not_pending_token_adoption():
    """Its public insert API treats the supplied token as an input, not output.

    A verifier that has already sampled token 9 from cache ``[1, 2, 7, 8]``
    needs the next decode call to *emit* 9. Re-inserting 9 instead processes
    it immediately and the public generator emits 10. The absent adoption
    operation is why the bounded lane first publishes target token 9 itself,
    then trims and replays 9 as the public generator input. All arrays here
    are tiny synthetic MLX values, not model inference.
    """
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator

    class FakePromptCache:
        def __init__(self, history: list[int]) -> None:
            self.history = list(history)

        def merge(self, caches):
            return FakeBatchCache([cache.history for cache in caches])

        def trim(self, n):
            del self.history[-n:]
            return n

    class FakeBatchCache:
        def __init__(self, rows: list[list[int]]) -> None:
            self.rows = [list(row) for row in rows]

        def append(self, inputs) -> None:
            for row, values in enumerate(inputs.tolist()):
                self.rows[row].extend(int(value) for value in values)

        def extract(self, index: int) -> FakePromptCache:
            return FakePromptCache(self.rows[index])

        def filter(self, keep: list[int]) -> None:
            self.rows = [self.rows[index] for index in keep]

    class FakeModel:
        def __call__(self, inputs, cache):
            cache[0].append(inputs)
            logits = []
            for row in inputs.tolist():
                row_logits = []
                for token in row:
                    winner = (int(token) + 1) % 16
                    row_logits.append(
                        [0.0 if index == winner else -1000.0 for index in range(16)]
                    )
                logits.append(row_logits)
            return mx.array(logits)

    def make_generator():
        return BatchGenerator(
            FakeModel(),
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            completion_batch_size=1,
            prefill_batch_size=1,
            stream=mx.new_stream(mx.cpu),
        )

    first = make_generator()
    resumed = make_generator()
    replayed = make_generator()
    try:
        uid = first.insert(
            [[7]],
            max_tokens=[8],
            caches=[[FakePromptCache([1, 2])]],
            all_tokens=[[1, 2]],
        )[0]
        assert [response.token for response in first.next_generated()] == [8]
        first_cache, _ = first.extract_cache([uid])[uid]
        assert first_cache[0].history == [1, 2, 7, 8]

        # A one-shot verifier would now hold this cache and a pending sampled
        # token 9. Public insert() consumes 9 as a new input, so it cannot
        # transfer that continuation without private GenerationBatch state.
        resumed_uid = resumed.insert(
            [[9]],
            max_tokens=[8],
            caches=[[FakePromptCache([1, 2, 7, 8])]],
            all_tokens=[[1, 2, 7, 8]],
        )[0]
        assert [response.token for response in resumed.next_generated()] == [10]
        resumed_cache, _ = resumed.extract_cache([resumed_uid])[resumed_uid]
        assert resumed_cache[0].history == [1, 2, 7, 8, 9, 10]

        # Once token 9 has been published by the bounded queue, exact replay
        # is sufficient: trim it from verified KV, then let insert() consume
        # it. The next public output is serial token 10, with exact history.
        verified_cache = FakePromptCache([1, 2, 7, 8, 9])
        assert verified_cache.trim(1) == 1
        replayed_uid = replayed.insert(
            [[9]],
            max_tokens=[8],
            caches=[[verified_cache]],
            all_tokens=[[1, 2, 7, 8]],
        )[0]
        assert [response.token for response in replayed.next_generated()] == [10]
        replayed_cache, _ = replayed.extract_cache([replayed_uid])[replayed_uid]
        assert replayed_cache[0].history == [1, 2, 7, 8, 9, 10]
    finally:
        first.close()
        resumed.close()
        replayed.close()
