# SPDX-License-Identifier: Apache-2.0
"""Exercise chat prefix identity and usage through completed DFlash requests."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from dflash_mlx.engine.events import SummaryEvent, TokenEvent

from omlx.engine.dflash import DFlashEngine


class _Tokenizer:
    clean_up_tokenization_spaces = False
    chat_template = "test chat template"
    unk_token_id = 0
    eos_token_id = 2

    def encode(self, prompt):
        # Deliberately identical across policies: their cache identities must
        # remain separate even when template options produce the same tokens.
        return [11, 12, 13, 14, 20, 21]

    def decode(self, tokens, **kwargs):
        return "reply"

    def apply_chat_template(self, messages, **kwargs):
        return "templated prompt"

    def convert_tokens_to_ids(self, tokens):
        vocab = {"<|im_start|>": 20, "assistant": 21}
        if isinstance(tokens, str):
            return vocab.get(tokens, self.unk_token_id)
        return [vocab.get(token, self.unk_token_id) for token in tokens]


class _PrefixStore:
    """Small snapshot-store double; the real PrefixCacheFlow builds its keys."""

    def __init__(self):
        self.prefixes = set()

    def lookup(self, tokens, key, **kwargs):
        hit = (key, tuple(tokens)) in self.prefixes
        return SimpleNamespace(
            matched_tokens=len(tokens) if hit else 0,
            hit_kind="l1_exact" if hit else "miss",
            snapshot=None,
            elapsed_ms=0.0,
        )

    def log_stats(self, **kwargs):
        pass


@pytest.fixture
def engine(monkeypatch):
    from dflash_mlx import runtime
    from dflash_mlx.server import prefix_cache_flow

    engine = DFlashEngine(model_name="test-model", draft_model_path="test-draft")
    engine._loaded = True
    engine._tokenizer_obj = _Tokenizer()
    engine._executor_tokenizer = engine._tokenizer_obj
    engine._draft_model = SimpleNamespace(target_layer_ids=(0,))
    engine._runtime_context = SimpleNamespace(
        runtime=SimpleNamespace(
            prefix_cache=True,
            target_fa_window=0,
            draft_sink_size=0,
            draft_window_size=1024,
        )
    )
    store = _PrefixStore()
    monkeypatch.setattr(
        prefix_cache_flow, "get_runtime_cache_manager", lambda *args, **kwargs: store
    )
    monkeypatch.setattr(
        prefix_cache_flow.SnapshotService,
        "from_request",
        lambda **kwargs: SimpleNamespace(key=kwargs["key"]),
    )
    monkeypatch.setattr(
        "omlx.engine.dflash.create_streaming_detokenizer", lambda *args, **kwargs: None
    )

    def generate_events(**kwargs):
        prompt = kwargs["prompt_tokens_override"]
        yield TokenEvent(42, 1, 1.0, 1)
        # Model execution is the only part replaced: completing prefill saves
        # the real flow's stable prefix under the real policy-dependent key.
        store.prefixes.add(
            (
                kwargs["snapshot_service"].key,
                tuple(prompt[: kwargs["stable_prefix_len"]]),
            )
        )
        yield SummaryEvent(
            elapsed_us=1000,
            prompt_token_count=len(prompt),
            generated_token_ids=(42,),
            generation_tokens=1,
            accepted_from_draft=1,
            acceptance_ratio=1.0,
            cycles_completed=1,
            phase_timings_us={},
        )

    monkeypatch.setattr(runtime, "stream_dflash_generate", generate_events)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr("omlx.engine_core.get_mlx_executor", lambda: executor)
        yield engine


async def _complete_chat(engine, *, streaming, role="user", policy=False):
    kwargs = dict(
        messages=[{"role": role, "content": "hello"}],
        max_tokens=1,
        chat_template_kwargs={"enable_thinking": policy},
        is_partial=role == "assistant",
    )
    if streaming:
        outputs = [output async for output in engine.stream_chat(**kwargs)]
        assert all(output.cached_tokens == 0 for output in outputs[:-1])
        assert outputs[-1].finished
        output = outputs[-1]
    else:
        output = await engine.chat(**kwargs)
    assert output.text == "reply"
    assert output.prompt_tokens == 6
    assert output.completion_tokens == 1
    assert output.finish_reason == "length"
    return output.cached_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_chat_cache_hits_are_isolated_by_template_policy(engine, streaming):
    counts = [
        await _complete_chat(engine, streaming=streaming, policy=policy)
        for policy in (False, False, True, True)
    ]
    # The final assistant marker is outside a user message's stable prefix.
    assert counts == [0, 4, 0, 4]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_assistant_continuation_reuses_the_full_prompt(engine, streaming):
    counts = [
        await _complete_chat(engine, streaming=streaming, role="assistant")
        for _ in range(2)
    ]
    # An assistant continuation must keep its final role marker in the cached
    # prefix. Omitting the request's last role would truncate this to 4.
    assert counts == [0, 6]
