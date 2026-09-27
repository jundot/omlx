# SPDX-License-Identifier: Apache-2.0
"""A SpecPrefill request that fails mid-decode must not wedge the engine.

The request runs through EngineCore with a tiny random-init llama target and
draft. The only injection is at the model boundary: the target raises once on
a decode forward while the sparse-prefill RoPE wrapper is installed.
"""

import asyncio

import mlx.core as mx
import pytest
from mlx_lm.models.llama import Model, ModelArgs

from omlx.engine_core import EngineConfig, EngineCore
from omlx.request import SamplingParams
from omlx.scheduler import SchedulerConfig

_ARGS = ModelArgs(
    model_type="llama",
    hidden_size=64,
    num_hidden_layers=2,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    rms_norm_eps=1e-5,
    vocab_size=1024,
)
_TIMEOUT = 30.0
_REQUEST_ID = "sp-1"


def _has_specprefill_rope(model) -> bool:
    return any(
        type(layer.self_attn.rope).__name__ == "_OffsetAdjustedRoPE"
        for layer in model.model.layers
    )


class _FaultOnDecode(Model):
    """Records every forward and raises once on an armed decode step.

    A decode step is a one-token forward after the request has produced
    output. Request fields are only read, from the scheduler's request table.
    """

    def __init__(self, args):
        super().__init__(args)
        self.requests = {}
        self.fault = None
        self.forwards = []
        self.at_fault = None
        self.fault_index = None

    def __call__(self, inputs, *args, **kwargs):
        request = self.requests.get(_REQUEST_ID)
        self.forwards.append(inputs[0].tolist())
        if (
            self.fault is not None
            and inputs.shape[1] == 1
            and request is not None
            and request.num_output_tokens > 0
        ):
            self.at_fault = {
                "cached_tokens": request.cached_tokens,
                "under_specprefill": _has_specprefill_rope(self),
            }
            self.fault_index = len(self.forwards)
            exc, self.fault = self.fault, None
            raise exc
        return super().__call__(inputs, *args, **kwargs)


async def _generate(engine, prompt, request_id, max_tokens, **kwargs):
    try:
        return await asyncio.wait_for(
            engine.generate(
                prompt,
                SamplingParams(max_tokens=max_tokens, temperature=0.0),
                request_id=request_id,
                **kwargs,
            ),
            _TIMEOUT,
        )
    except TimeoutError:
        pytest.fail(f"request {request_id} never finished (engine stuck)")


async def _run(tokenizer, fault, prompt, warm_prefix=None, **scheduler_kwargs):
    """Run a SpecPrefill request that faults on decode, then a plain request."""
    mx.random.seed(0)
    target = _FaultOnDecode(_ARGS)
    draft = Model(_ARGS)
    mx.eval(target.parameters(), draft.parameters())
    engine = EngineCore(
        target,
        tokenizer,
        EngineConfig(
            scheduler_config=SchedulerConfig(prefill_step_size=128, **scheduler_kwargs)
        ),
    )
    engine.scheduler.set_specprefill_draft_model(draft, draft_model_name=None)
    target.requests = engine.scheduler.requests
    await engine.start()
    try:
        if warm_prefix is not None:
            warm = await _generate(engine, warm_prefix, "warm-prefix", 4)
            assert warm.finished and warm.finish_reason in ("stop", "length")
        target.fault = fault
        try:
            first = await _generate(
                engine,
                prompt,
                _REQUEST_ID,
                16,
                specprefill=True,
                specprefill_threshold=256,
                specprefill_keep_pct=0.3,
            )
        except Exception as exc:
            first = exc
        end = len(target.forwards)
        second = await _generate(engine, [11, 12, 13, 14, 15, 16, 17, 18], "plain", 8)

        assert target.fault is None, "fault was never injected"
        assert target.at_fault["under_specprefill"]
        assert engine.scheduler._specprefill_active_request_id is None
        assert not _has_specprefill_rope(target)
        assert second.finished and second.finish_reason in ("stop", "length")
        return target, first, target.forwards[target.fault_index : end]
    finally:
        await engine.stop()


# A RuntimeError goes through fail_all_requests; the TypeError is treated as
# cache corruption and the request is re-prefilled.
@pytest.mark.parametrize(
    "fault, retried",
    [
        (RuntimeError("injected decode failure"), False),
        (TypeError("'NoneType' object is not subscriptable"), True),
    ],
    ids=["fail_all_requests", "cache_corruption_retry"],
)
def test_decode_failure_after_specprefill_releases_engine(
    mock_tokenizer, fault, retried
):
    prompt = [(i * 7) % 1000 + 10 for i in range(600)]
    _, first, _ = asyncio.run(_run(mock_tokenizer, fault, prompt))

    if retried:
        assert not isinstance(first, Exception), first
        assert first.finished and first.finish_reason in ("stop", "length")
    else:
        assert isinstance(first, Exception)
        assert "injected decode failure" in str(first)


def test_corruption_retry_after_prefix_hit_prefills_full_prompt(
    mock_tokenizer, tmp_path
):
    # A prefix of at least 30% of the prompt makes admission wait for the
    # warm request's async store before the lookup.
    prefix_len = 512
    prompt = [(i * 7) % 1000 + 10 for i in range(prefix_len)]
    prompt += [(i * 13) % 1000 + 10 for i in range(1024)]
    target, retried, retry_forwards = asyncio.run(
        _run(
            mock_tokenizer,
            TypeError("'NoneType' object is not subscriptable"),
            prompt,
            warm_prefix=prompt[:prefix_len],
            paged_cache_block_size=64,
            max_cache_blocks=64,
            paged_ssd_cache_dir=str(tmp_path / "ssd"),
        )
    )
    assert target.at_fault["cached_tokens"] > 0, "sp-1 did not hit the prefix cache"

    # The retry drops the hit, so the whole prompt is prefilled densely from
    # position 0 before the kickoff token.
    retry = [token for tokens in retry_forwards for token in tokens]
    assert retry[: len(prompt)] == prompt, (
        f"retry forwarded {retry[:6]}..., expected the full prompt "
        f"{prompt[:6]}... densely"
    )
    assert not isinstance(retried, Exception), retried
    assert retried.finished and retried.finish_reason in ("stop", "length")
