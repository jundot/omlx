# SPDX-License-Identifier: Apache-2.0
"""SpecPrefill after a restored prefix cache fails validation.

Two EngineCore sessions share one paged-SSD cache directory, as two server
runs do. Session 1 stores a warm prefix. Between the sessions the stored block
is rewritten on disk into a persistent hybrid cache block whose recurrent
state doesn't match the current model (recurrent slot validation failure).
Session 2 restores that block, scores the uncached suffix with the draft, and
only then does cache validation reject the restored state. The fallback must
prefill the full prompt densely instead of reusing indices scored against the
rejected cache.

The models are tiny random-init mlx_lm models. The on-disk rewrite is the only
fault; everything else is observed read-only at the model boundary (the target
and the draft record each forward).
"""

import asyncio
import glob
import json
import os

import mlx.core as mx
import pytest
from mlx_lm.models import qwen3_next
from mlx_lm.models.llama import Model as Llama
from mlx_lm.models.llama import ModelArgs as LlamaArgs

from omlx.engine_core import EngineConfig, EngineCore
from omlx.request import SamplingParams
from omlx.scheduler import SchedulerConfig

# Four layers alternating recurrent (ArraysCache) and attention (KVCache).
_TARGET_ARGS = qwen3_next.ModelArgs(
    model_type="qwen3_next",
    hidden_size=64,
    num_hidden_layers=4,
    intermediate_size=128,
    num_attention_heads=4,
    linear_num_value_heads=2,
    linear_num_key_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=0,
    num_experts_per_tok=0,
    decoder_sparse_step=1,
    shared_expert_intermediate_size=0,
    mlp_only_layers=[0, 1, 2, 3],
    moe_intermediate_size=0,
    rms_norm_eps=1e-5,
    vocab_size=1024,
    num_key_value_heads=2,
    rope_theta=10000.0,
    partial_rotary_factor=0.5,
    max_position_embeddings=4096,
    head_dim=16,
    full_attention_interval=2,
)
_DRAFT_ARGS = LlamaArgs(
    model_type="llama",
    hidden_size=64,
    num_hidden_layers=2,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    rms_norm_eps=1e-5,
    vocab_size=1024,
)
_TIMEOUT = 60.0
_MODEL_NAME = "tiny-hybrid"
_REQUEST_ID = "sp-hit"
_SPECPREFILL_ROPES = ("_OffsetAdjustedRoPE", "_PositionMappedRoPE")


class _Recording:
    """Records each forward's tokens and the request's cached_tokens, read-only."""

    def __init__(self, args):
        super().__init__(args)
        self.requests = {}
        self.forwards = []

    def __call__(self, inputs, *args, **kwargs):
        request = self.requests.get(_REQUEST_ID)
        self.forwards.append(
            {
                "tokens": inputs[0].tolist(),
                "cached_tokens": None if request is None else request.cached_tokens,
                "specprefill_rope": any(
                    type(layer.self_attn.rope).__name__ in _SPECPREFILL_ROPES
                    for layer in self.model.layers
                    if hasattr(layer, "self_attn")
                ),
            }
        )
        return super().__call__(inputs, *args, **kwargs)


class _Target(_Recording, qwen3_next.Model):
    pass


class _Draft(_Recording, Llama):
    pass


def _models():
    mx.random.seed(0)
    target = _Target(_TARGET_ARGS)
    draft = _Draft(_DRAFT_ARGS)
    mx.eval(target.parameters(), draft.parameters())
    return target, draft


def _engine(target, draft, tokenizer, cache_dir):
    engine = EngineCore(
        target,
        tokenizer,
        EngineConfig(
            scheduler_config=SchedulerConfig(
                prefill_step_size=128,
                paged_cache_block_size=64,
                max_cache_blocks=64,
                paged_ssd_cache_dir=str(cache_dir),
                model_name=_MODEL_NAME,
            )
        ),
    )
    engine.scheduler.set_specprefill_draft_model(draft, draft_model_name=None)
    return engine


async def _generate(engine, prompt, what, **kwargs):
    try:
        return await asyncio.wait_for(engine.generate(prompt, **kwargs), _TIMEOUT)
    except TimeoutError:
        pytest.fail(f"{what} did not finish within {_TIMEOUT:.0f}s")


async def _store_prefix(tokenizer, cache_dir, prefix):
    target, draft = _models()
    engine = _engine(target, draft, tokenizer, cache_dir)
    await engine.start()
    try:
        return await _generate(
            engine,
            prefix,
            "warm-prefix request",
            sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
            request_id="warm-prefix",
        )
    finally:
        await engine.stop()
        # Graceful shutdown drains the async store and flushes the SSD writer.
        engine.scheduler.shutdown()


def _stored_blocks(cache_dir):
    return glob.glob(os.path.join(cache_dir, "**", "*.safetensors"), recursive=True)


def _add_recurrent_slot(path):
    """External fault: give each recurrent layer one extra state on disk."""
    arrays, meta = mx.load(path, return_metadata=True)
    rewritten = {}
    for layer, cls in enumerate(json.loads(meta["layer_cache_types"])):
        count = int(meta[f"layer_{layer}_state_count"])
        states = [arrays[f"layer_{layer}_state_{i}"] for i in range(count)]
        if cls == "ArraysCache":
            states.append(mx.zeros_like(states[-1]))
            meta[f"layer_{layer}_state_count"] = str(len(states))
        for i, state in enumerate(states):
            rewritten[f"layer_{layer}_state_{i}"] = state
    mx.eval(list(rewritten.values()))
    tmp = path[: -len(".safetensors")] + ".rewrite.safetensors"
    mx.save_safetensors(tmp, rewritten, metadata=meta)
    os.replace(tmp, path)


async def _hit_rejected_prefix(tokenizer, cache_dir, prompt):
    target, draft = _models()
    engine = _engine(target, draft, tokenizer, cache_dir)
    target.requests = draft.requests = engine.scheduler.requests
    await engine.start()
    try:
        out = await _generate(
            engine,
            prompt,
            f"SpecPrefill request {_REQUEST_ID}",
            sampling_params=SamplingParams(max_tokens=8, temperature=0.0),
            request_id=_REQUEST_ID,
            specprefill=True,
            specprefill_threshold=256,
            specprefill_keep_pct=0.3,
        )
    finally:
        await engine.stop()
        engine.scheduler.shutdown()
    return target, draft, out


def test_specprefill_falls_back_to_dense_prefill_when_restored_cache_is_rejected(
    mock_tokenizer, tmp_path
):
    cache_dir = str(tmp_path / "ssd")
    prefix_len = 512
    prompt = [(i * 7) % 1000 + 10 for i in range(prefix_len)]
    prompt += [(i * 13) % 1000 + 10 for i in range(1024)]

    # (1) Session 1 builds a persistent prefix cache.
    warm = asyncio.run(_store_prefix(mock_tokenizer, cache_dir, prompt[:prefix_len]))
    assert warm.finished and warm.finish_reason in ("stop", "length"), warm
    blocks = _stored_blocks(cache_dir)
    assert len(blocks) == 1, f"expected one stored prefix block, found {blocks}"
    _add_recurrent_slot(blocks[0])

    target, draft, out = asyncio.run(
        _hit_rejected_prefix(mock_tokenizer, cache_dir, prompt)
    )

    # (2) Session 2 restores the stored prefix, and the draft scores only the
    # uncached suffix.
    assert draft.forwards, "the draft model was never called"
    cached = draft.forwards[0]["cached_tokens"]
    assert cached, "session 2 did not restore the stored prefix cache"
    scored = draft.forwards[0]["tokens"]
    assert scored == prompt[cached : cached + len(scored)]

    # (3) Recurrent slot validation rejects the restored state: the first
    # target forward runs with no cached prefix.
    forwards = target.forwards
    assert forwards, "the target model was never called"
    assert (
        forwards[0]["cached_tokens"] == 0
    ), f"first forward ran with cached_tokens={forwards[0]['cached_tokens']}"

    # (4) The model boundary receives the full prompt densely from prompt[0],
    # without the SpecPrefill RoPE wrapper.
    stream = [token for forward in forwards for token in forward["tokens"]]
    assert stream[: len(prompt)] == prompt, (
        f"after the rejected restore the target was fed {stream[:6]}... "
        f"({len(stream)} tokens), expected the full prompt {prompt[:6]}... "
        f"densely"
    )
    assert not any(forward["specprefill_rope"] for forward in forwards)

    # (5) The request finishes normally.
    assert out.finished and out.finish_reason in ("stop", "length"), out
