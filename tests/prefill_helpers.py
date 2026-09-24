"""Shared small-model fixtures for prefill contracts; no test-module imports."""

import importlib
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import KVCache

from omlx.request import Request, SamplingParams


@pytest.fixture
def model_factory(monkeypatch):
    models = []

    def create(model_type="llama", *, quantized=False):
        if model_type == "hy_v3":
            from omlx.patches.hy_v3 import apply_hy_v3_patch

            apply_hy_v3_patch()
        module = importlib.import_module(f"mlx_lm.models.{model_type}")
        stream = mx.new_thread_local_stream(mx.default_device())
        with mx.stream(stream):
            mx.random.seed(2718)
            arguments = {
                "model_type": model_type,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-05,
                "vocab_size": 64,
                "head_dim": 8,
                "max_position_embeddings": 2048,
                "rope_theta": 10000,
                "tie_word_embeddings": False,
            }
            if model_type in ("hy_v3",):
                arguments.update(
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=4,
                    head_dim=32,
                )
            if model_type == "hy_v3":
                arguments.update(
                    num_experts=4,
                    num_experts_per_tok=2,
                    num_shared_experts=1,
                    expert_hidden_dim=64,
                    first_k_dense_replace=1,
                    rope_parameters={"rope_theta": 10000, "rope_type": "default"},
                )
            model = module.Model(module.ModelArgs.from_dict(arguments))
            if quantized:
                model.set_dtype(mx.float16)
                nn.quantize(model, group_size=32, bits=4)
            model.eval()
            mx.eval(model.parameters())
        harness = SimpleNamespace(
            model=model,
            stream=stream,
            forwards=[],
            original_forward=type(model).__call__,
            record=True,
            tolerance=(
                (0.01 if model_type in ("hy_v3",) else 0.005) if quantized else 3e-05
            ),
        )

        def record_forward(instance, inputs, cache=None, **kwargs):
            if instance is model and harness.record:
                harness.forwards.append(
                    SimpleNamespace(
                        shape=tuple(inputs.shape),
                        cache_size=next(
                            layer.size()
                            for layer in cache
                            if isinstance(layer, KVCache) or hasattr(layer, "keys")
                        ),
                    )
                )
            return harness.original_forward(instance, inputs, cache=cache, **kwargs)

        monkeypatch.setattr(type(model), "__call__", record_forward)
        models.append(harness)
        return harness

    yield create
    for harness in models:
        mx.synchronize(harness.stream)


def make_request(request_id, length, *, start=1, max_tokens=6, **sampling):
    token_ids = [(start + token_index * 7) % 61 for token_index in range(length)]
    return Request(
        request_id=request_id,
        prompt=token_ids,
        prompt_token_ids=token_ids,
        num_prompt_tokens=length,
        sampling_params=SamplingParams(
            temperature=0, max_tokens=max_tokens, **sampling
        ),
        skip_cache_store=True,
    )
