# SPDX-License-Identifier: Apache-2.0
"""Text-only Flint serving through the native mlx-vlm backend."""

import asyncio
import json
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.cache import KVCache, RotatingKVCache
from mlx_vlm.models.flint.config import ModelConfig
from mlx_vlm.models.flint.flint import Model
from mlx_vlm.utils import load, load_model
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from omlx.engine.vlm import VLMBatchedEngine
from omlx.exceptions import InvalidRequestError
from omlx.models.vlm import VLMModelAdapter
from omlx.oq import _stream_calibration_supported, quantize_oq_streaming


def _tiny_config():
    return ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        no_rope_layer_indices=[0, 2],
        mlp_only_layers=[0],
        num_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        sliding_window=513,
        eos_token_id=[2],
    )


@pytest.fixture
def checkpoint(tmp_path):
    mx.random.seed(17)
    args = _tiny_config()
    model = Model(args)
    model.eval()
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    original = {}
    for key, value in tree_flatten(model.parameters()):
        key = key.removeprefix("language_model.")
        if ".switch_mlp." in key:
            prefix, projection = key.split(".switch_mlp.")
            for expert in range(args.num_experts):
                original[f"{prefix}.experts.{expert}.{projection}"] = value[expert]
        elif not key.endswith(".expert_bias"):
            original[key] = value
    source = tmp_path / "source"
    source.mkdir()
    config = {**asdict(args), "architectures": ["FlintForCausalLM"]}
    (source / "config.json").write_text(json.dumps(config))
    shards = [{}, {}]
    mapping = {}
    for index, (key, value) in enumerate(original.items()):
        shards[index % 2][key] = value
        mapping[key] = f"model-{index % 2}.safetensors"
    for index, shard in enumerate(shards):
        mx.save_safetensors(str(source / f"model-{index}.safetensors"), shard)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "hello": 3}
    vocab.update({f"t{index}": index for index in range(4, args.vocab_size)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = ByteLevel()
    processor = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    processor.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}hello{% endif %}"
    )
    processor.save_pretrained(source)
    yield source, model, original
    mx.synchronize()
    mx.clear_cache()


def test_original_cross_shard_checkpoint_strictly_loads(checkpoint):
    source, reference, _ = checkpoint
    loaded, processor = load(source, strict=True, trust_remote_code=False)
    tokens = mx.array([[1, 3, 4, 5, 6, 7]])
    adapter = VLMModelAdapter(loaded)
    assert adapter.model_type == "flint"
    assert adapter.layers is loaded.layers
    assert isinstance(processor, PreTrainedTokenizerFast)
    assert processor.encode("hello", add_special_tokens=False) == [3]
    assert processor.stopping_criteria is not None
    np.testing.assert_array_equal(
        np.asarray(adapter(tokens)), np.asarray(reference(tokens).logits)
    )

    engine = VLMBatchedEngine(str(source))
    engine._vlm_model = loaded
    engine._processor = processor
    engine._tokenizer = processor
    assert engine._apply_chat_template([{"role": "user", "content": "hello"}]) == (
        "hello hello"
    )


@pytest.mark.parametrize("prefix", ["", "language_model."])
@pytest.mark.parametrize("alias", ["expert_bias", "e_score_correction_bias"])
def test_native_sanitizer_preserves_router_bias(checkpoint, prefix, alias):
    _, model, original = checkpoint
    weights = {prefix + key: value for key, value in original.items()}
    bias = mx.array([0.125, -0.25], dtype=mx.bfloat16)
    weights[f"{prefix}model.layers.1.mlp.gate.{alias}"] = bias
    sanitized = model.sanitize(weights)
    assert mx.array_equal(
        sanitized["language_model.model.layers.1.mlp.expert_bias"], bias
    ).item()
    model.load_weights(list(sanitized.items()), strict=True)


@pytest.mark.parametrize("bits", [4, 6])
def test_prequantized_checkpoint_strictly_loads(checkpoint, tmp_path, bits):
    _, model, _ = checkpoint
    nn.quantize(model, group_size=64, bits=bits)
    output = tmp_path / "quantized"
    output.mkdir()
    config = {
        **asdict(model.config),
        "architectures": ["FlintForCausalLM"],
        "quantization": {"group_size": 64, "bits": bits},
    }
    (output / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(
        str(output / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    loaded = load_model(output, lazy=False, strict=True, trust_remote_code=False)
    tokens = mx.array([[1, 3, 4, 5]])
    logits = VLMModelAdapter(loaded)(tokens)
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(model(tokens).logits))
    assert mx.isfinite(logits).all().item()


@pytest.mark.parametrize("quantized", [False, True])
def test_adapter_native_cache_crosses_sliding_boundary(checkpoint, quantized):
    _, model, _ = checkpoint
    if quantized:
        nn.quantize(model, group_size=64, bits=4)
    adapter = VLMModelAdapter(model)
    tokens = (mx.arange(514)[None, :] % 100).astype(mx.int32)
    full = adapter(tokens)[:, -1, :]
    cache = adapter.make_cache()
    assert [type(item) for item in cache] == [KVCache, RotatingKVCache, KVCache]
    assert cache[1].max_size == 513
    cached_full = adapter(tokens, cache=cache)[:, -1, :]
    assert mx.array_equal(full, cached_full).item()
    cache = adapter.make_cache()
    mx.eval(adapter(tokens[:, :512], cache=cache))
    continued = adapter(tokens[:, 512:], cache=cache)[:, -1, :]
    assert mx.isfinite(continued).all().item()
    assert all(item.offset == 514 for item in cache)
    if not quantized:
        np.testing.assert_array_equal(np.asarray(full), np.asarray(continued))
    # Quantized row-shape differences are unresolved, not an equivalence gate.
    decoded = adapter(mx.array([[3]]), cache=cache)
    assert decoded.shape == (1, 1, 128)
    assert all(item.offset == 515 for item in cache)


@pytest.mark.parametrize("media", ["images", "audio", "videos"])
@pytest.mark.parametrize("loaded", [False, True])
def test_engine_rejects_media(checkpoint, media, loaded):
    source, model, _ = checkpoint
    engine = VLMBatchedEngine(str(source))
    if loaded:
        engine._vlm_model = model
    kwargs = {"images": [], media: [object()]}
    with pytest.raises(InvalidRequestError, match="Flint is a text-only model"):
        engine._prepare_vision_inputs([{"role": "user", "content": "hello"}], **kwargs)


@pytest.mark.asyncio
async def test_engine_serves_synthetic_flint(checkpoint):
    source, _, _ = checkpoint
    engine = VLMBatchedEngine(str(source))
    try:
        await asyncio.wait_for(engine.start(), timeout=30)
        assert engine.model_type == "flint"
        assert isinstance(engine._adapter, VLMModelAdapter)
        output = await asyncio.wait_for(
            engine.generate([1, 3, 4], max_tokens=2, temperature=0), timeout=30
        )
        assert output.prompt_tokens == 3
        assert 1 <= output.completion_tokens <= 2
        assert output.finish_reason in {"stop", "length"}

        async def collect_stream():
            return [
                chunk
                async for chunk in engine.stream_generate(
                    [1, 3, 4], max_tokens=2, temperature=0
                )
            ]

        chunks = await asyncio.wait_for(collect_stream(), timeout=30)
        assert chunks[-1].finished
        assert chunks[-1].prompt_tokens == output.prompt_tokens
        assert chunks[-1].completion_tokens == output.completion_tokens
        assert chunks[-1].finish_reason == output.finish_reason
        assert chunks[-1].text == output.text

        outputs = await asyncio.wait_for(
            asyncio.gather(
                engine.generate([1, 3, 4], max_tokens=2, temperature=0),
                engine.generate([1, 3, 5, 6], max_tokens=2, temperature=0),
            ),
            timeout=30,
        )
        assert [result.prompt_tokens for result in outputs] == [3, 4]
        assert all(result.finish_reason in {"stop", "length"} for result in outputs)
    finally:
        await engine.stop()


@pytest.mark.parametrize("enhanced", [False, True])
def test_oq_conversion_fails_before_loading_or_writing(checkpoint, tmp_path, enhanced):
    source, _, _ = checkpoint
    output = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="Flint oQ conversion and calibration"):
        quantize_oq_streaming(str(source), str(output), 4, enhanced=enhanced)
    assert not output.exists()
    assert not _stream_calibration_supported("flint")
