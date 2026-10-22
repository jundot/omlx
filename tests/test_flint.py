# SPDX-License-Identifier: Apache-2.0
"""Native text-only Flint serving and original-source oQ calibration."""

import asyncio
import json
from dataclasses import asdict
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.cache import KVCache, RotatingKVCache
from mlx_vlm.models.flint.config import ModelConfig
from mlx_vlm.models.flint.flint import Model
from mlx_vlm.models.flint.language import LanguageModel, SparseMoeBlock
from mlx_vlm.utils import load, load_model
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

import omlx.oq as oq
from omlx.engine.vlm import VLMBatchedEngine
from omlx.exceptions import InvalidRequestError
from omlx.models.vlm import VLMModelAdapter


def _tiny_config(window=513):
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
        sliding_window=window,
        eos_token_id=[2],
    )


@pytest.fixture
def checkpoint(tmp_path, request):
    mx.random.seed(17)
    args = _tiny_config(getattr(request, "param", 513))
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
    source, model, original = checkpoint
    weights = {prefix + key: value for key, value in original.items()}
    bias = mx.array([0.125, -0.25], dtype=mx.bfloat16)
    weights[f"{prefix}model.layers.1.mlp.gate.{alias}"] = bias
    sanitized = model.sanitize(weights)
    assert mx.array_equal(
        sanitized["language_model.model.layers.1.mlp.expert_bias"], bias
    ).item()
    sanitizer = oq._build_model_sanitizer(asdict(model.config), model_path=source)
    actual = sanitizer(weights)
    assert actual.keys() == sanitized.keys()
    for key in sanitized:
        assert mx.array_equal(actual[key], sanitized[key]).item()
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


def test_sanitizer_never_constructs_model_and_matches_native(checkpoint, monkeypatch):
    source, model, original = checkpoint
    config = asdict(model.config)
    expected = model.sanitize(original)
    for cls in (Model, LanguageModel, SparseMoeBlock):
        monkeypatch.setattr(
            cls, "__init__", Mock(side_effect=AssertionError("weights"))
        )
    sanitizer = oq._build_model_sanitizer(config, model_path=source)
    actual = sanitizer(original)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert mx.array_equal(actual[key], expected[key]).item()
    plan = oq._streamed_source_plan(source, config)
    assert plan.keys() == expected.keys()
    for key in expected:
        assert mx.array_equal(plan.pop(key), expected[key]).item()


@pytest.mark.parametrize("window,length", [(3, 2), (3, 3), (3, 6), (513, 514)])
def test_layer_walk_matches_native_masks_and_logits(window, length):
    mx.random.seed(5)
    model = Model(_tiny_config(window))
    model.eval()
    model.set_dtype(mx.bfloat16)
    tokens = (mx.arange(length)[None, :] % 100).astype(mx.int32)
    core = model.language_model.model
    inputs = core.embed_tokens(tokens)
    inputs, masks, state = oq._prepare_layer_inputs(model, model.layers, tokens, inputs)
    for index, layer in enumerate(model.layers):
        inputs, _ = oq._forward_layer_result(
            layer, inputs, masks[index], state, layer_idx=index
        )
    hidden = core.norm(inputs).astype(mx.float32)
    actual = model.language_model.lm_head(hidden)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(model(tokens).logits))
    collector = Mock(
        _original_modules={"language_model.lm_head": model.language_model.lm_head},
        entries={},
    )
    oq._collect_flint_lm_head_imatrix(model, inputs, collector)
    captured = collector.collect_dense.call_args.args[2]
    assert captured.dtype == mx.float32
    np.testing.assert_array_equal(
        np.asarray(captured), np.asarray(core(tokens).astype(mx.float32))
    )


@pytest.mark.parametrize("model_type", ["flint", "glm5_next"])
def test_shared_streamed_head_preserves_normalization_and_dtype(
    checkpoint, monkeypatch, model_type
):
    source, model, _ = checkpoint
    norm = nn.RMSNorm(model.config.hidden_size, eps=0.01)
    norm.weight = model.language_model.model.norm.weight
    head = model.language_model.lm_head
    monkeypatch.setattr(
        oq,
        "_streamed_source_plan",
        lambda *a: {
            "language_model.model.norm.weight": norm.weight,
            "language_model.lm_head.weight": head.weight,
        },
    )
    config = {"model_type": model_type}
    config.update(
        {"rms_norm_eps": 0.01}
        if model_type == "flint"
        else {"text_config": {"rms_norm_eps": 0.01}}
    )
    shape = (2, 6, 64) if model_type == "flint" else (2, 6, 2, 64)
    working = [mx.random.normal(shape).astype(mx.bfloat16) for _ in range(2)]
    expected = oq.OQImatrixCollector()
    actual = oq.OQImatrixCollector()
    collect = Mock(wraps=actual.collect_dense)
    monkeypatch.setattr(actual, "collect_dense", collect)
    for hidden in working:
        hidden = hidden.mean(axis=2) if hidden.ndim == 4 else hidden
        normalized = norm(hidden)
        if model_type == "flint":
            normalized = normalized.astype(mx.float32)
        expected.collect_dense("language_model.lm_head", head, normalized)
    assert oq._collect_streamed_untied_lm_head(source, config, actual, working) == 1
    dtype = mx.float32 if model_type == "flint" else mx.bfloat16
    assert all(call.args[2].dtype == dtype for call in collect.call_args_list)
    reference = expected.entries["language_model.lm_head"]
    entry = actual.entries["language_model.lm_head"]
    np.testing.assert_array_equal(entry.counts, reference.counts)
    np.testing.assert_array_equal(entry.in_sum2, reference.in_sum2)


def test_oq6_reuses_bf16_imatrix_but_measures_its_own_sensitivity(
    checkpoint, tmp_path, monkeypatch
):
    source, _, _ = checkpoint
    config = json.loads((source / "config.json").read_text())
    tokens = (mx.arange(128).reshape(8, 16) % 100).astype(mx.int32)
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    cache_path = tmp_path / "bf16.npz"
    cached = oq._load_or_collect_imatrix(
        str(source),
        config,
        cache_path=str(cache_path),
        reuse_cache=True,
        num_samples=2,
        seq_length=16,
        strict=True,
        trust_remote_code=False,
        stream_calibration=True,
        measure_sensitivity=True,
        sensitivity_oq_level=4,
        sensitivity_num_samples=2,
        sensitivity_seq_length=16,
        require_mtp_entries=False,
    )
    original_cache = cache_path.read_bytes()
    measure = oq._measure_sensitivity_streaming

    def measure_small(model_path, model_config, level, **kwargs):
        return measure(
            model_path,
            model_config,
            level,
            num_samples=2,
            seq_length=16,
            trust_remote_code=False,
        )

    measured = Mock(side_effect=measure_small)
    monkeypatch.setattr(oq, "_measure_sensitivity_streaming", measured)
    monkeypatch.setattr(
        oq,
        "_collect_imatrix_streaming",
        Mock(side_effect=AssertionError("must reuse BF16 statistics")),
    )
    output = tmp_path / "oq6"
    oq.quantize_oq_streaming(
        str(source),
        str(output),
        6,
        enhanced=True,
        stream_calibration=True,
        auto_proxy_sensitivity=False,
        imatrix_strict=True,
        imatrix_cache_path=str(cache_path),
        imatrix_reuse_cache=True,
        imatrix_num_samples=2,
        imatrix_seq_length=16,
        preserve_mtp=False,
    )
    measured.assert_called_once()
    assert measured.call_args.args[2] == 6
    assert cache_path.read_bytes() == original_cache
    report = json.loads((output / "oq_imatrix_report.json").read_text())
    assert report["cache_reused"]
    assert report["entry_count"] == len(cached.entries)
    assert report["applied"] and not report["missing"] and not report["mismatched"]
    loaded = load_model(output, lazy=False, strict=True, trust_remote_code=False)
    assert mx.isfinite(loaded(tokens[:1]).logits).all().item()


@pytest.mark.parametrize("checkpoint", [3, 513], indirect=True)
def test_streamed_imatrix_sensitivity_and_cache_match_resident(
    checkpoint, tmp_path, monkeypatch
):
    source, model, _ = checkpoint
    config = asdict(model.config)
    length = config["sliding_window"] + 3
    tokens = (mx.arange(8 * length).reshape(8, length) % 100).astype(mx.int32)
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    real_batch_plan = oq._oqe_calibration_batch_plan

    def batch_plan(*args, **kwargs):
        return {**real_batch_plan(*args, **kwargs), "micro_batch_size": 2}

    monkeypatch.setattr(oq, "_oqe_calibration_batch_plan", batch_plan)
    resident, resident_meta = oq._collect_imatrix_from_model(
        model,
        object(),
        config,
        calib_dataset=oq._OQE_CALIB_DATASET,
        num_samples=2,
        seq_length=length,
    )
    expected_sensitivity = oq._measure_sensitivity_from_model(
        model, object(), config, 4, num_samples=2, seq_length=length
    )
    monkeypatch.setattr(
        "mlx_vlm.utils.load_model",
        Mock(side_effect=AssertionError("must not load the whole model")),
    )
    monkeypatch.setattr(
        oq,
        "_build_proxy_for_sensitivity",
        Mock(side_effect=AssertionError("proxy forbidden")),
    )
    kwargs = dict(
        cache_path=str(tmp_path / "imatrix.npz"),
        reuse_cache=True,
        num_samples=2,
        seq_length=length,
        strict=True,
        trust_remote_code=False,
        stream_calibration=True,
        measure_sensitivity=True,
        sensitivity_oq_level=4,
        sensitivity_num_samples=2,
        sensitivity_seq_length=length,
        require_mtp_entries=False,
    )
    streamed = oq._load_or_collect_imatrix(str(source), config, **kwargs)
    assert "language_model.lm_head" in streamed.entries
    assert resident.keys() == streamed.entries.keys()
    assert streamed.metadata["processed_samples"] == resident_meta["processed_samples"]
    assert streamed.metadata["collection"]["load_kind"] == "streaming"
    if length < 513:
        assert len(streamed.metadata["collection"]["rounds"]) > 1
    assert streamed.metadata["layer_walk"] == "flint_native_masks_fp32_head_v1"
    for name, entry in resident.items():
        np.testing.assert_array_equal(streamed.entries[name].counts, entry.counts)
        np.testing.assert_array_equal(streamed.entries[name].in_sum2, entry.in_sum2)
    measured = streamed.metadata["collection"]["sensitivity_map"]
    assert measured.keys() == expected_sensitivity.keys()
    for index, score in expected_sensitivity.items():
        assert measured[index] == pytest.approx(score, abs=1e-6)
    assert oq._load_or_collect_imatrix(str(source), config, **kwargs).reused
    standalone = oq._measure_sensitivity_streaming(
        str(source), config, 4, num_samples=2, seq_length=length
    )
    assert standalone == pytest.approx(expected_sensitivity, abs=1e-6)
    incompatible = {**streamed.metadata, "layer_walk": "old_generic_mask"}
    assert not oq._oqe_cache_matches(streamed, incompatible)


@pytest.mark.parametrize("enhanced", [False, True])
@pytest.mark.parametrize("stream_calibration", [False, True])
def test_output_strictly_reloads_without_proxy(
    checkpoint, tmp_path, monkeypatch, enhanced, stream_calibration
):
    source, _, _ = checkpoint
    tokens = (mx.arange(128).reshape(8, 16) % 100).astype(mx.int32)
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    monkeypatch.setattr(
        oq,
        "_build_proxy_for_sensitivity",
        Mock(side_effect=AssertionError("proxy forbidden")),
    )
    output = tmp_path / "output"
    oq.quantize_oq_streaming(
        str(source),
        str(output),
        4,
        enhanced=enhanced,
        stream_calibration=stream_calibration,
        auto_proxy_sensitivity=False,
        imatrix_strict=True,
        imatrix_num_samples=2,
        imatrix_seq_length=16,
    )
    loaded = load_model(output, lazy=False, strict=True, trust_remote_code=False)
    assert mx.isfinite(loaded(tokens[:1]).logits).all().item()
    if enhanced:
        report = json.loads((output / "oq_imatrix_report.json").read_text())
        assert report["applied"] and not report["missing"] and not report["mismatched"]


@pytest.mark.parametrize(
    "explicit,exceeds_ram,expected",
    [
        (None, False, False),
        (None, True, True),
        (True, False, True),
        (False, True, False),
    ],
)
def test_existing_calibration_selection(explicit, exceeds_ram, expected):
    assert (
        oq._resolve_stream_calibration(
            explicit, model_exceeds_ram=exceeds_ram, model_type="flint"
        )
        is expected
    )


@pytest.mark.parametrize("failure", ["missing", "discovery"])
def test_oq_sanitizer_failure_never_falls_back(
    checkpoint, tmp_path, monkeypatch, failure
):
    source, model, _ = checkpoint
    sanitizer = Mock(wraps=oq._build_flint_sanitizer(asdict(model.config)))
    monkeypatch.setattr(
        oq,
        "_build_model_sanitizer",
        lambda *a, **kw: None if failure == "missing" else sanitizer,
    )
    discovery = Mock(side_effect=ValueError("unsupported sanitize operation"))
    monkeypatch.setattr(oq, "_discover_sanitize_plan", discovery)
    output = tmp_path / "failed"
    message = (
        "no model sanitizer for flint"
        if failure == "missing"
        else "Flint streaming sanitize discovery failed"
    )
    with pytest.raises(RuntimeError, match=message):
        oq.quantize_oq_streaming(
            str(source), str(output), 4, sensitivity_map_override={0: 1.0}
        )
    sanitizer.assert_not_called()
    assert not list(output.glob("*.safetensors"))
    assert not (output / "config.json").exists()
