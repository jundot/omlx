# SPDX-License-Identifier: Apache-2.0
"""K2 model math, quantization, and conditional decoding contracts."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import make_prompt_cache

from omlx.patches.k2_horizon import apply_k2_horizon_patch
from omlx.patches.k2_horizon.k2_horizon_model import GroupedRMSNorm, Model, ModelArgs
from omlx.patches.k2_horizon.uno_adapter import ConditionalLoRALinear
from omlx.patches.k2_horizon.uno_decode import UnoDecoder, acceptance_and_residual


def small_config(**overrides):
    return dict(
        dict(
            model_type="k2_horizon",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=128,
            rms_norm_eps=1e-5,
            layernorm_num_groups=2,
            mlp_only_layers=[],
            num_experts=0,
            num_experts_per_tok=0,
            moe_intermediate_size=0,
            num_shared_experts=0,
            moe_gate_bias=True,
            norm_topk_prob=True,
            router_score_func="sigmoid",
            router_scaling_factor=1,
            query_key_norm=False,
            rope_parameters={"rope_type": "default", "rope_theta": 10000},
        ),
        **overrides,
    )


@pytest.mark.parametrize("kind", ["dense", "moe", "mova", "yarn", "partial"])
def test_model_cache_and_quantization(kind):
    apply_k2_horizon_patch()
    config = small_config()
    if kind in ("moe", "mova"):
        config.update(
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            moe_intermediate_size=64,
        )
    if kind == "mova":
        config.update(
            mova_num_experts=4,
            mova_num_experts_per_tok=2,
            attention_gate_func="softplus",
        )
    if kind == "yarn":
        config["rope_parameters"].update(
            rope_type="yarn",
            factor=4,
            original_max_position_embeddings=128,
            beta_fast=32,
            beta_slow=1,
            attention_factor=1,
        )
    if kind == "partial":
        config["rope_head_dim"] = 8
    model = Model(ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    ids = mx.array([[1, 2, 3, 4]])
    full = model(ids)
    cache = make_prompt_cache(model)
    model(ids[:, :3], cache=cache)
    tail = model(ids[:, 3:], cache=cache)
    assert mx.allclose(full[:, -1].astype(mx.float32), tail[:, -1], atol=0.04).item()
    from mlx_lm.utils import quantize_model

    model, _ = quantize_model(model, config, group_size=32, bits=4)
    assert mx.all(mx.isfinite(model(ids))).item()
    if kind in ("moe", "mova"):
        assert not isinstance(model.layers[0].mlp.gate, nn.QuantizedLinear)


def test_grouped_norm_and_oq_protection():
    from omlx.oq import universal_quant_predicate

    x = mx.arange(64).reshape(1, 64).astype(mx.bfloat16)
    actual = GroupedRMSNorm(64, 2, 1e-5)(x)
    groups = x.astype(mx.float32).reshape(1, 2, 32)
    expected = (
        groups * mx.rsqrt(mx.mean(groups**2, -1, keepdims=True) + 1e-5)
    ).reshape(1, 64)
    assert mx.allclose(actual.astype(mx.float32), expected, atol=0.01).item()
    config = {"model_type": "k2_horizon"}
    layer = nn.Linear(64, 64)
    assert (
        universal_quant_predicate("model.layers.0.self_attn.v_router", layer, config)
        is False
    )
    assert (
        universal_quant_predicate("model.layers.0.self_attn.q_proj", layer, config)[
            "bits"
        ]
        == 8
    )


@pytest.mark.parametrize("quantized", [False, True])
def test_conditional_adapter_keeps_clean_rows(quantized):
    base = nn.Linear(64, 64, bias=False)
    base.set_dtype(mx.bfloat16)
    if quantized:
        base = base.to_quantized(group_size=32, bits=4)
    a, b = mx.ones((2, 64), mx.bfloat16), mx.ones((64, 2), mx.bfloat16)
    layer = ConditionalLoRALinear(base, a, b, 2)
    x = mx.ones((1, 2, 64), mx.bfloat16)
    actual = layer.conditional_forward(x, mx.array([[0, 1]]))
    assert mx.array_equal(actual[:, 0], base(x)[:, 0]).item()
    assert not mx.array_equal(actual[:, 1], base(x)[:, 1]).item()


def test_acceptance_residual():
    p, q = mx.array([[0.7, 0.3]]), mx.array([[0.4, 0.6]])
    flags, residual = acceptance_and_residual(p, q, mx.array([1]), mx.array([0.8]))
    assert not flags.item()
    assert mx.allclose(residual, mx.array([[1.0, 0.0]])).item()


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_adapter_loading_casts_to_base_activations(tmp_path, quantized, dtype):
    import json

    from omlx.patches.k2_horizon.uno_adapter import TARGETS, load_uno_adapter

    model = Model(ModelArgs.from_dict(small_config()))
    model.set_dtype(mx.bfloat16)
    tensors = {}
    for i, layer in enumerate(model.layers):
        for scope, names in TARGETS.items():
            for name in names:
                out_dims, in_dims = getattr(getattr(layer, scope), name).weight.shape
                prefix = f"model.layers.{i}.{scope}.{name}"
                tensors[f"{prefix}.lora_A.weight"] = mx.ones((2, in_dims), dtype)
                tensors[f"{prefix}.lora_B.weight"] = mx.ones((out_dims, 2), dtype)
    if quantized:
        from mlx_lm.utils import quantize_model

        model, _ = quantize_model(model, small_config(), group_size=32, bits=4)
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "IFM/K2-Horizon-0.9B",
                "r": 2,
                "lora_alpha": 16,
                "bias": "none",
                "fan_in_fan_out": False,
                "target_modules": [
                    name for names in TARGETS.values() for name in names
                ],
            }
        )
    )
    mx.save_safetensors(str(tmp_path / "adapter_model.safetensors"), tensors)
    info = load_uno_adapter(model, tmp_path, base_model_id="IFM/K2-Horizon-0.9B")
    assert info["pairs"] == 14
    assert model.layers[0].self_attn.q_proj.lora_a.dtype == mx.bfloat16


@pytest.mark.parametrize("quantized", [False, True])
def test_indexed_checkpoint_roundtrip(tmp_path, quantized):
    import json

    from mlx.utils import tree_flatten
    from mlx_lm import utils

    apply_k2_horizon_patch()
    config = small_config()
    model = Model(ModelArgs.from_dict(config))
    if quantized:
        model, config = utils.quantize_model(model, config, group_size=32, bits=4)
    weights = dict(tree_flatten(model.parameters()))
    shard = "pytorch_model-00001-of-00001.safetensors"
    mx.save_safetensors(str(tmp_path / shard), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in weights}})
    )
    restored, _ = utils.load_model(tmp_path)
    ids = mx.array([[1, 2, 3]])
    assert mx.array_equal(model(ids), restored(ids)).item()
    (tmp_path / shard).unlink()
    with pytest.raises(FileNotFoundError, match="Missing K2 checkpoint shard"):
        utils.load_model(tmp_path)


class _ScriptedModel:
    """A fixed categorical target with real MLX KV storage for rollback tests."""

    def __init__(self, reject_at=None):
        self.args = SimpleNamespace(vocab_size=128)
        self._uno_adapter_loaded = True
        self.reject_at = reject_at

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        self.cache = [KVCache()]
        return self.cache

    def __call__(self, inputs, cache=None, lora_mask=None):
        if cache is not None:
            values = inputs[:, None, :, None].astype(mx.float32)
            cache[0].update_and_fetch(values, values)
        length = inputs.shape[1]
        if lora_mask is not None:
            tokens = list(range(10, 10 + length))
        else:
            tokens = list(range(11, 11 + length))
            if self.reject_at is not None and self.reject_at < length - 1:
                tokens[self.reject_at] = 90
        return mx.where(
            mx.arange(128)[None, None, :] == mx.array(tokens)[None, :, None],
            0.0,
            -mx.inf,
        )


@pytest.mark.parametrize("reject_at", list(range(7)) + [None])
def test_each_rejection_frontier_and_all_accepted_preserve_real_kv(reject_at):
    model = _ScriptedModel(reject_at)
    decoder = UnoDecoder(model, eos_token_ids=[], block_size=8, temperature=0)
    iterator = decoder.generate([2, 3, 4], max_tokens=16)
    cycle = next(iterator)
    expected = (
        list(range(10, 19))
        if reject_at is None
        else list(range(10, 11 + reject_at)) + [90]
    )
    assert list(cycle.tokens) == expected
    assert cycle.accepted_proposals == (7 if reject_at is None else reject_at)
    assert cycle.cache_length == 3 + len(expected) - 1
    keys = model.cache[0].state[0]
    assert keys[0, 0, :, 0].tolist() == [2, 3, 4] + expected[:-1]
    iterator.close()


@pytest.mark.parametrize("eos_slot", range(9))
def test_eos_at_every_committed_slot_excludes_later_draft_tokens(eos_slot):
    model = _ScriptedModel()
    decoder = UnoDecoder(
        model, eos_token_ids=[10 + eos_slot], block_size=8, temperature=0
    )
    cycles = list(decoder.generate([2, 3, 4], max_tokens=16))
    assert len(cycles) == 1
    assert list(cycles[0].tokens) == list(range(10, 11 + eos_slot))
    assert cycles[0].accepted_proposals == min(7, eos_slot)
    assert cycles[0].finish_reason == "stop"
    assert model.cache[0].state[0][0, 0, :, 0].tolist() == [2, 3, 4] + list(
        range(10, 10 + eos_slot)
    )


@pytest.mark.parametrize("budget", range(9))
def test_budget_shorter_than_block_is_exact(budget):
    decoder = UnoDecoder(
        _ScriptedModel(), eos_token_ids=[], block_size=8, temperature=0
    )
    cycles = list(decoder.generate([2, 3], max_tokens=budget))
    assert [token for cycle in cycles for token in cycle.tokens] == list(
        range(10, 10 + budget)
    )
    if budget:
        assert cycles[-1].finish_reason == "length"


def test_mova_router_preserves_source_partition_rounding():
    from omlx.patches.k2_horizon.k2_horizon_model import router_logits

    x = mx.ones((1, 4), mx.bfloat16)
    weights = mx.array([[1, 1 / 256, -1, 0], [0, 0, 1 / 512, 0]], mx.bfloat16)
    partial = router_logits(x, weights, partitions=2)
    full = router_logits(x, weights, partitions=1)
    assert partial.tolist() == [[0, 1 / 512]]
    assert full.tolist() == [[1 / 256, 1 / 512]]
    assert mx.argmax(partial).item() == 1
    assert mx.argmax(full).item() == 0


def test_uno_rejects_untrained_block_size():
    with pytest.raises(ValueError, match="1, 8"):
        UnoDecoder(
            SimpleNamespace(_uno_adapter_loaded=True), eos_token_ids={0}, block_size=9
        )


def test_uno_prefill_does_not_accumulate_retired_kv_buffers():
    class BufferedModel(_ScriptedModel):
        def __call__(self, inputs, cache=None, lora_mask=None):
            values = mx.broadcast_to(
                inputs[:, None, :, None], (1, 8, inputs.shape[1], 64)
            ).astype(mx.float32)
            cache[0].update_and_fetch(values, values)
            return mx.zeros((1, inputs.shape[1], 128))

    model = BufferedModel()
    decoder = UnoDecoder(model, eos_token_ids=[], prefill_step_size=256)
    cached_bytes = []
    mx.clear_cache()

    def cancelled():
        mx.synchronize()
        cached_bytes.append(mx.get_cache_memory())
        return model.cache[0].offset >= 8192

    assert list(decoder.generate([2] * 8194, max_tokens=1, cancelled=cancelled)) == []
    resident_bytes = sum(value.nbytes for value in model.cache[0].state)
    assert max(cached_bytes) < 2 * resident_bytes


@pytest.mark.parametrize("cancel_warm", [False, True])
def test_uno_reuses_ssd_prefix_after_reload(tmp_path, cancel_warm):
    import threading

    from omlx.engine.uno import UnoEngine, _UnoPrefillGuard
    from omlx.memory_monitor import MemoryMonitor, set_model_info_from_model
    from omlx.scheduler import SchedulerConfig

    model = _ScriptedModel()
    model.layers = [None]
    model.args.num_hidden_layers = 1
    model.args.num_key_value_heads = model.args.num_attention_heads = 1
    model.args.head_dim = model.args.hidden_size = 1
    monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
    set_model_info_from_model(monitor, model)
    config = SchedulerConfig(
        model_name="k2-base",
        paged_ssd_cache_dir=str(tmp_path),
        paged_ssd_cache_max_size=1024**2,
        paged_cache_block_size=4,
    )
    options = dict(max_tokens=8, temperature=0, top_p=1, top_k=0, seed=0)
    prompt = list(range(2, 19))
    results = []
    for attempt in range(3 if cancel_warm else 2):
        engine = UnoEngine("base", adapter_path="unused", scheduler_config=config)
        engine._model = model
        engine._bundle = SimpleNamespace(block_size=4)
        engine._prefill_guard = _UnoPrefillGuard(monitor, 512)
        engine._executor_tokenizer = SimpleNamespace(
            decode=lambda ids: "", eos_token_ids=set()
        )
        engine._output_parser_factory = SimpleNamespace(
            thinking_marker_pairs=[],
            create_session_with_tools=lambda *_: SimpleNamespace(
                process_token=lambda token: SimpleNamespace(
                    stream_text=str(token), is_stop=False
                ),
                finalize=lambda: SimpleNamespace(
                    stream_text="", finish_reason=None, tool_calls=[]
                ),
            ),
        )
        prefix = engine._prefix_cache = engine._init_prefix_cache(model, monitor)
        output = []
        cancelled = threading.Event()

        def publish(
            value, output=output, event=cancelled, abort=cancel_warm and attempt == 1
        ):
            output.append(value)
            if abort:
                event.set()

        try:
            engine._run(prompt, options, [], cancelled, publish)
            results.append(output[-1])
            stats = engine.get_runtime_cache_stats()
            assert stats["block_size"] == 4
            assert stats["prefix_cache"].hits == int(attempt > 0)
            assert not prefix.paged_cache.request_tables
            assert not prefix._request_tables
            assert all(
                block.ref_count == 0
                for block in prefix.paged_cache.allocated_blocks.values()
                if not block.is_null
            )
        finally:
            prefix.paged_ssd_cache.close()
        assert prefix.paged_ssd_cache.get_stats().num_files > 0
    assert results[0].cached_tokens == 0
    assert results[-1].cached_tokens == 16
    assert results[0].tokens == results[-1].tokens
    if cancel_warm:
        assert not results[1].finished
        assert results[1].tokens == results[0].tokens[: len(results[1].tokens)]
    config.model_name = "another-base"
    isolated = UnoEngine("other", adapter_path="unused", scheduler_config=config)
    prefix = isolated._init_prefix_cache(model, monitor)
    try:
        assert prefix.fetch_cache("other", prompt)[0] is None
    finally:
        prefix.paged_ssd_cache.close()


@pytest.mark.parametrize("reject_at", [None, 2])
def test_uno_restored_prefix_preserves_only_verified_kv(reject_at):
    model = _ScriptedModel(reject_at)
    prompt = [2, 3, 4, 5, 6]
    cache = model.make_cache()
    model(mx.array([prompt[:3]]), cache=cache)
    decoder = UnoDecoder(model, eos_token_ids=[], block_size=8, temperature=0)
    cycles = list(decoder.generate(prompt, max_tokens=9, prompt_cache=cache))
    emitted = [token for cycle in cycles for token in cycle.tokens]
    assert cache[0].state[0][0, 0, :, 0].tolist() == prompt + emitted[:-1]
    assert cache[0].offset == len(prompt) + len(emitted) - 1
