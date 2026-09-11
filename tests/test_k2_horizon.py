# SPDX-License-Identifier: Apache-2.0
"""K2 model arithmetic, quantization, and cache checks."""

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


@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("length", [1, 8, 512])
def test_grouped_norm(groups, length):
    x = mx.arange(length * 64).reshape(1, length, 64).astype(mx.bfloat16)
    actual = GroupedRMSNorm(64, groups, 1e-5)(x)
    grouped = x.astype(mx.float32).reshape(1, length, groups, 64 // groups)
    expected = (
        grouped * mx.rsqrt(mx.mean(grouped**2, -1, keepdims=True) + 1e-5)
    ).reshape(x.shape)
    assert mx.allclose(actual.astype(mx.float32), expected, atol=0.01).item()


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
    cache = model.make_cache()
    model(mx.array([[2, 3]]), cache=cache)
    cycle = decoder.cycle(4, cache=cache, frontier=3, max_tokens=16)
    expected = (
        list(range(10, 19))
        if reject_at is None
        else list(range(10, 11 + reject_at)) + [90]
    )
    assert list(cycle.tokens) == expected
    assert cycle.accepted_proposals == (7 if reject_at is None else reject_at)
    assert all(c.offset == 3 + len(expected) - 1 for c in cache)
    keys = model.cache[0].state[0]
    assert keys[0, 0, :, 0].tolist() == [2, 3, 4] + expected[:-1]


def run_uno_cycles(decoder, prompt, max_tokens, cache=None):
    cache = make_prompt_cache(decoder.model) if cache is None else cache
    if cache[0].offset < len(prompt) - 1:
        decoder.model(mx.array([prompt[cache[0].offset : -1]]), cache=cache)
    seed, frontier = prompt[-1], len(prompt)
    while max_tokens:
        cycle = decoder.cycle(
            seed, cache=cache, frontier=frontier, max_tokens=max_tokens
        )
        yield cycle
        if cycle.tokens[-1] in decoder.eos:
            break
        seed = cycle.tokens[-1]
        frontier += len(cycle.tokens)
        max_tokens -= len(cycle.tokens)


@pytest.mark.parametrize("eos_slot", range(9))
def test_eos_at_every_committed_slot_excludes_later_draft_tokens(eos_slot):
    model = _ScriptedModel()
    decoder = UnoDecoder(
        model, eos_token_ids=[10 + eos_slot], block_size=8, temperature=0
    )
    cycles = list(run_uno_cycles(decoder, [2, 3, 4], 16))
    assert len(cycles) == 1
    assert list(cycles[0].tokens) == list(range(10, 11 + eos_slot))
    assert cycles[0].accepted_proposals == min(7, eos_slot)
    assert model.cache[0].state[0][0, 0, :, 0].tolist() == [2, 3, 4] + list(
        range(10, 10 + eos_slot)
    )


@pytest.mark.parametrize("budget", range(1, 9))
def test_budget_shorter_than_block_is_exact(budget):
    decoder = UnoDecoder(
        _ScriptedModel(), eos_token_ids=[], block_size=8, temperature=0
    )
    cycles = list(run_uno_cycles(decoder, [2, 3], budget))
    assert [token for cycle in cycles for token in cycle.tokens] == list(
        range(10, 10 + budget)
    )


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


def test_yarn_rotation_uses_each_batch_offset():
    from omlx.patches.k2_horizon.k2_horizon_model import YarnRoPE

    rope = YarnRoPE(
        SimpleNamespace(
            rope_head_dim=16,
            rope_theta=10000,
            rope_parameters=dict(
                attention_factor=1.0,
                original_max_position_embeddings=128,
                beta_fast=32,
                beta_slow=1,
                factor=4,
            ),
        )
    )
    x = mx.arange(2 * 4 * 8 * 16).reshape(2, 4, 8, 16).astype(mx.bfloat16)
    for length in (1, 8):
        values = x[:, :, :length]
        actual = rope(values, offset=mx.array([4096, 8192]))
        expected = mx.concatenate(
            [
                rope(values[i : i + 1], offset=offset)
                for i, offset in enumerate((4096, 8192))
            ]
        )
        assert mx.array_equal(actual, expected).item()


def test_router_bias_only_changes_selection():
    from omlx.patches.k2_horizon.k2_horizon_model import route

    x = mx.ones((2, 1, 4), mx.bfloat16)
    weight = mx.zeros((3, 4), mx.bfloat16)
    bias = mx.array([0.0, 0.1, 0.2])
    for top_k, scale in [(1, 1.0), (2, 2.5), (1, 3.0)]:
        indices, weights = route(x, weight, bias, top_k, scale)
        expected = mx.broadcast_to(mx.arange(3 - top_k, 3), indices.shape)
        assert mx.array_equal(mx.sort(indices), expected).item()
        assert mx.all(weights == scale / top_k).item()


@pytest.mark.parametrize(
    "top_k, expected",
    [
        (None, [[0.625, 0.375, 0, 0], [0, 0, 3 / 7, 4 / 7]]),
        (1, [[1, 0, 0, 0], [0, 0, 0, 1]]),
    ],
)
def test_uno_probabilities_restore_vocabulary_order(top_k, expected):
    from omlx.patches.k2_horizon.uno_decode import probabilities

    logits = mx.log(mx.array([[0.5, 0.3, 0.2, 0], [0.1, 0.2, 0.3, 0.4]]))
    actual = probabilities(logits, temperature=1, top_p=0.6, top_k=top_k)
    assert mx.allclose(actual, mx.array(expected), atol=1e-6).item()


def test_uno_rejects_untrained_block_size():
    with pytest.raises(ValueError, match="1, 8"):
        UnoDecoder(
            SimpleNamespace(_uno_adapter_loaded=True), eos_token_ids={0}, block_size=9
        )


@pytest.mark.parametrize("cancel_warm", [False, True])
def test_uno_reuses_ssd_prefix_after_reload(
    tmp_path, mock_tokenizer, monkeypatch, cancel_warm
):
    from omlx.patches.k2_horizon.compiled import install_compiled_blocks
    from omlx.patches.k2_horizon.uno_batch import install_cache_hooks
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig

    mx.random.seed(81)
    model = Model(ModelArgs.from_dict(small_config()))
    model._uno_adapter_loaded = model._omlx_uno_enabled = True
    model._omlx_uno_eos = []
    install_compiled_blocks(model)
    install_cache_hooks()
    mock_tokenizer.eos_token_id = None
    mock_tokenizer.convert_tokens_to_ids = lambda _: None
    config = SchedulerConfig(
        model_name="k2-base:k2-compiled-v1",
        paged_ssd_cache_dir=str(tmp_path),
        paged_ssd_cache_max_size=1024**2,
        paged_cache_block_size=4,
        hot_cache_max_size=0,
    )
    prompt = list(range(2, 19))
    calls = []
    original = UnoDecoder.cycle

    def record(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(UnoDecoder, "cycle", record)
    results = []
    for attempt in range(3 if cancel_warm else 2):
        scheduler = Scheduler(model, mock_tokenizer, config)
        request = Request(
            request_id=f"request-{attempt}",
            prompt=prompt,
            sampling_params=SamplingParams(max_tokens=24, temperature=0),
        )
        try:
            scheduler.add_request(request)
            for _ in range(100):
                scheduler.step()
                if cancel_warm and attempt == 1 and request.num_output_tokens >= 2:
                    scheduler.abort_request(request.request_id)
                if request.is_finished():
                    break
            assert request.is_finished()
            results.append((request.cached_tokens, list(request.output_token_ids)))
        finally:
            scheduler.shutdown()
    assert calls, "The scheduler must execute Uno cycles"
    assert len(results[0][1]) == 24
    assert results[0][0] == 0
    assert results[-1][0] == 16
    assert results[0][1] == results[-1][1]
    if cancel_warm:
        assert 0 < len(results[1][1]) < len(results[0][1])
        assert results[1][1] == results[0][1][: len(results[1][1])]


@pytest.mark.parametrize("reject_at", [None, 2])
def test_uno_restored_prefix_preserves_only_verified_kv(reject_at):
    model = _ScriptedModel(reject_at)
    prompt = [2, 3, 4, 5, 6]
    cache = model.make_cache()
    model(mx.array([prompt[:3]]), cache=cache)
    decoder = UnoDecoder(model, eos_token_ids=[], block_size=8, temperature=0)
    cycles = list(run_uno_cycles(decoder, prompt, 9, cache))
    emitted = [token for cycle in cycles for token in cycle.tokens]
    assert cache[0].state[0][0, 0, :, 0].tolist() == prompt + emitted[:-1]
    assert cache[0].offset == len(prompt) + len(emitted) - 1
