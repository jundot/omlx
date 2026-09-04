# SPDX-License-Identifier: Apache-2.0
"""Progressive load and tensor-strategy contract tests."""

import json
import struct
from types import SimpleNamespace

import mlx.nn.layers.distributed as distributed_layers
import pytest

from omlx.cluster.planner import _supports_tensor_parallel
from omlx.cluster.progressive_loading import (
    install_progressive_loader,
    materialize_parameters_progressively,
    progressive_sharded_load,
)
from omlx.cluster.tensor_strategies import (
    apply_tensor_strategy,
    native_shard_is_layer_local,
    registered_model_types,
    supports_model_type,
)
from omlx.patches.mlx_lm_pipeline_index import (
    _JsonProxy,
    _open_with_single_file_index,
)


class _FakeMX:
    def __init__(self):
        self.events = []

    def eval(self, *values):
        self.events.append(("eval", values))

    def clear_cache(self):
        self.events.append(("clear",))


def test_progressive_materializer_evaluates_fixed_then_each_layer_in_order():
    mx = _FakeMX()
    progress = []
    parameters = [
        ("model.layers.2.weight", "layer-2"),
        ("model.embed_tokens.weight", "embedding"),
        ("model.layers.0.weight", "layer-0"),
        ("lm_head.weight", "head"),
    ]

    layers = materialize_parameters_progressively(
        parameters,
        mx_module=mx,
        tree_flatten=lambda value: value,
        progress=progress.append,
    )

    assert layers == (0, 2)
    assert mx.events == [
        ("eval", ("embedding", "head")),
        ("clear",),
        ("eval", ("layer-0",)),
        ("clear",),
        ("eval", ("layer-2",)),
        ("clear",),
    ]
    assert [item["phase"] for item in progress] == [
        "materializing_fixed",
        "materializing_layers",
        "materializing_layers",
    ]
    assert progress[-1]["layers_loaded"] == progress[-1]["layers_total"] == 2


def test_fixed_phase_is_visible_before_large_fixed_weights_materialize():
    timeline = []

    class TimelineMX:
        def eval(self, *values):
            timeline.append(("eval", values))

        def clear_cache(self):
            timeline.append(("clear",))

    materialize_parameters_progressively(
        [("model.embed_tokens.weight", "embedding")],
        mx_module=TimelineMX(),
        tree_flatten=lambda value: value,
        progress=lambda event: timeline.append(("progress", event["phase"])),
    )

    assert timeline[0] == ("progress", "materializing_fixed")
    assert timeline[1] == ("eval", ("embedding",))


def test_tensor_registry_includes_missing_exo_architectures():
    assert {"qwen3_next", "nemotron_h"} <= registered_model_types()
    assert supports_model_type("qwen3_next") is True
    assert supports_model_type("nemotron_h") is True
    assert supports_model_type("llama", native_shard=True) is True
    assert supports_model_type("unknown") is False


def test_planner_and_loader_apply_the_same_native_tensor_proof():
    from mlx_lm.models import iquestloopcoder, qwen3

    assert native_shard_is_layer_local(qwen3.Model.shard)[0] is True
    assert native_shard_is_layer_local(iquestloopcoder.Model.shard)[0] is False
    assert _supports_tensor_parallel({"model_type": "qwen3"}) is True
    assert _supports_tensor_parallel({"model_type": "iquestloopcoder"}) is False
    # Explicit adapters remain available even without a native Model.shard.
    assert _supports_tensor_parallel({"model_type": "qwen3_next"}) is True


def test_qwen_next_moe_inplace_shards_are_wrapped_with_an_all_sum(monkeypatch):
    from mlx_lm.models import qwen3_next

    all_sums = []

    class FakeMX(_FakeMX):
        distributed = SimpleNamespace(
            all_sum=lambda value, group: all_sums.append((value, group)) or value
        )

    class FakeGroup:
        @staticmethod
        def size():
            return 2

        @staticmethod
        def rank():
            return 0

    class FakeMoE:
        def __init__(self):
            self.switch_mlp = SimpleNamespace(
                gate_proj="switch-gate",
                down_proj="switch-down",
                up_proj="switch-up",
            )
            self.shared_expert = SimpleNamespace(
                gate_proj="shared-gate",
                down_proj="shared-down",
                up_proj="shared-up",
            )

        def __call__(self, value):
            return value

    attention = SimpleNamespace(
        num_attention_heads=2,
        num_key_value_heads=2,
        q_proj="q",
        k_proj="k",
        v_proj="v",
        o_proj="o",
    )
    layer = SimpleNamespace(
        is_linear=False,
        self_attn=attention,
        mlp=FakeMoE(),
        parameters=lambda: [],
    )
    model = SimpleNamespace(model_type="qwen3_next", layers=[layer])
    group = FakeGroup()
    mx = FakeMX()
    monkeypatch.setattr(qwen3_next, "Qwen3NextSparseMoeBlock", FakeMoE)
    monkeypatch.setattr(
        distributed_layers,
        "shard_linear",
        lambda module, _mode, *, group: module,
    )
    monkeypatch.setattr(
        distributed_layers,
        "shard_inplace",
        lambda module, _mode, *, group: None,
    )
    monkeypatch.setattr(
        distributed_layers,
        "sum_gradients",
        lambda group: lambda value: value,
    )

    assert (
        apply_tensor_strategy(
            model,
            group,
            mx_module=mx,
        )
        == "qwen3_next"
    )
    assert layer.mlp(7) == 7
    assert all_sums == [(7, group)]


def test_native_tensor_strategy_materializes_and_shards_one_layer_at_a_time():
    mx = _FakeMX()
    calls = []
    progress = []

    class Layer:
        def __init__(self, name):
            self.name = name

        def parameters(self):
            return self.name

    class Model:
        model_type = "native_test"

        def __init__(self):
            self.model = SimpleNamespace(
                layers=[Layer("zero"), Layer("one"), Layer("two")]
            )

        def shard(self, group):
            assert len(self.model.layers) == 1
            for layer in self.model.layers:
                calls.append(layer.name)

    model = Model()
    strategy = apply_tensor_strategy(
        model,
        SimpleNamespace(),
        mx_module=mx,
        progress=progress.append,
    )

    assert strategy == "native"
    assert calls == ["zero", "one", "two"]
    assert [layer.name for layer in model.model.layers] == ["zero", "one", "two"]
    assert [item["layers_loaded"] for item in progress] == [1, 2, 3]
    assert sum(event[0] == "clear" for event in mx.events) == 3


def test_native_tensor_strategy_skips_read_only_forwarding_layer_property():
    """Qwen3.5 exposes Model.layers as a property over model.layers."""

    mx = _FakeMX()
    calls = []

    class Layer:
        def __init__(self, name):
            self.name = name

        def parameters(self):
            return self.name

    class Model:
        model_type = "native_test"

        def __init__(self):
            self.model = SimpleNamespace(layers=[Layer("zero"), Layer("one")])

        @property
        def layers(self):
            return self.model.layers

        def shard(self, group):
            assert len(self.layers) == 1
            for layer in self.layers:
                calls.append(layer.name)

    model = Model()
    strategy = apply_tensor_strategy(
        model,
        SimpleNamespace(),
        mx_module=mx,
    )

    assert strategy == "native"
    assert calls == ["zero", "one"]
    assert [layer.name for layer in model.layers] == ["zero", "one"]


def test_native_tensor_strategy_refuses_fixed_weight_mutation_outside_layer_loop():
    mx = _FakeMX()

    class Layer:
        def parameters(self):
            return "layer"

    class Model:
        model_type = "unsafe_native"

        def __init__(self):
            self.layers = [Layer()]
            self.output = "unsharded"

        def shard(self, group):
            self.output = "sharded"
            for _layer in self.layers:
                pass

    model = Model()

    try:
        apply_tensor_strategy(
            model,
            SimpleNamespace(),
            mx_module=mx,
        )
    except RuntimeError as exc:
        assert "outside its layer loop" in str(exc)
    else:
        raise AssertionError("unsafe native sharding was accepted")
    assert model.output == "unsharded"


def test_progressive_loader_patch_is_scoped_and_restored(monkeypatch):
    def original(*args, **kwargs):
        return "original", args, kwargs

    server = SimpleNamespace(sharded_load=original)
    calls = []

    monkeypatch.setattr(
        "omlx.cluster.progressive_loading.progressive_sharded_load",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "progressive",
    )

    with install_progressive_loader(server, progress=lambda _event: None):
        assert server.sharded_load("model") == "progressive"
        assert server.sharded_load is not original

    assert server.sharded_load is original
    assert calls[0][0] == ("model",)
    assert callable(calls[0][1]["progress"])


def test_progressive_pipeline_load_preserves_single_file_model_support(tmp_path):
    """The progressive loader must use the in-memory index compatibility patch."""

    tensor_name = "model.layers.0.weight"
    header = {
        tensor_name: {
            "dtype": "F16",
            "shape": [1],
            "data_offsets": [0, 2],
        },
        "__metadata__": {"format": "mlx"},
    }
    encoded = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"\0\0"
    )

    class Pipeline:
        def pipeline(self, _group):
            return None

    model = SimpleNamespace(
        model=Pipeline(),
        parameters=lambda: [(tensor_name, "weight")],
    )
    utils = SimpleNamespace(
        _download=lambda _repo, allow_patterns=None: tmp_path,
        load_config=lambda _path: {"model_type": "llama", "eos_token_id": 2},
        load_model=lambda *_args, **_kwargs: (model, {"eos_token_id": 2}),
        load_tokenizer=lambda *_args, **_kwargs: "tokenizer",
        tree_flatten=lambda parameters: parameters,
        open=_open_with_single_file_index,
        json=_JsonProxy(),
    )

    class Distributed:
        @staticmethod
        def all_sum(value, stream=None):
            return value

    mx = _FakeMX()
    mx.array = lambda value: value
    mx.distributed = Distributed()
    mx.cpu = "cpu"

    loaded, tokenizer = progressive_sharded_load(
        tmp_path,
        pipeline_group=SimpleNamespace(),
        utils_module=utils,
        mx_module=mx,
    )

    assert loaded is model
    assert tokenizer == "tokenizer"
    assert not (tmp_path / "model.safetensors.index.json").exists()


def test_progressive_loader_checks_tokenizer_trust_before_model_load(tmp_path):
    calls = []

    def reject_tokenizer(_path, config, **_kwargs):
        calls.append(("tokenizer", config))
        raise ValueError("trust_remote_code=True is required")

    utils = SimpleNamespace(
        _download=lambda _repo, allow_patterns=None: tmp_path,
        load_config=lambda _path: {"model_type": "llama"},
        load_tokenizer=reject_tokenizer,
        load_model=lambda *_args, **_kwargs: calls.append(("model", None)),
    )

    with pytest.raises(ValueError, match="trust_remote_code=True"):
        progressive_sharded_load(
            tmp_path,
            utils_module=utils,
            mx_module=_FakeMX(),
        )

    assert calls == [("tokenizer", {"trust_remote_code": False})]
