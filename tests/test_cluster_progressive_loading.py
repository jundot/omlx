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
    assert {
        "gemma4",
        "gemma4_text",
        "gemma4_unified",
        "kimi_k25",
        "nemotron_h",
        "qwen3_moe",
        "qwen3_next",
        "qwen3_vl",
        "qwen3_vl_moe",
    } <= registered_model_types()
    assert supports_model_type("qwen3_next") is True
    assert supports_model_type("nemotron_h") is True
    assert supports_model_type("qwen3_vl_moe") is True
    assert supports_model_type("gemma4_unified") is True
    assert supports_model_type("llama", native_shard=True) is True
    assert supports_model_type("unknown") is False


def test_planner_and_loader_apply_the_same_native_tensor_proof():
    from mlx_lm.models import iquestloopcoder, qwen3

    assert native_shard_is_layer_local(qwen3.Model.shard)[0] is True
    bound_owner = type("BoundShardOwner", (), {"shard": qwen3.Model.shard})()
    assert native_shard_is_layer_local(bound_owner.shard)[0] is True
    assert native_shard_is_layer_local(iquestloopcoder.Model.shard)[0] is False
    assert _supports_tensor_parallel({"model_type": "qwen3"}) is True
    assert _supports_tensor_parallel({"model_type": "iquestloopcoder"}) is False
    # Explicit adapters remain available even without a native Model.shard.
    assert _supports_tensor_parallel({"model_type": "qwen3_next"}) is True
    assert _supports_tensor_parallel({"model_type": "qwen3_moe"}) is True
    assert _supports_tensor_parallel({"model_type": "qwen3_vl"}) is True
    assert _supports_tensor_parallel({"model_type": "gemma4"}) is True
    assert _supports_tensor_parallel({"model_type": "kimi_k25"}) is True
    # Capability discovery follows the same official remapping MLX-LM uses
    # when it loads a checkpoint (kimi_k2 -> deepseek_v3).
    assert _supports_tensor_parallel({"model_type": "kimi_k2"}) is True


@pytest.mark.parametrize(
    "model_type, strategy, delegate_owner",
    [
        ("qwen3_vl", "qwen3_vl", "language_model"),
        ("kimi_k25", "kimi_k25", "outer"),
    ],
)
def test_audited_wrapper_delegates_shard_one_layer_at_a_time(
    model_type,
    strategy,
    delegate_owner,
):
    mx = _FakeMX()
    calls = []

    class Layer:
        def __init__(self, name):
            self.name = name
            if model_type == "qwen3_vl":
                self.self_attn = SimpleNamespace(n_heads=4, n_kv_heads=2)
            else:
                self.self_attn = SimpleNamespace(num_heads=4)

        def parameters(self):
            return self.name

    owner = SimpleNamespace(layers=[Layer("zero"), Layer("one")])

    def shard(_group):
        assert len(owner.layers) == 1
        calls.append(owner.layers[0].name)

    language_model = SimpleNamespace(model=owner)
    if delegate_owner == "language_model":
        language_model.shard = shard
    model = SimpleNamespace(
        model_type=model_type,
        language_model=language_model,
    )
    if delegate_owner == "outer":
        model.shard = shard
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    progress = []

    assert (
        apply_tensor_strategy(
            model,
            group,
            mx_module=mx,
            progress=progress.append,
        )
        == strategy
    )
    assert calls == ["zero", "one"]
    assert [layer.name for layer in owner.layers] == ["zero", "one"]
    assert [event["layers_loaded"] for event in progress] == [1, 2]


def test_audited_wrapper_rejects_bad_heads_before_mutating_any_layer():
    mx = _FakeMX()
    calls = []

    class Layer:
        def __init__(self, name, heads):
            self.name = name
            self.self_attn = SimpleNamespace(n_heads=heads, n_kv_heads=2)

        def parameters(self):
            return self.name

    owner = SimpleNamespace(layers=[Layer("good", 4), Layer("bad", 3)])
    language_model = SimpleNamespace(
        model=owner,
        shard=lambda _group: calls.append(owner.layers[0].name),
    )
    model = SimpleNamespace(model_type="qwen3_vl", language_model=language_model)

    with pytest.raises(ValueError, match="attention heads"):
        apply_tensor_strategy(
            model,
            SimpleNamespace(size=lambda: 2, rank=lambda: 0),
            mx_module=mx,
        )

    assert calls == []
    assert [layer.name for layer in owner.layers] == ["good", "bad"]


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


def test_native_tensor_strategy_shards_declared_auxiliary_blocks_progressively():
    mx = _FakeMX()
    calls = []
    progress = []

    class Layer:
        def __init__(self, name):
            self.name = name

        def parameters(self):
            return self.name

    class Auxiliary:
        def __init__(self, name):
            self.block = Layer(name)

        def parameters(self):
            return f"{self.block.name}-owner"

    class Model:
        model_type = "native_test"

        def __init__(self):
            self.model = SimpleNamespace(layers=[Layer("main-0"), Layer("main-1")])
            self.mtp = [Auxiliary("mtp-0"), Auxiliary("mtp-1")]

        def shard(self, group):
            for layer in self.model.layers:
                calls.append(layer.name)

        def _omlx_tensor_auxiliary_modules(self):
            return self.mtp

    model = Model()
    strategy = apply_tensor_strategy(
        model,
        SimpleNamespace(),
        mx_module=mx,
        progress=progress.append,
    )

    assert strategy == "native"
    assert calls == ["main-0", "main-1", "mtp-0", "mtp-1"]
    assert [layer.name for layer in model.model.layers] == ["main-0", "main-1"]
    auxiliary = [event for event in progress if event["phase"].startswith("tensor_aux")]
    assert [event["modules_loaded"] for event in auxiliary] == [1, 2]


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


def test_progressive_hybrid_load_shards_only_the_local_pipeline_stage(
    tmp_path,
    monkeypatch,
):
    """TP must see concrete stage layers, then restore global layer indices."""

    seen = []
    created = []

    class Layer:
        def __init__(self, index):
            self.index = index

        def parameters(self):
            return [(f"weight-{self.index}", self)]

    class Pipeline:
        def __init__(self):
            self.layers = [Layer(index) for index in range(4)]
            self.start_idx = 0
            self.end_idx = 4

        def pipeline(self, _group):
            self.start_idx = 2
            self.end_idx = 4
            self.layers[:2] = [None, None]

    class Model:
        model_type = "native_test"

        def __init__(self):
            self.model = Pipeline()
            created.append(self)

        def parameters(self):
            values = [("embed.weight", "embed")]
            values.extend(
                (f"model.layers.{index}.weight", layer)
                for index, layer in enumerate(self.model.layers)
                if layer is not None
            )
            return values

        def shard(self, _group):
            return None

    def fake_strategy(model, _group, *, mx_module, progress=None):
        seen.append([layer.index for layer in model.model.layers])
        assert all(layer is not None for layer in model.model.layers)
        return "hybrid-test"

    monkeypatch.setattr(
        "omlx.cluster.progressive_loading.apply_tensor_strategy",
        fake_strategy,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "embed.weight": "model.safetensors",
                    "model.layers.2.weight": "model.safetensors",
                    "model.layers.3.weight": "model.safetensors",
                }
            }
        )
    )

    class HybridMX(_FakeMX):
        def set_cache_limit(self, _value):
            return 1024

    utils = SimpleNamespace(
        _download=lambda _repo, allow_patterns=None: tmp_path,
        load_config=lambda _path: {"model_type": "native_test"},
        load_model=lambda *_args, **_kwargs: (Model(), {}),
        load_tokenizer=lambda *_args, **_kwargs: "tokenizer",
        tree_flatten=lambda parameters: parameters,
        open=open,
        json=json,
    )

    loaded, tokenizer = progressive_sharded_load(
        tmp_path,
        pipeline_group=SimpleNamespace(),
        tensor_group=SimpleNamespace(),
        utils_module=utils,
        mx_module=HybridMX(),
    )

    assert tokenizer == "tokenizer"
    assert loaded is created[-1]
    assert seen == [[2, 3]]
    assert loaded.model.layers[:2] == [None, None]
    assert [layer.index for layer in loaded.model.layers[2:]] == [2, 3]


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


def test_tensor_load_does_not_pin_pre_sharded_layer_arrays(monkeypatch):
    """The layer-snapshot list must not survive into the sharding pass.

    Regression: ``flat`` pinned every pre-shard array for the whole strategy
    run, so each materialized layer stayed resident in full next to its
    shard — a TP=2 rank grew toward ~1.5x the whole model until the load
    watchdog killed it (163 GiB against an 84 GiB plan), and a 128 GB node
    ran out of memory entirely. The pre-shard arrays must be unreachable the
    moment the strategy starts swapping in sharded slices.
    """
    import gc
    import weakref

    class Arr:
        def __init__(self, name):
            self.name = name

    class FakeModel:
        model_type = "native_test"

        def __init__(self):
            self.params = {
                "embed.weight": Arr("embed"),
                "lm_head.weight": Arr("head"),
                "mtp.0.weight": Arr("mtp"),
                "model.layers.0.weight": Arr("layer-0"),
                "model.layers.1.weight": Arr("layer-1"),
            }
            self.shard = lambda group: None

        def parameters(self):
            return self.params

    model = FakeModel()
    originals = {
        key: weakref.ref(value)
        for key, value in model.params.items()
        if ".layers." in key
    }
    embed_ref = weakref.ref(model.params["embed.weight"])
    head_ref = weakref.ref(model.params["lm_head.weight"])
    mtp_ref = weakref.ref(model.params["mtp.0.weight"])

    def fake_strategy(shard_model, group, *, mx_module, progress=None):
        # A large output projection must remain lazy until it can be sliced;
        # evaluating it in the replicated fixed phase causes the avoidable
        # full-head memory spike this loader is designed to prevent.
        assert "head" not in mx_module.evaluated
        assert "mtp" not in mx_module.evaluated
        for key in list(shard_model.params):
            if ".layers." in key:
                shard_model.params[key] = Arr(key + " shard")
        shard_model.params["lm_head.weight"] = Arr("head shard")
        shard_model.params["mtp.0.weight"] = Arr("mtp shard")
        gc.collect()
        # Checked *inside* the strategy: after this point a pinned original
        # would sit next to its materialized shard for the rest of the load.
        assert all(ref() is None for ref in originals.values())
        assert head_ref() is None
        assert mtp_ref() is None
        return "native"

    monkeypatch.setattr(
        "omlx.cluster.progressive_loading.apply_tensor_strategy", fake_strategy
    )

    class FakeMX:
        cpu = object()
        distributed = SimpleNamespace(
            all_sum=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("launcher rank_ready markers own the load barrier")
            )
        )

        def __init__(self):
            self.evaluated = []
            self.cache_limits = []

        def array(self, value):
            return value

        def eval(self, *values):
            self.evaluated.extend(
                value.name for value in values if isinstance(value, Arr)
            )

        def clear_cache(self):
            pass

        def set_cache_limit(self, value):
            self.cache_limits.append(value)
            return 1234

    fake_utils = SimpleNamespace(
        _download=lambda repo, allow_patterns=None: repo,
        load_config=lambda path: {},
        load_tokenizer=lambda path, config, eos_token_ids=None: "tokenizer",
        load_model=lambda path, **kwargs: (model, {}),
        tree_flatten=lambda params: list(params.items()),
    )

    fake_mx = FakeMX()
    loaded, tokenizer = progressive_sharded_load(
        "fake-repo",
        tensor_group=SimpleNamespace(),
        utils_module=fake_utils,
        mx_module=fake_mx,
    )

    assert loaded is model
    assert tokenizer == "tokenizer"
    gc.collect()
    assert all(ref() is None for ref in originals.values())
    assert embed_ref() is not None
    assert head_ref() is None
    assert mtp_ref() is None
    assert fake_mx.evaluated[:1] == ["embed"]
    assert "head shard" in fake_mx.evaluated
    assert "mtp shard" in fake_mx.evaluated
    assert fake_mx.cache_limits == [0, 1234]
