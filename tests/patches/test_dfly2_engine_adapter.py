import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.dfly2 import (
    DFly2Config,
    attach,
    make_cache,
    make_stages,
    map_official_weights,
)
from omlx.patches.deepseek_v41_dfly2.dsv41_bridge import DSV41TargetBridge


class _TinyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.mtp = make_stages(config)


def _config():
    config = ModelConfig(
        vocab_size=17,
        dim=32,
        moe_inter_dim=32,
        n_layers=2,
        n_mtp_layers=2,
        n_heads=1,
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=16,
        o_groups=1,
        o_lora_rank=32,
        window_size=4,
        compress_ratios=(0, 0, 0, 0),
        kv_source_layers=(),
        index_source_layers=(),
        index_n_heads=1,
        index_head_dim=32,
        index_topk=2,
        hc_mult=1,
        preserve_mtp=True,
    )
    config.dfly_config = {
        "block_size": 3,
        "mask_token_id": 0,
        "target_kv_layers": [0, 1],
        "target_index_layers": [0, 1],
        "index_topk": 2,
        "attentive_correction": {"max_block_slots": 16},
        "hidden_correction": {"intermediate_size": 16},
    }
    return config


def _official_tensors(model):
    tensors = {}
    for name, value in tree_flatten(model.parameters()):
        if not name.startswith("mtp."):
            continue
        if name.startswith("mtp.0.heads."):
            official = "model." + name.removeprefix("mtp.0.heads.")
        else:
            _mtp, stage, core, *suffix = name.split(".")
            assert core == "core"
            official = f"model.layers.{stage}." + ".".join(suffix)
        tensors[official] = mx.full(value.shape, 0.25, value.dtype)
    return tensors


def test_make_cache_and_stages_follow_dfly_config():
    config = _config()
    stages = make_stages(config)
    cache = make_cache(config)
    assert len(stages) == len(cache) == 2
    assert [item.target_layer for item in cache] == [0, 1]
    assert all(item.index_topk == 2 for item in cache)
    assert stages[0].heads.block_size == 3


def test_weight_mapping_is_strict_and_round_trips_real_name_patterns():
    model = _TinyModel(_config())
    tensors = _official_tensors(model)
    mapped = dict(map_official_weights(model, tensors))
    assert set(mapped) == {name for name, _ in tree_flatten(model.parameters()) if name.startswith("mtp.")}
    assert mx.array_equal(mapped["mtp.0.core.attn.wq_a.weight"], tensors["model.layers.0.attn.wq_a.weight"])
    assert mx.array_equal(mapped["mtp.0.heads.mtp_fusion.eh_proj.weight"], tensors["model.mtp_fusion.eh_proj.weight"])
    assert mx.array_equal(
        mapped["mtp.0.heads.hidden_correction.attention.q_in.weight"],
        tensors["model.hidden_correction.attention.q_in.weight"],
    )

    with pytest.raises(KeyError, match="unknown"):
        map_official_weights(model, {**tensors, "model.layers.9.attn.wq_a.weight": mx.zeros((1, 1))})
    missing = dict(tensors)
    missing.pop("model.layers.1.attn.wq_a.weight")
    with pytest.raises(KeyError, match="missing"):
        map_official_weights(model, missing)


class _FakeRuntime:
    def __init__(self, config):
        self.cfg = DFly2Config.from_config(config)
        self.seen_histories = None

    def block_hidden(self, _embed, hidden_rows, _anchor, start, histories, width):
        self.seen_histories = histories
        assert hidden_rows.shape == (1, 1, 32)
        assert start == 2
        return mx.zeros((1, width, 32))

    def correct(self, _embed, hidden, _previous):
        return hidden


class _Host:
    def __init__(self, config):
        self._config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.dfly = _FakeRuntime(config)
        self._dfly_cache = None


def _bridge():
    bridge = DSV41TargetBridge(target_layers=(0, 1), index_topk=2, local_window=0)
    for layer in (0, 1):
        bridge.record(
            layer,
            start=0,
            window_latents=mx.arange(64, dtype=mx.float32).reshape(1, 2, 32),
            global_positions=mx.zeros((0,), dtype=mx.int32),
            global_latents=mx.zeros((1, 0, 32)),
            # Anchor position 2 consumes picks from position 1: one real row
            # and one invalid sentinel that must remain masked.
            index_topk=mx.array([[[0, -1], [0, -1]]], dtype=mx.int32),
        )
    return bridge


def test_attached_methods_have_no_mtp_cache_and_need_target_forward():
    attach(_Host)
    host = _Host(_config())
    assert host.make_mtp_cache() == []
    assert host.dspark_append_context(mx.zeros((1, 1, 32)), []) is None
    with pytest.raises(RuntimeError, match="target forward"):
        host.dspark_forward(mx.zeros((1, 1, 32)), mx.array([[1]], dtype=mx.int32), [])


def test_attached_forward_builds_masked_bridge_histories():
    attach(_Host)
    host = _Host(_config())
    host._dfly_cache = _bridge()
    block, state = host.dspark_forward(
        mx.zeros((1, 1, 32)), mx.array([[1]], dtype=mx.int32), [], draft_length=2
    )
    mx.eval(block)
    assert state is None
    assert block.shape == (1, 3, 32)
    assert len(host.dfly.seen_histories) == 2
    for history in host.dfly.seen_histories:
        assert history.latents.shape == (1, 2, 32)
        assert history.valid.shape == (1, 2)
        assert history.valid.tolist() == [[True, False]]


def test_attach_installs_all_generic_generator_methods():
    class _Unpatched:
        pass

    attach(_Unpatched)
    for name in ("make_mtp_cache", "dspark_append_context", "dspark_forward", "dspark_step_logits"):
        assert callable(getattr(_Unpatched, name))


def test_attached_forward_rejects_mismatched_target_stage_offsets():
    attach(_Host)
    host = _Host(_config())
    host._dfly_cache = _bridge()
    host._dfly_cache.record(
        1,
        start=2,
        window_latents=mx.zeros((1, 1, 32)),
        global_positions=mx.zeros((0,), dtype=mx.int32),
        global_latents=mx.zeros((1, 0, 32)),
        index_topk=mx.zeros((1, 1, 2), dtype=mx.int32),
    )
    with pytest.raises(ValueError, match="stage offsets differ"):
        host.dspark_forward(mx.zeros((1, 1, 32)), mx.array([[1]], dtype=mx.int32), [])


def test_bind_bridge_installs_capture_and_tears_down(monkeypatch):
    """The seam must be reversible: a request may not leak its bridge."""
    from types import SimpleNamespace

    from omlx.patches.deepseek_v41.dfly2 import bind_bridge
    from omlx.patches.deepseek_v41_dfly2 import DSV41TargetBridge

    attentions = {layer: SimpleNamespace() for layer in (28, 32, 36)}
    model = SimpleNamespace(
        _config={"dfly_config": {"block_size": 6, "mask_token_id": 128799, "index_topk": 512,
                                  "target_kv_layers": [28, 32, 36], "target_index_layers": [28, 32, 36],
                                  "hidden_correction": {"intermediate_size": 5120}}},
        layers={layer: SimpleNamespace(self_attn=attention) for layer, attention in attentions.items()},
    )
    bridge = DSV41TargetBridge(target_layers=(28, 32, 36))

    teardown = bind_bridge(model, bridge)
    for attention in attentions.values():
        assert getattr(attention, "_omlx_dfly_capture", None) is not None
    assert model._dfly_cache is bridge

    teardown()
    for attention in attentions.values():
        assert not hasattr(attention, "_omlx_dfly_capture")
    assert model._dfly_cache is None


def test_bridge_anchor_and_rollback_drop_rejected_tail():
    """A rejected verify tail must never survive into the next draft."""
    import mlx.core as mx

    from omlx.patches.deepseek_v41_dfly2 import DSV41TargetBridge

    bridge = DSV41TargetBridge(target_layers=(28,), index_topk=2, local_window=2)
    with pytest.raises(RuntimeError):
        bridge.committed_anchor()

    def row(values):
        return mx.array([[[float(v), float(v)] for v in values]])

    def picks():
        return mx.array([[[0, 1]]], dtype=mx.int32)

    bridge.record(layer=28, start=0, window_latents=row([0, 1, 2]), global_positions=mx.array([], dtype=mx.int32),
                  global_latents=mx.zeros((1, 0, 2)), index_topk=mx.array([[[0, 1], [0, 1], [0, 1]]], dtype=mx.int32))
    assert bridge.committed_anchor() == 3
    bridge.record(layer=28, start=3, window_latents=row([3, 4]), global_positions=mx.array([], dtype=mx.int32),
                  global_latents=mx.zeros((1, 0, 2)), index_topk=mx.array([[[0, 1], [0, 1]]], dtype=mx.int32))
    assert bridge.committed_anchor() == 5
    bridge.rollback(keep_positions=bridge.committed_anchor())
    ends = [segment.end for segment in bridge._window[28]]
    assert max(ends) == 5, ends


def test_capture_at_position_zero_starts_a_fresh_bridge():
    """A new prompt must not draft against the previous request's rows."""
    import mlx.core as mx

    from omlx.patches.deepseek_v41_dfly2 import capture_attention_slice
    from omlx.patches.deepseek_v41_dfly2.dsv41_bridge import DSV41TargetBridge

    bridge = DSV41TargetBridge(target_layers=(28,), index_topk=2, local_window=2)
    row = mx.zeros((1, 2, 4))
    picks = mx.array([[[0, -1], [1, -1]]], dtype=mx.int32)

    capture_attention_slice(
        bridge=bridge, layer=28, start=40, ratio=4,
        window_latents=row, compressed_latents=row, compressed_start=40,
        native_index_topk=picks,
    )
    assert bridge._window[28]

    capture_attention_slice(
        bridge=bridge, layer=28, start=0, ratio=4,
        window_latents=row, compressed_latents=row, compressed_start=0,
        native_index_topk=picks,
    )
    ends = [segment.end for segment in bridge._window[28]]
    assert ends == [2], ends
