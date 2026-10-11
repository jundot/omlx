# SPDX-License-Identifier: Apache-2.0
"""KV accounting for hybrid linear-attention stacks (gated-delta-net + full attention).

Qwen3.5/3.6/3.8 dense interleave three linear-attention layers with one
full-attention layer. Only the full-attention layers grow a KV cache, so neither
the planner's reservation nor the per-rank prefill guard may charge every layer.
"""

from omlx.cluster.planner import _kv_bytes_per_token_per_layer

HYBRID_TEXT_CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 5120,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "num_hidden_layers": 64,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
}


def test_hybrid_kv_reservation_counts_only_full_attention_layers():
    config = {"model_type": "qwen3_5", "text_config": HYBRID_TEXT_CONFIG}
    per_layer = 4 * 256 * 2 * 2
    # 16 of 64 layers hold a KV cache: the per-layer average is a quarter.
    assert _kv_bytes_per_token_per_layer(config) == per_layer // 4
    # => 64 KiB per token over the whole model, not 256 KiB.
    assert _kv_bytes_per_token_per_layer(config) * 64 == 64 * 1024


def test_non_hybrid_kv_reservation_is_unchanged():
    config = {"num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256}
    assert _kv_bytes_per_token_per_layer(config) == 4 * 256 * 2 * 2
    all_full = dict(config, layer_types=["full_attention"] * 8)
    assert _kv_bytes_per_token_per_layer(all_full) == 4 * 256 * 2 * 2


def _classified_monitor(kv_layers):
    def fake_set_model_info_from_model(monitor, model):
        monitor.set_model_info(
            num_layers=64,
            num_kv_heads=4,
            head_dim=256,
            dtype_size=2,
            num_attention_heads=24,
            num_kv_cache_layers=kv_layers,
        )

    return fake_set_model_info_from_model


def test_rank_prefill_guard_charges_only_the_stage_full_attention_layers(monkeypatch):
    from omlx import memory_monitor
    from omlx.cluster.prefill_guard import rank_monitor

    # This stage owns 33 layers, 8 of them full attention (make_cache() on a stage).
    monkeypatch.setattr(
        memory_monitor, "set_model_info_from_model", _classified_monitor(8)
    )
    monitor = rank_monitor(object(), layer_count=33)
    assert monitor._num_kv_cache_layers == 8


def test_rank_prefill_guard_never_charges_more_than_the_stage_holds(monkeypatch):
    from omlx import memory_monitor
    from omlx.cluster.prefill_guard import rank_monitor

    # Classification covered the whole model (64 KV layers): clamp to the stage.
    monkeypatch.setattr(
        memory_monitor, "set_model_info_from_model", _classified_monitor(64)
    )
    assert rank_monitor(object(), layer_count=33)._num_kv_cache_layers == 33


def test_rank_prefill_guard_keeps_sliding_window_layers(monkeypatch):
    from omlx import memory_monitor
    from omlx.cluster.prefill_guard import rank_monitor

    def classified(monitor, model):
        monitor.set_model_info(
            num_layers=36,
            num_kv_heads=8,
            head_dim=64,
            dtype_size=2,
            num_attention_heads=64,
            num_kv_cache_layers=18,
            rotating_layer_specs=[(18, 128)],
        )

    monkeypatch.setattr(memory_monitor, "set_model_info_from_model", classified)
    tp = rank_monitor(object(), tensor_parallel_size=2)
    assert tp._rotating_layer_specs == ((18, 128),)
    assert tp.estimate_resident_kv_bytes(
        32768, chunk_tokens=2048
    ) > tp.estimate_prompt_kv_bytes(32768)
    # A 20-layer stage has room for only 2 sliding layers next to 18 full ones.
    assert rank_monitor(object(), layer_count=20)._rotating_layer_specs == ((2, 128),)
