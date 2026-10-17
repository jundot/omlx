# SPDX-License-Identifier: Apache-2.0
"""M1 key-split sizing for the tensor-op verify attention."""

from __future__ import annotations

from omlx.patches import qwen35_verify_sdpa_split as vs


def test_about_one_threadgroup_per_core_within_bounds():
    # M1 Max (32 cores), Qwen3.8 dense layout (4 KV heads)
    assert vs._m1_gqa_chunk(32768, 4, cores=32) == 4096  # exactly one wave
    # Qwen3.6 MoE layout (2 KV heads)
    assert vs._m1_gqa_chunk(32768, 2, cores=32) == 2048
    # never below 1024 keys while one wave fits, always a multiple of 64
    assert vs._m1_gqa_chunk(17000, 1, cores=32) == 1024
    for cores in (14, 16, 24, 32, 48, 64):
        for kv_len in (16385, 50000, 99999, 133112):
            for heads in (1, 2, 4, 8):
                chunk = vs._m1_gqa_chunk(kv_len, heads, cores=cores)
                assert 512 <= chunk <= 4096 and chunk % 64 == 0


def test_no_nearly_empty_second_wave():
    """Threadgroups (splits x KV heads) fit one wave or fill many; never just past the cores."""
    for cores in (14, 16, 24, 32, 48, 64):
        for heads in (1, 2, 4, 8):
            for kv_len in range(16385, 262145, 1021):
                chunk = vs._m1_gqa_chunk(kv_len, heads, cores=cores)
                groups = -(-kv_len // chunk) * heads
                assert groups <= cores or groups >= 6 * cores, (
                    cores,
                    heads,
                    kv_len,
                    chunk,
                    groups,
                )


def test_reported_long_context_cases():
    # M1 Ultra, 2 KV heads: 100k keys stays one wave, 133k leaves it (66 groups on 64 cores before)
    assert vs._m1_gqa_chunk(100710, 2, cores=64) == 3200
    assert vs._m1_gqa_chunk(133112, 2, cores=64) == 576
    # M1 Max, 2 KV heads: 76.8k keys used 4096 (38 groups on 32 cores)
    assert vs._m1_gqa_chunk(76800, 2, cores=32) == 640
    assert vs._m1_gqa_chunk(133112, 2, cores=32) == 1088


def test_more_cores_give_more_splits():
    # M1 Ultra (64 cores) needs twice the splits of an M1 Max for the same context.
    kv_len, kv_heads = 27063, 2
    max_splits = -(-kv_len // vs._m1_gqa_chunk(kv_len, kv_heads, cores=32))
    ultra_splits = -(-kv_len // vs._m1_gqa_chunk(kv_len, kv_heads, cores=64))
    assert max_splits == 16
    assert (
        ultra_splits == 27
    )  # one wave on 32 cores is 16 groups; 27 splits x 2 heads = 54 groups on 64 cores


def test_core_count_comes_from_the_hardware_with_an_m1_max_fallback(monkeypatch):
    from omlx.utils import hardware

    vs._m1_gpu_cores.cache_clear()
    monkeypatch.setattr(hardware, "get_gpu_core_count", lambda: 64)
    assert vs._m1_gqa_chunk(27063, 2) == vs._m1_gqa_chunk(27063, 2, cores=64)
    vs._m1_gpu_cores.cache_clear()
    monkeypatch.setattr(hardware, "get_gpu_core_count", lambda: None)
    assert vs._m1_gpu_cores() == 32
    assert vs._m1_gqa_chunk(27063, 2) == vs._m1_gqa_chunk(27063, 2, cores=32)
    vs._m1_gpu_cores.cache_clear()
