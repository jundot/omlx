import sys

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin-only native API")
def test_native_shared_cluster_cpu_matmul():
    mx = pytest.importorskip("mlx.core")
    from omlx.custom_kernels.qwen35_prefill import fast

    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    if not fast.has_symbol("qwen35_cpu_fp16_affine_qmm_t"):
        pytest.skip("Qwen3.5 native custom kernel is unavailable")
    if not fast.qwen35_cpu_shared_resource_available():
        pytest.skip("shared-cluster dispatch_apply is unavailable")

    rows, input_dim = 2048, 128
    cpu_outputs = gpu_outputs = 64
    x = mx.ones((1, rows, input_dim), dtype=mx.float16)
    cpu_weight = mx.ones((cpu_outputs, input_dim), dtype=mx.float16)
    gpu_weight = mx.zeros(
        (gpu_outputs, input_dim * 4 // 32), dtype=mx.uint32
    )
    gpu_scales = mx.zeros(
        (gpu_outputs, input_dim // 128), dtype=mx.float16
    )
    gpu_biases = mx.zeros_like(gpu_scales)

    result = fast.qwen35_cpu_fp16_affine_qmm_t(
        x,
        cpu_weight,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        4,
        group_size=128,
        cpu_threads=8,
        cpu_shared_resource=True,
    )
    mx.eval(result)

    expected = mx.full(
        (1, rows, cpu_outputs), input_dim, dtype=mx.float16
    )
    assert result.shape == (1, rows, cpu_outputs + gpu_outputs)
    assert bool(mx.all(mx.isfinite(result)).item())
    assert float(mx.max(mx.abs(result[..., :cpu_outputs] - expected)).item()) == 0
