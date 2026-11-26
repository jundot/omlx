# SPDX-License-Identifier: Apache-2.0
"""Match split affine Hy4 MoE verification with the stock projection path."""

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches.deepseek_v4.switch_layers import SwitchGLU  # noqa: E402
from omlx.patches.hy_v4 import (  # noqa: E402
    exact,
    language,
    verify_dispatch,
    verify_moe,
)
from omlx.patches.hy_v4.layers import ClampedSwiGLU  # noqa: E402


def test_dispatch_arms_only_short_non_row_exact_gpu_verify(monkeypatch):
    monkeypatch.setattr(verify_dispatch, "_supported_gpu", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert not verify_dispatch.eligible(4)
    try:
        language.set_dspark_verify_armed(True)
        assert verify_dispatch.eligible(2)
        assert verify_dispatch.eligible(6)
        assert not verify_dispatch.eligible(1)
        assert not verify_dispatch.eligible(7)
        exact.set_armed(True)
        assert not verify_dispatch.eligible(4)
    finally:
        language.set_dspark_verify_armed(False)
    assert not verify_dispatch.eligible(4)


@pytest.fixture
def fast_gpu():
    if not mx.metal.is_available() or not verify_dispatch._supported_gpu():
        pytest.skip("short Hy4 verify kernels require the validated MLX/Metal backend")
    with mx.stream(mx.gpu):
        yield


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("experts", [16, 24])
def test_fused_routed_experts_match_stock_slot_order(fast_gpu, bits, experts):
    mx.random.seed(315)
    switch = SwitchGLU(1536, 1536, experts, activation=ClampedSwiGLU(7.0))
    for name, precision in (("gate_proj", bits), ("up_proj", bits), ("down_proj", 4)):
        projection = switch[name]
        projection.weight = projection.weight.astype(mx.bfloat16)
        switch[name] = projection.to_quantized(group_size=64, bits=precision)
    switch.eval()
    mx.eval(switch.parameters())
    routed = (mx.random.normal((4, 1536)) * 0.2).astype(mx.bfloat16)
    indices = mx.argsort(mx.random.uniform(shape=(4, experts)), axis=-1)[..., :16]
    scores = mx.softmax(mx.random.normal((4, 16)), axis=-1)

    expected = switch(routed, indices, scores=scores, weighted_sum=True)
    actual = verify_moe.routed_experts(
        switch, routed, indices, scores, switch.activation.limit
    )
    if experts == 16:
        # The stock sorted path uses a matrix kernel at this route density.
        assert actual is None
        return
    assert actual is not None
    assert mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item()
