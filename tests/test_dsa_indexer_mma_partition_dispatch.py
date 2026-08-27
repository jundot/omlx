"""Architecture dispatch for the qualified DS4 MMA WM4xWN1 partition."""

import pytest

from omlx.custom_kernels.glm_moe_dsa import fast


@pytest.mark.parametrize(
    "device_info,expected",
    [
        ({"device_name": "Apple M3 Ultra", "architecture": "applegpu_g15d"}, True),
        ({"device_name": "Apple M5 Max", "architecture": "applegpu_g17s"}, False),
        ({"device_name": "Apple M3 Ultra", "architecture": "applegpu_g17s"}, False),
        ({"device_name": "Apple M6 Ultra"}, False),
    ],
)
def test_wm4_partition_requires_exact_g15d(monkeypatch, device_info, expected):
    monkeypatch.setattr(fast, "_EXT_MMA_WM4", True)
    assert fast.dsa_indexer_mma_wm4_wn1_eligible(device_info) is expected


def test_wm4_partition_requires_candidate_abi(monkeypatch):
    monkeypatch.setattr(fast, "_EXT_MMA_WM4", False)
    assert not fast.dsa_indexer_mma_wm4_wn1_eligible({"architecture": "applegpu_g15d"})
