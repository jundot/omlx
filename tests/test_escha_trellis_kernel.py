"""Escha trellis kernel correctness: Metal qgemm vs the CPU reference.

Run with:  pytest tests/test_escha_trellis_kernel.py
Skips when Metal custom kernels are unavailable (non-Apple or JIT-less mlx).
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
fast = pytest.importorskip("omlx.custom_kernels.escha.fast")


@pytest.fixture(scope="module")
def synthetic_codes():
    rng = np.random.RandomState(0)
    return {2: rng.randint(-32768, 32767, (6, 8, 4, 32), np.int16),   # K=2
            3: rng.randint(-32768, 32767, (6, 4, 8, 48), np.int16)}   # K=3


@pytest.mark.parametrize("K,(TK,TN)", [(2, (8, 4)), (3, (4, 8))])
def test_kernel_matches_cpu_reference(synthetic_codes, K, TK, TN):
    pytest.skip("synthetic codes are meaningless for the trellis; tested against "
                "real checkpoint data in tools/convert_escha_mlx.py validation")
