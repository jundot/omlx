# SPDX-License-Identifier: Apache-2.0
"""Partial-miss vision encoding: only uncached images hit the vision tower.

Multi-turn agent loops resend every historical screenshot each turn. When the
vision feature cache holds all but the newest image, the vision tower must
encode only the missing subset — not the whole batch (regression for the
~0.8 s/MP fixed per-turn TTFT floor on qwen-style models).
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.cache.vision_feature_cache import VisionFeatureSSDCache
from omlx.engine.vlm import VLMBatchedEngine

_MODEL = "fake-model"
# t*h*w per image: 4, 4, 8 rows → 1, 1, 2 merged tokens.
_GRIDS = [[1, 2, 2], [1, 2, 2], [1, 2, 4]]


class _FakeTower:
    spatial_merge_size = 2

    def __init__(self):
        self.calls = []
        self.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.bfloat16))
        )

    def __call__(self, pv, grid_thw):
        # Tag every output row with the first input row value so tests can
        # tell which images reached the tower.
        self.calls.append((int(pv.shape[0]), grid_thw.tolist()))
        rows = int(pv.shape[0])
        return mx.full((rows // 4, 8), float(pv[0, 0].item())), []


def _make_engine(tower, cache=None):
    eng = VLMBatchedEngine.__new__(VLMBatchedEngine)
    eng._vlm_model = SimpleNamespace(
        vision_tower=tower, config=SimpleNamespace(model_type="qwen4_exp")
    )
    eng._model_name = _MODEL
    eng._vision_cache = cache
    return eng


@pytest.fixture
def cache():
    c = VisionFeatureSSDCache(cache_dir=None, max_memory_entries=10)
    yield c
    c.close()


def _pixel_values_3imgs():
    parts = [
        mx.zeros((4, 1536)),
        mx.ones((4, 1536)),
        mx.full((8, 1536), 2.0),
    ]
    return mx.concatenate(parts, axis=0)


def test_partial_miss_encodes_only_missing_image(cache):
    cache.put("h0", _MODEL, mx.zeros((1, 8)))
    cache.put("h2", _MODEL, mx.full((2, 8), 2.0))

    tower = _FakeTower()
    eng = _make_engine(tower, cache)
    pv = _pixel_values_3imgs()
    grid = mx.array(_GRIDS)
    cached = [cache.get("h0", _MODEL), None, cache.get("h2", _MODEL)]

    out = eng._encode_missing_vision_features(
        pv,
        {"image_grid_thw": grid},
        cached,
        ["h0", "h1", "h2"],
        image_token_count=4,
    )

    assert out is not None
    assert out.shape == (4, 8)
    # Only image 1's 4 patch rows reached the tower.
    assert tower.calls == [(4, [[1, 2, 2]])]
    # Combined features are in prompt order: img0(1 row)=0, img1(1 row)=1,
    # img2(2 rows)=2.
    vals = out[:, 0].tolist()
    assert vals == [0.0, 1.0, 2.0, 2.0]
    # The newly encoded image landed in the cache.
    assert cache.get("h1", _MODEL) is not None


def test_all_missing_returns_none_to_force_full_encode(cache):
    tower = _FakeTower()
    eng = _make_engine(tower)
    pv = _pixel_values_3imgs()
    grid = mx.array(_GRIDS)

    out = eng._encode_missing_vision_features(
        pv,
        {"image_grid_thw": grid},
        [None, None, None],
        ["h0", "h1", "h2"],
        image_token_count=4,
    )
    assert out is None
    assert tower.calls == []


def test_non_qwen_model_returns_none(cache):
    tower = _FakeTower()
    eng = _make_engine(tower)
    eng._vlm_model.config.model_type = "gemma3"
    pv = _pixel_values_3imgs()
    grid = mx.array(_GRIDS)

    out = eng._encode_missing_vision_features(
        pv,
        {"image_grid_thw": grid},
        [mx.zeros((1, 8)), None, mx.zeros((2, 8))],
        ["h0", "h1", "h2"],
        image_token_count=4,
    )
    assert out is None
    assert tower.calls == []


def test_cached_shape_disagreeing_with_grid_falls_back(cache):
    tower = _FakeTower()
    eng = _make_engine(tower)
    pv = _pixel_values_3imgs()
    grid = mx.array(_GRIDS)
    # Image 0 cached with the wrong token count for its grid.
    cached = [mx.zeros((3, 8)), None, mx.zeros((2, 8))]

    out = eng._encode_missing_vision_features(
        pv,
        {"image_grid_thw": grid},
        cached,
        ["h0", "h1", "h2"],
        image_token_count=4,
    )
    assert out is None
    assert tower.calls == []


def test_token_count_mismatch_rejects_combination(cache):
    tower = _FakeTower()
    eng = _make_engine(tower)
    pv = _pixel_values_3imgs()
    grid = mx.array(_GRIDS)
    cached = [mx.zeros((1, 8)), None, mx.zeros((2, 8))]

    out = eng._encode_missing_vision_features(
        pv,
        {"image_grid_thw": grid},
        cached,
        ["h0", "h1", "h2"],
        image_token_count=99,  # disagrees with 4 feature tokens
    )
    assert out is None
