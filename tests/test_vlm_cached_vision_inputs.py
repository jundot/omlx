# SPDX-License-Identifier: Apache-2.0
"""Hash-first cached vision inputs: cached images skip the image processor.

Multi-turn agent loops resend every history image each turn. Once an image's
features AND patch grid are in the vision feature cache, the HF image
processor (resize/normalize/patchify) for that image is pure repeated work —
it used to run before the per-image feature cache was even consulted. These
tests cover ``VLMBatchedEngine._try_build_cached_vision_inputs``: the
processor-free token rebuild, the miss-only preprocessing path, and every
guard that must fall back to the full path.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from PIL import Image

from omlx.cache.vision_feature_cache import VisionFeatureSSDCache
from omlx.engine.vlm import VLMBatchedEngine
from omlx.utils.image import compute_per_image_hashes

_MODEL = "fake-model"

# Token ids for the fake tokenizer (qwen-style marker triple).
VS_ID = 1
VE_ID = 2
IT_ID = 3
VS_TOK = "<vs>"
VE_TOK = "<ve>"
IT_TOK = "<it>"
MARKER = VS_TOK + IT_TOK + VE_TOK
MERGE = 2


class _FakeTokenizer:
    """Deterministic plain-text tokenizer; marker tokens are fixed ids."""

    def convert_ids_to_tokens(self, token_id):
        return {VS_ID: VS_TOK, VE_ID: VE_TOK, IT_ID: IT_TOK}.get(
            token_id, f"<id{token_id}>"
        )

    def __call__(self, text, add_special_tokens=False):
        # The fast path tokenizes only marker-free text chunks.
        assert VS_TOK not in text and IT_TOK not in text and VE_TOK not in text
        return {"input_ids": [100 + ord(c) for c in text]}


class _FakeTower:
    spatial_merge_size = MERGE

    def __init__(self):
        self.calls = []
        self.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.bfloat16))
        )

    def __call__(self, pixel_values, grid_thw):
        self.calls.append((int(pixel_values.shape[0]), grid_thw.tolist()))
        rows = int(pixel_values.shape[0])
        return mx.full((rows // (MERGE * MERGE), 8), 5.0), []


def _make_engine(tower, cache):
    eng = VLMBatchedEngine.__new__(VLMBatchedEngine)
    eng._vlm_model = SimpleNamespace(
        vision_tower=tower,
        config=SimpleNamespace(
            model_type="qwen4_exp",
            vision_start_token_id=VS_ID,
            vision_end_token_id=VE_ID,
            image_token_id=IT_ID,
        ),
    )
    eng._model_name = _MODEL
    eng._vision_cache = cache
    eng._vision_cache_enabled = True
    eng._processor = SimpleNamespace(
        tokenizer=_FakeTokenizer(),
        image_processor=SimpleNamespace(merge_size=MERGE),
    )
    return eng


def _images():
    return [
        Image.new("RGB", (8, 8), (10, 20, 30)),
        Image.new("RGB", (8, 8), (40, 50, 60)),
    ]


@pytest.fixture
def cache():
    c = VisionFeatureSSDCache(cache_dir=None, max_memory_entries=10)
    yield c
    c.close()


# ── cache-layer grid plumbing ─────────────────────────────────────────


def test_put_get_grid_roundtrip_memory(cache):
    cache.put("h0", _MODEL, mx.zeros((1, 8)), grid=[1, 2, 2])
    assert cache.get_grid("h0", _MODEL) == [1, 2, 2]
    assert cache.get_grid("missing", _MODEL) is None


def test_legacy_entry_without_grid(cache):
    cache.put("h0", _MODEL, mx.zeros((1, 8)))
    assert cache.get("h0", _MODEL) is not None
    assert cache.get_grid("h0", _MODEL) is None


def test_grid_survives_ssd_roundtrip(tmp_path):
    cache = VisionFeatureSSDCache(
        cache_dir=tmp_path / "vc", max_memory_entries=10, max_size_bytes=10**7
    )
    cache.put("h0", _MODEL, mx.full((4, 8), 3.0), grid=[1, 4, 4])
    cache.close()

    reopened = VisionFeatureSSDCache(
        cache_dir=tmp_path / "vc", max_memory_entries=10, max_size_bytes=10**7
    )
    try:
        assert reopened.get_grid("h0", _MODEL) == [1, 4, 4]
        feat = reopened.get("h0", _MODEL)
        assert feat is not None and feat.shape == (4, 8)
        # Memory promotion carries the grid too.
        assert reopened.get_grid("h0", _MODEL) == [1, 4, 4]
    finally:
        reopened.close()


# ── fast path: token rebuild with all images cached ───────────────────


def test_all_hit_skips_image_processor(cache, monkeypatch):
    imgs = _images()
    hashes = compute_per_image_hashes(imgs)
    # img0 grid [1,4,4] -> 4 merged tokens; img1 [1,2,2] -> 1.
    cache.put(hashes[0], _MODEL, mx.full((4, 8), 7.0), grid=[1, 4, 4])
    cache.put(hashes[1], _MODEL, mx.full((1, 8), 9.0), grid=[1, 2, 2])

    def _no_processor(*a, **k):
        raise AssertionError("image processor must not run when all cached")

    monkeypatch.setattr("mlx_vlm.utils.prepare_inputs", _no_processor)
    eng = _make_engine(_FakeTower(), cache)

    out = eng._try_build_cached_vision_inputs(f"A{MARKER}B{MARKER}C", imgs)
    assert out is not None

    ids = out["input_ids"][0].tolist()
    assert ids == [
        100 + ord("A"),
        VS_ID, IT_ID, IT_ID, IT_ID, IT_ID, VE_ID,
        100 + ord("B"),
        VS_ID, IT_ID, VE_ID,
        100 + ord("C"),
    ]
    assert out["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 2]]
    assert out["mm_token_type_ids"][0].tolist() == [
        1 if t == IT_ID else 0 for t in ids
    ]
    assert out["attention_mask"].shape == (1, len(ids))
    # All cached: pixel_values is an empty non-None tensor (keeps the
    # model's multimodal branch without feeding the tower).
    assert out["pixel_values"].shape[0] == 0
    combined = out["cached_image_features"]
    assert combined.shape == (5, 8)
    # Prompt order preserved: img0 rows then img1 rows.
    assert float(combined[0, 0].item()) == 7.0
    assert float(combined[4, 0].item()) == 9.0


# ── fast path: partial miss ────────────────────────────────────────────


def test_partial_miss_processes_only_missing(cache, monkeypatch):
    imgs = _images()
    hashes = compute_per_image_hashes(imgs)
    cache.put(hashes[0], _MODEL, mx.full((4, 8), 7.0), grid=[1, 4, 4])

    seen = []

    def _fake_prepare(processor, images=None, prompts=None, **kw):
        seen.append(len(images))
        n = len(images)
        return {
            "pixel_values": mx.full((4 * n, 1536), 2.0),
            "image_grid_thw": mx.array([[1, 2, 2]] * n),
        }

    monkeypatch.setattr("mlx_vlm.utils.prepare_inputs", _fake_prepare)
    tower = _FakeTower()
    eng = _make_engine(tower, cache)

    out = eng._try_build_cached_vision_inputs(f"A{MARKER}B{MARKER}C", imgs)
    assert out is not None
    # Only the one missing image reached the processor...
    assert seen == [1]
    # ...and only it reached the vision tower.
    assert tower.calls == [(4, [[1, 2, 2]])]

    ids = out["input_ids"][0].tolist()
    assert ids.count(IT_ID) == 5  # 4 cached + 1 fresh
    assert out["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 2]]
    combined = out["cached_image_features"]
    assert combined.shape == (5, 8)
    assert float(combined[0, 0].item()) == 7.0
    assert float(combined[4, 0].item()) == 5.0
    # The miss is now cached with its grid for future turns.
    assert cache.get_grid(hashes[1], _MODEL) == [1, 2, 2]


# ── guards: every failure must fall back to the full path ─────────────


def test_guard_non_qwen_model(cache):
    eng = _make_engine(_FakeTower(), cache)
    eng._vlm_model.config.model_type = "gemma3"
    assert eng._try_build_cached_vision_inputs(f"A{MARKER}B", _images()) is None


def test_guard_marker_count_mismatch(cache):
    eng = _make_engine(_FakeTower(), cache)
    # Two images but the prompt carries only one marker triple.
    assert eng._try_build_cached_vision_inputs(f"A{MARKER}B", _images()) is None


def test_guard_nothing_cached(cache):
    eng = _make_engine(_FakeTower(), cache)
    assert (
        eng._try_build_cached_vision_inputs(f"A{MARKER}B{MARKER}C", _images())
        is None
    )


def test_guard_legacy_entry_without_grid(cache):
    # Feature cached before grids existed — without a grid the fast path
    # cannot rebuild placeholder expansion; must fall back.
    imgs = _images()
    hashes = compute_per_image_hashes(imgs)
    cache.put(hashes[0], _MODEL, mx.full((4, 8), 7.0))
    cache.put(hashes[1], _MODEL, mx.full((1, 8), 9.0), grid=[1, 2, 2])
    eng = _make_engine(_FakeTower(), cache)
    assert (
        eng._try_build_cached_vision_inputs(f"A{MARKER}B{MARKER}C", imgs)
        is None
    )


def test_guard_stale_grid_rejected(cache, monkeypatch):
    # Cached grid claims 16 merged tokens but the feature has 4 rows —
    # different resize regime; treat as a miss and reprocess that image.
    imgs = _images()
    hashes = compute_per_image_hashes(imgs)
    cache.put(hashes[0], _MODEL, mx.full((4, 8), 7.0), grid=[1, 4, 4])
    cache.put(hashes[1], _MODEL, mx.full((4, 8), 9.0), grid=[1, 8, 8])

    seen = []

    def _fake_prepare(processor, images=None, prompts=None, **kw):
        seen.append(len(images))
        return {
            "pixel_values": mx.full((4, 1536), 2.0),
            "image_grid_thw": mx.array([[1, 2, 2]]),
        }

    monkeypatch.setattr("mlx_vlm.utils.prepare_inputs", _fake_prepare)
    eng = _make_engine(_FakeTower(), cache)
    # image0 stays a hit; image1 is reprocessed with a fresh grid.
    out = eng._try_build_cached_vision_inputs(f"A{MARKER}B{MARKER}C", imgs)
    assert out is not None
    assert seen == [1]
    assert out["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 2]]
    assert cache.get_grid(hashes[1], _MODEL) == [1, 2, 2]
