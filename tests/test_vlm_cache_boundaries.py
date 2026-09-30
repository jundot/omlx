"""Cache identity must follow the final multimodal token sequence."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from PIL import Image

from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.engine.vlm import VLMBatchedEngine, _audio_feature_cache_key_ranges
from omlx.utils.image import compute_image_hash


def prepare_case(full, prefixes, *, grid=None, counts=(1, 1), model_type="qwen3_5"):
    engine = VLMBatchedEngine(model_name="boundary-test")
    engine._processor = MagicMock()
    engine._processor.image_processor.merge_size = 2
    engine._processor.apply_chat_template.side_effect = lambda messages, **kwargs: str(
        len(messages)
    )
    engine._vlm_model = MagicMock()
    engine._vlm_model.config.model_type = model_type
    engine._vlm_model.config.image_token_id = 99
    engine._vlm_model.get_input_embeddings.return_value = SimpleNamespace(
        inputs_embeds=mx.zeros((1, len(full), 1))
    )
    engine._vision_cache = None
    messages = [{"role": "user", "content": "text"}] * (2 * len(counts) + 1)
    ranges = [(2 * i + 1, count) for i, count in enumerate(counts)]
    engine._format_messages_for_vlm_template = MagicMock(
        return_value=(messages, ranges)
    )
    images = [Image.new("RGB", (4, 4), (i * 40, 0, 0)) for i in range(sum(counts))]

    def prepare(processor, images=None, prompts=None, **kwargs):
        index = int(prompts[0])
        ids = full if index == len(messages) else prefixes[index]
        data = {"input_ids": mx.array([ids]), "pixel_values": mx.zeros((1, 1))}
        if grid is not None and index == len(messages):
            data["image_grid_thw"] = mx.array(grid)
        return data

    with patch("mlx_vlm.utils.prepare_inputs", side_effect=prepare) as mocked:
        result = engine._prepare_vision_inputs(messages, images)
    return result, images, mocked.call_count


def reused_tokens(tokens, ranges_a, ranges_b, hash_a, hash_b):
    manager = PagedCacheManager(
        block_size=4, max_blocks=100, initial_blocks=100, model_name="boundary-test"
    )
    model = MagicMock()
    model.layers = [MagicMock()]
    cache = BlockAwarePrefixCache(model=model, paged_cache_manager=manager)
    keys = mx.ones((1, 1, len(tokens), 1))
    data = [{"state": (keys, keys), "cache_type": "KVCache", "class_name": "KVCache"}]

    def kwargs(ranges, image_hash):
        return dict(
            extra_keys=(image_hash,),
            extra_key_token_start=ranges[0][0],
            extra_key_ranges=[(start, (key,)) for start, key in ranges],
        )

    cache.store_cache("a", tokens, data, **kwargs(ranges_a, hash_a))
    table, _ = cache.fetch_cache("b", tokens, **kwargs(ranges_b, hash_b))
    return table.num_tokens if table else 0


@pytest.mark.parametrize("first_prefix_len", [10, 60])
def test_grid_boundaries_ignore_rerendered_reasoning(first_prefix_len):
    full = [1] * 4 + [99] * 4 + [2] * 4 + [99] * 4 + [3] * 4
    result, images, calls = prepare_case(
        full,
        {1: [1] * first_prefix_len, 3: [2] * 16},
        grid=[[1, 4, 4], [1, 4, 4]],
    )
    tokens, _, _, whole_hash, start, ranges = result
    assert tokens == full
    assert start == 4
    assert [a for a, _ in ranges] == [4, 12]
    assert calls == 1  # The full processor output already identifies image spans.
    changed = [(a, "different-" + h) for a, h in ranges]
    assert reused_tokens(tokens, ranges, changed, whole_hash, "changed") == 4
    later = [ranges[0], (ranges[1][0], "later-image")]
    assert reused_tokens(tokens, ranges, later, whole_hash, "later") == 12
    assert reused_tokens(tokens, ranges, ranges, whole_hash, whole_hash) == 20
    assert ranges[0][1] == compute_image_hash(images[:1])


def test_adjacent_images_and_multiple_images_per_turn():
    full = [1] * 4 + [99] * 12 + [2] * 4 + [99] * 4 + [3] * 4
    result, images, calls = prepare_case(
        full,
        {1: [1] * 4, 3: full[:20]},
        counts=(2, 1),
        grid=[[1, 4, 4], [1, 4, 8], [1, 4, 4]],
    )
    assert result[5] == [
        (4, compute_image_hash(images[:2])),
        (20, compute_image_hash(images)),
    ]
    assert calls == 1


@pytest.mark.parametrize(
    "prefixes, expected",
    [
        ({1: [1, 1, 7] * 10, 3: [1, 1, 99, 99, 7]}, [2, 4]),
        ({1: [1, 1, 99], 3: [1, 7]}, [1, 1]),
        ({1: [1, 1], 3: [1, 1, 99, 99, 2, 2]}, [2, 6]),
    ],
)
def test_non_grid_boundaries_use_only_matching_final_tokens(prefixes, expected):
    full = [1, 1, 99, 99, 2, 2, 99, 99, 3, 3]
    result, _, _ = prepare_case(full, prefixes, model_type="gemma3")
    assert [a for a, _ in result[5]] == expected
    assert result[0] == full


def test_grid_layout_mismatch_does_not_publish_partial_ranges():
    full = [1] * 4 + [99] * 4 + [2] * 4
    result, _, _ = prepare_case(
        full, {1: [1] * 4, 3: full}, grid=[[1, 4, 4], [1, 4, 4]]
    )
    assert result[4] == 0
    assert result[5] == []
    assert result[3] is not None  # Existing whole-request image key remains available.


@pytest.mark.parametrize("text_prefix", [0, 1, 3, 4, 5, 63, 64, 65])
@pytest.mark.parametrize("patch_grid", [[1, 2, 2], [1, 4, 8], [2, 4, 4]])
def test_grid_boundaries_at_block_edges(text_prefix, patch_grid):
    count = patch_grid[0] * patch_grid[1] * patch_grid[2] // 4
    full = [1] * text_prefix + [99] * count + [2] * 8
    result, _, calls = prepare_case(
        full, {1: [7] * 100}, counts=(1,), grid=[patch_grid]
    )
    assert result[4] == text_prefix
    assert result[5][0][0] == text_prefix
    assert calls == 1


def test_generic_receding_boundary_salts_with_all_images():
    result, images, _ = prepare_case(
        [1] * 4 + [99] * 4 + [2] * 4 + [99] * 4,
        {1: [1] * 4, 3: [1, 7]},
        model_type="gemma3",
    )
    from omlx.cache.paged_cache import resolve_block_extra_keys

    ranges = [(start, (key,)) for start, key in result[5]]
    assert resolve_block_extra_keys(4, extra_key_ranges=ranges) == (
        compute_image_hash(images),
    )


@pytest.mark.parametrize(
    "grid, tokens",
    [
        ([[1, 0, 4]], [99]),
        ([[1, 3, 3]], [99, 99]),
        ([[1, 4, 4]], [99, 99, 0, 99, 99]),
        ([[1, 4, 4]], [99, 99, 99]),
    ],
)
def test_invalid_grid_never_exposes_unkeyed_image_blocks(grid, tokens):
    result, _, _ = prepare_case(tokens, {1: [1, 2]}, counts=(1,), grid=grid)
    assert result[4] == 0
    assert result[5] == []
    assert result[3] is not None


AUDIO = 98


def prepare_audio_case(ids, features, mask):
    engine = VLMBatchedEngine(model_name="audio-boundary-test")
    engine._processor = MagicMock()
    engine._processor.apply_chat_template.return_value = "prompt"
    engine._vlm_model = MagicMock()
    engine._vlm_model.config.model_type = "gemma4"
    engine._vlm_model.config.audio_token_id = AUDIO
    engine._vlm_model.get_input_embeddings.return_value = SimpleNamespace(
        inputs_embeds=mx.zeros((1, len(ids), 1))
    )
    engine._vision_cache = None
    messages = [{"role": "user", "content": "text"}]
    engine._format_messages_for_vlm_template = MagicMock(return_value=(messages, []))

    def prepare(processor, images=None, prompts=None, **kwargs):
        return {
            "input_ids": mx.array([ids]),
            "input_features": features,
            "input_features_mask": mask,
        }

    clips = [(mx.zeros((16000,)), 16000)] * features.shape[0]
    with patch("mlx_vlm.utils.prepare_inputs", side_effect=prepare):
        return engine._prepare_vision_inputs(messages, [], audio=clips)


def test_equal_length_audio_clips_do_not_share_cached_blocks():
    # Placeholder tokens depend only on clip length; content must key the cache.
    ids = [1] * 4 + [AUDIO] * 4 + [2] * 4
    mask = mx.ones((1, 6), dtype=mx.bool_)
    a = prepare_audio_case(ids, mx.zeros((1, 6, 3)), mask)
    b = prepare_audio_case(ids, mx.ones((1, 6, 3)), mask)
    assert a[0] == b[0] == ids
    assert a[4] == 4
    assert [start for start, _ in a[5]] == [4]
    assert reused_tokens(ids, a[5], b[5], a[3], b[3]) == 4
    assert reused_tokens(ids, a[5], a[5], a[3], a[3]) == 12


def test_audio_clip_key_ignores_padding_from_longer_later_clip():
    one = _audio_feature_cache_key_ranges(
        [1, AUDIO, AUDIO, 2],
        mx.full((1, 3, 2), 0.5),
        mx.ones((1, 3), dtype=mx.bool_),
        AUDIO,
        [],
    )
    features = mx.concatenate(
        [
            mx.concatenate([mx.full((1, 3, 2), 0.5), mx.zeros((1, 2, 2))], axis=1),
            mx.ones((1, 5, 2)),
        ]
    )
    mask = mx.array([[True] * 3 + [False] * 2, [True] * 5])
    two = _audio_feature_cache_key_ranges(
        [1, AUDIO, AUDIO, 2, AUDIO, AUDIO, AUDIO, 3], features, mask, AUDIO, []
    )
    assert [start for start, _ in two] == [1, 4]
    assert two[0] == one[0]


def test_audio_mask_that_is_not_right_padding_keeps_full_row():
    # A mask whose valid frames are not a leading run must not drop content.
    mask = mx.array([[False, True, True]])
    a = _audio_feature_cache_key_ranges(
        [AUDIO], mx.array([[[1.0], [2.0], [3.0]]]), mask, AUDIO, []
    )
    b = _audio_feature_cache_key_ranges(
        [AUDIO], mx.array([[[1.0], [2.0], [4.0]]]), mask, AUDIO, []
    )
    assert a[0][1] != b[0][1]


def test_audio_keys_merge_with_image_ranges():
    ids = [99, 99, AUDIO, AUDIO, 1, 99, 99]
    args = (mx.zeros((1, 2, 2)), mx.ones((1, 2), dtype=mx.bool_), AUDIO)
    ranges = _audio_feature_cache_key_ranges(ids, *args, [(0, "i1"), (5, "i2")])
    changed = _audio_feature_cache_key_ranges(ids, *args, [(0, "i1"), (5, "x2")])
    assert [start for start, _ in ranges] == [0, 2, 5]
    assert ranges[0] == (0, "i1")
    assert ranges[:2] == changed[:2]
    assert ranges[2][1] != changed[2][1]


@pytest.mark.parametrize("audio_token_id", [AUDIO, None])
def test_unmatched_audio_runs_key_from_first_audio_token_or_start(audio_token_id):
    # Two token runs but one feature row: fall back to one whole-audio key.
    ids = [1, AUDIO, 2, AUDIO, 3]
    ranges = _audio_feature_cache_key_ranges(
        ids, mx.zeros((1, 2, 2)), None, audio_token_id, []
    )
    other = _audio_feature_cache_key_ranges(
        ids, mx.ones((1, 2, 2)), None, audio_token_id, []
    )
    assert [start for start, _ in ranges] == [1 if audio_token_id else 0]
    assert ranges[0][1] != other[0][1]
