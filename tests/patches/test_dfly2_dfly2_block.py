import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from omlx.patches.deepseek_v41_dfly2.dfly2_block import draft_block
from omlx.patches.deepseek_v41_dfly2.dsv41_bridge import DSV41History


def _identity_layer(x, _start, history):
    # The draft layer itself appends the block's B6 keys, so bridge history is
    # only K+W (=4 here), not K+W+B.
    assert history.indices.shape[-1] == 4
    return x


def test_draft_block_applies_fusion_then_previous_token_correction_left_to_right():
    seen_previous = []

    def embed(ids):
        return ids.astype(mx.float32)[..., None]

    def fuse(anchor, target):
        return anchor + target

    def correct(hidden, previous):
        seen_previous.append(np.asarray(previous))
        return hidden + previous

    def head(hidden):
        # Vocabulary logits choose previous+1 as greedy token.
        value = hidden[..., 0].astype(mx.int32)
        return (mx.arange(32, dtype=mx.int32)[None, :] == (value + 1)[..., None]).astype(mx.float32) * 10.0

    histories = [
        DSV41History(
            indices=mx.array([[0, 1, 2, 3]], dtype=mx.int32),
            valid=mx.ones((1, 4), dtype=mx.bool_),
            latents=mx.zeros((1, 4, 1)),
        )
    ]
    tokens, logits = draft_block(
        embed=embed,
        fusion=fuse,
        layers=(_identity_layer,),
        final_norm=lambda x: x,
        correct=correct,
        lm_head=head,
        histories=histories,
        bonus_token=mx.array([2], dtype=mx.int32),
        target_hidden=mx.array([[[3.0]]]),
        mask_token_id=0,
        block_size=4,
        start_position=5,
    )
    mx.eval(tokens, *logits)
    # Slot 1 sees bonus 2; later slots see earlier generated tokens.
    np.testing.assert_array_equal(tokens, [[3, 4, 5]])
    assert [int(item.reshape(-1)[0]) for item in seen_previous] == [2, 3, 4]
    assert len(logits) == 3


def test_draft_block_rejects_bridge_history_count_mismatch():
    try:
        draft_block(
            embed=lambda ids: ids.astype(mx.float32)[..., None],
            fusion=lambda a, _h: a,
            layers=(_identity_layer,),
            final_norm=lambda x: x,
            correct=lambda h, _p: h,
            lm_head=lambda h: mx.one_hot(mx.zeros(h.shape[:-1], mx.int32), 2),
            histories=(),
            bonus_token=mx.array([1], dtype=mx.int32),
            target_hidden=mx.zeros((1, 1, 1)),
            mask_token_id=0,
            block_size=2,
            start_position=1,
        )
    except ValueError as exc:
        assert "history" in str(exc)
    else:
        raise AssertionError("missing per-layer history was accepted")
