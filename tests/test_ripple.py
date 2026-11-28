# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx


def ripple_config():
    return dict(
        model_type="ricursive",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        layer_types=["sliding_attention", "full_attention"],
    )


def test_ripple_sanitizer_and_class_lookup():
    from mlx_lm.utils import _get_classes

    from omlx.oq import _build_model_sanitizer
    from omlx.patches.ripple import model as ripple

    config = ripple_config()
    sanitize = _build_model_sanitizer(config)
    assert sanitize is not None
    weights = {"x.weight": mx.ones((2,)), "rotary_emb.inv_freq": mx.ones((2,))}
    assert list(sanitize(weights)) == ["x.weight"]
    cls, args = _get_classes(config)
    assert cls is ripple.Model
    model = cls(args.from_dict(config))
    logits = model(mx.array([[1, 2]]), cache=model.make_cache())
    assert logits.shape == (1, 2, 128)
    assert mx.all(mx.isfinite(logits)).item()


def test_ripple_rejected_draft_crossing_window():
    from omlx.patches.mlx_lm_mtp import cache_rollback
    from omlx.patches.mlx_lm_mtp.batch_generator import _call_backbone_captured
    from omlx.patches.ripple.model import Model, ModelArgs

    cache_rollback.apply()
    model = Model(ModelArgs.from_dict(dict(ripple_config(), sliding_window=8)))
    a, b = model.make_cache(), model.make_cache()
    prefix = mx.array([[1, 2, 3, 4, 5, 6, 7]])
    mx.eval(model(prefix, cache=a), model(prefix, cache=b))
    block = mx.array([[8, 9, 10, 11, 12]])
    _, _, _, captures = _call_backbone_captured(model, block, b, 1, [0, 1])
    assert len(captures) == 2
    assert model.mtp_partial_rollback(b, 1, 4)
    for token in [8, 9]:
        mx.eval(model(mx.array([[token]]), cache=a))
    follow = mx.array([[70]])
    assert mx.allclose(model(follow, cache=a), model(follow, cache=b), atol=1e-4).item()


def test_ripple_draft_window_and_engine_eligibility():
    from omlx.speculative.dflash import context_bytes_per_row
    from omlx.utils.model_loading import dflash_batched_supported

    config = dict(
        architectures=["DFlash2DraftModel"],
        hidden_size=5120,
        num_attention_heads=80,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=5,
        num_target_layers=80,
        no_rope_layers=[4],
        sliding_window=4096,
        layer_types=["sliding_attention"] * 4 + ["full_attention"],
        dflash_config={"target_layer_ids": [1, 20, 39, 58, 77]},
    )
    assert dflash_batched_supported("ricursive", "batched")
    assert not dflash_batched_supported("ricursive", "vlm")
    assert context_bytes_per_row(config) == context_bytes_per_row(config, 16384)
    assert context_bytes_per_row(config) == 4 * context_bytes_per_row(config, 4096)
