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
