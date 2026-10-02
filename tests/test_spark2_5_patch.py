from mlx_lm.models import spark2_5
from mlx_lm.models.cache import KVCache, RotatingKVCache
import mlx.core as mx


def test_upstream_spark2_5_cached_decode_matches_full_prompt():
    args = spark2_5.ModelArgs(
        model_type="spark2_5",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_theta": 10000.0},
            "full_attention": {"rope_theta": 10000.0},
        },
    )
    mx.random.seed(1)
    model = spark2_5.Model(args)
    cache = model.make_cache()
    assert isinstance(cache[0], RotatingKVCache)
    assert isinstance(cache[1], KVCache)
    tokens = mx.array([[1, 2, 3, 4, 5, 6]])
    full = model(tokens)[:, -1]
    model(tokens[:, :-1], cache=cache)
    cached = model(tokens[:, -1:], cache=cache)[:, -1]
    assert mx.allclose(full, cached, atol=1e-5).item()
