"""resolve_vocab_size must size the grammar bitmask from the language model's
vocabulary. A composite (VLM) config's top-level ``vocab_size`` is often a
dataclass placeholder the checkpoint never sets (mlx-vlm's Qwen3-VL
``ModelConfig`` defaults it to 32000 while ``text_config.vocab_size`` is
151936); sizing the mask from it misaligned mask and logits, so structured
output on the VLM engine rejected every sampled token until max_tokens
(#3550)."""

from types import SimpleNamespace

from omlx.utils.tokenizer import resolve_vocab_size


def test_nested_text_config_wins_over_top_level_placeholder():
    config = SimpleNamespace(
        vocab_size=32000,  # mlx-vlm ModelConfig default, not from config.json
        text_config=SimpleNamespace(vocab_size=151936),
    )
    assert resolve_vocab_size(SimpleNamespace(config=config)) == 151936


def test_nested_text_config_as_dict():
    config = SimpleNamespace(vocab_size=32000, text_config={"vocab_size": 151936})
    assert resolve_vocab_size(SimpleNamespace(config=config)) == 151936


def test_top_level_vocab_size_for_plain_llm():
    assert (
        resolve_vocab_size(SimpleNamespace(config=SimpleNamespace(vocab_size=151936)))
        == 151936
    )


def test_top_level_vocab_size_when_text_config_has_none():
    config = SimpleNamespace(
        vocab_size=257152, text_config=SimpleNamespace(hidden_size=16)
    )
    assert resolve_vocab_size(SimpleNamespace(config=config)) == 257152


def test_args_fallback_when_config_is_missing():
    model = SimpleNamespace(args=SimpleNamespace(vocab_size=32064))
    assert resolve_vocab_size(model) == 32064


def test_config_takes_precedence_over_args():
    model = SimpleNamespace(
        config=SimpleNamespace(text_config={"vocab_size": 151936}),
        args=SimpleNamespace(vocab_size=32000),
    )
    assert resolve_vocab_size(model) == 151936


def test_unusable_values_are_skipped():
    assert resolve_vocab_size(None) is None
    assert resolve_vocab_size(SimpleNamespace()) is None
    assert (
        resolve_vocab_size(SimpleNamespace(config=SimpleNamespace(vocab_size=None)))
        is None
    )
    # zero, negative and bool placeholders are not vocabulary sizes
    config = SimpleNamespace(vocab_size=0, text_config={"vocab_size": True})
    assert resolve_vocab_size(SimpleNamespace(config=config)) is None
    config = SimpleNamespace(vocab_size=-1, text_config={"vocab_size": 0})
    assert resolve_vocab_size(SimpleNamespace(config=config)) is None
