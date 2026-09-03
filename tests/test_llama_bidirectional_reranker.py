# SPDX-License-Identifier: Apache-2.0
"""Tests for the native LlamaBidirectional sequence-classification reranker."""

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from omlx.model_discovery import (
    LLAMA_BIDIRECTIONAL_RERANKER_ARCHITECTURE,
    SUPPORTED_RERANKER_ARCHITECTURES,
)
from omlx.models.llama_bidirectional_reranker import (
    Model,
    ModelArgs,
    bidirectional_attention_mask,
)
from omlx.models.reranker import MLXRerankerModel

ARCH = LLAMA_BIDIRECTIONAL_RERANKER_ARCHITECTURE


def _config(**overrides) -> ModelArgs:
    values = {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "rope_theta": 500000.0,
        "id2label": {"0": "LABEL_0"},
    }
    values.update(overrides)
    return ModelArgs(**values)


def _tiny_model(**overrides) -> Model:
    model = Model(_config(**overrides))
    mx.eval(model.parameters())
    model.train(False)
    return model


# --- configuration -----------------------------------------------------------


def test_num_labels_is_derived_from_id2label() -> None:
    assert _config().num_labels == 1
    assert _config(id2label=None).num_labels == 1
    assert _config(id2label={"0": "a", "1": "b"}).num_labels == 2


def test_explicit_num_labels_wins_over_id2label() -> None:
    assert _config(num_labels=3, id2label={"0": "a"}).num_labels == 3


@pytest.mark.parametrize(
    "override",
    [
        {"architectures": ["LlamaBidirectionalModel"]},
        {"pooling": "cls"},
        {"temperature": 0.0},
        {"temperature": -1.0},
    ],
)
def test_config_fails_closed_on_unsupported_variants(override) -> None:
    with pytest.raises(ValueError):
        _config(**override)


# --- discovery and routing ---------------------------------------------------


def test_architecture_is_registered_as_a_supported_reranker() -> None:
    assert ARCH in SUPPORTED_RERANKER_ARCHITECTURES


def test_validate_architecture_accepts_without_directory_hint(tmp_path) -> None:
    # Deliberately a directory name with no "rerank" substring: this
    # architecture is self-disambiguating, unlike the CausalLM/VLM rerankers.
    model_dir = tmp_path / "plain-name"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architectures": [ARCH]}))

    MLXRerankerModel(str(model_dir))._validate_architecture()


def test_embedding_sibling_is_not_a_supported_reranker() -> None:
    assert "LlamaBidirectionalModel" not in SUPPORTED_RERANKER_ARCHITECTURES


def test_tokenizer_loader_rejects_an_unexpected_tokenizer_class(tmp_path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "LlamaTokenizer"})
    )

    with pytest.raises(ValueError, match="PreTrainedTokenizerFast"):
        MLXRerankerModel._load_llama_bidirectional_tokenizer(tmp_path)


# --- attention mask ----------------------------------------------------------


def test_mask_is_bidirectional_and_excludes_padding_keys() -> None:
    mask = bidirectional_attention_mask(mx.array([[1, 1, 1, 0]]), mx.float32)
    mx.eval(mask)

    assert mask.shape == (1, 1, 4, 4)
    # Both directions are visible between real tokens.
    assert mask[0, 0, 0, 2].item() == 0.0
    assert mask[0, 0, 2, 0].item() == 0.0
    # The padding column is masked for every query row.
    assert mask[0, 0, 0, 3].item() == float("-inf")
    assert mask[0, 0, 3, 3].item() == float("-inf")


def test_padding_rows_keep_a_finite_key_so_softmax_cannot_produce_nan() -> None:
    mask = bidirectional_attention_mask(mx.array([[1, 0]]), mx.float32)
    mx.eval(mask)

    # Row 1 is a padding query; it must still see the real token at column 0.
    assert mask[0, 0, 1, 0].item() == 0.0


def test_mask_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError):
        bidirectional_attention_mask(mx.ones((2, 3, 4)), mx.float32)


# --- weights -----------------------------------------------------------------


def test_score_head_is_root_level_and_bias_free() -> None:
    model = _tiny_model()
    flat = dict(tree_flatten(model.parameters()))

    assert "score.weight" in flat
    assert flat["score.weight"].shape == (1, 8)
    assert "score.bias" not in flat


def test_sanitize_preserves_the_checkpoint_namespace() -> None:
    model = _tiny_model()
    weights = {
        "model.embed_tokens.weight": mx.zeros((32, 8)),
        "score.weight": mx.zeros((1, 8)),
        "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((4,)),
        "lm_head.weight": mx.zeros((32, 8)),
    }

    sanitized = model.sanitize(weights)

    # score.weight stays at the root; a blanket "model." prefix would hide it.
    assert set(sanitized) == {"model.embed_tokens.weight", "score.weight"}


def test_sanitized_checkpoint_keys_load_into_the_model() -> None:
    model = _tiny_model()
    checkpoint = {
        key: mx.zeros(value.shape) for key, value in tree_flatten(model.parameters())
    }
    checkpoint["lm_head.weight"] = mx.zeros((32, 8))

    model.load_weights(list(model.sanitize(checkpoint).items()))
    mx.eval(model.parameters())


# --- forward -----------------------------------------------------------------


def test_forward_returns_a_raw_unsquashed_logit() -> None:
    model = _tiny_model()
    input_ids = mx.array([[1, 2, 3]])
    attention_mask = mx.ones((1, 3), dtype=mx.int32)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    mx.eval(output.pooler_output, output.last_hidden_state)

    assert output.pooler_output.shape == (1, 1)
    assert output.text_embeds is None

    # Recompute the head by hand. Any sigmoid, softmax or normalization in the
    # model path would break this equality.
    hidden = output.last_hidden_state
    pooled = mx.sum(hidden, axis=1) / 3.0
    expected = pooled @ model.score.weight.T
    assert mx.allclose(output.pooler_output, expected, atol=1e-5)


def test_temperature_scales_the_logit() -> None:
    warm = _tiny_model()
    cold = _tiny_model(temperature=2.0)
    cold.update(warm.parameters())
    mx.eval(cold.parameters())

    input_ids = mx.array([[1, 2, 3]])
    attention_mask = mx.ones((1, 3), dtype=mx.int32)
    warm_logit = warm(input_ids=input_ids, attention_mask=attention_mask).pooler_output
    cold_logit = cold(input_ids=input_ids, attention_mask=attention_mask).pooler_output
    mx.eval(warm_logit, cold_logit)

    assert mx.allclose(cold_logit, warm_logit / 2.0, atol=1e-5)


def test_padding_does_not_change_a_sequence_score() -> None:
    """Left padding must be inert: RoPE logits depend only on relative offsets
    and padding keys are masked, so a pair scores the same alone and in a
    padded batch. This is the sharpest check on mask and pooling correctness."""
    model = _tiny_model()

    alone = model(
        input_ids=mx.array([[5, 6, 7]]),
        attention_mask=mx.ones((1, 3), dtype=mx.int32),
    ).pooler_output
    # Same sequence, left-padded to length 5 inside a batch.
    batched = model(
        input_ids=mx.array([[0, 0, 5, 6, 7], [1, 2, 3, 4, 9]]),
        attention_mask=mx.array([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
    ).pooler_output
    mx.eval(alone, batched)

    assert mx.allclose(alone[0], batched[0], atol=1e-4)


# --- prompt and scoring path -------------------------------------------------


def test_prompt_template_matches_the_reference_byte_for_byte() -> None:
    rendered = MLXRerankerModel._format_llama_bidirectional_prompt("q text", "d text")

    assert rendered == "question:q text \n \n passage:d text"


class _RecordingProcessor:
    """Minimal tokenizer stand-in that records how it was called."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        batch = len(texts)
        return {
            "input_ids": [[1, 2, 3] for _ in range(batch)],
            "attention_mask": [[1, 1, 1] for _ in range(batch)],
        }


def _wired_reranker(model, processor) -> MLXRerankerModel:
    reranker = MLXRerankerModel("unused")
    reranker.model = model
    reranker.processor = processor
    reranker._loaded = True
    reranker._is_llama_bidirectional = True
    reranker._num_labels = 1
    return reranker


def test_rerank_defaults_to_the_8192_token_ceiling() -> None:
    processor = _RecordingProcessor()
    reranker = _wired_reranker(_tiny_model(), processor)

    reranker.rerank("query", ["doc one", "doc two"])

    assert MLXRerankerModel._DEFAULT_MAX_LENGTH_LLAMA_BIDIRECTIONAL == 8192
    assert processor.calls[0]["max_length"] == 8192


def test_rerank_encodes_single_templated_sequences_not_pairs() -> None:
    processor = _RecordingProcessor()
    reranker = _wired_reranker(_tiny_model(), processor)

    reranker.rerank("what is ml", ["ml is", "weather is"])

    call = processor.calls[0]
    # One positional list of fully rendered strings, never a (queries, docs) pair.
    assert call["texts"] == [
        "question:what is ml \n \n passage:ml is",
        "question:what is ml \n \n passage:weather is",
    ]
    assert call["padding"] is True
    assert call["truncation"] is True


def test_rerank_sorts_by_raw_score_and_counts_real_tokens() -> None:
    model = _tiny_model()
    # Force a deterministic, clearly signed ordering through the head.
    model.score.weight = mx.array([[1.0] + [0.0] * 7])
    mx.eval(model.parameters())
    reranker = _wired_reranker(model, _RecordingProcessor())

    output = reranker.rerank("query", ["doc one", "doc two"])

    assert len(output.scores) == 2
    assert output.indices == sorted(
        range(2), key=lambda i: output.scores[i], reverse=True
    )
    # Six real tokens across the two 3-token rows; no padding was emitted.
    assert output.total_tokens == 6


def test_rerank_rejects_a_multi_label_head() -> None:
    reranker = _wired_reranker(_tiny_model(num_labels=2), _RecordingProcessor())

    with pytest.raises(ValueError, match="single-label head"):
        reranker.rerank("query", ["doc"])
