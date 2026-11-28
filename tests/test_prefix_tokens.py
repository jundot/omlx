# SPDX-License-Identifier: Apache-2.0
"""Prefix token reuse returns exactly what a full encode returns."""

import copy
import random

import pytest

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from omlx.utils import prefix_tokens  # noqa: E402
from omlx.utils.prefix_tokens import (  # noqa: E402
    PrefixTokenCache,
    encode_prompt,
    prefix_token_cache_for,
)

_PIECES = [
    "def ", "return x\n", "    ", "\n\n", "\t", "日本語", "ü", "😀", "abc", "123", "...",
    "→", "\r\n", "class Cache:\n", " the ", "token", "<|im_end|>", "  \n",
]


def _text(seed, chars):
    rnd = random.Random(seed)
    out, n = [], 0
    while n < chars:
        piece = rnd.choice(_PIECES)
        out.append(piece)
        n += len(piece)
    return "".join(out)


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator([_text(i, 4000) for i in range(20)], trainer)
    return transformers.PreTrainedTokenizerFast(tokenizer_object=tok)


def _full(tokenizer, text):
    return tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).ids


def test_reused_prefix_matches_a_full_encode(tokenizer, monkeypatch):
    monkeypatch.setattr(prefix_tokens, "MIN_CHARS", 2_000)
    cache = prefix_token_cache_for(tokenizer)
    assert cache is not None
    doc = _text(99, 40_000)
    cache.encode(doc)
    rnd = random.Random(5)
    tails = ["", " ", "\n", "😀 tail", "ü", "<|im_end|>\n", "ab", "a" * 50]
    for _ in range(200):
        cut = rnd.randrange(3_000, len(doc))
        text = doc[:cut] + rnd.choice(tails) + doc[cut : cut + rnd.randrange(0, 200)]
        assert cache.encode(text) == _full(tokenizer, text)
    assert cache.reused > 150


def test_a_mismatched_margin_falls_back_to_a_full_encode(tokenizer, monkeypatch):
    monkeypatch.setattr(prefix_tokens, "MIN_CHARS", 2_000)
    cache = prefix_token_cache_for(tokenizer)
    doc = _text(7, 20_000)
    cache.encode(doc)
    text, ids, offsets = cache._items[0]
    # Corrupt the cached tokens around where the next re-encode is checked.
    cache._items[0] = (text, [i + 1 for i in ids], offsets)
    new = doc[:15_000] + " new question"
    assert cache.encode(new) == _full(tokenizer, new)
    assert cache.full == 2 and cache.reused == 0


def test_tokenizers_that_add_special_tokens_do_not_qualify(tokenizer):
    from tokenizers import processors

    with_bos = copy.deepcopy(tokenizer)
    with_bos.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="a $A", special_tokens=[("a", with_bos.convert_tokens_to_ids("a"))]
    )
    assert prefix_token_cache_for(with_bos) is None
    assert prefix_token_cache_for(object()) is None


def test_only_byte_level_bpe_qualifies():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator([_text(1, 2000)], trainers.WordLevelTrainer(special_tokens=["[UNK]"]))
    assert prefix_token_cache_for(transformers.PreTrainedTokenizerFast(tokenizer_object=tok)) is None


def test_short_prompts_use_the_tokenizer(tokenizer):
    cache = prefix_token_cache_for(tokenizer)
    assert encode_prompt(tokenizer, cache, "short prompt") == tokenizer.encode("short prompt")
    assert cache.full == 0 and cache.reused == 0


def test_a_copied_cache_starts_empty(tokenizer):
    cache = prefix_token_cache_for(tokenizer)
    cache.encode(_text(3, 40_000))
    copied = copy.deepcopy(cache)
    assert isinstance(copied, PrefixTokenCache) and copied._items == []
