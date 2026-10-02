# SPDX-License-Identifier: Apache-2.0
"""K2 BOS handling and tokenizer copy isolation."""

import copy
import json
from types import SimpleNamespace

import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from omlx.patches.k2_horizon.checkpoint import _patch_tokenizer


@pytest.mark.parametrize("model_type", ["k2_horizon", "llama", "gemma4", "qwen3"])
def test_rendered_bos_is_not_added_twice(tmp_path, model_type):
    bos = "<|ifm|begin_of_text|>"
    backend = Tokenizer(
        models.WordLevel({bos: 0, "hello": 1, "[UNK]": 2}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single=f"{bos} $A", special_tokens=[(bos, 0)]
    )
    hf = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token=bos, unk_token="[UNK]"
    )
    wrapped = TokenizerWrapper(hf, eos_token_ids={2})
    hf.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{bos} $A", special_tokens=[(bos, 0)]
    )
    object.__setattr__(wrapped, "add_eos_token", lambda _: None)
    utils = SimpleNamespace(load_tokenizer=lambda *_args, **_kwargs: wrapped)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    _patch_tokenizer(utils)
    tokenizer = utils.load_tokenizer(tmp_path)
    expected = [0, 1] if model_type == "k2_horizon" else [0, 0, 1]
    assert tokenizer.encode(bos + " hello") == expected
    assert tokenizer.encode("hello") == [0, 1]
    assert tokenizer.encode(bos + " hello", add_special_tokens=False) == [0, 1]
    assert tokenizer.encode(bos + " hello", add_special_tokens=True) == [0, 0, 1]
    clone = copy.deepcopy(tokenizer)
    assert clone.encode.__self__ is not tokenizer.encode.__self__
    assert clone.encode(bos + " hello") == expected
