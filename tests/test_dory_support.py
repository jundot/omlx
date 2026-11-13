# SPDX-License-Identifier: Apache-2.0
"""Small, offline checks of Dory registration, cache layout and tokenizer."""

import importlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from unittest.mock import patch

import mlx.core as mx
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from omlx.patches import dory


def test_registration_and_logical_cache_counts():
    dory.register_model()
    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes({"model_type": "dory"})
    assert model_cls.__module__ == "mlx_lm.models.dory"
    assert importlib.import_module("mlx_lm.models.dory").__file__.endswith("dory_model.py")
    for loops, expected in [(1, 36), (2, 56), (4, 96)]:
        args = args_cls.from_dict({
            "model_type": "dory", "hidden_size": 8, "intermediate_size": 16,
            "vocab_size": 16, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 4, "n_recurrent_loops": loops,
        })
        model = model_cls(args)
        caches = model.make_cache()
        assert len(caches) == expected
        assert len(model.layers) == 72
        if loops == 2:
            assert sum(type(c).__name__ == "RotatingKVCache" for c in caches) == 42
            assert sum(type(c).__name__ == "KVCache" for c in caches) == 14
    assert args_cls.from_dict({"model_type": "dory"}).n_recurrent_loops == 2


def test_concurrent_first_registration_and_import(monkeypatch):
    import mlx_lm.models as models_pkg
    from mlx_lm.utils import _get_classes

    name = "mlx_lm.models.dory"
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delattr(models_pkg, "dory", raising=False)
    original_spec = importlib.util.spec_from_file_location
    executing = threading.Event()
    release = threading.Event()
    registrar_started = threading.Event()
    importer_started = threading.Event()
    executions = []

    def delayed_spec(module_name, path):
        spec = original_spec(module_name, path)
        if module_name == name:
            loader = spec.loader

            class DelayedLoader:
                def create_module(self, spec):
                    return None

                def exec_module(self, module):
                    executions.append(module)
                    executing.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release registration")
                    loader.exec_module(module)

            spec.loader = DelayedLoader()
        return spec

    monkeypatch.setattr(importlib.util, "spec_from_file_location", delayed_spec)

    def second_registration():
        registrar_started.set()
        dory.register_model()
        return _get_classes({"model_type": "dory"})

    def ordinary_import():
        importer_started.set()
        return importlib.import_module(name)

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(dory.register_model)
        try:
            assert executing.wait(5)
            second = pool.submit(second_registration)
            third = pool.submit(ordinary_import)
            assert registrar_started.wait(5) and importer_started.wait(5)
            with pytest.raises(FutureTimeout):
                second.result(timeout=0.2)
            with pytest.raises(FutureTimeout):
                third.result(timeout=0.2)
        finally:
            release.set()
        first.result(timeout=5)
        model_cls, args_cls = second.result(timeout=5)
        imported = third.result(timeout=5)
    assert len(executions) == 1
    assert imported is executions[0] is sys.modules[name] is models_pkg.dory
    assert (model_cls, args_cls) == (imported.Model, imported.ModelArgs)


def test_failed_registration_removes_partial_module(monkeypatch):
    import mlx_lm.models as models_pkg

    name = "mlx_lm.models.dory"
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delattr(models_pkg, "dory", raising=False)
    original_spec = importlib.util.spec_from_file_location

    def failing_spec(module_name, path):
        spec = original_spec(module_name, path)
        if module_name == name:
            class FailingLoader:
                def create_module(self, spec):
                    return None

                def exec_module(self, module):
                    raise RuntimeError("registration failed")
            spec.loader = FailingLoader()
        return spec

    monkeypatch.setattr(importlib.util, "spec_from_file_location", failing_spec)
    with pytest.raises(RuntimeError, match="registration failed"):
        dory.register_model()
    assert name not in sys.modules
    assert not hasattr(models_pkg, "dory")


def test_pre_load_dispatch_registers_dory(tmp_path):
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    (tmp_path / "config.json").write_text('{"model_type":"dory"}')
    maybe_apply_pre_load_patches(str(tmp_path))
    from mlx_lm.utils import _get_classes

    assert _get_classes({"model_type": "dory"})[0].__module__ == "mlx_lm.models.dory"


def test_fused_checkpoint_weights():
    dory.register_model()
    from mlx_lm.models.dory import Model, ModelArgs

    model = Model(ModelArgs(hidden_size=8, intermediate_size=8, vocab_size=8,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                            num_hidden_layers=2, n_input_layers=0,
                            n_recurrent_layers=2, n_output_layers=0))
    weights = {f"layer.mixer.{name}_proj.weight": mx.full((1, 2), i)
               for i, name in enumerate(("q", "k", "v", "g"))}
    weights.update({f"layer.mixer.{name}_proj.weight": mx.full((1, 2), i)
                    for i, name in enumerate(("gate", "up"))})
    result = model.sanitize(weights)
    assert result["layer.mixer.qkvg_proj.weight"].tolist() == [[0, 0], [1, 1], [2, 2], [3, 3]]
    assert result["layer.mixer.gate_up_proj.weight"].tolist() == [[0, 0], [1, 1]]
    assert set(model.sanitize(result)) == set(result)


@pytest.fixture
def local_tokenizer(tmp_path):
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "[EOS]": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]")
    fast.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    fast.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "dory"}))
    return tmp_path


def test_known_validation_fallback_keeps_template_eos_and_ids(local_tokenizer):
    dory.register_model()
    from mlx_lm import utils

    err = ValueError("layer_types entries must be in allowed types; got 'mlp'")
    with patch.object(utils, "load_tokenizer", side_effect=err):
        wrapped = dory.load_tokenizer(local_tokenizer, {"trust_remote_code": False}, [2, 11])
    assert wrapped.encode("hello", add_special_tokens=False) == [1]
    assert wrapped.eos_token_ids == {2, 11}
    assert wrapped.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False) == "hello"


def test_unrelated_tokenizer_failure_is_not_swallowed(local_tokenizer):
    from mlx_lm import utils

    err = ValueError("unrelated validation failure")
    with patch.object(utils, "load_tokenizer", side_effect=err):
        with pytest.raises(ValueError, match="unrelated") as caught:
            dory.load_tokenizer(local_tokenizer)
    assert caught.value is err


def test_non_dory_known_error_is_not_swallowed(local_tokenizer):
    from mlx_lm import utils

    (local_tokenizer / "config.json").write_text('{"model_type":"llama"}')
    err = ValueError("layer_types entries must be in allowed types; got 'mlp'")
    with patch.object(utils, "load_tokenizer", side_effect=err):
        with pytest.raises(ValueError) as caught:
            dory.load_tokenizer(local_tokenizer)
    assert caught.value is err


def test_dory_load_path_uses_local_tokenizer_without_changing_mlx_lm(local_tokenizer, monkeypatch):
    from mlx_lm import utils
    from omlx.utils import model_loading

    dory.register_model()
    fake_model = object()
    monkeypatch.setattr(utils, "_download", lambda path, **kwargs: local_tokenizer)
    monkeypatch.setattr(utils, "load_model", lambda *a, **kw: (fake_model, {"eos_token_id": [2, 11]}))
    original = utils.load_tokenizer
    monkeypatch.setattr(utils, "load_tokenizer", lambda *a, **kw: (_ for _ in ()).throw(
        ValueError("layer_types entries must be in allowed types; got 'mlp'")))
    installed = utils.load_tokenizer
    model, tokenizer = model_loading.lm_load_compat(str(local_tokenizer))
    assert model is fake_model and tokenizer.eos_token_ids == {2, 11}
    assert utils.load_tokenizer is installed and original is not installed
