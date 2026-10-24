"""Exercise the vendor's model-local load overrides without importing MLX."""

import ast
import math
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

_VENDOR = (
    Path(__file__).resolve().parents[1]
    / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp"
)


def _method(filename, class_name, method_name, namespace):
    tree = ast.parse((_VENDOR / filename).read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(_VENDOR / filename), "exec"), namespace)
    return namespace[method_name]


def test_explicit_native_ple_storage_overrides_resident_global_before_allocation(
    tmp_path,
):
    disk = Mock(return_value=object())
    resident = Mock(side_effect=AssertionError("resident PLE must not be allocated"))
    namespace = {
        "super": lambda: SimpleNamespace(__init__=lambda: None),
        "nn": SimpleNamespace(Module=SimpleNamespace(__init__=lambda self: None)),
        "mx": np,
        "math": math,
        "TextConfig": SimpleNamespace,
        "_build_layer_multipliers": lambda *args: np.array([1], dtype=np.int64),
        "_find_nth_prime_after": lambda start, count: 7,
        "_PLE_RUNTIME_MODE": "resident",
        "_PLE_RUNTIME_MODEL_PATH": None,
        "DiskBackedShardedEmbedding": disk,
        "ShardedEmbedding": resident,
    }
    initialize = _method("language.py", "Qwen4ExpNGramEmbedding", "__init__", namespace)
    config = SimpleNamespace(
        ngram_size=2,
        heads_per_ngram=1,
        vocab_size=16,
        seed=1,
        eos_token_id=1,
        ngram_vocab_size_base=7,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=2,
        ple_storage_mode="mmap",
        ple_model_path=str(tmp_path),
        ple_tensor_prefixes={"1": "language_model.layers.1.ple.ngram_embedding"},
    )
    embedding = SimpleNamespace()
    initialize(embedding, config, 4, 1, 0)
    disk.assert_called_once_with(
        str(tmp_path), "language_model.layers.1.ple.ngram_embedding", 8, 4, 2
    )
    resident.assert_not_called()
    assert namespace["_PLE_RUNTIME_MODE"] == "resident"
    assert namespace["_PLE_RUNTIME_MODEL_PATH"] is None


def test_explicit_native_mtp_off_does_not_construct_global_mtp_module():
    mtp = Mock(side_effect=AssertionError("packed MTP has no bindings"))
    initialize = _method(
        "qwen4_exp.py",
        "Model",
        "__init__",
        {
            "nn": SimpleNamespace(Module=SimpleNamespace(__init__=lambda self: None)),
            "ModelConfig": SimpleNamespace,
            "VisionModel": lambda _: "vision",
            "LanguageModel": lambda *args: "language",
            "get_mtp_runtime": lambda: SimpleNamespace(enabled=True),
            "Qwen4ExpMTPModule": mtp,
        },
    )
    model = SimpleNamespace()
    initialize(
        model,
        SimpleNamespace(
            vision_config=object(), text_config=object(), mtp_enabled=False
        ),
    )
    assert model._mtp_enabled is False
    mtp.assert_not_called()


def test_explicit_residual_marker_skips_heuristic_and_mtp_weights():
    heuristic = Mock(side_effect=AssertionError("declared norms must not be guessed"))
    sanitize = _method(
        "qwen4_exp.py",
        "Model",
        "sanitize",
        {
            "get_ple_runtime_mode": lambda: "resident",
            "get_mtp_runtime": lambda: SimpleNamespace(enabled=True),
            "dequantize_fp8_weights": lambda weights, **kw: weights,
            "sanitize_key": lambda key: key,
            "_MTP_PREFIXES": ("mtp.",),
            "_NGRAM_SHARD_RE": re.compile(r"(never-matches)"),
            "_NGRAM_STORAGE_RE": re.compile(r"never-matches"),
            "_normalize_ones_centered_rmsnorm_weights": heuristic,
        },
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                ple_storage_mode="mmap",
                ple_layer_ids=(),
                tie_word_embeddings=False,
                num_experts=0,
                num_hidden_layers=0,
            )
        ),
        _mtp_enabled=False,
        _omlx_explicit_rmsnorm_residual=True,
    )
    residual = np.array([0.25], dtype=np.float32)
    weights = {"norm.weight": residual, "mtp.unbound.weight": np.ones(1)}
    actual = sanitize(model, weights)
    assert set(actual) == {"norm.weight"}
    assert actual["norm.weight"] is residual
    heuristic.assert_not_called()
