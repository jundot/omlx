# SPDX-License-Identifier: Apache-2.0
"""Tests for pylate ColBERT (per-token projected) embedding support.

Covers the layout predicate, the projection recipe and its per-token L2, the
declared activation function, pylate's residual rules, the width read from the
last Dense module, the RoPE config dialect, the safetensors weight layout and
the loader dispatch.
"""

import json
import math
from pathlib import Path

import mlx.core as mx
import pytest

from omlx.models.colbert_projection import (
    ColbertLayoutError,
    ColbertProjection,
    flatten_rope_parameters,
    is_pylate_colbert_dir,
    l2_normalize_tokens,
    resolve_colbert_layout,
)
from omlx.models.embedding import MLXEmbeddingModel

# The tiny fixture's projection, computed by hand:
#   h = [1, 2]
#   L0 linear   [[1,0],[0,1],[1,1]] -> [1, 2, 3]
#   L0 residual [[0,1],[1,0],[0,0]] -> [2, 1, 0]        => [3, 3, 3]
#   L1 linear   [[1,0,0],[0,1,0]]   -> [3, 3]
#   L1 residual [[0,1,0],[1,0,0]]   -> [3, 3]           => [6, 6]
#   L2 linear   [[1,0],[1,1]]       -> [6, 12]
#   [6, 12] / sqrt(180) = [0.4472135955, 0.8944271910]
TINY_HIDDEN = [1.0, 2.0]
TINY_PROJECTED = [0.4472135955, 0.8944271910]


def _assert_close(actual, expected, atol: float = 1e-6) -> None:
    """Compare nested arrays, reporting both sides when they differ."""
    assert mx.allclose(
        mx.array(actual), mx.array(expected), rtol=1e-5, atol=atol
    ).item(), f"expected {expected}, got {actual}"


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload))


def _write_dense_module(
    root: Path,
    name: str,
    linear,
    residual=None,
    *,
    use_residual=None,
    bias: bool = False,
    activation: str = "torch.nn.modules.linear.Identity",
) -> None:
    """Write one pylate Dense module. ``linear`` is the [out, in] torch layout."""
    directory = root / name
    directory.mkdir(parents=True)
    arrays = {"linear.weight": mx.array(linear, dtype=mx.float32)}
    if residual is not None:
        arrays["residual.weight"] = mx.array(residual, dtype=mx.float32)
    mx.save_safetensors(str(directory / "model.safetensors"), arrays)
    _write_json(
        directory / "config.json",
        {
            "in_features": len(linear[0]),
            "out_features": len(linear),
            "bias": bias,
            "activation_function": activation,
            "use_residual": (
                residual is not None if use_residual is None else use_residual
            ),
        },
    )


def _write_dense_config(
    root: Path,
    name: str,
    *,
    in_features: int,
    out_features: int,
    use_residual: bool,
    bias: bool = False,
    activation: str = "torch.nn.modules.linear.Identity",
) -> None:
    """Write a Dense module's config without any weight file."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(
        directory / "config.json",
        {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            "activation_function": activation,
            "use_residual": use_residual,
        },
    )


def _write_pipeline(root: Path, dense_names, *, model_type: str = "ColBERT") -> None:
    _write_json(root / "config_sentence_transformers.json", {"model_type": model_type})
    modules = [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer",
        }
    ]
    modules += [
        {
            "idx": index,
            "name": str(index),
            "path": name,
            "type": "pylate.models.Dense.Dense",
        }
        for index, name in enumerate(dense_names, start=1)
    ]
    _write_json(root / "modules.json", modules)


def _write_encoder_safetensors(root: Path) -> None:
    mx.save_safetensors(
        str(root / "model.safetensors"),
        {"final_norm.weight": mx.ones((2,), dtype=mx.float32)},
    )


def _tiny_colbert_dir(tmp_path: Path) -> Path:
    """A 2 -> 3 -> 2 -> 2 chain whose projection is computable by hand."""
    root = tmp_path / "tiny-colbert"
    root.mkdir()
    _write_pipeline(root, ["1_Dense", "2_Dense", "3_Dense"])
    _write_encoder_safetensors(root)
    _write_dense_module(
        root, "1_Dense", [[1, 0], [0, 1], [1, 1]], [[0, 1], [1, 0], [0, 0]]
    )
    _write_dense_module(root, "2_Dense", [[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [1, 0, 0]])
    _write_dense_module(root, "3_Dense", [[1, 0], [1, 1]])
    return root


def _one_dense_dir(
    tmp_path: Path,
    name: str,
    linear,
    residual=None,
    *,
    use_residual=None,
    activation: str = "torch.nn.modules.linear.Identity",
) -> Path:
    root = tmp_path / name
    root.mkdir()
    _write_pipeline(root, ["1_Dense"])
    _write_encoder_safetensors(root)
    _write_dense_module(
        root,
        "1_Dense",
        linear,
        residual,
        use_residual=use_residual,
        activation=activation,
    )
    return root


def _plain_embedding_dir(tmp_path: Path) -> Path:
    """An ordinary sentence-transformers embedding export (no ColBERT)."""
    root = tmp_path / "plain-embedder"
    root.mkdir()
    _write_json(
        root / "config_sentence_transformers.json",
        {"model_type": "SentenceTransformer", "similarity_fn_name": "cosine"},
    )
    _write_json(
        root / "modules.json",
        [
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.models.Transformer",
            },
            {
                "idx": 1,
                "name": "1",
                "path": "1_Pooling",
                "type": "sentence_transformers.models.Pooling",
            },
        ],
    )
    return root


class TestColbertLayoutDetection:
    """`is_pylate_colbert_dir` separates a ColBERT export from a plain embedder."""

    def test_recognizes_a_pylate_colbert_layout(self, tmp_path):
        root = _tiny_colbert_dir(tmp_path)

        assert is_pylate_colbert_dir(root) is True

    def test_recognizes_colbert_by_maxsim_similarity_fn(self, tmp_path):
        root = tmp_path / "maxsim-colbert"
        root.mkdir()
        _write_pipeline(root, ["1_Dense"])
        _write_json(
            root / "config_sentence_transformers.json",
            {"similarity_fn_name": "MaxSim"},
        )
        assert is_pylate_colbert_dir(root) is True

    def test_recognizes_colbert_by_pylate_module_type(self, tmp_path):
        root = tmp_path / "pylate-colbert"
        root.mkdir()
        _write_json(
            root / "modules.json",
            [
                {
                    "idx": 1,
                    "name": "1_Dense",
                    "path": "1_Dense",
                    "type": "pylate.models.Dense.Dense",
                }
            ],
        )
        assert is_pylate_colbert_dir(root) is True

    def test_recognizes_colbert_by_architecture(self, tmp_path):
        root = tmp_path / "hf-colbert"
        root.mkdir()
        _write_json(
            root / "modules.json",
            [
                {
                    "idx": 1,
                    "name": "1_Dense",
                    "path": "1_Dense",
                    "type": "sentence_transformers.models.Dense",
                }
            ],
        )
        _write_json(
            root / "config.json",
            {"architectures": ["HF_ColBERT"]},
        )
        assert is_pylate_colbert_dir(root) is True

    def test_rejects_an_ordinary_embedding_export(self, tmp_path):
        root = _plain_embedding_dir(tmp_path)

        assert is_pylate_colbert_dir(root) is False

    def test_rejects_the_colbert_declaration_without_dense_modules(self, tmp_path):
        root = tmp_path / "colbert-without-projection"
        root.mkdir()
        _write_pipeline(root, [])

        assert is_pylate_colbert_dir(root) is False

    def test_rejects_a_directory_with_no_metadata(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()

        assert is_pylate_colbert_dir(root) is False


class TestColbertLayoutResolution:
    """`resolve_colbert_layout` reads the chain, its width and its weight source."""

    def test_reads_the_output_width_from_the_last_dense_module(self, tmp_path):
        root = tmp_path / "wide-colbert"
        root.mkdir()
        _write_pipeline(root, ["1_Dense", "2_Dense"])
        _write_encoder_safetensors(root)
        _write_dense_module(root, "1_Dense", [[1, 0], [0, 1]])
        _write_dense_module(root, "2_Dense", [[1, 0] for _ in range(7)])

        layout = resolve_colbert_layout(root)

        assert layout.output_dim == 7
        assert [spec.out_features for spec in layout.dense] == [2, 7]
        assert layout.encoder_weights == root / "model.safetensors"

    def test_rejects_an_inconsistent_projection_chain(self, tmp_path):
        root = tmp_path / "broken-colbert"
        root.mkdir()
        _write_pipeline(root, ["1_Dense", "2_Dense"])
        _write_encoder_safetensors(root)
        _write_dense_module(root, "1_Dense", [[1, 0], [0, 1], [1, 1]])
        _write_dense_module(root, "2_Dense", [[1, 0] for _ in range(5)])

        with pytest.raises(ColbertLayoutError, match="inconsistent"):
            resolve_colbert_layout(root)

    def test_rejects_a_directory_with_no_weight_artifact(self, tmp_path):
        root = tmp_path / "weightless-colbert"
        root.mkdir()
        _write_pipeline(root, ["1_Dense"])
        _write_dense_config(
            root, "1_Dense", in_features=2, out_features=3, use_residual=False
        )

        with pytest.raises(ColbertLayoutError, match="model.safetensors"):
            resolve_colbert_layout(root)

    def test_rejects_a_missing_encoder_safetensors(self, tmp_path):
        root = _tiny_colbert_dir(tmp_path)
        (root / "model.safetensors").unlink()

        with pytest.raises(ColbertLayoutError, match="model.safetensors"):
            resolve_colbert_layout(root)


class TestColbertProjectionMath:
    """The projection recipe, layer by layer, and its per-token L2."""

    def test_applies_linear_plus_learned_residual_per_layer(self, tmp_path):
        projection = ColbertProjection.from_layout(
            resolve_colbert_layout(_tiny_colbert_dir(tmp_path))
        )

        result = projection(mx.array([TINY_HIDDEN], dtype=mx.float32))

        _assert_close(result.tolist(), [TINY_PROJECTED])

    def test_residual_is_a_learned_matmul_not_an_identity_skip(self, tmp_path):
        # A shape-changing layer's residual is a second learned MatMul; an
        # identity skip is not even shape-compatible here (2 -> 3).
        root = _one_dense_dir(
            tmp_path,
            "swap",
            [[1, 0], [0, 1], [1, 1]],
            [[0, 1], [1, 0], [0, 0]],
        )
        projection = ColbertProjection.from_layout(resolve_colbert_layout(root))

        result = projection(mx.array([[1.0, 2.0]], dtype=mx.float32))

        # linear -> [1, 2, 3]; residual -> [2, 1, 0]; sum -> [3, 3, 3].
        _assert_close(result.tolist(), [[3 / 27**0.5] * 3])

    def test_the_final_layer_has_no_residual(self, tmp_path):
        # Doubling the linear map is observable only if the input were added too.
        root = _one_dense_dir(tmp_path, "final", [[1, 0], [0, 2]])
        projection = ColbertProjection.from_layout(resolve_colbert_layout(root))

        result = projection(mx.array([[1.0, 2.0]], dtype=mx.float32))

        _assert_close(result.tolist(), [[0.2425356250, 0.9701425001]])

    def test_every_projected_row_is_l2_normalized(self, tmp_path):
        projection = ColbertProjection.from_layout(
            resolve_colbert_layout(_tiny_colbert_dir(tmp_path))
        )

        result = projection(mx.array([[1.0, 2.0], [3.0, -4.0]], dtype=mx.float32))
        norms = mx.linalg.norm(result, axis=-1)

        assert norms.tolist() == pytest.approx([1.0, 1.0], abs=1e-6)

    def test_normalization_is_per_token_not_per_batch(self):
        vectors = mx.array([[3.0, 4.0], [30.0, 40.0]], dtype=mx.float32)

        normalized = l2_normalize_tokens(vectors)

        _assert_close(normalized.tolist(), [[0.6, 0.8], [0.6, 0.8]])

    def test_a_degenerate_row_stays_zero_instead_of_nan(self):
        vectors = mx.array([[0.0, 0.0], [3.0, 4.0]], dtype=mx.float32)

        normalized = l2_normalize_tokens(vectors)

        _assert_close(normalized.tolist(), [[0.0, 0.0], [0.6, 0.8]])

    def test_rejects_a_transposed_weight_matrix(self, tmp_path):
        # config declares 2 -> 3 but the file stores [2, 3], the [in, out] layout.
        root = tmp_path / "transposed"
        root.mkdir()
        _write_pipeline(root, ["1_Dense"])
        _write_encoder_safetensors(root)
        (root / "1_Dense").mkdir()
        _write_json(
            root / "1_Dense" / "config.json",
            {
                "in_features": 2,
                "out_features": 3,
                "bias": False,
                "use_residual": False,
            },
        )
        mx.save_safetensors(
            str(root / "1_Dense" / "model.safetensors"),
            {"linear.weight": mx.zeros((2, 3), dtype=mx.float32)},
        )
        layout = resolve_colbert_layout(root)

        with pytest.raises(ColbertLayoutError, match="torch nn.Linear layout"):
            ColbertProjection.from_layout(layout)


class TestRopeParameterDialect:
    """The transformers>=5 nested RoPE dialect must reach mlx-embeddings' fields."""

    def test_maps_nested_thetas_onto_the_flat_fields(self):
        config = {
            "hidden_size": 768,
            "rope_parameters": {
                "full_attention": {"rope_theta": 160000, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 160000, "rope_type": "default"},
            },
        }

        resolved = flatten_rope_parameters(config)

        assert resolved["global_rope_theta"] == 160000.0
        assert resolved["local_rope_theta"] == 160000.0

    def test_keeps_an_explicitly_declared_flat_theta(self):
        config = {
            "global_rope_theta": 500000.0,
            "rope_parameters": {"full_attention": {"rope_theta": 160000}},
        }

        resolved = flatten_rope_parameters(config)

        assert resolved["global_rope_theta"] == 500000.0

    def test_leaves_a_config_without_rope_parameters_alone(self):
        config = {"local_rope_theta": 10000.0}

        resolved = flatten_rope_parameters(config)

        assert resolved == {"local_rope_theta": 10000.0}


class _ConstantHiddenEncoder:
    """A raw encoder stand-in returning the same hidden row for every token."""

    def __init__(self, hidden_row):
        self.hidden_row = hidden_row
        self.batch_shapes = []

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch, width = input_ids.shape
        self.batch_shapes.append((batch, width))
        row = mx.array(self.hidden_row, dtype=mx.float32)
        return {
            "last_hidden_state": mx.broadcast_to(
                row, (batch, width, len(self.hidden_row))
            )
        }


def _colbert_backed_model(tmp_path: Path) -> MLXEmbeddingModel:
    """An `MLXEmbeddingModel` wired to the tiny projection, without loading weights."""
    path = _tiny_colbert_dir(tmp_path)
    layout = resolve_colbert_layout(path)
    model = MLXEmbeddingModel(str(path))
    model._loaded = True
    model.model = _ConstantHiddenEncoder(TINY_HIDDEN)
    model._colbert_projection = ColbertProjection.from_layout(layout)
    model._colbert_layout = layout
    model._hidden_size = layout.output_dim
    return model


class TestColbertEmbedTokenIds:
    """`embed_token_ids` applies the projection for a ColBERT model."""

    def test_projects_and_normalizes_every_token(self, tmp_path):
        model = _colbert_backed_model(tmp_path)

        output = model.embed_token_ids([[7, 8, 9], [3]])

        assert [len(row) for row in output.token_embeddings] == [3, 1]
        _assert_close(
            output.token_embeddings[0],
            [TINY_PROJECTED, TINY_PROJECTED, TINY_PROJECTED],
        )
        _assert_close(output.token_embeddings[1], [TINY_PROJECTED])
        assert output.dimensions == 2
        assert output.total_tokens == 4

    def test_reports_the_projected_width_not_the_encoder_width(self, tmp_path):
        model = _colbert_backed_model(tmp_path)

        output = model.embed_token_ids([[1]])

        assert output.dimensions == 2
        assert model.hidden_size == 2

    def test_the_pooled_path_is_refused(self, tmp_path):
        model = _colbert_backed_model(tmp_path)

        with pytest.raises(ValueError, match="late-interaction"):
            model.embed(["a document"])

    def test_get_model_info_reports_the_projection(self, tmp_path):
        model = _colbert_backed_model(tmp_path)

        info = model.get_model_info()

        assert info["late_interaction"] is True
        assert info["projection_dim"] == 2
        assert info["projection_layers"] == 3
        assert info["encoder_hidden_size"] == 2


class TestActivationFunctions:
    """The module's declared ``activation_function`` is honored exactly."""

    def test_relu_is_applied_to_the_linear_output(self, tmp_path):
        root = _one_dense_dir(
            tmp_path,
            "relu",
            [[1, 0], [0, 1]],
            activation="torch.nn.modules.activation.ReLU",
        )
        projection = ColbertProjection.from_layout(resolve_colbert_layout(root))

        result = projection(mx.array([[-1.0, 2.0]], dtype=mx.float32))

        _assert_close(result.tolist(), [[0.0, 1.0]])

    def test_gelu_is_the_exact_erf_form(self, tmp_path):
        root = _one_dense_dir(
            tmp_path,
            "gelu",
            [[1, 0], [0, 1]],
            activation="torch.nn.modules.activation.GELU",
        )
        projection = ColbertProjection.from_layout(resolve_colbert_layout(root))

        result = projection(mx.array([[1.0, -1.0]], dtype=mx.float32))

        vector = [
            0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))
            for value in (1.0, -1.0)
        ]
        norm = math.sqrt(sum(value * value for value in vector))
        _assert_close(result.tolist(), [[value / norm for value in vector]])

    def test_tanh_is_applied(self, tmp_path):
        root = _one_dense_dir(
            tmp_path,
            "tanh",
            [[1, 0], [0, 1]],
            activation="torch.nn.modules.activation.Tanh",
        )
        projection = ColbertProjection.from_layout(resolve_colbert_layout(root))

        result = projection(mx.array([[0.5, -2.0]], dtype=mx.float32))

        vector = [math.tanh(0.5), math.tanh(-2.0)]
        norm = math.sqrt(sum(value * value for value in vector))
        _assert_close(result.tolist(), [[value / norm for value in vector]])

    def test_activation_is_applied_before_the_residual(self, tmp_path):
        # pylate computes activation(2x) then adds x; applying the activation
        # after the residual would yield relu(3x) instead.
        root = _one_dense_dir(
            tmp_path,
            "order",
            [[2, 0], [0, 2]],
            use_residual=True,
            activation="torch.nn.modules.activation.ReLU",
        )
        projection = ColbertProjection.from_layout(resolve_colbert_layout(root))

        result = projection(mx.array([[-1.0, 2.0]], dtype=mx.float32))

        vector = [-1.0, 6.0]  # relu(2 * x) + x
        norm = math.sqrt(sum(value * value for value in vector))
        _assert_close(result.tolist(), [[value / norm for value in vector]])

    def test_a_missing_activation_key_means_identity(self, tmp_path):
        root = _one_dense_dir(tmp_path, "no-key", [[1, 0], [0, 1]])
        _write_json(
            root / "1_Dense" / "config.json",
            {
                "in_features": 2,
                "out_features": 2,
                "bias": False,
                "use_residual": False,
            },
        )
        projection = ColbertProjection.from_layout(resolve_colbert_layout(root))

        result = projection(mx.array([[3.0, 4.0]], dtype=mx.float32))

        _assert_close(result.tolist(), [[0.6, 0.8]])

    def test_an_unsupported_activation_is_rejected(self, tmp_path):
        root = _one_dense_dir(
            tmp_path,
            "sigmoid",
            [[1, 0], [0, 1]],
            activation="torch.nn.modules.activation.Sigmoid",
        )

        with pytest.raises(ColbertLayoutError, match="Sigmoid"):
            resolve_colbert_layout(root)

    def test_a_non_string_activation_is_rejected(self, tmp_path):
        root = _one_dense_dir(tmp_path, "numeric", [[1, 0], [0, 1]])
        _write_json(
            root / "1_Dense" / "config.json",
            {
                "in_features": 2,
                "out_features": 2,
                "bias": False,
                "use_residual": False,
                "activation_function": 5,
            },
        )

        with pytest.raises(ColbertLayoutError, match="non-string"):
            resolve_colbert_layout(root)


class TestResidualSemantics:
    """pylate's residual is a learned MatMul only when the width changes."""

    def test_same_width_residual_is_an_identity_skip(self, tmp_path):
        root = _one_dense_dir(
            tmp_path, "identity-residual", [[1, 0], [0, 1]], use_residual=True
        )
        projection = ColbertProjection.from_layout(resolve_colbert_layout(root))

        result = projection(mx.array([[1.0, 2.0]], dtype=mx.float32))

        _assert_close(result.tolist(), [[1 / 5**0.5, 2 / 5**0.5]])

    def test_same_width_residual_with_weights_is_rejected(self, tmp_path):
        root = _one_dense_dir(
            tmp_path,
            "identity-residual-weights",
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]],
        )

        with pytest.raises(ColbertLayoutError, match="identity skip"):
            ColbertProjection.from_layout(resolve_colbert_layout(root))

    def test_a_shape_changing_residual_still_needs_its_matrix(self, tmp_path):
        root = _one_dense_dir(
            tmp_path,
            "missing-residual",
            [[1, 0], [0, 1], [1, 1]],
            use_residual=True,
        )

        with pytest.raises(ColbertLayoutError, match="residual.weight"):
            ColbertProjection.from_layout(resolve_colbert_layout(root))


class TestColbertLoadDispatch:
    """``MLXEmbeddingModel.load`` routes a ColBERT directory to its own loader."""

    @staticmethod
    def _colbert_dir(tmp_path: Path) -> Path:
        root = _tiny_colbert_dir(tmp_path)
        _write_json(root / "config.json", {"model_type": "modernbert"})
        return root

    @staticmethod
    def _patch_mlx_embeddings(monkeypatch) -> None:
        class _StubModelArgs:
            @classmethod
            def from_dict(cls, config):
                return dict(config)

        class _StubEncoder:
            def __init__(self):
                self.loaded_weights = None
                self.training = True

            def parameters(self):
                return {"final_norm.weight": mx.ones((2,), dtype=mx.float32)}

            def load_weights(self, weights, strict=True):
                self.loaded_weights = dict(weights)

            def train(self, mode):
                self.training = mode

        class _StubWrapper:
            def __init__(self, args):
                self.model = _StubEncoder()

        class _StubModule:
            ModelArgs = _StubModelArgs
            Model = _StubWrapper

        import importlib

        monkeypatch.setattr(importlib, "import_module", lambda name: _StubModule)

    def test_dispatch_tokenizer_fallback_and_reload(self, tmp_path, monkeypatch):
        root = self._colbert_dir(tmp_path)
        self._patch_mlx_embeddings(monkeypatch)

        import transformers

        tokenizer_calls: list[dict] = []
        tokenizer = object()

        def fake_from_pretrained(path, **kwargs):
            tokenizer_calls.append(dict(kwargs))
            if len(tokenizer_calls) == 1:
                raise ValueError("no slow tokenizer for this checkpoint")
            return tokenizer

        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
        )

        model = MLXEmbeddingModel(str(root))
        model.load()

        info = model.get_model_info()
        assert info["late_interaction"] is True
        assert info["projection_dim"] == 2
        assert model.hidden_size == 2
        assert model.processor is tokenizer
        # The slow tokenizer is tried first, then the regular (fast) load.
        assert tokenizer_calls[0]["use_fast"] is False
        assert "use_fast" not in tokenizer_calls[1]
        assert model.model.training is False
        assert list(model.model.loaded_weights) == ["final_norm.weight"]
        with pytest.raises(ValueError, match="late-interaction"):
            model.embed(["a query"])

        model.close()
        assert model._colbert_projection is None
        assert model._colbert_layout is None
        assert model._loaded is False

        model.load()
        assert model._colbert_projection is not None
