# SPDX-License-Identifier: Apache-2.0
"""Tests for CausalLM-based reranker support."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from safetensors.numpy import save_file

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from omlx.models.reranker import MLXRerankerModel, RerankOutput


class TestXLMRobertaReranker:
    """Tests for native XLM-RoBERTa sequence-classification rerankers."""

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_load_xlm_roberta_switches_to_eval_mode(self, tmp_path):
        """Native reranker load must disable dropout for deterministic scores."""
        from mlx.utils import tree_flatten
        from omlx.models.xlm_roberta import Model, ModelArgs

        config = {
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaForSequenceClassification"],
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "vocab_size": 16,
            "num_attention_heads": 1,
            "intermediate_size": 8,
            "max_position_embeddings": 8,
            "attention_probs_dropout_prob": 0.5,
            "hidden_dropout_prob": 0.5,
            "classifier_dropout": 0.5,
            "pad_token_id": 1,
            "num_labels": 1,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        source_model = Model(ModelArgs(**config))
        mx.save_safetensors(
            str(tmp_path / "model.safetensors"),
            {name: value for name, value in tree_flatten(source_model.parameters())},
        )

        loader = MLXRerankerModel(str(tmp_path))
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=MagicMock(),
        ):
            loaded_model, _ = loader._load_xlm_roberta()

        assert loaded_model.training is False


class TestCausalLMReranker:
    """Tests for CausalLM reranker (e.g., Qwen3-Reranker) functionality."""

    def _make_model_dir(self, tmp_path, name="Qwen3-Reranker-0.6B"):
        """Create a mock model directory with CausalLM reranker config."""
        model_dir = tmp_path / name
        model_dir.mkdir()
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        return model_dir

    def test_validate_architecture_accepts_causal_lm_reranker(self, tmp_path):
        """CausalLM architecture is accepted when directory name contains 'reranker'."""
        model_dir = self._make_model_dir(tmp_path, "Qwen3-Reranker-0.6B")
        model = MLXRerankerModel(str(model_dir))
        # Should not raise
        model._validate_architecture()

    def test_validate_architecture_rejects_plain_causal_lm(self, tmp_path):
        """CausalLM architecture is rejected when directory name lacks reranker hint."""
        model_dir = self._make_model_dir(tmp_path, "Qwen3-0.6B")
        model = MLXRerankerModel(str(model_dir))
        with pytest.raises(ValueError, match="does not contain"):
            model._validate_architecture()

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_rerank_causal_lm_scoring(self, tmp_path):
        """Test _rerank_causal_lm produces correct scores from mocked logits."""
        model_dir = self._make_model_dir(tmp_path)

        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model._loaded = True
        model._token_true_id = 9693  # "yes"
        model._token_false_id = 2152  # "no"
        model._prefix_tokens = [1, 2, 3]
        model._suffix_tokens = [4, 5]

        # Mock tokenizer: return simple token IDs for each document
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[10, 11, 12], [20, 21, 22]],
        }
        model.processor = mock_tokenizer

        # Mock model forward pass: return logits where "yes" > "no" for doc 0,
        # and "no" > "yes" for doc 1

        call_count = [0]

        def mock_forward(input_ids):
            vocab_size = 10000
            seq_len = input_ids.shape[1]
            logits = mx.zeros((1, seq_len, vocab_size))
            # Set logits at last position
            last_pos = np.zeros(vocab_size)
            if call_count[0] == 0:
                # Doc 0: yes=5.0, no=0.0 → high relevance
                last_pos[9693] = 5.0
                last_pos[2152] = 0.0
            else:
                # Doc 1: yes=0.0, no=5.0 → low relevance
                last_pos[9693] = 0.0
                last_pos[2152] = 5.0
            call_count[0] += 1
            # Construct logits with the last position set
            logits_np = np.zeros((1, seq_len, vocab_size), dtype=np.float32)
            logits_np[0, -1, :] = last_pos
            return mx.array(logits_np)

        model.model = MagicMock(side_effect=mock_forward)

        result = model._rerank_causal_lm("test query", ["relevant doc", "irrelevant doc"])

        assert isinstance(result, RerankOutput)
        assert len(result.scores) == 2
        # Doc 0 should have high score (yes >> no)
        assert result.scores[0] > 0.9
        # Doc 1 should have low score (no >> yes)
        assert result.scores[1] < 0.1
        # Sorted indices: doc 0 first
        assert result.indices == [0, 1]
        assert result.total_tokens > 0

    def test_rerank_causal_lm_empty_documents(self, tmp_path):
        """Test rerank with empty document list returns empty result."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model._loaded = True

        result = model.rerank("test query", [])
        assert result.scores == []
        assert result.indices == []
        assert result.total_tokens == 0

    def test_rerank_dispatches_to_causal_lm(self, tmp_path):
        """Test that rerank() dispatches to _rerank_causal_lm when _is_causal_lm is True."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model._loaded = True

        mock_result = RerankOutput(scores=[0.9], indices=[0], total_tokens=10)
        with patch.object(model, "_rerank_causal_lm", return_value=mock_result) as mock_method:
            result = model.rerank("query", ["doc"])
            mock_method.assert_called_once()
            assert result.scores == [0.9]

    def test_max_length_default_for_causal_lm(self, tmp_path):
        """Test that CausalLM reranker uses 8192 as effective max_length by default."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model._loaded = True

        mock_result = RerankOutput(scores=[0.5], indices=[0], total_tokens=10)
        with patch.object(model, "_rerank_causal_lm", return_value=mock_result) as mock_method:
            model.rerank("query", ["doc"])
            # max_length=None should use default 8192 for CausalLM
            args, _ = mock_method.call_args
            assert args[2] == 8192  # query, documents, max_length

    def test_max_length_explicit_override(self, tmp_path):
        """Test that explicit max_length is respected even for CausalLM."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model._loaded = True

        mock_result = RerankOutput(scores=[0.5], indices=[0], total_tokens=10)
        with patch.object(model, "_rerank_causal_lm", return_value=mock_result) as mock_method:
            model.rerank("query", ["doc"], max_length=1024)
            args, _ = mock_method.call_args
            assert args[2] == 1024

    def test_max_length_512_explicit_respected_for_causal_lm(self, tmp_path):
        """Test that explicitly passing max_length=512 is respected (not overridden)."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model._loaded = True

        mock_result = RerankOutput(scores=[0.5], indices=[0], total_tokens=10)
        with patch.object(model, "_rerank_causal_lm", return_value=mock_result) as mock_method:
            model.rerank("query", ["doc"], max_length=512)
            args, _ = mock_method.call_args
            assert args[2] == 512


class TestCausalLMPromptAffixes:
    """Tests for prefix/suffix extraction across chat template shapes."""

    # The reranker-native template Qwen/Qwen3-Reranker-0.6B ships as
    # chat_template.jinja since its 2026-04 sentence-transformers update.
    # It only understands system/query/document roles and drops user messages.
    _NATIVE_TEMPLATE = (
        '{%- set instruction = messages | selectattr("role", "eq", "system") '
        '| map(attribute="content") | first | default("Given a web search '
        'query, retrieve relevant passages that answer the query") -%}\n'
        '{%- set query_text = messages | selectattr("role", "eq", "query") '
        '| map(attribute="content") | first -%}\n'
        '{%- set document_text = messages | selectattr("role", "eq", '
        '"document") | map(attribute="content") | first -%}\n'
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be "
        '"yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
        "<Instruct>: {{ instruction }}\n"
        "<Query>: {{ query_text }}\n"
        "<Document>: {{ document_text }}<|im_end|>\n"
        "<|im_start|>assistant\n"
        # The upstream file ends with "</think>\n\n\n"; jinja strips exactly
        # one trailing newline, so the rendered suffix ends with "</think>\n\n"
        # — byte-identical to the standard-template path.
        "<think>\n\n</think>\n\n\n"
    )

    _EXPECTED_PREFIX = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be "
        '"yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )
    _EXPECTED_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    class _StandardTokenizer:
        """Mimics a standard system/user chat template (e.g., Qwen3 ChatML)."""

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            rendered = ""
            for message in messages:
                rendered += (
                    f"<|im_start|>{message['role']}\n"
                    f"{message['content']}<|im_end|>\n"
                )
            if add_generation_prompt:
                rendered += "<|im_start|>assistant\n"
            return rendered

    class _NativeTokenizer:
        """Mock tokenizer that renders a hard-coded reranker-native Jinja
        template (mirroring the upstream Qwen3-Reranker chat_template.jinja)."""

        def __init__(self, template):
            self._template = template

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            jinja2 = pytest.importorskip("jinja2")
            return (
                jinja2.Environment()
                .from_string(self._template)
                .render(
                    messages=messages,
                    add_generation_prompt=add_generation_prompt,
                )
            )

    def test_standard_template_extracts_affixes(self):
        """Sentinel split on a system/user template yields prefix and suffix."""
        model = MLXRerankerModel("unused")
        prefix, suffix = model._extract_causal_lm_affixes(self._StandardTokenizer())

        assert prefix == self._EXPECTED_PREFIX
        assert suffix == self._EXPECTED_SUFFIX

    def test_native_template_extracts_affixes(self):
        """The reranker-native template (query/document roles) is detected
        after the standard system/user attempt falls through, and yields the
        same affixes as the standard template path."""
        model = MLXRerankerModel("unused")
        tokenizer = self._NativeTokenizer(self._NATIVE_TEMPLATE)

        role_calls = []
        original_apply = tokenizer.apply_chat_template

        def recording_apply(messages, **kwargs):
            role_calls.append([m["role"] for m in messages])
            return original_apply(messages, **kwargs)

        tokenizer.apply_chat_template = recording_apply
        prefix, suffix = model._extract_causal_lm_affixes(tokenizer)

        assert prefix == self._EXPECTED_PREFIX
        assert suffix == self._EXPECTED_SUFFIX
        # The standard system/user attempt must run first and fall through
        # (the native template drops the user message, so the sentinel never
        # appears), then the native query/document attempt succeeds.
        assert role_calls == [
            ["system", "user"],
            ["system", "query", "document"],
        ]

    def test_native_template_rendering_error_falls_through(self):
        """A template that raises on both shapes surfaces both errors."""
        model = MLXRerankerModel("unused")
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = RuntimeError("bad template")

        with pytest.raises(
            ValueError, match="Could not extract CausalLM reranker"
        ) as excinfo:
            model._extract_causal_lm_affixes(tokenizer)

        # Both attempts' errors are in the message, and the original exception
        # is chained for debugging.
        assert "bad template" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_incompatible_template_raises_value_error(self):
        """A template matching neither shape raises instead of mis-splitting,
        and the error includes both rendered attempts."""
        model = MLXRerankerModel("unused")
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "static output, no slots"

        with pytest.raises(ValueError, match="query/document attempt") as excinfo:
            model._extract_causal_lm_affixes(tokenizer)

        assert "static output, no slots" in str(excinfo.value)


class TestJinaReranker:
    """Focused tests for Jina listwise reranker internals."""

    def _make_jina_model_dir(self, tmp_path, name="jina-reranker-v3-mlx"):
        """Create a mock model directory with Jina architecture config."""
        model_dir = tmp_path / name
        model_dir.mkdir()
        config = {
            "model_type": "qwen3",
            "architectures": ["JinaForRanking"],
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        return model_dir

    def test_resolve_token_id_uses_fallback_paths(self):
        """_resolve_token_id should resolve IDs from decoder and convert fallback."""
        model = MLXRerankerModel("unused")

        class _TokenInfo:
            def __init__(self, content):
                self.content = content

        tokenizer = MagicMock()
        tokenizer.added_tokens_decoder = {
            32000: _TokenInfo("<|embed_token|>"),
        }
        tokenizer.convert_tokens_to_ids.side_effect = lambda token: (
            32001 if token == "<|rerank_token|>" else None
        )
        tokenizer.get_added_vocab.return_value = {}

        assert model._resolve_token_id(tokenizer, "<|embed_token|>") == 32000
        assert model._resolve_token_id(tokenizer, "<|rerank_token|>") == 32001

    def test_format_jina_prompt_upstream_parity_invariants(self):
        """_format_jina_prompt should preserve upstream prompt shape and token placement."""
        model = MLXRerankerModel("unused")

        query = "what is green tea"
        docs = ["green tea health benefits", "coffee market prices"]
        instruction = "Prioritize passages that directly answer the question."

        prompt_with_instruction = model._format_jina_prompt(
            query,
            docs,
            instruction=instruction,
        )

        expected_system_prompt = (
            "You are a search relevance expert who can determine a ranking of the "
            "passages based on how relevant they are to the query. If the query is "
            "a question, how relevant a passage is depends on how well it answers "
            "the question. If not, try to analyze the intent of the query and "
            "assess how well each passage satisfies the intent. If an instruction "
            "is provided, you should follow the instruction when determining the "
            "ranking."
        )

        assert expected_system_prompt in prompt_with_instruction
        assert '<passage id="0">' in prompt_with_instruction
        assert '<passage id="1">' in prompt_with_instruction
        assert prompt_with_instruction.index(
            '<passage id="0">'
        ) < prompt_with_instruction.index("<query>")
        assert (
            '<passage id="0">\ngreen tea health benefits<|embed_token|>\n</passage>'
            in prompt_with_instruction
        )
        assert (
            "<query>\nwhat is green tea<|rerank_token|>\n</query>"
            in prompt_with_instruction
        )
        assert (
            "<instruct>\n"
            "Prioritize passages that directly answer the question.\n"
            "</instruct>\n" in prompt_with_instruction
        )
        assert (
            "<|im_start|>assistant\n<think>\n\n</think>\n\n" in prompt_with_instruction
        )
        assert "</query><|im_end|>" in prompt_with_instruction

        prompt_without_instruction = model._format_jina_prompt(query, docs)
        assert "<instruct>" not in prompt_without_instruction

    def test_load_jina_projector_missing_file_raises_clear_error(self, tmp_path):
        """Missing projector.safetensors should raise a clear FileNotFoundError."""
        model = MLXRerankerModel("unused")

        with pytest.raises(FileNotFoundError, match="projector.safetensors"):
            model._load_jina_projector(tmp_path)

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_get_jina_hidden_states_accepts_3d_tensor(self):
        """_get_jina_hidden_states should return 3D backbone outputs unchanged."""
        model = MLXRerankerModel("unused")
        expected = mx.array(np.zeros((1, 4, 8), dtype=np.float32))

        model.model = MagicMock()
        model.model.model = MagicMock(return_value=expected)

        input_ids = mx.array([[1, 2, 3, 4]])
        actual = model._get_jina_hidden_states(input_ids)

        assert actual.shape == (1, 4, 8)
        assert np.allclose(np.array(actual.tolist()), np.array(expected.tolist()))

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_get_jina_hidden_states_expands_2d_tensor(self):
        """_get_jina_hidden_states should expand 2D backbone outputs to batch form."""
        model = MLXRerankerModel("unused")
        returned = mx.array(np.zeros((4, 8), dtype=np.float32))

        model.model = MagicMock()
        model.model.model = MagicMock(return_value=returned)

        input_ids = mx.array([[1, 2, 3, 4]])
        actual = model._get_jina_hidden_states(input_ids)

        assert actual.shape == (1, 4, 8)

    def test_get_jina_hidden_states_missing_backbone_raises_clear_error(self):
        """_get_jina_hidden_states should fail clearly when model.model is missing."""
        model = MLXRerankerModel("unused")
        model.model = object()

        with pytest.raises(ValueError, match="Could not find Jina model backbone"):
            model._get_jina_hidden_states("input_ids")

    def test_get_jina_hidden_states_rejects_unsupported_output(self):
        """_get_jina_hidden_states should reject non-tensor backbone outputs."""
        model = MLXRerankerModel("unused")

        class _UnsupportedOutput:
            pass

        model.model = MagicMock()
        model.model.model = MagicMock(return_value=_UnsupportedOutput())

        with pytest.raises(
            ValueError, match="did not return hidden states as a tensor"
        ):
            model._get_jina_hidden_states("input_ids")

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_get_jina_hidden_states_rejects_invalid_tensor_rank(self):
        """_get_jina_hidden_states should reject tensor outputs with unsupported rank."""
        model = MLXRerankerModel("unused")
        invalid = mx.array(np.zeros((1, 2, 3, 4), dtype=np.float32))
        model.model = MagicMock()
        model.model.model = MagicMock(return_value=invalid)
        input_ids = mx.array([[1, 2, 3, 4]])
        with pytest.raises(ValueError, match="Jina hidden states must be rank 2 or 3"):
            model._get_jina_hidden_states(input_ids)

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_load_jina_projector_two_layer_mlp(self, tmp_path):
        """Projector should apply linear1 -> ReLU -> linear2 exactly."""
        model_dir = self._make_jina_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))

        w1 = np.zeros((512, 1024), dtype=np.float32)
        w2 = np.zeros((512, 512), dtype=np.float32)

        w1[0, 0] = 1.5
        w1[1, 1] = -2.0
        w1[2, 2] = 0.5

        w2[0, 0] = 1.0
        w2[1, 1] = -3.0
        w2[3, 2] = 2.0

        save_file(
            {
                "linear1.weight": w1,
                "linear2.weight": w2,
            },
            str(model_dir / "projector.safetensors"),
        )

        projector = model._load_jina_projector(model_dir)

        x = np.zeros((2, 1024), dtype=np.float32)
        x[0, 0] = 2.0
        x[0, 1] = 1.0
        x[0, 2] = 4.0
        x[1, 0] = -3.0
        x[1, 1] = 5.0
        x[1, 2] = -2.0

        projected = projector(mx.array(x))
        mx.eval(projected)

        expected = np.maximum(x @ w1.T, 0.0) @ w2.T
        actual = np.array(projected.tolist(), dtype=np.float32)
        assert np.allclose(actual, expected, atol=1e-6)

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_rerank_jina_returns_scores_and_sorted_indices(self, tmp_path):
        """_rerank_jina should produce per-doc scores and descending indices."""
        model_dir = self._make_jina_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._loaded = True
        model._is_jina_reranker = True
        model._doc_embed_token_id = 2001
        model._query_embed_token_id = 2002
        model._jina_projector = lambda x: x

        class _Tokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                ids = []
                for piece in text.replace("\n", " ").split():
                    if "<|rerank_token|>" in piece:
                        ids.append(2002)
                        remainder = piece.replace("<|rerank_token|>", "")
                        if remainder:
                            ids.append(7)
                    elif "<|embed_token|>" in piece:
                        ids.append(2001)
                        remainder = piece.replace("<|embed_token|>", "")
                        if remainder:
                            ids.append(7)
                    else:
                        ids.append(7)
                return ids

            def decode(self, token_ids, skip_special_tokens=False):
                del skip_special_tokens
                return " ".join(["tok"] * len(token_ids))

        model.processor = _Tokenizer()

        def _fake_hidden_states(input_ids):
            token_ids = input_ids[0].tolist()
            hidden_states = np.zeros((1, len(token_ids), 2), dtype=np.float32)
            doc_vectors = ([0.6, 0.8], [0.95, 0.1], [-0.2, 0.0])
            doc_idx = 0
            for pos, token_id in enumerate(token_ids):
                if token_id == 2002:
                    hidden_states[0, pos, :] = np.array([1.0, 0.0], dtype=np.float32)
                elif token_id == 2001 and doc_idx < len(doc_vectors):
                    hidden_states[0, pos, :] = np.array(
                        doc_vectors[doc_idx], dtype=np.float32
                    )
                    doc_idx += 1
            return mx.array(hidden_states)

        with patch.object(
            model, "_get_jina_hidden_states", side_effect=_fake_hidden_states
        ):
            result = model._rerank_jina(
                "query", ["doc a", "doc b", "doc c"], max_length=256
            )

        assert len(result.scores) == 3
        assert result.scores[1] > result.scores[0] > result.scores[2]
        assert result.indices == [1, 0, 2]
        assert result.total_tokens > 0

    def test_rerank_dispatch_and_max_length_for_jina(self, tmp_path):
        """rerank() should dispatch to _rerank_jina and honor max_length semantics."""
        model_dir = self._make_jina_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._loaded = True
        model._is_jina_reranker = True

        mock_result = RerankOutput(scores=[0.9], indices=[0], total_tokens=10)
        with patch.object(
            model, "_rerank_jina", return_value=mock_result
        ) as mock_method:
            model.rerank("query", ["doc"])
            args, _ = mock_method.call_args
            assert args[2] == 8192

        with patch.object(
            model, "_rerank_jina", return_value=mock_result
        ) as mock_method:
            model.rerank("query", ["doc"], max_length=1024)
            args, _ = mock_method.call_args
            assert args[2] == 1024


class TestRerankerCompileFallback:
    """Tests for reranker compiled path fallback behavior."""

    def _make_model_dir(self, tmp_path, name="bge-reranker-v2-m3"):
        """Create a mock model directory with SequenceClassification config."""
        model_dir = tmp_path / name
        model_dir.mkdir()
        config = {
            "model_type": "modernbert",
            "architectures": ["ModernBertForSequenceClassification"],
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        return model_dir

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_compiled_path_fallback_on_failure(self, tmp_path):
        """Test that _rerank_seq_classification falls back to eager on compile failure."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._loaded = True
        model._is_causal_lm = False
        model._is_compiled = True
        model._compiled_seq_logits = MagicMock(
            side_effect=RuntimeError("compile fail")
        )

        # Mock processor
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": [[1, 2, 3, 4]],
            "attention_mask": [[1, 1, 1, 1]],
        }
        model.processor = mock_processor

        # Mock model to return pooler_output
        mock_outputs = MagicMock(spec=[])
        mock_outputs.pooler_output = mx.array([[0.85]])
        model.model = MagicMock(return_value=mock_outputs)

        result = model._rerank_seq_classification("query", ["doc"])

        assert len(result.scores) == 1
        # Compiled path failed, eager path should have been used
        model.model.assert_called_once()

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_eager_path_when_not_compiled(self, tmp_path):
        """Test that _rerank_seq_classification uses eager path when not compiled."""
        model_dir = self._make_model_dir(tmp_path)
        model = MLXRerankerModel(str(model_dir))
        model._loaded = True
        model._is_causal_lm = False
        model._is_compiled = False
        model._compiled_seq_logits = None

        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        model.processor = mock_processor

        mock_outputs = MagicMock(spec=[])
        mock_outputs.pooler_output = mx.array([[0.7]])
        model.model = MagicMock(return_value=mock_outputs)

        result = model._rerank_seq_classification("query", ["doc"])

        assert len(result.scores) == 1
        model.model.assert_called_once()

    def test_try_compile_skips_causal_lm(self, tmp_path):
        """Test that _try_compile returns False for causal-lm rerankers."""
        model_dir = tmp_path / "Qwen3-Reranker-0.6B"
        model_dir.mkdir()
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        model = MLXRerankerModel(str(model_dir))
        model._is_causal_lm = True
        model.model = MagicMock()

        result = model._try_compile()

        assert result is False
        assert model._compiled_seq_logits is None


class TestRerankerClose:
    """Tests for reranker unload resource release."""

    def test_close_releases_compiled_model_and_processor_resources(self):
        """close() should drop wrapper references before clearing MLX caches."""
        model = MLXRerankerModel("test-model")
        model.model = MagicMock()
        model.processor = MagicMock()
        model._loaded = True
        model._num_labels = 1
        model._is_causal_lm = True
        model._is_jina_reranker = True
        model._is_vl_reranker = True
        model._token_true_id = 1
        model._token_false_id = 2
        model._doc_embed_token_id = 3
        model._query_embed_token_id = 4
        model._jina_projector = MagicMock()
        model._prefix_tokens = [5]
        model._suffix_tokens = [6]
        model._is_compiled = True
        model._compiled_seq_logits = MagicMock()

        with (
            patch("omlx.models.reranker.gc.collect") as collect,
            patch("omlx.models.reranker.mx") as mock_mx,
            patch(
                "omlx.models.reranker.clear_thread_compile_cache"
            ) as clear_compile_cache,
        ):
            model.close()

        assert model.model is None
        assert model.processor is None
        assert model._compiled_seq_logits is None
        assert model._loaded is False
        assert model._num_labels is None
        assert model._is_causal_lm is False
        assert model._is_jina_reranker is False
        assert model._is_vl_reranker is False
        assert model._token_true_id is None
        assert model._token_false_id is None
        assert model._doc_embed_token_id is None
        assert model._query_embed_token_id is None
        assert model._jina_projector is None
        assert model._prefix_tokens is None
        assert model._suffix_tokens is None
        assert model._is_compiled is False
        mock_mx.synchronize.assert_called_once()
        mock_mx.clear_cache.assert_called_once()
        clear_compile_cache.assert_called_once()
        assert collect.call_count == 2
