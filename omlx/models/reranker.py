# SPDX-License-Identifier: Apache-2.0
"""
MLX Reranker Model wrapper.

This module provides a wrapper for document reranking using SequenceClassification
and CausalLM-based reranker models on Apple's MLX framework.

Supports:
- ModernBertForSequenceClassification (via mlx-embeddings)
- XLMRobertaForSequenceClassification (omlx native implementation)
- CausalLM-based rerankers (e.g., Qwen3-Reranker) via yes/no logit scoring
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

from ..model_discovery import (
    CAUSAL_LM_RERANKER_ARCHITECTURES,
    SUPPORTED_RERANKER_ARCHITECTURES,
)

logger = logging.getLogger(__name__)


@dataclass
class RerankOutput:
    """Output from rerank operation."""

    scores: list[float]
    """Relevance scores for each document (0 to 1)."""

    indices: list[int]
    """Document indices sorted by score (descending)."""

    total_tokens: int
    """Total number of tokens processed."""


class MLXRerankerModel:
    """
    Wrapper for document reranking on Apple's MLX framework.

    Supports two reranking paradigms:

    1. SequenceClassification models (encoder-based):
       - ModernBertForSequenceClassification (via mlx-embeddings)
       - XLMRobertaForSequenceClassification (omlx native implementation)

    2. CausalLM-based rerankers (decoder-based):
       - Qwen3-Reranker and similar models that use yes/no logit scoring
       - Uses instruction prompts and extracts relevance from token logits

    Example:
        >>> model = MLXRerankerModel("BAAI/bge-reranker-v2-m3")
        >>> model.load()
        >>> output = model.rerank("What is ML?", ["ML is...", "Weather is..."])
        >>> print(output.scores)  # [0.95, 0.12]
    """

    # CausalLM reranker prompt template (Qwen3-Reranker format)
    _CAUSAL_LM_SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the "
        'Query and the Instruct provided. Note that the answer can only be '
        '"yes" or "no".'
    )
    _CAUSAL_LM_DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def __init__(self, model_name: str):
        """
        Initialize the MLX reranker model.

        Args:
            model_name: HuggingFace model name or local path
        """
        self.model_name = model_name

        self.model = None
        self.processor = None
        self._loaded = False
        self._num_labels: int | None = None
        self._is_causal_lm = False
        self._token_true_id: int | None = None
        self._token_false_id: int | None = None
        self._prefix_tokens: list[int] | None = None
        self._suffix_tokens: list[int] | None = None

    def _get_architecture(self) -> str | None:
        """Get the model architecture from config.json."""
        config_path = Path(self.model_name) / "config.json"
        if not config_path.exists():
            return None

        try:
            with open(config_path) as f:
                config = json.load(f)
            architectures = config.get("architectures", [])
            return architectures[0] if architectures else None
        except (json.JSONDecodeError, IOError):
            return None

    def _load_xlm_roberta(self) -> Tuple[Any, Any]:
        """Load XLMRoberta model using omlx native implementation."""
        import mlx.core as mx
        from mlx.utils import tree_unflatten
        from safetensors import safe_open
        from transformers import AutoTokenizer

        from .xlm_roberta import Model, ModelArgs

        model_path = Path(self.model_name)

        # Load config
        with open(model_path / "config.json") as f:
            config_dict = json.load(f)

        config = ModelArgs(**{
            k: v for k, v in config_dict.items()
            if k in ModelArgs.__dataclass_fields__
        })

        # Create model
        model = Model(config)

        # Load weights
        weights = {}
        weight_files = list(model_path.glob("*.safetensors"))
        for wf in weight_files:
            with safe_open(wf, framework="mlx") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)

        # Sanitize weights (remove "roberta." prefix, etc.)
        weights = model.sanitize(weights)

        # Load weights into model
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        return model, tokenizer

    def _load_causal_lm(self) -> Tuple[Any, Any]:
        """Load a CausalLM-based reranker model using mlx-lm."""
        from mlx_lm import load as mlx_lm_load

        model_path = str(self.model_name)
        model, tokenizer_wrapper = mlx_lm_load(model_path)

        # mlx-lm returns a TokenizerWrapper; unwrap to get the underlying
        # transformers tokenizer which supports __call__ for batch encoding.
        tokenizer = getattr(tokenizer_wrapper, "_tokenizer", tokenizer_wrapper)

        # Resolve yes/no token IDs from tokenizer
        self._token_true_id = tokenizer.convert_tokens_to_ids("yes")
        self._token_false_id = tokenizer.convert_tokens_to_ids("no")

        if self._token_true_id is None or self._token_false_id is None:
            raise ValueError(
                "Could not find 'yes'/'no' token IDs in tokenizer. "
                "This model may not be a compatible CausalLM reranker."
            )

        # Pre-compute prefix and suffix tokens for the prompt template.
        # Use apply_chat_template() for portability across tokenizer formats,
        # then split on a sentinel to extract prefix/suffix boundaries.
        _SENTINEL = "<<__CONTENT_SENTINEL__>>"
        messages = [
            {"role": "system", "content": self._CAUSAL_LM_SYSTEM_PROMPT},
            {"role": "user", "content": _SENTINEL},
        ]
        template_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        parts = template_str.split(_SENTINEL)
        if len(parts) != 2:
            raise ValueError(
                f"Chat template produced unexpected format; "
                f"could not split on sentinel. Template: {template_str!r}"
            )
        prefix = parts[0]
        # Append <think> block for models that use thinking-then-answering format
        suffix = parts[1] + "<think>\n\n</think>\n\n"

        self._prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

        logger.info(
            f"CausalLM reranker tokens: yes={self._token_true_id}, "
            f"no={self._token_false_id}, "
            f"prefix_len={len(self._prefix_tokens)}, "
            f"suffix_len={len(self._suffix_tokens)}"
        )

        return model, tokenizer

    def load(self) -> None:
        """Load the model and processor/tokenizer."""
        if self._loaded:
            return

        # Check architecture before loading
        self._validate_architecture()

        arch = self._get_architecture()
        logger.info(f"Loading reranker model: {self.model_name} (arch={arch})")

        try:
            if arch in CAUSAL_LM_RERANKER_ARCHITECTURES:
                # CausalLM-based reranker (e.g., Qwen3-Reranker)
                self.model, self.processor = self._load_causal_lm()
                self._is_causal_lm = True
                self._num_labels = 2  # yes/no
            elif arch == "XLMRobertaForSequenceClassification":
                # Use omlx native implementation
                self.model, self.processor = self._load_xlm_roberta()
                self._num_labels = getattr(self.model.config, "num_labels", None)
            else:
                # Use mlx-embeddings for other architectures (ModernBert, etc.)
                from mlx_embeddings import load
                self.model, self.processor = load(self.model_name)

                # Get num_labels from model config
                if hasattr(self.model, "config"):
                    config = self.model.config
                    self._num_labels = getattr(config, "num_labels", None)

            self._loaded = True
            logger.info(
                f"Reranker model loaded successfully: {self.model_name} "
                f"(arch={arch}, num_labels={self._num_labels}, "
                f"causal_lm={self._is_causal_lm})"
            )

        except ImportError as e:
            raise ImportError(
                "mlx-lm, mlx-embeddings, or transformers is required for reranking. "
                "Install with: pip install mlx-lm mlx-embeddings transformers"
            ) from e
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No safetensors weight files found for '{self.model_name}'. "
                f"Reranker models require weights in safetensors format. "
                f"If this is a PyTorch model, use an MLX-converted version "
                f"(e.g., from mlx-community on HuggingFace)."
            )
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    # Default max_length per model type
    _DEFAULT_MAX_LENGTH_SEQ_CLASSIFICATION = 512
    _DEFAULT_MAX_LENGTH_CAUSAL_LM = 8192

    def rerank(
        self,
        query: str,
        documents: list[str],
        max_length: int | None = None,
    ) -> RerankOutput:
        """
        Rerank documents by relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank
            max_length: Maximum token length for each query-document pair.
                If None, uses model-appropriate default (512 for encoder,
                8192 for CausalLM).

        Returns:
            RerankOutput with scores, sorted indices, and token count
        """
        if not self._loaded:
            self.load()

        if not documents:
            return RerankOutput(scores=[], indices=[], total_tokens=0)

        if self._is_causal_lm:
            effective_max_length = (
                max_length
                if max_length is not None
                else self._DEFAULT_MAX_LENGTH_CAUSAL_LM
            )
            return self._rerank_causal_lm(query, documents, effective_max_length)
        else:
            effective_max_length = (
                max_length
                if max_length is not None
                else self._DEFAULT_MAX_LENGTH_SEQ_CLASSIFICATION
            )
            return self._rerank_seq_classification(
                query, documents, effective_max_length
            )

    def _rerank_causal_lm(
        self,
        query: str,
        documents: list[str],
        max_length: int = 8192,
    ) -> RerankOutput:
        """
        Rerank using CausalLM yes/no logit scoring (e.g., Qwen3-Reranker).

        Constructs instruction prompts, runs per-document forward passes, and
        extracts relevance scores from the logits of yes/no tokens at the last
        position. Each document is processed individually since mlx-lm models
        generate their own causal mask internally and don't accept an external
        padding mask.
        """
        import mlx.core as mx

        tokenizer = self.processor
        prefix_tokens = self._prefix_tokens
        suffix_tokens = self._suffix_tokens

        # Compute max tokens available for the instruction content
        max_content_tokens = max_length - len(prefix_tokens) - len(suffix_tokens)

        # Format and tokenize each query-document pair
        pairs_text = []
        for doc in documents:
            content = (
                f"<Instruct>: {self._CAUSAL_LM_DEFAULT_INSTRUCTION}\n"
                f"<Query>: {query}\n"
                f"<Document>: {doc}"
            )
            pairs_text.append(content)

        # Tokenize content parts (without prefix/suffix)
        content_encodings = tokenizer(
            pairs_text,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=max_content_tokens,
            add_special_tokens=False,
        )

        # Assemble full token sequences: prefix + content + suffix
        all_input_ids = []
        for content_ids in content_encodings["input_ids"]:
            full_ids = prefix_tokens + content_ids + suffix_tokens
            all_input_ids.append(full_ids)

        # Per-document forward pass and score extraction.
        # mlx-lm models generate their own causal attention mask internally
        # and don't support external padding masks, so we process each
        # document individually to ensure correct attention computation.
        scores = []
        total_tokens = 0
        for ids in all_input_ids:
            input_ids = mx.array([ids])  # (1, seq_len)
            logits = self.model(input_ids)
            # Extract yes/no logits at the last position
            last_logits = logits[0, -1, :]
            true_logit = last_logits[self._token_true_id]
            false_logit = last_logits[self._token_false_id]
            paired = mx.array([false_logit, true_logit])
            probs = mx.softmax(paired)
            mx.eval(probs)
            scores.append(probs[1].item())
            total_tokens += len(ids)

        # Sort indices by score (descending)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in indexed_scores]

        return RerankOutput(
            scores=scores,
            indices=sorted_indices,
            total_tokens=total_tokens,
        )

    def _rerank_seq_classification(
        self,
        query: str,
        documents: list[str],
        max_length: int = 512,
    ) -> RerankOutput:
        """Rerank using SequenceClassification models (encoder-based)."""
        import mlx.core as mx

        # Get the underlying tokenizer from TokenizerWrapper (mlx-embeddings only)
        # Don't unwrap transformers tokenizers which also have _tokenizer attribute
        processor = self.processor
        processor_class = type(processor).__name__
        if processor_class == "TokenizerWrapper" and hasattr(processor, "_tokenizer"):
            processor = processor._tokenizer

        # Tokenize query-document pairs
        # SequenceClassification models expect pairs as (query, document)
        pairs = [(query, doc) for doc in documents]

        # Batch encode all pairs
        inputs = processor(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="np",
        )

        # Convert to MLX arrays
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract scores from pooler_output
        # pooler_output shape: (batch_size, num_labels)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            logits = outputs.pooler_output
        else:
            raise ValueError(
                "Model output does not contain pooler_output. "
                "Ensure the model is a SequenceClassification model."
            )

        # Ensure computation is done
        mx.eval(logits)

        # Extract relevance scores
        # For binary classification (num_labels=1), score is already sigmoid applied
        # For multi-class, take the positive class probability
        if logits.shape[-1] == 1:
            # Binary classification: sigmoid already applied by model
            scores = logits.squeeze(-1).tolist()
        else:
            # Multi-class: take last column (typically "relevant" class)
            scores = logits[:, -1].tolist()

        # Sort indices by score (descending)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in indexed_scores]

        # Count tokens
        total_tokens = self._count_tokens(query, documents)

        return RerankOutput(
            scores=scores,
            indices=sorted_indices,
            total_tokens=total_tokens,
        )

    def _count_tokens(self, query: str, documents: list[str]) -> int:
        """Count total tokens in query-document pairs."""
        total = 0

        processor = self.processor
        processor_class = type(processor).__name__
        if processor_class == "TokenizerWrapper" and hasattr(processor, "_tokenizer"):
            processor = processor._tokenizer

        def get_token_count(text: str, add_special: bool = True) -> int:
            """Get token count for text, handling different tokenizer types."""
            if hasattr(processor, "encode"):
                tokens = processor.encode(text, add_special_tokens=add_special)
                # Handle different return types
                if isinstance(tokens, list):
                    return len(tokens)
                elif hasattr(tokens, "ids"):
                    # tokenizers.Encoding object
                    return len(tokens.ids)
                else:
                    return len(tokens)
            else:
                # Fallback to word count estimate
                return len(text.split()) + (2 if add_special else 0)

        # Count query tokens once
        query_len = get_token_count(query, add_special=True)

        # Count document tokens
        for doc in documents:
            doc_len = get_token_count(doc, add_special=False)
            # Each pair includes query + doc + special tokens
            total += query_len + doc_len + 3  # [CLS], [SEP], [SEP]

        return total

    @property
    def num_labels(self) -> int | None:
        """Get the number of classification labels."""
        return self._num_labels

    def _validate_architecture(self) -> None:
        """
        Validate that the model architecture is supported.

        Raises:
            ValueError: If the architecture is not supported
        """
        config_path = Path(self.model_name) / "config.json"
        if not config_path.exists():
            # If no config.json, let mlx-embeddings handle validation
            return

        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read config.json: {e}")
            return

        architectures = config.get("architectures", [])
        if not architectures:
            return

        arch = architectures[0]

        # CausalLM reranker architectures require the directory name heuristic
        # to distinguish from regular LLMs with the same architecture.
        if arch in CAUSAL_LM_RERANKER_ARCHITECTURES:
            from ..model_discovery import _is_causal_lm_reranker

            if not _is_causal_lm_reranker(Path(self.model_name)):
                raise ValueError(
                    f"Architecture {arch} is a CausalLM that can be used as a "
                    f"reranker, but the model directory name "
                    f"'{Path(self.model_name).name}' does not contain "
                    f"'reranker' or 'rerank'. Please rename the directory or "
                    f"use the correct model."
                )
            return

        if arch not in SUPPORTED_RERANKER_ARCHITECTURES:
            supported_list = ", ".join(
                sorted(SUPPORTED_RERANKER_ARCHITECTURES | CAUSAL_LM_RERANKER_ARCHITECTURES)
            )
            raise ValueError(
                f"Unsupported reranker architecture: {arch}. "
                f"Currently supported architectures: {supported_list}."
            )

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "num_labels": self._num_labels,
        }

        # Try to get model config
        if hasattr(self.model, "config"):
            config = self.model.config
            info.update(
                {
                    "model_type": getattr(config, "model_type", None),
                    "hidden_size": getattr(config, "hidden_size", None),
                    "max_position_embeddings": getattr(
                        config, "max_position_embeddings", None
                    ),
                }
            )

        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXRerankerModel model={self.model_name} status={status}>"
