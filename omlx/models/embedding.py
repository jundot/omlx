# SPDX-License-Identifier: Apache-2.0
"""
MLX Embedding Model wrapper.

This module provides a wrapper around mlx-embeddings for generating
text embeddings using Apple's MLX framework, with native fallback
for XLMRoBERTa and BERT embedding models.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingOutput:
    """Output from embedding generation."""

    embeddings: List[List[float]]
    """List of embedding vectors, one per input text."""

    total_tokens: int
    """Total number of tokens in the input."""

    dimensions: int = 0
    """Dimension of each embedding vector."""


class MLXEmbeddingModel:
    """
    Wrapper around mlx-embeddings for generating text embeddings.

    This class provides a unified interface for loading and running
    embedding models using Apple's MLX framework.

    Supports:
    - Native XLMRoBERTa embedding (no mlx-embeddings dependency)
    - Native BERT embedding (no mlx-embeddings dependency)
    - mlx-embeddings fallback for other architectures

    Example:
        >>> model = MLXEmbeddingModel("mlx-community/all-MiniLM-L6-v2-4bit")
        >>> output = model.embed(["Hello, world!", "How are you?"])
        >>> print(len(output.embeddings))  # 2
    """

    def __init__(self, model_name: str):
        """
        Initialize the MLX embedding model.

        Args:
            model_name: HuggingFace model name or local path
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._loaded = False
        self._hidden_size: Optional[int] = None
        self._using_native = False

    def _load_native(self) -> bool:
        """
        Try to load using native omlx implementations (xlm_roberta, bert).

        Returns True if native loading succeeded, False otherwise.
        """
        import mlx.core as mx
        from mlx.utils import tree_unflatten
        from safetensors import safe_open
        from transformers import AutoTokenizer

        model_path = Path(self.model_name)
        config_path = model_path / "config.json"
        if not config_path.exists():
            logger.debug(f"No config.json at {model_path}, native loading skipped")
            return False

        try:
            with open(config_path) as f:
                config_dict = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.debug(f"Failed to read config.json, native loading skipped")
            return False

        architectures = config_dict.get("architectures", [])
        model_type = config_dict.get("model_type", "")
        arch = architectures[0] if architectures else ""

        # Determine if this is a native-supportable embedding architecture
        native_arches = {"XLMRobertaModel", "BertModel", "BertForMaskedLM"}
        if arch not in native_arches:
            logger.debug(
                f"Architecture '{arch}' not natively supported for embedding, "
                f"trying mlx-embeddings"
            )
            return False

        try:
            from .xlm_roberta import Model, ModelArgs

            # Build ModelArgs from config, only using known fields
            known_fields = {f.name for f in ModelArgs.__dataclass_fields__.values()}
            model_config = {
                k: v for k, v in config_dict.items() if k in known_fields
            }
            # Ensure architectures is set correctly
            model_config["architectures"] = architectures

            config = ModelArgs(**model_config)

            # Create model
            model_instance = Model(config)

            # Load weights from safetensors
            weights = {}
            weight_files = list(model_path.glob("*.safetensors"))
            if not weight_files:
                logger.debug(f"No safetensors files found in {model_path}")
                return False

            for wf in weight_files:
                with safe_open(wf, framework="mlx") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)

            # Sanitize and load
            weights = model_instance.sanitize(weights)
            model_instance.load_weights(list(weights.items()))
            mx.eval(model_instance.parameters())

            # Load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            self.model = model_instance
            self.processor = tokenizer
            self._hidden_size = config.hidden_size
            self._loaded = True
            self._using_native = True
            logger.info(
                f"Embedding model loaded natively: {self.model_name} "
                f"(arch={arch}, hidden_size={config.hidden_size})"
            )
            return True

        except Exception as e:
            logger.debug(f"Native loading failed for {self.model_name}: {e}")
            return False

    def load(self) -> None:
        """Load the model and processor/tokenizer."""
        if self._loaded:
            return

        # 1. Try native loading first (xlm_roberta, bert)
        if self._load_native():
            return

        # 2. Fallback to mlx-embeddings
        try:
            from mlx_embeddings import load

            logger.info(f"Loading embedding model via mlx-embeddings: {self.model_name}")

            self.model, self.processor = load(self.model_name)

            # Get hidden size from model config
            if hasattr(self.model, "config"):
                config = self.model.config
                self._hidden_size = getattr(config, "hidden_size", None)
                if self._hidden_size is None:
                    if hasattr(config, "text_config"):
                        self._hidden_size = getattr(
                            config.text_config, "hidden_size", None
                        )

            self._loaded = True
            self._using_native = False
            logger.info(
                f"Embedding model loaded successfully: {self.model_name} "
                f"(hidden_size={self._hidden_size})"
            )

        except ImportError:
            raise ImportError(
                "mlx-embeddings is required for embedding generation. "
                "Install with: pip install mlx-embeddings"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No safetensors weight files found for '{self.model_name}'. "
                f"Embedding models require weights in safetensors format. "
                f"If this is a PyTorch model, use an MLX-converted version "
                f"(e.g., from mlx-community on HuggingFace)."
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ) -> EmbeddingOutput:
        """
        Generate embeddings for input texts.

        Args:
            texts: List of input texts
            max_length: Maximum token length for each text
            padding: Whether to pad shorter sequences
            truncation: Whether to truncate longer sequences

        Returns:
            EmbeddingOutput with embeddings and token count
        """
        if not self._loaded:
            self.load()

        import mlx.core as mx

        # Normalize input
        if isinstance(texts, str):
            texts = [texts]

        # Get tokenizer (may be transformers tokenizer or mlx_embeddings processor)
        processor = self.processor
        if hasattr(processor, "_tokenizer"):
            processor = processor._tokenizer

        if self._using_native:
            # Native mode: use transformers tokenizer directly
            if hasattr(processor, "__call__"):
                encoded = processor(
                    texts,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    return_tensors="np",
                )
                input_ids = mx.array(encoded["input_ids"])
                attention_mask = mx.array(encoded["attention_mask"])
            else:
                # tokenizers.Tokenizer: encode() returns Encoding, need .ids
                encoded_ids = []
                for t in texts:
                    enc = processor.encode(t, add_special_tokens=True)
                    ids = list(enc.ids)[:max_length]
                    encoded_ids.append(ids)
                max_len = max(len(ids) for ids in encoded_ids)
                padded = [ids + [0] * (max_len - len(ids)) for ids in encoded_ids]
                input_ids = mx.array(padded)
                attention_mask = mx.array([[1] * max_len for _ in padded])

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.text_embeds

        else:
            # mlx-embeddings mode: use generate()
            from mlx_embeddings import generate

            outputs = generate(
                self.model,
                processor,
                texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
            )

            # Extract embeddings from output
            if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                embeddings_array = outputs.text_embeds
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings_array = outputs.pooler_output
            elif (
                hasattr(outputs, "last_hidden_state")
                and outputs.last_hidden_state is not None
            ):
                import mlx.core as mx
                embeddings_array = mx.mean(outputs.last_hidden_state, axis=1)
            else:
                raise ValueError(
                    "Model output does not contain expected embedding fields "
                    "(text_embeds, pooler_output, or last_hidden_state)"
                )

            embeddings = embeddings_array

        # Ensure computation is done
        mx.eval(embeddings)

        # Convert to Python list
        embeddings_list = embeddings.tolist()

        # Count tokens
        total_tokens = self._count_tokens(texts)

        # Get dimensions
        dimensions = len(embeddings_list[0]) if embeddings_list else 0

        return EmbeddingOutput(
            embeddings=embeddings_list,
            total_tokens=total_tokens,
            dimensions=dimensions,
        )

    def _count_tokens(self, texts: List[str]) -> int:
        """Count total tokens in input texts."""
        total = 0
        processor = self.processor

        for text in texts:
            if hasattr(processor, "encode"):
                tokens = processor.encode(text, add_special_tokens=True)
                if isinstance(tokens, list):
                    total += len(tokens)
                elif hasattr(tokens, "shape"):
                    total += tokens.shape[-1] if tokens.ndim > 0 else 1
                else:
                    total += len(tokens)
            elif hasattr(processor, "tokenizer"):
                tokens = processor.tokenizer.encode(text, add_special_tokens=True)
                total += len(tokens) if isinstance(tokens, list) else len(list(tokens))
            elif hasattr(processor, "_tokenizer"):
                tokens = processor._tokenizer.encode(text, add_special_tokens=True)
                total += len(tokens) if isinstance(tokens, list) else len(list(tokens))
            else:
                total += len(text.split()) + 2

        return total

    @property
    def hidden_size(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self._hidden_size

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "hidden_size": self._hidden_size,
            "native_implementation": self._using_native,
        }

        if hasattr(self.model, "config"):
            config = self.model.config
            info.update(
                {
                    "model_type": getattr(config, "model_type", None),
                    "vocab_size": getattr(config, "vocab_size", None),
                    "max_position_embeddings": getattr(
                        config, "max_position_embeddings", None
                    ),
                }
            )

        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        impl = "native" if self._using_native else "mlx-embeddings"
        return (
            f"<MLXEmbeddingModel model={self.model_name} "
            f"status={status} impl={impl}>"
        )
