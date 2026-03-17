# SPDX-License-Identifier: Apache-2.0
"""
MLX Embedding Model wrapper.

This module provides a wrapper around mlx-embeddings for generating
text embeddings using Apple's MLX framework.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx

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
        self._is_compiled = False
        self._compiled_embed = None

    def load(self) -> None:
        """Load the model and processor/tokenizer."""
        if self._loaded:
            return

        try:
            from mlx_embeddings import load

            logger.info(f"Loading embedding model: {self.model_name}")

            self.model, self.processor = load(self.model_name)

            # Get hidden size from model config
            if hasattr(self.model, "config"):
                config = self.model.config
                self._hidden_size = getattr(config, "hidden_size", None)
                if self._hidden_size is None and hasattr(config, "text_config"):
                    self._hidden_size = getattr(config.text_config, "hidden_size", None)

            # Compile a primitive-only embedding forward path
            self._is_compiled = self._try_compile()

            self._loaded = True
            logger.info(
                f"Embedding model loaded successfully: {self.model_name} "
                f"(hidden_size={self._hidden_size}, compiled={self._is_compiled})"
            )

        except ImportError:
            raise ImportError(
                "mlx-embeddings is required for embedding generation. "
                "Install with: pip install mlx-embeddings"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No safetensors weight files found for '{self.model_name}'. "
                f"mlx-embeddings requires models in safetensors format. "
                f"If this is a PyTorch model, use an MLX-converted version "
                f"(e.g., from mlx-community on HuggingFace)."
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _extract_embeddings_array(self, outputs):
        """Extract embedding tensor from model outputs."""
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            return outputs.text_embeds
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return mx.mean(outputs.last_hidden_state, axis=1)
        raise ValueError(
            "Model output does not contain expected embedding fields "
            "(text_embeds, pooler_output, or last_hidden_state)"
        )

    def _try_compile(self) -> bool:
        """
        Compile a primitive-output embedding forward function.

        Root-cause fix:
        - Compiling model.__call__ directly can return arrays without primitives
          for some embedding/reranker models, causing eval() runtime errors.
        - We compile a narrower function that returns only the final embedding array.
        """
        base_model = self.model

        try:
            def _compiled_embed(inputs):
                outputs = base_model(**inputs)
                return self._extract_embeddings_array(outputs)

            # NOTE: shapeless=True can fail output shape inference for AddMM
            # in some embedding models; use default compile mode.
            self._compiled_embed = mx.compile(_compiled_embed)

            # Warmup: verify compilation actually works with a dummy forward pass
            test_inputs = {"input_ids": mx.zeros((1, 4), dtype=mx.int32)}
            _ = self._compiled_embed(test_inputs)

            logger.info(
                f"mx.compile enabled for {self.model_name} "
                f"(primitive embedding path)"
            )
            return True
        except Exception as e:
            logger.info(f"mx.compile unavailable for {self.model_name}: {e}")
            self._compiled_embed = None
            return False

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

        from mlx_embeddings import generate
        from mlx_embeddings.utils import prepare_inputs

        # Normalize input
        if isinstance(texts, str):
            texts = [texts]

        # Get the underlying tokenizer from TokenizerWrapper
        processor = self.processor
        if hasattr(processor, "_tokenizer"):
            processor = processor._tokenizer

        embeddings_array = None

        # Compile path: run primitive-output compiled function
        if self._is_compiled and self._compiled_embed is not None:
            try:
                inputs = prepare_inputs(
                    processor,
                    None,
                    texts,
                    max_length,
                    padding,
                    truncation,
                    None,
                )
                if not isinstance(inputs, dict):
                    inputs = dict(inputs)
                embeddings_array = self._compiled_embed(inputs)
            except Exception as e:
                logger.warning(
                    f"compiled embedding path failed for {self.model_name}: {e}; "
                    f"disabling compile and falling back to eager generate()"
                )
                self._is_compiled = False
                self._compiled_embed = None

        # Eager path fallback
        if embeddings_array is None:
            outputs = generate(
                self.model,
                processor,
                texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
            )
            embeddings_array = self._extract_embeddings_array(outputs)

        # Ensure computation is done
        mx.eval(embeddings_array)

        # Convert to Python list
        embeddings = embeddings_array.tolist()

        # Count tokens
        total_tokens = self._count_tokens(texts)

        # Get dimensions
        dimensions = len(embeddings[0]) if embeddings else 0

        return EmbeddingOutput(
            embeddings=embeddings,
            total_tokens=total_tokens,
            dimensions=dimensions,
        )

    def _count_tokens(self, texts: List[str]) -> int:
        """Count total tokens in input texts."""
        total = 0

        for text in texts:
            if hasattr(self.processor, "encode"):
                # Standard tokenizer
                tokens = self.processor.encode(text, add_special_tokens=True)
                if isinstance(tokens, list):
                    total += len(tokens)
                elif hasattr(tokens, "shape"):
                    total += tokens.shape[-1] if tokens.ndim > 0 else 1
                else:
                    total += len(tokens)
            elif hasattr(self.processor, "tokenizer"):
                # Processor with nested tokenizer
                tokens = self.processor.tokenizer.encode(text, add_special_tokens=True)
                total += len(tokens) if isinstance(tokens, list) else len(list(tokens))
            else:
                # Fallback: estimate based on whitespace
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
        }

        # Try to get model config
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
        return f"<MLXEmbeddingModel model={self.model_name} status={status}>"
