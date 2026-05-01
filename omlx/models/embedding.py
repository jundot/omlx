# SPDX-License-Identifier: Apache-2.0
"""
MLX Embedding Model wrapper.

This module provides a wrapper around mlx-embeddings for generating
text embeddings using Apple's MLX framework, with native fallback
for XLMRoBERTa and BERT embedding models.
"""

import json
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
from mlx.utils import tree_flatten

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

    def __init__(self, model_name: str, trust_remote_code: bool = False):
        """
        Initialize the MLX embedding model.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Allow execution of custom Python shipped inside
                the model repository. Off by default for security (issue #926).
        """
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code

        self.model = None
        self.processor = None
        self._loaded = False
        self._hidden_size: Optional[int] = None
        self._using_native = False
        self._is_compiled = False
        self._compiled_embed = None
        self._remap_input_ids_to_inputs = False

    def _load_native(self) -> bool:
        """
        Try to load using native omlx implementations (xlm_roberta, bert).

        Returns True if native loading succeeded, False otherwise.
        """
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
            logger.debug("Failed to read config.json, native loading skipped")
            return False

        architectures = config_dict.get("architectures", [])
        arch = architectures[0] if architectures else ""

        native_arches = {"XLMRobertaModel", "BertModel", "BertForMaskedLM"}
        if arch not in native_arches:
            logger.debug(
                f"Architecture '{arch}' not natively supported for embedding, "
                "trying mlx-embeddings"
            )
            return False

        try:
            from .xlm_roberta import Model, ModelArgs

            known_fields = {f.name for f in ModelArgs.__dataclass_fields__.values()}
            model_config = {
                k: v for k, v in config_dict.items() if k in known_fields
            }
            model_config["architectures"] = architectures

            config = ModelArgs(**model_config)
            model_instance = Model(config)

            weights = {}
            weight_files = list(model_path.glob("*.safetensors"))
            if not weight_files:
                logger.debug(f"No safetensors files found in {model_path}")
                return False

            for wf in weight_files:
                with safe_open(wf, framework="mlx") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)

            weights = model_instance.sanitize(weights)
            self._validate_native_weights(model_instance, weights)
            model_instance.load_weights(list(weights.items()), strict=False)
            mx.eval(model_instance.parameters())

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    use_fast=False,
                    trust_remote_code=self.trust_remote_code,
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=self.trust_remote_code,
                )

            self.model = model_instance
            self.processor = tokenizer
            self._hidden_size = config.hidden_size
            self._loaded = True
            self._using_native = True
            self._is_compiled = False
            self._compiled_embed = None
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

            self.model, self.processor = load(
                self.model_name,
                tokenizer_config={"trust_remote_code": self.trust_remote_code},
            )

            if hasattr(self.model, "config"):
                config = self.model.config
                self._hidden_size = getattr(config, "hidden_size", None)
                if self._hidden_size is None and hasattr(config, "text_config"):
                    self._hidden_size = getattr(config.text_config, "hidden_size", None)

            self._using_native = False
            self._detect_input_key_remapping()
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
                f"Embedding models require weights in safetensors format. "
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

    def _validate_native_weights(
        self, model_instance, weights: Dict[str, Any]
    ) -> None:
        """Reject native checkpoints with missing or shape-incompatible core weights."""
        expected_weights = dict(tree_flatten(model_instance.parameters()))
        expected_weight_names = set(expected_weights.keys())
        provided_weight_names = set(weights.keys())
        missing_weight_names = expected_weight_names - provided_weight_names

        optional_missing_prefixes = ("pooler.",)
        required_missing = sorted(
            name
            for name in missing_weight_names
            if not name.startswith(optional_missing_prefixes)
        )
        if required_missing:
            preview = ", ".join(required_missing[:10])
            suffix = "..." if len(required_missing) > 10 else ""
            raise ValueError(
                "Native embedding checkpoint is missing required weights: "
                f"{preview}{suffix}"
            )

        shape_mismatches = []
        for name in expected_weight_names & provided_weight_names:
            expected_shape = tuple(expected_weights[name].shape)
            provided_shape = tuple(weights[name].shape)
            if expected_shape != provided_shape:
                shape_mismatches.append((name, expected_shape, provided_shape))

        if shape_mismatches:
            preview = ", ".join(
                f"{name}: expected {expected_shape}, got {provided_shape}"
                for name, expected_shape, provided_shape in shape_mismatches[:5]
            )
            suffix = "..." if len(shape_mismatches) > 5 else ""
            raise ValueError(
                "Native embedding checkpoint has incompatible weight shapes: "
                f"{preview}{suffix}"
            )

    def _uses_custom_embedding_inputs(self, processor) -> bool:
        """Return True when processor exposes a custom embedding input API."""
        for attr_name in ("prepare_embedding_inputs", "prepare_model_inputs"):
            try:
                inspect.getattr_static(processor, attr_name)
                return True
            except AttributeError:
                continue
        return False

    def _normalize_embedding_inputs(
        self,
        inputs: Union[str, Dict[str, str], List[str], List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        """Normalize embedding inputs into item dicts."""
        if not inputs:
            return []
        if isinstance(inputs, str):
            return [{"text": inputs}]
        if isinstance(inputs, dict):
            return [dict(inputs)]
        first = inputs[0]
        if isinstance(first, str):
            return [{"text": text} for text in inputs]
        return [dict(item) for item in inputs]

    def _prepare_embedding_inputs(
        self,
        processor,
        inputs: Union[List[str], List[Dict[str, str]]],
        max_length: int,
        padding: bool,
        truncation: bool,
    ):
        """
        Prepare inputs for embedding inference.

        Some embedding processors, such as qwen3_vl in mlx-embeddings, expose
        a higher-level embedding API instead of the tokenizer-style
        ``processor(texts, ...)`` path. Reuse that official extension point
        when available to avoid positional-argument mismatches.
        """
        normalized_inputs = self._normalize_embedding_inputs(inputs)

        if self._uses_custom_embedding_inputs(processor):
            if hasattr(processor, "prepare_embedding_inputs"):
                return processor.prepare_embedding_inputs(
                    normalized_inputs, return_tensors="mlx"
                )
            return processor.prepare_model_inputs(
                normalized_inputs, return_tensors="mlx"
            )

        if any("image" in item for item in normalized_inputs):
            raise ValueError(
                f"Embedding model '{self.model_name}' does not support image inputs"
            )

        from mlx_embeddings.utils import prepare_inputs

        return prepare_inputs(
            processor,
            None,
            [item.get("text", "") for item in normalized_inputs],
            max_length,
            padding,
            truncation,
            None,
        )

    def _detect_input_key_remapping(self) -> None:
        """Check if the model accepts `inputs` instead of `input_ids` and cache the result."""
        try:
            params = inspect.signature(self.model.__call__).parameters
            self._remap_input_ids_to_inputs = (
                "input_ids" not in params and "inputs" in params
            )
        except (TypeError, ValueError):
            self._remap_input_ids_to_inputs = False

    def _adapt_model_inputs_for_call(
        self, model_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rename prepared inputs to match the embedding model call signature."""
        adapted_inputs = dict(model_inputs)
        if self._remap_input_ids_to_inputs and "input_ids" in adapted_inputs:
            adapted_inputs["inputs"] = adapted_inputs.pop("input_ids")
        return adapted_inputs

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
                outputs = base_model(**self._adapt_model_inputs_for_call(inputs))
                return self._extract_embeddings_array(outputs)

            self._compiled_embed = mx.compile(_compiled_embed)

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
        inputs: Union[str, List[str], List[Dict[str, str]]],
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

        normalized_inputs = self._normalize_embedding_inputs(inputs)
        input_texts = [item["text"] for item in normalized_inputs if "text" in item]
        has_image_inputs = any("image" in item for item in normalized_inputs)

        processor = self.processor
        uses_custom_embedding_inputs = self._uses_custom_embedding_inputs(processor)
        if hasattr(processor, "_tokenizer") and not uses_custom_embedding_inputs:
            processor = processor._tokenizer

        if has_image_inputs and (self._using_native or not uses_custom_embedding_inputs):
            raise ValueError(
                f"Embedding model '{self.model_name}' does not support image inputs"
            )

        embeddings_array = None
        total_tokens: Optional[int] = None

        if self._using_native:
            if hasattr(processor, "__call__"):
                encoded = processor(
                    input_texts,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    return_tensors="np",
                )
                input_ids = mx.array(encoded["input_ids"])
                attention_mask = mx.array(encoded["attention_mask"])
            else:
                encoded_ids = []
                masks = []
                for text in input_texts:
                    enc = processor.encode(text, add_special_tokens=True)
                    ids = list(enc.ids)[:max_length]
                    encoded_ids.append(ids)
                max_len = max(len(ids) for ids in encoded_ids)
                padded = []
                for ids in encoded_ids:
                    pad_len = max_len - len(ids)
                    padded.append(ids + [0] * pad_len)
                    masks.append([1] * len(ids) + [0] * pad_len)
                input_ids = mx.array(padded)
                attention_mask = mx.array(masks)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings_array = self._extract_embeddings_array(outputs)
            total_tokens = self._count_prepared_tokens(
                {"attention_mask": attention_mask, "input_ids": input_ids}
            )
        else:
            if self._is_compiled and self._compiled_embed is not None:
                try:
                    inputs = self._prepare_embedding_inputs(
                        processor,
                        normalized_inputs,
                        max_length,
                        padding,
                        truncation,
                    )
                    if not isinstance(inputs, dict):
                        inputs = dict(inputs)
                    total_tokens = self._count_prepared_tokens(inputs)
                    embeddings_array = self._compiled_embed(inputs)
                except Exception as e:
                    logger.warning(
                        f"compiled embedding path failed for {self.model_name}: {e}; "
                        "disabling compile and falling back to eager generate()"
                    )
                    self._is_compiled = False
                    self._compiled_embed = None
                    total_tokens = None

            if embeddings_array is None:
                if uses_custom_embedding_inputs:
                    inputs = self._prepare_embedding_inputs(
                        processor,
                        normalized_inputs,
                        max_length,
                        padding,
                        truncation,
                    )
                    if not isinstance(inputs, dict):
                        inputs = dict(inputs)
                    outputs = self.model(**self._adapt_model_inputs_for_call(inputs))
                    total_tokens = self._count_prepared_tokens(inputs)
                else:
                    from mlx_embeddings import generate

                    outputs = generate(
                        self.model,
                        processor,
                        input_texts,
                        max_length=max_length,
                        padding=padding,
                        truncation=truncation,
                    )
                embeddings_array = self._extract_embeddings_array(outputs)

        mx.eval(embeddings_array)
        embeddings = embeddings_array.tolist()
        if total_tokens is None:
            total_tokens = self._count_tokens(normalized_inputs)
        dimensions = len(embeddings[0]) if embeddings else 0

        return EmbeddingOutput(
            embeddings=embeddings,
            total_tokens=total_tokens,
            dimensions=dimensions,
        )

    def _count_tokens(
        self, inputs: Union[List[str], List[Dict[str, str]]]
    ) -> int:
        """Count total tokens in input texts."""
        total = 0
        processor = self.processor

        for item in self._normalize_embedding_inputs(inputs):
            text = item.get("text")
            if not text:
                continue
            if hasattr(processor, "encode"):
                tokens = processor.encode(text, add_special_tokens=True)
                if isinstance(tokens, list):
                    total += len(tokens)
                elif hasattr(tokens, "shape"):
                    total += tokens.shape[-1] if tokens.ndim > 0 else 1
                elif hasattr(tokens, "ids"):
                    total += len(tokens.ids)
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

    def _count_prepared_tokens(self, prepared_inputs: Dict[str, Any]) -> int:
        """Count tokens from prepared model inputs, including multimodal tokens."""
        attention_mask = prepared_inputs.get("attention_mask")
        if attention_mask is not None:
            try:
                return int(mx.sum(attention_mask).item())
            except (TypeError, ValueError):
                pass
            if isinstance(attention_mask, list):
                return int(sum(sum(row) if isinstance(row, list) else row for row in attention_mask))
            if hasattr(attention_mask, "tolist"):
                values = attention_mask.tolist()
                if values and isinstance(values[0], list):
                    return int(sum(sum(row) for row in values))
                return int(sum(values))

        input_ids = prepared_inputs.get("input_ids")
        if input_ids is None:
            return 0
        if hasattr(input_ids, "shape"):
            if len(input_ids.shape) == 0:
                return 1
            if len(input_ids.shape) == 1:
                return int(input_ids.shape[0])
            return int(input_ids.shape[0] * input_ids.shape[1])
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return int(sum(len(row) for row in input_ids))
            return int(len(input_ids))
        return 0

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
            "compiled": self._is_compiled,
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
