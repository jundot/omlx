# SPDX-License-Identifier: Apache-2.0
"""
MLX Embedding Model wrapper.

This module provides a wrapper around mlx-embeddings for generating
text embeddings using Apple's MLX framework, with native fallback
for XLMRoBERTa, BERT, and Qwen2-decoder embedding models.
"""

import gc
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
from mlx.utils import tree_flatten

from ..utils.image import validate_image_data_uri
from .base_model import last_token_pool, mean_pooling, normalize_embeddings
from .mlx_embeddings_compat import (
    patch_qwen3_vl_processor_for_torch_free_image_loading,
)

logger = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_MAX_LENGTH = 512
_TOKENIZER_MAX_LENGTH_SENTINEL = 10**18
_CONTEXT_LENGTH_ATTRS = (
    "max_position_embeddings",
    "max_seq_len",
    "max_seq_length",
    "seq_length",
    "n_positions",
)
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}

# Pooling modes whose result depends on the attention mask. CLS reads a fixed
# position, so it is mask-independent.
_MASK_AWARE_POOLING_MODES = ("mean", "lasttoken")


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
    - Native Qwen2-decoder embedding (last-token + L2; jina-code, gte-Qwen2)
      — mlx-embeddings has no qwen2 module
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
        self._pooling_mode: Optional[str] = None
        self._pooling_source: str = "not resolved"

    # Fallbacks for MLX conversions that dropped the sentence-transformers
    # metadata. Reviewed against the concrete checkpoints on the Hub: none of
    # the MLX BGE-M3 / Qwen3-Embedding / Qwen3-VL-Embedding repos ship
    # ``1_Pooling/config.json``, so detection alone never fires for exactly the
    # models this fix is meant to serve.
    _FAMILY_POOLING = (
        ("qwen3-vl-embedding", "lasttoken"),
        ("qwen3-embedding", "lasttoken"),
        ("bge-m3", "cls"),
    )

    def _pooling_config_path(self, model_path: Path) -> Optional[Path]:
        """Locate the Pooling module through ``modules.json``.

        sentence-transformers describes its pipeline there; the Pooling module
        carries its own directory in ``path`` (usually ``1_Pooling``, but not
        always). Hardcoding ``1_Pooling`` misses non-standard exports, and some
        MLX repos keep the ``modules.json`` entry while dropping the directory
        itself -- so the path is also checked for existence.
        """
        modules_path = model_path / "modules.json"
        if modules_path.is_file():
            try:
                with open(modules_path) as fh:
                    modules = json.load(fh)
                for module in modules if isinstance(modules, list) else []:
                    if "Pooling" in str(module.get("type", "")):
                        candidate = model_path / str(module.get("path", ""))
                        cfg = candidate / "config.json"
                        if cfg.is_file():
                            return cfg
            except (OSError, ValueError) as e:
                logger.debug(f"Could not read modules.json for {self.model_name}: {e}")
        # Fall back to the conventional location for exports without modules.json.
        conventional = model_path / "1_Pooling" / "config.json"
        return conventional if conventional.is_file() else None

    @staticmethod
    def _pooling_mode_from_config(cfg: Dict[str, Any]) -> Optional[str]:
        """Read either config dialect.

        Recent sentence-transformers releases write a single
        ``{"pooling_mode": "lasttoken"}`` string; older ones set one boolean per
        mode. Qwen3-VL-Embedding uses the newer form, so a boolean-only parser
        silently ignores a pooling config that is actually present.
        """
        declared = cfg.get("pooling_mode")
        if isinstance(declared, str):
            alias = {
                "cls": "cls",
                "mean": "mean",
                "lasttoken": "lasttoken",
                "last_token": "lasttoken",
            }
            mode = alias.get(declared.strip().lower())
            if mode:
                return mode
        for key, mode in (
            ("pooling_mode_cls_token", "cls"),
            ("pooling_mode_lasttoken", "lasttoken"),
            ("pooling_mode_mean_tokens", "mean"),
        ):
            if cfg.get(key):
                return mode
        return None

    def _resolve_pooling_mode(self) -> Tuple[Optional[str], str]:
        """Resolve the pooling mode and report where it came from.

        Returns ``(mode, source)``. ``source`` is logged at load time so an
        operator can tell a declared mode from an inferred one without reading
        the code -- silent pooling changes are exactly what makes retrieval
        quality drift unnoticed.
        """
        model_path = Path(self.model_name)
        cfg_path = self._pooling_config_path(model_path)
        if cfg_path is not None:
            try:
                with open(cfg_path) as fh:
                    cfg = json.load(fh)
                mode = self._pooling_mode_from_config(cfg)
                if mode:
                    return mode, str(cfg_path.relative_to(model_path))
            except (OSError, ValueError) as e:
                logger.debug(f"Could not read pooling config for {self.model_name}: {e}")

        haystack = str(self.model_name).lower()
        config_path = model_path / "config.json"
        if config_path.is_file():
            try:
                with open(config_path) as fh:
                    haystack += " " + str(json.load(fh).get("_name_or_path", "")).lower()
            except (OSError, ValueError):
                pass
        for needle, mode in self._FAMILY_POOLING:
            if needle in haystack:
                return mode, f"known family ({needle})"
        return None, "not declared"

    def _load_native(self) -> bool:
        """
        Try to load using native omlx implementations (xlm_roberta, bert).

        Returns True if native loading succeeded, False otherwise.
        """
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

        native_arch_modules = {
            "XLMRobertaModel": "xlm_roberta",
            "BertModel": "xlm_roberta",
            "BertForMaskedLM": "xlm_roberta",
            "Qwen2ForCausalLM": "qwen2_embedding",
        }
        module_name = native_arch_modules.get(arch)
        if module_name is None:
            logger.debug(
                f"Architecture '{arch}' not natively supported for embedding, "
                "trying mlx-embeddings"
            )
            return False

        try:
            from importlib import import_module

            native_module = import_module(f"{__package__}.{module_name}")
            Model = native_module.Model
            ModelArgs = native_module.ModelArgs

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
                weights.update(mx.load(str(wf)))

            weights = model_instance.sanitize(weights)
            self._validate_native_weights(model_instance, weights)
            model_instance.load_weights(list(weights.items()), strict=False)
            mx.eval(model_instance.parameters())
            # Embedding inference must be deterministic: put the model in eval
            # mode so dropout (p>0 in XLM-RoBERTa/BERT) is disabled. Without this
            # every /v1/embeddings call applies random dropout, producing
            # non-deterministic, corrupted vectors.
            model_instance.train(False)

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

        self._pooling_mode, self._pooling_source = self._resolve_pooling_mode()
        if self._pooling_mode:
            logger.info(
                f"Embedding pooling for {self.model_name}: "
                f"{self._pooling_mode} (source: {self._pooling_source})"
            )

        # 1. Try native loading first (xlm_roberta, bert)
        if self._load_native():
            return

        # 2. Fallback to mlx-embeddings
        try:
            patch_qwen3_vl_processor_for_torch_free_image_loading()
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

    def _extract_embeddings_array(self, outputs, attention_mask=None):
        """Extract embedding tensor from model outputs as a 2D (batch, hidden) array."""
        # Honour the checkpoint's own declaration first. Some MLX conversions of
        # sentence-transformers models expose a mean-pooled ``text_embeds`` while
        # the checkpoint was trained for CLS pooling (e.g. BGE-M3), which silently
        # degrades retrieval instead of failing. Mode is resolved once at load().
        #
        # Pooling goes through the existing mask-aware helpers: a bare
        # ``[:, -1]`` is only correct under left padding and an unmasked
        # ``mean(axis=1)`` averages pad tokens in, both of which corrupt vectors
        # in mixed-length batches while single inputs still look fine. The
        # result is L2-normalized like every other path out of this method.
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if self._pooling_mode and last_hidden is not None and last_hidden.ndim == 3:
            if self._pooling_mode == "cls":
                return normalize_embeddings(last_hidden[:, 0, :])
            if self._pooling_mode == "lasttoken":
                return normalize_embeddings(
                    last_token_pool(last_hidden, attention_mask)
                )
            if self._pooling_mode == "mean" and attention_mask is not None:
                return normalize_embeddings(
                    mean_pooling(last_hidden, attention_mask)
                )

        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            embeddings = outputs.text_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        elif (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            embeddings = outputs.last_hidden_state
        else:
            raise ValueError(
                "Model output does not contain expected embedding fields "
                "(text_embeds, pooler_output, or last_hidden_state)"
            )

        # Some models (e.g. ModernBERT loaded as MaskedLM) skip pooling and return
        # per-token features with shape (batch, seq_len, hidden). Mean pool so
        # /v1/embeddings always emits one vector per input.
        if embeddings.ndim == 3:
            embeddings = mx.mean(embeddings, axis=1)
        return embeddings

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

    @staticmethod
    def _positive_context_length(value: Any) -> Optional[int]:
        """Return a usable positive context length from config/tokenizer metadata."""
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if 0 < value < _TOKENIZER_MAX_LENGTH_SENTINEL:
            return value
        return None

    @classmethod
    def _get_config_value(cls, config: Any, key: str) -> Optional[int]:
        if config is None:
            return None
        if isinstance(config, dict):
            return cls._positive_context_length(config.get(key))
        return cls._positive_context_length(getattr(config, key, None))

    @classmethod
    def _context_length_from_config(cls, config: Any) -> Optional[int]:
        """Read context length from model config objects or dictionaries."""
        for key in _CONTEXT_LENGTH_ATTRS:
            value = cls._get_config_value(config, key)
            if value is not None:
                return value

        for nested_key in ("text_config", "language_config"):
            nested = None
            if isinstance(config, dict):
                nested = config.get(nested_key)
            elif config is not None:
                nested = getattr(config, nested_key, None)
            for key in _CONTEXT_LENGTH_ATTRS:
                value = cls._get_config_value(nested, key)
                if value is not None:
                    return value

        return None

    def _resolve_max_length(self, max_length: int | None) -> int:
        """Resolve embedding token length when callers omit an explicit limit."""
        if max_length is not None:
            value = self._positive_context_length(max_length)
            if value is None:
                raise ValueError("max_length must be a positive integer")
            return value

        for config in (
            getattr(self.model, "config", None),
            getattr(self.processor, "config", None),
        ):
            value = self._context_length_from_config(config)
            if value is not None:
                return value

        processor = self.processor
        tokenizers = [
            processor,
            getattr(processor, "tokenizer", None),
            getattr(processor, "_tokenizer", None),
        ]
        for tokenizer in tokenizers:
            for attr_name in ("model_max_length", "max_length"):
                value = self._positive_context_length(
                    getattr(tokenizer, attr_name, None)
                )
                if value is not None:
                    return value

        return _DEFAULT_EMBEDDING_MAX_LENGTH

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

    def _eager_forward_with_mask(
        self,
        processor,
        normalized_inputs,
        input_texts,
        max_length: int,
        padding: bool,
        truncation: bool,
    ):
        """Eager forward for tokenizer-style processors, keeping the mask.

        ``mlx_embeddings.generate`` is ``prepare_inputs`` + ``model(**inputs)``
        and returns only the outputs, dropping the mask it just built. When the
        checkpoint declares a mask-aware pooling mode, run those same two steps
        here so the mask survives into pooling; the compiled path already has
        it. Without this, a right-padded batch pools a pad token whenever
        compile is off or has fallen back. Any preparation failure returns to
        ``generate()`` unchanged, so this can only add a mask, never remove a
        working path.
        """
        if self._pooling_mode in _MASK_AWARE_POOLING_MODES:
            inputs = None
            try:
                prepared = self._prepare_embedding_inputs(
                    processor, normalized_inputs, max_length, padding, truncation
                )
                inputs = dict(prepared)
            except Exception as e:
                # Only input preparation is guarded: a forward error must
                # surface as it does on every other path, not silently rerun.
                logger.warning(
                    "masked eager forward unavailable for %s (%s); falling back "
                    "to generate() without a mask",
                    self.model_name,
                    e,
                )
            if inputs is not None:
                outputs = self.model(**self._adapt_model_inputs_for_call(inputs))
                return outputs, inputs.get("attention_mask")

        from mlx_embeddings import generate

        outputs = generate(
            self.model,
            processor,
            input_texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
        )
        return outputs, None

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
        compile_env = os.getenv("OMLX_EMBEDDING_COMPILE", "1").strip().lower()
        if compile_env in _FALSE_ENV_VALUES:
            logger.info(
                "mx.compile disabled for %s by OMLX_EMBEDDING_COMPILE",
                self.model_name,
            )
            self._compiled_embed = None
            return False

        base_model = self.model

        try:
            def _compiled_embed(inputs):
                outputs = base_model(**self._adapt_model_inputs_for_call(inputs))
                # The mask travels with the request; mask-aware pooling needs it.
                return self._extract_embeddings_array(
                    outputs, inputs.get("attention_mask")
                )

            self._compiled_embed = mx.compile(_compiled_embed)

            # Real requests arrive with a traced attention_mask (the tokenizer
            # path always emits one). A mask-less probe lets a model whose
            # forward branches on the mask pass here — its internally-built
            # default mask is a tracing constant — and then the first real
            # request trips the traced-mask branch and silently disables
            # compile for the rest of the process (issue #2447). The reranker
            # probe already includes the mask.
            test_inputs = {
                "input_ids": mx.zeros((1, 4), dtype=mx.int32),
                "attention_mask": mx.ones((1, 4), dtype=mx.int32),
            }
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

    def close(self) -> None:
        """Release model, processor, and compiled embedding resources."""
        if not self._loaded and self.model is None and self.processor is None:
            self._compiled_embed = None
            self._is_compiled = False
            return

        logger.info(
            "Releasing embedding model resources: %s "
            "(compiled=%s, native=%s)",
            self.model_name,
            self._is_compiled,
            self._using_native,
        )

        self._compiled_embed = None
        self._is_compiled = False

        self.model = None
        self.processor = None
        self._hidden_size = None
        self._loaded = False
        self._using_native = False
        self._remap_input_ids_to_inputs = False

        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        gc.collect()

    def embed(
        self,
        inputs: Union[str, List[str], List[Dict[str, str]]],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> EmbeddingOutput:
        """
        Generate embeddings for input texts.

        Args:
            texts: List of input texts
            max_length: Maximum token length for each text. If omitted, use
                model/tokenizer metadata when available.
            padding: Whether to pad shorter sequences
            truncation: Whether to truncate longer sequences

        Returns:
            EmbeddingOutput with embeddings and token count
        """
        if not self._loaded:
            self.load()

        max_length = self._resolve_max_length(max_length)
        normalized_inputs = self._normalize_embedding_inputs(inputs)
        for item in normalized_inputs:
            image_ref = item.get("image")
            if isinstance(image_ref, str):
                item["image"] = validate_image_data_uri(
                    image_ref,
                    field="items[].image",
                )
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
                    ids = list(enc.ids)
                    if truncation:
                        ids = ids[:max_length]
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
            embeddings_array = self._extract_embeddings_array(
                outputs, attention_mask
            )
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
                # Both eager paths must carry the prepared mask the compiled
                # path passes: pooling is only correct with it.
                eager_mask = None
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
                    eager_mask = inputs.get("attention_mask")
                else:
                    outputs, eager_mask = self._eager_forward_with_mask(
                        processor,
                        normalized_inputs,
                        input_texts,
                        max_length,
                        padding,
                        truncation,
                    )
                embeddings_array = self._extract_embeddings_array(outputs, eager_mask)

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
