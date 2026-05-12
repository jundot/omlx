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
from typing import Any, Dict, Tuple

import mlx.core as mx

from ..model_discovery import (
    CAUSAL_LM_RERANKER_ARCHITECTURES,
    MULTIMODAL_RERANKER_ARCHITECTURES,
    SUPPORTED_RERANKER_ARCHITECTURES,
    _is_causal_lm_reranker,
)
from ..utils.image import load_image
from .mlx_embeddings_compat import (
    patch_qwen3_vl_processor_for_torch_free_image_loading,
)

logger = logging.getLogger(__name__)


def _coerce_item_to_text(item: Any) -> str:
    """Reduce a rerank input (str or dict with 'text') to plain text.

    Used by text-only reranker paths so dict inputs stay compatible with
    callers that previously passed bare strings.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text", "") or ""
    return str(item)


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

    def __init__(self, model_name: str, trust_remote_code: bool = False):
        """
        Initialize the MLX reranker model.

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
        self._num_labels: int | None = None
        self._is_causal_lm = False
        self._is_jina_reranker = False
        self._is_vl_reranker = False
        self._token_true_id: int | None = None
        self._token_false_id: int | None = None
        self._doc_embed_token_id: int | None = None
        self._query_embed_token_id: int | None = None
        self._jina_projector = None
        self._prefix_tokens: list[int] | None = None
        self._suffix_tokens: list[int] | None = None
        self._is_compiled = False
        self._compiled_seq_logits = None

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
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=self.trust_remote_code
        )

        return model, tokenizer

    def _load_vl_reranker(self) -> Tuple[Any, Any]:
        """Load a multimodal reranker (e.g., Qwen3-VL-Reranker) via mlx-embeddings.

        mlx-embeddings exposes a unified `load()` + `model.process()` API that
        handles both embedding and reranking variants of Qwen3-VL. Reranker vs
        embedder is decided by the input dict shape at inference time.
        """
        patch_qwen3_vl_processor_for_torch_free_image_loading()
        from mlx_embeddings import load as mlx_emb_load

        return mlx_emb_load(
            str(self.model_name),
            tokenizer_config={"trust_remote_code": self.trust_remote_code},
        )

    def _build_vl_item(
        self, item: "str | dict[str, Any]"
    ) -> Dict[str, Any]:
        """Normalize a rerank input into the mlx-embeddings VL item format.

        Accepts either a bare string (text) or a dict with 'text' and/or
        'image' keys. Image values are strings (URL / base64 data URI / local
        path) and get loaded via omlx's shared image loader.
        """
        if isinstance(item, str):
            return {"text": item}
        if not isinstance(item, dict):
            return {"text": str(item)}

        result: Dict[str, Any] = {}
        text = item.get("text")
        if text:
            result["text"] = text
        image_ref = item.get("image")
        if image_ref:
            if isinstance(image_ref, str):
                result["image"] = load_image(image_ref)
            else:
                # Already a PIL image or similar — pass through
                result["image"] = image_ref
        if not result:
            raise ValueError(
                "VL reranker item must have at least 'text' or 'image'."
            )
        return result

    def _rerank_vl(
        self,
        query: "str | dict[str, Any]",
        documents: "list[str] | list[dict[str, Any]]",
        max_length: int,
    ) -> RerankOutput:
        """Rerank using mlx-embeddings' multimodal model.process() API."""
        query_item = self._build_vl_item(query)
        doc_items = [self._build_vl_item(d) for d in documents]

        inputs = {
            "instruction": self._CAUSAL_LM_DEFAULT_INSTRUCTION,
            "query": query_item,
            "documents": doc_items,
        }

        scores = self.model.process(inputs, processor=self.processor)
        mx.eval(scores)
        scores_list = [float(s) for s in scores.tolist()]
        indices = sorted(
            range(len(scores_list)),
            key=lambda i: scores_list[i],
            reverse=True,
        )

        return RerankOutput(
            scores=scores_list,
            indices=indices,
            total_tokens=0,
        )

    def _load_causal_lm(self) -> Tuple[Any, Any]:
        """Load a CausalLM-based reranker model using mlx-lm."""
        from mlx_lm import load as mlx_lm_load

        from ..utils.model_loading import maybe_load_custom_quantization

        model_path = str(self.model_name)
        tokenizer_config = {"trust_remote_code": self.trust_remote_code}
        custom_loaded = maybe_load_custom_quantization(
            model_path,
            is_vlm=False,
        )
        if custom_loaded is not None:
            model, tokenizer_wrapper = custom_loaded
        else:
            loaded = mlx_lm_load(
                model_path,
                tokenizer_config=tokenizer_config,
            )
            model = loaded[0]
            tokenizer_wrapper = loaded[1]

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

    def _load_jina_reranker(self) -> Tuple[Any, Any]:
        """
        Load a Jina v3 reranker model using mlx-lm.

        Jina v3 reranker uses special-token hidden states + projector + cosine
        similarity for listwise scoring.
        """
        from mlx_lm import load as mlx_lm_load

        from ..utils.model_loading import maybe_load_custom_quantization

        model_path = str(self.model_name)
        tokenizer_config = {"trust_remote_code": self.trust_remote_code}
        custom_loaded = maybe_load_custom_quantization(
            model_path,
            is_vlm=False,
        )
        if custom_loaded is not None:
            model, tokenizer_wrapper = custom_loaded
        else:
            loaded = mlx_lm_load(
                model_path,
                tokenizer_config=tokenizer_config,
            )
            model = loaded[0]
            tokenizer_wrapper = loaded[1]

        # mlx-lm returns a TokenizerWrapper; unwrap to get the underlying
        # transformers tokenizer which supports __call__ for batch encoding.
        tokenizer = getattr(tokenizer_wrapper, "_tokenizer", tokenizer_wrapper)

        doc_embed_token_id = self._resolve_token_id(tokenizer, "<|embed_token|>")
        query_embed_token_id = self._resolve_token_id(tokenizer, "<|rerank_token|>")

        if doc_embed_token_id is None or query_embed_token_id is None:
            raise ValueError(
                "Could not resolve required Jina special tokens "
                "('<|embed_token|>', '<|rerank_token|>'). "
                "This model may not be a compatible Jina v3 reranker."
            )

        self._doc_embed_token_id = doc_embed_token_id
        self._query_embed_token_id = query_embed_token_id
        self._jina_projector = self._load_jina_projector(self.model_name)

        logger.info(
            f"Jina reranker tokens: embed_token={doc_embed_token_id}, "
            f"rerank_token={query_embed_token_id}"
        )

        return model, tokenizer

    def _resolve_token_id(self, tokenizer: Any, token_text: str) -> int | None:
        """Resolve token IDs across tokenizer implementations."""
        added_tokens = getattr(tokenizer, "added_tokens_decoder", {}) or {}
        for tid, tinfo in added_tokens.items():
            content = ""
            if isinstance(tinfo, str):
                content = tinfo
            elif hasattr(tinfo, "content"):
                content = tinfo.content
            elif isinstance(tinfo, dict):
                content = tinfo.get("content", "")

            if content == token_text:
                return int(tid)

        convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_tokens_to_ids):
            try:
                token_id = convert_tokens_to_ids(token_text)
            except Exception:
                token_id = None

            if isinstance(token_id, int) and token_id >= 0:
                unk_token_id = getattr(tokenizer, "unk_token_id", None)
                if unk_token_id is None or token_id != unk_token_id:
                    return token_id

        get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
        if callable(get_added_vocab):
            try:
                added_vocab = get_added_vocab() or {}
            except Exception:
                added_vocab = {}

            token_id = added_vocab.get(token_text)
            if isinstance(token_id, int):
                return token_id

        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            try:
                vocab = get_vocab() or {}
            except Exception:
                vocab = {}

            token_id = vocab.get(token_text)
            if isinstance(token_id, int):
                return token_id

        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                encoded = encode(token_text, add_special_tokens=False)
            except TypeError:
                encoded = encode(token_text)
            except Exception:
                encoded = None

            if hasattr(encoded, "ids"):
                encoded = encoded.ids

            if (
                isinstance(encoded, list)
                and len(encoded) == 1
                and isinstance(encoded[0], int)
            ):
                return encoded[0]

        return None

    def _load_jina_projector(self, model_dir: str | Path):
        """Load Jina projector weights and return a projection callable."""
        model_path = Path(model_dir)
        projector_path = model_path / "projector.safetensors"
        if not projector_path.exists():
            raise FileNotFoundError(
                f"Missing Jina projector file: {projector_path}. "
                "Expected projector.safetensors for JinaForRanking models."
            )

        from safetensors import safe_open

        weights = {}
        with safe_open(projector_path, framework="mlx") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

        required_keys = ("linear1.weight", "linear2.weight")
        missing_keys = [key for key in required_keys if key not in weights]
        if missing_keys:
            raise ValueError(
                f"Jina projector is malformed: missing keys {missing_keys} in "
                f"{projector_path}. "
                f"Available keys: {sorted(weights.keys())}"
            )

        linear1_weight = weights["linear1.weight"]
        linear2_weight = weights["linear2.weight"]

        if len(linear1_weight.shape) != 2 or len(linear2_weight.shape) != 2:
            raise ValueError(
                "Jina projector weights must be 2D matrices: "
                f"linear1.weight={linear1_weight.shape}, "
                f"linear2.weight={linear2_weight.shape}."
            )

        if linear1_weight.shape != (512, 1024) or linear2_weight.shape != (512, 512):
            raise ValueError(
                "Unexpected Jina projector shapes. Expected "
                "linear1.weight=(512, 1024) and linear2.weight=(512, 512), "
                f"got linear1.weight={linear1_weight.shape}, "
                f"linear2.weight={linear2_weight.shape}."
            )

        def _project(x):
            if x.shape[-1] != linear1_weight.shape[1]:
                raise ValueError(
                    "Jina projector input dim mismatch for linear1: "
                    f"input={x.shape[-1]}, expected={linear1_weight.shape[1]}."
                )
            hidden = x @ mx.transpose(linear1_weight)
            hidden = mx.maximum(hidden, 0)
            return hidden @ mx.transpose(linear2_weight)

        return _project

    def _sanitize_jina_text(self, text: str) -> str:
        """Strip conflicting special tokens from user-provided text."""
        sanitized = str(text)
        sanitized = sanitized.replace("<|embed_token|>", " ")
        sanitized = sanitized.replace("<|rerank_token|>", " ")
        sanitized = sanitized.replace("<|score_token|>", " ")
        sanitized = sanitized.replace("<|im_start|>", " ")
        sanitized = sanitized.replace("<|im_end|>", " ")
        return sanitized.strip()

    def _format_jina_prompt(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
    ) -> str:
        """Format a listwise Jina reranking prompt."""
        sanitized_query = self._sanitize_jina_text(query)
        sanitized_docs = [self._sanitize_jina_text(doc) for doc in documents]
        sanitized_instruction = (
            self._sanitize_jina_text(instruction) if instruction is not None else None
        )

        user_content = (
            f"I will provide you with {len(sanitized_docs)} passages, each indicated "
            f"by a numerical identifier. Rank the passages based on their relevance "
            f"to query: {sanitized_query}\n"
        )
        if sanitized_instruction:
            user_content += f"<instruct>\n{sanitized_instruction}\n</instruct>\n"

        doc_prompts = [
            f'<passage id="{idx}">\n{doc}<|embed_token|>\n</passage>'
            for idx, doc in enumerate(sanitized_docs)
        ]
        user_content += "\n".join(doc_prompts) + "\n"
        user_content += f"<query>\n{sanitized_query}<|rerank_token|>\n</query>"

        system_prompt = (
            "You are a search relevance expert who can determine a ranking of the "
            "passages based on how relevant they are to the query. If the query is "
            "a question, how relevant a passage is depends on how well it answers "
            "the question. If not, try to analyze the intent of the query and "
            "assess how well each passage satisfies the intent. If an instruction "
            "is provided, you should follow the instruction when determining the "
            "ranking."
        )

        return (
            "<|im_start|>system\n"
            f"{system_prompt}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_content}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

    def _get_jina_hidden_states(self, input_ids):
        """Extract final hidden states from the Jina mlx-lm backbone."""

        backbone = getattr(self.model, "model", None)
        if backbone is None or not callable(backbone):
            model_type = type(self.model).__name__ if self.model is not None else "None"
            raise ValueError(
                "Could not find Jina model backbone (model.model). "
                f"The mlx-lm model wrapper may have changed: {model_type}."
            )

        hidden_states = backbone(input_ids)

        if not hasattr(hidden_states, "shape"):
            raise ValueError("Jina backbone did not return hidden states as a tensor.")

        if len(hidden_states.shape) == 2:
            return mx.expand_dims(hidden_states, axis=0)

        if len(hidden_states.shape) != 3:
            raise ValueError(
                "Jina hidden states must be rank 2 or 3. "
                f"Got shape: {hidden_states.shape}"
            )

        return hidden_states

    def _cosine_similarity(self, query_vec, doc_vecs, eps: float = 1e-8):
        """Compute cosine similarity between one query vector and many docs."""
        if len(query_vec.shape) == 2:
            query_vec = query_vec[0]
        if len(doc_vecs.shape) == 1:
            doc_vecs = mx.expand_dims(doc_vecs, axis=0)

        query_norm = mx.linalg.norm(query_vec)
        doc_norms = mx.linalg.norm(doc_vecs, axis=-1)
        denom = mx.maximum(doc_norms * query_norm, eps)
        numer = mx.sum(doc_vecs * query_vec, axis=-1)
        return numer / denom

    def load(self) -> None:
        """Load the model and processor/tokenizer."""
        if self._loaded:
            return

        # Check architecture before loading
        self._validate_architecture()

        arch = self._get_architecture()
        logger.info(f"Loading reranker model: {self.model_name} (arch={arch})")

        try:
            if arch in MULTIMODAL_RERANKER_ARCHITECTURES:
                # Multimodal reranker (e.g., Qwen3-VL-Reranker) via mlx-embeddings
                self.model, self.processor = self._load_vl_reranker()
                self._is_vl_reranker = True
                self._num_labels = 1
            elif arch == "JinaForRanking":
                # Jina v3 reranker: listwise hidden-state scoring + projector
                self.model, self.processor = self._load_jina_reranker()
                self._is_jina_reranker = True
                self._num_labels = 1
            elif arch in CAUSAL_LM_RERANKER_ARCHITECTURES:
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
                patch_qwen3_vl_processor_for_torch_free_image_loading()
                from mlx_embeddings import load

                self.model, self.processor = load(
                    self.model_name,
                    tokenizer_config={"trust_remote_code": self.trust_remote_code},
                )

                # Get num_labels from model config
                if hasattr(self.model, "config"):
                    config = self.model.config
                    self._num_labels = getattr(config, "num_labels", None)

            # Try mx.compile for persistent Metal kernel caching
            self._is_compiled = self._try_compile()

            self._loaded = True
            logger.info(
                f"Reranker model loaded successfully: {self.model_name} "
                f"(arch={arch}, num_labels={self._num_labels}, "
                f"causal_lm={self._is_causal_lm}, vl={self._is_vl_reranker}, "
                f"compiled={self._is_compiled})"
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

    def _try_compile(self) -> bool:
        """Compile reranker scoring path to return primitive logits arrays.

        Root-cause fix:
        - Compiling model.__call__ directly can yield arrays without primitives
          in some MLX output containers.
        - Compile a narrow function that returns logits only.
        """
        if self._is_causal_lm or self._is_vl_reranker:
            # CausalLM / VL reranker paths use custom scoring (yes/no logits or
            # mlx-embeddings model.process). VL forward needs pixel_values and
            # lacks pooler_output, so the compile wrapper here wouldn't apply.
            logger.info(
                f"mx.compile skipped for {self.model_name}"
            )
            self._compiled_seq_logits = None
            return False

        base_model = self.model
        if not callable(base_model):
            return False
        try:

            def _compiled_seq_logits(inputs):
                outputs = base_model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    return outputs.pooler_output
                raise ValueError(
                    "Model output does not contain pooler_output. "
                    "Ensure the model is a SequenceClassification model."
                )

            # NOTE: use default compile mode. shapeless=True can fail shape
            # inference for some linear ops in embedding/reranker stacks.
            self._compiled_seq_logits = mx.compile(_compiled_seq_logits)

            # Warmup: verify compilation actually works with a dummy forward pass
            test_inputs = {
                "input_ids": mx.zeros((1, 4), dtype=mx.int32),
                "attention_mask": mx.ones((1, 4), dtype=mx.int32),
            }
            _ = self._compiled_seq_logits(test_inputs)

            logger.info(
                f"mx.compile enabled for {self.model_name} "
                f"(primitive reranker logits path)"
            )
            return True
        except Exception as e:
            logger.info(
                f"mx.compile unavailable for {self.model_name}: {e}"
            )
            self._compiled_seq_logits = None
            return False

    # Default max_length per model type
    _DEFAULT_MAX_LENGTH_SEQ_CLASSIFICATION = 512
    _DEFAULT_MAX_LENGTH_CAUSAL_LM = 8192

    def rerank(
        self,
        query: "str | dict",
        documents: "list[str] | list[dict]",
        max_length: int | None = None,
    ) -> RerankOutput:
        """
        Rerank documents by relevance to the query.

        Args:
            query: The search query. String for text-only rerankers. Dict with
                'text' and/or 'image' for multimodal rerankers.
            documents: List of documents to rerank. Each item can be a string
                or a dict with 'text' and/or 'image' keys.
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

        if self._is_vl_reranker:
            effective_max_length = (
                max_length
                if max_length is not None
                else self._DEFAULT_MAX_LENGTH_CAUSAL_LM
            )
            return self._rerank_vl(query, documents, effective_max_length)

        # Text-only paths: coerce dict inputs down to text so existing
        # _rerank_* methods keep their str-only contract.
        query_str = _coerce_item_to_text(query)
        docs_str = [_coerce_item_to_text(d) for d in documents]

        if self._is_jina_reranker:
            effective_max_length = (
                max_length
                if max_length is not None
                else self._DEFAULT_MAX_LENGTH_CAUSAL_LM
            )
            return self._rerank_jina(query_str, docs_str, effective_max_length)
        elif self._is_causal_lm:
            effective_max_length = (
                max_length
                if max_length is not None
                else self._DEFAULT_MAX_LENGTH_CAUSAL_LM
            )
            return self._rerank_causal_lm(query_str, docs_str, effective_max_length)
        else:
            effective_max_length = (
                max_length
                if max_length is not None
                else self._DEFAULT_MAX_LENGTH_SEQ_CLASSIFICATION
            )
            return self._rerank_seq_classification(
                query_str, docs_str, effective_max_length
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
        if not callable(tokenizer):
            raise ValueError("CausalLM reranker tokenizer is not initialized.")
        if prefix_tokens is None or suffix_tokens is None:
            raise ValueError("CausalLM reranker prompt tokens are not initialized.")
        if not callable(self.model):
            raise ValueError("CausalLM reranker model is not initialized.")

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

    def _rerank_jina(
        self,
        query: str,
        documents: list[str],
        max_length: int = 8192,
    ) -> RerankOutput:
        """
        Rerank using Jina v3 listwise embedding-based scoring.

        Builds multi-document prompts, extracts hidden states at special token
        positions, applies the projector, and computes query-document cosine
        similarities. Uses deterministic greedy chunking under max_length.
        """
        tokenizer = self.processor
        doc_embed_token_id = self._doc_embed_token_id
        query_embed_token_id = self._query_embed_token_id
        projector = self._jina_projector
        if tokenizer is None:
            raise ValueError("Jina reranker tokenizer is not initialized.")

        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise ValueError("Jina reranker tokenizer does not provide encode().")

        if (
            doc_embed_token_id is None
            or query_embed_token_id is None
            or projector is None
        ):
            raise ValueError(
                "Jina reranker is not fully initialized. "
                "Missing special-token IDs or projector."
            )

        def _to_token_ids(text: str) -> list[int]:
            encoded = encode(text, add_special_tokens=False)
            if hasattr(encoded, "ids"):
                return list(encoded.ids)
            return list(encoded)

        decode = getattr(tokenizer, "decode", None)

        def _truncate_doc_to_fit(
            query_text: str, doc_text: str
        ) -> Tuple[str, list[int]]:
            doc_token_ids = _to_token_ids(doc_text)
            if not doc_token_ids:
                prompt = self._format_jina_prompt(query_text, [""])
                prompt_ids = _to_token_ids(prompt)[:max_length]
                return "", prompt_ids

            best_doc = ""
            best_ids: list[int] = []
            lo = 0
            hi = len(doc_token_ids)
            while lo <= hi:
                mid = (lo + hi) // 2
                if callable(decode):
                    candidate_doc = decode(
                        doc_token_ids[:mid], skip_special_tokens=False
                    )
                else:
                    candidate_doc = doc_text[:mid]

                prompt = self._format_jina_prompt(query_text, [candidate_doc])
                prompt_ids = _to_token_ids(prompt)
                if len(prompt_ids) <= max_length:
                    best_doc = candidate_doc
                    best_ids = prompt_ids
                    lo = mid + 1
                else:
                    hi = mid - 1

            if not best_ids:
                raise ValueError(
                    "Could not fit even a minimally truncated document into max_length. "
                    f"max_length={max_length}"
                )

            return best_doc, best_ids

        sanitized_query = self._sanitize_jina_text(query)
        sanitized_docs = [self._sanitize_jina_text(doc) for doc in documents]

        scores = [0.0] * len(documents)
        total_tokens = 0
        start = 0
        while start < len(sanitized_docs):
            chunk_doc_indices: list[int] = []
            chunk_docs: list[str] = []
            chunk_input_ids: list[int] | None = None
            cursor = start

            while cursor < len(sanitized_docs):
                candidate_docs = chunk_docs + [sanitized_docs[cursor]]
                candidate_prompt = self._format_jina_prompt(
                    sanitized_query, candidate_docs
                )
                candidate_ids = _to_token_ids(candidate_prompt)

                if len(candidate_ids) <= max_length:
                    chunk_docs = candidate_docs
                    chunk_doc_indices.append(cursor)
                    chunk_input_ids = candidate_ids
                    cursor += 1
                    continue

                if chunk_docs:
                    break

                truncated_doc, truncated_ids = _truncate_doc_to_fit(
                    sanitized_query,
                    sanitized_docs[cursor],
                )
                chunk_docs = [truncated_doc]
                chunk_doc_indices = [cursor]
                chunk_input_ids = truncated_ids
                cursor += 1
                break

            if chunk_input_ids is None or not chunk_doc_indices:
                raise ValueError("Failed to create a valid Jina reranker chunk.")

            input_array = mx.array([chunk_input_ids])
            hidden_states = self._get_jina_hidden_states(input_array)

            query_positions = [
                pos
                for pos, token_id in enumerate(chunk_input_ids)
                if token_id == query_embed_token_id
            ]
            if not query_positions:
                raise ValueError(
                    "Jina prompt does not contain '<|rerank_token|>' in tokenized input."
                )

            doc_positions = [
                pos
                for pos, token_id in enumerate(chunk_input_ids)
                if token_id == doc_embed_token_id
            ]
            if len(doc_positions) < len(chunk_docs):
                raise ValueError(
                    "Jina prompt/doc mismatch: detected fewer '<|embed_token|>' "
                    "positions than documents in chunk."
                )

            selected_doc_positions = doc_positions[: len(chunk_docs)]
            query_hidden = hidden_states[0, query_positions[0], :]
            doc_hidden = hidden_states[0, selected_doc_positions, :]

            query_vec = projector(query_hidden)
            doc_vecs = projector(doc_hidden)
            similarities = self._cosine_similarity(query_vec, doc_vecs)
            mx.eval(similarities)

            chunk_scores = similarities.tolist()
            for original_idx, score in zip(chunk_doc_indices, chunk_scores):
                scores[original_idx] = float(score)

            total_tokens += len(chunk_input_ids)
            start = cursor

        # Sort by score descending
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
        if not callable(processor):
            raise ValueError("SequenceClassification processor is not initialized.")

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

        # Forward pass (compiled primitive logits path when available)
        logits = None
        if self._is_compiled and self._compiled_seq_logits is not None:
            try:
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                logits = self._compiled_seq_logits(model_inputs)
            except Exception as e:
                logger.warning(
                    f"compiled reranker path failed for {self.model_name}: {e}; "
                    f"disabling compile and falling back to eager forward()"
                )
                self._is_compiled = False
                self._compiled_seq_logits = None

        if logits is None:
            if not callable(self.model):
                raise ValueError("SequenceClassification model is not initialized.")
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
            if not _is_causal_lm_reranker(Path(self.model_name)):
                raise ValueError(
                    f"Architecture {arch} is a CausalLM that can be used as a "
                    f"reranker, but the model directory name "
                    f"'{Path(self.model_name).name}' does not contain "
                    f"'reranker' or 'rerank'. Please rename the directory or "
                    f"use the correct model."
                )
            return

        # Multimodal reranker architectures share the arch string with VLM chat
        # models; use the same dir-name heuristic to disambiguate.
        if arch in MULTIMODAL_RERANKER_ARCHITECTURES:
            if not _is_causal_lm_reranker(Path(self.model_name)):
                raise ValueError(
                    f"Architecture {arch} is a VLM that can be used as a "
                    f"reranker, but the model directory name "
                    f"'{Path(self.model_name).name}' does not contain "
                    f"'reranker' or 'rerank'. Please rename the directory or "
                    f"use the correct model."
                )
            return

        if arch not in SUPPORTED_RERANKER_ARCHITECTURES:
            supported_list = ", ".join(
                sorted(
                    SUPPORTED_RERANKER_ARCHITECTURES
                    | CAUSAL_LM_RERANKER_ARCHITECTURES
                    | MULTIMODAL_RERANKER_ARCHITECTURES
                )
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
