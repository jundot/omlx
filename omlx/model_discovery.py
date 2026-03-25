# SPDX-License-Identifier: Apache-2.0
"""
Model discovery for oMLX multi-model serving.

This module scans a model directory and discovers available models,
estimating memory usage for each.

Supports:
- LLM models: Use BatchedEngine for continuous batching with paged KV cache
- VLM models: Use VLMBatchedEngine for vision-language model inference
- Embedding models: Use EmbeddingEngine for batch embedding generation
- Reranker models: Use RerankerEngine for document reranking
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ModelType = Literal["llm", "vlm", "embedding", "reranker"]
EngineType = Literal["batched", "vlm", "embedding", "reranker"]

# Known VLM (Vision-Language Model) types from mlx-vlm
VLM_MODEL_TYPES = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5_moe",
    "gemma3",
    "llava",
    "llava_next",
    "llava-qwen2",
    "mllama",
    "idefics3",
    "internvl_chat",
    "phi3_v",
    "paligemma",
    "mistral3",
    "pixtral",
    "molmo",
    "bunny_llama",
    "multi_modality",
    "florence2",
    "deepseekocr",
    "deepseekocr_2",
    "dots_ocr",
    "glm_ocr",
    "minicpmv",
    "phi4_siglip",
    "phi4mm",
}

# Known VLM architectures
VLM_ARCHITECTURES = {
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "MllamaForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
    "InternVLChatModel",
    "Idefics3ForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
    "Phi3VForCausalLM",
    "Pixtral",
    "MolmoForCausalLM",
    "Florence2ForConditionalGeneration",
}

# Known embedding model types from mlx-embeddings
EMBEDDING_MODEL_TYPES = {
    "bert",
    "xlm-roberta",
    "xlm_roberta",
    "modernbert",
    "siglip",
    "colqwen2_5",
    "colqwen2-5",
    "lfm2",
}

# Model types that have both embedding and LLM variants.
# These require architecture-based disambiguation via EMBEDDING_ARCHITECTURES.
AMBIGUOUS_EMBEDDING_MODEL_TYPES = {
    "qwen3",
    "gemma3-text",
    "gemma3_text",
}

# Known embedding architectures
EMBEDDING_ARCHITECTURES = {
    "BertModel",
    "BertForMaskedLM",
    "XLMRobertaModel",
    "XLMRobertaForMaskedLM",
    "ModernBertModel",
    "ModernBertForMaskedLM",
    "Qwen3ForTextEmbedding",
    "SiglipModel",
    "SiglipVisionModel",
    "SiglipTextModel",
}

# Supported reranker architectures
SUPPORTED_RERANKER_ARCHITECTURES = {
    "ModernBertForSequenceClassification",  # via mlx-embeddings
    "XLMRobertaForSequenceClassification",  # omlx native implementation
}

# CausalLM-based reranker architectures.
# These are standard CausalLM models fine-tuned for reranking via yes/no logit scoring.
# Detected by architecture + heuristic (model name or tokenizer hints).
CAUSAL_LM_RERANKER_ARCHITECTURES = {
    "Qwen3ForCausalLM",
    "JinaForRanking",  # Jina v3 reranker: uses <|score_token|> logits
}

# CausalLM-based embedding architectures.
# These use a standard CausalLM architecture but are fine-tuned for embeddings
# (no lm_head weights). Detected by architecture + directory name heuristic.
CAUSAL_LM_EMBEDDING_ARCHITECTURES = {
    "Qwen3ForCausalLM",  # Qwen3-Embedding uses CausalLM arch without lm_head
}

# Unsupported reranker architectures (future support)
UNSUPPORTED_RERANKER_ARCHITECTURES = {
    "BertForSequenceClassification",
    "Qwen3ForSequenceClassification",
}

# All known reranker architectures (for model type detection)
RERANKER_ARCHITECTURES = SUPPORTED_RERANKER_ARCHITECTURES | UNSUPPORTED_RERANKER_ARCHITECTURES

# Unsupported model types — detected and skipped during discovery.
# These models require audio endpoints (/v1/audio/*) that oMLX doesn't implement.
# Only top-level config fields are checked; nested audio_config/tts_config in
# multimodal models (e.g., MiniCPM-o) won't trigger this.
UNSUPPORTED_MODEL_TYPES = {
    "whisper",
    "qwen3_tts",
}

UNSUPPORTED_ARCHITECTURES = {
    "WhisperForConditionalGeneration",
}


@dataclass
class DiscoveredModel:
    """Information about a discovered model."""

    model_id: str  # Directory name (e.g., "llama-3b")
    model_path: str  # Full path to model directory
    model_type: ModelType  # "llm", "vlm", "embedding", or "reranker"
    engine_type: EngineType  # "batched", "vlm", "embedding", or "reranker"
    estimated_size: int  # Estimated memory usage in bytes
    config_model_type: str = ""  # Raw model_type from config.json (e.g., "deepseekocr_2")


def _is_unsupported_model(model_path: Path) -> bool:
    """
    Check if model is an unsupported type (TTS, ASR, etc.).

    Only checks top-level config fields. Multimodal models with nested
    audio_config/tts_config (e.g., MiniCPM-o) are not affected.
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    architectures = config.get("architectures", [])
    for arch in architectures:
        if arch in UNSUPPORTED_ARCHITECTURES:
            return True

    model_type = config.get("model_type", "")
    normalized = model_type.lower().replace("-", "_")
    return normalized in UNSUPPORTED_MODEL_TYPES or model_type in UNSUPPORTED_MODEL_TYPES


def _is_causal_lm_reranker(model_path: Path) -> bool:
    """
    Heuristic check for CausalLM models fine-tuned as rerankers.

    CausalLM rerankers (e.g., Qwen3-Reranker) use the same architecture as
    their base LLMs but are fine-tuned to output yes/no logits for relevance
    scoring. We detect them by checking the model directory name for "reranker"
    or "rerank" keywords, since config.json is identical to a standard LLM.
    """
    name_lower = model_path.name.lower()
    return "reranker" in name_lower or "rerank" in name_lower


def _is_causal_lm_embedding(model_path: Path) -> bool:
    """
    Heuristic check for CausalLM models fine-tuned as embedding models.

    CausalLM embeddings (e.g., Qwen3-Embedding) use the same architecture as
    their base LLMs but are fine-tuned for embeddings and ship without lm_head
    weights. We detect them by checking the model directory name for "embedding"
    or "embed" keywords, since config.json is identical to a standard LLM.
    """
    name_lower = model_path.name.lower()
    return "embedding" in name_lower or "embed" in name_lower


def detect_model_type(model_path: Path) -> ModelType:
    """
    Detect model type from config.json.

    Checks:
    1. architectures field for reranker-specific classes (SequenceClassification)
    2. architectures field for embedding-specific classes
    3. model_type field against known embedding types (unambiguous only)
    4. VLM detection via architectures, model_type, or vision_config presence

    Args:
        model_path: Path to model directory

    Returns:
        Model type: "llm", "vlm", "embedding", or "reranker"
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        return "llm"

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        return "llm"

    # Check architectures field for reranker first (more specific)
    architectures = config.get("architectures", [])
    for arch in architectures:
        if arch in RERANKER_ARCHITECTURES:
            return "reranker"

    # Check for CausalLM-based rerankers (e.g., Qwen3-Reranker).
    # These use a standard CausalLM architecture but are fine-tuned for reranking
    # via yes/no logit scoring. Detected by architecture + model directory name hint.
    for arch in architectures:
        if arch in CAUSAL_LM_RERANKER_ARCHITECTURES:
            if _is_causal_lm_reranker(model_path):
                return "reranker"

    # Check for CausalLM-based embeddings (e.g., Qwen3-Embedding).
    # These use a standard CausalLM architecture but are fine-tuned for embeddings
    # and ship without lm_head weights. Detected by architecture + directory name hint.
    for arch in architectures:
        if arch in CAUSAL_LM_EMBEDDING_ARCHITECTURES:
            if _is_causal_lm_embedding(model_path):
                return "embedding"

    # Check architectures field for embedding (before model_type to avoid
    # false positives from ambiguous model types like qwen3, gemma3-text)
    for arch in architectures:
        if arch in EMBEDDING_ARCHITECTURES:
            return "embedding"

    # Check model_type field for unambiguous embedding types
    model_type = config.get("model_type", "")
    # Normalize: replace hyphens with underscores and lowercase
    normalized_type = model_type.lower().replace("-", "_")

    if normalized_type in EMBEDDING_MODEL_TYPES or model_type in EMBEDDING_MODEL_TYPES:
        return "embedding"

    # Ambiguous embedding types (have both embedding and LLM variants):
    # only classified as embedding if architecture matched above
    if (
        normalized_type in AMBIGUOUS_EMBEDDING_MODEL_TYPES
        or model_type in AMBIGUOUS_EMBEDDING_MODEL_TYPES
    ):
        logger.info(
            f"Model type '{model_type}' has both embedding and LLM variants, "
            f"but architecture {architectures} is not an embedding architecture "
            "— treating as LLM"
        )

    # Check for VLM: architectures field
    for arch in architectures:
        if arch in VLM_ARCHITECTURES:
            return "vlm"

    # Check for VLM: model_type field (only if vision capabilities are present)
    # Some model families (e.g., qwen3_5_moe) have both VLM and text-only variants.
    # Text-only quants won't have vision_config in their config.json.
    if normalized_type in VLM_MODEL_TYPES:
        if "vision_config" in config:
            return "vlm"
        logger.info(
            f"Model type '{model_type}' is in VLM_MODEL_TYPES but no vision_config found — "
            "treating as LLM (text-only quant)"
        )

    # Check for VLM: presence of vision_config (fallback heuristic)
    # Catch-all for VLMs that aren't yet listed in VLM_MODEL_TYPES.
    if "vision_config" in config:
        return "vlm"

    return "llm"


def estimate_model_size(model_path: Path) -> int:
    """
    Estimate model memory usage from safetensors/bin file sizes.

    MLX keeps quantized weights in compressed form, so file size ≈ memory usage.

    Args:
        model_path: Path to model directory

    Returns:
        Estimated memory usage in bytes
    """
    total_size = 0

    # Primary: safetensors files
    safetensors_files = list(model_path.glob("*.safetensors"))
    for f in safetensors_files:
        total_size += f.stat().st_size

    # Fallback: .bin files (older PyTorch format)
    if total_size == 0:
        for f in model_path.glob("*.bin"):
            # Filter out non-weight files
            name_lower = f.name.lower()
            if "optimizer" in name_lower or "training" in name_lower:
                continue
            total_size += f.stat().st_size

    # Also check in subdirectories (some models store weights in subfolders)
    if total_size == 0:
        for f in model_path.glob("**/*.safetensors"):
            total_size += f.stat().st_size

    if total_size == 0:
        raise ValueError(f"No model weights found in {model_path}")

    # Add overhead for runtime buffers (~5%)
    overhead_factor = 1.05

    return int(total_size * overhead_factor)


def _is_adapter_dir(path: Path) -> bool:
    """Check if a directory contains a LoRA/PEFT adapter (has adapter_config.json)."""
    return (path / "adapter_config.json").exists()


def _is_model_dir(path: Path) -> bool:
    """Check if a directory contains a valid model (has config.json)."""
    return (path / "config.json").exists() and not _is_adapter_dir(path)


def _register_model(
    models: dict[str, DiscoveredModel],
    model_dir: Path,
    model_id: str,
) -> None:
    """Try to register a single model directory into the models dict."""
    try:
        if _is_unsupported_model(model_dir):
            logger.info(
                f"Skipping unsupported model: {model_id} "
                "(TTS/ASR models require audio endpoints not implemented in oMLX)"
            )
            return

        model_type = detect_model_type(model_dir)
        if model_type == "embedding":
            engine_type: EngineType = "embedding"
        elif model_type == "reranker":
            engine_type = "reranker"
        elif model_type == "vlm":
            engine_type = "vlm"
        else:
            engine_type = "batched"
        estimated_size = estimate_model_size(model_dir)

        # Read raw config model_type for sub-type detection (e.g., OCR models)
        config_model_type = ""
        try:
            import json
            with open(model_dir / "config.json") as f:
                config_model_type = json.load(f).get("model_type", "")
        except Exception:
            pass

        models[model_id] = DiscoveredModel(
            model_id=model_id,
            model_path=str(model_dir),
            model_type=model_type,
            engine_type=engine_type,
            estimated_size=estimated_size,
            config_model_type=config_model_type,
        )

        size_gb = estimated_size / (1024**3)
        logger.info(
            f"Discovered model: {model_id} "
            f"(type: {model_type}, engine: {engine_type}, size: {size_gb:.2f}GB)"
        )
    except Exception as e:
        logger.error(f"Failed to discover model {model_id}: {e}")


def discover_models(model_dir: Path) -> dict[str, DiscoveredModel]:
    """
    Scan model directory with two-level discovery.

    Supports both flat and organized directory layouts:

    Flat (one level):
        model_dir/
        ├── llama-3b/          → model_id: "llama-3b"
        │   ├── config.json
        │   └── *.safetensors
        └── qwen-7b/           → model_id: "qwen-7b"

    Organized (two levels):
        model_dir/
        ├── mlx-community/
        │   ├── llama-3b/      → model_id: "llama-3b"
        │   └── qwen-7b/       → model_id: "qwen-7b"
        └── Qwen/
            └── Qwen3-8B/      → model_id: "Qwen3-8B"

    If a first-level subdirectory has config.json, it's treated as a model.
    Otherwise, its children are scanned for models (organization folder).

    Args:
        model_dir: Path to directory containing model subdirectories

    Returns:
        Dictionary mapping model_id to DiscoveredModel
    """
    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")

    if not model_dir.is_dir():
        raise ValueError(f"Model directory is not a directory: {model_dir}")

    models: dict[str, DiscoveredModel] = {}

    for subdir in sorted(model_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue

        if _is_adapter_dir(subdir):
            logger.info(
                f"Skipping LoRA adapter: {subdir.name} "
                "(oMLX does not support LoRA/PEFT adapters)"
            )
        elif _is_model_dir(subdir):
            # Level 1: direct model folder
            _register_model(models, subdir, subdir.name)
        else:
            # Level 2: organization folder — scan children
            has_children = False
            for child in sorted(subdir.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                if _is_adapter_dir(child):
                    logger.info(
                        f"Skipping LoRA adapter: {child.name} "
                        "(oMLX does not support LoRA/PEFT adapters)"
                    )
                elif _is_model_dir(child):
                    has_children = True
                    _register_model(models, child, child.name)

            if not has_children:
                logger.debug(
                    f"Skipping {subdir.name}: no config.json found "
                    f"(not a model or organization folder)"
                )

    # Fallback: if no models found and the directory itself is a model, register it.
    # This supports pointing directly at a single model folder, e.g.:
    #   /Models/Qwen3.5-9B-MLX-4bit/  (contains config.json and weight files)
    if not models and _is_model_dir(model_dir):
        _register_model(models, model_dir, model_dir.name)

    return models


def discover_models_from_dirs(
    model_dirs: list[Path],
) -> dict[str, DiscoveredModel]:
    """
    Scan multiple model directories and merge results.

    Each directory is scanned using discover_models(). On model_id conflicts,
    the first directory's model takes priority (earlier directory wins).

    Args:
        model_dirs: List of paths to directories containing model subdirectories

    Returns:
        Dictionary mapping model_id to DiscoveredModel
    """
    merged: dict[str, DiscoveredModel] = {}

    for model_dir in model_dirs:
        if not model_dir.exists():
            logger.warning(f"Model directory does not exist, skipping: {model_dir}")
            continue
        if not model_dir.is_dir():
            logger.warning(f"Not a directory, skipping: {model_dir}")
            continue

        try:
            discovered = discover_models(model_dir)
        except ValueError as e:
            logger.warning(f"Skipping directory {model_dir}: {e}")
            continue

        for model_id, info in discovered.items():
            if model_id in merged:
                logger.warning(
                    f"Duplicate model_id '{model_id}' found in {model_dir}, "
                    f"keeping version from {merged[model_id].model_path}"
                )
                continue
            merged[model_id] = info

    return merged


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f}PB"
