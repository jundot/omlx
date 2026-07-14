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
- Audio STT models: Use STTEngine for speech-to-text (Whisper, Qwen3-ASR, ...)
- Audio TTS models: Use TTSEngine for text-to-speech (Qwen3-TTS, Kokoro, ...)
"""

import contextlib
import json
import logging
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ModelType = Literal["llm", "vlm", "embedding", "reranker", "audio_stt", "audio_tts", "audio_sts"]
EngineType = Literal["batched", "vlm", "embedding", "reranker", "audio_stt", "audio_tts", "audio_sts"]
RootFingerprint = tuple[int, int]

# Known VLM (Vision-Language Model) types from mlx-vlm
VLM_MODEL_TYPES = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5_moe",
    "gemma3",
    "gemma4",
    "gemma4_unified",
    "diffusion_gemma",
    "llava",
    "llava_next",
    "llava-qwen2",
    "llava_qwen2",  # underscore form — matches FastVLM checkpoints on disk
    "mllama",
    "idefics3",
    "internvl_chat",
    "phi3_v",
    "paligemma",
    "mistral3",
    "pixtral",
    "molmo",
    "molmo2",
    "bunny_llama",
    "multi_modality",
    "florence2",
    "deepseekocr",
    "deepseekocr_2",
    "dots_ocr",
    "glm_ocr",
    "minimax_m3_vl",
    "minicpmv",
    "phi4_siglip",
    "phi4mm",
    "youtu_vl",
}

# Text-only model families that are implemented in mlx-vlm rather than
# mlx-lm. They still use the VLM engine because that path loads mlx-vlm
# models and adapts their language model to oMLX's scheduler.
VLM_NATIVE_TEXT_MODEL_TYPES = {
    "cohere2_moe",
    "minimax_m3",
}

# Speculative-decoding "helper" checkpoints (dFlash / MTP / assistant drafters)
# are never meant to be served as standalone chat models. Some declare a
# distinctive top-level model_type — an ``*_assistant`` (e.g. gemma4_assistant)
# or ``*_mtp`` (e.g. qwen3_5_mtp) marker — but DFlash draft checkpoints declare
# a plain model_type (e.g. ``qwen3``) and are only distinguishable by their
# architecture name (``DFlashDraftModel``) or a drafter-only config block
# (``dflash_config``). Keep these in sync with the drafter resolution in
# engine_pool.py (~1498) and the dflash gate in engine/dflash.py when new
# drafter families are added.
HELPER_CONFIG_MODEL_TYPE_SUFFIXES = ("_assistant", "_mtp")
_HELPER_ARCH_TOKENS = ("draft", "assistant", "mtp")
_HELPER_CONFIG_KEYS = ("dflash_config",)


def is_helper_config_model_type(config_model_type: str | None) -> bool:
    """True when ``config_model_type`` marks a speculative-decoding drafter.

    These are the raw top-level ``model_type`` values from a checkpoint's
    config.json (e.g. ``gemma4_assistant``, ``qwen3_5_mtp``). Note this misses
    DFlash drafts, whose model_type is a plain ``qwen3`` — use
    :func:`is_helper_model_config` when the full config dict is available.
    """
    if not isinstance(config_model_type, str) or not config_model_type:
        return False
    mt = config_model_type.lower()
    return mt.endswith(HELPER_CONFIG_MODEL_TYPE_SUFFIXES)


def is_helper_model_config(config: dict) -> bool:
    """True when a parsed config.json marks a speculative-decoding drafter.

    Catches three intrinsic signals, any of which is definitive:
    an ``*_assistant`` / ``*_mtp`` model_type, a drafter architecture name
    (e.g. ``DFlashDraftModel``), or a drafter-only config block
    (e.g. ``dflash_config``). These are helper checkpoints backing
    dFlash / MTP / assistant speculative decoding, not chat models.
    """
    if not isinstance(config, dict):
        return False
    if is_helper_config_model_type(config.get("model_type")):
        return True
    if any(key in config for key in _HELPER_CONFIG_KEYS):
        return True
    architectures = config.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    elif not isinstance(architectures, list | tuple | set):
        return False
    for arch in architectures:
        arch_lower = str(arch).lower()
        if any(token in arch_lower for token in _HELPER_ARCH_TOKENS):
            return True
    return False


# Known VLM architectures
VLM_ARCHITECTURES = {
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "MllamaForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
    "Gemma4ForConditionalGeneration",
    "InternVLChatModel",
    "Idefics3ForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
    "Phi3VForCausalLM",
    "Pixtral",
    "MolmoForCausalLM",
    "Molmo2ForConditionalGeneration",
    "LlavaQwen2ForCausalLM",  # apple/FastVLM (all sizes)
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
}

# Model types that have both embedding and LLM variants.
# These require architecture-based disambiguation via EMBEDDING_ARCHITECTURES.
AMBIGUOUS_EMBEDDING_MODEL_TYPES = {
    "qwen3",
    "gemma3-text",
    "gemma3_text",
    "lfm2",
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
    "JinaForRanking",  # Jina v3 listwise reranker
}

# CausalLM-based reranker architectures.
# These are standard CausalLM models fine-tuned for reranking via yes/no logit scoring.
# Detected by architecture + heuristic (model name or tokenizer hints).
CAUSAL_LM_RERANKER_ARCHITECTURES = {
    "Qwen3ForCausalLM",
}

# CausalLM-based embedding architectures.
# These use a standard CausalLM architecture but are fine-tuned for embeddings
# (no lm_head weights). Detected by architecture + directory name heuristic.
CAUSAL_LM_EMBEDDING_ARCHITECTURES = {
    "Qwen3ForCausalLM",  # Qwen3-Embedding uses CausalLM arch without lm_head
    "Qwen2ForCausalLM",  # jina-code-embeddings & similar; only treated as an
    # embedding when the dir-name heuristic (_is_causal_lm_embedding) also
    # matches, so Qwen2/Qwen2.5 chat models are unaffected.
}

# Multimodal (VLM-based) reranker architectures.
# These share an architecture with VLM chat models but are fine-tuned for
# reranking. Loaded via mlx-embeddings' model.process() API. Distinguished
# from VLM chat by directory name heuristic.
MULTIMODAL_RERANKER_ARCHITECTURES = {
    "Qwen3VLForConditionalGeneration",  # Qwen3-VL-Reranker
}

# Multimodal (VLM-based) embedding architectures.
# Same arch as the reranker variant; distinguished by directory name hint.
MULTIMODAL_EMBEDDING_ARCHITECTURES = {
    "Qwen3VLForConditionalGeneration",  # Qwen3-VL-Embedding
}

# Unsupported reranker architectures (future support)
UNSUPPORTED_RERANKER_ARCHITECTURES = {
    "BertForSequenceClassification",
    "Qwen3ForSequenceClassification",
}

# All known reranker architectures (for model type detection)
RERANKER_ARCHITECTURES = SUPPORTED_RERANKER_ARCHITECTURES | UNSUPPORTED_RERANKER_ARCHITECTURES

# Unsupported model types — detected and skipped during discovery.
# Only top-level config fields are checked; nested audio_config/tts_config in
# multimodal models (e.g., MiniCPM-o) won't trigger this.
# Note: "whisper" and "qwen3_tts" were previously listed here but are now
# handled as audio types (audio_stt / audio_tts) — see AUDIO_* sets below.
UNSUPPORTED_MODEL_TYPES: set[str] = set()

UNSUPPORTED_ARCHITECTURES: set[str] = set()

# ---------------------------------------------------------------------------
# Audio model detection — dynamically loaded from mlx-audio when available
# ---------------------------------------------------------------------------
#
# mlx-audio maintains MODEL_REMAPPING dicts (model_type → module directory)
# and model directories under mlx_audio/{stt,tts,sts}/models/. We read these
# at import time so oMLX automatically recognises new audio model families
# when mlx-audio is updated.  Falls back to static sets when mlx-audio is
# not installed.
#
# Some base LLM model_types (qwen3, llama, …) collide with mlx-audio TTS
# model directory names because mlx-audio extends these architectures for
# audio.  We exclude them so a plain Qwen3 LLM is not misdetected as TTS.

_LLM_TYPE_COLLISIONS = {"qwen3", "llama", "dense"}


def _build_audio_detection_sets():
    """Build STT/TTS/STS model-type sets from mlx-audio at import time.

    Returns (stt_types, tts_types, sts_types) where each is a set of
    model_type strings that should trigger audio detection.
    """
    try:
        from pathlib import Path as _P

        import mlx_audio as _mla

        _base = _P(_mla.__file__).parent

        def _dir_names(subdir: str) -> set:
            d = _base / subdir / "models"
            if d.is_dir():
                return {p.name for p in d.iterdir()
                        if p.is_dir() and not p.name.startswith("__")}
            return set()

        # TTS: MODEL_REMAPPING keys + model dir names
        from mlx_audio.tts.utils import MODEL_REMAPPING as _tts_remap
        tts = set(_tts_remap.keys()) | _dir_names("tts")

        # STT: MODEL_REMAPPING keys + model dir names
        from mlx_audio.stt.utils import MODEL_REMAPPING as _stt_remap
        stt = set(_stt_remap.keys()) | _dir_names("stt")

        # STS: model dir names only (no unified utils/remapping)
        sts = _dir_names("sts")

        # Strip base-LLM names that collide with audio model dirs
        tts -= _LLM_TYPE_COLLISIONS
        stt -= _LLM_TYPE_COLLISIONS

        logger.debug(
            "Audio detection sets loaded from mlx-audio: "
            "STT=%d, TTS=%d, STS=%d", len(stt), len(tts), len(sts),
        )
        return stt, tts, sts

    except Exception:
        logger.debug("mlx-audio not available — using static audio detection sets")
        # Static fallback so model discovery still works without mlx-audio
        _stt = {"whisper", "qwen3_asr", "parakeet", "qwen2_audio"}
        _tts = {"qwen3_tts", "kokoro", "chatterbox", "vibevoice", "vibevoice_streaming", "kugelaudio", "audiodit"}
        _sts = {"deepfilternet", "mossformer2_se", "sam_audio", "lfm_audio"}
        return _stt, _tts, _sts


AUDIO_STT_MODEL_TYPES, AUDIO_TTS_MODEL_TYPES, AUDIO_STS_MODEL_TYPES = (
    _build_audio_detection_sets()
)

# Architecture-based detection — these are checked before model_type and
# are always static because architecture strings are stable identifiers.
AUDIO_STT_ARCHITECTURES = {
    "WhisperForConditionalGeneration",
    "Qwen3ASRForConditionalGeneration",
    "ParakeetForCTC",
    "Qwen2AudioForConditionalGeneration",
}

AUDIO_TTS_ARCHITECTURES = {
    "KokoroForConditionalGeneration",
    "Qwen3TTSForConditionalGeneration",
    "ChatterboxForConditionalGeneration",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceStreamingForConditionalGenerationInference",
    "KugelAudioForConditionalGeneration",
}

AUDIO_STS_ARCHITECTURES = {
    "DeepFilterNetModel",
    "MossFormer2SEModel",
    "SAMAudio",
    "LFM2AudioModel",
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
    thinking_default: bool | None = None  # True if model thinks by default, False if not, None if unknown
    preserve_thinking_default: bool | None = None  # True when template supports preserve_thinking (Qwen 3.6+)
    model_context_length: int | None = None  # Declared context length from config.json (None if unknown)
    source_type: str = "local"  # "local" or "hf_cache"
    source_repo_id: str | None = None  # HuggingFace repo id for cache-backed models
    is_helper: bool = False  # Speculative-decoding drafter (dFlash/Assistant/MTP)


@dataclass
class ModelDiscoveryResult:
    """Discovered models plus per-root scan and structural-presence evidence.

    ``root_models`` contains loadable models. ``root_model_entries`` also
    includes structurally present local folders and HF cache entries that are
    temporarily incomplete, so persistence cleanup can remain conservative
    without exposing those entries as active models.
    """

    models: dict[str, DiscoveredModel]
    root_complete: dict[Path, bool]
    root_models: dict[Path, frozenset[str]]
    root_model_entries: dict[Path, frozenset[str]]
    root_fingerprints: dict[Path, RootFingerprint | None]


@dataclass
class _DiscoveryHealth:
    """Mutable scan state shared by one root's best-effort traversal."""

    complete: bool = True


def _model_root_key(path: Path) -> Path:
    """Return the normalized physical key used by discovery reports."""
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError):
        return path.expanduser().absolute()


def _root_fingerprint(
    path: Path,
    *,
    health: _DiscoveryHealth | None = None,
) -> RootFingerprint | None:
    """Return the physical root identity without hiding access failures."""
    try:
        root_stat = path.stat()
    except OSError as e:
        if health is not None:
            health.complete = False
        logger.warning(
            "Could not fingerprint model root %s: %s: %s",
            path,
            type(e).__name__,
            e,
        )
        return None
    return int(root_stat.st_dev), int(root_stat.st_ino)


@dataclass(frozen=True)
class HfCacheEntry:
    """Resolved HuggingFace Hub cache entry."""

    snapshot_path: Path
    model_id: str
    source_repo_id: str


def _is_unsupported_model(model_path: Path) -> bool:
    """
    Check if model is an unsupported type that should be skipped during discovery.

    Audio models (STT/TTS) are NOT unsupported — they are detected as
    "audio_stt" or "audio_tts" by detect_model_type() and served via
    their own engine types.

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


def _has_sentence_transformers_embedding_pipeline(model_path: Path) -> bool:
    """
    Detect sentence-transformers style embedding exports via modules.json.

    This allows oMLX to recognize embedding exports whose base transformer
    architecture is ambiguous (for example gemma3_text) but which include
    sentence-transformers pooling/normalization modules.
    """
    modules_path = model_path / "modules.json"
    if not modules_path.exists():
        return False

    try:
        with open(modules_path) as f:
            modules = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    if not isinstance(modules, list):
        return False

    module_types = {
        module.get("type", "")
        for module in modules
        if isinstance(module, dict)
    }
    if "sentence_transformers.models.Transformer" not in module_types:
        return False

    return any(
        module_type.startswith("sentence_transformers.models.")
        and module_type != "sentence_transformers.models.Transformer"
        for module_type in module_types
    )


def _looks_like_kokoro_config(config: dict) -> bool:
    """Return True for Kokoro exports that omit HF ``model_type``.

    mlx-community Kokoro conversions (e.g. Kokoro-82M-bf16) keep the original
    Kokoro config — top-level ``istftnet`` + ``plbert`` sections and a
    ``vocab`` table — with no HF-style ``model_type``/``architectures``.
    mlx-audio loads them fine, but oMLX must classify them as TTS during
    discovery or they fall through to the LLM engine, whose loader only
    matches ``model*.safetensors`` and fails with a misleading
    "No safetensors found" error.
    """
    if not isinstance(config, dict):
        return False
    return (
        isinstance(config.get("istftnet"), dict)
        and isinstance(config.get("plbert"), dict)
        and isinstance(config.get("vocab"), dict)
    )


def _looks_like_nemo_asr_config(config: dict) -> bool:
    """Return True for NeMo ASR exports that omit HF ``model_type``.

    NVIDIA Parakeet TDT/CTC MLX conversions keep the original NeMo ASR
    training config instead of a HuggingFace-style ``model_type`` or
    ``architectures`` field.  mlx-audio can load these models by name, but
    oMLX must still classify them as STT during discovery or they fall through
    to the LLM engine and fail with a misleading ``'model_type'`` error.

    Only top-level NeMo ASR module targets are considered so multimodal models
    with nested ``audio_config`` sections are not misclassified.
    """
    if not isinstance(config, dict):
        return False

    module_targets: list[str] = []
    for key in ("preprocessor", "encoder", "decoder", "joint"):
        value = config.get(key)
        if isinstance(value, dict):
            target = value.get("_target_")
            if isinstance(target, str):
                module_targets.append(target.lower())

    if not any("nemo.collections.asr" in target for target in module_targets):
        return False

    # NeMo ASR configs include an audio preprocessor plus tokenizer/decoder
    # metadata.  Requiring these keeps the heuristic narrow while covering
    # Parakeet TDT exports whose config has no model_type at all.
    preprocessor = config.get("preprocessor")
    has_audio_preprocessor = isinstance(preprocessor, dict) and (
        "audio" in str(preprocessor.get("_target_", "")).lower()
        or "melspectrogram" in str(preprocessor.get("_target_", "")).lower()
    )
    has_asr_head = isinstance(config.get("decoder"), dict) or isinstance(
        config.get("joint"), dict
    )
    has_tokenizer = isinstance(config.get("tokenizer"), dict)
    return has_audio_preprocessor and has_asr_head and has_tokenizer


def _has_vision_subconfig(config: dict) -> bool:
    """
    Return True if ``config`` carries evidence of a vision sub-config.

    Three keys cover the conventions in the wild:

    - ``vision_config`` — most VLMs (Qwen2-VL, Gemma3, LLaVA-Next, ...).
    - ``vit_config`` — Molmo / Molmo2 family.
    - ``mm_vision_tower`` — older LLaVA family including FastVLM's
      ``llava_qwen2``. The check is non-empty-only: a config-stub text-only
      quant could in principle declare a tower path it doesn't ship weights
      for, but in practice bf16 FastVLM ships a real path string.

    Used by the VLM classifier in :func:`detect_model_type` and by other
    paths (``oq``, admin model info) that need to ask "is this a VLM?".
    """
    return (
        "vision_config" in config
        or "vit_config" in config
        or bool(config.get("mm_vision_tower"))
    )


def _architecture_indicates_causal_lm(architectures: list[str]) -> bool:
    """True when ``architectures`` describe a text causal LM (not mlx-audio STS).

    Liquid LFM text checkpoints (LFM2, LFM2.5 MoE, etc.) use ``lfm*`` model
    types and ``*ForCausalLM`` classes. mlx-audio LFM STS uses ``LFM2AudioModel``
    and is handled earlier via :data:`AUDIO_STS_ARCHITECTURES`.
    """
    return any("causallm" in arch.lower() for arch in architectures)


def detect_model_type(model_path: Path) -> ModelType:
    """
    Detect model type from config.json.

    Checks:
    1. architectures field for reranker-specific classes (SequenceClassification)
    2. CausalLM-based reranker/embedding detection (architecture + directory name)
    3. sentence-transformers pipeline detection via modules.json
    4. architectures field for embedding-specific classes
    5. model_type field against known embedding types (unambiguous only)
    6. VLM detection via architectures, model_type, or vision sub-config
       presence (``vision_config`` / ``vit_config`` / non-empty
       ``mm_vision_tower`` — see :func:`_has_vision_subconfig`)
    7. Audio model detection (STT/TTS/STS)

    Args:
        model_path: Path to model directory

    Returns:
        Model type: "llm", "vlm", "embedding", "reranker", "audio_stt", "audio_tts", or "audio_sts"
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

    # Check for multimodal (VLM-based) rerankers and embeddings.
    # Same architecture string as VLM chat models; distinguished by the
    # directory name heuristic. Must come before VLM detection below so
    # the reranker/embedding hint wins over default VLM classification.
    for arch in architectures:
        if arch in MULTIMODAL_RERANKER_ARCHITECTURES and _is_causal_lm_reranker(model_path):
            return "reranker"
        if arch in MULTIMODAL_EMBEDDING_ARCHITECTURES and _is_causal_lm_embedding(model_path):
            return "embedding"

    if _has_sentence_transformers_embedding_pipeline(model_path):
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

    if normalized_type in VLM_NATIVE_TEXT_MODEL_TYPES:
        logger.info(
            f"{model_type} detected as mlx-vlm native text model"
        )
        return "vlm"

    # Check for VLM: architectures field
    # Some text-only quants (e.g., unsloth/gemma-4-31b-it-MLX-8bit) keep the VLM
    # architecture name but strip vision_config and vision weights.
    # For model families known to have text-only variants, require evidence
    # of a vision sub-config — see :func:`_has_vision_subconfig` for the
    # three keys we accept (``vision_config``, ``vit_config``,
    # ``mm_vision_tower``).
    for arch in architectures:
        if arch in VLM_ARCHITECTURES:
            if normalized_type in VLM_MODEL_TYPES and not _has_vision_subconfig(config):
                logger.info(
                    f"Architecture '{arch}' is a VLM architecture but no "
                    "vision_config / vit_config / mm_vision_tower found — "
                    "treating as LLM (text-only quant)"
                )
                break
            return "vlm"

    # Check for VLM: model_type field (only if vision capabilities are present)
    # Some model families (e.g., qwen3_5_moe) have both VLM and text-only variants.
    # Text-only quants won't carry a vision sub-config. gemma4_unified and
    # diffusion_gemma are exceptions: they are served by mlx-vlm regardless of
    # vision_config presence in config.json.
    if normalized_type in VLM_MODEL_TYPES:
        if normalized_type in {"gemma4_unified", "diffusion_gemma"}:
            logger.info(
                f"{model_type} detected as VLM (mlx-vlm native model)"
            )
            return "vlm"
        if _has_vision_subconfig(config):
            return "vlm"
        logger.info(
            f"Model type '{model_type}' is in VLM_MODEL_TYPES but no "
            "vision_config / vit_config / mm_vision_tower found — "
            "treating as LLM (text-only quant)"
        )

    # Check for VLM: presence of a vision sub-config (fallback heuristic).
    # Catch-all for VLMs that aren't yet listed in VLM_MODEL_TYPES.
    if _has_vision_subconfig(config):
        return "vlm"

    # Check for audio models — architectures take priority over model_type.
    # Only top-level architectures/model_type are inspected; nested audio_config
    # inside multimodal models (e.g., MiniCPM-o) does not trigger this path.
    #
    # Architecture check first (unambiguous):
    for arch in architectures:
        if arch in AUDIO_STT_ARCHITECTURES:
            return "audio_stt"

    # NeMo ASR exports such as mlx-community/parakeet-tdt-0.6b-v3 ship a
    # NeMo training config without HF model_type/architectures.  They are
    # still STT models and mlx-audio can load them by directory/repo name.
    if _looks_like_nemo_asr_config(config):
        return "audio_stt"
    # Kokoro exports similarly ship a bare original config (istftnet/plbert).
    if _looks_like_kokoro_config(config):
        return "audio_tts"
    for arch in architectures:
        if arch in AUDIO_TTS_ARCHITECTURES:
            return "audio_tts"
    for arch in architectures:
        if arch in AUDIO_STS_ARCHITECTURES:
            return "audio_sts"

    # model_type check (dynamically loaded from mlx-audio when available).
    # Check TTS before STT because some model_type values (e.g. "vibevoice")
    # appear in both sets — TTS is the more common category for these.
    if normalized_type in AUDIO_TTS_MODEL_TYPES or model_type in AUDIO_TTS_MODEL_TYPES:
        return "audio_tts"
    if normalized_type in AUDIO_STT_MODEL_TYPES or model_type in AUDIO_STT_MODEL_TYPES:
        return "audio_stt"
    if normalized_type in AUDIO_STS_MODEL_TYPES or model_type in AUDIO_STS_MODEL_TYPES:
        return "audio_sts"
    # mlx-audio LFM STS may use an "lfm*" model_type without a known architecture
    # string yet. Liquid LFM *text* checkpoints share that prefix — disambiguate
    # with CausalLM architecture names (LFM2 / LFM2.5 MoE, future lfm* LMs).
    if normalized_type.startswith("lfm") and normalized_type not in EMBEDDING_MODEL_TYPES:
        if _architecture_indicates_causal_lm(architectures):
            return "llm"
        return "audio_sts"

    return "llm"


def detect_thinking_default(model_path: Path) -> bool | None:
    """Detect whether a model's chat template enables thinking by default.

    Inspects the Jinja chat template for ``enable_thinking`` references and
    determines the default behaviour:

    * **True** — model thinks by default (e.g. Qwen 3.x: only suppresses
      thinking when ``enable_thinking is false``).
    * **False** — model suppresses thinking by default (e.g. Gemma 4: only
      enables thinking when ``enable_thinking`` is truthy,
      ``default(false)``).
    * **None** — template does not reference ``enable_thinking`` (model has
      no thinking toggle).
    """
    # Try standalone Jinja file first, then tokenizer_config.json
    template_text = None
    jinja_path = model_path / "chat_template.jinja"
    if jinja_path.exists():
        with contextlib.suppress(OSError):
            template_text = jinja_path.read_text(encoding="utf-8")

    if template_text is None:
        tc_path = model_path / "tokenizer_config.json"
        if tc_path.exists():
            try:
                with open(tc_path) as f:
                    tc = json.load(f)
                template_text = tc.get("chat_template")
            except Exception:
                pass

    if not template_text or "enable_thinking" not in template_text:
        return None

    # Heuristic: if the template only disables thinking when explicitly
    # ``enable_thinking is false``, then thinking is ON by default.
    # If the template requires ``enable_thinking`` to be truthy or uses
    # ``default(false)``, then thinking is OFF by default.
    if "enable_thinking is false" in template_text:
        return True  # ON by default (Qwen pattern)
    if "default(false)" in template_text or "enable_thinking)" in template_text:
        return False  # OFF by default (Gemma pattern)

    return None


# Context-length keys, in priority order. Order mirrors HuggingFace
# Transformers conventions: ``max_position_embeddings`` is the canonical
# field for decoder-only LLMs; ``max_seq_len`` / ``seq_length`` show up on
# Llama / Mistral / Qwen forks; ``n_positions`` is the GPT-2 lineage.
_CONTEXT_LENGTH_KEYS = (
    "max_position_embeddings",
    "max_seq_len",
    "max_seq_length",
    "seq_length",
    "n_positions",
)

# tokenizer_config.json's ``model_max_length`` is Transformers' fallback
# field. Transformers seeds it with ``int(1e30)`` when the tokenizer has
# no real cap, and downstream code distinguishes that sentinel from a
# real long context. Anything above ~1e18 is treated as the sentinel.
_TOKENIZER_MAX_LENGTH_SENTINEL = 10**18


def _read_model_context_length(model_path: Path) -> int | None:
    """Discover the declared context length from a model's config files.

    Resolution order:

    1. Top-level ``config.json`` keys (``max_position_embeddings`` first,
       then the rest of :data:`_CONTEXT_LENGTH_KEYS`).
    2. Nested ``text_config`` / ``language_config`` keys (used by VLM
       wrappers and Qwen-style MoE configs that put the language head in
       a sub-object).
    3. ``tokenizer_config.json``'s ``model_max_length`` — but only when
       it is a finite positive integer, since Transformers seeds it with
       ``int(1e30)`` as a "no cap" sentinel.

    Returns:
        Positive integer context length, or ``None`` when no usable
        value was found in any of the above.
    """
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                model_config = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Failed to read config.json for {model_path}: {e}")
            model_config = {}

        for key in _CONTEXT_LENGTH_KEYS:
            value = model_config.get(key)
            if isinstance(value, int) and value > 0:
                return value

        for nest_key in ("text_config", "language_config"):
            nested = model_config.get(nest_key)
            if isinstance(nested, dict):
                for key in _CONTEXT_LENGTH_KEYS:
                    value = nested.get(key)
                    if isinstance(value, int) and value > 0:
                        return value

    tc_path = model_path / "tokenizer_config.json"
    if tc_path.exists():
        try:
            with open(tc_path, encoding="utf-8") as f:
                tc = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Failed to read tokenizer_config.json for {model_path}: {e}")
            tc = {}

        value = tc.get("model_max_length")
        if isinstance(value, int) and 0 < value < _TOKENIZER_MAX_LENGTH_SENTINEL:
            return value

    return None


def detect_preserve_thinking(model_path: Path) -> bool | None:
    """Detect whether a model's chat template supports ``preserve_thinking``.

    Qwen 3.6+ templates strip ``<think>`` blocks from historical assistant
    turns by default and only keep them when ``preserve_thinking`` is true.
    Stripping breaks KV prefix cache reuse, so we default to True when the
    template supports this flag.

    Returns:
        True if the template references ``preserve_thinking`` (should be
        enabled), None otherwise (template has no such flag).
    """
    template_text = None
    jinja_path = model_path / "chat_template.jinja"
    if jinja_path.exists():
        with contextlib.suppress(OSError):
            template_text = jinja_path.read_text(encoding="utf-8")

    if template_text is None:
        tc_path = model_path / "tokenizer_config.json"
        if tc_path.exists():
            try:
                with open(tc_path) as f:
                    tc = json.load(f)
                template_text = tc.get("chat_template")
            except Exception:
                pass

    if not template_text or "preserve_thinking" not in template_text:
        return None

    return True


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


def _path_exists(path: Path, *, health: _DiscoveryHealth | None = None) -> bool:
    """Check path existence without hiding an access failure as absence."""
    try:
        path.stat()
    except FileNotFoundError:
        return False
    except OSError as e:
        if health is not None:
            health.complete = False
        logger.warning(
            "Skipping inaccessible model metadata %s: %s: %s",
            path,
            type(e).__name__,
            e,
        )
        return False
    return True


def _is_adapter_dir(path: Path, *, health: _DiscoveryHealth | None = None) -> bool:
    """Check if a directory contains a LoRA/PEFT adapter (has adapter_config.json)."""
    return _path_exists(path / "adapter_config.json", health=health)


def _is_model_dir(path: Path, *, health: _DiscoveryHealth | None = None) -> bool:
    """Check if a directory contains a valid model (has config.json)."""
    return _path_exists(path / "config.json", health=health) and not _is_adapter_dir(
        path, health=health
    )


def model_directory_access_error(path: Path) -> str | None:
    """Return a user-facing error if a model directory cannot be scanned."""
    try:
        if not path.exists():
            return f"Model directory does not exist: {path}"
        if not path.is_dir():
            return f"Model directory is not a directory: {path}"
        next(path.iterdir(), None)
    except OSError as e:
        return (
            f"Model directory is not readable: {path} "
            f"({type(e).__name__}: {e})"
        )
    return None


def model_directory_write_error(path: Path, *, create: bool = False) -> str | None:
    """Return a user-facing error if a model directory cannot be written."""
    try:
        if not path.exists():
            if create:
                path.mkdir(parents=True, exist_ok=True)
            else:
                return f"Model directory does not exist: {path}"
        if not path.is_dir():
            return f"Model directory is not a directory: {path}"
    except OSError as e:
        return (
            f"Model directory is not writable: {path} "
            f"({type(e).__name__}: {e})"
        )

    access_error = model_directory_access_error(path)
    if access_error is not None:
        return access_error

    try:
        with tempfile.NamedTemporaryFile(
            prefix=".omlx-write-test-",
            dir=path,
            delete=True,
        ) as f:
            f.write(b"")
            f.flush()
    except OSError as e:
        return (
            f"Model directory is not writable: {path} "
            f"({type(e).__name__}: {e})"
        )

    return None


def _iter_readable_entries(
    path: Path,
    context: str,
    *,
    health: _DiscoveryHealth | None = None,
) -> list[Path]:
    """Return sorted directory entries, or an empty list when scanning fails."""
    try:
        return sorted(path.iterdir())
    except OSError as e:
        if health is not None:
            health.complete = False
        logger.warning(
            "Skipping unreadable %s %s: %s: %s",
            context,
            path,
            type(e).__name__,
            e,
        )
        return []


def _is_readable_dir(
    path: Path,
    context: str,
    *,
    health: _DiscoveryHealth | None = None,
) -> bool:
    try:
        return stat.S_ISDIR(path.stat().st_mode)
    except OSError as e:
        if health is not None:
            health.complete = False
        logger.warning(
            "Skipping inaccessible %s %s: %s: %s",
            context,
            path,
            type(e).__name__,
            e,
        )
        return False


def _decode_hf_cache_model_id(path: Path) -> tuple[str, str] | None:
    """Decode models--org--repo into (route_safe_id, repo_id)."""
    name = path.name
    if not name.startswith("models--"):
        return None

    encoded = name[len("models--"):]
    if not encoded:
        return None

    parts = encoded.split("--")
    if len(parts) == 1:
        return parts[0], parts[0]

    repo_name = "--".join(parts[1:])
    return f"{parts[0]}--{repo_name}", f"{parts[0]}/{repo_name}"


def _resolve_hf_cache_entry(
    path: Path,
    *,
    health: _DiscoveryHealth | None = None,
) -> HfCacheEntry | None:
    """Resolve an HF Hub cache entry (models--Org--Name/) to its active snapshot.

    Returns an HfCacheEntry or None if not a valid HF model cache entry.
    """
    decoded = _decode_hf_cache_model_id(path)
    if decoded is None:
        return None
    model_id, source_repo_id = decoded

    snapshots_dir = path / "snapshots"
    if not _is_readable_dir(
        snapshots_dir,
        "HF cache snapshots directory",
        health=health,
    ):
        return None

    for ref_name in ("main", "master"):
        try:
            commit_hash = (path / "refs" / ref_name).read_text().strip()
        except OSError:
            continue
        snapshot = snapshots_dir / commit_hash
        if _is_readable_dir(snapshot, "HF cache referenced snapshot", health=health):
            return HfCacheEntry(snapshot, model_id, source_repo_id)

    snapshots = [
        p
        for p in _iter_readable_entries(
            snapshots_dir,
            "HF cache snapshots",
            health=health,
        )
        if _is_readable_dir(p, "HF cache snapshot", health=health)
    ]
    if not snapshots:
        return None
    if len(snapshots) == 1:
        return HfCacheEntry(snapshots[0], model_id, source_repo_id)

    latest_snapshot: Path | None = None
    latest_mtime = float("-inf")
    for snapshot in snapshots:
        try:
            mtime = snapshot.stat().st_mtime
        except OSError as e:
            if health is not None:
                health.complete = False
            logger.warning(
                "Skipping inaccessible HF cache snapshot %s: %s: %s",
                snapshot,
                type(e).__name__,
                e,
            )
            continue
        if mtime > latest_mtime:
            latest_snapshot = snapshot
            latest_mtime = mtime
    if latest_snapshot is None:
        return None
    return HfCacheEntry(latest_snapshot, model_id, source_repo_id)


def _safetensors_has_mlx_metadata(
    path: Path,
    *,
    shards: list[Path] | None = None,
) -> bool:
    """Return True if any model safetensors shard declares MLX format."""
    try:
        from safetensors import safe_open
    except Exception as e:
        logger.debug(f"safetensors import failed while checking {path}: {e}")
        return False

    candidates = shards if shards is not None else list(path.glob("model*.safetensors"))
    for shard in sorted(candidates):
        try:
            with safe_open(str(shard), framework="numpy") as f:
                metadata = f.metadata() or {}
        except Exception as e:
            logger.debug(f"Could not read safetensors metadata from {shard}: {e}")
            continue
        if str(metadata.get("format", "")).lower() == "mlx":
            return True
    return False


_MLX_NAME_RE = re.compile(r"(^|[-_/])mlx($|[-_/])", re.IGNORECASE)

# Speculative-decoding helper checkpoints (e.g. z-lab/Qwen3.6-27B-DFlash) are
# MLX-loadable drafts even though their safetensors are saved in PyTorch ("pt")
# format and the repo name carries no "mlx" token, so the HF-cache MLX
# heuristic must recognise them explicitly or they vanish from the draft-model
# picker (#1643). Detection delegates to is_helper_model_config so the drafter
# markers stay in sync with helper flagging and engine_pool's drafter
# resolution.
def _is_helper_checkpoint(model_path: Path) -> bool:
    """True if ``model_path`` is a speculative-decoding helper checkpoint.

    Must never raise on an unreadable entry: unlike the config reads inside
    _register_model, this runs outside discover_models' per-model guard, so
    an escaped exception would abort the whole scan. UnicodeDecodeError from
    a non-UTF-8 config.json is a ValueError, not a JSONDecodeError.
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False
    return is_helper_model_config(config)


def _is_hf_cache_mlx_compatible(
    model_dir: Path,
    source_repo_id: str,
    *,
    health: _DiscoveryHealth | None = None,
) -> bool:
    """Heuristic for HF cache entries that can be loaded without conversion."""
    if not _is_model_dir(model_dir, health=health):
        return False
    shards = [
        entry
        for entry in _iter_readable_entries(
            model_dir,
            "HF cache snapshot",
            health=health,
        )
        if entry.name.startswith("model") and entry.suffix == ".safetensors"
    ]
    if not shards:
        logger.debug(
            f"Skipping HF cache model without model*.safetensors: {source_repo_id}"
        )
        return False
    if _safetensors_has_mlx_metadata(model_dir, shards=shards):
        return True

    repo_lower = source_repo_id.lower()
    if repo_lower.startswith("mlx-community/") or _MLX_NAME_RE.search(source_repo_id):
        logger.info(
            f"Treating HF cache model as MLX-compatible by repo name: {source_repo_id}"
        )
        return True

    if _is_helper_checkpoint(model_dir):
        logger.info(
            f"Treating HF cache model as MLX-compatible speculative-decoding "
            f"helper: {source_repo_id}"
        )
        return True

    logger.debug(f"Skipping non-MLX HF cache model: {source_repo_id}")
    return False


def _register_model(
    models: dict[str, DiscoveredModel],
    model_dir: Path,
    model_id: str,
    *,
    source_type: str = "local",
    source_repo_id: str | None = None,
) -> None:
    """Try to register a single model directory into the models dict."""
    if model_id in models:
        logger.warning(
            f"Duplicate model_id '{model_id}' found in {model_dir}, "
            f"keeping version from {models[model_id].model_path}"
        )
        return
    try:
        if _is_unsupported_model(model_dir):
            logger.info(f"Skipping unsupported model: {model_id}")
            return

        model_type = detect_model_type(model_dir)
        if model_type == "embedding":
            engine_type: EngineType = "embedding"
        elif model_type == "reranker":
            engine_type = "reranker"
        elif model_type == "vlm":
            engine_type = "vlm"
        elif model_type == "audio_stt":
            engine_type = "audio_stt"
        elif model_type == "audio_tts":
            engine_type = "audio_tts"
        elif model_type == "audio_sts":
            engine_type = "audio_sts"
        else:
            engine_type = "batched"
        estimated_size = estimate_model_size(model_dir)

        # Read raw config model_type for sub-type detection (e.g., OCR models)
        # and flag speculative-decoding drafters (dFlash/Assistant/MTP).
        config_model_type = ""
        is_helper = False
        try:
            import json
            with open(model_dir / "config.json") as f:
                _config = json.load(f)
            config_model_type = _config.get("model_type", "")
            is_helper = is_helper_model_config(_config)
        except Exception:
            pass

        thinking_default = detect_thinking_default(model_dir)
        preserve_thinking_default = detect_preserve_thinking(model_dir)
        model_context_length = _read_model_context_length(model_dir)

        models[model_id] = DiscoveredModel(
            model_id=model_id,
            model_path=str(model_dir),
            model_type=model_type,
            engine_type=engine_type,
            estimated_size=estimated_size,
            config_model_type=config_model_type,
            thinking_default=thinking_default,
            preserve_thinking_default=preserve_thinking_default,
            model_context_length=model_context_length,
            source_type=source_type,
            source_repo_id=source_repo_id,
            is_helper=is_helper,
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
    return discover_models_with_status(model_dir).models


def discover_models_with_status(
    model_dir: Path,
    *,
    hf_cache_entries_only: bool = False,
) -> ModelDiscoveryResult:
    """Discover models and report whether the root was scanned completely.

    Discovery remains best-effort for normal serving: unreadable entries and
    malformed/incomplete model directories are skipped. Callers making
    destructive persistence decisions can combine ``root_complete`` with
    ``root_model_entries`` so access failures defer cleanup while a skipped but
    structurally present folder protects only its own state.
    ``hf_cache_entries_only`` records cache-entry presence without loading
    model metadata; it protects persisted state while HF visibility is disabled.
    """
    root_key = _model_root_key(model_dir)
    access_error = model_directory_access_error(model_dir)
    if access_error is not None:
        if "not readable" in access_error:
            logger.warning("Skipping directory %s: %s", model_dir, access_error)
            return ModelDiscoveryResult(
                {},
                {root_key: False},
                {root_key: frozenset()},
                {root_key: frozenset()},
                {root_key: None},
            )
        raise ValueError(access_error)

    models: dict[str, DiscoveredModel] = {}
    model_entries: set[str] = set()
    root_has_model_artifact = False
    health = _DiscoveryHealth()
    start_fingerprint = _root_fingerprint(model_dir, health=health)

    for subdir in _iter_readable_entries(model_dir, "model directory", health=health):
        if subdir.name.startswith("."):
            continue
        if not _is_readable_dir(subdir, "model entry", health=health):
            root_has_model_artifact |= subdir.suffix in {".safetensors", ".bin"}
            continue

        if hf_cache_entries_only:
            hf_decoded = _decode_hf_cache_model_id(subdir)
            if hf_decoded is not None:
                model_entries.add(hf_decoded[0])
            continue

        if _is_adapter_dir(subdir, health=health):
            model_entries.add(subdir.name)
            logger.info(
                f"Skipping LoRA adapter: {subdir.name} "
                "(oMLX does not support LoRA/PEFT adapters)"
            )
        elif _is_model_dir(subdir, health=health):
            # Level 1: direct model folder
            model_entries.add(subdir.name)
            _register_model(models, subdir, subdir.name)
        else:
            # HF Hub cache entry: models--Org--Name/snapshots/<hash>/
            hf_decoded = _decode_hf_cache_model_id(subdir)
            if hf_decoded is not None:
                # The cache entry itself is still present even when it has no
                # resolvable snapshot yet. Keep this separate from successfully
                # discovered models so reconciliation can protect prior state
                # without exposing a non-loadable model.
                model_entries.add(hf_decoded[0])
                hf_resolved = _resolve_hf_cache_entry(subdir, health=health)
                if hf_resolved is None:
                    continue
                if _is_hf_cache_mlx_compatible(
                    hf_resolved.snapshot_path,
                    hf_resolved.source_repo_id,
                    health=health,
                ):
                    _register_model(
                        models,
                        hf_resolved.snapshot_path,
                        hf_resolved.model_id,
                        source_type="hf_cache",
                        source_repo_id=hf_resolved.source_repo_id,
                    )
                continue

            # Level 2: organization folder — scan children
            # Keep the top-level folder name as conservative structural
            # evidence too. It is harmless for ordinary organization folders,
            # and protects a formerly loadable direct model while its metadata
            # is temporarily absent.
            model_entries.add(subdir.name)
            has_children = False
            for child in _iter_readable_entries(subdir, "model group", health=health):
                if not _is_readable_dir(
                    child, "model group entry", health=health
                ) or child.name.startswith("."):
                    continue
                model_entries.add(child.name)
                if _is_adapter_dir(child, health=health):
                    logger.info(
                        f"Skipping LoRA adapter: {child.name} "
                        "(oMLX does not support LoRA/PEFT adapters)"
                    )
                elif _is_model_dir(child, health=health):
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
    root_is_model = (
        not hf_cache_entries_only
        and not models
        and _is_model_dir(model_dir, health=health)
    )
    if root_is_model:
        model_entries.add(model_dir.name)
        _register_model(models, model_dir, model_dir.name)
    elif not hf_cache_entries_only and not models and root_has_model_artifact:
        # A directly configured single-model root may temporarily lose its
        # config while weight files are still present.
        model_entries.add(model_dir.name)

    end_fingerprint = _root_fingerprint(model_dir, health=health)
    stable_fingerprint = start_fingerprint
    if start_fingerprint != end_fingerprint:
        health.complete = False
        stable_fingerprint = None
        logger.warning(
            "Model root identity changed during discovery; treating scan as "
            "incomplete: %s",
            model_dir,
        )

    return ModelDiscoveryResult(
        models,
        {root_key: health.complete},
        {root_key: frozenset(models)},
        {root_key: frozenset(set(models) | model_entries)},
        {root_key: stable_fingerprint},
    )


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
    return discover_models_from_dirs_with_status(model_dirs).models


def discover_models_from_dirs_with_status(
    model_dirs: list[Path],
) -> ModelDiscoveryResult:
    """Discover from multiple roots and retain per-root scan completeness."""
    merged: dict[str, DiscoveredModel] = {}
    root_complete: dict[Path, bool] = {}
    root_models: dict[Path, frozenset[str]] = {}
    root_model_entries: dict[Path, frozenset[str]] = {}
    root_fingerprints: dict[Path, RootFingerprint | None] = {}

    for model_dir in model_dirs:
        try:
            result = discover_models_with_status(model_dir)
        except ValueError as e:
            logger.warning(f"Skipping directory {model_dir}: {e}")
            root_key = _model_root_key(model_dir)
            root_complete[root_key] = False
            root_models[root_key] = frozenset()
            root_model_entries[root_key] = frozenset()
            root_fingerprints[root_key] = None
            continue

        root_complete.update(result.root_complete)
        root_models.update(result.root_models)
        root_model_entries.update(result.root_model_entries)
        root_fingerprints.update(result.root_fingerprints)

        for model_id, info in result.models.items():
            if model_id in merged:
                logger.warning(
                    f"Duplicate model_id '{model_id}' found in {model_dir}, "
                    f"keeping version from {merged[model_id].model_path}"
                )
                continue
            merged[model_id] = info

    return ModelDiscoveryResult(
        merged,
        root_complete,
        root_models,
        root_model_entries,
        root_fingerprints,
    )


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f}PB"
