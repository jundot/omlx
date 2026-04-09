# SPDX-License-Identifier: Apache-2.0
"""Tests for audio model discovery (INV-01).

Verifies that STT and TTS model types are correctly detected during model
discovery, and that previously UNSUPPORTED_MODEL_TYPES ("whisper", "qwen3_tts")
are now surfaced as audio types rather than silently skipped.

All tests run without mlx-audio installed — only config.json parsing is tested.
"""

import json
from pathlib import Path

import pytest

from omlx.model_discovery import (
    DiscoveredModel,
    _is_unsupported_model,
    detect_model_type,
    discover_models,
    estimate_model_size,
    AUDIO_STS_MODEL_TYPES,
    AUDIO_STS_ARCHITECTURES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(path: Path, config: dict) -> None:
    (path / "config.json").write_text(json.dumps(config))


def _write_weights(path: Path, size: int = 1000) -> None:
    (path / "model.safetensors").write_bytes(b"0" * size)


def _make_model(path: Path, config: dict, size: int = 1000) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_config(path, config)
    _write_weights(path, size)


# ---------------------------------------------------------------------------
# TestDetectAudioModelType
# ---------------------------------------------------------------------------


class TestDetectAudioModelType:
    """Tests that audio model types are detected correctly."""

    def test_whisper_architecture_returns_audio_stt(self, tmp_path):
        """WhisperForConditionalGeneration architecture -> audio_stt."""
        _write_config(tmp_path, {
            "model_type": "whisper",
            "architectures": ["WhisperForConditionalGeneration"],
        })
        assert detect_model_type(tmp_path) == "audio_stt"

    def test_qwen3_asr_architecture_returns_audio_stt(self, tmp_path):
        """Qwen3ASRForConditionalGeneration architecture -> audio_stt."""
        _write_config(tmp_path, {
            "model_type": "qwen3_asr",
            "architectures": ["Qwen3ASRForConditionalGeneration"],
        })
        assert detect_model_type(tmp_path) == "audio_stt"

    def test_whisper_model_type_returns_audio_stt(self, tmp_path):
        """model_type="whisper" without known STT architecture -> audio_stt."""
        _write_config(tmp_path, {
            "model_type": "whisper",
        })
        assert detect_model_type(tmp_path) == "audio_stt"

    def test_whisper_custom_architecture_returns_audio_stt(self, tmp_path):
        """Custom whisper architecture still detected by model_type."""
        _write_config(tmp_path, {
            "model_type": "whisper",
            "architectures": ["SomeCustomWhisperArch"],
        })
        assert detect_model_type(tmp_path) == "audio_stt"

    def test_qwen3_tts_model_type_returns_audio_tts(self, tmp_path):
        """model_type="qwen3_tts" -> audio_tts."""
        _write_config(tmp_path, {
            "model_type": "qwen3_tts",
            "architectures": ["Qwen3TTSForConditionalGeneration"],
        })
        assert detect_model_type(tmp_path) == "audio_tts"

    def test_kokoro_architecture_returns_audio_tts(self, tmp_path):
        """KokoroForConditionalGeneration architecture -> audio_tts."""
        _write_config(tmp_path, {
            "model_type": "kokoro",
            "architectures": ["KokoroForConditionalGeneration"],
        })
        assert detect_model_type(tmp_path) == "audio_tts"

    def test_qwen3_tts_no_architectures_returns_audio_tts(self, tmp_path):
        """qwen3_tts model_type alone is sufficient for audio_tts detection."""
        _write_config(tmp_path, {
            "model_type": "qwen3_tts",
        })
        assert detect_model_type(tmp_path) == "audio_tts"

    def test_vibevoice_model_type_returns_audio_tts(self, tmp_path):
        """model_type='vibevoice' (no architecture) -> audio_tts."""
        _write_config(tmp_path, {"model_type": "vibevoice"})
        assert detect_model_type(tmp_path) == "audio_tts"

    def test_vibevoice_streaming_architecture_returns_audio_tts(self, tmp_path):
        """VibeVoice streaming architecture -> audio_tts."""
        _write_config(tmp_path, {
            "model_type": "vibevoice_streaming",
            "architectures": ["VibeVoiceStreamingForConditionalGenerationInference"],
        })
        assert detect_model_type(tmp_path) == "audio_tts"

    def test_qwen2_audio_model_type_returns_audio_stt(self, tmp_path):
        """model_type='qwen2_audio' -> audio_stt."""
        _write_config(tmp_path, {"model_type": "qwen2_audio"})
        assert detect_model_type(tmp_path) == "audio_stt"

    def test_qwen2_audio_architecture_returns_audio_stt(self, tmp_path):
        """Qwen2AudioForConditionalGeneration architecture -> audio_stt."""
        _write_config(tmp_path, {
            "model_type": "qwen2_audio",
            "architectures": ["Qwen2AudioForConditionalGeneration"],
        })
        assert detect_model_type(tmp_path) == "audio_stt"

    def test_kugelaudio_model_type_returns_audio_tts(self, tmp_path):
        """model_type='kugelaudio' -> audio_tts."""
        _write_config(tmp_path, {"model_type": "kugelaudio"})
        assert detect_model_type(tmp_path) == "audio_tts"

    def test_kugelaudio_architecture_returns_audio_tts(self, tmp_path):
        """KugelAudioForConditionalGeneration architecture -> audio_tts."""
        _write_config(tmp_path, {
            "model_type": "kugelaudio",
            "architectures": ["KugelAudioForConditionalGeneration"],
        })
        assert detect_model_type(tmp_path) == "audio_tts"

    def test_audiodit_model_type_returns_audio_tts(self, tmp_path):
        """model_type='audiodit' (LongCat-AudioDiT) -> audio_tts."""
        _write_config(tmp_path, {"model_type": "audiodit"})
        assert detect_model_type(tmp_path) == "audio_tts"


# ---------------------------------------------------------------------------
# TestAudioNotUnsupported
# ---------------------------------------------------------------------------


class TestAudioNotUnsupported:
    """Audio models should no longer be flagged as unsupported."""

    def test_whisper_no_longer_unsupported(self, tmp_path):
        """After audio support, whisper should NOT be returned by _is_unsupported_model."""
        _write_config(tmp_path, {
            "model_type": "whisper",
            "architectures": ["WhisperForConditionalGeneration"],
        })
        # When audio is supported, _is_unsupported_model must return False
        assert _is_unsupported_model(tmp_path) is False

    def test_qwen3_tts_no_longer_unsupported(self, tmp_path):
        """After audio support, qwen3_tts should NOT be returned by _is_unsupported_model."""
        _write_config(tmp_path, {
            "model_type": "qwen3_tts",
        })
        assert _is_unsupported_model(tmp_path) is False


# ---------------------------------------------------------------------------
# TestNonAudioRegressions
# ---------------------------------------------------------------------------


class TestNonAudioRegressions:
    """Non-audio model types must not be affected by the audio changes."""

    def test_llm_still_detected(self, tmp_path):
        _write_config(tmp_path, {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
        assert detect_model_type(tmp_path) == "llm"

    def test_vlm_still_detected(self, tmp_path):
        _write_config(tmp_path, {
            "model_type": "qwen2_5_vl",
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        })
        assert detect_model_type(tmp_path) == "vlm"

    def test_embedding_still_detected(self, tmp_path):
        _write_config(tmp_path, {"model_type": "bert", "architectures": ["BertModel"]})
        assert detect_model_type(tmp_path) == "embedding"

    def test_reranker_still_detected(self, tmp_path):
        _write_config(tmp_path, {
            "model_type": "modernbert",
            "architectures": ["ModernBertForSequenceClassification"],
        })
        assert detect_model_type(tmp_path) == "reranker"

    def test_qwen2_causal_lm_not_audio(self, tmp_path):
        _write_config(tmp_path, {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
        result = detect_model_type(tmp_path)
        assert result not in ("audio_stt", "audio_tts")

    def test_missing_config_defaults_to_llm(self, tmp_path):
        assert detect_model_type(tmp_path) == "llm"

    def test_qwen2_causal_lm_not_sts(self, tmp_path):
        _write_config(tmp_path, {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
        result = detect_model_type(tmp_path)
        assert result != "audio_sts"

    def test_multimodal_with_nested_audio_config_not_audio(self, tmp_path):
        """MiniCPM-o has nested audio_config but is a VLM, not audio_stt/tts."""
        _write_config(tmp_path, {
            "model_type": "minicpmv",
            "architectures": ["MiniCPMO"],
            "vision_config": {"model_type": "siglip_vision_model"},
            "audio_config": {"model_type": "whisper"},
        })
        result = detect_model_type(tmp_path)
        assert result not in ("audio_stt", "audio_tts")


# ---------------------------------------------------------------------------
# TestDiscoverModelsIncludesAudio
# ---------------------------------------------------------------------------


class TestDiscoverModelsIncludesAudio:
    """discover_models() must return audio models with correct engine_type."""

    def test_discover_stt_model(self, tmp_path):
        """STT model included in discover_models results."""
        stt_dir = tmp_path / "whisper-large-v3"
        _make_model(stt_dir, {
            "model_type": "whisper",
            "architectures": ["WhisperForConditionalGeneration"],
        })

        models = discover_models(tmp_path)
        assert "whisper-large-v3" in models
        assert models["whisper-large-v3"].model_type == "audio_stt"

    def test_discover_stt_engine_type(self, tmp_path):
        """STT model has engine_type='stt' (or 'audio_stt')."""
        stt_dir = tmp_path / "whisper-small"
        _make_model(stt_dir, {"model_type": "whisper"})

        models = discover_models(tmp_path)
        assert "whisper-small" in models
        assert models["whisper-small"].engine_type in ("stt", "audio_stt")

    def test_discover_tts_model(self, tmp_path):
        """TTS model included in discover_models results."""
        tts_dir = tmp_path / "Qwen3-TTS"
        _make_model(tts_dir, {"model_type": "qwen3_tts"})

        models = discover_models(tmp_path)
        assert "Qwen3-TTS" in models
        assert models["Qwen3-TTS"].model_type == "audio_tts"

    def test_discover_tts_engine_type(self, tmp_path):
        """TTS model has engine_type='tts' (or 'audio_tts')."""
        tts_dir = tmp_path / "kokoro-tts"
        _make_model(tts_dir, {
            "model_type": "kokoro",
            "architectures": ["KokoroForConditionalGeneration"],
        })

        models = discover_models(tmp_path)
        assert "kokoro-tts" in models
        assert models["kokoro-tts"].engine_type in ("tts", "audio_tts")

    def test_discover_audio_and_llm_together(self, tmp_path):
        """Audio and LLM models coexist in the same discovery result."""
        llm_dir = tmp_path / "llama-3b"
        _make_model(llm_dir, {"model_type": "llama"})

        stt_dir = tmp_path / "whisper-tiny"
        _make_model(stt_dir, {"model_type": "whisper"})

        tts_dir = tmp_path / "qwen3-tts"
        _make_model(tts_dir, {"model_type": "qwen3_tts"})

        models = discover_models(tmp_path)
        assert len(models) == 3
        assert models["llama-3b"].model_type == "llm"
        assert models["whisper-tiny"].model_type == "audio_stt"
        assert models["qwen3-tts"].model_type == "audio_tts"

    def test_discover_stt_has_correct_fields(self, tmp_path):
        """DiscoveredModel fields are populated correctly for an STT model."""
        stt_dir = tmp_path / "whisper-base"
        _make_model(stt_dir, {"model_type": "whisper"}, size=2048)

        models = discover_models(tmp_path)
        model = models["whisper-base"]

        assert model.model_id == "whisper-base"
        assert model.model_path == str(stt_dir)
        assert model.model_type == "audio_stt"
        assert model.estimated_size == int(2048 * 1.05)


# ---------------------------------------------------------------------------
# TestEstimateModelSizeAudio
# ---------------------------------------------------------------------------


class TestEstimateModelSizeAudio:
    """estimate_model_size works for audio model directories."""

    def test_estimate_stt_safetensors(self, tmp_path):
        """Audio model with safetensors returns estimated size."""
        (tmp_path / "model.safetensors").write_bytes(b"0" * 4096)
        size = estimate_model_size(tmp_path)
        assert size == int(4096 * 1.05)

    def test_estimate_tts_multiple_safetensors(self, tmp_path):
        """Audio model with multiple safetensors shards."""
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"0" * 2000)
        (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"0" * 3000)
        size = estimate_model_size(tmp_path)
        assert size == int(5000 * 1.05)


# ---------------------------------------------------------------------------
# TestDetectSTSModelType
# ---------------------------------------------------------------------------


class TestDetectSTSModelType:
    """Tests that STS model types are detected correctly."""

    def test_sam_audio_model_type_returns_audio_sts(self, tmp_path):
        """model_type='sam_audio' -> audio_sts."""
        _write_config(tmp_path, {"model_type": "sam_audio"})
        assert detect_model_type(tmp_path) == "audio_sts"

    def test_sam_audio_architecture_returns_audio_sts(self, tmp_path):
        """SAMAudio architecture -> audio_sts."""
        _write_config(tmp_path, {
            "model_type": "sam_audio",
            "architectures": ["SAMAudio"],
        })
        assert detect_model_type(tmp_path) == "audio_sts"

    def test_deepfilternet_model_type_returns_audio_sts(self, tmp_path):
        """model_type='deepfilternet' -> audio_sts."""
        _write_config(tmp_path, {"model_type": "deepfilternet"})
        assert detect_model_type(tmp_path) == "audio_sts"

    def test_deepfilternet_architecture_returns_audio_sts(self, tmp_path):
        """DeepFilterNetModel architecture -> audio_sts."""
        _write_config(tmp_path, {
            "model_type": "deepfilternet",
            "architectures": ["DeepFilterNetModel"],
        })
        assert detect_model_type(tmp_path) == "audio_sts"

    def test_mossformer2_se_model_type_returns_audio_sts(self, tmp_path):
        """model_type='mossformer2_se' (mlx-audio dir name) -> audio_sts."""
        _write_config(tmp_path, {"model_type": "mossformer2_se"})
        assert detect_model_type(tmp_path) == "audio_sts"

    def test_mossformer2_architecture_returns_audio_sts(self, tmp_path):
        """MossFormer2SEModel architecture -> audio_sts (regardless of model_type)."""
        _write_config(tmp_path, {
            "model_type": "mossformer2",
            "architectures": ["MossFormer2SEModel"],
        })
        assert detect_model_type(tmp_path) == "audio_sts"

    def test_lfm2_audio_architecture_returns_audio_sts(self, tmp_path):
        """LFM2AudioModel architecture -> audio_sts."""
        _write_config(tmp_path, {
            "model_type": "lfm2_audio",
            "architectures": ["LFM2AudioModel"],
        })
        assert detect_model_type(tmp_path) == "audio_sts"

    def test_sts_model_types_set_contains_expected_entries(self):
        """AUDIO_STS_MODEL_TYPES contains the expected STS model families."""
        # When mlx-audio is installed these come from directory scanning;
        # names use underscores matching the mlx-audio module dirs.
        assert "deepfilternet" in AUDIO_STS_MODEL_TYPES
        assert "sam_audio" in AUDIO_STS_MODEL_TYPES

    def test_sts_architectures_set_contains_expected_entries(self):
        """AUDIO_STS_ARCHITECTURES contains the key architecture names."""
        assert "DeepFilterNetModel" in AUDIO_STS_ARCHITECTURES
        assert "MossFormer2SEModel" in AUDIO_STS_ARCHITECTURES
        assert "SAMAudio" in AUDIO_STS_ARCHITECTURES
        assert "LFM2AudioModel" in AUDIO_STS_ARCHITECTURES


# ---------------------------------------------------------------------------
# TestDiscoverModelsIncludesSTS
# ---------------------------------------------------------------------------


class TestDiscoverModelsIncludesSTS:
    """discover_models() must return STS models with correct engine_type."""

    def test_discover_sts_sam_audio_model(self, tmp_path):
        """SAMAudio STS model included in discover_models results."""
        sts_dir = tmp_path / "sam-audio-base"
        _make_model(sts_dir, {
            "model_type": "sam_audio",
            "architectures": ["SAMAudio"],
        })

        models = discover_models(tmp_path)
        assert "sam-audio-base" in models
        assert models["sam-audio-base"].model_type == "audio_sts"

    def test_discover_sts_engine_type(self, tmp_path):
        """STS model has engine_type='audio_sts'."""
        sts_dir = tmp_path / "deepfilternet3"
        _make_model(sts_dir, {"model_type": "deepfilternet"})

        models = discover_models(tmp_path)
        assert "deepfilternet3" in models
        assert models["deepfilternet3"].engine_type == "audio_sts"

    def test_discover_sts_deepfilternet_model(self, tmp_path):
        """DeepFilterNet STS model is discovered correctly."""
        sts_dir = tmp_path / "DeepFilterNet3"
        _make_model(sts_dir, {
            "model_type": "deepfilternet",
            "architectures": ["DeepFilterNetModel"],
        })

        models = discover_models(tmp_path)
        assert "DeepFilterNet3" in models
        assert models["DeepFilterNet3"].model_type == "audio_sts"

    def test_discover_sts_mossformer2_model(self, tmp_path):
        """MossFormer2 STS model is discovered correctly."""
        sts_dir = tmp_path / "mossformer2-se-48k"
        _make_model(sts_dir, {
            "model_type": "mossformer2",
            "architectures": ["MossFormer2SEModel"],
        })

        models = discover_models(tmp_path)
        assert "mossformer2-se-48k" in models
        assert models["mossformer2-se-48k"].model_type == "audio_sts"

    def test_discover_lfm2_audio_model(self, tmp_path):
        """LFM2 audio model (via LFM2AudioModel architecture) is discovered as audio_sts."""
        sts_dir = tmp_path / "LFM2.5-Audio-1B"
        _make_model(sts_dir, {
            "model_type": "lfm2_audio",
            "architectures": ["LFM2AudioModel"],
        })

        models = discover_models(tmp_path)
        assert "LFM2.5-Audio-1B" in models
        assert models["LFM2.5-Audio-1B"].model_type == "audio_sts"

    def test_discover_all_audio_types_together(self, tmp_path):
        """STT, TTS, and STS models coexist in the same discovery result."""
        _make_model(tmp_path / "llama-3b", {"model_type": "llama"})
        _make_model(tmp_path / "whisper-tiny", {"model_type": "whisper"})
        _make_model(tmp_path / "qwen3-tts", {"model_type": "qwen3_tts"})
        _make_model(tmp_path / "deepfilternet3", {"model_type": "deepfilternet"})

        models = discover_models(tmp_path)
        assert len(models) == 4
        assert models["llama-3b"].model_type == "llm"
        assert models["whisper-tiny"].model_type == "audio_stt"
        assert models["qwen3-tts"].model_type == "audio_tts"
        assert models["deepfilternet3"].model_type == "audio_sts"

    def test_discover_sts_has_correct_fields(self, tmp_path):
        """DiscoveredModel fields are populated correctly for an STS model."""
        sts_dir = tmp_path / "sam-audio-fp16"
        _make_model(sts_dir, {"model_type": "sam_audio"}, size=2048)

        models = discover_models(tmp_path)
        model = models["sam-audio-fp16"]

        assert model.model_id == "sam-audio-fp16"
        assert model.model_path == str(sts_dir)
        assert model.model_type == "audio_sts"
        assert model.engine_type == "audio_sts"
        assert model.estimated_size == int(2048 * 1.05)
