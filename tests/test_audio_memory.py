# SPDX-License-Identifier: Apache-2.0
"""Tests for audio engine memory tracking in EnginePool (INV-06).

Verifies that audio (STT/TTS) engines participate in the same LRU memory
management lifecycle as LLM/VLM/embedding engines:
  - Loading updates _current_model_memory
  - Unloading decrements _current_model_memory
  - last_access is updated on get_engine()
  - Audio engines are eligible for LRU eviction unless pinned
  - _find_lru_victim() can select an audio model
  - Pre-load eviction evicts audio when memory is tight

All tests run with mocked engines — mlx-audio is not required.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine_pool import EngineEntry, EnginePool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audio_model_dir(tmp_path):
    """Model directory with one LLM and one STT model (small sizes for fast tests)."""
    llm_dir = tmp_path / "llama-3b"
    llm_dir.mkdir()
    (llm_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (llm_dir / "model.safetensors").write_bytes(b"0" * 1024)  # ~1KB

    stt_dir = tmp_path / "whisper-tiny"
    stt_dir.mkdir()
    (stt_dir / "config.json").write_text(json.dumps({
        "model_type": "whisper",
        "architectures": ["WhisperForConditionalGeneration"],
    }))
    (stt_dir / "model.safetensors").write_bytes(b"0" * 2048)  # ~2KB

    tts_dir = tmp_path / "kokoro-tts"
    tts_dir.mkdir()
    (tts_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_tts"}))
    (tts_dir / "model.safetensors").write_bytes(b"0" * 1536)  # ~1.5KB

    return tmp_path


@pytest.fixture
def pool_with_audio(audio_model_dir):
    """EnginePool with audio + LLM models, generous memory limit."""
    pool = EnginePool(max_model_memory=10 * 1024**3)
    pool.discover_models(str(audio_model_dir))
    return pool


# ---------------------------------------------------------------------------
# TestAudioMemoryTracking
# ---------------------------------------------------------------------------


class TestAudioMemoryTracking:
    """Loading and unloading audio engines updates _current_model_memory."""

    @pytest.mark.asyncio
    async def test_loading_stt_updates_memory(self, pool_with_audio):
        """Loading an STT engine increments _current_model_memory."""
        pool = pool_with_audio
        assert pool.current_model_memory == 0

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()
        mock_engine.stop = AsyncMock()

        with patch("omlx.engine_pool.STTEngine", return_value=mock_engine, create=True):
            await pool.get_engine("whisper-tiny")

        assert pool.current_model_memory > 0

    @pytest.mark.asyncio
    async def test_loading_tts_updates_memory(self, pool_with_audio):
        """Loading a TTS engine increments _current_model_memory."""
        pool = pool_with_audio

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()
        mock_engine.stop = AsyncMock()

        with patch("omlx.engine_pool.TTSEngine", return_value=mock_engine, create=True):
            await pool.get_engine("kokoro-tts")

        assert pool.current_model_memory > 0

    @pytest.mark.asyncio
    async def test_unloading_stt_decrements_memory(self, pool_with_audio):
        """Unloading an STT engine decrements _current_model_memory."""
        pool = pool_with_audio

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()
        mock_engine.stop = AsyncMock()

        with patch("omlx.engine_pool.STTEngine", return_value=mock_engine, create=True):
            await pool.get_engine("whisper-tiny")
            memory_after_load = pool.current_model_memory
            assert memory_after_load > 0

            await pool._unload_engine("whisper-tiny")

        assert pool.current_model_memory < memory_after_load

    @pytest.mark.asyncio
    async def test_unload_clears_engine_reference(self, pool_with_audio):
        """After unload, EngineEntry.engine is None."""
        pool = pool_with_audio

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()
        mock_engine.stop = AsyncMock()

        with patch("omlx.engine_pool.STTEngine", return_value=mock_engine, create=True):
            await pool.get_engine("whisper-tiny")
            await pool._unload_engine("whisper-tiny")

        assert pool._entries["whisper-tiny"].engine is None


# ---------------------------------------------------------------------------
# TestAudioLastAccess
# ---------------------------------------------------------------------------


class TestAudioLastAccess:
    """last_access is updated when an audio engine is retrieved."""

    @pytest.mark.asyncio
    async def test_get_engine_updates_last_access(self, pool_with_audio):
        """get_engine() updates last_access timestamp for audio entry."""
        pool = pool_with_audio
        entry = pool._entries["whisper-tiny"]
        assert entry.last_access == 0.0

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()

        with patch("omlx.engine_pool.STTEngine", return_value=mock_engine, create=True):
            with patch("time.time", return_value=1234.0):
                await pool.get_engine("whisper-tiny")

        assert entry.last_access == 1234.0

    @pytest.mark.asyncio
    async def test_second_get_engine_refreshes_last_access(self, pool_with_audio):
        """Second call to get_engine() refreshes last_access."""
        pool = pool_with_audio

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()

        with patch("omlx.engine_pool.STTEngine", return_value=mock_engine, create=True):
            with patch("time.time", return_value=1000.0):
                await pool.get_engine("whisper-tiny")

            with patch("time.time", return_value=2000.0):
                await pool.get_engine("whisper-tiny")

        assert pool._entries["whisper-tiny"].last_access == 2000.0


# ---------------------------------------------------------------------------
# TestAudioLRUEviction
# ---------------------------------------------------------------------------


class TestAudioLRUEviction:
    """Audio engines are eligible for LRU eviction by default."""

    def test_audio_entry_not_pinned_by_default(self, pool_with_audio):
        """STT and TTS entries are not pinned by default."""
        pool = pool_with_audio
        assert pool._entries["whisper-tiny"].is_pinned is False
        assert pool._entries["kokoro-tts"].is_pinned is False

    def test_find_lru_victim_can_select_stt(self, pool_with_audio):
        """_find_lru_victim() returns STT model when it is the oldest loaded entry."""
        pool = pool_with_audio

        # Mark whisper-tiny as loaded and older than all others
        mock_engine = MagicMock()
        mock_engine.has_active_requests.return_value = False
        pool._entries["whisper-tiny"].engine = mock_engine
        pool._entries["whisper-tiny"].last_access = 10.0

        victim = pool._find_lru_victim()
        assert victim == "whisper-tiny"

    def test_find_lru_victim_selects_oldest_audio_over_newer_llm(self, pool_with_audio):
        """_find_lru_victim() picks the oldest entry regardless of model type."""
        pool = pool_with_audio

        mock_stt = MagicMock()
        mock_stt.has_active_requests.return_value = False
        pool._entries["whisper-tiny"].engine = mock_stt
        pool._entries["whisper-tiny"].last_access = 50.0  # Older

        mock_llm = MagicMock()
        mock_llm.has_active_requests.return_value = False
        pool._entries["llama-3b"].engine = mock_llm
        pool._entries["llama-3b"].last_access = 100.0  # Newer

        victim = pool._find_lru_victim()
        assert victim == "whisper-tiny"

    def test_pinned_audio_not_evicted(self, pool_with_audio):
        """Pinned audio engine is skipped by _find_lru_victim()."""
        pool = pool_with_audio

        mock_stt = MagicMock()
        mock_stt.has_active_requests.return_value = False
        pool._entries["whisper-tiny"].engine = mock_stt
        pool._entries["whisper-tiny"].last_access = 50.0
        pool._entries["whisper-tiny"].is_pinned = True  # pinned

        mock_llm = MagicMock()
        mock_llm.has_active_requests.return_value = False
        pool._entries["llama-3b"].engine = mock_llm
        pool._entries["llama-3b"].last_access = 100.0  # Newer but not pinned

        victim = pool._find_lru_victim()
        # whisper-tiny is pinned — llama-3b must be the victim
        assert victim == "llama-3b"


# ---------------------------------------------------------------------------
# TestAudioPinning
# ---------------------------------------------------------------------------


class TestAudioPinning:
    """Audio engines can be pinned to prevent eviction."""

    def test_discover_with_pinned_audio(self, audio_model_dir):
        """discover_models() with pinned_models pins the audio entry."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(audio_model_dir), pinned_models=["whisper-tiny"])

        assert pool._entries["whisper-tiny"].is_pinned is True
        assert pool._entries["llama-3b"].is_pinned is False

    def test_pinned_audio_not_selected_as_lru_victim(self, audio_model_dir):
        """Pinned audio model is excluded from LRU eviction candidates."""
        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(audio_model_dir), pinned_models=["whisper-tiny"])

        pool._entries["whisper-tiny"].engine = MagicMock()
        pool._entries["whisper-tiny"].last_access = 1.0  # Oldest

        pool._entries["llama-3b"].engine = MagicMock()
        pool._entries["llama-3b"].last_access = 99.0

        victim = pool._find_lru_victim()
        assert victim != "whisper-tiny"


# ---------------------------------------------------------------------------
# TestAudioPreLoadEviction
# ---------------------------------------------------------------------------


class TestAudioPreLoadEviction:
    """Pre-load eviction works when loading an audio model requires freeing memory."""

    @pytest.fixture
    def tight_audio_pool(self, tmp_path):
        """Pool tight enough that only one model fits at a time."""
        llm_dir = tmp_path / "llama-3b"
        llm_dir.mkdir()
        (llm_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (llm_dir / "model.safetensors").write_bytes(b"0" * 1024)

        stt_dir = tmp_path / "whisper-tiny"
        stt_dir.mkdir()
        (stt_dir / "config.json").write_text(json.dumps({
            "model_type": "whisper",
            "architectures": ["WhisperForConditionalGeneration"],
        }))
        (stt_dir / "model.safetensors").write_bytes(b"0" * 2048)

        # Limit: allow one model but not both simultaneously
        pool = EnginePool(max_model_memory=2500)
        pool.discover_models(str(tmp_path))
        return pool

    @pytest.mark.asyncio
    async def test_loading_stt_evicts_llm(self, tight_audio_pool):
        """When memory is tight, loading STT evicts the loaded LLM."""
        pool = tight_audio_pool

        mock_llm = MagicMock()
        mock_llm.start = AsyncMock()
        mock_llm.stop = AsyncMock()
        mock_llm.has_active_requests.return_value = False

        mock_stt = MagicMock()
        mock_stt.start = AsyncMock()
        mock_stt.stop = AsyncMock()
        mock_stt.has_active_requests.return_value = False

        with patch("omlx.engine_pool.BatchedEngine", return_value=mock_llm):
            await pool.get_engine("llama-3b")

        with patch("omlx.engine_pool.STTEngine", return_value=mock_stt, create=True):
            await pool.get_engine("whisper-tiny")

        # llama-3b should have been evicted
        mock_llm.stop.assert_called_once()
        assert pool._entries["llama-3b"].engine is None
        assert pool._entries["whisper-tiny"].engine is not None
