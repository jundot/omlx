# SPDX-License-Identifier: Apache-2.0
"""Tests for realtime STT and realtime voice: engine sessions and WebSockets.

All tests run with mocked models/engines — mlx-audio decode paths are not
exercised here (that needs a real model; see the slow smoke procedure).
"""

import asyncio
import base64
import io
import json
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlx_vlm.models.lfm3_audio import Model as Lfm3AudioModel
from mlx_vlm.models.lfm3_audio import ModelConfig as Lfm3AudioConfig
from mlx_vlm.models.lfm3_audio.codec import CodecConfig, LiquidCodec
from PIL import Image

from omlx.api.audio_routes import _realtime_api_key, realtime_router
from omlx.engine import lfm3_realtime
from omlx.engine.qwen3_omni_realtime import Qwen3OmniVoiceBackend
from omlx.engine.realtime_voice import (
    FunctionCall,
    RealtimeVoiceError,
    RealtimeVoiceSession,
    VoiceBackend,
    VoiceStep,
    ToolOutput,
    VoiceTurn,
    turn_messages,
)
from omlx.engine.stt import (
    RealtimeTranscriptionSession,
    STTEngine,
    _SegmentedRealtimeBackend,
    _split_audio_segments,
    _VoxtralRealtimeBackend,
    _WhisperRealtimeBackend,
)
from omlx.engine.vlm import _VOICE_BACKENDS, VLMBatchedEngine
from omlx.model_discovery import REALTIME_VOICE_MODEL_TYPES, is_realtime_stt_model


class TestRealtimeSttClassification:
    """Discovery-level realtime capability flag."""

    def test_whisper_is_realtime(self):
        assert is_realtime_stt_model("audio_stt", "whisper")

    def test_voxtral_realtime_is_realtime(self):
        assert is_realtime_stt_model("audio_stt", "voxtral_realtime")

    def test_whole_buffer_stt_families_are_realtime(self):
        assert is_realtime_stt_model("audio_stt", "qwen3_asr")
        assert is_realtime_stt_model("audio_stt", "parakeet")

    def test_non_stt_types_are_not_realtime(self):
        assert not is_realtime_stt_model("llm", "whisper")
        assert not is_realtime_stt_model("audio_tts", "whisper")


class TestSupportsRealtimeStt:
    """Engine-level capability introspection on the loaded model."""

    def test_unloaded_engine_not_supported(self):
        assert not STTEngine("m").supports_realtime_stt()

    def test_whisper_like_model_supported(self):
        engine = STTEngine("m")
        engine._model = SimpleNamespace(generate_streaming=lambda: None)
        assert engine.supports_realtime_stt()

    def test_voxtral_like_model_supported(self):
        engine = STTEngine("m")
        engine._model = SimpleNamespace(create_streaming_session=lambda: None)
        assert engine.supports_realtime_stt()

    def test_plain_stt_model_supported(self):
        engine = STTEngine("m")
        engine._model = SimpleNamespace(generate=lambda: None)
        assert engine.supports_realtime_stt()


class TestRealtimeSessionLifecycle:
    @pytest.mark.asyncio
    async def test_second_session_rejected_until_release(self):
        engine = STTEngine("m")
        engine._model = SimpleNamespace(generate_streaming=lambda: None)

        session = await engine.create_realtime_session()
        with pytest.raises(RuntimeError, match="already active"):
            await engine.create_realtime_session()

        await session.release()
        session2 = await engine.create_realtime_session()
        await session2.release()

    @pytest.mark.asyncio
    async def test_unsupported_model_raises(self):
        engine = STTEngine("m")
        engine._model = SimpleNamespace()
        with pytest.raises(RuntimeError, match="realtime"):
            await engine.create_realtime_session()

    @pytest.mark.asyncio
    async def test_unstarted_engine_raises(self):
        with pytest.raises(RuntimeError, match="not started"):
            await STTEngine("m").create_realtime_session()


class TestPcm16Conversion:
    def test_feed_pcm16_converts_to_normalized_float32(self):
        received = []
        backend = SimpleNamespace(feed=received.append)
        session = RealtimeTranscriptionSession(
            MagicMock(spec=STTEngine), backend, "act"
        )

        pcm = struct.pack("<5h", 0, 16384, -16384, 32767, -32768)
        session.feed_pcm16(pcm)

        assert len(received) == 1
        arr = received[0]
        assert arr.dtype == np.float32
        assert arr[0] == 0.0
        assert abs(arr[1] - 0.5) < 1e-3
        assert abs(arr[2] + 0.5) < 1e-3
        assert arr[4] == -1.0

    def test_feed_pcm16_ignores_empty_payload(self):
        received = []
        backend = SimpleNamespace(feed=received.append)
        session = RealtimeTranscriptionSession(
            MagicMock(spec=STTEngine), backend, "act"
        )
        session.feed_pcm16(b"")
        assert received == []


class TestWhisperBackendBuffering:
    def test_take_respects_chunk_threshold(self):
        backend = _WhisperRealtimeBackend(model=None)
        backend.feed(np.zeros(1000, dtype=np.float32))
        assert backend._take(16000) is None

        backend.feed(np.zeros(15000, dtype=np.float32))
        merged = backend._take(16000)
        assert merged is not None
        assert len(merged) == 16000

        # Buffer drained after take
        assert backend._take(1) is None


class _FakeVoxtralSession:
    def __init__(self):
        self.fed = []
        self.closed = False
        self._deltas = ["hello ", "world"]
        self._emitted = 0

    @property
    def done(self):
        return self.closed and self._emitted >= len(self._deltas)

    def feed(self, samples):
        self.fed.append(samples)

    def close(self):
        self.closed = True

    def step(self, max_decode_tokens=4):
        if self._emitted < len(self._deltas):
            delta = self._deltas[self._emitted]
            self._emitted += 1
            return [delta]
        return []


class TestVoxtralBackend:
    def test_feed_poll_close_flow(self):
        model = SimpleNamespace(create_streaming_session=_FakeVoxtralSession)
        backend = _VoxtralRealtimeBackend(model)
        backend.start_sync()

        backend.feed(np.zeros(10, dtype=np.float32))
        deltas = backend.poll_sync()
        rest = backend.close_sync()

        assert "".join(deltas + rest) == "hello world"

    def test_unstarted_backend_is_inert(self):
        backend = _VoxtralRealtimeBackend(SimpleNamespace())
        backend.feed(np.zeros(4, dtype=np.float32))
        assert backend.poll_sync() == []
        assert backend.close_sync() == []


def _tone(seconds: float, amplitude: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * 16000)) / 16000
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16000), dtype=np.float32)


class _FakeWholeBufferModel:
    def __init__(self):
        self.seconds: list[float] = []

    def generate(self, audio, **kwargs):
        self.seconds.append(len(audio) / 16000)
        return SimpleNamespace(text=f" part{len(self.seconds)} ")


class TestSegmentedRealtimeBackend:
    def test_pause_commits_utterance_and_joins_with_space(self):
        model = _FakeWholeBufferModel()
        backend = _SegmentedRealtimeBackend(model)

        backend.feed(np.concatenate([_tone(1.0), _silence(1.0)]))
        first = backend.poll_sync()
        backend.feed(np.concatenate([_tone(1.0), _silence(1.0)]))
        second = backend.poll_sync()

        assert first == ["part1"]
        assert second == [" part2"]
        assert backend.close_sync() == []

    def test_silence_and_clicks_dropped_open_utterance_flushed_on_close(self):
        model = _FakeWholeBufferModel()
        backend = _SegmentedRealtimeBackend(model)

        backend.feed(np.concatenate([_silence(5.0), _tone(0.1), _silence(1.0)]))
        assert backend.poll_sync() == []
        assert model.seconds == []

        backend.feed(_tone(1.0))
        assert backend.poll_sync() == []
        assert backend.close_sync() == ["part1"]
        # Only the lead-in survives the dropped silence and click.
        assert model.seconds == [pytest.approx(1.3)]

    def test_speech_without_pause_cut_at_quiet_point(self):
        model = _FakeWholeBufferModel()
        backend = _SegmentedRealtimeBackend(model)

        backend.feed(np.concatenate([_tone(27.0), _tone(0.1, 0.0), _tone(8.0)]))
        assert backend.poll_sync() == ["part1"]
        assert backend.close_sync() == [" part2"]
        assert 26.9 < model.seconds[0] < 27.2


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


class _StubRealtimeSession:
    """Engine-session stand-in with scripted poll results."""

    def __init__(self, polls=None, final=None):
        self.fed = b""
        self.released = False
        self._polls = list(polls or [])
        self._final = list(final or [])

    def feed_pcm16(self, data):
        self.fed += data

    async def poll(self):
        if self._polls:
            return self._polls.pop(0)
        await asyncio.sleep(0)
        return []

    async def close(self):
        return list(self._final)

    async def release(self):
        self.released = True


def _make_realtime_engine(stub):
    engine = MagicMock(spec=STTEngine)
    engine.supports_realtime_stt.return_value = True
    engine.create_realtime_session = AsyncMock(return_value=stub)
    return engine


def _ws_client(engine):
    app = FastAPI()
    app.include_router(realtime_router)

    pool = MagicMock()
    pool.get_engine = AsyncMock(return_value=engine)
    return (
        patch("omlx.api.audio_routes._get_engine_pool", return_value=pool),
        patch("omlx.api.audio_routes._verify_ws_api_key", return_value=True),
        patch("omlx.api.audio_routes._resolve_model", side_effect=lambda m: m),
        TestClient(app),
    )


WS_PATH = "/v1/audio/transcriptions/realtime"


class TestRealtimeWebSocket:
    def test_happy_path_delta_stop_done(self):
        stub = _StubRealtimeSession(polls=[["hello "]], final=["world"])
        engine = _make_realtime_engine(stub)
        p1, p2, p3, client = _ws_client(engine)
        with p1, p2, p3, client, client.websocket_connect(WS_PATH) as ws:
            ws.send_json({"type": "start", "model": "whisper-tiny", "api_key": "k"})
            assert ws.receive_json() == {"type": "ready"}

            ws.send_bytes(b"\x00\x00" * 1600)
            assert ws.receive_json() == {
                "type": "transcript.delta",
                "delta": "hello ",
            }

            ws.send_json({"type": "stop"})
            assert ws.receive_json() == {
                "type": "transcript.delta",
                "delta": "world",
            }
            done = ws.receive_json()
            assert done["type"] == "transcript.done"
            assert done["text"] == "hello world"

        assert stub.released
        assert len(stub.fed) == 3200

    def test_invalid_api_key_rejected(self):
        stub = _StubRealtimeSession()
        engine = _make_realtime_engine(stub)
        p1, _, p3, client = _ws_client(engine)
        with (
            p1,
            patch("omlx.api.audio_routes._verify_ws_api_key", return_value=False),
            p3,
            client,
            client.websocket_connect(WS_PATH) as ws,
        ):
            ws.send_json({"type": "start", "model": "m", "api_key": "bad"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "API key" in msg["detail"]
        assert not stub.released

    def test_non_stt_engine_rejected(self):
        engine = MagicMock()  # not an STTEngine
        p1, p2, p3, client = _ws_client(engine)
        with p1, p2, p3, client, client.websocket_connect(WS_PATH) as ws:
            ws.send_json({"type": "start", "model": "llama", "api_key": "k"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "not a speech-to-text" in msg["detail"]

    def test_non_realtime_stt_rejected(self):
        engine = MagicMock(spec=STTEngine)
        engine.supports_realtime_stt.return_value = False
        p1, p2, p3, client = _ws_client(engine)
        with p1, p2, p3, client, client.websocket_connect(WS_PATH) as ws:
            ws.send_json({"type": "start", "model": "qwen3-asr", "api_key": "k"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "does not support realtime" in msg["detail"]

    def test_busy_engine_rejected(self):
        engine = MagicMock(spec=STTEngine)
        engine.supports_realtime_stt.return_value = True
        engine.create_realtime_session = AsyncMock(
            side_effect=RuntimeError(
                "A realtime transcription session is already active"
            )
        )
        p1, p2, p3, client = _ws_client(engine)
        with p1, p2, p3, client, client.websocket_connect(WS_PATH) as ws:
            ws.send_json({"type": "start", "model": "whisper", "api_key": "k"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "already active" in msg["detail"]

    def test_missing_model_rejected(self):
        engine = _make_realtime_engine(_StubRealtimeSession())
        p1, p2, p3, client = _ws_client(engine)
        with p1, p2, p3, client, client.websocket_connect(WS_PATH) as ws:
            ws.send_json({"type": "start", "api_key": "k"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "model" in msg["detail"].lower()

    def test_bad_first_message_rejected(self):
        engine = _make_realtime_engine(_StubRealtimeSession())
        p1, p2, p3, client = _ws_client(engine)
        with p1, p2, p3, client, client.websocket_connect(WS_PATH) as ws:
            ws.send_json({"type": "hello"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    def test_client_disconnect_releases_session(self):
        stub = _StubRealtimeSession()
        engine = _make_realtime_engine(stub)
        p1, p2, p3, client = _ws_client(engine)
        with p1, p2, p3, client, client.websocket_connect(WS_PATH) as ws:
            ws.send_json({"type": "start", "model": "whisper", "api_key": "k"})
            assert ws.receive_json() == {"type": "ready"}
            # Context-manager exit closes the socket without a stop message
        assert stub.released


# ---------------------------------------------------------------------------
# Long-audio handling (segmented file streaming + realtime window rotation)
# ---------------------------------------------------------------------------


SR = 16000


class TestSplitAudioSegments:
    def test_short_audio_single_segment(self):
        samples = np.ones(10 * SR, dtype=np.float32)
        segments = _split_audio_segments(samples, SR, segment_seconds=30.0)
        assert len(segments) == 1
        assert len(segments[0]) == len(samples)

    def test_segments_reconstruct_input(self):
        rng = np.random.default_rng(0)
        samples = rng.standard_normal(95 * SR).astype(np.float32)
        segments = _split_audio_segments(samples, SR, segment_seconds=30.0)
        assert len(segments) >= 3
        assert all(len(s) <= 30 * SR for s in segments)
        np.testing.assert_array_equal(np.concatenate(segments), samples)

    def test_cut_lands_in_quiet_zone(self):
        # Loud signal with a silent gap at 27..28s: the 30 s boundary should
        # move back into the gap.
        samples = np.ones(40 * SR, dtype=np.float32)
        samples[27 * SR : 28 * SR] = 0.0
        segments = _split_audio_segments(
            samples, SR, segment_seconds=30.0, search_seconds=5.0
        )
        cut = len(segments[0])
        assert 27 * SR <= cut <= 28 * SR

    def test_tiny_tail_merged_into_previous(self):
        samples = np.ones(int(30.02 * SR), dtype=np.float32)
        segments = _split_audio_segments(
            samples, SR, segment_seconds=30.0, search_seconds=0.0
        )
        assert sum(len(s) for s in segments) == len(samples)
        assert all(len(s) >= int(0.1 * SR) for s in segments)


class _FakeAlignAttDecoder:
    """Scripted StreamingDecoder stand-in for window/delta logic tests."""

    def __init__(self):
        self.reset_calls = 0
        self.tokenizer = SimpleNamespace(decode=lambda toks: "".join(toks))

    def decode_chunk(self, mel, is_last=False):
        raise AssertionError("tests drive _emit_stable_delta directly")

    def reset(self):
        self.reset_calls += 1


class TestWhisperEmitStableDelta:
    """_emit_stable_delta receives the decoder's FULL window hypothesis."""

    def _backend(self):
        backend = _WhisperRealtimeBackend(model=None)
        backend._decoder = _FakeAlignAttDecoder()
        return backend

    def test_deltas_are_append_only(self):
        backend = self._backend()
        assert backend._emit_stable_delta(["hello"], is_last=False) == "hello"
        assert (
            backend._emit_stable_delta(["hello", " world"], is_last=False) == " world"
        )
        # Unchanged hypothesis emits nothing
        assert backend._emit_stable_delta(["hello", " world"], is_last=False) == ""
        assert backend._emit_stable_delta([], is_last=False) == ""

    def test_trailing_incomplete_char_withheld_until_complete(self):
        backend = self._backend()
        # Hypothesis ends in a replacement char (partial CJK bytes)
        assert backend._emit_stable_delta(["a", "�"], is_last=False) == "a"
        # Completing token resolves the pending char; only now it is emitted
        backend._decoder.tokenizer = SimpleNamespace(decode=lambda toks: "a한")
        assert backend._emit_stable_delta(["a", "한"], is_last=False) == "한"

    def test_replacement_chars_stripped_from_emitted_delta(self):
        backend = self._backend()
        # Interior U+FFFD never reaches the client; bookkeeping keeps the
        # raw decode so later diffs stay consistent.
        assert backend._emit_stable_delta(["a", "�", "b"], is_last=False) == "ab"
        assert backend._window_text == "a�b"

    def test_is_last_strips_trailing_replacement_char(self):
        backend = self._backend()
        assert backend._emit_stable_delta(["a", "�"], is_last=True) == "a"

    def test_upstream_duplicated_slice_does_not_duplicate_output(self):
        # The decoder's per-call result.tokens can contain re-emitted old
        # content, but the full hypothesis does not — diffing against the
        # hypothesis must emit only the true continuation.
        backend = self._backend()
        first = "we can create a report and today we focus on"
        assert backend._emit_stable_delta([first], is_last=False) == first
        full = first + " reports."
        assert backend._emit_stable_delta([full], is_last=True) == " reports."

    def test_revision_aligned_by_suffix_overlap(self):
        backend = self._backend()
        first = "welcome everyone please log in"
        assert backend._emit_stable_delta([first], is_last=False) == first
        # Full-window re-decode revised the beginning but continues the tail
        revised = "Welcome, everyone please log in with the shared account"
        delta = backend._emit_stable_delta([revised], is_last=True)
        assert delta == " with the shared account"
        assert backend._window_text == first + delta

    def test_unalignable_revision_dropped_mid_window(self):
        backend = self._backend()
        assert backend._emit_stable_delta(["hello world"], is_last=False)
        # Hypothesis still in flux: dropping avoids duplicated spam
        assert (
            backend._emit_stable_delta(["completely different text"], is_last=False)
            == ""
        )

    def test_unalignable_revision_recovered_at_flush(self):
        backend = self._backend()
        assert backend._emit_stable_delta(["hello world"], is_last=False)
        # Closing flush emits the settled hypothesis past the common
        # prefix — bounded echo beats losing the revised region.
        assert (
            backend._emit_stable_delta(["hello brave new world"], is_last=True)
            == "brave new world"
        )


class TestWhisperWindowRotation:
    def _backend_with_stub_decode(self):
        backend = _WhisperRealtimeBackend(model=None)
        backend._decoder = _FakeAlignAttDecoder()
        calls = []

        def _decode(samples, is_last):
            calls.append((len(samples), is_last))
            if is_last:
                backend._window_tokens = []
                backend._window_text = ""
            return "txt"

        backend._decode = _decode
        return backend, calls

    def test_rotates_before_mel_cap(self):
        backend, calls = self._backend_with_stub_decode()
        loud = np.ones(SR, dtype=np.float32) * 0.5
        for _ in range(40):
            backend.feed(loud.copy())
            backend.poll_sync()
        # Every chunk under 28 s stays in-window; the chunk that would cross
        # the cap closes the window (is_last=True) and resets counters.
        assert any(is_last for _, is_last in calls)
        first_rotation = next(i for i, (_, il) in enumerate(calls) if il)
        assert first_rotation <= 28
        assert backend._window_samples < 28 * SR

    def test_quiet_chunk_triggers_early_rotation(self):
        backend, calls = self._backend_with_stub_decode()
        loud = np.ones(SR, dtype=np.float32) * 0.5
        quiet = np.zeros(SR, dtype=np.float32)
        for _ in range(23):
            backend.feed(loud.copy())
            backend.poll_sync()
        assert not any(is_last for _, is_last in calls)
        backend.feed(quiet)
        backend.poll_sync()
        assert calls[-1][1] is True
        assert backend._window_samples == 0

    def test_first_decode_waits_for_larger_chunk(self):
        backend = _WhisperRealtimeBackend(model=None)
        backend.feed(np.ones(SR, dtype=np.float32))
        # 1 s buffered but no decoder yet: below FIRST_CHUNK_SECONDS
        assert backend.poll_sync() == []
        backend.feed(np.ones(SR, dtype=np.float32))
        backend._decode = lambda samples, is_last: f"got {len(samples)}"
        assert backend.poll_sync() == [f"got {2 * SR}"]

    def test_rotation_redetects_language_unless_pinned(self):
        backend = _WhisperRealtimeBackend(model=None)
        backend._decoder = _FakeAlignAttDecoder()
        backend._language = "ko"  # auto-detected earlier
        backend._rotate_window()
        assert backend._decoder is None
        assert backend._language is None

        pinned = _WhisperRealtimeBackend(model=None, language="ko")
        pinned._decoder = _FakeAlignAttDecoder()
        pinned._rotate_window()
        assert pinned._decoder is not None
        assert pinned._decoder.reset_calls == 1
        assert pinned._language == "ko"


class TestSegmentedTranscribeStream:
    @pytest.mark.asyncio
    async def test_whisper_streams_per_segment_with_autodetect(self, monkeypatch):
        import omlx.engine.stt as stt_mod

        decoded = []

        class FakeWhisperModel:
            def generate_streaming(self):
                pass

            def generate(self, segment, **kwargs):
                decoded.append((len(segment), kwargs))
                idx = len(decoded)
                return SimpleNamespace(
                    text=f" segment {idx}",
                    language="ko" if idx % 2 else "en",
                    segments=[],
                )

        samples = np.random.default_rng(1).standard_normal(65 * SR)
        monkeypatch.setattr(
            stt_mod,
            "_load_audio_samples",
            lambda path: samples.astype(np.float32),
        )

        engine = STTEngine("whisper-test")
        engine._model = FakeWhisperModel()

        chunks = [c async for c in engine.transcribe_stream("/fake.wav")]

        assert len(chunks) == len(decoded) >= 3
        assert sum(n for n, _ in decoded) == len(samples)
        # No language pin: each segment auto-detects (language not forced)
        assert all("language" not in kw for _, kw in decoded)
        assert [c["language"] for c in chunks[:2]] == ["ko", "en"]
        assert "".join(c["text"] for c in chunks).startswith(" segment 1 segment 2")


class _FakeQwen3Model:
    """Fake matching the qwen3_asr contract driven by the token-id path."""

    sample_rate = 16000
    config = SimpleNamespace(support_languages=["Korean", "English"])

    def __init__(self, token_script, decode_fn):
        self._script = token_script
        self._tokenizer = SimpleNamespace(decode=decode_fn)
        self.calls = []

    def extract_language(self, text):
        if "<asr_text>" in text and text.startswith("language "):
            return (
                text[len("language ") : text.find("<asr_text>")].strip(),
                text[text.find("<asr_text>") + len("<asr_text>") :],
            )
        return "English", text

    def stream_generate(self, audio, **kwargs):
        self.calls.append(kwargs)
        for t in self._script:
            yield t, None


def _decode_with_split_char(ids):
    """Tokens 3+4 form one Hangul char; either alone decodes broken."""
    out = []
    i = 0
    table = {1: "language Korean<asr_text>", 2: "안", 4: "�", 5: " 하세요"}
    while i < len(ids):
        if ids[i] == 3:
            if i + 1 < len(ids) and ids[i + 1] == 4:
                out.append("녕")
                i += 2
                continue
            out.append("�")
            i += 1
            continue
        out.append(table[ids[i]])
        i += 1
    return "".join(out)


class TestQwen3TokenIdStreaming:
    """transcribe_stream drives Qwen3-ASR via raw token ids (CJK-safe)."""

    @pytest.mark.asyncio
    async def test_cjk_chars_never_split(self, monkeypatch):
        import omlx.engine.stt as stt_mod

        monkeypatch.setattr(
            stt_mod,
            "_load_audio_samples",
            lambda path: np.zeros(3 * SR, dtype=np.float32),
        )
        model = _FakeQwen3Model([1, 2, 3, 4, 5], _decode_with_split_char)
        engine = STTEngine("qwen3-test")
        engine._model = model

        chunks = [c async for c in engine.transcribe_stream("/fake.wav")]
        texts = [c["text"] for c in chunks if c["text"]]

        assert "".join(texts) == "안녕 하세요"
        assert all("�" not in t for t in texts)
        # Auto-detect prefix never leaks into the transcript
        assert all("<asr_text>" not in t and "language" not in t for t in texts)
        assert chunks[0]["language"] == "Korean"
        # Cumulative token count reported for the usage line
        assert chunks[-1]["generation_tokens"] == 5

    @pytest.mark.asyncio
    async def test_language_hint_normalized_and_forwarded(self, monkeypatch):
        import omlx.engine.stt as stt_mod

        monkeypatch.setattr(
            stt_mod,
            "_load_audio_samples",
            lambda path: np.zeros(3 * SR, dtype=np.float32),
        )
        table = {2: "안", 5: " 하세요"}
        model = _FakeQwen3Model([2, 5], lambda ids: "".join(table[i] for i in ids))
        engine = STTEngine("qwen3-test")
        engine._model = model

        chunks = [c async for c in engine.transcribe_stream("/fake.wav", language="ko")]
        # ISO hint mapped to the full name qwen3 expects
        assert model.calls[0]["language"] == "korean"
        assert "".join(c["text"] for c in chunks) == "안 하세요"


class TestSegmentedTranscribeStreamLanguage:
    @pytest.mark.asyncio
    async def test_explicit_language_forwarded(self, monkeypatch):
        import omlx.engine.stt as stt_mod

        seen = {}

        class FakeWhisperModel:
            def generate_streaming(self):
                pass

            def generate(self, segment, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(text=" ok", language="ko", segments=[])

        monkeypatch.setattr(
            stt_mod,
            "_load_audio_samples",
            lambda path: np.zeros(5 * SR, dtype=np.float32),
        )
        engine = STTEngine("whisper-test")
        engine._model = FakeWhisperModel()

        chunks = [c async for c in engine.transcribe_stream("/fake.wav", language="ko")]
        assert seen["language"] == "ko"
        assert chunks[0]["text"] == " ok"


# ---------------------------------------------------------------------------
# Realtime voice WebSocket (/v1/realtime)
# ---------------------------------------------------------------------------


class _StubVoiceBackend(VoiceBackend):
    """Backend stand-in: each reply is ``steps`` chunks of text and audio."""

    voices = ("chelsie", "ethan")
    default_instructions = "Speak briefly."
    supports_tools = True
    input_image = True

    def __init__(self, steps=2, delay=0.0, fail=None, calls=()):
        self.steps, self.delay, self.fail = steps, delay, fail
        # Function calls that end each reply, one entry per reply.
        self.calls = list(calls)
        self.turns, self.inputs, self.truncated = [], [], []
        self.stopped, self.closed, self._left = 0, False, 0

    def begin_turn(self, turn, instructions, voice, tools):
        if self.fail is not None:
            raise self.fail
        audio = 0 if turn.audio is None else len(turn.audio)
        self.turns.append((audio, instructions, voice))
        self.inputs.append((turn, tools))
        self._left = self.steps
        return 7

    def step(self, seconds, cancel):
        time.sleep(self.delay)
        self._left -= 1
        audio = np.full(240, 0.5, np.float32)
        step = VoiceStep(text="hi ", audio=audio, slots=3, done=self._left <= 0)
        if step.done and self.calls:
            step.function_calls = self.calls.pop(0)
        return step

    def stop(self):
        self.stopped += 1

    def truncate(self, played_seconds):
        self.truncated.append(played_seconds)

    def close(self):
        self.closed = True


def _voice_engine(backend, executor):
    engine = MagicMock(spec=VLMBatchedEngine)
    engine._engine = SimpleNamespace(_mlx_executor=executor)
    engine._speech_active = 0
    engine.supports_realtime_voice = True
    session = RealtimeVoiceSession(engine, backend)
    engine.create_realtime_session = AsyncMock(return_value=session)
    return engine


def _voice_client(engine, key_ok=True):
    app = FastAPI()
    app.include_router(realtime_router)
    pool = MagicMock()
    pool.get_engine = AsyncMock(return_value=engine)
    pool.release_engine = AsyncMock()
    pool.get_abort_requested_reason = MagicMock(return_value=None)
    patches = (
        patch("omlx.api.audio_routes._get_engine_pool", return_value=pool),
        patch("omlx.api.audio_routes._verify_ws_api_key", return_value=key_ok),
        patch("omlx.api.audio_routes._resolve_model", side_effect=lambda m: m),
    )
    return patches, pool, TestClient(app)


def _pcm(seconds):
    samples = np.zeros(int(24000 * seconds), dtype="<i2").tobytes()
    return base64.b64encode(samples).decode()


def _png_url():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_turn_puts_each_tool_screenshot_in_its_own_response():
    shot, page = ToolOutput('{"ok": true}', ["shot"]), ToolOutput("page")
    tool_turn = VoiceTurn(tool_outputs=[shot, page])
    assert turn_messages(tool_turn) == [
        {
            "role": "tool",
            "content": [{"type": "text", "text": '{"ok": true}'}, {"type": "image"}],
        },
        {"role": "tool", "content": "page"},
    ]
    spoken = VoiceTurn(audio=np.zeros(1), images=["screen"], tool_outputs=[shot])
    assert turn_messages(spoken)[-1] == {
        "role": "user",
        "content": [{"type": "image"}, {"type": "audio"}],
    }
    assert spoken.all_images == ["shot", "screen"]


class _TrimCache:
    def __init__(self):
        self.trimmed = 0

    def trim(self, count):
        self.trimmed += count


def test_qwen_stop_while_writing_a_call_keeps_only_the_speech():
    backend = object.__new__(Qwen3OmniVoiceBackend)
    backend.cache = [_TrimCache(), _TrimCache()]
    backend.offset = backend.position = 100
    # Two spoken tokens, then <tool_call> (9) and call text; the last token is unfed.
    backend.reply, backend.speech_end, backend.unfed = [1, 2, 9, 3, 4], 2, 4
    backend.im_end, backend.thinker_done, backend.done = 0, False, False
    backend.stop()
    assert backend.reply == [1, 2] and backend.pending == [0] and backend.done
    assert backend.cache[0].trimmed == 2 and backend.offset == backend.position == 98


def _eventually(check, timeout=2.0):
    # The server finishes cleanup after the test client closes the socket.
    deadline = time.monotonic() + timeout
    while not check() and time.monotonic() < deadline:
        time.sleep(0.01)
    return check()


def _until(ws, kind):
    events = []
    while True:
        events.append(ws.receive_json())
        if events[-1]["type"] in (kind, "error"):
            return events


VOICE_PATH = "/v1/realtime?model=voice"


class TestRealtimeVoiceWebSocket:
    def setup_method(self):
        self.executor = ThreadPoolExecutor(1, thread_name_prefix="mlx-engine-test")

    def teardown_method(self):
        self.executor.shutdown()

    def test_spoken_turn_round_trip(self):
        backend = _StubVoiceBackend()
        engine = _voice_engine(backend, self.executor)
        (p1, p2, p3), pool, client = _voice_client(engine)
        with p1, p2, p3, client, client.websocket_connect(VOICE_PATH) as ws:
            created = ws.receive_json()
            assert created["type"] == "session.created"
            assert created["session"]["audio"]["input"]["turn_detection"] is None
            # Client instructions replace this default for the whole session.
            assert created["session"]["instructions"] == "Speak briefly."
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {"instructions": "Be brief.", "voice": "ethan"},
                }
            )
            updated = ws.receive_json()
            assert updated["session"]["audio"]["output"]["voice"] == "ethan"
            ws.send_json({"type": "input_audio_buffer.append", "audio": _pcm(0.2)})
            ws.send_json({"type": "input_audio_buffer.commit"})
            committed = ws.receive_json()
            assert committed["type"] == "input_audio_buffer.committed"
            ws.send_json({"type": "response.create"})
            events = _until(ws, "response.done")
        kinds = [e["type"] for e in events]
        assert kinds[:2] == ["response.created", "response.output_item.added"]
        audio = [e for e in events if e["type"] == "response.output_audio.delta"]
        pcm = np.frombuffer(base64.b64decode(audio[0]["delta"]), dtype="<i2")
        assert len(pcm) == 240 and pcm[0] == 16383
        done = events[-1]["response"]
        assert done["status"] == "completed"
        assert done["output"][0]["content"][0]["transcript"] == "hi hi "
        assert done["usage"]["input_tokens"] == 7
        assert done["usage"]["output_tokens"] == 6
        assert backend.turns == [(4800, "Be brief.", "ethan")]
        assert _eventually(lambda: pool.release_engine.await_count == 1)
        pool.release_engine.assert_awaited_once_with("voice")
        assert backend.closed
        engine.release_realtime_session.assert_called_once()
        assert engine._speech_active == 0

    def test_failed_reply_reports_its_error(self):
        full = RealtimeVoiceError("conversation_too_long", "the context is full")
        engine = _voice_engine(_StubVoiceBackend(fail=full), self.executor)
        (p1, p2, p3), _, client = _voice_client(engine)
        with p1, p2, p3, client, client.websocket_connect(VOICE_PATH) as ws:
            ws.receive_json()
            ws.send_json({"type": "input_audio_buffer.append", "audio": _pcm(0.2)})
            ws.send_json({"type": "input_audio_buffer.commit"})
            ws.receive_json()
            ws.send_json({"type": "response.create"})
            error = _until(ws, "response.done")[-1]["error"]
            assert error["code"] == "conversation_too_long"
            done = _until(ws, "response.done")[-1]
        assert done["response"]["status"] == "failed"

    def test_image_and_function_call_round_trip(self):
        call = FunctionCall("web_search", '{"query": "mlx"}')
        backend = _StubVoiceBackend(calls=[[call]])
        engine = _voice_engine(backend, self.executor)
        (p1, p2, p3), _, client = _voice_client(engine)
        tool = {"type": "function", "name": "web_search", "parameters": {}}
        with p1, p2, p3, client, client.websocket_connect(VOICE_PATH) as ws:
            created = ws.receive_json()
            assert created["session"]["capabilities"] == {
                "input_image": True,
                "tools": True,
            }
            ws.send_json({"type": "session.update", "session": {"tools": [tool]}})
            assert ws.receive_json()["session"]["tools"] == [tool]
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": _png_url()}],
                    },
                }
            )
            assert ws.receive_json()["type"] == "conversation.item.added"
            ws.send_json({"type": "input_audio_buffer.append", "audio": _pcm(0.2)})
            ws.send_json({"type": "input_audio_buffer.commit"})
            ws.receive_json()
            ws.send_json({"type": "response.create"})
            events = _until(ws, "response.done")
            args = next(
                e
                for e in events
                if e["type"] == "response.function_call_arguments.done"
            )
            assert args["name"] == "web_search" and args["arguments"] == call.arguments
            output = events[-1]["response"]["output"]
            assert [item["type"] for item in output] == ["message", "function_call"]
            assert output[1]["call_id"] == args["call_id"]

            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "call_unknown",
                        "output": "{}",
                    },
                }
            )
            assert ws.receive_json()["error"]["code"] == "item_not_found"
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": args["call_id"],
                        "output": '{"ok": true}',
                    },
                }
            )
            assert ws.receive_json()["type"] == "conversation.item.added"
            # An image right after an output is that tool's screenshot.
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": _png_url()}],
                    },
                }
            )
            assert ws.receive_json()["type"] == "conversation.item.added"
            # A tool output alone opens the next reply.
            ws.send_json({"type": "response.create"})
            assert _until(ws, "response.done")[-1]["response"]["status"] == "completed"
        (first, tools), (second, _) = backend.inputs
        assert len(first.images) == 1 and first.audio is not None
        assert tools == [
            {"type": "function", "function": {"name": "web_search", "parameters": {}}}
        ]
        assert second.audio is None and not second.images
        assert [o.text for o in second.tool_outputs] == ['{"ok": true}']
        assert len(second.tool_outputs[0].images) == 1

    def test_truncate_drops_outputs_sent_for_the_cut_reply(self):
        call = FunctionCall("web_search", "{}")
        backend = _StubVoiceBackend(calls=[[call]])
        engine = _voice_engine(backend, self.executor)
        (p1, p2, p3), _, client = _voice_client(engine)
        tool = {"type": "function", "name": "web_search"}
        with p1, p2, p3, client, client.websocket_connect(VOICE_PATH) as ws:
            ws.receive_json()
            ws.send_json({"type": "session.update", "session": {"tools": [tool]}})
            ws.receive_json()
            ws.send_json({"type": "input_audio_buffer.append", "audio": _pcm(0.2)})
            ws.send_json({"type": "input_audio_buffer.commit"})
            ws.receive_json()
            ws.send_json({"type": "response.create"})
            done = _until(ws, "response.done")[-1]["response"]
            reply_id, call_id = done["output"][0]["id"], done["output"][1]["call_id"]
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": "{}",
                    },
                }
            )
            ws.receive_json()
            # The user talks over the reply before its next reply starts.
            ws.send_json(
                {
                    "type": "conversation.item.truncate",
                    "item_id": reply_id,
                    "audio_end_ms": 0,
                }
            )
            assert ws.receive_json()["type"] == "conversation.item.truncated"
            ws.send_json({"type": "input_audio_buffer.append", "audio": _pcm(0.2)})
            ws.send_json({"type": "input_audio_buffer.commit"})
            ws.receive_json()
            ws.send_json({"type": "response.create"})
            _until(ws, "response.done")
        turn, _ = backend.inputs[-1]
        assert turn.audio is not None and not turn.tool_outputs

    def test_images_and_tools_need_a_capable_model(self):
        backend = _StubVoiceBackend()
        backend.input_image = backend.supports_tools = False
        engine = _voice_engine(backend, self.executor)
        (p1, p2, p3), _, client = _voice_client(engine)
        image = {"type": "input_image", "image_url": _png_url()}
        with p1, p2, p3, client, client.websocket_connect(VOICE_PATH) as ws:
            ws.receive_json()
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {"tools": [{"type": "function", "name": "f"}]},
                }
            )
            assert ws.receive_json()["error"]["code"] == "unsupported"
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [image]},
                }
            )
            assert ws.receive_json()["error"]["code"] == "unsupported"
            # A malformed part is refused without closing the session.
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": ["x"]},
                }
            )
            assert ws.receive_json()["error"]["code"] == "unsupported"

    def test_gpu_power_limit_holds_for_the_session(self):
        engine = _voice_engine(_StubVoiceBackend(), self.executor)
        (p1, p2, p3), pool, client = _voice_client(engine)
        limiter = MagicMock()
        p4 = patch("omlx.api.audio_routes._get_gpu_power_limiter", return_value=limiter)
        p5 = patch(
            "omlx.api.audio_routes._realtime_gpu_power_limit",
            side_effect=lambda watts, model: watts,
        )
        with p1, p2, p3, p4, p5, client:
            with client.websocket_connect(f"{VOICE_PATH}&gpu_power_limit_w=12") as ws:
                ws.receive_json()
                limiter.acquire.assert_called_once_with(12)
                limiter.release.assert_not_called()
            assert _eventually(lambda: limiter.release.call_count == 1)
            limiter.release.assert_called_once_with(12)
            with client.websocket_connect(f"{VOICE_PATH}&gpu_power_limit_w=1") as ws:
                assert ws.receive_json()["error"]["code"] == "invalid_value"
        # The bad value is refused before the model is leased.
        assert pool.get_engine.await_count == 1

    def test_rejects_server_vad_and_missing_turn(self):
        engine = _voice_engine(_StubVoiceBackend(), self.executor)
        (p1, p2, p3), _, client = _voice_client(engine)
        with p1, p2, p3, client, client.websocket_connect(VOICE_PATH) as ws:
            ws.receive_json()
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {"turn_detection": {"type": "server_vad"}},
                }
            )
            assert ws.receive_json()["error"]["code"] == "unsupported"
            ws.send_json({"type": "input_audio_buffer.commit"})
            assert (
                ws.receive_json()["error"]["code"] == "input_audio_buffer_commit_empty"
            )
            ws.send_json({"type": "response.create"})
            assert ws.receive_json()["error"]["code"] == "input_audio_buffer_empty"

    def test_cancel_truncate_and_locked_voice(self):
        backend = _StubVoiceBackend(steps=10**6, delay=0.01)
        engine = _voice_engine(backend, self.executor)
        (p1, p2, p3), _, client = _voice_client(engine)
        with p1, p2, p3, client, client.websocket_connect(VOICE_PATH) as ws:
            ws.receive_json()
            ws.send_json({"type": "input_audio_buffer.append", "audio": _pcm(0.2)})
            ws.send_json({"type": "input_audio_buffer.commit"})
            ws.receive_json()
            ws.send_json({"type": "response.create"})
            created = _until(ws, "response.output_item.added")
            item_id = created[-1]["item"]["id"]
            ws.send_json({"type": "response.create"})
            ws.send_json({"type": "response.cancel"})
            events = _until(ws, "response.done")
            errors = [e for e in events if e["type"] == "error"]
            if errors:
                assert (
                    errors[0]["error"]["code"]
                    == "conversation_already_has_active_response"
                )
                events = _until(ws, "response.done")
            assert events[-1]["response"]["status"] == "cancelled"
            ws.send_json(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "audio_end_ms": 500,
                }
            )
            truncated = _until(ws, "conversation.item.truncated")[-1]
            assert truncated["audio_end_ms"] == 500
            ws.send_json({"type": "session.update", "session": {"voice": "ethan"}})
            assert ws.receive_json()["error"]["code"] == "session_locked"
        assert backend.stopped == 1
        assert backend.truncated == [0.5]

    def test_rejected_connections(self):
        engine = _voice_engine(_StubVoiceBackend(), self.executor)
        (p1, p2, p3), _, client = _voice_client(engine, key_ok=False)
        with p1, p2, p3, client, client.websocket_connect(VOICE_PATH) as ws:
            assert ws.receive_json()["error"]["code"] == "invalid_api_key"

        stt = MagicMock(spec=STTEngine)
        (p1, p2, p3), pool, client = _voice_client(stt)
        with p1, p2, p3, client, client.websocket_connect(VOICE_PATH) as ws:
            assert ws.receive_json()["error"]["code"] == "unsupported_model"
        assert _eventually(lambda: pool.release_engine.await_count == 1)

        busy = _voice_engine(_StubVoiceBackend(), self.executor)
        busy.create_realtime_session = AsyncMock(
            side_effect=RealtimeVoiceError("session_limit", "busy")
        )
        (p1, p2, p3), _, client = _voice_client(busy)
        with p1, p2, p3, client, client.websocket_connect(VOICE_PATH) as ws:
            assert ws.receive_json()["error"]["code"] == "session_limit"

    def test_api_key_from_header_or_subprotocol(self):
        header = SimpleNamespace(headers={"authorization": "Bearer k1"}, scope={})
        protocol = SimpleNamespace(
            headers={},
            scope={"subprotocols": ["realtime", "openai-insecure-api-key.k2"]},
        )
        assert _realtime_api_key(header) == "k1"
        assert _realtime_api_key(protocol) == "k2"

    def test_voice_backends_cover_discovery_types(self):
        assert set(_VOICE_BACKENDS) == REALTIME_VOICE_MODEL_TYPES


# ---------------------------------------------------------------------------
# LFM3 voice backend on a tiny random LFM3-Audio
# ---------------------------------------------------------------------------

_TINY_IDS = {
    "audio_token_id": 250,
    "audio_out_token_id": 251,
    "audio_start_token_id": 252,
    "text_end_token_id": 253,
    "mixed_start_token_id": 254,
    "eos_token_id": [255, 253],
}


def _tiny_lfm(layer_types, hidden, **extra):
    return {
        "hidden_size": hidden,
        "intermediate_size": 2 * hidden,
        "num_hidden_layers": len(layer_types),
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 260,
        "layer_types": layer_types,
        **extra,
    }


@pytest.fixture(scope="module")
def tiny_lfm3():
    # With these weights the reply ends on its own before the slot limit.
    mx.random.seed(3)
    config = Lfm3AudioConfig(
        lfm=_tiny_lfm(
            ["conv", "full_attention", "conv", "sliding_attention"],
            64,
            sliding_window=8,
        ),
        encoder={
            "lfm": _tiny_lfm(
                ["conv", "full_attention"],
                32,
                model_type="lfm2",
                max_position_embeddings=128,
                norm_eps=1e-5,
                conv_bias=False,
                conv_L_cache=3,
                block_multiple_of=16,
                block_ffn_dim_multiplier=1.0,
                block_auto_adjust_ff_dim=True,
            ),
            "dim_conv": 16,
            "dim_out": 32,
            "dim_in": 128,
            "sliding_window": 32,
        },
        depthformer=_tiny_lfm(["full_attention"], 32, head_dim=8, vocab_size=1),
        **_TINY_IDS,
    )
    model = Lfm3AudioModel(config)
    codec = LiquidCodec(CodecConfig(encoder_dim=64, quantizer_dim=32))
    codec.decoder.istft.window = mx.array(np.hanning(1281)[:-1], dtype=mx.float32)
    mx.eval(model.parameters(), codec.parameters())
    object.__setattr__(model, "_codec", codec)
    return model


class _Tokenizer:
    # Special tokens a test maps onto ids of the tiny vocabulary.
    special: dict = {}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + i % 26) for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.special.get(token, -1)


def _tiny_backend(model, *turns):
    backend = lfm3_realtime.LFM3VoiceBackend(
        model, SimpleNamespace(tokenizer=_Tokenizer())
    )
    queue = iter(turns)
    backend._turn_inputs = lambda turn, system, tools: (list(next(queue)), {})
    return backend


TOOLS = [{"type": "function", "function": {"name": "web_search"}}]


def _first_choice(model, ids):
    cache = model.make_cache()
    embeds = model.get_input_embeddings(mx.array([ids])).inputs_embeds
    hidden = model.language_model.model(None, cache, embeds)[:, -1]
    logits = model.language_model.model.embed_tokens.as_linear(hidden)
    return int(mx.argmax(logits, axis=-1).item())


def test_lfm3_tools_open_speech_like_the_plain_prompt(tiny_lfm3):
    ids = list(range(10, 30))
    plain = _tiny_backend(tiny_lfm3, [*ids, 254])
    plain.begin_turn(VoiceTurn(np.zeros(2400, np.float32)), "", None, [])
    # With tools and a first choice other than <tool_call>, the opener is fed after the prompt.
    tooled = _tiny_backend(tiny_lfm3, ids)
    tooled.begin_turn(VoiceTurn(np.zeros(2400, np.float32)), "", None, TOOLS)
    assert tooled.mode == lfm3_realtime.INTERLEAVED and not tooled.reply
    assert np.allclose(np.array(tooled.hidden), np.array(plain.hidden), atol=1e-4)
    assert tooled._offset() == plain._offset()


def test_lfm3_call_reply_is_silent_and_leaves_no_partial_call(tiny_lfm3, monkeypatch):
    monkeypatch.setattr(lfm3_realtime, "MAX_REPLY_SLOTS", 30)
    ids = list(range(10, 30))
    first = _first_choice(tiny_lfm3, ids)
    monkeypatch.setattr(_Tokenizer, "special", {"<tool_call>": first})
    # Call text that never closes, until the slot limit stops it.
    monkeypatch.setattr(lfm3_realtime, "sample", lambda *args: mx.array([40]))
    backend = _tiny_backend(tiny_lfm3, ids)
    backend.begin_turn(VoiceTurn(np.zeros(2400, np.float32)), "", None, TOOLS)
    texts, audio = [], 0
    while True:
        step = backend.step(0.01, threading.Event())
        texts.append(step.text)
        audio += step.audio.size
        if step.done:
            break
    assert audio == 0 and not "".join(texts) and not step.function_calls
    assert backend._offset() == len(ids)
    assert _slots(backend.pending) == [(255, None)]


def test_lfm3_call_that_ends_early_turns_into_speech(tiny_lfm3, monkeypatch):
    ids = list(range(10, 30))
    first = _first_choice(tiny_lfm3, ids)
    monkeypatch.setattr(_Tokenizer, "special", {"<tool_call>": first})
    backend = _tiny_backend(tiny_lfm3, ids)
    backend.begin_turn(VoiceTurn(np.zeros(2400, np.float32)), "", None, TOOLS)
    assert backend.call_ids == [first]
    monkeypatch.setattr(lfm3_realtime, "sample", lambda *args: mx.array([255]))
    backend._slot()
    assert backend.mode == lfm3_realtime.INTERLEAVED and not backend.call_ids
    # Back at the prompt's end, with the interleaved opener fed.
    assert backend._offset() == len(ids) + 1 and not backend.finished


def test_lfm3_failed_first_turn_keeps_no_cache(tiny_lfm3, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("out of memory")

    backend = _tiny_backend(tiny_lfm3, list(range(10, 30)))
    monkeypatch.setattr(type(tiny_lfm3), "get_input_embeddings", fail)
    with pytest.raises(RuntimeError):
        backend.begin_turn(VoiceTurn(np.zeros(2400, np.float32)), "", None, TOOLS)
    # The retry renders the system prompt and tools again.
    assert backend.cache is None and backend.tools == []


def test_lfm3_reads_its_xml_tool_call(tiny_lfm3):
    backend = _tiny_backend(tiny_lfm3)
    xml = "<tool_call>\n<function=web_search>\n<parameter=query>\nmlx\n</parameter>\n</function>\n</tool_call>"
    backend.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: xml)
    backend.tools, backend.call_ids = TOOLS, [1]
    (call,) = backend._parse_calls()
    assert call.name == "web_search" and json.loads(call.arguments) == {"query": "mlx"}


def _slots(slots):
    return [
        (token, None if codes is None else codes.tolist()) for token, codes in slots
    ]


def _finish(backend, seconds):
    while not backend.step(seconds, threading.Event()).done:
        pass


def test_lfm3_voice_loop_matches_stream_slots(tiny_lfm3, monkeypatch):
    monkeypatch.setattr(lfm3_realtime, "MAX_REPLY_SLOTS", 90)
    ids = [*range(10, 30), 254]
    mx.random.seed(0)
    reference = _slots(
        tiny_lfm3.stream_slots(
            mx.array([ids]), max_tokens=90, **tiny_lfm3.audio_sampling
        )
    )
    assert any(codes is not None for _, codes in reference)
    for seconds in (0.0, 0.01, 10.0):
        backend = _tiny_backend(tiny_lfm3, ids)
        backend.begin_turn(VoiceTurn(np.zeros(2400, np.float32)), "", None, [])
        mx.random.seed(0)
        _finish(backend, seconds)
        assert _slots(backend.reply) == reference


def test_lfm3_truncate_then_next_turn_matches_fresh_prefill(tiny_lfm3, monkeypatch):
    monkeypatch.setattr(lfm3_realtime, "MAX_REPLY_SLOTS", 60)
    # The next turn is shorter than the sliding window and the conv state,
    # so every restored cache layer shows in its last hidden state.
    turn1, turn2 = [*range(10, 30), 254], [40, 254]
    eos_code = tiny_lfm3.config.audio_eos_code
    backend = _tiny_backend(tiny_lfm3, turn1, turn2)
    backend.begin_turn(VoiceTurn(np.zeros(2400, np.float32)), "", None, [])
    mx.random.seed(1)
    _finish(backend, 0.01)
    spoken = [c for _, c in backend.reply if c is not None and c[0].item() != eos_code]
    assert len(spoken) > 2
    backend.truncate(2 * lfm3_realtime.FRAME_SECONDS)
    heard = backend.pending
    assert sum(c is not None and c[0].item() != eos_code for _, c in heard) == 2
    assert heard[-2][1][0].item() == eos_code and heard[-1][0] == 255
    backend.truncate(0.0)
    pending = list(backend.pending)
    assert _slots(pending) == [(255, None)]
    backend.begin_turn(VoiceTurn(np.zeros(2400, np.float32)), "", None, [])

    plain = backend._embed
    embeds = mx.concatenate(
        [
            plain([(t, None) for t in turn1]),
            plain(pending),
            plain([(t, None) for t in turn2]),
        ]
    )
    fresh = tiny_lfm3.language_model.model(None, tiny_lfm3.make_cache(), embeds[None])
    assert np.allclose(np.array(backend.hidden), np.array(fresh[:, -1]), atol=1e-4)


def test_lfm3_truncate_before_any_frame_drops_the_reply(tiny_lfm3):
    backend = _tiny_backend(tiny_lfm3, [*range(10, 30), 254])
    backend.begin_turn(VoiceTurn(np.zeros(2400, np.float32)), "", None, [])
    prefill = backend._offset()
    mx.random.seed(1)
    # An interleaved reply opens with text slots, before any speech frame.
    for _ in range(3):
        backend._slot()
    assert not backend.frames and backend._offset() > prefill
    backend.stop()
    backend.truncate(0.0)
    assert backend._offset() == prefill
    assert _slots(backend.pending) == [(255, None)]


def test_codec_stream_matches_full_decode(tiny_lfm3):
    codec = tiny_lfm3.codec()
    mx.random.seed(2)
    codes = mx.random.randint(0, 2016, (12, 24)).astype(mx.int32)
    full = np.asarray(codec.decode(codes))
    for chunks in ([24], [1] * 24, [2, 7, 15]):
        stream = lfm3_realtime.CodecStream(codec)
        parts, start = [], 0
        for size in chunks:
            parts.append(stream.push([codes[:, i] for i in range(start, start + size)]))
            start += size
        parts.append(stream.flush())
        pcm = np.concatenate(parts)
        assert pcm.shape == full.shape
        assert np.abs(pcm - full).max() < 1e-4
