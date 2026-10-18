# SPDX-License-Identifier: Apache-2.0
"""Tests for realtime STT: engine sessions and the WebSocket endpoint.

All tests run with mocked models/engines — mlx-audio decode paths are not
exercised here (that needs a real model; see the slow smoke procedure).
"""

import asyncio
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.engine.stt import (
    RealtimeTranscriptionSession,
    STTEngine,
    _is_qwen3_realtime_model,
    _Qwen3RealtimeBackend,
    _split_audio_segments,
    _VoxtralRealtimeBackend,
    _WhisperRealtimeBackend,
)
from omlx.model_discovery import is_realtime_stt_model


class TestRealtimeSttClassification:
    """Discovery-level realtime capability flag."""

    def test_whisper_is_realtime(self):
        assert is_realtime_stt_model("audio_stt", "whisper")

    def test_voxtral_realtime_is_realtime(self):
        assert is_realtime_stt_model("audio_stt", "voxtral_realtime")

    def test_qwen3_asr_is_realtime(self):
        # Served by _Qwen3RealtimeBackend (server-side endpointing), not by
        # an incremental-input decoder — see TestQwen3RealtimeBackend.
        assert is_realtime_stt_model("audio_stt", "qwen3_asr")

    def test_other_stt_families_are_not_realtime(self):
        assert not is_realtime_stt_model("audio_stt", "parakeet")
        assert not is_realtime_stt_model("audio_stt", "")

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

    def test_plain_stt_model_not_supported(self):
        engine = STTEngine("m")
        engine._model = SimpleNamespace(generate=lambda: None)
        assert not engine.supports_realtime_stt()

    def test_qwen3_model_supported_without_streaming_symbols(self):
        engine = STTEngine("m")
        engine._model = _FakeQwen3RtModel()
        assert engine.supports_realtime_stt()

    def test_admission_is_module_pinned_not_symbol_pinned(self):
        # A different whole-buffer model exposing stream_generate must
        # NOT be admitted: only a module carrying _Qwen3RealtimeBackend
        # semantics may pass (UP-11: symbol whitelisting alone would
        # crash foreign models in whisper code paths).
        engine = STTEngine("m")
        engine._model = SimpleNamespace(
            stream_generate=lambda: None, generate=lambda: None
        )
        assert not engine.supports_realtime_stt()


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
        engine._model = SimpleNamespace(generate=lambda: None)
        with pytest.raises(RuntimeError, match="realtime"):
            await engine.create_realtime_session()

    @pytest.mark.asyncio
    async def test_unstarted_engine_raises(self):
        with pytest.raises(RuntimeError, match="not started"):
            await STTEngine("m").create_realtime_session()

    @pytest.mark.asyncio
    async def test_qwen3_model_dispatches_to_qwen3_backend(self):
        engine = STTEngine("m")
        engine._model = _FakeQwen3RtModel()

        session = await engine.create_realtime_session(language="cs")
        assert isinstance(session._backend, _Qwen3RealtimeBackend)
        # "cs" is not in _ISO_TO_STT_LANG: the hint passes through verbatim
        assert session._backend._language == "cs"
        await session.release()

    @pytest.mark.asyncio
    async def test_whisper_symbol_model_still_dispatches_to_whisper(self):
        engine = STTEngine("m")
        engine._model = SimpleNamespace(generate_streaming=lambda: None)

        session = await engine.create_realtime_session()
        assert isinstance(session._backend, _WhisperRealtimeBackend)
        await session.release()

    @pytest.mark.asyncio
    async def test_unmatched_model_never_reaches_whisper_catch_all(self):
        # supports_realtime_stt() normally rejects such a model at the
        # route; if a future gate change lets one through, dispatch must
        # fail explicitly instead of feeding it to the whisper decoder
        # (the latent hole behind the UP-10 oQ8 crash shape).
        engine = STTEngine("m")
        engine._model = SimpleNamespace(generate=lambda: None)
        engine.supports_realtime_stt = lambda: True  # bypass the gate
        with pytest.raises(RuntimeError, match="no realtime backend"):
            await engine.create_realtime_session()


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


# ---------------------------------------------------------------------------
# Qwen3-ASR realtime backend (server-side per-utterance endpointing)
# ---------------------------------------------------------------------------


_QWEN3_MODULE = "mlx_audio.stt.models.qwen3_asr.qwen3_asr"
_RT_FRAME = 1600  # one 0.1 s slice at the realtime rate (16 kHz)


class _FakeQwen3RtModel:
    """Whole-buffer Qwen3-ASR stand-in for the realtime path.

    ``generate`` mirrors the per-closed-utterance call the backend makes
    through the same whole-buffer path as the file endpoint. Scripted
    replies / failures let the tests pin the decode budget, the
    degenerate-output guard and session survival without a checkpoint.
    """

    def __init__(self, texts=None, fail_times=0, expects_language_names=False):
        self.calls = []
        self._texts = list(texts or [])
        self._fail_times = fail_times
        if expects_language_names:
            self.config = SimpleNamespace(support_languages=["English", "Czech"])

    def generate(self, audio, language=None, max_tokens=None):
        i = len(self.calls)
        self.calls.append(
            SimpleNamespace(
                audio=np.asarray(audio),
                language=language,
                max_tokens=max_tokens,
            )
        )
        if i < self._fail_times:
            raise RuntimeError("simulated decode failure")
        return SimpleNamespace(
            text=self._texts[i] if i < len(self._texts) else "ok",
            language="english",
        )


# The engine admits Qwen3 models by their mlx-audio module path, so the
# fake must claim it (this is also the contract a real checkpoint upgrade
# would break, loudly: the gate falls back to segmented mode).
_FakeQwen3RtModel.__module__ = _QWEN3_MODULE


def _voiced(n, frame=_RT_FRAME):
    # RMS 0.05 clears VOICE_RMS (0.01); amplitude is irrelevant otherwise
    return [np.full(frame, 0.05, dtype=np.float32) for _ in range(n)]


def _quiet(n, frame=_RT_FRAME):
    return [np.zeros(frame, dtype=np.float32) for _ in range(n)]


def _drain(backend, frames):
    """Feed frame by frame, polling each time (tighter than the real 1 s
    session cadence; upper-bounds commit promptness)."""
    out = []
    for f in frames:
        backend.feed(f)
        out.extend(backend.poll_sync())
    return out


class TestQwen3RealtimeBackend:
    def test_utterance_committed_after_trailing_silence(self):
        model = _FakeQwen3RtModel(texts=["hello world"])
        out = _drain(_Qwen3RealtimeBackend(model), _voiced(5) + _quiet(8))

        assert out == ["hello world"]
        assert len(model.calls) == 1
        # 0.7 s silence closes; kept = speech + TAIL_MARGIN (0.2 s)
        assert len(model.calls[0].audio) == 7 * _RT_FRAME
        # 0.7 s -> 24*0.7+64 = 80.8 -> floor MIN_TOKENS
        assert model.calls[0].max_tokens == 96

    def test_preroll_keeps_the_first_syllable(self):
        model = _FakeQwen3RtModel()
        out = _drain(
            _Qwen3RealtimeBackend(model),
            _quiet(10) + _voiced(5) + _quiet(8),
        )

        assert out == ["ok"]
        # Onset picks up the 0.3 s preroll (3 frames) + 5 voiced + 0.2 s
        # tail margin (2 frames) = 10 frames
        assert len(model.calls[0].audio) == 10 * _RT_FRAME

    def test_cough_is_not_an_utterance(self):
        model = _FakeQwen3RtModel()
        out = _drain(_Qwen3RealtimeBackend(model), _voiced(2) + _quiet(8))

        assert out == []
        assert model.calls == []  # decode never paid for a 0.2 s blip

    def test_long_monologue_force_cut_bounds_decode(self):
        model = _FakeQwen3RtModel()
        backend = _Qwen3RealtimeBackend(model)
        out = _drain(backend, _voiced(260))  # 26 s without a pause

        assert len(out) == 1  # cut at MAX_SEGMENT_SECONDS, no silence needed
        assert len(model.calls[0].audio) == 250 * _RT_FRAME
        # 25 s -> 24*25+64 = 664 -> capped at MAX_TOKENS_CAP
        assert model.calls[0].max_tokens == 512

        rest = backend.close_sync()
        assert len(rest) == 1  # the open 1 s tail still lands on close
        assert len(model.calls) == 2

    def test_degenerate_spiral_dropped_session_continues(self):
        # Measured field shape (mruu report): a broken checkpoint emitted
        # 32k chars from 2.1 s of audio. The guard must swallow that
        # utterance — never deliver the wall — and keep listening.
        model = _FakeQwen3RtModel(texts=["x" * 100, "fine"])
        out = _drain(
            _Qwen3RealtimeBackend(model),
            _voiced(5) + _quiet(8) + _voiced(5) + _quiet(8),
        )

        assert out == ["fine"]  # first: 0.7 s audio, cap 42 chars -> dropped
        assert len(model.calls) == 2

    def test_failed_decode_drops_one_segment_not_the_session(self):
        model = _FakeQwen3RtModel(fail_times=1)
        out = _drain(
            _Qwen3RealtimeBackend(model),
            _voiced(5) + _quiet(8) + _voiced(5) + _quiet(8),
        )

        assert out == ["ok"]  # poison segment gone, second one delivered
        assert len(model.calls) == 2

    def test_close_flushes_open_utterance(self):
        model = _FakeQwen3RtModel(texts=["tail words"])
        backend = _Qwen3RealtimeBackend(model)
        assert _drain(backend, _voiced(5)) == []  # no silence yet

        assert backend.close_sync() == ["tail words"]
        assert len(model.calls[0].audio) == 5 * _RT_FRAME

    def test_quiet_stream_produces_nothing(self):
        model = _FakeQwen3RtModel()
        backend = _Qwen3RealtimeBackend(model)

        assert _drain(backend, _quiet(50)) == []
        assert backend.close_sync() == []
        assert model.calls == []

    def test_language_hint_normalized_for_name_expecting_models(self):
        model = _FakeQwen3RtModel(expects_language_names=True)
        backend = _Qwen3RealtimeBackend(model, language="en")
        _drain(backend, _voiced(5) + _quiet(8))

        assert model.calls[0].language == "english"

    def test_language_hint_verbatim_for_unknown_iso(self):
        model = _FakeQwen3RtModel(expects_language_names=True)
        backend = _Qwen3RealtimeBackend(model, language="cs")
        _drain(backend, _voiced(5) + _quiet(8))

        assert model.calls[0].language == "cs"


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
    from omlx.api.audio_routes import realtime_router

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
            client,client.websocket_connect(WS_PATH) as ws
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
            side_effect=RuntimeError("A realtime transcription session is already active")
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
        samples[27 * SR: 28 * SR] = 0.0
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
            backend._emit_stable_delta(["hello", " world"], is_last=False)
            == " world"
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
            backend._emit_stable_delta(
                ["completely different text"], is_last=False
            )
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
        loud = (np.ones(SR, dtype=np.float32) * 0.5)
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
        loud = (np.ones(SR, dtype=np.float32) * 0.5)
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
            stt_mod, "_load_audio_samples",
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
                text[len("language "): text.find("<asr_text>")].strip(),
                text[text.find("<asr_text>") + len("<asr_text>"):],
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
            stt_mod, "_load_audio_samples",
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
            stt_mod, "_load_audio_samples",
            lambda path: np.zeros(3 * SR, dtype=np.float32),
        )
        table = {2: "안", 5: " 하세요"}
        model = _FakeQwen3Model(
            [2, 5], lambda ids: "".join(table[i] for i in ids)
        )
        engine = STTEngine("qwen3-test")
        engine._model = model

        chunks = [
            c async for c in engine.transcribe_stream("/fake.wav", language="ko")
        ]
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
            stt_mod, "_load_audio_samples",
            lambda path: np.zeros(5 * SR, dtype=np.float32),
        )
        engine = STTEngine("whisper-test")
        engine._model = FakeWhisperModel()

        chunks = [
            c async for c in engine.transcribe_stream("/fake.wav", language="ko")
        ]
        assert seen["language"] == "ko"
        assert chunks[0]["text"] == " ok"
