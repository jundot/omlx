# SPDX-License-Identifier: Apache-2.0
"""Live integration test for TTS HTTP streaming against a running oMLX server.

This test verifies the real transport path using httpx streaming against a
server process started separately (for example in tmux). It is intended for
real-model validation of the Phase 1 TTS streaming implementation.

Required environment variables:
- OMLX_TTS_MODEL: model ID exposed by /v1/models

Optional environment variables:
- OMLX_BASE_URL: server base URL (default: http://127.0.0.1:8000)
- OMLX_TTS_VOICE: voice to use (default: Chelsie)
- OMLX_API_KEY: API key if auth is enabled

Run with:
  OMLX_TTS_MODEL=Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
  pytest tests/integration/test_audio_tts_streaming_integration.py -m "integration or slow" -s -v
"""

import os
import time

import httpx
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BASE_URL = os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TTS_MODEL = os.environ.get("OMLX_TTS_MODEL")
TTS_VOICE = os.environ.get("OMLX_TTS_VOICE", "Chelsie")
API_KEY = os.environ.get("OMLX_API_KEY")


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _streaming_test_text() -> str:
    """Return a long, multi-sentence text that should force multiple TTS segments."""
    part1 = (
        "Hello, this is a long-form streaming verification for oMLX. "
        "We want the first audio bytes to arrive before the complete response has finished generating. "
        "The phrasing is intentionally a little longer than a short demo sentence so that the server must work through meaningful content."
    )
    part2 = (
        "Now we continue with a second paragraph-length sentence that should preserve the same Qwen custom voice characteristics across boundaries. "
        "If Phase 1 streaming is implemented correctly, this part should arrive later in the same HTTP response instead of being buffered until everything is done."
    )
    part3 = (
        "Finally, we add one more sentence to increase the chance of multiple synthesis calls and multiple transport writes. "
        "This makes the test more realistic for long assistant responses in agent or chat workflows."
    )
    return f"{part1} {part2} {part3}"


@pytest.mark.asyncio
async def test_live_tts_streaming_emits_multiple_http_chunks():
    """Verify that a running oMLX server emits incremental audio over HTTP streaming."""
    if not TTS_MODEL:
        pytest.skip("Set OMLX_TTS_MODEL to run live TTS streaming integration test")

    timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0)
    async with httpx.AsyncClient(base_url=BASE_URL, headers=_headers(), timeout=timeout) as client:
        # Verify server is reachable and the target model is exposed.
        models_resp = await client.get("/v1/models")
        models_resp.raise_for_status()
        model_ids = {m["id"] for m in models_resp.json().get("data", [])}
        assert TTS_MODEL in model_ids, (
            f"Model {TTS_MODEL!r} not found in /v1/models. Available: {sorted(model_ids)}"
        )

        payload = {
            "model": TTS_MODEL,
            "input": _streaming_test_text(),
            "voice": TTS_VOICE,
            "response_format": "wav",
            "stream": True,
        }

        chunk_timestamps: list[float] = []
        raw_chunks: list[bytes] = []
        t0 = time.perf_counter()

        async with client.stream("POST", "/v1/audio/speech", json=payload) as response:
            response.raise_for_status()
            assert "audio/wav" in response.headers.get("content-type", "")

            async for chunk in response.aiter_raw():
                if not chunk:
                    continue
                chunk_timestamps.append(time.perf_counter())
                raw_chunks.append(chunk)

        t_done = time.perf_counter()
        assert raw_chunks, "No streaming audio chunks received"

        first_chunk = raw_chunks[0]
        full_body = b"".join(raw_chunks)
        t_first = chunk_timestamps[0] - t0
        total_time = t_done - t0

        # Basic WAV shape: one header at the beginning, audio bytes afterwards.
        assert first_chunk.startswith(b"RIFF"), first_chunk[:32]
        assert b"WAVE" in first_chunk[:64], first_chunk[:64]
        assert len(full_body) > 4096, f"Unexpectedly small audio body: {len(full_body)} bytes"
        assert full_body.count(b"RIFF") == 1, "Expected one WAV header for the streamed response"

        # Real transport assertions: we expect multiple received chunks and earlier first audio than total completion.
        assert len(raw_chunks) >= 2, (
            "Expected multiple HTTP chunks from the live streaming response, "
            f"got {len(raw_chunks)}"
        )
        assert t_first < total_time, (
            f"First chunk did not arrive before completion: first={t_first:.2f}s total={total_time:.2f}s"
        )

        # Stronger signal that bytes arrived incrementally, not all at once.
        inter_chunk_gap = chunk_timestamps[-1] - chunk_timestamps[0]
        assert inter_chunk_gap > 0.05, (
            "All chunks arrived effectively at once; expected observable incremental delivery. "
            f"gap={inter_chunk_gap:.3f}s, chunks={len(raw_chunks)}"
        )

        print(
            f"Streaming verified for {TTS_MODEL}: chunks={len(raw_chunks)}, "
            f"first_byte={t_first:.2f}s, total={total_time:.2f}s, "
            f"inter_chunk_gap={inter_chunk_gap:.2f}s, bytes={len(full_body)}"
        )
