#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a streamed TTS sample counting from one to thirty.

This is a live utility script for a running oMLX server. It sends one
continuous TTS request to /v1/audio/speech, writes the raw streamed WAV bytes,
then writes a finalized WAV with corrected RIFF/data sizes for normal players.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import time
import wave
from pathlib import Path

import httpx


DEFAULT_TEXT = (
    "one, two, three, four, five, six, seven, eight, nine, ten, "
    "eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, "
    "eighteen, nineteen, twenty, twenty one, twenty two, twenty three, "
    "twenty four, twenty five, twenty six, twenty seven, twenty eight, "
    "twenty nine, thirty."
)

DEFAULT_INSTRUCTIONS = (
    "Count naturally in one continuous, calm voice at an even pace. "
    "Avoid dramatic pauses or restarts."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a one-to-thirty TTS sample through oMLX streaming."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000"),
        help="oMLX server base URL.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "OMLX_TTS_MODEL", "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
        ),
        help="TTS model ID exposed by /v1/models.",
    )
    parser.add_argument(
        "--voice",
        default=os.environ.get("OMLX_TTS_VOICE", "vivian"),
        help="Voice name to use for the TTS model.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OMLX_API_KEY", "oMLX"),
        help="Bearer API key. Use an empty string to omit Authorization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for generated text, WAV, and metadata files.",
    )
    parser.add_argument(
        "--prefix",
        default="qwen_tts_count_1_to_30_vivian_continuous",
        help="Filename prefix for generated artifacts.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--instructions",
        default=DEFAULT_INSTRUCTIONS,
        help="Optional synthesis instructions.",
    )
    return parser.parse_args()


def finalize_stream_wav(raw_path: Path, final_path: Path) -> dict[str, float | int]:
    raw = bytearray(raw_path.read_bytes())
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE" or raw[36:40] != b"data":
        raise RuntimeError(f"Unexpected WAV layout in {raw_path}")

    raw[4:8] = struct.pack("<I", len(raw) - 8)
    raw[40:44] = struct.pack("<I", len(raw) - 44)
    final_path.write_bytes(raw)

    with wave.open(str(final_path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        return {
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "sample_rate_hz": sample_rate,
            "frames": frames,
            "duration_seconds": round(frames / sample_rate, 3),
        }


async def generate(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    text_path = output_dir / f"{args.prefix}.txt"
    raw_path = output_dir / f"{args.prefix}_stream.wav"
    final_path = output_dir / f"{args.prefix}_finalized.wav"
    meta_path = output_dir / f"{args.prefix}.json"

    text_path.write_text(args.text + "\n")

    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "instructions": args.instructions,
        "response_format": "wav",
        "stream": True,
        "speed": 1.0,
    }

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0)

    chunk_sizes: list[int] = []
    chunk_offsets: list[float] = []
    t0 = time.perf_counter()

    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        timeout=timeout,
    ) as client:
        async with client.stream("POST", "/v1/audio/speech", json=payload) as response:
            response.raise_for_status()
            with raw_path.open("wb") as raw_file:
                async for chunk in response.aiter_raw():
                    if not chunk:
                        continue
                    raw_file.write(chunk)
                    raw_file.flush()
                    chunk_sizes.append(len(chunk))
                    chunk_offsets.append(time.perf_counter() - t0)

    finalized_audio = finalize_stream_wav(raw_path, final_path)

    meta = {
        "model": args.model,
        "voice": args.voice,
        "instructions": args.instructions,
        "text_file": str(text_path.resolve()),
        "raw_stream_file": str(raw_path.resolve()),
        "finalized_file": str(final_path.resolve()),
        "text": args.text,
        "chunks": len(chunk_sizes),
        "bytes": raw_path.stat().st_size,
        "first_byte_seconds": round(chunk_offsets[0], 3) if chunk_offsets else None,
        "total_seconds": round(chunk_offsets[-1], 3) if chunk_offsets else None,
        "inter_chunk_gap_seconds": (
            round(chunk_offsets[-1] - chunk_offsets[0], 3)
            if len(chunk_offsets) > 1
            else 0
        ),
        "chunk_sizes_head": chunk_sizes[:8],
        "chunk_offsets_head_seconds": [round(offset, 3) for offset in chunk_offsets[:8]],
        "finalized_audio": finalized_audio,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    meta = asyncio.run(generate(parse_args()))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
