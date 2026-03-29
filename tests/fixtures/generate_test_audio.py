#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the minimal WAV fixture used by audio tests.

Run once to regenerate: python tests/fixtures/generate_test_audio.py
The output file (test_audio.wav) is committed and used by test_audio_stt.py.
"""

import io
import struct
import wave
from pathlib import Path

HERE = Path(__file__).parent


def make_wav(
    path: Path,
    duration_secs: float = 0.5,
    sample_rate: int = 16000,
    n_channels: int = 1,
    sample_width: int = 2,
) -> None:
    """Write a minimal WAV file with silence."""
    n_samples = int(sample_rate * duration_secs)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    print(f"Wrote {path}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    make_wav(HERE / "test_audio.wav")
