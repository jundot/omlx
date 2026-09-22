# SPDX-License-Identifier: Apache-2.0
"""Tests for MiMo V2.6 audio tokenization and processor integration."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from omlx.patches.mimo_v2.audio import (
    MiMoAudioProcessor,
    group_audio_codes,
)


def test_group_audio_codes_pads_time_axis_with_zero_code():
    codes = mx.arange(6 * 20).reshape(6, 20)

    grouped = group_audio_codes(codes)
    mx.eval(grouped)

    assert grouped.shape == (2, 4, 20)
    assert grouped[0].tolist() == codes[:4].tolist()
    assert grouped[1, :2].tolist() == codes[4:].tolist()
    assert grouped[1, 2:].tolist() == [[1024] * 20, [1024] * 20]


def test_audio_processor_expands_placeholders_and_returns_codes(monkeypatch):
    class FakeTokenizer:
        pass

    class FakeBase:
        tokenizer = FakeTokenizer()
        image_processor = object()
        video_processor = None

        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return {"input_ids": mx.array([[1, 2, 3]])}

    class FakeEncoder:
        def __init__(self):
            self.calls = 0

        def encode(self, _mel):
            self.calls += 1
            return mx.full((self.calls + 3, 20), self.calls)

    fake_tokenizer = SimpleNamespace(
        config=SimpleNamespace(),
        encoder=FakeEncoder(),
    )
    base = FakeBase()
    processor = MiMoAudioProcessor(base, "/unused")
    monkeypatch.setattr(processor, "_load_audio_tokenizer", lambda: fake_tokenizer)
    monkeypatch.setattr(
        "omlx.patches.mimo_v2.audio.mel_spectrogram",
        lambda audio, _config: mx.array(audio),
    )

    result = processor(
        text=["before<|audio_pad|>middle<|audio_pad|>after"],
        audios=[np.zeros(4), np.zeros(5)],
        return_tensors="mlx",
    )

    # Four and five tokenizer frames become one and two grouped audio tokens.
    assert base.calls[0]["text"] == [
        "before<|audio_pad|>middle<|audio_pad|><|audio_pad|>after"
    ]
    assert result["audio_codes"].shape == (3, 4, 20)
    assert result["audio_codes"][0].tolist() == [[1] * 20] * 4
    assert result["audio_codes"][1].tolist() == [[2] * 20] * 4
    assert result["audio_codes"][2, :1].tolist() == [[2] * 20]
    assert result["audio_codes"][2, 1:].tolist() == [[1024] * 20] * 3


def test_audio_processor_preserves_text_only_requests():
    class FakeBase:
        tokenizer = object()
        image_processor = object()
        video_processor = None

        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return {"input_ids": mx.array([[7]])}

    base = FakeBase()
    processor = MiMoAudioProcessor(base, "/unused")

    result = processor(text=["hello"], return_tensors="mlx")

    assert result["input_ids"].tolist() == [[7]]
    assert base.calls[0]["text"] == ["hello"]
    assert "audios" not in base.calls[0]
