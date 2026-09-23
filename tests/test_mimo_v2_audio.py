# SPDX-License-Identifier: Apache-2.0
"""Tests for MiMo V2.6 audio tokenization and processor integration."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from omlx.patches.mimo_v2.audio import (
    AudioTokenizerEncoder,
    MiMoAudioProcessor,
    _load_codebook_weights,
    _LocalTransformer,
    audio_cache_key_ranges,
    group_audio_codes,
    mel_spectrogram,
)


def test_audio_cache_keys_preserve_prefixes_before_changed_media():
    ids = [1, 2, 99, 99, 3, 4, 5, 99, 99, 6]
    codes = mx.zeros((4, 4, 20), dtype=mx.int32)
    original = audio_cache_key_ranges(
        ids, codes, 99, [(0, "first-image"), (6, "second-image")]
    )
    changed_codes = mx.concatenate([codes[:2], mx.ones_like(codes[2:])])
    changed_audio = audio_cache_key_ranges(
        ids, changed_codes, 99, [(0, "first-image"), (6, "second-image")]
    )
    changed_first_audio = audio_cache_key_ranges(
        ids,
        mx.concatenate([mx.ones_like(codes[:2]), codes[2:]]),
        99,
        [(0, "first-image"), (6, "second-image")],
    )
    changed_image = audio_cache_key_ranges(
        ids, codes, 99, [(0, "first-image"), (6, "different-image")]
    )

    assert [start for start, _ in original] == [0, 2, 6, 7]
    assert original[0] == (0, "first-image")
    assert original[:3] == changed_audio[:3]
    assert original[3][1] != changed_audio[3][1]
    assert original[0] == changed_first_audio[0]
    assert all(a[1] != b[1] for a, b in zip(original[1:], changed_first_audio[1:]))
    assert original[:2] == changed_image[:2]
    assert all(a[1] != b[1] for a, b in zip(original[2:], changed_image[2:]))
    assert original == audio_cache_key_ranges(
        ids, codes, 99, [(0, "first-image"), (6, "second-image")]
    )


def test_mel_spectrogram_matches_torchaudio_power_default(monkeypatch):
    import transformers.audio_utils as audio_utils

    captured = {}
    monkeypatch.setattr(
        audio_utils,
        "mel_filter_bank",
        lambda **_kwargs: np.ones((5, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        audio_utils,
        "window_function",
        lambda *_args, **_kwargs: np.ones(8, dtype=np.float32),
    )

    def fake_spectrogram(*_args, **kwargs):
        captured.update(kwargs)
        return np.ones((4, 3), dtype=np.float32)

    monkeypatch.setattr(audio_utils, "spectrogram", fake_spectrogram)
    config = SimpleNamespace(
        nfft=8,
        sampling_rate=24_000,
        n_mels=4,
        fmin=0,
        fmax=None,
        window_size=8,
        hop_length=2,
    )

    mel_spectrogram(np.zeros(16, dtype=np.float32), config)

    assert captured["power"] == 2.0


def test_audio_tokenizer_uses_reference_hybrid_attention_schedule():
    encoder = AudioTokenizerEncoder(
        {
            "d_model": 8,
            "encoder_attention_heads": 2,
            "n_mels": 4,
            "kernel_size": 3,
            "stride_size": 2,
            "hybrid_attention": True,
            "swa_per_block": 2,
            "encoder_attn_window_size": [128, 0],
            "encoder_layers": 4,
            "encoder_ffn_dim": 16,
            "encoder_causal": True,
            "encoder_skip_layer_id": 3,
            "avg_pooler": 2,
            "codebook_size": [8],
            "num_quantizers": 2,
            "rope_theta": 10_000,
        }
    )

    assert [layer.self_attn.window_size for layer in encoder.layers] == [128, -1, 128, -1]


def test_audio_bridge_local_transformer_uses_reference_causal_mask():
    seen = []

    class CapturingLayer(nn.Module):
        def __call__(self, x, mask=None):
            seen.append(mask)
            return x

    transformer = _LocalTransformer.__new__(_LocalTransformer)
    nn.Module.__init__(transformer)
    transformer.layers = [CapturingLayer()]
    transformer.norm = nn.Identity()

    transformer(mx.zeros((2, 4, 8)))

    assert len(seen) == 1
    assert seen[0].tolist() == [
        [0.0, -1e9, -1e9, -1e9],
        [0.0, 0.0, -1e9, -1e9],
        [0.0, 0.0, 0.0, -1e9],
        [0.0, 0.0, 0.0, 0.0],
    ]


def test_load_codebook_weights_installs_private_mlx_parameters():
    layers = [
        SimpleNamespace(_codebook=SimpleNamespace(embed=mx.zeros((2, 3))))
        for _ in range(2)
    ]
    model = SimpleNamespace(
        encoder=SimpleNamespace(
            quantizer=SimpleNamespace(vq=SimpleNamespace(layers=layers))
        )
    )
    first = mx.ones((2, 3))
    second = mx.full((2, 3), 2)

    _load_codebook_weights(
        model,
        {
            "encoder.quantizer.vq.layers.0._codebook.embed": first,
            "encoder.quantizer.vq.layers.1._codebook.embed": second,
        },
    )

    assert mx.array_equal(layers[0]._codebook.embed, first)
    assert mx.array_equal(layers[1]._codebook.embed, second)


def test_group_audio_codes_pads_time_axis_by_repeating_last_code():
    codes = mx.arange(6 * 20).reshape(6, 20)

    grouped = group_audio_codes(codes)
    mx.eval(grouped)

    assert grouped.shape == (2, 4, 20)
    assert grouped[0].tolist() == codes[:4].tolist()
    assert grouped[1, :2].tolist() == codes[4:].tolist()
    assert grouped[1, 2:].tolist() == [codes[-1].tolist(), codes[-1].tolist()]


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
    assert result["audio_codes"][2, 1:].tolist() == [[2] * 20] * 3


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
