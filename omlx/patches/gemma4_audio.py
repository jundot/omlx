# SPDX-License-Identifier: Apache-2.0
"""Several audio clips per request for mlx-vlm Gemma 4.

mlx-vlm passes only the first clip unless the processor sets
``supports_multiple_audio``. With several clips, two more defects appear:
- ``ceil(ms / 40)`` placeholders can exceed the encoder rows by one.
- The encoder keeps padded rows, so ``masked_scatter`` feeds a shorter clip's
  padding into the placeholders of the next clip.
The patch counts placeholders with the encoder arithmetic, as HF Transformers
does, and drops padded rows before the scatter.
"""

import mlx.core as mx
import numpy as np

_PATCHED = False


def _encoder_rows(feature_extractor, num_samples: int) -> int:
    # Mel frames, then two stride-2 convs with kernel 3 and padding 1.
    frame_length = feature_extractor.frame_length
    rows = (
        num_samples + frame_length // 2 - frame_length - 1
    ) // feature_extractor.hop_length + 1
    if rows <= 0:
        return 0
    for _ in range(2):
        rows = (rows - 1) // 2 + 1
    return rows


def _compact_rows(encodings: mx.array, pad_mask: mx.array):
    keep = np.flatnonzero(~np.array(pad_mask).reshape(-1))
    rows = encodings.reshape(-1, encodings.shape[-1])[mx.array(keep)]
    return rows[None], mx.zeros((1, rows.shape[0]), dtype=mx.bool_)


def apply_gemma4_audio_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return False

    from mlx_vlm.models.gemma4.audio import AudioEncoder
    from mlx_vlm.models.gemma4.audio_feature_extractor import (
        Gemma4AudioFeatureExtractor,
    )
    from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor

    count_placeholders = Gemma4Processor._compute_audio_num_tokens
    encode = AudioEncoder.__call__

    def _compute_audio_num_tokens(self, audio_waveform, sampling_rate):
        # gemma4_unified embeds fixed sample chunks, so its stock count is exact.
        if not isinstance(self.feature_extractor, Gemma4AudioFeatureExtractor):
            return count_placeholders(self, audio_waveform, sampling_rate)
        rows = _encoder_rows(self.feature_extractor, len(audio_waveform))
        return min(rows, self.audio_seq_length)

    def _encode_compact(self, audio_mel, audio_mel_mask):
        return _compact_rows(*encode(self, audio_mel, audio_mel_mask))

    Gemma4Processor.supports_multiple_audio = True
    Gemma4Processor._compute_audio_num_tokens = _compute_audio_num_tokens
    AudioEncoder.__call__ = _encode_compact
    _PATCHED = True
    return True
