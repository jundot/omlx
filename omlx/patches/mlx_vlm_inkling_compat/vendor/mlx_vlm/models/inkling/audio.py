import mlx.core as mx
import mlx.nn as nn

from .config import AudioConfig


class AudioModel(nn.Module):
    """dMel audio front end (HiggsAudioV2-style): each of ``n_mel_bins`` mel channels is
    discretized into ``mel_vocab_size`` buckets; per-channel bins are offset into a single
    embedding table, looked up, and summed, then RMS-normed into LM space."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.model_type = config.model_type
        self.n_mel_bins = config.n_mel_bins
        self.mel_vocab_size = config.mel_vocab_size
        self.embed_audio_tokens = nn.Embedding(
            config.n_mel_bins * config.mel_vocab_size, config.text_hidden_size
        )
        self.norm = nn.RMSNorm(config.text_hidden_size, eps=config.rms_norm_eps)

    def __call__(self, audio_input_ids):
        """audio_input_ids: [..., frames, n_mel_bins] of bucket indices -> [..., frames, hidden]."""
        offsets = mx.arange(self.n_mel_bins) * self.mel_vocab_size
        embeds = self.embed_audio_tokens(audio_input_ids + offsets)
        embeds = embeds.sum(axis=-2)
        return self.norm(embeds)
