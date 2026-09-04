from typing import Any, Optional

import mlx.core as mx

from ..cache import KVCache
from ..deepseekocr.language import DeepseekV2Model
from ..deepseekocr.language import LanguageModel as DeepseekLanguageModel
from .config import TextConfig


class RingSlidingKVCache(KVCache):
    """Unlimited-OCR R-SWA cache.

    Upstream keeps the full prompt/prefill cache, then appends generated tokens
    until a small decode ring is full. Afterwards, new decode keys/values
    overwrite that ring while absolute positions keep increasing.
    """

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = int(window_size)
        self.prefill_length: Optional[int] = None
        self._ring_pos = 0

    def update_and_fetch(self, keys, values):
        seq_len = int(keys.shape[2])

        if self.prefill_length is None:
            if seq_len > 1:
                return super().update_and_fetch(keys, values)
            self.prefill_length = self.offset

        if self.keys is None or self.offset < self.prefill_length + self.window_size:
            keys, values = super().update_and_fetch(keys, values)
            if self.offset >= self.prefill_length + self.window_size:
                self._ring_pos = 0
            return keys, values

        for idx in range(seq_len):
            slot = self.prefill_length + self._ring_pos
            self.keys[..., slot : slot + 1, :] = keys[..., idx : idx + 1, :]
            self.values[..., slot : slot + 1, :] = values[..., idx : idx + 1, :]
            self._ring_pos = (self._ring_pos + 1) % self.window_size

        self.offset += seq_len
        return (
            self.keys[..., : self.prefill_length + self.window_size, :],
            self.values[..., : self.prefill_length + self.window_size, :],
        )

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        kwargs.pop("window_size", None)
        # Prefill/warmup is standard causal attention. Steady-state decode uses
        # q_len=1 and attends over the retained prompt plus ring slots, matching
        # upstream SlidingWindowLlamaAttention.
        if self.prefill_length is not None and self.offset >= (
            self.prefill_length + self.window_size
        ):
            if N == 1 and not return_array:
                return None
        return super().make_mask(
            N, return_array=return_array, window_size=None, **kwargs
        )

    @property
    def state(self):
        if self.keys is None:
            return None, None
        end = (
            self.offset
            if self.prefill_length is None
            else min(self.offset, self.prefill_length + self.window_size)
        )
        return self.keys[..., :end, :], self.values[..., :end, :]

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = 0 if self.keys is None else self.keys.shape[2]
        self.prefill_length = None
        self._ring_pos = 0

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.window_size,
                    -1 if self.prefill_length is None else self.prefill_length,
                    self.offset,
                    self._ring_pos,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        window_size, prefill_length, offset, ring_pos = map(int, v)
        self.window_size = window_size
        self.prefill_length = None if prefill_length < 0 else prefill_length
        self.offset = offset
        self._ring_pos = ring_pos


class DeepseekV2RingSlidingModel(DeepseekV2Model):
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            h = self.embed_tokens(x)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = self._make_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)

    def _make_attention_mask(self, h: mx.array, cache: Optional[Any]):
        if cache is not None and hasattr(cache, "make_mask"):
            return cache.make_mask(h.shape[1], return_array=False, window_size=None)
        if h.shape[1] == 1:
            return None
        return "causal"


class LanguageModel(DeepseekLanguageModel):
    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.model = DeepseekV2RingSlidingModel(config)

    def make_cache(self):
        window_size = self.config.sliding_window_size or self.config.sliding_window
        if window_size is None:
            return [KVCache() for _ in self.layers]
        return [RingSlidingKVCache(window_size) for _ in self.layers]


__all__ = ["LanguageModel", "RingSlidingKVCache"]
