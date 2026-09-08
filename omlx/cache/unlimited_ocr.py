# SPDX-License-Identifier: Apache-2.0
"""Native Unlimited-OCR ring cache on oMLX's serial generation lane.

Import lazily after the Unlimited-OCR compatibility loader has run: the pinned
mlx-vlm may obtain the native model package from oMLX's vendor namespace.
"""

from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache


class OMLXRingSlidingKVCache(RingSlidingKVCache):
    """Keep the native attention layout; reject lossy batch conversions."""

    def __init__(self, window_size):
        super().__init__(window_size)
        self._prefill_end = None

    def set_prefill_end(self, end):
        """Declare the N-1 boundary before any size-one prefill chunks."""
        if self.prefill_length is not None or end < self.offset:
            raise ValueError("Cannot reopen or move back the ring prefill boundary")
        self._prefill_end = end

    def update_and_fetch(self, keys, values):
        if self._prefill_end is not None:
            if self.offset < self._prefill_end:
                if self.offset + keys.shape[2] > self._prefill_end:
                    raise ValueError("Prefill chunk crosses the ring decode boundary")
                # Native ring code treats every size-one call as decode. The
                # scheduler knows whether it is still processing prompt KV.
                return KVCache.update_and_fetch(self, keys, values)
            self._prefill_end = None
        return super().update_and_fetch(keys, values)

    @classmethod
    def merge(cls, caches):
        if len(caches) != 1:
            raise ValueError("Unlimited-OCR ring caches require serial requests")
        return caches[0]

    def to_batch(self, left_padding):
        if list(left_padding) != [0]:
            raise ValueError(
                "Unlimited-OCR ring caches require serial unpadded requests"
            )
        return self

    def filter(self, batch_indices):
        indices = list(batch_indices)
        if indices == [0]:
            return
        if not indices:
            self.keys = self.values = None
            self.offset = 0
            self.prefill_length = None
            self._ring_pos = 0
            self._prefill_end = None
            return
        raise ValueError("Unlimited-OCR ring caches require serial requests")

    def extract(self, idx):
        if int(idx) != 0:
            raise IndexError("Unlimited-OCR ring cache only has row 0")
        return self

    def extend(self, other):
        raise ValueError("Unlimited-OCR ring caches require serial requests")

    def is_trimmable(self):
        # Prefill prefixes are contiguous; decode has overwritten history.
        return self.prefill_length is None

    def trim(self, n):
        if not self.is_trimmable():
            raise ValueError("Cannot trim a decoded Unlimited-OCR ring cache")
        return super().trim(n)
