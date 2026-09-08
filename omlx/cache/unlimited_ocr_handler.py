# SPDX-License-Identifier: Apache-2.0
"""Persist the intact prompt prefix, never overwritten OCR decode-ring slots."""

from .type_handlers import CacheType, KVCacheHandler


class RingSlidingKVCacheHandler(KVCacheHandler):
    @property
    def cache_type(self):
        return CacheType.RING_SLIDING_KVCACHE

    def serialize_state(self, cache_obj):
        keys, values = cache_obj.state
        if keys is None or values is None:
            return keys, values
        prefix_length = cache_obj.prefill_length
        end = (
            min(cache_obj.offset, prefix_length)
            if prefix_length is not None
            else cache_obj.offset
        )
        return keys[:, :, :end, :], values[:, :, :end, :]

    def serialize_meta_state(self, cache_obj):
        keys, _ = self.serialize_state(cache_obj)
        length = 0 if keys is None else keys.shape[2]
        # Storage is a prefill prefix, even when captured after decoding.
        return (str(cache_obj.window_size), "-1", str(length), "0")

    def extract_state(self, cache_obj):
        keys, values = self.serialize_state(cache_obj)
        return {
            "keys": keys,
            "values": values,
            "offset": 0 if keys is None else keys.shape[2],
            "cache_type": self.cache_type.value,
        }

    @staticmethod
    def validate_prefix_metadata(meta_state):
        """Return window/length only for intact, not post-decode, captures."""
        if not isinstance(meta_state, (list, tuple)) or len(meta_state) != 4:
            raise ValueError("Missing or invalid Unlimited-OCR ring metadata")
        window, prefix_length, stored_length, ring_pos = map(int, meta_state)
        if window <= 0 or prefix_length != -1 or ring_pos != 0 or stored_length <= 0:
            raise ValueError("Invalid Unlimited-OCR intact-prefix metadata")
        return window, stored_length

    def reconstruct_cache(self, state, meta_state=None):
        from .unlimited_ocr import OMLXRingSlidingKVCache

        window, _ = self.validate_prefix_metadata(meta_state)
        keys, values = state.get("keys"), state.get("values")
        if (
            keys is None
            or values is None
            or keys.ndim != 4
            or values.ndim != 4
            or keys.shape[:3] != values.shape[:3]
            or keys.shape[0] != 1
            or keys.shape[2] <= 0
        ):
            raise ValueError("Invalid Unlimited-OCR intact-prefix cache state")
        cache = OMLXRingSlidingKVCache(window)
        # Deduplicated chains use the first block's capture metadata. Its
        # stored_length may be shorter than this extended, concatenated prefix;
        # per-block tensor-length validation already happens in prefix_cache.
        cache.state = (keys, values)
        # Do not restore the old decode boundary. The request may extend this
        # prefix; the native cache establishes its boundary on first decode.
        return cache
