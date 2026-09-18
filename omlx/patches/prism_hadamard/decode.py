# SPDX-License-Identifier: Apache-2.0
"""Preserve allocated KV capacity through single-row Qwen3.5 decode."""

from mlx_vlm.models.cache import BatchKVCache, KVCache
from mlx_vlm.models.qwen3_5.language import Qwen3_5Model


class CapacityPreservingModel(Qwen3_5Model):
    """Use the existing unbatched decoder without extracting/copying its KV.

    mlx-vlm's singleton batch path extracts every KV layer, forwards one token,
    then merges every layer back into a tightly sized BatchKVCache. That drops
    spare capacity and copies the full prefix on the next token too. At long
    contexts the retired buffers can exhaust the serving process's pool.

    An unpadded one-row cache already has the unbatched layout. Borrow its
    backing arrays and restore only the updated references/offsets. Padded,
    quantized, multi-row, and prefill calls retain the upstream path.
    """

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        position_ids=None,
        capture_layer_ids=None,
        hidden_sink=None,
    ):
        rows = None
        borrowed = []
        if (
            cache is not None
            and inputs is not None
            and inputs.shape[:2] == (1, 1)
            and hidden_sink is None
            and capture_layer_ids is None
            and type(cache[self.fa_idx]) is BatchKVCache
        ):
            candidates = [
                (i, entry)
                for i, entry in enumerate(cache)
                if isinstance(entry, BatchKVCache)
            ]
            if candidates and all(
                type(entry) is BatchKVCache
                and entry.keys is not None
                and entry.keys.shape[0] == 1
                and entry.left_padding.tolist() == [0]
                and entry._right_padding is None
                for _, entry in candidates
            ):
                rows = list(cache)
                for i, entry in candidates:
                    row = KVCache()
                    row.keys, row.values = entry.keys, entry.values
                    row.offset = entry._idx
                    rows[i] = row
                    borrowed.append((entry, row))

        output = super().__call__(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            capture_layer_ids=capture_layer_ids,
            cache=rows if rows is not None else cache,
            position_ids=position_ids,
            hidden_sink=hidden_sink,
        )
        for entry, row in borrowed:
            entry.keys, entry.values = row.keys, row.values
            entry.offset = entry.offset + (row.offset - entry._idx)
            entry._idx = row.offset
        return output
