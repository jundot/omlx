# SPDX-License-Identifier: Apache-2.0
"""Storage-only block deltas for DeepSeek V4.1's packed cumulative state."""

try:
    import mlx.core as mx
except ImportError:
    mx = None

DELTA_CLASS = "DeepseekV41Delta"
CACHE_CLASS = "DeepseekV41Cache"


def _ratio(meta):
    if tuple(meta) == ("deepseek_v41", "2"):
        return None  # Legacy full snapshots remain readable without compaction.
    if tuple(meta[:2]) != ("deepseek_v41", "3") or len(meta) != 3:
        raise ValueError("Unknown V4.1 cache metadata")
    ratio = int(meta[2])
    if ratio < 0:
        raise ValueError("Invalid V4.1 compression ratio")
    return ratio


def compact_state(state, meta, start, end):
    """Return bounded state plus one block's rows and an absolute range record."""
    ratio = _ratio(meta)
    if ratio is None:
        return state
    if not 0 <= start < end:
        raise ValueError("Invalid V4.1 snapshot token range")
    row_start, row_end = (start // ratio, end // ratio) if ratio else (0, 0)
    record = [1, start, end, row_start, row_end, ratio]
    if len(state) == 8:
        if state[7].tolist() != record:
            raise ValueError("V4.1 delta belongs to another boundary")
        return state
    if len(state) != 7 or state[0].tolist() != [end]:
        raise ValueError("V4.1 snapshot offset does not match its boundary")
    result = list(state)
    for slot in (2, 3):
        value = state[slot]
        if (
            value.ndim != 3
            or value.shape[:2] != (1, row_end)
            or value.dtype != mx.uint8
        ):
            raise ValueError("Invalid cumulative V4.1 packed state")
        # Gather materializes only the delta; a contiguous slice may retain
        # the entire cumulative allocation in an in-memory snapshot.
        result[slot] = (
            mx.take(value, mx.arange(row_start, row_end), axis=1)
            if row_end > row_start
            else mx.zeros((1, 0, value.shape[2]), value.dtype)
        )
    result.append(mx.array(record, mx.int64))
    return tuple(result)


def compact_snapshot(extracted, token_count, block_size):
    """Compact temporary boundary files and in-memory boundary snapshots alike."""
    if token_count <= 0 or block_size <= 0:
        return extracted
    start = ((token_count - 1) // block_size) * block_size
    for layer in extracted:
        if layer.get("class_name") == CACHE_CLASS:
            layer["state"] = compact_state(
                layer["state"], layer.get("meta_state", ()), start, token_count
            )
    return extracted


def restore_chain(markers, metas, token_count):
    """Validate every absolute range before joining packed rows; reject gaps."""
    from ..patches.deepseek_v41.cache import DeepseekV41Cache

    parts = {2: [], 3: []}
    cursor = rows = 0
    last = last_meta = None
    for marker, meta in zip(markers, metas, strict=True):
        if (
            not isinstance(marker, tuple)
            or len(marker) != 3
            or marker[0] != "__nstate__"
        ):
            raise ValueError("Missing V4.1 boundary state")
        name, state = marker[1:]
        if name == CACHE_CLASS and len(state) == 7:
            # A legacy full checkpoint can anchor a subsequent delta chain.
            last, last_meta = list(state), meta
            cursor = int(state[0].item())
            ratio = _ratio(meta)
            if cursor <= 0 or state[0].shape != (1,):
                raise ValueError("Invalid V4.1 full snapshot offset")
            for slot in (2, 3):
                if (
                    state[slot].ndim != 3
                    or state[slot].shape[0] != 1
                    or state[slot].dtype != mx.uint8
                ):
                    raise ValueError("Invalid V4.1 full packed state")
            rows = state[2].shape[1]
            if state[3].shape[1] != rows or (
                ratio is not None and rows != (cursor // ratio if ratio else 0)
            ):
                raise ValueError("V4.1 KV/index lengths diverge")
            parts = {slot: [state[slot]] for slot in (2, 3)}
            continue
        if name != DELTA_CLASS or len(state) != 8:
            raise ValueError("Unknown V4.1 persisted state")
        record = state[7]
        if record.shape != (6,) or record.dtype != mx.int64:
            raise ValueError("Invalid V4.1 delta range record")
        version, start, end, first, stop, ratio = record.tolist()
        if (
            version != 1
            or ratio != _ratio(meta)
            or start != cursor
            or end <= start
            or first != rows
            or (first, stop) != ((start // ratio, end // ratio) if ratio else (0, 0))
            or state[0].tolist() != [end]
        ):
            raise ValueError("Incomplete or out-of-order V4.1 delta chain")
        if last_meta is not None and _ratio(last_meta) not in (None, ratio):
            raise ValueError("V4.1 compression ratio changed across blocks")
        for slot in (2, 3):
            value = state[slot]
            if (
                value.ndim != 3
                or value.shape[:2] != (1, stop - first)
                or value.dtype != mx.uint8
                or (parts[slot] and value.shape[2:] != parts[slot][-1].shape[2:])
            ):
                raise ValueError("V4.1 packed delta shape mismatch")
            parts[slot].append(value)
        cursor, rows, last, last_meta = end, stop, list(state[:7]), meta
    if last is None or cursor != token_count:
        raise ValueError("V4.1 restored offset does not match the cache hit")
    for slot in (2, 3):
        last[slot] = mx.concatenate(parts[slot], axis=1)
    return DeepseekV41Cache.from_state(last, last_meta)
