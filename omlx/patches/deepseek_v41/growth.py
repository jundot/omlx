# SPDX-License-Identifier: MIT
"""In-place growth for the packed compressed KV and index-key cache slots.

Decode appends at most a few rows per token to slots that hold the whole
context. A concatenation copies the full slot on every token, so each slot
keeps two buffers instead:

- The cache holds a view of the current buffer.
- A short append writes the rows the idle buffer is missing (the previous
  append and the new rows) into the idle buffer and makes it current.
- Nothing else refers to the idle buffer by then, so MLX updates it in place.

Values never change: if a stray reference blocks the in-place update, MLX
copies instead. Replacing, slicing or extracting a slot outside this module
breaks the chain; the next append then starts a new buffer from the prefix.
"""

import mlx.core as mx

# Longer appends (prefill chunks) allocate one exact buffer, so prefill memory
# is unchanged.
_MAX_ROWS = 64


def _capacity(end, rows):
    if rows > _MAX_ROWS:
        return (max(end, 256) + 255) // 256 * 256
    return (end + max(4096, end // 16) + 255) // 256 * 256


def _registry(cache):
    buffers = getattr(cache, "_ds41_buffers", None)
    if buffers is None:
        buffers = cache._ds41_buffers = {}
    return buffers


def append(cache, slot, previous, values, start):
    """Return ``previous[:, :start]`` followed by ``values`` as a buffer view."""
    buffers = _registry(cache)
    entry = buffers.get(slot)
    chain = (
        entry is not None
        and cache[slot] is entry["view"]
        and start <= entry["end"]
        and previous.shape[1] >= start
    )
    if not values.shape[1]:
        # Nothing to add: keep the current view so the chain continues.
        if chain and start == entry["end"]:
            return entry["view"]
        return previous[:, :start]
    end = start + values.shape[1]
    short = values.shape[1] <= _MAX_ROWS
    buffer = None
    if chain and short:
        idle, valid = entry["idle"], min(entry["valid"], start)
        entry["idle"] = None
        if (
            idle is not None
            and idle.shape[1] >= end
            and idle.shape[0] == values.shape[0]
            and idle.shape[2:] == values.shape[2:]
            and idle.dtype == values.dtype
        ):
            update = (
                values
                if valid == start
                else mx.concatenate([previous[:, valid:start], values], 1)
            )
            # The registry no longer holds ``idle``, so this update can reuse
            # its storage.
            idle[:, valid:end] = update
            buffer = idle
        del idle
    if buffer is None:
        parts = [previous[:, :start], values] if start else [values]
        capacity = _capacity(end, values.shape[1])
        if capacity > end:
            parts.append(
                mx.zeros(
                    (values.shape[0], capacity - end, *values.shape[2:]), values.dtype
                )
            )
        buffer = mx.concatenate(parts, 1)
    keep = chain and short
    view = buffer[:, :end]
    buffers[slot] = {
        "buffer": buffer,
        "view": view,
        "end": end,
        # The buffer behind ``previous`` holds valid rows [0, start).
        "idle": entry["buffer"] if keep else None,
        "valid": start if keep else 0,
    }
    return view


def truncate(cache, slot, length):
    """Set ``cache[slot]`` to its first ``length`` rows and keep the chain."""
    view = cache[slot][:, :length]
    entry = _registry(cache).get(slot)
    if entry is not None and cache[slot] is entry["view"] and length <= entry["end"]:
        entry["view"], entry["end"] = view, length
    cache[slot] = view
    return view
