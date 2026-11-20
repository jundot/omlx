# SPDX-License-Identifier: Apache-2.0
"""Adjacent-pair (GPT-J) RoPE helpers for Hy4.

Hy4 rotates channel pairs (2i, 2i+1). On text, its interleaved M-RoPE reads
the same position on all three axes, which is ordinary RoPE. Image tokens read
distinct time, height and width positions (``Positions``, ``rope_index``).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class TrailingRoPE(nn.Module):
    """RoPE on the trailing ``dims`` channels of each head.

    Hy4's indexers rotate the last ``qk_rope_head_dim`` channels, the opposite
    end from DeepSeek-V3.2's indexer.
    """

    def __init__(self, dims: int, base: float):
        super().__init__()
        self.dims = dims
        self.base = base

    def __call__(self, x: mx.array, offset=0) -> mx.array:
        head = x.shape[-1] - self.dims
        rotated = mx.fast.rope(
            x[..., head:],
            self.dims,
            traditional=True,
            base=self.base,
            scale=1.0,
            offset=offset,
        )
        if head == 0:
            return rotated
        return mx.concatenate([x[..., :head], rotated], axis=-1)


def mrope_channels(dims: int, section) -> np.ndarray | None:
    """Position plane (0 = time, 1 = height, 2 = width) of each frequency.

    The planes interleave: channels 1, 4, 7, ... read height, 2, 5, 8, ... read
    width (``section[1]`` and ``section[2]`` of each) and the rest read time.
    Returns None without a ``mrope_section``.
    """
    if section is None:
        return None
    if len(section) != 3 or sum(section) != dims // 2:
        raise ValueError(
            f"mrope_section {list(section)} does not cover {dims} channels"
        )
    planes = np.zeros(dims // 2, dtype=np.int32)
    planes[1 : section[1] * 3 : 3] = 1
    planes[2 : section[2] * 3 : 3] = 2
    return planes


def mrope_tables(planes: mx.array, dims: int, base: float, channels) -> tuple:
    """(cos, sin), each [B, L, dims // 2], of [3, B, L] position planes."""
    inv_freq = base ** (-mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    per_channel = mx.take(planes.astype(mx.float32), mx.array(channels), axis=0)
    theta = mx.moveaxis(per_channel, 0, -1) * inv_freq
    return mx.cos(theta), mx.sin(theta)


def rotate_pairs(x: mx.array, cos: mx.array, sin: mx.array, dims: int) -> mx.array:
    """Rotate the trailing ``dims`` channels of ``x`` by per-row cos/sin."""
    half = dims // 2
    head = x.shape[-1] - dims
    r = x[..., head:].astype(mx.float32).reshape(*x.shape[:-1], half, 2)
    even, odd = r[..., 0], r[..., 1]
    rotated = mx.stack([even * cos - odd * sin, odd * cos + even * sin], axis=-1)
    rotated = rotated.reshape(*x.shape[:-1], dims).astype(x.dtype)
    if head == 0:
        return rotated
    return mx.concatenate([x[..., :head], rotated], axis=-1)


def rope_at(x: mx.array, positions: mx.array, dims: int, base: float) -> mx.array:
    """RoPE of the trailing ``dims`` channels at explicit positions.

    ``x`` is [..., N, D]; ``positions`` holds one position per row N and may
    carry leading batch dimensions.
    """
    inv_freq = base ** (-mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    theta = positions.astype(mx.float32)[..., None] * inv_freq
    return rotate_pairs(x, mx.cos(theta), mx.sin(theta), dims)


def rotate(rope, x: mx.array, offset, positions) -> mx.array:
    """``rope`` at ``offset``, or at the call's ``positions`` when it has them."""
    return rope(x, offset) if positions is None else positions.rotate(x, rope)


class Positions:
    """Rotary positions of one forward call.

    ``planes`` holds the [3, B, L] time, height and width positions of the
    call's tokens. A ``general`` call (merged image embeddings) rotates with
    per-token tables; otherwise the three planes are the consecutive text
    positions ``offset + arange(L)`` and the stock RoPE kernels run with that
    per-row ``offset``. ``history`` is the [3, B, slots] position of every
    cache slot, current call included, when a cache keeps it.
    """

    def __init__(self, planes: mx.array, general: bool, section=None):
        self.planes = planes
        self.general = general
        self.section = section
        self.offset = None if general else planes[0, :, 0]
        self.history = None
        self._tables = {}

    def tables(self, base: float, dims: int) -> tuple:
        key = (base, dims)
        if key not in self._tables:
            self._tables[key] = self.tables_of(self.planes, base, dims)
        return self._tables[key]

    def tables_of(self, planes: mx.array, base: float, dims: int) -> tuple:
        """cos/sin of arbitrary [3, B, N] planes under this call's layout."""
        if self.section is None and self.general:
            raise ValueError("Image positions need rope_parameters.mrope_section")
        channels = mrope_channels(dims, self.section)
        if channels is None:
            # Equal text planes: any plane gives the same table.
            channels = np.zeros(dims // 2, dtype=np.int32)
        return mrope_tables(planes, dims, base, channels)

    def row(self, index: int) -> Positions:
        """The positions of row ``index`` alone, as its decode step had them."""
        row = Positions(
            self.planes[:, :, index : index + 1], self.general, self.section
        )
        row.history = self.history
        return row

    def rotate(self, x: mx.array, rope) -> mx.array:
        """``rope`` (nn.RoPE or TrailingRoPE) applied to [B, H, L, D] ``x``."""
        if not self.general:
            return rope(x, self.offset)
        cos, sin = self.tables(rope.base, rope.dims)
        return rotate_pairs(x, cos[:, None], sin[:, None], rope.dims)


def rope_index(
    input_ids: np.ndarray,
    image_token_id: int,
    grids,
    merge_size: int,
    attention_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Time, height and width positions of every token ([3, B, L]) and the
    per-row delta between the next decode position and the token count.

    Text advances all three planes together. An image of ``t x h x w`` patch
    cells fills ``t * (h / m) * (w / m + 1)`` placeholder slots (a newline slot
    ends every merged row). Its rows and columns count from the image's first
    position, each frame starts a block of ``max(rows, columns + 1)`` positions
    after the previous one, and text resumes after the image's largest
    position. Padding slots (``attention_mask == 0``) keep position 0.
    """
    ids = np.asarray(input_ids)
    batch, length = ids.shape
    planes = np.zeros((3, batch, length), dtype=np.int32)
    deltas = np.zeros((batch, 1), dtype=np.int64)
    pending = iter(grids)
    for row in range(batch):
        valid = np.ones(length, dtype=bool)
        if attention_mask is not None:
            valid = np.asarray(attention_mask)[row].astype(bool)
        is_image = ids[row][valid] == image_token_id
        parts, current, i = [], 0, 0
        while i < len(is_image):
            j = i
            while j < len(is_image) and is_image[j] == is_image[i]:
                j += 1
            if not is_image[i]:
                parts.append(np.broadcast_to(np.arange(j - i) + current, (3, j - i)))
                current += j - i
                i = j
                continue
            # A run of image slots can hold several back-to-back images.
            while i < j:
                grid = next(pending, None)
                if grid is None:
                    raise ValueError("The prompt holds more images than image_grid_thw")
                t, h, w = (int(v) for v in grid)
                rows, cols = h // merge_size, w // merge_size + 1
                if i + t * rows * cols > j:
                    raise ValueError("Image tokens do not match image_grid_thw")
                time = np.arange(t) * max(rows, cols)
                time, up, left = np.meshgrid(
                    time, np.arange(rows), np.arange(cols), indexing="ij"
                )
                block = np.stack([time, up + time, left + time]).reshape(3, -1)
                parts.append(block + current)
                current = int(block.max()) + 1 + current
                i += t * rows * cols
        positions = np.concatenate(parts, axis=1) if parts else np.zeros((3, 0))
        planes[:, row, valid] = positions
        deltas[row, 0] = (positions.max() + 1 if positions.size else 0) - len(is_image)
    if next(pending, None) is not None:
        raise ValueError("image_grid_thw lists more images than the prompt holds")
    return planes, deltas
