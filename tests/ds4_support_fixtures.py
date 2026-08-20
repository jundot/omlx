# SPDX-License-Identifier: Apache-2.0
"""Pinned DS4 support-tree fixtures used by tests."""

from omlx.ds4_support import DS4_SERVER_BINARY, DS4_SUPPORT_FILES

PINNED_DS4_METAL_FILES: tuple[str, ...] = (
    "flash_attn.metal",
    "dense.metal",
    "moe.metal",
    "dsv4_hc.metal",
    "unary.metal",
    "dsv4_kv.metal",
    "dsv4_rope.metal",
    "dsv4_misc.metal",
    "argsort.metal",
    "cpy.metal",
    "concat.metal",
    "get_rows.metal",
    "sum_rows.metal",
    "softmax.metal",
    "repeat.metal",
    "glu.metal",
    "norm.metal",
    "bin.metal",
    "set_rows.metal",
)


def pinned_ds4_support_relative_paths(
    *,
    include_binary: bool = True,
) -> tuple[str, ...]:
    """Return expected paths for the pinned upstream DS4 test fixture."""
    paths: list[str] = []
    if include_binary:
        paths.append(DS4_SERVER_BINARY)
    paths.extend(DS4_SUPPORT_FILES)
    paths.extend(f"metal/{name}" for name in PINNED_DS4_METAL_FILES)
    return tuple(paths)
