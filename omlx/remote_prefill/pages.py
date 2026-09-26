# SPDX-License-Identifier: Apache-2.0
"""Turn a frame of exported cache pages into token-major key and value arrays."""

from __future__ import annotations

from typing import Any

from .wire import DTYPE_BYTES, LayerExport, WireError

# Unsigned types that carry each element width's bits unchanged.
_CARRIER = {2: "uint16", 4: "uint32"}


def carrier_dtype(mx: Any, layer: LayerExport) -> Any:
    """The unsigned MLX type a frame's bytes are copied in, before they are reinterpreted."""
    return getattr(mx, _CARRIER[DTYPE_BYTES[layer.dtype]])


def frame_shape(layer: LayerExport, rows: int) -> tuple[int, ...]:
    """The array shape of a frame holding `rows` page rows of `layer`."""
    return tuple(
        rows if dim == "block" else size for dim, size in zip(layer.dims, layer.shape)
    )


def _flatten(part: Any, dims: list[str], inner: tuple[str, ...]) -> Any:
    """Order `part` as block, token, then `inner`, and merge block and token into one axis."""
    order = [dims.index(name) for name in ("block", "token", *inner)]
    moved = part.transpose(order)
    return moved.reshape(-1, *moved.shape[2:])


def split_pages(mx: Any, layer: LayerExport, pages: Any) -> tuple[Any, Any]:
    """Keys and values as [tokens, heads, head_size]; for MLA the latent and rope parts as [tokens, n]."""
    dims = list(layer.dims)
    typed = pages.view(getattr(mx, layer.dtype))
    if layer.kind == "mla":
        if set(dims) != {"block", "token", "latent"}:
            raise WireError(f"layer {layer.index} MLA pages have dims {layer.dims}")
        flat = _flatten(typed, dims, ("latent",))
        if layer.latent_size + layer.rope_size != flat.shape[-1] or not layer.rope_size:
            raise WireError(
                f"layer {layer.index} latent and rope sizes do not fill its rows"
            )
        return flat[:, : layer.latent_size], flat[:, layer.latent_size :]
    if "kv" in dims:
        axis = dims.index("kv")
        if typed.shape[axis] != 2:
            raise WireError(
                f"layer {layer.index} kv dim holds {typed.shape[axis]} parts"
            )
        keys, values = (
            mx.squeeze(part, axis) for part in mx.split(typed, 2, axis=axis)
        )
        dims.pop(axis)
    elif "kv_head_dim" in dims:
        axis = dims.index("kv_head_dim")
        keys, values = mx.split(typed, 2, axis=axis)
        dims[axis] = "head_dim"
    else:
        raise WireError(f"layer {layer.index} pages hold neither kv nor kv_head_dim")
    if sorted(dims) != sorted(("block", "token", "head", "head_dim")):
        raise WireError(f"layer {layer.index} attention pages have dims {layer.dims}")
    keys, values = (
        _flatten(part, dims, ("head", "head_dim")) for part in (keys, values)
    )
    if keys.shape[1:] != (layer.heads, layer.head_size):
        raise WireError(
            f"layer {layer.index} pages hold {keys.shape[1:]} heads by size, "
            f"not {(layer.heads, layer.head_size)}"
        )
    return keys, values
