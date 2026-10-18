"""Packed prefill: chunks of several requests in one forward.

Token-wise layers (embeddings, norms, projections, MLP/MoE, hyper-connections)
see one packed sequence of shape (1, T). Each registered sequence mixer
(attention, GDN/KDA, PLE) runs once per row with that row's own cache, mask
and positions, which are the arguments a single-request chunk passes. Rows
never share a cache, so they can start at any offset and use any length.

Every layer receives a ``PackedRows`` stand-in instead of a cache. It raises
on any access that a registered mixer does not handle, so an unregistered
cache consumer fails the forward instead of mixing rows.

A row matches its single-request chunk bit for bit when every token-wise op
picks the same kernel for the packed forward as for the row alone. mlx picks
some kernels by row count, so:
- registered row ops (row-count dependent and cheap) also run once per row;
- rows shorter than ``packed_min_row_tokens`` stay out of packs.
"""

from __future__ import annotations

import contextvars
import importlib
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

PACKED_MASK = object()

# Fewer rows take mlx's matrix-vector projection kernels (and model decode
# kernels) alone, where a packed forward takes the matrix kernels.
_MIN_PACKED_ROW_TOKENS = 64
# mlx runs sorted expert rows through its segmented gather kernel only from
# this many rows per expert on average; fewer take the per-row kernel.
_GATHER_ROWS_PER_EXPERT = 4
# MoE families whose packed rows match their single-request chunks. Qwen MoE
# routers and shared-expert gates pick kernels by row count at any length, so
# holding their short rows back would cost throughput without a match.
_EXACT_MOE_FAMILIES = {"glm5_next", "hy_v3"}

_active_batch: contextvars.ContextVar[PackedBatch | None] = contextvars.ContextVar(
    "omlx_packed_batch", default=None
)


@dataclass(frozen=True)
class PackedRow:
    """One request's next chunk, with the cache it extends."""

    request_id: str
    tokens: mx.array
    cache: list[Any]
    rope_delta: float = 0.0


class PackedBatch:
    def __init__(self, rows: Sequence[PackedRow]):
        self.rows = tuple(rows)
        spans = []
        start = 0
        for row in self.rows:
            length = int(row.tokens.shape[-1])
            spans.append((start, start + length))
            start += length
        self.spans = tuple(spans)
        self.total_tokens = start


class PackedRows:
    """Per-layer cache stand-in that owns the row caches of one forward."""

    # Model-level mask and offset helpers read these.
    offset = 0
    left_padding = None
    lengths = None

    def __init__(self, batch: PackedBatch, caches: Sequence[Any]):
        self.__dict__["batch"] = batch
        self.__dict__["caches"] = tuple(caches)

    def make_mask(self, *args, **kwargs):
        return PACKED_MASK

    def __getitem__(self, index):
        return PackedRows(self.batch, [cache[index] for cache in self.caches])

    def __bool__(self):
        return True

    def __getattr__(self, name):
        raise AttributeError(f"packed prefill does not support cache access {name!r}")


def packed_batch_of(cache: Any) -> PackedBatch | None:
    """Return the batch behind a model-level cache list, if it is packed."""
    if isinstance(cache, (list, tuple)) and cache and isinstance(cache[0], PackedRows):
        return cache[0].batch
    return None


def _slice_tokens(value: Any, start: int, end: int, total: int) -> Any:
    if isinstance(value, mx.array):
        if value.ndim >= 2 and value.shape[-1] == total:
            return value[..., start:end]
        return value
    if (
        isinstance(value, tuple)
        and value
        and all(isinstance(item, mx.array) for item in value)
    ):
        # (cos, sin) position embeddings keep tokens on axis 1.
        return tuple(
            item[:, start:end] if item.ndim >= 2 and item.shape[1] == total else item
            for item in value
        )
    return value


def _wrap_mixer(
    cls: type,
    mask_fn: Callable[[Any, Any], Any],
    *,
    x_arg: str,
    token_args: tuple[str, ...] = (),
) -> None:
    current = cls.__call__
    if getattr(current, "_omlx_packed_mixer", False):
        return
    signature = inspect.signature(current)
    var_kwargs = [
        p.name
        for p in signature.parameters.values()
        if p.kind is inspect.Parameter.VAR_KEYWORD
    ]
    has_mask = "mask" in signature.parameters

    def packed_call(self, *args, **kwargs):
        # Decode and unpacked prefill skip the argument binding.
        if not isinstance(kwargs.get("cache"), PackedRows) and not any(
            isinstance(arg, PackedRows) for arg in args
        ):
            return current(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        rows = bound.arguments.get("cache")
        if not isinstance(rows, PackedRows):
            return current(self, *args, **kwargs)
        arguments = dict(bound.arguments)
        del arguments["self"]
        extra = {}
        for name in var_kwargs:
            extra.update(arguments.pop(name, {}) or {})
        x = arguments[x_arg]
        total = x.shape[1]
        outputs = []
        for cache, (start, end) in zip(rows.caches, rows.batch.spans):
            row_args = dict(arguments)
            row_x = x[:, start:end]
            row_args[x_arg] = row_x
            row_args["cache"] = cache
            if has_mask:
                row_args["mask"] = mask_fn(row_x, cache)
            for name in token_args:
                if row_args.get(name) is not None:
                    row_args[name] = row_args[name][:, start:end]
            for name in ("position_ids", "position_embeddings"):
                if row_args.get(name) is not None:
                    row_args[name] = _slice_tokens(row_args[name], start, end, total)
            outputs.append(current(self, **row_args, **extra))
        return mx.concatenate(outputs, axis=1)

    packed_call._omlx_packed_mixer = True
    packed_call.__wrapped__ = current
    cls.__call__ = packed_call


def _wrap_row_op(cls: type) -> None:
    current = cls.__call__
    if getattr(current, "_omlx_packed_row_op", False):
        return

    def row_call(self, x, *args, **kwargs):
        batch = _active_batch.get()
        if batch is None or x.ndim < 2 or x.shape[1] != batch.total_tokens:
            return current(self, x, *args, **kwargs)
        outputs = [
            current(self, x[:, start:end], *args, **kwargs)
            for start, end in batch.spans
        ]
        if isinstance(outputs[0], tuple):
            return tuple(mx.concatenate(parts, axis=1) for parts in zip(*outputs))
        return mx.concatenate(outputs, axis=1)

    row_call._omlx_packed_row_op = True
    row_call.__wrapped__ = current
    cls.__call__ = row_call


def _qwen3_5_mixers():
    q = importlib.import_module("mlx_vlm.models.qwen3_5.language")
    return [
        (q.Qwen3_5GatedDeltaNet, q._create_qwen3_5_ssm_mask, "inputs", ()),
        (q.Qwen3_5Attention, q._create_qwen3_5_attention_mask, "x", ()),
    ]


def _qwen4_exp_mixers():
    q = importlib.import_module("mlx_vlm.models.qwen3_5.language")
    q4 = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
    return [
        (q4.Qwen4ExpGatedDeltaNet, q._create_qwen3_5_ssm_mask, "inputs", ()),
        (q4.Qwen4ExpAttention, q._create_qwen3_5_attention_mask, "x", ()),
        # PLE mixes the sequence through its n-gram history and short conv.
        (
            q4.Qwen4ExpPLELayer,
            q._create_qwen3_5_ssm_mask,
            "hidden_states",
            ("input_ids",),
        ),
    ]


def _glm5_next_mixers():
    base = importlib.import_module("mlx_vlm.models.base")
    g = importlib.import_module("mlx_vlm.models.glm5_next.language")
    return [
        (g.Glm5NextLinearAttention, base.create_ssm_mask, "inputs", ()),
        (
            g.Glm5NextSparseAttention,
            lambda x, cache: base.create_attention_mask(x, cache[0], return_array=True),
            "x",
            (),
        ),
    ]


def _glm5_next_row_ops():
    g = importlib.import_module("mlx_vlm.models.glm5_next.language")
    # The fp32 router GEMM picks its kernel by row count on non-NAX GPUs.
    return [g.Glm5NextMoEGate]


def _mlx_lm_qwen3_5_mixers():
    base = importlib.import_module("mlx_lm.models.base")
    q = importlib.import_module("mlx_lm.models.qwen3_5")
    return [
        (q.GatedDeltaNet, base.create_ssm_mask, "inputs", ()),
        (q.Attention, base.create_attention_mask, "x", ()),
    ]


def _hy_v3_mixers():
    base = importlib.import_module("mlx_lm.models.base")
    h = importlib.import_module("mlx_lm.models.hy_v3")
    return [(h.Attention, base.create_attention_mask, "x", ())]


def _hy_v3_row_ops():
    h = importlib.import_module("mlx_lm.models.hy_v3")
    # Same fp32 router input as GLM-5.3. The narrow shared-expert MLP takes
    # mlx's split-K matmul for short rows, with a split count set by row count.
    return [h.MoEGate, h.MLP]


# Families keyed by model type: VLM adapters by the wrapped language model,
# mlx-lm models by ``args.model_type``. A family is added once its packed rows
# are checked against single-request chunks on the real model.
_VLM_FAMILIES = {
    "qwen3_5": _qwen3_5_mixers,
    "qwen3_5_moe": _qwen3_5_mixers,
    "qwen4_exp": _qwen4_exp_mixers,
    "glm5_next": _glm5_next_mixers,
}
_MLX_LM_FAMILIES = {"qwen3_5_moe": _mlx_lm_qwen3_5_mixers, "hy_v3": _hy_v3_mixers}
_ROW_OPS = {"glm5_next": _glm5_next_row_ops, "hy_v3": _hy_v3_row_ops}


def _is_vlm_adapter(model: Any) -> bool:
    return hasattr(model, "_language_model") and hasattr(model, "_vlm_model")


def packed_prefill_family(model: Any) -> str | None:
    """Return the registered family of ``model``, or None if unsupported."""
    if _is_vlm_adapter(model):
        family = getattr(model, "model_type", None)
        return family if family in _VLM_FAMILIES else None
    family = getattr(getattr(model, "args", None), "model_type", None)
    return family if family in _MLX_LM_FAMILIES else None


def install_packed_prefill(model: Any) -> bool:
    """Wrap the family's sequence mixers; safe to repeat after other patches."""
    family = packed_prefill_family(model)
    if family is None:
        return False
    registry = _VLM_FAMILIES if _is_vlm_adapter(model) else _MLX_LM_FAMILIES
    for cls, mask_fn, x_arg, token_args in registry[family]():
        _wrap_mixer(cls, mask_fn, x_arg=x_arg, token_args=token_args)
    for cls in _ROW_OPS.get(family, list)():
        _wrap_row_op(cls)
    return True


def packed_min_row_tokens(model: Any) -> int:
    """Shortest chunk that may share a forward with other rows."""
    args = getattr(getattr(model, "_language_model", model), "args", None)
    experts = getattr(args, "n_routed_experts", None) or getattr(
        args, "num_experts", None
    )
    top_k = getattr(args, "num_experts_per_tok", None)
    if packed_prefill_family(model) in _EXACT_MOE_FAMILIES and experts and top_k:
        gather_rows = -(-_GATHER_ROWS_PER_EXPERT * int(experts) // int(top_k))
        return max(_MIN_PACKED_ROW_TOKENS, gather_rows)
    return _MIN_PACKED_ROW_TOKENS


def _cache_offset(cache: list[Any]) -> int:
    # Same source as VLMModelAdapter: the first layer cache with an offset.
    for layer_cache in cache:
        if hasattr(layer_cache, "offset"):
            offset = layer_cache.offset
            return int(offset.item()) if isinstance(offset, mx.array) else int(offset)
    return 0


def _position_ids(model: Any, batch: PackedBatch) -> mx.array | None:
    if not getattr(model, "_uses_mrope", False):
        return None
    # Each row gets the positions a single-row text chunk would.
    rows = [
        model._position_ids_from_starts(
            mx.array([_cache_offset(row.cache) + int(row.rope_delta)], dtype=mx.int32),
            1,
            end - begin,
            qwen4_text_prefill_positions=True,
        )
        for row, (begin, end) in zip(batch.rows, batch.spans)
    ]
    return mx.concatenate(rows, axis=-1)


def run_packed_prefill(model: Any, rows: Sequence[PackedRow]) -> Any:
    """Run one forward over ``rows`` and return the lazy model output.

    Prefill callers evaluate only the row caches, so logits stay unbuilt.
    """
    batch = PackedBatch(rows)
    layer_count = len(batch.rows[0].cache)
    caches = [
        PackedRows(batch, [row.cache[layer] for row in batch.rows])
        for layer in range(layer_count)
    ]
    inputs = mx.concatenate([row.tokens for row in batch.rows], axis=1)
    token = _active_batch.set(batch)
    try:
        if not _is_vlm_adapter(model):
            return model(inputs, cache=caches)
        kwargs = {}
        position_ids = _position_ids(model, batch)
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if model.model_type == "qwen4_exp":
            kwargs["skip_logits"] = True
        return model._language_model(inputs, cache=caches, **kwargs)
    finally:
        _active_batch.reset(token)
