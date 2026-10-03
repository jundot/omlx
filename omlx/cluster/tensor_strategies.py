# SPDX-License-Identifier: Apache-2.0
"""Capability-gated, layer-at-a-time tensor-parallel sharding.

MLX model sharding is architecture-specific. This module keeps an explicit
registry for oMLX adapters and uses a carefully bounded native fallback for
models whose installed MLX-LM class already implements ``shard()``.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class TensorStrategy:
    name: str
    model_types: tuple[str, ...]
    source: str


_ADAPTERS: dict[str, Callable[..., None]] = {}


def _register(
    strategy: TensorStrategy,
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    def decorator(function: Callable[..., None]) -> Callable[..., None]:
        for model_type in strategy.model_types:
            if model_type in _ADAPTERS:
                raise RuntimeError(f"duplicate tensor strategy for {model_type}")
            _ADAPTERS[model_type] = function
        function._omlx_tensor_strategy = strategy  # type: ignore[attr-defined]
        return function

    return decorator


QWEN3_NEXT = TensorStrategy(
    name="qwen3_next",
    model_types=("qwen3_next", "qwen3_next_moe"),
    source="oMLX adapter derived from Exo's section-aware Qwen strategy",
)
NEMOTRON_H = TensorStrategy(
    name="nemotron_h",
    model_types=("nemotron_h",),
    source="oMLX adapter derived from Exo's attention/Mamba/MoE strategy",
)
QWEN4_EXP = TensorStrategy(
    name="qwen4_exp",
    model_types=("qwen4_exp", "qwen4_exp_text"),
    source="oMLX adapter: GDN/attention/MoE sharding + rank-local PLE n-gram table",
)


def registered_model_types() -> frozenset[str]:
    return frozenset(_ADAPTERS)


def supports_model_type(model_type: str, *, native_shard: bool = False) -> bool:
    return bool(native_shard or model_type in _ADAPTERS)


def _model_type(model: Any) -> str:
    for candidate in (
        getattr(model, "model_type", None),
        getattr(getattr(model, "args", None), "model_type", None),
        getattr(getattr(model, "config", None), "model_type", None),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return type(model).__name__.lower()


def _emit(
    callback: ProgressCallback | None,
    *,
    strategy: str,
    layer: int,
    loaded: int,
    total: int,
) -> None:
    if callback is not None:
        callback(
            {
                "phase": "tensor_sharding",
                "strategy": strategy,
                "layer": layer,
                "layers_loaded": loaded,
                "layers_total": total,
            }
        )


def _common_layer_owner(model: Any) -> tuple[Any, list[Any]]:
    """Find the concrete module that owns the mutable transformer layer list."""

    queue = [model]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        descriptor = inspect.getattr_static(type(candidate), "layers", None)
        read_only_property = (
            isinstance(descriptor, property) and descriptor.fset is None
        )
        layers = getattr(candidate, "layers", None)
        if isinstance(layers, list) and not read_only_property:
            return candidate, layers
        for name in ("model", "backbone", "language_model", "transformer"):
            child = getattr(candidate, name, None)
            if child is not None and not isinstance(child, (str, bytes)):
                queue.append(child)
    raise RuntimeError(
        f"tensor strategy cannot locate the layer container on {type(model).__name__}"
    )


def _native_layerwise_shard(
    model: Any,
    group: Any,
    mx: Any,
    progress: ProgressCallback | None,
) -> None:
    """Run a native ``shard()`` one materialized layer at a time.

    Native MLX-LM implementations in the pinned release iterate only their
    layer list. Temporarily presenting one layer preserves their
    architecture-specific logic while ensuring the unsharded layer is
    materialized before FAST_SYNCH sees any sharding graph.
    """

    supported, reason = native_shard_is_layer_local(getattr(model, "shard", None))
    if not supported:
        raise RuntimeError(
            "native tensor strategy cannot safely shard one layer at a time: "
            + reason
        )
    owner, layers = _common_layer_owner(model)
    original = list(layers)
    if not original or any(layer is None for layer in original):
        raise RuntimeError("native tensor sharding requires a complete model")
    total = len(original)
    try:
        for index, layer in enumerate(original):
            mx.eval(layer.parameters())
            owner.layers = [layer]
            model.shard(group)
            mx.eval(layer.parameters())
            mx.clear_cache()
            _emit(
                progress,
                strategy="native",
                layer=index,
                loaded=index + 1,
                total=total,
            )
    finally:
        owner.layers = original


def native_shard_is_layer_local(shard: Any) -> tuple[bool, str]:
    """Prove repeated native ``shard()`` calls cannot re-shard fixed weights.

    Progressive loading temporarily exposes one transformer layer and invokes
    the installed architecture's native method once per layer. That is safe
    only when the method's mutating work lives in one top-level layer loop.
    Architectures with embedding/head sharding outside that loop must get an
    explicit adapter instead of being guessed at runtime.
    """

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(shard)))
    except (OSError, TypeError, IndentationError, SyntaxError):
        return False, "native shard source is unavailable for validation"
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )
    if function is None:
        return False, "native shard method could not be inspected"

    layer_loops = 0
    for statement in function.body:
        if isinstance(statement, ast.For):
            iterator = statement.iter
            if not isinstance(iterator, ast.Attribute) or iterator.attr != "layers":
                return False, "native shard iterates a non-layer top-level collection"
            layer_loops += 1
            continue
        if isinstance(
            statement,
            (
                ast.Assert,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Import,
                ast.ImportFrom,
            ),
        ):
            continue
        if isinstance(statement, ast.Expr):
            if isinstance(statement.value, ast.Constant) and isinstance(
                statement.value.value, str
            ):
                continue
            return False, "native shard performs work outside its layer loop"
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if all(isinstance(target, ast.Name) for target in targets):
                continue
            return False, "native shard mutates model state outside its layer loop"
        if isinstance(statement, ast.Return) and statement.value is None:
            continue
        return False, "native shard has unsupported control flow outside its layer loop"
    if layer_loops != 1:
        return False, f"native shard has {layer_loops} top-level layer loops"
    return True, "native shard is confined to one layer loop"


def _require_divisible(value: int, divisor: int, label: str) -> None:
    if value <= 0 or value % divisor:
        raise ValueError(f"{label} ({value}) is not divisible by {divisor} ranks")


def _wrap_sharded_moe(inner: Any, group: Any, mx: Any) -> Any:
    """Add the collective that in-place expert weight slicing does not provide."""

    import mlx.nn as nn
    from mlx.nn.layers.distributed import sum_gradients

    class ShardedMoE(nn.Module):
        def __init__(self, module: Any):
            super().__init__()
            self.inner = module

        def __call__(self, value: Any, *args: Any, **kwargs: Any) -> Any:
            value = sum_gradients(group)(value)
            output = self.inner(value, *args, **kwargs)
            return mx.distributed.all_sum(output, group=group)

    return ShardedMoE(inner)


def _uneven_group_ranges(total_groups: int, size: int) -> list[tuple[int, int]]:
    """Contiguous per-rank ``[lo, hi)`` group ranges covering ``total_groups``.

    When ``total_groups`` is not divisible by ``size`` the first
    ``total_groups % size`` ranks receive one extra group. Low ranks get the
    slightly larger shard on purpose: rank 0 is the coordinator, usually the
    higher-memory node, so it absorbs the few-percent skew.
    """

    base, rem = divmod(total_groups, size)
    ranges: list[tuple[int, int]] = []
    lo = 0
    for r in range(size):
        hi = lo + base + (1 if r < rem else 0)
        ranges.append((lo, hi))
        lo = hi
    return ranges


def _shard_switch_mlp_uneven(
    switch_mlp: Any, group: Any, mx: Any, rank: int, size: int
) -> None:
    """Group-aligned (possibly uneven) tensor-parallel split of a quantized MoE.

    ``fc1`` is column-parallel — each rank owns a contiguous block of
    intermediate neurons — and ``fc2`` is row-parallel over that same block.
    ``fc2``'s intermediate axis is the quantization-group axis, so the split
    must land on group boundaries. When the group count is not divisible by the
    world size (Nemotron-H's MoE has 29 groups; world size 2 wants 14.5) the
    even ``mx.split`` inside ``shard_inplace`` raises. We slice explicit,
    possibly unequal, group ranges instead. The recombining ``all_sum`` in
    :func:`_wrap_sharded_moe` is shape-agnostic, so unequal per-rank widths sum
    back to the full result exactly (verified to fp noise on mlx 0.31.x).
    """

    from mlx.nn.layers.distributed import shard_inplace

    fc1 = switch_mlp.fc1
    fc2 = switch_mlp.fc2

    # Non-quantized experts have no group constraint; the stock even split is
    # correct and simpler.
    if not hasattr(fc2, "scales"):
        shard_inplace(fc1, "all-to-sharded", group=group)
        shard_inplace(fc2, "sharded-to-all", group=group)
        return

    # One scales column per quant group along fc2's intermediate (contraction)
    # axis. This is the axis that must divide the world size and, at TP=2 for
    # Nemotron-H, does not (29 is prime).
    groups = int(fc2.scales.shape[-1])
    lo, hi = _uneven_group_ranges(groups, size)[rank]

    # fc1: column-parallel. Its output rows *are* the intermediate neurons, so
    # slicing the same group block keeps fc1's output aligned with fc2's input.
    # Output is axis 1 of the 3D (experts, out, in) expert tensors.
    neurons_per_group = int(fc1.weight.shape[1]) // groups
    nlo, nhi = lo * neurons_per_group, hi * neurons_per_group
    fc1.weight = mx.contiguous(fc1.weight[:, nlo:nhi, :])
    if hasattr(fc1, "scales"):
        fc1.scales = mx.contiguous(fc1.scales[:, nlo:nhi, :])
    if getattr(fc1, "biases", None) is not None:
        fc1.biases = mx.contiguous(fc1.biases[:, nlo:nhi, :])

    # fc2: row-parallel over the packed intermediate axis. Packed columns per
    # group = packed width / group count (8 for 4-bit: 32/4 values per uint32);
    # scales/biases carry exactly one column per group.
    packed_per_group = int(fc2.weight.shape[-1]) // groups
    plo, phi = lo * packed_per_group, hi * packed_per_group
    fc2.weight = mx.contiguous(fc2.weight[..., plo:phi])
    fc2.scales = mx.contiguous(fc2.scales[..., lo:hi])
    if getattr(fc2, "biases", None) is not None:
        fc2.biases = mx.contiguous(fc2.biases[..., lo:hi])


@_register(QWEN3_NEXT)
def _shard_qwen3_next(
    model: Any,
    group: Any,
    mx: Any,
    progress: ProgressCallback | None,
) -> None:
    from mlx.nn.layers.distributed import shard_inplace, shard_linear
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    layers = list(model.layers)
    size = int(group.size())
    rank = int(group.rank())
    total = len(layers)
    for index, layer in enumerate(layers):
        mx.eval(layer.parameters())
        if layer.is_linear:
            attention = layer.linear_attn
            _require_divisible(attention.num_k_heads, size, "linear key heads")
            _require_divisible(attention.num_v_heads, size, "linear value heads")
            attention.in_proj_qkvz = shard_linear(
                attention.in_proj_qkvz,
                "all-to-sharded",
                group=group,
            )
            attention.in_proj_ba = shard_linear(
                attention.in_proj_ba,
                "all-to-sharded",
                group=group,
            )
            attention.out_proj = shard_linear(
                attention.out_proj,
                "sharded-to-all",
                group=group,
            )

            key_dim = int(attention.key_dim)
            value_dim = int(attention.value_dim)
            key_shard = key_dim // size
            value_shard = value_dim // size
            indices = mx.concatenate(
                [
                    mx.arange(rank * key_shard, (rank + 1) * key_shard),
                    mx.arange(
                        key_dim + rank * key_shard,
                        key_dim + (rank + 1) * key_shard,
                    ),
                    mx.arange(
                        2 * key_dim + rank * value_shard,
                        2 * key_dim + (rank + 1) * value_shard,
                    ),
                ]
            )
            attention.conv1d.weight = mx.contiguous(attention.conv1d.weight[indices])
            if getattr(attention.conv1d, "bias", None) is not None:
                attention.conv1d.bias = mx.contiguous(attention.conv1d.bias[indices])
            attention.conv1d.groups = key_shard * 2 + value_shard
            heads = attention.num_v_heads // size
            attention.A_log = mx.contiguous(
                attention.A_log[rank * heads : (rank + 1) * heads]
            )
            attention.dt_bias = mx.contiguous(
                attention.dt_bias[rank * heads : (rank + 1) * heads]
            )
            attention.num_k_heads //= size
            attention.num_v_heads //= size
            attention.key_dim //= size
            attention.value_dim //= size
            attention.conv_dim = attention.key_dim * 2 + attention.value_dim
        else:
            attention = layer.self_attn
            _require_divisible(
                attention.num_attention_heads,
                size,
                "attention heads",
            )
            _require_divisible(
                attention.num_key_value_heads,
                size,
                "KV heads",
            )
            attention.q_proj = shard_linear(
                attention.q_proj,
                "all-to-sharded",
                group=group,
            )
            attention.k_proj = shard_linear(
                attention.k_proj,
                "all-to-sharded",
                group=group,
            )
            attention.v_proj = shard_linear(
                attention.v_proj,
                "all-to-sharded",
                group=group,
            )
            attention.o_proj = shard_linear(
                attention.o_proj,
                "sharded-to-all",
                group=group,
            )
            attention.num_attention_heads //= size
            attention.num_key_value_heads //= size

        mlp = layer.mlp
        if isinstance(mlp, Qwen3NextSparseMoeBlock):
            for name, sharding in (
                ("gate_proj", "all-to-sharded"),
                ("down_proj", "sharded-to-all"),
                ("up_proj", "all-to-sharded"),
            ):
                shard_inplace(
                    getattr(mlp.switch_mlp, name),
                    sharding,
                    group=group,
                )
                shard_inplace(
                    getattr(mlp.shared_expert, name),
                    sharding,
                    group=group,
                )
            layer.mlp = _wrap_sharded_moe(mlp, group, mx)
        else:
            mlp.gate_proj = shard_linear(
                mlp.gate_proj, "all-to-sharded", group=group
            )
            mlp.down_proj = shard_linear(
                mlp.down_proj, "sharded-to-all", group=group
            )
            mlp.up_proj = shard_linear(
                mlp.up_proj, "all-to-sharded", group=group
            )
        mx.eval(layer.parameters())
        mx.clear_cache()
        _emit(
            progress,
            strategy=QWEN3_NEXT.name,
            layer=index,
            loaded=index + 1,
            total=total,
        )


@_register(NEMOTRON_H)
def _shard_nemotron_h(
    model: Any,
    group: Any,
    mx: Any,
    progress: ProgressCallback | None,
) -> None:
    from mlx.nn.layers.distributed import shard_inplace, shard_linear
    from mlx_lm.models.nemotron_h import (
        NemotronHAttention,
        NemotronHMamba2Mixer,
        NemotronHMoE,
    )

    layers = list(model.layers)
    size = int(group.size())
    rank = int(group.rank())
    total = len(layers)
    for index, layer in enumerate(layers):
        mx.eval(layer.parameters())
        mixer = layer.mixer
        if isinstance(mixer, NemotronHAttention):
            _require_divisible(mixer.num_heads, size, "attention heads")
            _require_divisible(mixer.num_key_value_heads, size, "KV heads")
            mixer.q_proj = shard_linear(
                mixer.q_proj, "all-to-sharded", group=group
            )
            mixer.k_proj = shard_linear(
                mixer.k_proj, "all-to-sharded", group=group
            )
            mixer.v_proj = shard_linear(
                mixer.v_proj, "all-to-sharded", group=group
            )
            mixer.o_proj = shard_linear(
                mixer.o_proj, "sharded-to-all", group=group
            )
            mixer.num_heads //= size
            mixer.num_key_value_heads //= size
        elif isinstance(mixer, NemotronHMamba2Mixer):
            _require_divisible(mixer.num_heads, size, "Mamba heads")
            _require_divisible(mixer.n_groups, size, "Mamba groups")
            heads = mixer.num_heads // size
            groups = mixer.n_groups // size
            intermediate = heads * mixer.head_dim
            bc = groups * mixer.ssm_state_size
            full_intermediate = mixer.intermediate_size
            full_bc = mixer.n_groups * mixer.ssm_state_size
            indices = mx.concatenate(
                [
                    mx.arange(rank * intermediate, (rank + 1) * intermediate),
                    mx.arange(
                        full_intermediate + rank * intermediate,
                        full_intermediate + (rank + 1) * intermediate,
                    ),
                    mx.arange(
                        2 * full_intermediate + rank * bc,
                        2 * full_intermediate + (rank + 1) * bc,
                    ),
                    mx.arange(
                        2 * full_intermediate + full_bc + rank * bc,
                        2 * full_intermediate + full_bc + (rank + 1) * bc,
                    ),
                    mx.arange(
                        2 * full_intermediate
                        + 2 * full_bc
                        + rank * heads,
                        2 * full_intermediate
                        + 2 * full_bc
                        + (rank + 1) * heads,
                    ),
                ]
            )
            mixer.in_proj.weight = mx.contiguous(mixer.in_proj.weight[indices])
            # ``in_proj`` is frequently quantized (per-tensor override in the
            # checkpoint's quantization dict). Its scales/biases carry one row
            # per weight row, so they must be gathered with the *same* row
            # ``indices`` — otherwise the sharded module keeps full-height
            # scales against a half-height weight and the first Mamba forward
            # fails a shape check. Row slicing is group-safe because quant
            # groups run along the input (column) axis, untouched here.
            if hasattr(mixer.in_proj, "scales"):
                mixer.in_proj.scales = mx.contiguous(mixer.in_proj.scales[indices])
            if getattr(mixer.in_proj, "biases", None) is not None:
                mixer.in_proj.biases = mx.contiguous(mixer.in_proj.biases[indices])
            # The affine layer bias (mamba_proj_bias) is per output row too.
            if getattr(mixer.in_proj, "bias", None) is not None:
                mixer.in_proj.bias = mx.contiguous(mixer.in_proj.bias[indices])
            mixer.out_proj = shard_linear(
                mixer.out_proj, "sharded-to-all", group=group
            )
            conv_indices = mx.concatenate(
                [
                    mx.arange(rank * intermediate, (rank + 1) * intermediate),
                    mx.arange(
                        full_intermediate + rank * bc,
                        full_intermediate + (rank + 1) * bc,
                    ),
                    mx.arange(
                        full_intermediate + full_bc + rank * bc,
                        full_intermediate + full_bc + (rank + 1) * bc,
                    ),
                ]
            )
            mixer.conv1d.weight = mx.contiguous(mixer.conv1d.weight[conv_indices])
            if getattr(mixer.conv1d, "bias", None) is not None:
                mixer.conv1d.bias = mx.contiguous(mixer.conv1d.bias[conv_indices])
            mixer.conv1d.groups = intermediate + 2 * bc
            start = rank * heads
            end = start + heads
            mixer.dt_bias = mx.contiguous(mixer.dt_bias[start:end])
            mixer.A_log = mx.contiguous(mixer.A_log[start:end])
            mixer.D = mx.contiguous(mixer.D[start:end])
            mixer.norm.weight = mx.contiguous(
                mixer.norm.weight[
                    rank * intermediate : (rank + 1) * intermediate
                ]
            )
            mixer.num_heads = heads
            mixer.n_groups = groups
            mixer.intermediate_size = intermediate
            mixer.conv_dim = intermediate + 2 * bc
            mixer.heads_per_group = heads // groups
        elif isinstance(mixer, NemotronHMoE):
            # Routed experts: group-aligned split that tolerates a quant-group
            # count not divisible by the world size (Nemotron-H has 29).
            _shard_switch_mlp_uneven(mixer.switch_mlp, group, mx, rank, size)
            if hasattr(mixer, "shared_experts"):
                shard_inplace(
                    mixer.shared_experts.up_proj,
                    "all-to-sharded",
                    group=group,
                )
                shard_inplace(
                    mixer.shared_experts.down_proj,
                    "sharded-to-all",
                    group=group,
                )
            layer.mixer = _wrap_sharded_moe(mixer, group, mx)
        mx.eval(layer.parameters())
        mx.clear_cache()
        _emit(
            progress,
            strategy=NEMOTRON_H.name,
            layer=index,
            loaded=index + 1,
            total=total,
        )


def _shard_qwen4_exp_ple(layer: Any, group: Any, rank: int, size: int) -> None:
    """Rewire the PLE table to expose only this rank's local shard range.

    The n-gram embedding is 128 mmap-backed shard tables (~29 GiB raw,
    ~16 GiB quantized).  Rather than materializing them into RAM (which
    would blow the rank budget), we leave the lazy mmap'd arrays in place
    and only wrap the table so that each rank's forward reads only its
    local subset.  The all_sum in the wrapper reconstructs the full
    embedding from the partial outputs — each row is produced by exactly
    one rank; everything else contributes zeros.
    """
    import mlx.nn as nn

    class _LocalPLEShards(nn.Module):
        """Rank-local slice of the qwen4_exp n-gram table, combined with all_sum."""

        def __init__(self, inner: Any, lo: int, hi: int, grp: Any):
            super().__init__()
            self.rows = int(inner.rows)
            self.dim = int(inner.dim)
            self._lo = lo
            self._hi = hi
            self._group = grp
            for i in range(lo, hi):
                setattr(self, f"shard_{i}", getattr(inner, f"shard_{i}"))

        def __call__(self, gid: Any) -> Any:
            import mlx.core as mx
            import numpy as np

            flat = gid.reshape(-1)
            shard_of = np.array(flat // self.rows, copy=False)
            row_of = flat % self.rows
            out = mx.zeros((flat.size, self.dim), dtype=mx.float32)
            for s in np.unique(shard_of).tolist():
                if not (self._lo <= s < self._hi):
                    continue
                sel = mx.array(np.nonzero(shard_of == s)[0])
                emb = getattr(self, f"shard_{s}")(mx.take(row_of, sel))
                out = mx.put_along_axis(
                    out, sel[:, None], emb.astype(mx.float32), axis=0
                )
            out = mx.distributed.all_sum(out, group=self._group)
            return out.reshape(*gid.shape, self.dim)

    ple = layer.ple
    sharded = ple.ple_embedding.ngram_embedding
    if getattr(sharded, "shard_sizes", None) is not None:
        # DiskBackedShardedEmbedding: the table streams from SSD via mmap and
        # owns no resident weights — nothing to split, nothing to rewire.
        return
    n = int(sharded.n_shards)
    _require_divisible(n, size, "PLE shards")
    lo, hi = _uneven_group_ranges(n, size)[rank]
    # Rewire — do NOT mx.eval the shards; they stay as lazy mmap'd arrays
    # and are streamed from SSD on each forward touch.
    ple.ple_embedding.ngram_embedding = _LocalPLEShards(sharded, lo, hi, group)


@_register(QWEN4_EXP)
def _shard_qwen4_exp(
    model: Any,
    group: Any,
    mx: Any,
    progress: ProgressCallback | None,
) -> None:
    import gc as _gc

    from mlx.nn.layers.distributed import shard_inplace, shard_linear
    from mlx.utils import tree_flatten, tree_unflatten
    try:
        from mlx_lm.models.qwen4_exp import SparseMoeBlock
    except ImportError:
        SparseMoeBlock = None

    _, layers = _common_layer_owner(model)
    layers = list(layers)
    size = int(group.size())
    rank = int(group.rank())
    total = len(layers)
    for index, layer in enumerate(layers):
        _old_children = [
            value
            for name, value in layer.named_modules()
            if name and name.count(".") == 0
        ]
        if getattr(layer, "ple", None) is not None:
            _shard_qwen4_exp_ple(layer, group, rank, size)
        # NOTE: no leading full-layer eval. The shard ops bind lazy slices of
        # the lazy mmap'd checkpoint arrays; the rebind below materializes
        # ONLY the sharded slices. Materializing the full layer first makes
        # the base un-releasable (measured: full + half resident per layer).
        if layer.layer_type == "linear_attention":
            attn = layer.linear_attn
            _require_divisible(attn.n_k, size, "linear key heads")
            _require_divisible(attn.n_v, size, "linear value heads")
            key_dim = int(attn.key_dim)
            value_dim = int(attn.value_dim)
            attn.in_proj_qkv = shard_linear(
                attn.in_proj_qkv,
                "all-to-sharded",
                segments=[key_dim, 2 * key_dim],
                group=group,
            )
            attn.in_proj_z = shard_linear(
                attn.in_proj_z, "all-to-sharded", group=group
            )
            attn.in_proj_b = shard_linear(attn.in_proj_b, "all-to-sharded", group=group)
            attn.in_proj_a = shard_linear(attn.in_proj_a, "all-to-sharded", group=group)
            attn.out_proj = shard_linear(attn.out_proj, "sharded-to-all", group=group)
            key_dim = int(attn.key_dim)
            value_dim = int(attn.value_dim)
            key_shard = key_dim // size
            value_shard = value_dim // size
            indices = mx.concatenate(
                [
                    mx.arange(rank * key_shard, (rank + 1) * key_shard),
                    mx.arange(
                        key_dim + rank * key_shard,
                        key_dim + (rank + 1) * key_shard,
                    ),
                    mx.arange(
                        2 * key_dim + rank * value_shard,
                        2 * key_dim + (rank + 1) * value_shard,
                    ),
                ]
            )
            attn.conv1d.weight = mx.contiguous(attn.conv1d.weight[indices])
            if getattr(attn.conv1d, "bias", None) is not None:
                attn.conv1d.bias = mx.contiguous(attn.conv1d.bias[indices])
            attn.conv1d.groups = key_shard * 2 + value_shard
            heads = attn.n_v // size
            attn.A_log = mx.contiguous(attn.A_log[rank * heads : (rank + 1) * heads])
            attn.dt_bias = mx.contiguous(
                attn.dt_bias[rank * heads : (rank + 1) * heads]
            )
            attn.n_k //= size
            attn.n_v //= size
            attn.key_dim //= size
            attn.value_dim //= size
            attn.conv_dim = attn.key_dim * 2 + attn.value_dim
        else:
            attn = layer.self_attn
            n_heads = getattr(attn, "n_heads", getattr(attn, "num_attention_heads", None))
            n_kv_heads = getattr(attn, "n_kv_heads", getattr(attn, "num_key_value_heads", None))
            _require_divisible(n_heads, size, "attention heads")
            _require_divisible(n_kv_heads, size, "KV heads")
            attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=group)
            attn.k_proj = shard_linear(attn.k_proj, "all-to-sharded", group=group)
            attn.v_proj = shard_linear(attn.v_proj, "all-to-sharded", group=group)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            if hasattr(attn, "n_heads"):
                attn.n_heads //= size
            if hasattr(attn, "num_attention_heads"):
                attn.num_attention_heads //= size
            if hasattr(attn, "n_kv_heads"):
                attn.n_kv_heads //= size
            if hasattr(attn, "num_key_value_heads"):
                attn.num_key_value_heads //= size
            # The QSA indexer is intentionally replicated: it produces the
            # sparse keep-mask that must be bit-identical on every rank, and
            # it is a negligible fraction of the weights.
        mlp = layer.mlp
        if (SparseMoeBlock is not None and isinstance(mlp, SparseMoeBlock)) or (
            hasattr(mlp, "switch_mlp") and hasattr(mlp, "shared_expert")
        ):
            for name, sharding in (
                ("gate_proj", "all-to-sharded"),
                ("down_proj", "sharded-to-all"),
                ("up_proj", "all-to-sharded"),
            ):
                shard_inplace(
                    getattr(mlp.switch_mlp, name),
                    sharding,
                    group=group,
                )
                shard_inplace(
                    getattr(mlp.shared_expert, name),
                    sharding,
                    group=group,
                )
            layer.mlp = _wrap_sharded_moe(mlp, group, mx)
        # mlx's sharded params bind as lazy slices of the full materialized
        # arrays; materializing them does NOT release the base (measured on a
        # 2-rank ring: full + half stay resident). Force explicit contiguous
        # copies and drop the pre-shard arrays, or every rank accumulates the
        # FULL model plus its shard.
        mx.eval(layer.parameters())
        mx.clear_cache()
        _emit(
            progress,
            strategy=QWEN4_EXP.name,
            layer=index,
            loaded=index + 1,
            total=total,
        )


def apply_tensor_strategy(
    model: Any,
    group: Any,
    *,
    mx_module: Any,
    progress: ProgressCallback | None = None,
) -> str:
    """Shard ``model`` with the registered adapter or its native implementation."""

    model_type = _model_type(model)
    adapter = _ADAPTERS.get(model_type)
    if adapter is not None:
        strategy = adapter._omlx_tensor_strategy  # type: ignore[attr-defined]
        adapter(model, group, mx_module, progress)
        return strategy.name
    if not callable(getattr(model, "shard", None)):
        raise RuntimeError(
            f"tensor parallelism is unsupported for model type {model_type!r}: "
            "no registered strategy and no native shard method"
        )
    _native_layerwise_shard(model, group, mx_module, progress)
    return "native"


__all__ = [
    "NEMOTRON_H",
    "QWEN3_NEXT",
    "QWEN4_EXP",
    "TensorStrategy",
    "apply_tensor_strategy",
    "native_shard_is_layer_local",
    "registered_model_types",
    "supports_model_type",
]
