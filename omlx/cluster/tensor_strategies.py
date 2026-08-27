# SPDX-License-Identifier: Apache-2.0
"""Capability-gated, layer-at-a-time tensor-parallel sharding.

MLX model sharding is architecture-specific. This module keeps an explicit
registry for oMLX adapters and uses a carefully bounded native fallback for
models whose installed MLX-LM class already implements ``shard()``.
"""

from __future__ import annotations

import ast
import gc
import inspect
import os
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], None]

_VOCAB_PARALLEL_MODE_ENV = "OMLX_CLUSTER_VOCAB_PARALLEL"
_VOCAB_PARALLEL_MIN_BYTES_ENV = "OMLX_CLUSTER_VOCAB_PARALLEL_MIN_BYTES"
_VOCAB_PARALLEL_MIN_BYTES = 256 * 1024**2
_LAZY_NATIVE_SHARD_ENV = "OMLX_TP_LAZY_NATIVE_SHARD"

# Full-model TP parity, not the small projection fixture, is authoritative.
# Dense Qwen3.5/3.8 currently diverges when its output vocabulary is split:
# the layer shards remain correct, but coordinator reconstruction produces
# repeated/invalid text after the first few tokens. Keep the exact replicated
# head until that model-specific path has a physical parity certificate.
_VOCAB_PARALLEL_UNQUALIFIED_MODEL_TYPES = frozenset(
    {"qwen3_5", "qwen3_5_moe"}
)


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
QWEN3_MOE = TensorStrategy(
    name="qwen3_moe",
    model_types=("qwen3_moe", "qwen3_vl_moe"),
    source="oMLX adapter derived from Exo's Qwen MoE strategy",
)
QWEN3_VL = TensorStrategy(
    name="qwen3_vl",
    model_types=("qwen3_vl",),
    source="audited layer-wise delegate to MLX-LM's Qwen3 strategy",
)
GEMMA4 = TensorStrategy(
    name="gemma4",
    model_types=("gemma4", "gemma4_text", "gemma4_unified"),
    source="oMLX adapter derived from Exo's Gemma 4 strategy",
)
KIMI_K25 = TensorStrategy(
    name="kimi_k25",
    model_types=("kimi_k25",),
    source="audited layer-wise delegate to MLX-LM's DeepSeek-V3 strategy",
)
NEMOTRON_H = TensorStrategy(
    name="nemotron_h",
    model_types=("nemotron_h",),
    source="oMLX adapter derived from Exo's attention/Mamba/MoE strategy",
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


def _gather_vocab_logits(local_logits: Any, group: Any, mx: Any) -> Any:
    """Gather vocabulary shards while preserving logits on the final axis."""

    if local_logits.ndim < 1:
        raise ValueError("vocabulary-parallel logits require at least one axis")
    vocab_first = mx.contiguous(mx.swapaxes(local_logits, 0, -1))
    gathered = mx.distributed.all_gather(vocab_first, group=group)
    return mx.swapaxes(gathered, 0, -1)


def _vocab_parallel_mode() -> str:
    value = os.environ.get(_VOCAB_PARALLEL_MODE_ENV, "auto").strip().lower()
    aliases = {
        "1": "on",
        "true": "on",
        "yes": "on",
        "0": "off",
        "false": "off",
        "no": "off",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "on", "off"}:
        raise ValueError(
            f"{_VOCAB_PARALLEL_MODE_ENV} must be auto, on, or off; got {value!r}"
        )
    return value


def _vocab_parallel_min_bytes() -> int:
    raw = os.environ.get(
        _VOCAB_PARALLEL_MIN_BYTES_ENV,
        str(_VOCAB_PARALLEL_MIN_BYTES),
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_VOCAB_PARALLEL_MIN_BYTES_ENV} must be an integer; got {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError(f"{_VOCAB_PARALLEL_MIN_BYTES_ENV} must be non-negative")
    return value


def _module_nbytes(module: Any) -> int:
    from mlx.utils import tree_flatten

    return sum(int(value.nbytes) for _, value in tree_flatten(module.parameters()))


def _lm_head_is_tied(model: Any, owner: Any, head: Any) -> bool:
    """Conservatively detect heads backed by the input embedding table."""

    for candidate in (
        model,
        owner,
        getattr(model, "args", None),
        getattr(model, "config", None),
        getattr(owner, "args", None),
        getattr(owner, "config", None),
    ):
        tied = getattr(candidate, "tie_word_embeddings", None)
        if tied is True:
            return True

    head_weight = getattr(head, "weight", None)
    if head_weight is None:
        return False
    roots = (
        model,
        owner,
        getattr(model, "model", None),
        getattr(owner, "model", None),
        getattr(model, "backbone", None),
        getattr(owner, "backbone", None),
        getattr(model, "transformer", None),
        getattr(owner, "transformer", None),
    )
    for root in roots:
        if root is None:
            continue
        for name in (
            "embed_tokens",
            "embeddings",
            "embedding",
            "tok_embeddings",
            "token_embeddings",
            "word_embeddings",
            "wte",
        ):
            embedding = getattr(root, name, None)
            if getattr(embedding, "weight", None) is head_weight:
                return True
    return False


def _find_untied_lm_head(model: Any) -> tuple[Any, Any] | None:
    """Return the owner and standalone output projection, if one exists."""

    owners = (
        model,
        getattr(model, "language_model", None),
        getattr(model, "_language_model", None),
    )
    seen: set[int] = set()
    for owner in owners:
        if owner is None:
            continue
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        head = getattr(owner, "lm_head", None)
        if (
            head is not None
            and not getattr(head, "_omlx_vocab_parallel", False)
            and not _lm_head_is_tied(model, owner, head)
        ):
            return owner, head
    return None


def _make_vocab_parallel_head(head: Any, group: Any, mx: Any) -> Any:
    """Create a row-sharded output projection with exact gathered logits."""

    import mlx.nn as nn

    if not isinstance(head, (nn.Linear, nn.QuantizedLinear)):
        raise TypeError(f"unsupported output head type {type(head).__name__}")

    size = int(group.size())
    rank = int(group.rank())
    output_dims = int(head.weight.shape[0])
    rows = output_dims // size
    start = rank * rows
    stop = start + rows

    class VocabParallelLinear(nn.Linear):
        def __init__(self, source: Any) -> None:
            nn.Module.__init__(self)
            self.group = group
            self.weight = mx.contiguous(source.weight[start:stop])
            if "bias" in source:
                self.bias = mx.contiguous(source.bias[start:stop])
            self._omlx_vocab_parallel = True
            self._omlx_output_dims = output_dims

        def _local_logits(self, value: Any) -> Any:
            # Calling the base class preserves architecture patches installed
            # on ``nn.Linear.__call__``. DeepSeek's DSpark verifier relies on
            # that hook for M=1-equivalent multi-row reduction order.
            return super().__call__(value)

        def _gather_logits(self, local_logits: Any) -> Any:
            if not getattr(self, "_omlx_gather_vocab_logits", True):
                return local_logits
            return _gather_vocab_logits(local_logits, self.group, mx)

        def __call__(self, value: Any) -> Any:
            return self._gather_logits(self._local_logits(value))

        def _extra_repr(self) -> str:
            in_dims = int(self.weight.shape[-1])
            return (
                f"input_dims={in_dims}, output_dims={self._omlx_output_dims}, "
                f"local_output_dims={int(self.weight.shape[0])}"
            )

    class VocabParallelQuantizedLinear(nn.QuantizedLinear):
        def __init__(self, source: Any) -> None:
            nn.Module.__init__(self)
            self.group = group
            self.weight = mx.contiguous(source.weight[start:stop])
            self.scales = mx.contiguous(source.scales[start:stop])
            if getattr(source, "biases", None) is not None:
                self.biases = mx.contiguous(source.biases[start:stop])
            else:
                self.biases = None
            if "bias" in source:
                self.bias = mx.contiguous(source.bias[start:stop])
            self.group_size = source.group_size
            self.bits = source.bits
            self.mode = getattr(source, "mode", "affine")
            self._omlx_vocab_parallel = True
            self._omlx_output_dims = output_dims
            self.freeze()

        def _local_logits(self, value: Any) -> Any:
            return super().__call__(value)

        def _gather_logits(self, local_logits: Any) -> Any:
            if not getattr(self, "_omlx_gather_vocab_logits", True):
                return local_logits
            return _gather_vocab_logits(local_logits, self.group, mx)

        def __call__(self, value: Any) -> Any:
            return self._gather_logits(self._local_logits(value))

        def _extra_repr(self) -> str:
            in_dims = int(self.weight.shape[-1]) * 32 // int(self.bits)
            return (
                f"input_dims={in_dims}, output_dims={self._omlx_output_dims}, "
                f"local_output_dims={int(self.weight.shape[0])}, "
                f"group_size={self.group_size}, bits={self.bits}, mode={self.mode}"
            )

    wrapper = (
        VocabParallelQuantizedLinear
        if isinstance(head, nn.QuantizedLinear)
        else VocabParallelLinear
    )
    return wrapper(head)


def _shard_output_head(
    model: Any,
    group: Any,
    mx: Any,
    progress: ProgressCallback | None = None,
) -> bool:
    """Shard a large untied vocabulary head, preserving the model interface."""

    mode = _vocab_parallel_mode()
    if mode == "off":
        return False

    model_type = _model_type(model)
    if model_type in _VOCAB_PARALLEL_UNQUALIFIED_MODEL_TYPES:
        reason = (
            f"vocabulary parallelism is not parity-qualified for {model_type}; "
            "the exact replicated output head is retained"
        )
        model._omlx_vocab_parallel_disabled_reason = reason
        if mode == "on":
            raise RuntimeError(reason)
        return False

    found = _find_untied_lm_head(model)
    if found is None:
        if mode == "on":
            raise RuntimeError("forced vocabulary parallelism found no untied lm_head")
        return False
    size = int(group.size())
    if size < 2:
        return False
    owner, head = found
    head_bytes = _module_nbytes(head)
    if mode == "auto" and head_bytes < _vocab_parallel_min_bytes():
        return False

    output_dims = int(head.weight.shape[0])
    if output_dims <= 0 or output_dims % size:
        message = (
            f"lm_head vocabulary ({output_dims}) is not divisible by {size} ranks"
        )
        if mode == "on":
            raise RuntimeError(message)
        return False

    try:
        sharded = _make_vocab_parallel_head(head, group, mx)
    except (KeyError, TypeError, ValueError) as exc:
        if mode == "on":
            raise RuntimeError(f"cannot shard lm_head: {exc}") from exc
        return False

    owner.lm_head = sharded
    model._omlx_vocab_parallel_head = True
    model._omlx_output_vocab_size = output_dims
    if progress is not None:
        progress(
            {
                "phase": "tensor_output_head",
                "strategy": "vocab_parallel",
                "rank": int(group.rank()),
                "ranks": size,
                "head_bytes": head_bytes,
                "local_head_bytes": _module_nbytes(sharded),
                "vocab_size": output_dims,
                "local_vocab_size": output_dims // size,
            }
        )
    return True


def _shard_auxiliary_vocab_heads(
    model: Any,
    group: Any,
    mx: Any,
    progress: ProgressCallback | None = None,
) -> int:
    """Shard adapter-declared vocabulary projections outside ``lm_head``.

    Some speculative decoders own another projection into the model
    vocabulary. DeepSeek-V4 Flash's DSpark Markov correction is the important
    example: leaving it replicated would both waste work and make its full
    vocabulary bias incompatible with a local ``lm_head`` shard.

    Adapters expose ``(owner, attribute_name)`` pairs through
    ``_omlx_tensor_vocab_modules``. The contract is deliberately explicit;
    guessing arbitrary linears by shape could corrupt tied embeddings or
    architecture-specific heads.
    """

    factory = getattr(model, "_omlx_tensor_vocab_modules", None)
    if not callable(factory):
        return 0
    modules = list(factory())
    if not modules:
        return 0

    expected_vocab = int(getattr(model, "_omlx_output_vocab_size", 0) or 0)
    sharded = 0
    replacements: list[Any] = []
    for index, entry in enumerate(modules):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise RuntimeError(
                "tensor vocabulary module contract must return (owner, name) pairs"
            )
        owner, name = entry
        head = getattr(owner, name, None)
        if head is None:
            raise RuntimeError(f"tensor vocabulary module {name!r} is missing")
        if getattr(head, "_omlx_vocab_parallel", False):
            continue
        output_dims = int(getattr(getattr(head, "weight", None), "shape", (0,))[0])
        if expected_vocab and output_dims != expected_vocab:
            raise RuntimeError(
                f"tensor vocabulary module {name!r} has {output_dims} rows; "
                f"expected {expected_vocab}"
            )
        replacement = _make_vocab_parallel_head(head, group, mx)
        setattr(owner, name, replacement)
        replacements.append(replacement)
        sharded += 1
        if progress is not None:
            progress(
                {
                    "phase": "tensor_auxiliary_output_head",
                    "strategy": "vocab_parallel",
                    "module": index,
                    "name": name,
                    "rank": int(group.rank()),
                    "ranks": int(group.size()),
                    "vocab_size": output_dims,
                    "local_vocab_size": int(replacement.weight.shape[0]),
                }
            )

    if sharded == len(modules):
        model._omlx_vocab_parallel_aux_heads = tuple(replacements)
        model._omlx_distributed_mtp_vocab_ready = True
    return sharded


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


def _native_auxiliary_modules(
    model: Any,
    group: Any,
    mx: Any,
    progress: ProgressCallback | None,
) -> int:
    """Shard adapter-declared auxiliary transformer blocks progressively.

    Speculative heads commonly store one or more full transformer blocks
    outside the backbone's ``layers`` list. A model patch may expose those
    owning modules through ``_omlx_tensor_auxiliary_modules``. We temporarily
    present each block to the already-validated native ``shard()`` method,
    preserving architecture-specific tensor slicing without replicating the
    auxiliary block on every rank.
    """

    factory = getattr(model, "_omlx_tensor_auxiliary_modules", None)
    if not callable(factory):
        return 0
    modules = list(factory())
    if not modules:
        return 0
    supported, reason = native_shard_is_layer_local(getattr(model, "shard", None))
    if not supported:
        raise RuntimeError(
            "auxiliary tensor sharding requires a layer-local native strategy: "
            + reason
        )

    owner, layers = _common_layer_owner(model)
    original = list(layers)
    total = len(modules)
    try:
        for index, module in enumerate(modules):
            shard_layer = getattr(module, "block", module)
            if not callable(getattr(module, "parameters", None)):
                raise RuntimeError(
                    "tensor auxiliary module has no parameter tree: "
                    f"{type(module).__name__}"
                )
            mx.eval(module.parameters())
            owner.layers = [shard_layer]
            model.shard(group)
            mx.eval(module.parameters())
            mx.clear_cache()
            if progress is not None:
                progress(
                    {
                        "phase": "tensor_auxiliary_sharding",
                        "strategy": "native",
                        "module": index,
                        "modules_loaded": index + 1,
                        "modules_total": total,
                    }
                )
    finally:
        owner.layers = original
    return total


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
    lazy_mode = os.environ.get(_LAZY_NATIVE_SHARD_ENV, "auto").strip().lower()
    if lazy_mode not in {"auto", "1", "true", "on", "0", "false", "off"}:
        raise ValueError(
            f"{_LAZY_NATIVE_SHARD_ENV} must be auto, on, or off; got {lazy_mode!r}"
        )
    lazy_before_shard = lazy_mode in {"1", "true", "on"} or (
        lazy_mode == "auto" and _model_type(model).startswith("deepseek_v4")
    )
    try:
        for index, layer in enumerate(original):
            if not lazy_before_shard:
                mx.eval(layer.parameters())
            owner.layers = [layer]
            model.shard(group)
            mx.eval(layer.parameters())
            # Drop the full lazy source graph before the next 3–4 GB layer is
            # touched. Synchronize first so the contiguous local slice owns its
            # bytes, then collect Python graph cycles and return freed Metal
            # buffers instead of wiring them until the 128 GB rank stalls.
            synchronize = getattr(mx, "synchronize", None)
            if callable(synchronize):
                synchronize()
            gc.collect()
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


def _trusted_delegate_layerwise_shard(
    owner: Any,
    shard: Callable[[Any], None],
    group: Any,
    mx: Any,
    progress: ProgressCallback | None,
    *,
    strategy: str,
) -> None:
    """Run an explicitly audited wrapper strategy one layer at a time.

    Some multimodal MLX-LM models only forward to a text model's native
    ``shard()``. The general AST proof intentionally rejects such wrappers
    because it cannot prove what the delegate mutates. Registered adapters
    may use this helper after naming the concrete layer owner and delegate;
    arbitrary native methods still go through :func:`native_shard_is_layer_local`.
    """

    layers = getattr(owner, "layers", None)
    if not isinstance(layers, list):
        raise RuntimeError(
            f"{strategy} tensor strategy cannot locate its audited layer owner"
        )
    original = list(layers)
    if not original or any(layer is None for layer in original):
        raise RuntimeError(f"{strategy} tensor sharding requires a complete model")
    total = len(original)
    try:
        for index, layer in enumerate(original):
            mx.eval(layer.parameters())
            owner.layers = [layer]
            shard(group)
            mx.eval(layer.parameters())
            mx.clear_cache()
            _emit(
                progress,
                strategy=strategy,
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

    # Inspect the stable class function rather than a transient bound-method
    # wrapper. After progressive sharding has evaluated dozens of large MLX
    # module trees, Python 3.13 can fail source recovery for the bound object
    # even though ``type(model).shard`` still points at the same auditable
    # function. Auxiliary MTP stages repeat this proof after the backbone loop.
    shard = getattr(shard, "__func__", shard)
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


@_register(QWEN3_MOE)
def _shard_qwen3_moe(
    model: Any,
    group: Any,
    mx: Any,
    progress: ProgressCallback | None,
) -> None:
    """Shard Qwen3 MoE text models and their Qwen3-VL wrapper."""

    from mlx.nn.layers.distributed import shard_inplace, shard_linear
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    layers = list(model.layers)
    size = int(group.size())
    total = len(layers)
    for layer in layers:
        attention = layer.self_attn
        _require_divisible(attention.n_heads, size, "attention heads")
        _require_divisible(attention.n_kv_heads, size, "KV heads")

    for index, layer in enumerate(layers):
        mx.eval(layer.parameters())
        attention = layer.self_attn
        attention.q_proj = shard_linear(
            attention.q_proj, "all-to-sharded", group=group
        )
        attention.k_proj = shard_linear(
            attention.k_proj, "all-to-sharded", group=group
        )
        attention.v_proj = shard_linear(
            attention.v_proj, "all-to-sharded", group=group
        )
        attention.o_proj = shard_linear(
            attention.o_proj, "sharded-to-all", group=group
        )
        attention.n_heads //= size
        attention.n_kv_heads //= size

        mlp = layer.mlp
        if isinstance(mlp, Qwen3MoeSparseMoeBlock):
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
            strategy=QWEN3_MOE.name,
            layer=index,
            loaded=index + 1,
            total=total,
        )


@_register(QWEN3_VL)
def _shard_qwen3_vl(
    model: Any,
    group: Any,
    mx: Any,
    progress: ProgressCallback | None,
) -> None:
    """Use the bundled Qwen3 text sharder behind the multimodal wrapper."""

    language_model = model.language_model
    owner = language_model.model
    size = int(group.size())
    for layer in owner.layers:
        attention = layer.self_attn
        _require_divisible(attention.n_heads, size, "attention heads")
        _require_divisible(attention.n_kv_heads, size, "KV heads")
    _trusted_delegate_layerwise_shard(
        owner,
        language_model.shard,
        group,
        mx,
        progress,
        strategy=QWEN3_VL.name,
    )


@_register(GEMMA4)
def _shard_gemma4(
    model: Any,
    group: Any,
    mx: Any,
    progress: ProgressCallback | None,
) -> None:
    """Shard Gemma 4 attention, dense MLPs, and optional routed experts."""

    from mlx.nn.layers.distributed import shard_inplace, shard_linear

    layers = list(model.layers)
    size = int(group.size())
    total = len(layers)
    for layer in layers:
        attention = layer.self_attn
        _require_divisible(attention.n_heads, size, "attention heads")
        _require_divisible(attention.n_kv_heads, size, "KV heads")

    for index, layer in enumerate(layers):
        mx.eval(layer.parameters())
        attention = layer.self_attn
        attention.q_proj = shard_linear(
            attention.q_proj, "all-to-sharded", group=group
        )
        if attention.has_kv:
            attention.k_proj = shard_linear(
                attention.k_proj, "all-to-sharded", group=group
            )
            if not attention.use_k_eq_v:
                attention.v_proj = shard_linear(
                    attention.v_proj, "all-to-sharded", group=group
                )
        attention.o_proj = shard_linear(
            attention.o_proj, "sharded-to-all", group=group
        )
        attention.n_heads //= size
        attention.n_kv_heads //= size

        mlp = layer.mlp
        mlp.gate_proj = shard_linear(
            mlp.gate_proj, "all-to-sharded", group=group
        )
        mlp.down_proj = shard_linear(
            mlp.down_proj, "sharded-to-all", group=group
        )
        mlp.up_proj = shard_linear(
            mlp.up_proj, "all-to-sharded", group=group
        )

        if layer.enable_moe:
            experts = layer.experts
            for name, sharding in (
                ("gate_proj", "all-to-sharded"),
                ("down_proj", "sharded-to-all"),
                ("up_proj", "all-to-sharded"),
            ):
                shard_inplace(
                    getattr(experts.switch_glu, name),
                    sharding,
                    group=group,
                )
            layer.experts = _wrap_sharded_moe(experts, group, mx)
        mx.eval(layer.parameters())
        mx.clear_cache()
        _emit(
            progress,
            strategy=GEMMA4.name,
            layer=index,
            loaded=index + 1,
            total=total,
        )


@_register(KIMI_K25)
def _shard_kimi_k25(
    model: Any,
    group: Any,
    mx: Any,
    progress: ProgressCallback | None,
) -> None:
    """Use Kimi K2.5's audited delegate to the DeepSeek-V3 text sharder."""

    owner = model.language_model.model
    size = int(group.size())
    for layer in owner.layers:
        _require_divisible(layer.self_attn.num_heads, size, "attention heads")
    _trusted_delegate_layerwise_shard(
        owner,
        model.shard,
        group,
        mx,
        progress,
        strategy=KIMI_K25.name,
    )


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


def apply_tensor_strategy(
    model: Any,
    group: Any,
    *,
    mx_module: Any,
    progress: ProgressCallback | None = None,
) -> str:
    """Shard transformer layers and any eligible standalone output head."""

    model_type = _model_type(model)
    adapter = _ADAPTERS.get(model_type)
    if adapter is not None:
        strategy = adapter._omlx_tensor_strategy  # type: ignore[attr-defined]
        adapter(model, group, mx_module, progress)
        strategy_name = strategy.name
    else:
        if not callable(getattr(model, "shard", None)):
            raise RuntimeError(
                f"tensor parallelism is unsupported for model type {model_type!r}: "
                "no registered strategy and no native shard method"
            )
        _native_layerwise_shard(model, group, mx_module, progress)
        _native_auxiliary_modules(model, group, mx_module, progress)
        strategy_name = "native"
    output_sharded = _shard_output_head(model, group, mx_module, progress)
    if output_sharded:
        _shard_auxiliary_vocab_heads(model, group, mx_module, progress)
    return strategy_name


__all__ = [
    "GEMMA4",
    "KIMI_K25",
    "NEMOTRON_H",
    "QWEN3_MOE",
    "QWEN3_NEXT",
    "QWEN3_VL",
    "TensorStrategy",
    "_gather_vocab_logits",
    "_shard_auxiliary_vocab_heads",
    "_shard_output_head",
    "apply_tensor_strategy",
    "native_shard_is_layer_local",
    "registered_model_types",
    "supports_model_type",
]
