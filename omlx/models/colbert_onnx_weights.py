# SPDX-License-Identifier: Apache-2.0
"""Read a pinned ColBERT ONNX export as MLX parameter arrays.

A pylate ColBERT repository can ship two weight sets that are *not* the same
numbers: the sentence-transformers ``model.safetensors`` and the exported
``model.onnx``. Measured on the pinned checkpoint (``lightonai/mLateOn``):

    projection 1_Dense.linear.weight   corr 0.99996321   max|d| 1.114e-03
    encoder embeddings.tok_embeddings  corr 0.99998670   max|d| 7.002e-03

so serving one set where the other produced the frozen retrieval measurements
silently changes the representation (min per-token cosine ~0.99 between them).
Which set is authoritative is a provenance decision, not something oMLX can infer,
so the loader makes the choice explicit and reports it (see
``ColbertLayout.weight_source``).

This module is the ONNX side of that choice. It reads the weight tensors out of
the graph itself:

* norm/embedding weights are ordinary initializers named ``bert.<path>``;
* attention and MLP matrices are anonymous ``onnx::MatMul_N`` operands of nodes
  whose *name* preserves the original module path
  (``/bert/layers.7/mlp/Wo/MatMul``), because the exporter hoisted them out of
  the module tree;
* the projection is ``/projection_layers.N/{linear,residual}/MatMul``.

Supported export layout
-----------------------
This reader targets the historical Optimum-style ColBERT export published for
the pinned checkpoint, ``lightonai/mLateOn`` revision
``edd378f99593c0ac8a15518b97ad89786b02685e``: ``bert.*`` initializers,
``/bert/layers.N/{attn/Wqkv,attn/Wo,mlp/Wi,mlp/Wo}/MatMul`` encoder nodes, and the
``/projection_layers.N/{linear,residual}/MatMul`` projection nodes.

Newer ``torch.onnx.export`` *dynamo* re-exports of the same model use a different
naming scheme (``transformer.*`` tensors, ``dense_layers.*`` projection nodes,
``node_*`` node names) and are **not** supported here; they are refused with an
explicit error. Serve such a directory from ``model.safetensors`` (the default
weight source, and what the loader prefers whenever it is present) instead.

External data
-------------
A graph whose weights live in sidecar files (``model.onnx`` +
``model.onnx_data``/``*.bin``, which ONNX uses once a model exceeds 2 GB) is
resolved against the *graph's own directory*, not the process working directory,
so loading does not depend on where the server was started.

Quantized exports
-----------------
``model_int8.onnx`` is a *dynamically quantized* export -- its matrices are
stored as int8 with a scale (and usually a zero point) and fed through
``MatMulInteger``. Those are dequantized here. onnxruntime's per-channel
quantizer stores one scale per *output feature*, i.e. the last axis of the ONNX
MatMul B ``[in, out]``; per-tensor (scalar) scales also occur. Those are the only
scale shapes this reader accepts -- any other shape cannot be broadcast against
the weight along that axis, and applying it in a transposed orientation would
quietly corrupt every vector, so it is a hard error. (The per-channel axis for
MatMul weights is empirically the last axis in onnxruntime's dynamic
quantization; older exports using axis 0 on a square weight would be
indistinguishable from that and are not supported.)

Such a graph also quantizes its *activations* per MatMul
(``DynamicQuantizeLinear``), which a float forward pass cannot reproduce, so int8
weights are an approximation of that artifact rather than bit parity with it.
Whether a graph is quantized is decided from its operators, not its file name.

Import of ``onnx`` is deferred to call time so oMLX keeps working without it.
"""

import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from .colbert_projection import ColbertDenseSpec, ColbertLayoutError

logger = logging.getLogger(__name__)

# Candidate graph files live with the layout detection (colbert_projection),
# which needs to know them without importing this module.

# Module-tree prefix the exporter used for the transformer, and the matching
# initializer prefix (ONNX initializer names carry no leading slash).
_NODE_PREFIX = "/bert/"
_INITIALIZER_PREFIX = "bert."

# Ops that can carry a weight operand.
_MATMUL_OPS = frozenset({"MatMul", "MatMulInteger", "Gemm"})

_INTEGER_DTYPES = frozenset({"int8", "uint8", "int16", "uint16", "int32"})

# Ops that only appear in a graph whose matrices/activations were quantized.
_QUANTIZED_OPS = frozenset(
    {
        "MatMulInteger",
        "DynamicQuantizeLinear",
        "QLinearMatMul",
        "QLinearConv",
        "DequantizeLinear",
    }
)

# What this reader understands, quoted in every layout error so an unsupported
# (e.g. newer dynamo) export says what to do instead of just failing.
_SUPPORTED_LAYOUT = (
    "'bert.*' initializers, '/bert/layers.N/.../MatMul' encoder node names and "
    "'/projection_layers.N/{linear,residual}/MatMul' projection node names"
)
_SUPPORTED_REVISION = (
    "lightonai/mLateOn revision edd378f99593c0ac8a15518b97ad89786b02685e"
)
_UNSUPPORTED_REMEDY = (
    "Newer torch.onnx dynamo exports ('transformer.*' tensors, "
    "'dense_layers.*' projection nodes) are not supported: serve a directory "
    "that ships model.safetensors, which is the default weight source."
)


@dataclass(frozen=True)
class OnnxDenseWeights:
    """One projection module read from the graph, in the MLX/torch [out, in] layout."""

    linear: mx.array
    residual: mx.array | None


@dataclass(frozen=True)
class OnnxColbertWeights:
    """Every weight the ColBERT loader needs, read from one ONNX graph."""

    graph_path: Path
    quantized: bool
    """Whether the graph stores its matrices as integers (detected from its ops)."""

    encoder: dict[str, mx.array]
    """MLX ``ModernBertModel`` parameter name -> array, all in MLX layout."""

    projection: tuple[OnnxDenseWeights, ...]
    """Projection modules in pipeline order."""


def graph_is_quantized(graph: Any) -> bool:
    """Whether ``graph`` stores its matrices as integers (dynamic quantization).

    Decided from the graph's operators rather than the file name: the name is
    only the pinned exporter's convention, and a graph named ``model.onnx`` can
    still carry ``MatMulInteger``.
    """
    return any(node.op_type in _QUANTIZED_OPS for node in graph.node)


def _import_onnx():
    """Import the ``onnx`` package, or explain how to get it."""
    try:
        import onnx
    except ImportError as e:  # pragma: no cover - exercised via the message only
        raise ColbertLayoutError(
            "reading ColBERT weights from an ONNX graph requires the 'onnx' "
            "package, which is not installed. Install it (pip install onnx) or "
            "point the model at a directory that ships model.safetensors."
        ) from e
    return onnx


def _as_mx(array: Any) -> mx.array:
    return mx.array(array)


def _initializer_array(tensor: Any, numpy_helper: Any, base_dir: str) -> Any:
    """Convert one initializer, resolving external-data sidecars against ``base_dir``."""
    return numpy_helper.to_array(tensor, base_dir=base_dir)


def _scales_broadcast(values_shape: tuple[int, ...], scales_shape: tuple[int, ...]):
    """Whether a quantizer scale shape can be applied along the weight's last axis.

    onnxruntime's per-channel quantization writes one scale per output feature
    (the last axis of the ONNX MatMul B ``[in, out]``); per-tensor writes a
    scalar. Anything else is rejected by the caller.
    """
    if len(values_shape) != 2:
        return False
    if len(scales_shape) == 0:
        return True
    if len(scales_shape) == 1:
        return scales_shape[0] in (1, values_shape[-1])
    if len(scales_shape) == 2:
        return scales_shape[-1] in (1, values_shape[-1]) and scales_shape[0] in (
            1,
            values_shape[0],
        )
    return False


def _dequantize(
    quantized: Any,
    scale: Any,
    zero_point: Any,
    numpy_helper: Any,
    base_dir: str,
    name: str,
) -> Any:
    """``(q - zero_point) * scale``, the inverse of ONNX integer quantization."""
    values = _initializer_array(quantized, numpy_helper, base_dir).astype("float32")
    scales = _initializer_array(scale, numpy_helper, base_dir).astype("float32")
    if not _scales_broadcast(tuple(values.shape), tuple(scales.shape)):
        raise ColbertLayoutError(
            f"quantized ONNX weight {name} has shape {tuple(values.shape)} but its "
            f"scale has shape {tuple(scales.shape)}, which cannot be applied along "
            "the output-feature (last) axis. Only per-tensor (scalar) and "
            "per-output-feature scales are supported; a transposed or axis-0 "
            "quantizer layout is refused rather than silently mis-scaled."
        )
    if zero_point is not None:
        zeros = _initializer_array(zero_point, numpy_helper, base_dir).astype("float32")
        if tuple(zeros.shape) != tuple(scales.shape):
            raise ColbertLayoutError(
                f"quantized ONNX weight {name} has scale shape "
                f"{tuple(scales.shape)} but zero-point shape {tuple(zeros.shape)}; "
                "ONNX requires them to match."
            )
        values = values - zeros
    return values * scales


def _weight_operand(
    operand: str | None,
    initializers: dict[str, Any],
    numpy_helper: Any,
    base_dir: str,
) -> Any | None:
    """Resolve a MatMul B operand to a float array, dequantizing when needed.

    Returns None when the operand is an activation (no initializer), which is how
    attention score matmuls are filtered out.
    """
    if operand is None:
        return None
    tensor = initializers.get(operand)
    if tensor is None:
        return None
    array = _initializer_array(tensor, numpy_helper, base_dir)
    if array.dtype.name not in _INTEGER_DTYPES:
        return array
    # Dynamically quantized weight: "<name>_quantized" with a sibling
    # "<name>_scale" and (usually) "<name>_zero_point".
    base = operand[: -len("_quantized")] if operand.endswith("_quantized") else operand
    scale = initializers.get(f"{base}_scale")
    if scale is None:
        raise ColbertLayoutError(
            f"quantized ONNX weight {operand} has no {base}_scale beside it"
        )
    return _dequantize(
        tensor,
        scale,
        initializers.get(f"{base}_zero_point"),
        numpy_helper,
        base_dir,
        operand,
    )


def _module_path(node_name: str) -> str | None:
    """``/bert/layers.7/mlp/Wo/MatMul`` -> ``layers.7.mlp.Wo``."""
    if not node_name.startswith(_NODE_PREFIX):
        return None
    stem = node_name[len(_NODE_PREFIX) :]
    if "/" not in stem:
        return None
    return stem.rsplit("/", 1)[0].replace("/", ".")


def _encoder_weights(
    graph: Any, numpy_helper: Any, base_dir: str
) -> dict[str, mx.array]:
    """MLX parameter name -> array for everything under ``/bert/``.

    Two naming systems meet here. Norms and the token embedding are ordinary
    initializers (``bert.final_norm.weight``), so the ``bert.`` prefix is simply
    dropped. The attention and MLP matrices are anonymous MatMul operands, so the
    node name -- which still records the module path -- is the only mapping left;
    their operand is transposed from the ONNX ``[in, out]`` MatMul B layout into
    the ``[out, in]`` layout ``nn.Linear`` uses.
    """
    initializers = {init.name: init for init in graph.initializer}
    weights: dict[str, mx.array] = {}

    for init in graph.initializer:
        if init.name.startswith(_INITIALIZER_PREFIX):
            weights[init.name[len(_INITIALIZER_PREFIX) :]] = _as_mx(
                _initializer_array(init, numpy_helper, base_dir)
            )

    for node in graph.node:
        if node.op_type not in _MATMUL_OPS or len(node.input) < 2:
            continue
        module = _module_path(node.name)
        if module is None:
            continue
        operand = _weight_operand(node.input[1], initializers, numpy_helper, base_dir)
        if operand is None:
            # No initializer operand: an attention score matmul, not a weight.
            continue
        if operand.ndim != 2:
            continue
        weights[f"{module}.weight"] = _as_mx(operand.T)

    return weights


def _projection_weights(
    graph: Any,
    dense: Sequence[ColbertDenseSpec],
    numpy_helper: Any,
    base_dir: str,
) -> tuple[OnnxDenseWeights, ...]:
    """Read ``/projection_layers.{i}/{linear,residual}`` in pipeline order."""
    initializers = {init.name: init for init in graph.initializer}
    by_prefix: dict[str, Any] = {}
    for node in graph.node:
        if node.op_type in _MATMUL_OPS and node.name.startswith("/projection_layers."):
            by_prefix[node.name] = node

    projection: list[OnnxDenseWeights] = []
    for index, spec in enumerate(dense):
        if spec.bias:
            raise ColbertLayoutError(
                f"{spec.directory.name}/config.json sets bias=true, which the "
                "ONNX weight source does not support (the pinned exports are "
                "bias-free). Use the safetensors source for this checkpoint."
            )
        if spec.residual_is_identity:
            # pylate creates no residual MatMul for a same-width layer; the
            # projection applies an identity skip instead.
            stray = next(
                (
                    name
                    for name in by_prefix
                    if name.startswith(f"/projection_layers.{index}/residual/")
                ),
                None,
            )
            if stray is not None:
                raise ColbertLayoutError(
                    f"{spec.directory.name}/config.json sets use_residual on a "
                    f"{spec.in_features} -> {spec.out_features} layer, which pylate "
                    f"implements as an identity skip, but the ONNX graph also has a "
                    f"residual node {stray}. The artifact is inconsistent."
                )
        found = {}
        for role in ("linear", "residual"):
            if role == "residual" and not spec.needs_residual_matrix:
                # A same-width module is an identity skip and has no residual
                # MatMul; the last module usually has no residual at all.
                continue
            prefix = f"/projection_layers.{index}/{role}/"
            node = next(
                (node for name, node in by_prefix.items() if name.startswith(prefix)),
                None,
            )
            if node is None:
                raise ColbertLayoutError(
                    f"ONNX graph has no {prefix}* MatMul for projection module "
                    f"{index} ({spec.directory.name}). This reader supports the "
                    f"ColBERT export layout with {_SUPPORTED_LAYOUT} "
                    f"({_SUPPORTED_REVISION}). {_UNSUPPORTED_REMEDY}"
                )
            operand = _weight_operand(
                node.input[1] if len(node.input) > 1 else None,
                initializers,
                numpy_helper,
                base_dir,
            )
            if operand is None:
                raise ColbertLayoutError(
                    f"ONNX graph node {node.name} has no initializer weight operand"
                )
            if operand.ndim != 2:
                raise ColbertLayoutError(
                    f"ONNX graph node {node.name} has a {operand.ndim}-D weight "
                    "operand; a projection MatMul needs a 2-D [in, out] matrix"
                )
            found[role] = operand
        projection.append(
            OnnxDenseWeights(
                linear=_as_mx(found["linear"].T),
                residual=(
                    _as_mx(found["residual"].T) if spec.needs_residual_matrix else None
                ),
            )
        )
    return tuple(projection)


def read_colbert_onnx(
    graph_path: Path,
    dense: Sequence[ColbertDenseSpec],
) -> OnnxColbertWeights:
    """Read every encoder and projection weight out of ``graph_path``.

    External-data sidecars are resolved against the graph's directory, so a
    >2 GB export loads regardless of the current working directory.

    Raises:
        ColbertLayoutError: when ``onnx`` is unavailable or the graph does not
            carry the weights in the supported ColBERT export layout.
    """
    onnx = _import_onnx()
    from onnx import numpy_helper

    graph_path = Path(graph_path)
    base_dir = str(graph_path.parent)
    model = onnx.load(str(graph_path), load_external_data=False)
    graph = model.graph

    encoder = _encoder_weights(graph, numpy_helper, base_dir)
    if not encoder:
        raise ColbertLayoutError(
            f"{graph_path} carries no weights in the ColBERT export layout this "
            f"reader supports ({_SUPPORTED_LAYOUT}; {_SUPPORTED_REVISION}). "
            f"{_UNSUPPORTED_REMEDY}"
        )
    projection = _projection_weights(graph, dense, numpy_helper, base_dir)
    quantized = graph_is_quantized(graph)

    # Release the protobuf (which holds a full copy of the raw tensor bytes)
    # before the caller builds the model.
    del model, graph
    gc.collect()

    logger.info(
        "Read %d encoder tensors and %d projection modules from %s%s",
        len(encoder),
        len(projection),
        graph_path,
        " (dynamically quantized: matrices dequantized)" if quantized else "",
    )
    return OnnxColbertWeights(
        graph_path=graph_path,
        quantized=quantized,
        encoder=encoder,
        projection=projection,
    )
