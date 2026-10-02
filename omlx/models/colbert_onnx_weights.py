# SPDX-License-Identifier: Apache-2.0
"""Read a pinned ColBERT ONNX export as MLX parameter arrays.

A pylate ColBERT repository can ship two weight sets that are *not* the same
numbers: the sentence-transformers ``model.safetensors`` and the exported
``model.onnx``. Measured on ``lightonai/LateOn-multilingual`` (both files from the
same HF revision, ``edd378f99593c0ac8a15518b97ad89786b02685e``):

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
* ``model_int8.onnx`` is a *dynamically quantized* export -- its matrices are
  stored as int8 with a scale (and zero point) and fed through ``MatMulInteger``.
  Those are dequantized here, but note that such a graph also quantizes its
  *activations* per MatMul (``DynamicQuantizeLinear``), which a float forward pass
  cannot reproduce, so int8 weights are an approximation of that artifact rather
  than bit parity with it.

Import of ``onnx`` is deferred to call time so oMLX keeps working without it.
"""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class OnnxDenseWeights:
    """One projection module read from the graph, in the MLX/torch [out, in] layout."""

    linear: mx.array
    residual: Optional[mx.array]


@dataclass(frozen=True)
class OnnxColbertWeights:
    """Every weight the ColBERT loader needs, read from one ONNX graph."""

    graph_path: Path
    quantized: bool
    """Whether the graph stores its matrices as int8 + scale (dynamic quantization)."""

    encoder: Dict[str, mx.array]
    """MLX ``ModernBertModel`` parameter name -> array, all in MLX layout."""

    projection: Tuple[OnnxDenseWeights, ...]
    """Projection modules in pipeline order."""


def is_quantized_graph(graph_path: Path) -> bool:
    """Whether ``graph_path`` is named as a quantized export."""
    return "int8" in Path(graph_path).name


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


def _initializer_array(tensor: Any, numpy_helper: Any) -> Any:
    return numpy_helper.to_array(tensor)


def _dequantize(
    quantized: Any,
    scale: Any,
    zero_point: Any,
    numpy_helper: Any,
) -> Any:
    """``(q - zero_point) * scale``, the inverse of ONNX integer quantization."""
    values = _initializer_array(quantized, numpy_helper).astype("float32")
    scales = _initializer_array(scale, numpy_helper).astype("float32")
    if zero_point is not None:
        values = values - _initializer_array(zero_point, numpy_helper).astype("float32")
    return values * scales


def _weight_operand(
    operand: Optional[str],
    initializers: Dict[str, Any],
    numpy_helper: Any,
) -> Optional[Any]:
    """Resolve a MatMul B operand to a float array, dequantizing when needed.

    Returns None when the operand is an activation (no initializer), which is how
    attention score matmuls are filtered out.
    """
    if operand is None:
        return None
    tensor = initializers.get(operand)
    if tensor is None:
        return None
    array = _initializer_array(tensor, numpy_helper)
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
        tensor, scale, initializers.get(f"{base}_zero_point"), numpy_helper
    )


def _module_path(node_name: str) -> Optional[str]:
    """``/bert/layers.7/mlp/Wo/MatMul`` -> ``layers.7.mlp.Wo``."""
    if not node_name.startswith(_NODE_PREFIX):
        return None
    stem = node_name[len(_NODE_PREFIX) :]
    if "/" not in stem:
        return None
    return stem.rsplit("/", 1)[0].replace("/", ".")


def _encoder_weights(graph: Any, numpy_helper: Any) -> Dict[str, mx.array]:
    """MLX parameter name -> array for everything under ``/bert/``.

    Two naming systems meet here. Norms and the token embedding are ordinary
    initializers (``bert.final_norm.weight``), so the ``bert.`` prefix is simply
    dropped. The attention and MLP matrices are anonymous MatMul operands, so the
    node name -- which still records the module path -- is the only mapping left;
    their operand is transposed from the ONNX ``[in, out]`` MatMul B layout into
    the ``[out, in]`` layout ``nn.Linear`` uses.
    """
    initializers = {init.name: init for init in graph.initializer}
    weights: Dict[str, mx.array] = {}

    for init in graph.initializer:
        if init.name.startswith(_INITIALIZER_PREFIX):
            weights[init.name[len(_INITIALIZER_PREFIX) :]] = _as_mx(
                _initializer_array(init, numpy_helper)
            )

    for node in graph.node:
        if node.op_type not in _MATMUL_OPS or len(node.input) < 2:
            continue
        module = _module_path(node.name)
        if module is None:
            continue
        operand = _weight_operand(node.input[1], initializers, numpy_helper)
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
) -> Tuple[OnnxDenseWeights, ...]:
    """Read ``/projection_layers.{i}/{linear,residual}`` in pipeline order."""
    initializers = {init.name: init for init in graph.initializer}
    by_prefix: Dict[str, Any] = {}
    for node in graph.node:
        if node.op_type in _MATMUL_OPS and node.name.startswith("/projection_layers."):
            by_prefix[node.name] = node

    projection: List[OnnxDenseWeights] = []
    for index, spec in enumerate(dense):
        if spec.bias:
            raise ColbertLayoutError(
                f"{spec.directory.name}/config.json sets bias=true, which the "
                "ONNX weight source does not support (the pinned exports are "
                "bias-free). Use the safetensors source for this checkpoint."
            )
        found = {}
        for role in ("linear", "residual"):
            if role == "residual" and not spec.use_residual:
                # The last projection module has no residual MatMul at all: the
                # export omits the node rather than emitting a zero one.
                continue
            prefix = f"/projection_layers.{index}/{role}/"
            node = next(
                (node for name, node in by_prefix.items() if name.startswith(prefix)),
                None,
            )
            if node is None:
                raise ColbertLayoutError(
                    f"ONNX graph has no {prefix}* MatMul for projection module "
                    f"{index} ({spec.directory.name})"
                )
            operand = _weight_operand(
                node.input[1] if len(node.input) > 1 else None,
                initializers,
                numpy_helper,
            )
            if operand is None:
                raise ColbertLayoutError(
                    f"ONNX graph node {node.name} has no initializer weight operand"
                )
            found[role] = operand
        projection.append(
            OnnxDenseWeights(
                linear=_as_mx(found["linear"].T),
                residual=_as_mx(found["residual"].T) if spec.use_residual else None,
            )
        )
    return tuple(projection)


def read_colbert_onnx(
    graph_path: Path,
    dense: Sequence[ColbertDenseSpec],
) -> OnnxColbertWeights:
    """Read every encoder and projection weight out of ``graph_path``.

    Raises:
        ColbertLayoutError: when ``onnx`` is unavailable or the graph does not
            carry the weights a ColBERT forward pass needs.
    """
    onnx = _import_onnx()
    from onnx import numpy_helper

    graph_path = Path(graph_path)
    model = onnx.load(str(graph_path), load_external_data=False)
    graph = model.graph

    encoder = _encoder_weights(graph, numpy_helper)
    if not encoder:
        raise ColbertLayoutError(
            f"{graph_path} carries no initializers under '{_INITIALIZER_PREFIX}'"
        )
    projection = _projection_weights(graph, dense, numpy_helper)
    quantized = is_quantized_graph(graph_path)

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
