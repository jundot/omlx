# SPDX-License-Identifier: Apache-2.0
"""Projected per-token (ColBERT / late-interaction) support for pylate exports.

``MLXEmbeddingModel.embed_token_ids`` returns the encoder's ``last_hidden_state``
verbatim. That is the right contract for a plain embedder -- Qwen3-Embedding is
served that way -- but it is the wrong contract for a pylate ColBERT model, whose
output is a *projected*, per-token L2-normalized vector.

The projection is not part of the transformer checkpoint. sentence-transformers
describes it in ``modules.json`` as a chain of ``pylate.models.Dense.Dense``
modules. The pinned ONNX export folds those modules **and** the final per-token
L2 into the graph, so its output is the contract to reproduce. Reading that graph
back pins the recipe exactly::

    h = encoder(input_ids, attention_mask)     # (B, T, 768)
    x = h @ W0.linear + h @ W0.residual        # (B, T, 1536)
    x = x @ W1.linear + x @ W1.residual        # (B, T, 768)
    x = x @ W2.linear                          # (B, T, 128)
    x = x / clip(||x||_2, axis=-1, eps)        # per-token L2

Two facts the graph makes explicit and that are easy to get wrong:

* ``residual`` is a **second learned MatMul over the same input**, added to the
  ``linear`` result -- not an identity skip, which is not even shape-compatible
  here (``1_Dense`` maps 768 -> 1536).
* The final Dense has no residual, and the per-token L2 is part of the model
  contract rather than a caller convention. The export spells it
  ``ReduceL2 -> Clip(min=1e-12) -> Div``.

Weight sources
--------------

A ColBERT repository can ship two weight sets that are **not the same numbers**:
the sentence-transformers ``model.safetensors`` and the exported ``model.onnx``
(measured on ``lightonai/LateOn-multilingual``: projection weights correlate
0.99996 with a 1.1e-3 max difference, giving ~0.99 per-token cosine between the
two outputs). Serving one set where the other produced the frozen retrieval
measurements silently changes the representation, so the source is explicit in
the layout, logged at load, reported by ``get_model_info``, and overridable with
``OMLX_COLBERT_WEIGHT_SOURCE``.

Orientation differs between the two: ``pylate.models.Dense.Dense`` wraps
``torch.nn.Linear``, so its safetensors weight is ``[out_features,
in_features]`` and the operation is ``x @ W.T``; the ONNX MatMul B operand is
``[in_features, out_features]``. The ONNX reader transposes at the boundary so
both sources share one ``[out, in]`` forward path, and the loader validates every
shape so a transposed export fails loudly instead of quietly producing noise.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

# `model_type` written by pylate/sentence-transformers for a ColBERT checkpoint.
COLBERT_MODEL_TYPE = "ColBERT"

# Where a ColBERT projection can be read from. They are not interchangeable; see
# the module docstring.
WEIGHT_SOURCE_SAFETENSORS = "safetensors"
WEIGHT_SOURCE_ONNX = "onnx"
WEIGHT_SOURCES = (WEIGHT_SOURCE_ONNX, WEIGHT_SOURCE_SAFETENSORS)

# Selects the source for a directory that ships both. Anything else (unset,
# "auto") keeps the default preference below.
WEIGHT_SOURCE_ENV = "OMLX_COLBERT_WEIGHT_SOURCE"

# Candidate ONNX graphs, most preferred first: the fp32 export defined the frozen
# measurements, the int8 export is a quantized derivative of it.
ONNX_GRAPH_NAMES: Tuple[str, ...] = ("model.onnx", "model_int8.onnx")

_SENTENCE_TRANSFORMERS_CONFIG = "config_sentence_transformers.json"
_MODULES_FILE = "modules.json"
_WEIGHTS_FILE = "model.safetensors"

# The per-token L2 the export folds into the graph: ReduceL2 -> Clip(min) -> Div.
# Anything at or below this is a degenerate row (the export yields a zero
# vector); real token vectors are many orders of magnitude above it.
_L2_EPS = 1e-12

# HF's nested rope dialect -> mlx-embeddings' flat ModernBERT fields.
_ROPE_PARAMETER_FIELDS = (
    ("full_attention", "global_rope_theta"),
    ("sliding_attention", "local_rope_theta"),
)


class ColbertLayoutError(ValueError):
    """A directory that claims to be a pylate ColBERT export cannot be served."""


@dataclass(frozen=True)
class ColbertDenseSpec:
    """One ``pylate.models.Dense.Dense`` module of a ColBERT projection chain."""

    directory: Path
    """Directory holding the module's ``config.json`` (and safetensors, if any)."""

    in_features: int
    """Width consumed by this module."""

    out_features: int
    """Width produced by this module."""

    use_residual: bool
    """Whether ``residual.weight`` is applied to the same input and added."""

    bias: bool
    """Whether ``linear.bias`` (and ``residual.bias``) exist."""

    weights: Optional[Path] = None
    """``model.safetensors`` for the safetensors source; None when ONNX-sourced."""


@dataclass(frozen=True)
class ColbertLayout:
    """A pylate ColBERT export that oMLX can load and serve per-token."""

    model_path: Path
    """Root of the sentence-transformers export."""

    weight_source: str
    """``onnx`` or ``safetensors`` -- which artifact supplies the weights."""

    dense: Tuple[ColbertDenseSpec, ...]
    """Projection modules, in pipeline order."""

    encoder_weights: Optional[Path] = None
    """``model.safetensors`` holding the encoder weights (safetensors source)."""

    onnx_graph: Optional[Path] = None
    """``model.onnx`` / ``model_int8.onnx`` supplying the weights (ONNX source)."""

    @property
    def output_dim(self) -> int:
        """Projected per-token width, read from the last Dense module."""
        return self.dense[-1].out_features

    @property
    def layer_count(self) -> int:
        """Number of Dense modules in the projection chain."""
        return len(self.dense)

    @property
    def encoder_input_dim(self) -> int:
        """Width the first Dense module consumes (the encoder's hidden size)."""
        return self.dense[0].in_features

    @property
    def weights_path(self) -> Path:
        """The file the weights actually came from."""
        if self.weight_source == WEIGHT_SOURCE_ONNX and self.onnx_graph is not None:
            return self.onnx_graph
        assert self.encoder_weights is not None
        return self.encoder_weights


def _load_json(path: Path) -> Optional[Any]:
    """Read a JSON file, returning None for anything unreadable."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _is_dense_module(module_type: Any) -> bool:
    """Whether a ``modules.json`` entry is a projection Dense.

    pylate registers ``pylate.models.Dense.Dense``; accept any dotted ``*.Dense``
    so a renamed pylate package does not silently drop the projection.
    """
    return isinstance(module_type, str) and module_type.endswith(".Dense")


def _dense_modules(model_path: Path) -> List[Dict[str, Any]]:
    """Projection module entries from ``modules.json``, in pipeline order."""
    modules = _load_json(model_path / _MODULES_FILE)
    if not isinstance(modules, list):
        return []
    return [
        module
        for module in modules
        if isinstance(module, dict) and _is_dense_module(module.get("type"))
    ]


def is_pylate_colbert_dir(model_path: Path) -> bool:
    """Whether ``model_path`` is a pylate ColBERT sentence-transformers export.

    The marker is the ColBERT declaration in ``config_sentence_transformers.json``
    *plus* at least one Dense module in ``modules.json``. A plain embedding export
    also ships ``modules.json`` (Transformer/Pooling/Normalize) but never the
    ColBERT declaration, and a directory that merely has "colbert" in its name is
    not evidence. Both files are read without raising so this stays a predicate.
    """
    model_path = Path(model_path)
    declared = _load_json(model_path / _SENTENCE_TRANSFORMERS_CONFIG)
    if not isinstance(declared, dict):
        return False
    if declared.get("model_type") != COLBERT_MODEL_TYPE:
        return False
    return bool(_dense_modules(model_path))


def find_onnx_graph(model_path: Path) -> Optional[Path]:
    """The best ONNX graph in ``model_path``, or None."""
    model_path = Path(model_path)
    for name in ONNX_GRAPH_NAMES:
        candidate = model_path / name
        if candidate.is_file():
            return candidate
    return None


def _requested_weight_source(explicit: Optional[str]) -> Optional[str]:
    """Resolve the requested source from an argument, then the environment."""
    value = explicit if explicit is not None else os.getenv(WEIGHT_SOURCE_ENV)
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in ("", "auto"):
        return None
    if normalized not in WEIGHT_SOURCES:
        raise ColbertLayoutError(
            f"unknown ColBERT weight source {value!r}; expected one of "
            f"{', '.join(WEIGHT_SOURCES)} (set through {WEIGHT_SOURCE_ENV})"
        )
    return normalized


def resolve_weight_source(
    model_path: Path,
    explicit: Optional[str] = None,
) -> str:
    """Choose where a ColBERT export takes its weights from.

    An explicit request (argument or ``OMLX_COLBERT_WEIGHT_SOURCE``) wins. With
    nothing requested, the ONNX graph is preferred when present: it is the
    artifact the frozen retrieval measurements came from, and where both exist
    they are not the same numbers, so the choice must not be silent.
    """
    requested = _requested_weight_source(explicit)
    if requested is not None:
        return requested
    if find_onnx_graph(Path(model_path)) is not None:
        return WEIGHT_SOURCE_ONNX
    return WEIGHT_SOURCE_SAFETENSORS


def _dense_spec(
    model_path: Path,
    module: Dict[str, Any],
    require_safetensors: bool,
) -> ColbertDenseSpec:
    """Resolve one Dense module entry into a validated spec."""
    relative = str(module.get("path") or "")
    if not relative:
        raise ColbertLayoutError(
            f"Dense module entry {module!r} in {_MODULES_FILE} has no path"
        )
    directory = Path(model_path) / relative
    config = _load_json(directory / "config.json")
    if not isinstance(config, dict):
        raise ColbertLayoutError(
            f"ColBERT projection module {directory} has no readable config.json"
        )
    try:
        in_features = int(config["in_features"])
        out_features = int(config["out_features"])
    except (KeyError, TypeError, ValueError) as e:
        raise ColbertLayoutError(
            f"ColBERT projection module {directory} does not declare integer "
            f"in_features/out_features: {config!r}"
        ) from e

    weights = directory / _WEIGHTS_FILE
    if require_safetensors and not weights.is_file():
        raise ColbertLayoutError(
            f"ColBERT projection module {directory} has no {_WEIGHTS_FILE}, which "
            f"the safetensors weight source requires. Set "
            f"{WEIGHT_SOURCE_ENV}={WEIGHT_SOURCE_ONNX} to read this checkpoint "
            "from its ONNX graph instead."
        )
    return ColbertDenseSpec(
        directory=directory,
        in_features=in_features,
        out_features=out_features,
        use_residual=bool(config.get("use_residual")),
        bias=bool(config.get("bias")),
        weights=weights if weights.is_file() else None,
    )


def resolve_colbert_layout(
    model_path: Path,
    weight_source: Optional[str] = None,
) -> ColbertLayout:
    """Validate a pylate ColBERT export and describe its projection chain.

    Raises:
        ColbertLayoutError: when the directory is a ColBERT export but cannot be
            served -- no projection modules, no usable weight artifact, or an
            inconsistent projection chain.
    """
    model_path = Path(model_path)
    modules = _dense_modules(model_path)
    if not modules:
        raise ColbertLayoutError(
            f"{model_path} declares model_type '{COLBERT_MODEL_TYPE}' but lists no "
            f"Dense projection module in {_MODULES_FILE}"
        )

    source = resolve_weight_source(model_path, weight_source)
    dense = tuple(
        _dense_spec(
            model_path,
            module,
            require_safetensors=(source == WEIGHT_SOURCE_SAFETENSORS),
        )
        for module in modules
    )
    for previous, current in zip(dense, dense[1:]):
        if previous.out_features != current.in_features:
            raise ColbertLayoutError(
                f"ColBERT projection chain is inconsistent: {previous.directory.name} "
                f"outputs {previous.out_features} but {current.directory.name} "
                f"consumes {current.in_features}"
            )

    encoder_weights = model_path / _WEIGHTS_FILE
    onnx_graph = find_onnx_graph(model_path)
    if source == WEIGHT_SOURCE_SAFETENSORS and not encoder_weights.is_file():
        raise ColbertLayoutError(
            f"ColBERT export at {model_path} has no {_WEIGHTS_FILE} for its "
            "encoder. Set "
            f"{WEIGHT_SOURCE_ENV}={WEIGHT_SOURCE_ONNX} to read this checkpoint "
            "from its ONNX graph instead."
        )
    if source == WEIGHT_SOURCE_ONNX and onnx_graph is None:
        raise ColbertLayoutError(
            f"ColBERT export at {model_path} has no ONNX graph "
            f"({' or '.join(ONNX_GRAPH_NAMES)}) to take weights from"
        )

    layout = ColbertLayout(
        model_path=model_path,
        weight_source=source,
        dense=dense,
        encoder_weights=encoder_weights if encoder_weights.is_file() else None,
        onnx_graph=onnx_graph,
    )
    logger.debug(
        "ColBERT layout at %s: source=%s (%s), %d Dense modules, output dim %d",
        model_path,
        layout.weight_source,
        layout.weights_path.name,
        len(dense),
        layout.output_dim,
    )
    return layout


def flatten_rope_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """Copy HF's nested ``rope_parameters`` thetas onto mlx-embeddings' flat fields.

    ModernBERT configs written by transformers>=5 -- this checkpoint included --
    declare RoPE as::

        "rope_parameters": {"full_attention":    {"rope_theta": 160000},
                            "sliding_attention": {"rope_theta": 160000}}

    while mlx-embeddings' ``modernbert.ModelArgs`` reads the flat
    ``global_rope_theta`` / ``local_rope_theta`` fields. Neither flat key is
    present in such a config, so ``local_rope_theta`` silently falls back to its
    10000 default and every sliding-window layer (two of every three here) runs
    RoPE at the wrong base. Measured against the same checkpoint's transformers
    implementation: min per-token cosine 0.99672 with the default, 0.99999966
    once the nested thetas are applied.

    An explicitly declared flat field always wins, so configs in the older
    dialect are untouched.
    """
    nested = config.get("rope_parameters")
    if not isinstance(nested, dict):
        return dict(config)
    resolved = dict(config)
    for section, field in _ROPE_PARAMETER_FIELDS:
        if field in resolved:
            continue
        entry = nested.get(section)
        if not isinstance(entry, dict):
            continue
        theta = entry.get("rope_theta")
        if isinstance(theta, (int, float)) and not isinstance(theta, bool):
            resolved[field] = float(theta)
    return resolved


def l2_normalize_tokens(vectors: mx.array, eps: float = _L2_EPS) -> mx.array:
    """Per-token L2 normalization, as the pinned export computes it.

    The graph is ``ReduceL2`` -> ``Clip(min=eps)`` -> ``Div``, so a degenerate
    all-zero row divides by ``eps`` and stays zero instead of producing NaN.
    """
    norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / mx.clip(norms, eps, None)


def _checked_matrix(
    matrix: Optional[mx.array],
    key: str,
    spec: ColbertDenseSpec,
) -> Optional[mx.array]:
    """Validate one ``[out_features, in_features]`` matrix."""
    if matrix is None:
        return None
    expected = (spec.out_features, spec.in_features)
    if tuple(matrix.shape) != expected:
        raise ColbertLayoutError(
            f"{key} for {spec.directory.name} has shape {tuple(matrix.shape)}, but "
            f"its config.json declares [out_features, in_features] = {expected} "
            "(torch nn.Linear layout). A transposed export cannot be projected by "
            "matrix multiplication."
        )
    return matrix


class ColbertProjection:
    """The Dense chain plus the final per-token L2.

    Mirrors the pinned ONNX export's ``/projection_layers.*`` and ``/ReduceL2``
    subgraph. Weights are plain ``mx.array`` attributes rather than an
    ``nn.Module``: the chain is five fixed matrices with no other state, and the
    encoder it accompanies owns dtype policy. MLX promotes the activations to the
    wider of the two dtypes, so a reduced-precision checkpoint cannot silently
    truncate the projection.
    """

    def __init__(self, dense: Sequence[ColbertDenseSpec]):
        if not dense:
            raise ValueError("a ColBERT projection requires at least one Dense module")
        self.dense: Tuple[ColbertDenseSpec, ...] = tuple(dense)
        self._linear: List[mx.array] = []
        self._linear_bias: List[Optional[mx.array]] = []
        self._residual: List[Optional[mx.array]] = []
        self._residual_bias: List[Optional[mx.array]] = []
        self._loaded = False

    @property
    def output_dim(self) -> int:
        """Projected per-token width."""
        return self.dense[-1].out_features

    @property
    def layer_count(self) -> int:
        """Number of Dense modules in the chain."""
        return len(self.dense)

    @classmethod
    def from_layout(cls, layout: ColbertLayout) -> "ColbertProjection":
        """Load every Dense module of a safetensors-sourced layout."""
        projection = cls(layout.dense)
        for spec in layout.dense:
            if spec.weights is None:
                raise ColbertLayoutError(
                    f"{spec.directory} has no {_WEIGHTS_FILE} to load the "
                    "projection from"
                )
            weights = mx.load(str(spec.weights))
            linear = _checked_matrix(
                weights.get("linear.weight"), "linear.weight", spec
            )
            if linear is None:
                raise ColbertLayoutError(
                    f"{spec.weights} has no linear.weight (keys: "
                    f"{sorted(weights.keys())})"
                )
            residual = (
                _checked_matrix(weights.get("residual.weight"), "residual.weight", spec)
                if spec.use_residual
                else None
            )
            if spec.use_residual and residual is None:
                raise ColbertLayoutError(
                    f"{spec.directory.name}/config.json sets use_residual but "
                    f"{spec.weights} has no residual.weight (keys: "
                    f"{sorted(weights.keys())})"
                )
            projection._linear.append(linear)
            projection._linear_bias.append(
                weights.get("linear.bias") if spec.bias else None
            )
            projection._residual.append(residual)
            projection._residual_bias.append(
                weights.get("residual.bias")
                if spec.bias and residual is not None
                else None
            )
        return projection._finalize()

    @classmethod
    def from_matrices(
        cls,
        dense: Sequence[ColbertDenseSpec],
        linear: Sequence[Optional[mx.array]],
        residual: Sequence[Optional[mx.array]],
    ) -> "ColbertProjection":
        """Build a projection from already-loaded ``[out, in]`` matrices."""
        projection = cls(dense)
        if (
            len(linear) != projection.layer_count
            or len(residual) != projection.layer_count
        ):
            raise ColbertLayoutError(
                f"projection chain length mismatch: {projection.layer_count} Dense "
                f"modules but {len(linear)} linear / {len(residual)} residual matrices"
            )
        for index, spec in enumerate(projection.dense):
            matrix = _checked_matrix(linear[index], f"projection_layers.{index}", spec)
            if matrix is None:
                raise ColbertLayoutError(
                    f"projection module {index} ({spec.directory.name}) has no linear "
                    "weight matrix"
                )
            if spec.use_residual and residual[index] is None:
                raise ColbertLayoutError(
                    f"projection module {index} ({spec.directory.name}) sets "
                    "use_residual but has no residual weight matrix"
                )
            projection._linear.append(matrix)
            projection._linear_bias.append(None)
            projection._residual.append(residual[index])
            projection._residual_bias.append(None)
        return projection._finalize()

    def _finalize(self) -> "ColbertProjection":
        """Evaluate the loaded matrices once, at load time."""
        mx.eval(
            [
                array
                for array in (
                    self._linear
                    + self._residual
                    + self._linear_bias
                    + self._residual_bias
                )
                if array is not None
            ]
        )
        self._loaded = True
        return self

    def __call__(self, hidden: mx.array) -> mx.array:
        """Project ``(batch, seq_len, encoder_dim)`` to unit per-token vectors."""
        if not self._loaded:
            raise RuntimeError("ColBERT projection weights are not loaded")
        x = hidden
        for index, spec in enumerate(self.dense):
            y = x @ mx.transpose(self._linear[index])
            bias = self._linear_bias[index]
            if bias is not None:
                y = y + bias
            if spec.use_residual:
                y = y + x @ mx.transpose(self._residual[index])
                residual_bias = self._residual_bias[index]
                if residual_bias is not None:
                    y = y + residual_bias
            x = y
        return l2_normalize_tokens(x)

    def __repr__(self) -> str:
        return (
            f"<ColbertProjection layers={self.layer_count} "
            f"output_dim={self.output_dim} loaded={self._loaded}>"
        )
