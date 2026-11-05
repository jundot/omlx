# SPDX-License-Identifier: Apache-2.0
"""Projected per-token (ColBERT / late-interaction) support for pylate exports.

``MLXEmbeddingModel.embed_token_ids`` returns the encoder's ``last_hidden_state``
verbatim. That is the right contract for a plain embedder -- Qwen3-Embedding is
served that way -- but it is the wrong contract for a pylate ColBERT model, whose
output is a *projected*, per-token L2-normalized vector.

The projection is not part of the transformer checkpoint. sentence-transformers
describes it in ``modules.json`` as a chain of ``pylate.models.Dense.Dense``
modules. Each module is a ``torch.nn.Linear`` plus an optional residual, followed
by the module's declared activation function. Reproducing
``pylate.models.Dense.Dense.forward`` exactly gives::

    y = activation(h @ W.linear.T + b.linear)               # torch Linear
    if use_residual:
        y = y + (h if in_features == out_features else h @ W.residual.T)
    h = y
    ...
    h = h / clip(||h||_2, axis=-1, eps)                     # per-token L2

Three facts the module config makes explicit and that are easy to get wrong:

* The activation is applied to the *linear* output, **before** the residual is
  added, because that is the order in pylate's ``Dense.forward``. The published
  ColBERT projections declare ``Identity``, but a checkpoint that declares
  anything else is applied faithfully -- or rejected (see ``_ACTIVATION_MODULES``)
  rather than served un-activated.
* ``residual`` is an identity skip when ``in_features == out_features``; pylate
  only builds a ``residual`` submodule (a second learned ``nn.Linear``, with
  ``bias=False``) when the layer changes width. So a shape-changing layer adds
  ``h @ W.residual.T`` and a same-width layer adds ``h``.
* The per-token L2 is part of the model contract rather than a caller
  convention. The pinned export spells it ``ReduceL2 -> Clip(min=1e-12) -> Div``,
  which is ``torch.nn.functional.normalize``; ``_L2_EPS`` matches that clamp.

Weight sources
--------------
A ColBERT repository can ship two weight sets that are **not the same numbers**:
the sentence-transformers ``model.safetensors`` and the exported ``model.onnx``
(measured on the pinned checkpoint: projection weights correlate 0.99996 with a
1.1e-3 max difference, giving ~0.99 per-token cosine between the two outputs).
Serving one set where the other produced the frozen retrieval measurements
silently changes the representation, so the source is explicit in the layout,
logged at load, reported by ``get_model_info``, and overridable with
``OMLX_COLBERT_WEIGHT_SOURCE``.

``model.safetensors`` is the default source: it is native to MLX and every
sentence-transformers export ships it, whereas the ONNX graph is optional and
only readable in a specific historical export layout (see
``colbert_onnx_weights``). Both sources share this module's ``[out, in]``
forward path: the ONNX reader transposes at the boundary, and the loader
validates every shape so a transposed export fails loudly instead of quietly
producing noise.

Orientation: ``pylate.models.Dense.Dense`` wraps ``torch.nn.Linear``, so its
safetensors weight is ``[out_features, in_features]`` and the operation is
``x @ W.T``; the ONNX MatMul B operand is ``[in_features, out_features]``.
"""

import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
ONNX_GRAPH_NAMES: tuple[str, ...] = ("model.onnx", "model_int8.onnx")

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

# Activations a ``pylate.models.Dense.Dense`` module may declare. sentence-
# transformers serializes the torch module as a dotted path in the module's
# config.json (newer builds; older ones wrote the bare class name), so both are
# accepted. Anything else is a hard error: applying the projection without its
# declared activation would silently move every vector.
_ACTIVATION_MODULES: dict[str, str] = {
    "torch.nn.modules.linear.Identity": "identity",
    "torch.nn.Identity": "identity",
    "torch.nn.modules.activation.ReLU": "relu",
    "torch.nn.ReLU": "relu",
    "torch.nn.modules.activation.GELU": "gelu",
    "torch.nn.GELU": "gelu",
    "torch.nn.modules.activation.Tanh": "tanh",
    "torch.nn.Tanh": "tanh",
    # Older sentence-transformers builds serialized only the class name.
    "Identity": "identity",
    "ReLU": "relu",
    "GELU": "gelu",
    "Tanh": "tanh",
}

# The kernels behind ``_ACTIVATION_MODULES``. These mirror the torch modules'
# forward passes; ``gelu`` is torch's default (exact, erf-based) form. A torch
# ``nn.GELU(approximate="tanh")`` serializes the same class path and therefore
# cannot be distinguished here.
_ACTIVATIONS = {
    "identity": lambda x: x,
    "relu": lambda x: mx.maximum(x, 0.0),
    "gelu": lambda x: 0.5 * x * (1.0 + mx.erf(x * (1.0 / math.sqrt(2.0)))),
    "tanh": lambda x: mx.tanh(x),
}


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
    """Whether the input is added back after the linear projection."""

    bias: bool
    """Whether ``linear.bias`` exists."""

    weights: Path | None = None
    """``model.safetensors`` for the safetensors source; None when ONNX-sourced."""

    activation: str = "identity"
    """Declared activation: ``identity``, ``relu``, ``gelu`` or ``tanh``."""

    @property
    def residual_is_identity(self) -> bool:
        """Whether pylate adds the input itself instead of a residual MatMul.

        ``pylate.models.Dense.Dense`` only builds its ``residual`` submodule when
        the layer changes width, so a same-width ``use_residual`` module is an
        identity skip and carries no ``residual.weight`` at all.
        """
        return self.use_residual and self.in_features == self.out_features

    @property
    def needs_residual_matrix(self) -> bool:
        """Whether this module must carry a ``residual.weight`` matrix."""
        return self.use_residual and self.in_features != self.out_features


@dataclass(frozen=True)
class ColbertLayout:
    """A pylate ColBERT export that oMLX can load and serve per-token."""

    model_path: Path
    """Root of the sentence-transformers export."""

    weight_source: str
    """``onnx`` or ``safetensors`` -- which artifact supplies the weights."""

    dense: tuple[ColbertDenseSpec, ...]
    """Projection modules, in pipeline order."""

    encoder_weights: Path | None = None
    """``model.safetensors`` holding the encoder weights (safetensors source)."""

    onnx_graph: Path | None = None
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
        if self.encoder_weights is None:
            raise ColbertLayoutError(
                f"ColBERT layout at {self.model_path} has no encoder weight file"
            )
        return self.encoder_weights


def _load_json(path: Path) -> Any | None:
    """Read a JSON file, returning None for anything unreadable."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _is_dense_module(module_type: Any) -> bool:
    """Whether a ``modules.json`` entry is a projection Dense.

    pylate registers ``pylate.models.Dense.Dense``; accept any dotted ``*.Dense``
    or bare ``Dense`` so a renamed pylate package does not silently drop the projection.
    """
    return isinstance(module_type, str) and (
        module_type == "Dense" or module_type.endswith(".Dense")
    )


def _dense_modules(model_path: Path) -> list[dict[str, Any]]:
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
    """Whether ``model_path`` is a pylate / sentence-transformers ColBERT export.

    Detected when projection Dense modules are present in ``modules.json`` and
    any of the following ColBERT indicators match:
    1. ``config_sentence_transformers.json`` declares ``model_type`` as ColBERT
       (or MultiVectorEncoder), or ``similarity_fn_name`` as MaxSim/MeanMaxSim.
    2. Any module in ``modules.json`` is from ``pylate`` (e.g. ``pylate.models.Dense.Dense``).
    3. ``config.json`` declares a ColBERT architecture (e.g. ``HF_ColBERT``) or model_type.
    """
    model_path = Path(model_path)
    dense_modules = _dense_modules(model_path)
    if not dense_modules:
        return False

    raw_modules = _load_json(model_path / _MODULES_FILE)
    if isinstance(raw_modules, list):
        for mod in raw_modules:
            if isinstance(mod, dict) and "pylate" in str(mod.get("type", "")).lower():
                return True

    declared = _load_json(model_path / _SENTENCE_TRANSFORMERS_CONFIG)
    if isinstance(declared, dict):
        model_type = str(declared.get("model_type", "")).lower()
        if model_type in ("colbert", "multivectorencoder"):
            return True
        similarity_fn = str(declared.get("similarity_fn_name", "")).lower()
        if similarity_fn in ("maxsim", "meanmaxsim"):
            return True

    config = _load_json(model_path / "config.json")
    if isinstance(config, dict):
        if str(config.get("model_type", "")).lower() == "colbert":
            return True
        architectures = config.get("architectures", [])
        if isinstance(architectures, list):
            for arch in architectures:
                if "colbert" in str(arch).lower():
                    return True

    return False


def find_onnx_graph(model_path: Path) -> Path | None:
    """The best ONNX graph in ``model_path``, or None."""
    model_path = Path(model_path)
    for name in ONNX_GRAPH_NAMES:
        candidate = model_path / name
        if candidate.is_file():
            return candidate
    return None


def _requested_weight_source(explicit: str | None) -> str | None:
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
    explicit: str | None = None,
) -> str:
    """Choose where a ColBERT export takes its weights from.

    An explicit request (argument or ``OMLX_COLBERT_WEIGHT_SOURCE``) wins. With
    nothing requested, safetensors is preferred when present (since onnx is an
    optional dependency and safetensors is native to MLX). When safetensors is
    not present, fallback to the ONNX graph if available.
    """
    requested = _requested_weight_source(explicit)
    if requested is not None:
        return requested
    model_path = Path(model_path)
    if (model_path / _WEIGHTS_FILE).is_file():
        return WEIGHT_SOURCE_SAFETENSORS
    if find_onnx_graph(model_path) is not None:
        return WEIGHT_SOURCE_ONNX
    return WEIGHT_SOURCE_SAFETENSORS


def _resolve_activation(config: dict[str, Any], directory: Path) -> str:
    """Map a module's declared ``activation_function`` onto a local kernel."""
    declared = config.get("activation_function")
    if declared is None:
        # pylate's Dense defaults to nn.Identity() when the key is absent.
        return "identity"
    if not isinstance(declared, str):
        raise ColbertLayoutError(
            f"{directory.name}/config.json declares a non-string "
            f"activation_function {declared!r}"
        )
    name = _ACTIVATION_MODULES.get(declared)
    if name is None:
        supported = ", ".join(sorted(_ACTIVATION_MODULES))
        raise ColbertLayoutError(
            f"{directory.name}/config.json declares activation_function "
            f"{declared!r}, which this loader cannot reproduce. Supported "
            f"activations: {supported}. Refusing to serve an un-activated "
            "projection."
        )
    return name


def _dense_spec(
    model_path: Path,
    module: dict[str, Any],
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
        activation=_resolve_activation(config, directory),
    )


def resolve_colbert_layout(
    model_path: Path,
    weight_source: str | None = None,
) -> ColbertLayout:
    """Validate a pylate ColBERT export and describe its projection chain.

    Raises:
        ColbertLayoutError: when the directory is a ColBERT export but cannot be
            served -- no projection modules, no usable weight artifact, an
            unsupported activation, or an inconsistent projection chain.
    """
    model_path = Path(model_path)
    modules = _dense_modules(model_path)
    if not modules:
        raise ColbertLayoutError(
            f"{model_path} is detected as a ColBERT export but lists no "
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


def flatten_rope_parameters(config: dict[str, Any]) -> dict[str, Any]:
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
    matrix: mx.array | None,
    key: str,
    spec: ColbertDenseSpec,
) -> mx.array | None:
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


def _identity_residual_conflict(
    spec: ColbertDenseSpec, label: str
) -> ColbertLayoutError:
    """Error for a same-width layer that also ships learned residual weights."""
    return ColbertLayoutError(
        f"{spec.directory.name}/config.json sets use_residual on a "
        f"{spec.in_features} -> {spec.out_features} layer, which pylate implements "
        f"as an identity skip and for which it creates no residual submodule, but "
        f"{label} also carries residual weights. The artifact is inconsistent."
    )


class ColbertProjection:
    """The Dense chain plus the final per-token L2.

    Mirrors the pinned ONNX export's ``/projection_layers.*`` and ``/ReduceL2``
    subgraph. Weights are plain ``mx.array`` attributes rather than an
    ``nn.Module``: the chain is a few fixed matrices with no other state, and the
    encoder it accompanies owns dtype policy. MLX promotes the activations to the
    wider of the two dtypes, so a reduced-precision checkpoint cannot silently
    truncate the projection.
    """

    def __init__(self, dense: Sequence[ColbertDenseSpec]):
        if not dense:
            raise ValueError("a ColBERT projection requires at least one Dense module")
        for spec in dense:
            if spec.activation not in _ACTIVATIONS:
                raise ValueError(
                    f"unknown ColBERT projection activation {spec.activation!r} for "
                    f"{spec.directory}"
                )
        self.dense: tuple[ColbertDenseSpec, ...] = tuple(dense)
        self._linear: list[mx.array] = []
        self._linear_bias: list[mx.array | None] = []
        # None is pylate's identity skip for a same-width use_residual layer.
        self._residual: list[mx.array | None] = []
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
            if spec.needs_residual_matrix and residual is None:
                raise ColbertLayoutError(
                    f"{spec.directory.name}/config.json sets use_residual on a "
                    f"{spec.in_features} -> {spec.out_features} layer, which pylate "
                    f"implements with residual.weight, but {spec.weights} does not "
                    f"carry it (keys: {sorted(weights.keys())})"
                )
            if spec.residual_is_identity and residual is not None:
                raise _identity_residual_conflict(spec, str(spec.weights))
            projection._linear.append(linear)
            projection._linear_bias.append(
                weights.get("linear.bias") if spec.bias else None
            )
            projection._residual.append(residual)
        return projection._finalize()

    @classmethod
    def from_matrices(
        cls,
        dense: Sequence[ColbertDenseSpec],
        linear: Sequence[mx.array | None],
        residual: Sequence[mx.array | None],
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
            residual_matrix = (
                _checked_matrix(
                    residual[index], f"projection_layers.{index}.residual", spec
                )
                if spec.use_residual
                else None
            )
            if spec.needs_residual_matrix and residual_matrix is None:
                raise ColbertLayoutError(
                    f"projection module {index} ({spec.directory.name}) sets "
                    "use_residual but has no residual weight matrix"
                )
            if spec.residual_is_identity and residual_matrix is not None:
                raise _identity_residual_conflict(spec, f"projection module {index}")
            projection._linear.append(matrix)
            projection._linear_bias.append(None)
            projection._residual.append(residual_matrix)
        return projection._finalize()

    def _finalize(self) -> "ColbertProjection":
        """Evaluate the loaded matrices once, at load time."""
        mx.eval(
            [
                array
                for array in (self._linear + self._residual + self._linear_bias)
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
            # pylate applies the activation to the linear output, before the
            # residual is added (pylate.models.Dense.Dense.forward).
            y = _ACTIVATIONS[spec.activation](y)
            if spec.use_residual:
                residual = self._residual[index]
                # None is the identity skip of a same-width layer.
                y = y + (x if residual is None else x @ mx.transpose(residual))
            x = y
        return l2_normalize_tokens(x)

    def __repr__(self) -> str:
        return (
            f"<ColbertProjection layers={self.layer_count} "
            f"output_dim={self.output_dim} loaded={self._loaded}>"
        )
