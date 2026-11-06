# SPDX-License-Identifier: Apache-2.0
"""Projected per-token (ColBERT / late-interaction) support for pylate exports.

``MLXEmbeddingModel.embed_token_ids`` returns the encoder's ``last_hidden_state``
verbatim. That is the right contract for a plain embedder -- Qwen3-Embedding is
served that way -- but it is the wrong contract for a pylate ColBERT model, whose
output is a *projected*, per-token L2-normalized vector.

The projection is not part of the transformer checkpoint. sentence-transformers
describes it in ``modules.json`` as a chain of ``pylate.models.Dense.Dense``
modules, each with its own subdirectory (``1_Dense``, ``2_Dense``, ...) holding a
``config.json`` and a ``model.safetensors``. Each module is a ``torch.nn.Linear``
plus an optional residual, followed by the module's declared activation function.
Reproducing ``pylate.models.Dense.Dense.forward`` exactly gives::

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
  convention: it is ``torch.nn.functional.normalize``, whose clamp ``_L2_EPS``
  matches.

Weight layout
-------------
A pylate export ships its weights as safetensors: the transformer weights in the
export's top-level ``model.safetensors`` (mapped onto mlx-embeddings'
``ModernBertModel`` parameter names), plus each projection module's own
``model.safetensors``. The loader validates every Dense matrix shape, so a
transposed export fails loudly instead of quietly producing noise.

Orientation: ``pylate.models.Dense.Dense`` wraps ``torch.nn.Linear``, so its
safetensors weight is ``[out_features, in_features]`` and the operation is
``x @ W.T``.
"""

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# `model_type` written by pylate/sentence-transformers for a ColBERT checkpoint.
COLBERT_MODEL_TYPE = "ColBERT"

_SENTENCE_TRANSFORMERS_CONFIG = "config_sentence_transformers.json"
_MODULES_FILE = "modules.json"
_WEIGHTS_FILE = "model.safetensors"

# The clamp the per-token L2 uses (torch.nn.functional.normalize's default).
# Anything at or below this is a degenerate row (a zero vector); real token
# vectors are many orders of magnitude above it.
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
    """Directory holding the module's ``config.json`` and ``model.safetensors``."""

    in_features: int
    """Width consumed by this module."""

    out_features: int
    """Width produced by this module."""

    use_residual: bool
    """Whether the input is added back after the linear projection."""

    bias: bool
    """Whether ``linear.bias`` exists."""

    weights: Path
    """The module's ``model.safetensors``."""

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

    dense: tuple[ColbertDenseSpec, ...]
    """Projection modules, in pipeline order."""

    encoder_weights: Path
    """The export's top-level ``model.safetensors`` (encoder weights)."""

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
    if not weights.is_file():
        raise ColbertLayoutError(
            f"ColBERT projection module {directory} has no {_WEIGHTS_FILE}. A "
            "pylate ColBERT export must ship model.safetensors for the encoder "
            "and for every Dense module."
        )
    return ColbertDenseSpec(
        directory=directory,
        in_features=in_features,
        out_features=out_features,
        use_residual=bool(config.get("use_residual")),
        bias=bool(config.get("bias")),
        weights=weights,
        activation=_resolve_activation(config, directory),
    )


def resolve_colbert_layout(model_path: Path) -> ColbertLayout:
    """Validate a pylate ColBERT export and describe its projection chain.

    Raises:
        ColbertLayoutError: when the directory is a ColBERT export but cannot be
            served -- no projection modules, missing ``model.safetensors``
            weights, an unsupported activation, or an inconsistent projection
            chain.
    """
    model_path = Path(model_path)
    modules = _dense_modules(model_path)
    if not modules:
        raise ColbertLayoutError(
            f"{model_path} is detected as a ColBERT export but lists no "
            f"Dense projection module in {_MODULES_FILE}"
        )

    dense = tuple(_dense_spec(model_path, module) for module in modules)
    for previous, current in zip(dense, dense[1:]):
        if previous.out_features != current.in_features:
            raise ColbertLayoutError(
                f"ColBERT projection chain is inconsistent: {previous.directory.name} "
                f"outputs {previous.out_features} but {current.directory.name} "
                f"consumes {current.in_features}"
            )

    encoder_weights = model_path / _WEIGHTS_FILE
    if not encoder_weights.is_file():
        raise ColbertLayoutError(
            f"ColBERT export at {model_path} has no {_WEIGHTS_FILE} for its "
            "encoder. A pylate ColBERT export must ship model.safetensors for "
            "the encoder and for every Dense module."
        )

    layout = ColbertLayout(
        model_path=model_path,
        dense=dense,
        encoder_weights=encoder_weights,
    )
    logger.debug(
        "ColBERT layout at %s: %d Dense modules, output dim %d, weights %s",
        model_path,
        len(dense),
        layout.output_dim,
        encoder_weights.name,
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
    """Per-token L2 normalization, as pylate / ``torch.nn.functional.normalize``.

    A degenerate all-zero row divides by ``eps`` and stays zero instead of
    producing NaN.
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

    Weights are plain ``mx.array`` attributes rather than an ``nn.Module``: the
    chain is a few fixed matrices with no other state, and the encoder it
    accompanies owns dtype policy. MLX promotes the activations to the wider of
    the two dtypes, so a reduced-precision checkpoint cannot silently truncate
    the projection.
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
        """Load every Dense module of the layout."""
        projection = cls(layout.dense)
        for spec in layout.dense:
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
