# SPDX-License-Identifier: Apache-2.0
"""SenseNova-Loop model module for oMLX (self-contained Loop/MHC architecture).

This module gives the official ``sensenova_loop`` model_type a real
``mlx_lm.models.sensenova_loop`` identity so ``_get_classes`` can import it
directly -- no config rewriting, no ``MODEL_REMAPPING`` detour.

Unlike the earlier thin-shell (which re-exported the *same* qwen3_5 class
objects and therefore made every qwen3_5 user share the Loop/MHC monkey
patches), this version defines **subclasses** of the qwen3_5 classes and
**inlines** the SenseNova Loop / MHC architecture directly into those
subclass method bodies (``DecoderLayer.__call__``, ``Qwen3_5TextModel.__call__``,
``TextModel.make_cache``, ``Model.sanitize``, ``ModelArgs.from_dict``,
``Model.__init__``).  No monkey-patching of ``cls.__call__ = ...`` etc. is
needed -- the architecture is part of the class definition itself.  This
matches the official ``mlx_lm.models.nanbeige`` / ``iquestloopcoder`` pattern,
where each Loop model is a self-contained model class with the loop/MHC logic
built in.

The subclass chain is deliberately threaded so a ``Model`` instance builds a
``TextModel`` which builds a ``Qwen3_5TextModel`` (all subclasses), and the
class-level ``__init__`` bodies mirror qwen3_5.py but reference the local
subclasses.  The MTP-specific slot layout / rollback live in the separate
``sensenova_loop_mtp`` module (independent of this architecture module).

Registration: ``apply()`` calls ``_register_module()``, which installs this
module into ``sys.modules["mlx_lm.models.sensenova_loop"]`` and as
``mlx_lm.models.sensenova_loop`` (same pattern as ``bailing_hybrid`` /
``deepseek_v41_mlx_lm``).  This registration is independent of MTP enablement
-- the Loop/MHC architecture loads whenever a ``sensenova_loop`` model is
loaded, with or without an MTP head.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import math
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# Use absolute imports (not ``from . import ...``) so this module works both
# when imported as ``omlx.patches.mlx_lm_mtp.sensenova_loop`` (the adapter
# ``apply()`` imports ``_register_module`` from here) and when loaded by
# ``_register_module`` as ``mlx_lm.models.sensenova_loop`` (its ``__package__``
# is ``mlx_lm.models``, so a relative ``.qwen3_5`` would resolve against the
# wrong package in the first case).
from mlx_lm.models import qwen3_5 as _q35
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.qwen3_5 import (
    create_attention_mask,
    create_ssm_mask,
    TextModelArgs,
)

logger = logging.getLogger(__name__)

_MODULE_NAME = "mlx_lm.models.sensenova_loop"
_SHELL_MARKER = "__is_sensenova_loop_shell"
_REGISTERED = False


# ---------------------------------------------------------------------------
# Loop execution plan (SenseNova Loop shared-weight architecture)
# ---------------------------------------------------------------------------

def _execution_plan(loop_settings: Optional[dict], num_layers: int) -> Optional[Tuple[int, ...]]:
    """Physical-layer execution order (ported from modeling_sensenova_loop.execution_plan).

    Returns None without a loop; otherwise::

        tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, num_layers))

    where ``[start, end)`` is the loop interval and ``count`` is the number of
    visits per loop slice.
    """
    if not loop_settings:
        return None
    start = loop_settings.get("start")
    end = loop_settings.get("end")
    count = loop_settings.get("eval_count")
    if loop_settings.get("counts") != [count]:
        raise ValueError("Only fixed-count Loop checkpoints are supported")
    if not (0 <= start < end <= num_layers and count >= 1):
        raise ValueError("Invalid Loop interval or eval_count")
    return tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, num_layers))


def install_loop(text_model: Any, loop_settings: Optional[dict]) -> None:
    """Install the loop execution plan on a model (sets ``_loop_plan``).

    ``text_model`` is an MLX TextModel whose ``.model`` is the inner
    ``Qwen3_5TextModel`` (containing ``.layers``).
    """
    if not loop_settings:
        return
    tm = text_model.model
    plan = _execution_plan(loop_settings, len(tm.layers))
    if plan is None:
        return
    tm._loop_plan = plan
    # Reset and point at the first slot of each type in the execution plan.
    tm.ssm_idx = None
    tm.fa_idx = None
    for slot, phys in enumerate(plan):
        layer = tm.layers[phys]
        if tm.ssm_idx is None and layer.is_linear:
            tm.ssm_idx = slot
        if tm.fa_idx is None and not layer.is_linear:
            tm.fa_idx = slot
        if tm.ssm_idx is not None and tm.fa_idx is not None:
            break


def is_loop_model(model: Any) -> bool:
    """Whether an inner model (with ``.layers``) is a SenseNova Loop architecture."""
    return getattr(model, "_loop_plan", None) is not None


# ---------------------------------------------------------------------------
# MHC (multi-head cross/cycle) -- ported from the mx version of
# modeling_sensenova_loop.py.
#
# Between the ``[start, end)`` physical layers it carries multi-stream routing
# across ``streams`` residual streams (pre gate + post scale + sinkhorn mixing).
# These are architecture components; they are consumed by ``install_mhc`` /
# ``DecoderLayer.__call__`` and have zero coupling with the generic Qwen3.5
# code.
# ---------------------------------------------------------------------------

def _mhc_sinkhorn(logits, iterations):
    """Normalize routing logits across rows and columns in log space (mx port)."""
    logits = logits.astype(mx.float32)
    for _ in range(iterations):
        logits = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits - mx.logsumexp(logits, axis=-2, keepdims=True)
    return mx.exp(logits)


def _mhc_enter(hidden, physical, settings):
    """Expand to residual streams at the first mHC layer, or restore their shape."""
    if physical == settings["start"]:
        return mx.contiguous(
            mx.broadcast_to(
                hidden[..., None, :],
                (*hidden.shape[:-1], settings["streams"], hidden.shape[-1]),
            )
        )
    return hidden.reshape(*hidden.shape[:-1], settings["streams"], -1)


def _mhc_leave(residual, physical, settings):
    """Reduce streams at the last mHC layer; otherwise carry them forward."""
    if physical == settings["end"] - 1:
        return mx.mean(residual, axis=-2)
    return mx.reshape(residual, (*residual.shape[:-2], -1))


class MHCRouter(nn.Module):
    """Multi-stream residual router: pre gate + post scale + sinkhorn mixing.

    Ports the mx semantics of modeling_sensenova_loop.MHCRouter: weights and
    biases are stored in FP32 (the model may be cast to BF16 overall, but the
    routing parameters must stay FP32).

    Must subclass ``nn.Module``: mlx's ``load_weights`` walks the module tree
    for parameters; as a plain class, ``mhc_attn.weight/bias`` would not be
    registered as submodule parameters and ``load_weights`` would raise
    ``Received N parameters not in model``.
    """

    def __init__(self, dim, settings):
        super().__init__()
        self.streams = settings["streams"]
        self.iterations = settings["iterations"]
        self.fixed_identity = settings["fixed_identity"]
        self.pre_scale = settings.get("pre_scale", 1.0)
        streams = self.streams
        output_size = 2 * streams + (0 if self.fixed_identity else streams * streams)
        # Weights/biases are stored in FP32.
        self.weight = mx.zeros((output_size, streams * dim), dtype=mx.float32)
        self.bias = mx.zeros((output_size,), dtype=mx.float32)
        # Bias init (mirrors the torch logic).
        init_probs = {2: (0.25, 0.75), 4: (0.0625, 0.1875, 0.25, 0.5)}[streams]
        b = self.bias
        for i, p in enumerate(init_probs):
            b[i] = float(mx.log(mx.array(p) / (1 - mx.array(p))))
        if not self.fixed_identity:
            diag = math.log((1 - (streams - 1) / 16) * 16)
            b[2 * streams:] = (mx.eye(streams) * diag).reshape(-1)
        self.bias = b

    def __call__(self, residual):
        return self._row_call(residual)

    def _row_call(self, residual):
        flat = mx.reshape(
            residual.astype(mx.float32), (*residual.shape[:-2], -1)
        )
        normalized = flat * mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + 1e-6)
        logits = normalized @ self.weight.T + self.bias
        streams = self.streams
        pre = mx.sigmoid(logits[..., :streams])
        post = 2 * mx.sigmoid(logits[..., streams:2 * streams])
        if self.fixed_identity:
            mix = mx.broadcast_to(
                mx.eye(streams, dtype=mx.float32),
                (*logits.shape[:-1], streams, streams),
            )
        else:
            mix = _mhc_sinkhorn(
                logits[..., 2 * streams:].reshape(*logits.shape[:-1], streams, streams),
                self.iterations,
            )
        # Matches the original: the branch stays in residual.dtype (BF16);
        # only logits/weight/bias math runs in FP32.
        branch = mx.sum(pre.astype(residual.dtype)[..., None] * residual, axis=-2)
        if self.pre_scale != 1.0:
            branch = branch * self.pre_scale
        return branch, post, mix

    @staticmethod
    def merge(residual, branch, post, mix):
        return MHCRouter._merge_rows(residual, branch, post, mix)

    @staticmethod
    def _merge_rows(residual, branch, post, mix):
        # Matches the original: the result stays in residual.dtype (BF16);
        # mix/post are computed first, then cast back.
        mixed = mx.matmul(mix.astype(residual.dtype), residual)
        update = post.astype(branch.dtype)[..., None] * branch[..., None, :]
        return mixed + update


def install_mhc(model: Any, mhc_settings: Optional[dict]) -> None:
    """Install MHC modules on the model (inject instance attributes only).

    Unlike the earlier monkey-patch version, ``DecoderLayer.__call__`` already
    carries the MHC-aware branch (it reads ``self._mhc_settings``); this
    function only injects the per-layer routing modules and settings.  No
    class-level patching is performed.
    """
    if not mhc_settings:
        return
    start, end = mhc_settings["start"], mhc_settings["end"]
    for i in range(start, min(end, len(model.layers))):
        layer = model.layers[i]
        layer._mhc_physical = i
        layer._mhc_settings = dict(mhc_settings)
        layer.mhc_attn = MHCRouter(model.args.hidden_size, mhc_settings)
        layer.mhc_ffn = MHCRouter(model.args.hidden_size, mhc_settings)


def configure_model(model: Any, args: Any) -> None:
    """Install Loop + MHC uniformly from ``ModelArgs`` (orchestration entry).

    Called at the end of ``Model.__init__``: reads ``args.mhc`` /
    ``args.loop_settings`` (attached as instance attributes by
    ``ModelArgs.from_dict``) and calls :func:`install_mhc` /
    :func:`install_loop` respectively.

    Non-SenseNova models (both fields None) are a no-op, matching the generic
    Qwen3.5 behavior.
    """
    install_mhc(model, getattr(args, "mhc", None))
    install_loop(model, getattr(args, "loop_settings", None))


# ---------------------------------------------------------------------------
# Subclass chain (threaded so the Loop/MHC architecture is part of these
# class definitions, not monkey-patched onto the shared qwen3_5 originals).
# ---------------------------------------------------------------------------

class ModelArgs(_q35.ModelArgs):
    """Loop-aware ModelArgs (mhc / loop_settings attach via from_dict)."""

    @classmethod
    def from_dict(cls, params: dict):
        instance = super().from_dict(params)
        # Read mhc/loop_settings from top-level keys (if present) and attach
        # as instance attributes. BaseModelArgs.from_dict ignores unknown
        # keys, so they must be read explicitly.
        if isinstance(params, dict):
            instance.mhc = params.get("mhc")
            instance.loop_settings = params.get("loop_settings")
        else:
            instance.mhc = getattr(instance, "mhc", None)
            instance.loop_settings = getattr(instance, "loop_settings", None)
        return instance


class DecoderLayer(_q35.DecoderLayer):
    """Loop/MHC-aware decoder layer.

    ``__call__`` carries the MHC-aware branch: when ``_mhc_settings`` is
    present (injected by ``install_mhc``), it routes the hidden state through
    the multi-stream residual router instead of the plain qwen3_5 path.
    """

    def __call__(self, x, mask=None, cache=None, n_confirmed=0):
        settings = getattr(self, "_mhc_settings", None)
        if settings is None:
            return super().__call__(x, mask, cache, n_confirmed=n_confirmed)
        physical = self._mhc_physical
        residual = _mhc_enter(x, physical, settings)
        branch, post, mix = self.mhc_attn(residual)
        branch = self.input_layernorm(
            mx.contiguous(branch.astype(self.input_layernorm.weight.dtype))
        )
        if self.is_linear:
            if n_confirmed:
                r = self.linear_attn(branch, mask, cache, n_confirmed=n_confirmed)
            else:
                r = self.linear_attn(branch, mask, cache)
        else:
            r = self.self_attn(branch, mask, cache)
        residual = MHCRouter.merge(residual, r, post, mix)
        branch, post, mix = self.mhc_ffn(residual)
        branch = self.mlp(
            self.post_attention_layernorm(
                mx.contiguous(branch.astype(self.post_attention_layernorm.weight.dtype))
            )
        )
        return _mhc_leave(MHCRouter.merge(residual, branch, post, mix), physical, settings)


class Qwen3_5TextModel(_q35.Qwen3_5TextModel):
    """Loop-aware inner text model (loop forward in ``__call__``).

    The ``__init__`` mirrors ``qwen3_5.Qwen3_5TextModel.__init__`` but builds
    the local ``DecoderLayer`` subclass so the MHC-aware ``__call__`` is
    reached at runtime.
    """

    def __init__(self, args: TextModelArgs):
        # PipelineMixin.__init__ sets pipeline_rank/pipeline_size/start_idx/
        # end_idx (qwen3_5.Qwen3_5TextModel does the same via super().__init__).
        # Using nn.Module.__init__ directly would leave start_idx unset and
        # break ``pipeline_layers`` when ``install_mhc`` reads ``model.layers``.
        _q35.PipelineMixin.__init__(self)
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        n_confirmed: int = 0,
        capture_layer_ids=None,
        capture_sink=None,
    ):
        loop_plan = getattr(self, "_loop_plan", None)
        if loop_plan is None:
            return super().__call__(
                inputs,
                cache,
                input_embeddings,
                n_confirmed,
                capture_layer_ids,
                capture_sink,
            )

        # Loop mode: reuse physical layers per the execution plan, with an
        # independent cache slot per visit.
        if getattr(self, "pipeline_size", 1) > 1:
            raise ValueError("Loop execution does not support model parallelism")
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(loop_plan)

        fa_mask = None
        ssm_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        wanted = set(capture_layer_ids or ())
        captured = {}
        for slot, phys in enumerate(loop_plan):
            layer = self.layers[phys]
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(
                hidden_states, mask=mask, cache=cache[slot], n_confirmed=n_confirmed
            )
            if slot in wanted:
                captured[slot] = hidden_states
        if capture_sink is not None:
            capture_sink.extend(captured[i] for i in capture_layer_ids or ())

        # Consistent with qwen35_model's Qwen3_5TextModel.__call__ convention:
        # return pre-norm hidden (the outer TextModel.__call__ applies norm to
        # produce logits).
        return hidden_states


class TextModel(_q35.TextModel):
    """Loop-aware TextModel (loop-aware ``make_cache``).

    The ``__init__`` mirrors ``qwen3_5.TextModel.__init__`` but builds the
    local ``Qwen3_5TextModel`` subclass so the loop/MHC forward is reached at
    runtime.
    """

    def __init__(self, args: TextModelArgs):
        # Call the qwen3_5.TextModel.__init__ chain (qwen35_model patches it to
        # attach the Lightning MTP head when mtp is active).  That builds a
        # qwen3_5.Qwen3_5TextModel internally; we then rebuild ``self.model``
        # as the local subclass so the loop/MHC forward (in the subclass) is
        # reached.  The MTP head lives on ``self`` (``self.mtp``), not on
        # ``self.model``, so it is unaffected by the rebuild.  Weights are not
        # bound yet at __init__ time (mlx_lm binds them afterwards via
        # model.update), so discarding the temporary qwen3_5 inner model is
        # safe.
        super().__init__(args)
        self.model = Qwen3_5TextModel(args)

    def make_cache(self):
        loop_plan = getattr(self.model, "_loop_plan", None)
        if loop_plan is not None:
            from mlx_lm.models.cache import ArraysCache, KVCache

            return [
                ArraysCache(size=2)
                if self.model.layers[phys].is_linear
                else KVCache()
                for phys in loop_plan
            ]
        return super().make_cache()


class Model(_q35.Model):
    """Loop-aware Model (configure_model in ``__init__`` + ``sanitize`` fix).

    The ``__init__`` mirrors ``qwen3_5.Model.__init__`` but builds the local
    ``TextModel`` subclass and installs Loop + MHC at the end.
    """

    def __init__(self, args: ModelArgs):
        nn.Module.__init__(self)
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))
        # Install Loop + MHC uniformly (no-op for non-SenseNova).
        configure_model(self.language_model, args)

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            # SenseNova lm_head: language_model.lm_head (no .model.).
            if key.startswith("model.language_model.lm_head"):
                key = "language_model.lm_head" + key[len("model.language_model.lm_head"):]
            # loop_gate: not consumed by the adapter, strip.
            elif "loop_gate" in key:
                continue
            # Everything else goes to the original sanitize logic (the
            # model.language_model prefix is preserved by the original).
            sanitized[key] = value
        return super().sanitize(sanitized)


def _classes_from_qwen35() -> tuple[Any, Any, Any, Any]:
    """Return the local subclass objects.

    Returns ``(Model, ModelArgs, TextModel, TextModelArgs)``.
    """
    return Model, ModelArgs, TextModel, TextModelArgs


def _register_module() -> None:
    """Install ``mlx_lm.models.sensenova_loop`` as a subclass-based module.

    Idempotent: re-runs after a prior registration are no-ops unless the
    module was removed from ``sys.modules`` by a reload.
    """
    global _REGISTERED
    existing = sys.modules.get(_MODULE_NAME)
    if existing is not None and getattr(existing, _SHELL_MARKER, False):
        _REGISTERED = True
        return

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(Path(__file__)))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {_MODULE_NAME} from {__file__}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    setattr(module, _SHELL_MARKER, True)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        models_pkg = importlib.import_module("mlx_lm.models")
        models_pkg.sensenova_loop = module
    except BaseException:
        if sys.modules.get(_MODULE_NAME) is module:
            if existing is None:
                sys.modules.pop(_MODULE_NAME)
            else:
                sys.modules[_MODULE_NAME] = existing
        raise

    _REGISTERED = True
    logger.info("Registered %s (self-contained Loop/MHC class over qwen3_5)", _MODULE_NAME)


def is_registered() -> bool:
    """Whether the sensenova_loop module is currently registered."""
    return _REGISTERED


def apply() -> bool:
    """Register the ``mlx_lm.models.sensenova_loop`` module (idempotent).

    This registers the self-contained Loop/MHC architecture.  It is
    **independent of MTP enablement** -- the architecture loads whenever a
    ``sensenova_loop`` model is loaded.  The MTP-specific slot layout /
    rollback live in ``sensenova_loop_mtp`` and are applied separately via
    ``sensenova_loop_mtp.apply_mtp()``.
    """
    try:
        _register_module()
    except Exception as e:  # pragma: no cover
        logger.debug("sensenova_loop module registration skipped: %s", e)
        return False
    return True


# Re-export the module-level helpers the loop forward needs (see
# Qwen3_5TextModel.__call__, which uses the qwen3_5 module functions).  These
# are module functions in qwen3_5.py; re-exporting keeps the patch target
# module self-consistent.
__all__ = [
    "Model",
    "ModelArgs",
    "TextModel",
    "TextModelArgs",
    "Qwen3_5TextModel",
    "DecoderLayer",
    "create_attention_mask",
    "create_ssm_mask",
    "install_loop",
    "install_mhc",
    "configure_model",
    "is_loop_model",
]
