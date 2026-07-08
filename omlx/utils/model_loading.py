# SPDX-License-Identifier: Apache-2.0
"""Model loading helpers with post-load transforms."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)

_VLM_TEXT_PREFIX = "language_model."
# HF/checkpoint order vs runtime module-tree order for the VLM text stack.
# ``sanitize`` swaps the former to the latter; class_predicate matches the latter.
_CKPT_TEXT_PREFIX = "model.language_model."
_RUNTIME_TEXT_PREFIX = "language_model.model."

_MLX_LM_LOAD_CONFIG_PATCHED = False

# mlx_lm.load dropped trust_remote_code in some releases. Check once at
# import time so call sites can pass it safely across versions.
def _mlx_lm_load_accepts_trust_remote_code() -> bool:
    try:
        import inspect
        from mlx_lm import load as _lm_load
        return "trust_remote_code" in inspect.signature(_lm_load).parameters
    except Exception:
        return False

_LM_LOAD_ACCEPTS_TRC = _mlx_lm_load_accepts_trust_remote_code()


def lm_load_compat(path_or_repo: str, *, trust_remote_code: bool = False, **kwargs):
    """Wrapper around mlx_lm.load that forwards trust_remote_code only when supported."""
    from mlx_lm import load
    if _LM_LOAD_ACCEPTS_TRC:
        kwargs["trust_remote_code"] = trust_remote_code
    return load(path_or_repo, **kwargs)


def expand_per_layer_quant_keys(cfg: dict) -> dict:
    """Add module-tree-path variants of per-layer quantization keys.

    mlx-lm's ``nn.quantize`` class_predicate matches the runtime module-tree
    path directly (``if p in config["quantization"]``), but oQ / HF
    checkpoints key per-layer overrides by other conventions:

    - bare safetensors tensor base name (``"lm_head"``), which the VLM text
      tree nests under ``language_model.`` (``"language_model.lm_head"``).
    - HF checkpoint order ``model.language_model.layers.N.*``, which
      ``sanitize`` swaps to module-tree order
      ``language_model.model.layers.N.*``.

    Without the matching variant the lookup misses, the global bits are used,
    and the layer is built at the wrong bit-width.

    Mutates *cfg* in place and returns it for convenience.
    """
    for config_key in ("quantization", "quantization_config"):
        quant = cfg.get(config_key)
        if not isinstance(quant, dict):
            continue
        extras: dict[str, dict] = {}
        for key, val in quant.items():
            if not isinstance(val, dict):
                continue
            if key.startswith(_CKPT_TEXT_PREFIX):
                # model.language_model.X -> language_model.model.X
                variant = _RUNTIME_TEXT_PREFIX + key[len(_CKPT_TEXT_PREFIX) :]
            elif key.startswith(_VLM_TEXT_PREFIX):
                # language_model.X -> X
                variant = key[len(_VLM_TEXT_PREFIX) :]
            else:
                # X -> language_model.X
                variant = _VLM_TEXT_PREFIX + key
            if variant not in quant and variant not in extras:
                extras[variant] = val
        if extras:
            quant.update(extras)
    return cfg


def expand_glm_moe_dsa_fused_quant_keys(cfg: dict) -> dict:
    """Add quantization specs for GLM DSA fused MoE gate/up layers.

    The oMLX GLM DSA patch fuses ``switch_mlp.gate_proj`` and
    ``switch_mlp.up_proj`` into ``switch_mlp.gate_up_proj``.  mlx-lm's loader
    chooses a module's quantizer from ``config["quantization"][path]`` before
    falling back to the global quantization settings.  GLM-5.1-MXFP4-Q8 ships
    per-layer MXFP4 specs for the split gate/up modules, but no fused path
    entry, so the fallback incorrectly quantizes ``gate_up_proj`` as affine and
    strict loading asks for missing ``gate_up_proj.biases`` tensors.

    Mutates *cfg* in place and returns it for convenience.
    """
    if cfg.get("model_type") != "glm_moe_dsa":
        return cfg

    for config_key in ("quantization", "quantization_config"):
        quant = cfg.get(config_key)
        if not isinstance(quant, dict):
            continue

        extras: dict[str, dict] = {}
        for gate_path, gate_spec in list(quant.items()):
            if not gate_path.endswith(".mlp.switch_mlp.gate_proj"):
                continue
            if not isinstance(gate_spec, dict):
                continue

            base_path = gate_path[: -len(".gate_proj")]
            up_path = f"{base_path}.up_proj"
            fused_path = f"{base_path}.gate_up_proj"
            if fused_path in quant:
                continue

            up_spec = quant.get(up_path)
            if isinstance(up_spec, dict) and up_spec == gate_spec:
                extras[fused_path] = dict(gate_spec)

        if extras:
            quant.update(extras)

    return cfg


def _patch_mlx_lm_load_config() -> None:
    """Wrap ``mlx_lm.utils.load_config`` to expand per-layer quant keys."""
    global _MLX_LM_LOAD_CONFIG_PATCHED
    if _MLX_LM_LOAD_CONFIG_PATCHED:
        return

    try:
        import mlx_lm.utils as _lu
    except ImportError:
        return

    _original = _lu.load_config

    def _patched(model_path, *args, **kwargs):
        cfg = _original(model_path, *args, **kwargs)
        expand_per_layer_quant_keys(cfg)
        expand_glm_moe_dsa_fused_quant_keys(cfg)
        return cfg

    _lu.load_config = _patched
    _MLX_LM_LOAD_CONFIG_PATCHED = True


def maybe_apply_pre_load_patches(
    model_name: str,
    model_settings: Any | None = None,
    for_vlm: bool = False,
) -> None:
    """Apply patches that need to run *before* mlx_lm.load() runs.

    Dispatches:

    - DeepSeek V4 patch (PR 1192) when ``config.json`` declares a
      ``deepseek_v4*`` model_type.
    - Step 3.7 Flash text-only wrapper (PR 1325) when ``config.json``
      declares ``model_type == "step3p7"``.
    - Llama 4 attention offset patch when ``config.json`` declares
      ``model_type == "llama4"`` directly or under ``text_config``.
    - GLM-5.2 ``glm_moe_dsa`` patch (mlx-lm PR 1410) when ``config.json``
      declares ``model_type == "glm_moe_dsa"``. Required because pinned
      mlx-lm exposes it as a bare DeepSeek-V3.2 subclass and cannot load
      checkpoints whose shared DSA layers carry no indexer weights.
    - Native MTP patch (PR 990 + PR 15) when the config declares MTP heads
      on a supported model_type. Always applied for sanitize correctness;
      head attachment is gated by ``model_settings.mtp_enabled``.
    - mlx-vlm side MTP runtime + nested-visual patches when ``for_vlm`` is
      True. Required so persisted ``mtp.*`` weights can bind to the
      LanguageModel tree even when ``mtp_enabled`` is False (otherwise
      strict load fails on a Qwen3.6 *-mtp VLM and the engine falls back
      to LLM, losing vision). VLMBatchedEngine passes ``for_vlm=True``;
      BatchedEngine / DFlashEngine / LLM loaders keep the default.
    - mlx-vlm MoE VLM sanitize patch when ``for_vlm`` is True and the
      checkpoint is a Qwen3.6 MoE VLM without declared MTP heads.
      Pre-converted mlx-lm exports ship ``switch_mlp`` weights; stock
      mlx-vlm ``sanitize`` unconditionally pops ``experts.gate_up_proj``
      and crashes with KeyError unless the mlx_vlm_mtp sanitize replacement
      is installed first. ``for_vlm=True`` is only passed by
      ``VLMBatchedEngine``, so no separate ``vision_config`` gate is needed.
    Both patches inject modules into ``sys.modules`` and replace mlx-lm
    internals; gating keeps non-affected models at zero cost.

    Safe to call repeatedly; the patches are idempotent.
    """
    # Reset the process-wide MTP flag so non-MTP-compatible models (or
    # models with mtp_enabled=False) are not polluted by a prior model
    # load that left the flag True.
    from ..patches.mlx_lm_mtp import set_mtp_active

    set_mtp_active(False)

    _patch_mlx_lm_load_config()

    config_path = Path(model_name) / "config.json"
    if not config_path.exists():
        return
    try:
        config = json.loads(config_path.read_text())
    except Exception as e:
        logger.debug(
            "Could not read %s for pre-load patch dispatch: %s", config_path, e
        )
        return

    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type.startswith("deepseek_v4"):
        from ..patches.deepseek_v4 import apply_deepseek_v4_patch

        if apply_deepseek_v4_patch():
            logger.info("DeepSeek V4 pre-load patch applied for %s", model_name)

    if model_type == "step3p7":
        from ..patches.step3p7 import apply_step3p7_patch

        if apply_step3p7_patch():
            logger.info("Step 3.7 pre-load patch applied for %s", model_name)

    text_config = config.get("text_config")
    text_model_type = (
        text_config.get("model_type") if isinstance(text_config, dict) else None
    )
    if model_type == "llama4" or text_model_type == "llama4":
        from ..patches.llama4_attention import apply_llama4_attention_patch

        if apply_llama4_attention_patch():
            logger.info("Llama 4 attention patch applied for %s", model_name)

    if model_type == "glm_moe_dsa":
        from ..patches.glm_moe_dsa import apply_glm_moe_dsa_patch

        if apply_glm_moe_dsa_patch():
            logger.info("GLM MoE DSA pre-load patch applied for %s", model_name)

    minimax_m3_types = {"minimax_m3", "minimax_m3_vl"}
    if for_vlm and (
        model_type in minimax_m3_types or text_model_type in minimax_m3_types
    ):
        from ..patches.mlx_vlm_minimax_m3_compat import (
            apply_mlx_vlm_minimax_m3_compat_patch,
        )

        if apply_mlx_vlm_minimax_m3_compat_patch():
            logger.info(
                "MiniMax M3 mlx-vlm compatibility patch applied for %s",
                model_name,
            )

        from ..patches.minimax_m3_sparse_attention import (
            apply_minimax_m3_sparse_attention_patch,
        )

        if apply_minimax_m3_sparse_attention_patch():
            logger.info(
                "MiniMax M3 sparse attention patch applied for %s",
                model_name,
            )

    # Apply the MTP patch whenever the model has MTP heads on a compatible
    # model_type — even when mtp_enabled is False. The patch is required
    # for *sanitize correctness*: stock mlx-lm Model.sanitize triggers a
    # +1 norm shift whenever it sees mtp.* keys (assuming a raw HF
    # checkpoint), which double-shifts an already-converted MLX model and
    # corrupts the output (garbage tokens). PR 990's sanitize gates the
    # shift on "unsanitized conv1d" instead.
    #
    # Whether the model actually attaches an MTP head — and therefore
    # whether BatchGenerator runs the MTP draft+verify cycle — is gated
    # by a process-wide flag set just before mlx_lm.load() runs. With
    # mtp_enabled=False the patch is still active so sanitize behaves
    # correctly, but Model.__init__ skips ``self.mtp = MTPModule(args)``;
    # the resulting model is indistinguishable from a stock model that
    # never had MTP heads.
    if _is_mtp_compatible(config, model_type):
        mtp_enabled = bool(
            model_settings is not None and getattr(model_settings, "mtp_enabled", False)
        )
        from ..patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch,
            set_mtp_active,
            set_mtp_depth,
        )

        if apply_mlx_lm_mtp_patch():
            set_mtp_active(mtp_enabled)
            # mtp_num_draft_tokens is the MAX draft depth; an adaptive
            # controller picks 1..max per sequence from rolling accept/latency
            # estimates, so prose/chat settles at 1 and predictable text
            # climbs. Set it to 1 for a fixed depth-1 cycle. Note: depth >= 2
            # verify forwards route through the verify-shape qmm kernels
            # (M >= 3), whose numerics can diverge from the unrouted path at
            # bf16 tail-ULP level.
            depth = getattr(model_settings, "mtp_num_draft_tokens", None)
            set_mtp_depth(int(depth) if depth else 3)
            if mtp_enabled:
                logger.info(
                    "Native MTP patch applied for %s (model_type=%s, active)",
                    model_name,
                    model_type,
                )
            else:
                logger.debug(
                    "Native MTP patch applied for %s for sanitize correctness "
                    "(model has MTP heads but mtp_enabled=False; head not attached)",
                    model_name,
                )

        # mlx-vlm side: only relevant when entering through VLMBatchedEngine
        # (e.g. ``qwen3_5_moe`` with vision_config). The mlx-lm patch alone
        # can't attach an MTP head to the mlx-vlm classes — apply the
        # parallel runtime patch so MTPModule is instantiated on
        # ``LanguageModel.__init__``.
        #
        # Applied regardless of ``mtp_enabled``: with MTP off, persisted
        # ``mtp.*`` weights still need a binding site on the language model
        # tree or mlx-vlm's strict load_weights fails with "parameters not
        # in model" (issue #1404). MTP decode invocation stays gated by
        # ``is_mtp_active()`` downstream, so MTP off + module attached
        # behaves identically to a stock no-MTP model at inference time
        # (with a small constant memory cost for the unused MTPModule).
        #
        # ``for_vlm=False`` skips this branch on BatchedEngine / DFlashEngine
        # paths so mlx-vlm classes are not touched when the load goes
        # through mlx-lm only.
        if for_vlm:
            try:
                from ..patches.mlx_vlm_mtp import (
                    apply_mlx_vlm_mtp_patch,
                    apply_mlx_vlm_mtp_runtime_patch,
                    set_mtp_attach_enabled,
                )
            except Exception:
                pass
            else:
                # Decide attach-vs-skip BEFORE applying the runtime patch
                # because the patch wraps ``LanguageModel.__init__`` which
                # reads the flag at instantiation. Some Qwen3.6 MoE VLM
                # exports (unsloth UD MLX builds, issue #1426) declare
                # ``mtp_num_hidden_layers > 0`` in config.json but ship no
                # ``mtp.*`` weights; attaching MTPModule there causes
                # strict load_weights to fail with "Missing N parameters"
                # and silently downgrade the engine to LLM, dropping
                # vision. Scan the index for actual mtp.* keys and skip
                # attachment when they're absent.
                has_mtp_weights = _checkpoint_has_mtp_weights(model_name)
                set_mtp_attach_enabled(has_mtp_weights)

                # Sanitize-preservation patch runs unconditionally: the
                # stock mlx-vlm Model.sanitize strips every ``mtp.*`` key,
                # so without this an MTP head with persisted weights would
                # load at random init (0% accept). When mtp.* weights are
                # absent the patch is a no-op on the affected paths.
                if apply_mlx_vlm_mtp_patch():
                    if mtp_enabled:
                        logger.info(
                            "mlx-vlm MTP sanitize patch applied for %s",
                            model_name,
                        )
                    else:
                        logger.debug(
                            "mlx-vlm MTP sanitize patch applied for %s "
                            "(mtp_enabled=False; allows persisted mtp.* "
                            "weights to bind)",
                            model_name,
                        )
                if apply_mlx_vlm_mtp_runtime_patch():
                    if not has_mtp_weights:
                        logger.info(
                            "mlx-vlm runtime MTP patch applied for %s "
                            "(config declares mtp heads but checkpoint "
                            "ships no mtp.* weights; MTPModule attachment "
                            "skipped to keep strict load_weights happy)",
                            model_name,
                        )
                    elif mtp_enabled:
                        logger.info(
                            "mlx-vlm runtime MTP patch applied for %s",
                            model_name,
                        )
                    else:
                        logger.debug(
                            "mlx-vlm runtime MTP patch applied for %s "
                            "(mtp_enabled=False; head attached for weight "
                            "load only)",
                            model_name,
                        )
    elif model_settings is not None and getattr(model_settings, "mtp_enabled", False):
        logger.warning(
            "mtp_enabled=True for %s but model is incompatible "
            "(model_type=%r, mtp_heads=%s); MTP path will be inactive",
            model_name,
            model_type,
            _has_mtp_heads(config),
        )

    # Pre-converted mlx-lm Qwen3.6 MoE VLMs (e.g. mlx-community mxfp4) ship
    # switch_mlp weights under language_model.model.* and often declare
    # mtp_num_hidden_layers=0. The mlx_vlm_mtp sanitize replacement skips
    # unfuse when switch_mlp is already present; stock mlx-vlm sanitize
    # unconditionally pops experts.gate_up_proj and VLM load fails with
    # KeyError → LLM fallback (vision silently dropped, issue #1261). That
    # sanitize patch was previously only wired through _is_mtp_compatible
    # above; apply it here for non-MTP MoE VLMs. Runtime MTP patch stays in
    # the branch above.
    if (
        for_vlm
        and model_type
        and model_type.startswith("qwen3_5_moe")
        and not _is_mtp_compatible(config, model_type)
    ):
        try:
            from ..patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_patch
        except Exception as e:
            logger.debug("qwen3_6 MoE VLM sanitize patch import failed: %s", e)
        else:
            if apply_mlx_vlm_mtp_patch():
                logger.debug(
                    "mlx-vlm qwen3_6 MoE VLM sanitize patch applied for %s "
                    "(no MTP heads; switch_mlp load correctness)",
                    model_name,
                )

    # qwen3_5_moe covers Qwen3.6 too (HF config sets model_type=qwen3_5_moe).
    # The nested-visual sanitize wrap remaps language_model.model.visual.*
    # to vision_tower.* for Qwen3.6's nested ViT layout. Wraps whichever
    # Model.sanitize is current (stock mlx-vlm or mlx_vlm_mtp runtime), so
    # the call has to land after apply_mlx_vlm_mtp_runtime_patch above.
    # VLM-only: dflash / mlx-lm paths never instantiate mlx-vlm classes,
    # so touching them there is just dead weight.
    if for_vlm and model_type and model_type.startswith("qwen3_5_moe"):
        try:
            from ..patches.qwen3_6_nested_visual import (
                apply_qwen3_6_nested_visual_patch,
            )
        except Exception as e:
            logger.debug("qwen3_6 nested-visual patch import failed: %s", e)
        else:
            if apply_qwen3_6_nested_visual_patch():
                logger.info(
                    "qwen3_6 nested-visual sanitize wrap applied for %s",
                    model_name,
                )


def _has_mtp_heads(config: dict) -> bool:
    """True iff the model config declares any MTP head layers."""
    if int(config.get("mtp_num_hidden_layers", 0) or 0) > 0:
        return True
    if int(config.get("num_nextn_predict_layers", 0) or 0) > 0:
        return True
    text_cfg = config.get("text_config") or {}
    if int(text_cfg.get("mtp_num_hidden_layers", 0) or 0) > 0:
        return True
    if int(text_cfg.get("num_nextn_predict_layers", 0) or 0) > 0:
        return True
    return False


_MTP_WEIGHT_PREFIXES = (
    "mtp.",
    "language_model.mtp.",
    "model.mtp.",
    "model.language_model.mtp.",
)


def _nextn_weight_prefixes(model_path: str | Path) -> tuple[str, ...]:
    """Weight-key prefixes for MTP layers stored as extra decoder layers.

    DeepSeek-V3-style checkpoints (GLM-5.2 among them) keep their MTP head
    as ``model.layers.<num_hidden_layers + i>.*`` rather than ``mtp.*``;
    the model patch's sanitize remaps them at load/convert time, so for
    detection purposes those layers count as MTP weights.
    """
    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except Exception:
        return ()
    cfgs = (config, config.get("text_config") or {})
    n_mtp = max(int(c.get("num_nextn_predict_layers", 0) or 0) for c in cfgs)
    if n_mtp <= 0:
        return ()
    n_main = max(int(c.get("num_hidden_layers", 0) or 0) for c in cfgs)
    if n_main <= 0:
        return ()
    return tuple(f"model.layers.{n_main + i}." for i in range(n_mtp))


def _checkpoint_has_mtp_weights(model_path: str | Path) -> bool:
    """True iff the checkpoint at *model_path* ships any MTP weight tensor.

    Matches both the ``mtp.*`` naming and the nextn layout (extra decoder
    layers past ``num_hidden_layers``, see ``_nextn_weight_prefixes``).

    Some Qwen3.6 MoE VLM exports declare ``mtp_num_hidden_layers > 0`` in
    ``config.json`` but strip the MTP weights during conversion (e.g.
    ``unsloth/Qwen3.6-35B-A3B-UD-MLX-*bit``). Attaching ``MTPModule`` for
    such a checkpoint causes mlx-vlm's strict ``load_weights`` to fail with
    "Missing N parameters: language_model.mtp.*", the engine falls back to
    LLM, and vision is silently dropped (issue #1426).

    Reads ``model.safetensors.index.json`` when present (no shard I/O).
    Falls back to the first safetensors shard's metadata header. Returns
    False when neither resolves — callers treat that as "no MTP weights"
    (the conservative choice: skip MTPModule attachment).
    """
    p = Path(model_path)
    if not p.is_dir():
        return False

    prefixes = _MTP_WEIGHT_PREFIXES + _nextn_weight_prefixes(p)

    index_path = p / "model.safetensors.index.json"
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text())
            weight_map = data.get("weight_map") or {}
            return any(k.startswith(prefixes) for k in weight_map)
        except Exception as e:
            logger.debug("Failed to read %s for mtp weight scan: %s", index_path, e)

    shards = sorted(p.glob("*.safetensors"))
    if not shards:
        return False
    try:
        import safetensors
    except Exception as e:
        logger.debug("safetensors import failed for mtp weight scan: %s", e)
        return False

    for shard in shards:
        try:
            with safetensors.safe_open(str(shard), framework="numpy") as f:
                for k in f.keys():
                    if k.startswith(prefixes):
                        return True
        except Exception as e:
            logger.debug("Failed to read %s header for mtp weight scan: %s", shard, e)
    return False


def _is_mtp_compatible(config: dict, model_type: str | None) -> bool:
    """Decide whether the native MTP patch can be applied to this model.

    Supports Qwen3.5/3.6 (mlx-lm PR 990), DeepSeek-V4-Flash (Blaizzy/mlx-lm
    fork PR 15) and GLM-5.2 (glm_moe_dsa). The model also has to declare
    MTP heads in the config; otherwise the patch is a no-op.
    """
    if not _has_mtp_heads(config):
        return False
    if not model_type:
        return False
    return (
        model_type.startswith("qwen3_5")
        or model_type.startswith("qwen3_6")
        or model_type.startswith("deepseek_v4")
        or model_type == "glm_moe_dsa"
    )


def load_text_model(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
    model_settings: Any | None = None,
):
    """Load an LLM model/tokenizer pair via mlx-lm."""
    maybe_apply_pre_load_patches(model_name, model_settings=model_settings)
    trust_remote_code = (
        bool(getattr(model_settings, "trust_remote_code", False))
        if model_settings is not None
        else False
    )
    return lm_load_compat(
        model_name,
        tokenizer_config=tokenizer_config,
        trust_remote_code=trust_remote_code,
    )


def materialize_lazy_state(model: Any) -> None:
    """Force-evaluate every mx.array in the model tree on the loader thread.

    mlx-vlm's load() runs `mx.eval(model.language_model.parameters())`, which
    leaves frozen buffers (RoPE freqs and similar) plus sibling sub-trees
    (vision_tower, audio_tower) as lazy arrays bound to the loader thread's
    default stream. When a different thread (e.g. an EngineCore per-engine
    executor introduced in #1304) later runs forward, mx.eval hits "no
    Stream(gpu, X) in current thread" because those lazy ops target a stream
    that only exists on the loader thread. Materializing the whole tree here
    makes every leaf array safe to read from any thread afterwards.
    """
    arrays = [v for _, v in tree_flatten(model) if isinstance(v, mx.array)]
    if arrays:
        mx.eval(arrays)


def apply_post_load_transforms(model: Any, model_settings: Any = None) -> Any:
    """Apply optional post-load model transforms based on settings.

    Currently supports:
    - IndexCache: skip redundant indexer computation in DSA layers

    Args:
        model: A loaded mlx-lm model instance.
        model_settings: A ModelSettings instance (or None).

    Returns:
        The (possibly patched) model.
    """
    if model_settings is None:
        return model

    index_cache_freq = getattr(model_settings, "index_cache_freq", None)
    if index_cache_freq is not None and index_cache_freq >= 2:
        from ..patches.index_cache import apply_index_cache

        applied = apply_index_cache(model, index_cache_freq)
        if applied:
            logger.info(f"IndexCache applied: freq={index_cache_freq}")

    return model


def maybe_load_custom_quantization(
    model_name: str,
    *,
    is_vlm: bool,
) -> tuple[Any, Any] | None:
    """Load models that require a custom upstream quantization loader.

    Returns ``None`` when the model does not declare a known custom
    quantization method. The custom loaders (e.g. paroquant) handle
    their own tokenizer/processor wiring, so omlx's tokenizer_config
    and trust_remote_code are not forwarded.
    """
    config_path = Path(model_name) / "config.json"
    if not config_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text())
    except Exception as e:
        logger.debug(
            "Could not read %s for custom quantization dispatch: %s",
            config_path,
            e,
        )
        return None

    quant_config = config.get("quantization_config")
    quant_method = quant_config.get("quant_method") if quant_config else None

    if not quant_method:
        return None

    if quant_method.lower() == "paroquant":
        try:
            from paroquant.inference.backends.mlx.load import load as paro_load
        except ImportError as e:
            raise ImportError(
                "This model uses ParoQuant. Install it separately with: "
                'pip install "paroquant[mlx]"'
            ) from e

        model, processor, loaded_is_vlm = paro_load(model_name, force_text=not is_vlm)
        if is_vlm and not loaded_is_vlm:
            raise ValueError(
                "ParoQuant loader returned a text-only model for VLM load: "
                f"{model_name}"
            )
    else:
        # The quant method may be already supported by mlx-lm; simply return None.
        return None

    return model, processor
