# SPDX-License-Identifier: Apache-2.0
"""SenseNova-Loop DFlash2 VLM adaptation (5-feature fc, VLM engine).

The VLM engine uses components (mlx_vlm.speculative.dflash,
mlx_vlm.models.qwen3_5 LanguageModel.__call__ + Qwen3_5BatchInvariantForward),
and the native VLM ``sensenova_loop`` body already supports:

  * 56-slot cache (execution_plan) -- make_cache uses self.layers=execution_layers;
  * 5-feature capture -- Qwen3_5Model.__call__ uses execution_layers +
    hidden_sink (4 layer slots) + return_hidden appends post-norm h (5th feature);
  * rollback transaction -- SpeculativeCacheTransaction walks all cache leaves.

So the VLM patch only needs to:

  1. bypass the loop9b LanguageModel.__call__ speculative_verify ValueError;
  2. make Qwen3_5BatchInvariantForward._model use execution_layers (56)
     instead of model.layers (32 physical);
  3. parse target_feature_sources in DFlash2Config;
  4. size DFlashDraftModel fc from the feature count;
  5. validate_dflash_target against the execution plan (56).

All changes are monkey-patches; the VLM internals (mlx_vlm/models/*,
mlx_vlm/speculative/*) are untouched.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

# The marker used inside ``target_feature_sources`` for the backbone's final
# post-norm hidden state.
_FINAL_POSTNORM = "final_postnorm"

# SenseNova-Loop model types that carry a DFlash2 5-feature drafter.
_SENSENOVA_LOOP_MODEL_TYPES = ("sensenova_loop", "sensenova_loop_text")

# ---------------------------------------------------------------------------
# 1. DFlash2Config: parse target_feature_sources from the checkpoint config.
# ---------------------------------------------------------------------------


def _patch_dflash2_config() -> None:
    """Teach ``DFlash2Config.from_dict`` to carry ``target_feature_sources``."""
    try:
        from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
    except Exception as e:  # pragma: no cover
        logger.debug("DFlash2 config patch skipped: %s", e)
        return

    if getattr(DFlash2Config, "_sl_dflash_tfs_patched", False):
        return

    orig_from_dict = DFlash2Config.from_dict.__func__

    def _patched_from_dict(cls, params):
        config = orig_from_dict(cls, params)
        tfs = params.get("target_feature_sources")
        if tfs is None:
            dflash = params.get("dflash_config") or {}
            tfs = dflash.get("target_feature_sources")
        if tfs is not None:
            config.target_feature_sources = list(tfs)
        return config

    DFlash2Config.from_dict = classmethod(_patched_from_dict)
    DFlash2Config.from_hf_dict = classmethod(_patched_from_dict)
    setattr(DFlash2Config, "_sl_dflash_tfs_patched", True)
    logger.debug("DFlash2Config.target_feature_sources parse patched")


# ---------------------------------------------------------------------------
# 2. DFlashDraftModel.__init__: fc input width from target_feature_sources.
# ---------------------------------------------------------------------------


def _patch_dflash_draft_model() -> None:
    """Make ``DFlashDraftModel.__init__`` size ``fc`` from the feature count."""
    try:
        from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import (
            DFlashDraftModel,
        )
    except Exception as e:  # pragma: no cover
        logger.debug("DFlashDraftModel patch skipped: %s", e)
        return

    if getattr(DFlashDraftModel, "_sl_dflash_fc_patched", False):
        return

    orig_init = DFlashDraftModel.__init__

    def _patched_init(self, config):
        tfs = getattr(config, "target_feature_sources", None)
        if tfs is not None and len(tfs) > len(config.target_layer_ids):
            # Widen target_layer_ids so base __init__ sizes fc as
            # len(tfs)*hidden_size; restore so capture keeps the real ids.
            saved = config.target_layer_ids
            config.target_layer_ids = list(tfs)
            try:
                orig_init(self, config)
            finally:
                config.target_layer_ids = saved
            return
        orig_init(self, config)

    DFlashDraftModel.__init__ = _patched_init
    setattr(DFlashDraftModel, "_sl_dflash_fc_patched", True)
    logger.debug("DFlashDraftModel fc width patched")


# ---------------------------------------------------------------------------
# 3. validate_dflash_target: count execution layers for SenseNova-Loop.
# ---------------------------------------------------------------------------


def _patch_validate_dflash_target() -> None:
    """Teach ``validate_dflash_target`` the 56-slot loop execution plan.

    SenseNova-Loop maps 32 physical layers to 56 execution slots via
    ``_execution_plan`` (VLM-side attr). ``num_target_layers`` counts
    execution slots, so the layer-count check must compare against
    ``len(_execution_plan)``, not the physical count.
    """
    try:
        from mlx_vlm.speculative.drafters import compatibility as compat
    except Exception as e:  # pragma: no cover
        logger.debug("validate_dflash_target patch skipped: %s", e)
        return

    orig = getattr(compat, "validate_dflash_target", None)
    logger.warning("SL-PVT: entering orig=%s patched=%s",
        getattr(orig, "__name__", type(orig).__name__) if orig else None,
        getattr(orig, "_sl_dflash_patched", False))
    # Force-install the VLM _patched (uses _execution_plan=56). Skip only
    # when this file's _patched is already installed.
    if orig is None:
        return
    if orig is getattr(compat, "_sl_vlm_patched_fn", None):
        logger.warning("SL-PVT: already VLM patched, skip")
        return

    def _patched(config, target_model, algorithm):
        try:
            language_model = compat._target_metadata(target_model)[0]
        except Exception as _e:
            language_model = None
            logger.warning("SL-VALIDATE: _target_metadata failed: %s", _e)
        logger.warning(
            "SL-VALIDATE: %s target=%s lm=%s plan?=%s",
            algorithm,
            type(target_model).__name__,
            type(language_model).__name__ if language_model else "None",
            hasattr(language_model, "_execution_plan") if language_model else "N/A",
        )
        # VLM loop9b: LanguageModel.layers = execution_layers (56);
        # _execution_plan lives on TextModel (language_model.model).
        inner = getattr(language_model, "model", None)
        plan = getattr(language_model, "_execution_plan", None)
        if plan is None and inner is not None:
            plan = getattr(inner, "_execution_plan", None)
        if plan is None:
            # Not SenseNova-Loop: stock validator unchanged.
            return orig(config, target_model, algorithm)
        saved = config.num_target_layers
        config.num_target_layers = len(plan)
        # orig's _target_metadata uses inner.layers (32 physical) for
        # layer_count; loop9b needs execution_layers (56 slots). Temporarily
        # swap inner.layers to execution_layers so orig sees 56, then restore.
        saved_layers = None
        if inner is not None:
            saved_layers = getattr(inner, "layers", None)
            exec_layers = getattr(inner, "execution_layers", None)
            if exec_layers is not None:
                logger.warning(
                    "SL-VALIDATE: inner.layers %d -> execution_layers %d",
                    len(saved_layers) if saved_layers is not None else -1,
                    len(exec_layers),
                )
                inner.layers = exec_layers
        try:
            return orig(config, target_model, algorithm)
        finally:
            config.num_target_layers = saved
            if inner is not None and saved_layers is not None:
                inner.layers = saved_layers

    compat.validate_dflash_target = _patched
    for _mod_name in (
        "mlx_vlm.speculative.drafters.dflash2.dflash2",
        "mlx_vlm.speculative.drafters.dspark.dspark",
    ):
        try:
            _mod = importlib.import_module(_mod_name)
            if getattr(_mod, "validate_dflash_target", None) is not _patched:
                _mod.validate_dflash_target = _patched
        except Exception as _e:  # pragma: no cover
            logger.debug("validate_dflash_target rebind %s failed: %s", _mod_name, _e)
    # Rebind DFlash2DraftModel.validate_target_compatibility directly so
    # binding always goes through _patched, even if the module-level name
    # is re-imported.
    try:
        from mlx_vlm.speculative.drafters.dflash2.dflash2 import (
            DFlash2DraftModel,
        )

        def _vtc(self, target_model):
            return _patched(self.config, target_model, "DFlash2")

        DFlash2DraftModel.validate_target_compatibility = _vtc
        logger.warning("SL-VTC-REBIND: DFlash2DraftModel.validate_target_compatibility -> _vtc ok")
    except Exception as _e:  # pragma: no cover
        logger.warning("SL-VTC-REBIND-FAIL: DFlash2DraftModel validate_target_compatibility rebind failed: %s", _e)
    setattr(_patched, "_sl_dflash_patched", True)
    # Mark this file's VLM _patched for _patch_validate_dflash_target.
    compat._sl_vlm_patched_fn = _patched
    logger.debug("validate_dflash_target patched for execution plan")


# ---------------------------------------------------------------------------
# 4. Qwen3_5BatchInvariantForward._model: use execution_layers for loop9b.
# ---------------------------------------------------------------------------


def _patch_verifier_model_layers() -> None:
    """Make the Qwen3.5 exact verifier traverse ``execution_layers`` (56).

    Native ``Qwen3_5BatchInvariantForward._model`` uses ``model.layers``
    (loop9b TextModel.layers = 32 physical), but loop9b needs
    ``capture_layer_ids = [3, 27, 51, 54]`` to capture **execution slots**
    (56). Slots 51/54 exceed 32 physical layers, so the native verifier
    cannot capture them.

    Adapt: when ``model`` has ``execution_layers`` (loop9b TextModel), use
    ``model.execution_layers`` (56) instead of ``model.layers`` (32); the
    cache list is sized by execution-slot count too.
    """
    try:
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward,
        )
    except Exception as e:  # pragma: no cover
        logger.debug("verifier model layers patch skipped: %s", e)
        return

    if getattr(Qwen3_5BatchInvariantForward, "_sl_verifier_layers_patched", False):
        return

    orig_model = Qwen3_5BatchInvariantForward._model

    def _patched_model(
        self, model, inputs, cache, inputs_embeds, position_ids,
        capture_layer_ids, hidden_sink,
    ):
        # loop9b TextModel: traverse execution slots (56), not physical layers.
        layers = getattr(model, "execution_layers", None)
        if layers is None:
            return orig_model(
                self, model, inputs, cache, inputs_embeds, position_ids,
                capture_layer_ids, hidden_sink,
            )
        helpers = self._helpers()
        hidden = (
            model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        )
        if cache is None:
            cache = [None] * len(layers)

        fa_cache = cache[model.fa_idx]
        ssm_cache = cache[model.ssm_idx]
        fa_mask = helpers._create_qwen3_5_attention_mask(hidden, fa_cache)
        ssm_mask = helpers._create_qwen3_5_ssm_mask(hidden, ssm_cache)
        decode_left_padding = (
            getattr(fa_cache, "_qwen3_5_decode_left_padding", None)
            if isinstance(fa_mask, str) and fa_mask == "left_padded_decode"
            else None
        )
        helpers._set_qwen3_5_decode_left_padding(cache, layers, decode_left_padding)

        position_embeddings = None
        if position_ids is not None:
            for layer in layers:
                if not layer.is_linear:
                    if not layer.self_attn.rotary_emb.fused_apply:
                        position_embeddings = layer.self_attn.rotary_emb(
                            hidden, position_ids
                        )
                    break

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for index, (layer, layer_cache) in enumerate(zip(layers, cache)):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            hidden = self._layer(
                layer,
                hidden,
                layer_mask,
                layer_cache,
                position_ids,
                position_embeddings,
            )
            if hidden_sink is not None and index in capture_set:
                hidden_sink.append(hidden)

        return model.norm(hidden)

    Qwen3_5BatchInvariantForward._model = _patched_model
    setattr(Qwen3_5BatchInvariantForward, "_sl_verifier_layers_patched", True)
    logger.debug("Qwen3.5 verifier _model uses execution_layers for loop9b")


# ---------------------------------------------------------------------------
# 4b. Qwen3_5BatchInvariantForward.verify: enlarge speculation capacity.
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def _patch_loop_mtp_rollback() -> None:
    """Add ``mtp_partial_rollback`` to loop9b ``LanguageModel``.

    The generic DFlash target hooks (``drafter_target.apply``) are skipped
    when the target already ships a native ``rollback_speculative_cache``
    (loop9b does). That skip also omits ``mtp_partial_rollback``, so the
    VLM chain verify finds no hook and falls back to a standard step on
    every cycle, disabling DFlash speculative gain. Install the equivalent
    trim-based partial rollback here (idempotent, loop9b-only) so the chain
    verify can roll the 56-slot execution-plan cache back to the accepted
    prefix.
    """
    try:
        from mlx_vlm.models.sensenova_loop import language as sl_lang
    except Exception as e:  # pragma: no cover
        logger.debug("loop mtp_partial_rollback patch skipped: %s", e)
        return

    cls = sl_lang.LanguageModel
    if getattr(cls, "_sl_mtp_partial_rollback_patched", False):
        return
    if hasattr(cls, "mtp_partial_rollback"):
        setattr(cls, "_sl_mtp_partial_rollback_patched", True)
        return

    def _mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Trim the execution-plan cache to ``accepted`` verified drafts.

        Mirrors ``drafter_target.mtp_partial_rollback``: each non-None slot
        trims the rejected tail; all slots must report the trimmed count.
        """
        rejected = int(num_drafts) - int(accepted)
        if rejected <= 0:
            return True
        results = []
        for idx, c in enumerate(cache):
            if c is None:
                continue
            # loop9b's linear-attention slots use mlx_vlm ArraysCache, a
            # recurrent (compressed) state that has NO token window to trim.
            # Its state does not retain rejected token history, so skipping
            # the trim is semantically correct and avoids AttributeError.
            if not hasattr(c, "trim"):
                logger.debug(
                    "SL-DIAG trim[%d] SKIP (no trim; type=%s.%s)",
                    idx, type(c).__module__, type(c).__name__,
                )
                results.append(True)
                continue
            try:
                r = c.trim(rejected)
            except Exception as e:
                logger.debug(
                    "SL-DIAG trim[%d] EXC: %s (type=%s.%s)",
                    idx, e, type(c).__module__, type(c).__name__,
                )
                results.append(False)
                continue
            if r != rejected:
                logger.debug(
                    "SL-DIAG trim[%d]=%s != rejected=%s (type=%s.%s)",
                    idx, r, rejected, type(c).__module__, type(c).__name__,
                )
                results.append(False)
            else:
                results.append(True)
        return all(results)

    cls.mtp_partial_rollback = _mtp_partial_rollback
    setattr(cls, "_sl_mtp_partial_rollback_patched", True)
    logger.debug("loop9b LanguageModel mtp_partial_rollback added")


def apply() -> bool:
    """Apply all VLM SenseNova-Loop DFlash2 adaptations (idempotent)."""
    logger.warning("SL-APPLY: sensenova_loop_dflash_vlm.apply() called")
    _patch_dflash2_config()
    _patch_dflash_draft_model()
    _patch_validate_dflash_target()
    _patch_verifier_model_layers()
    # Multi-visit loop spec patches live in the shared module (used by both
    # DFlash2 and VLM MTP). Import here so DFlash2 and MTP share one source.
    from omlx.patches.sensenova_loop_spec_common import (
        _patch_loop_language_model_call,
        _patch_loop_validate_speculation,
        _patch_loop_verify_capacity,
    )
    _patch_loop_verify_capacity()
    _patch_loop_validate_speculation()
    # LanguageModel.__call__ bypass lives in the shared module; DFlash2 uses
    # its own guard flag so MTP and DFlash2 patches coexist independently.
    _patch_loop_language_model_call("_sl_spec_verify_bypassed_dflash")
    _patch_loop_mtp_rollback()
    return True


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG)
    print("Applying VLM dflash adaptation...")
    apply()
    print("Applied.")
