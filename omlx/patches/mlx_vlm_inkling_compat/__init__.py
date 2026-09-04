# SPDX-License-Identifier: Apache-2.0
"""Inkling (Thinking Machines) compatibility layer for the pinned mlx-vlm.

The `inkling` model package landed in mlx-vlm PR #1637 and was extended
for Inkling Small in PR #1756 (pcuenca/mlx-vlm@73e4cb6), both newer than
oMLX's mlx-vlm pin (`78b96eb`). This vendors the PR-head model package
plus the three shared modules it imports that do not exist at the pin
(`mlx_vlm.models.{mlp,switch_layers,activations}` — the pin sources
SwitchGLU from mlx-lm, whose activation contract differs), and wires the
discovery surface oMLX needs:

- installs the vendored packages onto the real `mlx_vlm.models` namespace
  (`__path__` append) so `get_model_and_args` can import them; relative
  imports (`..base`, `..cache`) resolve against the real pinned mlx-vlm.
  If a future pin bump ships these modules upstream, the real ones win
  automatically (the vendor path is searched last).
- adds `inkling_mm_model -> inkling` to `mlx_vlm.utils.MODEL_REMAPPING`
  (checkpoints declare `model_type: "inkling_mm_model"`).
- registers `inkling`/`inkling_mm_model` in `prompt_utils.MODEL_CONFIG`
  and wraps `get_message_json` with LIST_WITH_IMAGE_FIRST semantics
  (image parts first, then text), matching PR #1756. Audio parts are not
  emitted yet: oMLX serves inkling text+vision first, and the vendored
  processor rejects audio arrays with a clear error.
- registers the vendored torch-free `InklingProcessor` with
  `AutoProcessor` (the reference processors only exist in transformers
  5.14+, above oMLX's <5.13 pin).
- wraps `mlx_vlm.utils.load_config` to translate a ModelOpt
  `hf_quant_config.json` (quant_algo NVFP4) into MLX quantization config
  `{group_size: 16, bits: 4, mode: "nvfp4"}`, porting PR #1756's
  `load_model` hunk, so the official NVFP4 checkpoint loads natively on
  stock MLX 0.32 nvfp4 kernels.
- forces `Model.sanitize` before quantization only when an MLX-format Inkling
  checkpoint still carries source `model.llm.*`/`model.visual.*` keys. This
  supports community affine conversions without re-sanitizing oMLX's native
  QKVR/MTP output layout.
"""

from __future__ import annotations

import importlib
import json
import logging
from contextvars import ContextVar
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"

# Checkpoints carry "inkling_mm_model"; the module directory is "inkling".
_MODEL_TYPES = ("inkling", "inkling_mm_model")
_MODULE_NAME = "inkling"

_APPLIED = False
_FORCE_SANITIZE_DIR: ContextVar[Path | None] = ContextVar(
    "omlx_inkling_force_sanitize_dir", default=None
)
_RAW_WEIGHT_PREFIXES = ("model.llm.", "model.visual.", "model.audio.", "model.mtp.")


def apply_mlx_vlm_inkling_compat_patch() -> bool:
    """Install the vendored inkling module and mlx-vlm discovery hooks."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        _install_vendor_namespace()
        _import_vendor_modules()

        import mlx_vlm.prompt_utils as prompt_utils
        import mlx_vlm.utils as vlm_utils

        _patch_model_remapping(vlm_utils)
        _patch_prompt_utils(prompt_utils)
        _patch_load_config(vlm_utils)
        _patch_force_sanitize(vlm_utils)
        _register_auto_processor()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Inkling mlx-vlm compat patch failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("Inkling mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _install_vendor_namespace() -> None:
    import mlx_vlm
    import mlx_vlm.models

    _append_package_path(mlx_vlm, _VENDOR_MLX_VLM)
    _append_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_str = str(path)
    if path_str not in package_path:
        package_path.append(path_str)


def _import_vendor_modules() -> None:
    # The shared shims must be importable before the model package, whose
    # language module does `from ..mlp import SwiGLUMLP` etc.
    importlib.import_module("mlx_vlm.models.activations")
    importlib.import_module("mlx_vlm.models.mlp")
    importlib.import_module("mlx_vlm.models.switch_layers")
    importlib.import_module(f"mlx_vlm.models.{_MODULE_NAME}")


def _patch_model_remapping(vlm_utils: Any) -> None:
    remapping = getattr(vlm_utils, "MODEL_REMAPPING", None)
    if isinstance(remapping, dict):
        remapping.setdefault("inkling_mm_model", _MODULE_NAME)


def _is_inkling(model_name: Any) -> bool:
    return isinstance(model_name, str) and model_name.lower() in _MODEL_TYPES


def _patch_prompt_utils(prompt_utils: Any) -> None:
    # apply_chat_template short-circuits to text-only formatting for any
    # model_type missing from MODEL_CONFIG, dropping image parts entirely.
    # Register both keys; the mapped format is never consulted because the
    # get_message_json wrapper intercepts inkling first.
    model_config = getattr(prompt_utils, "MODEL_CONFIG", None)
    message_format = getattr(prompt_utils, "MessageFormat", None)
    if isinstance(model_config, dict) and message_format is not None:
        placeholder = getattr(message_format, "LIST_WITH_IMAGE_FIRST", None) or getattr(
            message_format, "LIST_WITH_IMAGE", None
        )
        if placeholder is not None:
            for model_type in _MODEL_TYPES:
                model_config.setdefault(model_type, placeholder)

    original = getattr(prompt_utils, "get_message_json", None)
    if original is None or getattr(original, "_omlx_inkling_compat", False):
        return

    def patched_get_message_json(
        model_name: str,
        prompt: str,
        role: str = "user",
        skip_image_token: bool = False,
        skip_audio_token: bool = False,
        num_images: int = 0,
        num_audios: int = 0,
        **kwargs,
    ):
        if _is_inkling(model_name):
            text = "" if prompt is None else str(prompt)
            content: list[dict[str, Any]] | str
            if role == "user" and not skip_image_token and num_images > 0:
                content = [{"type": "image"} for _ in range(num_images)]
                content.append({"type": "text", "text": text})
            else:
                content = text
            return {"role": role, "content": content}
        return original(
            model_name,
            prompt,
            role=role,
            skip_image_token=skip_image_token,
            skip_audio_token=skip_audio_token,
            num_images=num_images,
            num_audios=num_audios,
            **kwargs,
        )

    patched_get_message_json._omlx_inkling_compat = True
    patched_get_message_json._omlx_original = original
    prompt_utils.get_message_json = patched_get_message_json


def _patch_load_config(vlm_utils: Any) -> None:
    original = getattr(vlm_utils, "load_config", None)
    if original is None or getattr(original, "_omlx_inkling_compat", False):
        return

    def patched_load_config(model_path, **kwargs):
        config = original(model_path, **kwargs)
        try:
            if (
                not isinstance(config, dict)
                or config.get("model_type") not in _MODEL_TYPES
            ):
                return config
            if not config.get("quantization") and not config.get(
                "quantization_config"
            ):
                hf_quant_path = Path(model_path) / "hf_quant_config.json"
                if hf_quant_path.exists():
                    with open(hf_quant_path, encoding="utf-8") as f:
                        hf_quant = json.load(f).get("quantization", {})
                    if hf_quant.get("quant_algo") == "NVFP4":
                        # nvfp4 implies group_size=16 (PR #1756). The
                        # sanitize in the vendored model passes the packed
                        # expert codes/scales through byte-identically.
                        config["quantization"] = config["quantization_config"] = {
                            "group_size": 16,
                            "bits": 4,
                            "mode": "nvfp4",
                        }

            from mlx_vlm.models.inkling.config import build_qkvr_fusion_policy

            text_config = config.setdefault("text_config", {})
            quantization = config.get("quantization")
            if not isinstance(quantization, dict):
                quantization = config.get("quantization_config")
            mtp_config = config.get("mtp_config") or {}
            mtp_layers = int(
                text_config.get("mtp_num_hidden_layers")
                or mtp_config.get("num_nextn_predict_layers")
                or mtp_config.get("num_mtp_layers")
                or 0
            )
            trunk_policy = build_qkvr_fusion_policy(
                quantization,
                int(text_config.get("num_hidden_layers") or 0),
                "language_model.model.layers.{layer_idx}.self_attn.",
            )
            mtp_policy = build_qkvr_fusion_policy(
                quantization,
                mtp_layers,
                "language_model.mtp.blocks.{layer_idx}.transformer_block.self_attn.",
            )
            text_config["qkvr_fused_layers"] = trunk_policy
            text_config["mtp_qkvr_fused_layers"] = mtp_policy
        except Exception as exc:  # noqa: BLE001
            logger.debug("Inkling hf_quant_config translation failed: %s", exc)
        return config

    patched_load_config._omlx_inkling_compat = True
    patched_load_config._omlx_original = original
    vlm_utils.load_config = patched_load_config


def _has_raw_inkling_weights(model_path: Any) -> bool:
    """Return whether an Inkling checkpoint still uses its source key layout."""
    path = Path(model_path)
    try:
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") not in _MODEL_TYPES:
            return False

        for index_path in sorted(path.glob("*.safetensors.index.json")):
            index = json.loads(index_path.read_text())
            keys = (index.get("weight_map") or {}).keys()
            if any(key.startswith(_RAW_WEIGHT_PREFIXES) for key in keys):
                return True

        import safetensors

        shards = sorted(
            shard
            for shard in path.glob("*.safetensors")
            if not shard.name.endswith("consolidated.safetensors")
        )
        for shard in shards:
            with safetensors.safe_open(shard, framework="np") as handle:
                if any(
                    key.startswith(_RAW_WEIGHT_PREFIXES)
                    for key in handle.keys()  # noqa: SIM118 - safe_open is not a dict
                ):
                    return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Inkling raw-layout detection failed for %s: %s", path, exc)
    return False


def _patch_force_sanitize(vlm_utils: Any) -> None:
    """Run the pinned mlx-vlm sanitizer for raw MLX-format Inkling weights.

    The pinned loader trusts ``format=mlx`` and skips ``Model.sanitize``. Community
    Inkling conversions retain the source ``model.llm.*`` naming despite that
    marker, so their keys and quantization sidecars must still be remapped before
    ``nn.quantize`` inspects the weight dictionary. The context-local directory
    keeps the metadata override scoped to this checkpoint and safe across threads.
    """
    import safetensors

    original_load_model = getattr(vlm_utils, "load_model", None)
    if original_load_model is None or getattr(
        original_load_model, "_omlx_inkling_force_sanitize", False
    ):
        return

    original_safe_open = safetensors.safe_open

    class _SafeOpenMetadataWrapper:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def metadata(self):
            metadata = self._inner.metadata()
            if isinstance(metadata, dict) and metadata.get("format") == "mlx":
                metadata = dict(metadata)
                metadata.pop("format", None)
            return metadata

    def patched_safe_open(filename, *args, **kwargs):
        handle = original_safe_open(filename, *args, **kwargs)
        target_dir = _FORCE_SANITIZE_DIR.get()
        if target_dir is None:
            return handle
        try:
            path = Path(filename).resolve()
        except TypeError:
            return handle
        if path.parent == target_dir and path.suffix == ".safetensors":
            return _SafeOpenMetadataWrapper(handle)
        return handle

    def patched_load_model(model_path, lazy=False, **kwargs):
        if not _has_raw_inkling_weights(model_path):
            return original_load_model(model_path, lazy=lazy, **kwargs)
        target_dir = Path(model_path).resolve()
        token = _FORCE_SANITIZE_DIR.set(target_dir)
        try:
            logger.info("Forcing sanitize for raw Inkling weights in %s", target_dir)
            return original_load_model(model_path, lazy=lazy, **kwargs)
        finally:
            _FORCE_SANITIZE_DIR.reset(token)

    patched_safe_open._omlx_inkling_force_sanitize = True
    patched_safe_open._omlx_original = original_safe_open
    patched_load_model._omlx_inkling_force_sanitize = True
    patched_load_model._omlx_original = original_load_model
    safetensors.safe_open = patched_safe_open
    vlm_utils.load_model = patched_load_model


def _register_auto_processor() -> None:
    from mlx_vlm.models.base import install_auto_processor_patch

    # Import through the installed namespace (the vendor tree has no
    # package __init__ chain of its own).
    processing = importlib.import_module(
        f"mlx_vlm.models.{_MODULE_NAME}.processing_inkling"
    )
    install_auto_processor_patch(list(_MODEL_TYPES), processing.InklingProcessor)


__all__ = ["apply_mlx_vlm_inkling_compat_patch", "is_applied"]
