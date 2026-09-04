# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 compatibility layer for newer mlx-vlm pins.

The mlx-vlm e390667 pin removed the out-of-tree MiniMax M3 implementation
that oMLX currently depends on. This module keeps the compatibility surface
small: it vendors the removed model/parser modules and restores only the
server-facing helpers oMLX uses.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"
_MINIMAX_MODEL_TYPES = {"minimax_m3", "minimax_m3_vl"}
_MINIMAX_ARCHITECTURES = {
    "MiniMaxM3ForCausalLM",
    "MiniMaxM3SparseForCausalLM",
}

_APPLIED = False
_IGNORED_LAYER_STACK: list[tuple[str, ...]] = []


def apply_mlx_vlm_minimax_m3_compat_patch() -> bool:
    """Install vendored MiniMax M3 modules and targeted mlx-vlm wrappers."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        _install_vendor_namespace()
        _import_vendor_modules()

        import mlx_vlm.prompt_utils as prompt_utils
        import mlx_vlm.utils as vlm_utils

        _patch_utils(vlm_utils)
        _patch_prompt_utils(prompt_utils)
    except Exception as exc:  # noqa: BLE001
        logger.debug("MiniMax M3 mlx-vlm compat patch failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("MiniMax M3 mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _install_vendor_namespace() -> None:
    import mlx_vlm
    import mlx_vlm.models
    import mlx_vlm.tool_parsers

    _append_package_path(mlx_vlm, _VENDOR_MLX_VLM)
    _append_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")
    _append_package_path(mlx_vlm.tool_parsers, _VENDOR_MLX_VLM / "tool_parsers")


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_str = str(path)
    if path_str not in package_path:
        package_path.append(path_str)


def _import_vendor_modules() -> None:
    # Importing the processor module installs mlx-vlm's AutoProcessor hook for
    # minimax_m3_vl via install_auto_processor_patch().
    for module_name in (
        "mlx_vlm.models.minimax_m3_vl.processing_minimax_m3_vl",
        "mlx_vlm.models.minimax_m3_vl",
        "mlx_vlm.models.minimax_m3",
        "mlx_vlm.tool_parsers.minimax_m3",
    ):
        importlib.import_module(module_name)


def _patch_utils(vlm_utils: Any) -> None:
    _patch_skip_multimodal_module(vlm_utils)
    _patch_get_model_and_args(vlm_utils)
    _patch_process_inputs(vlm_utils)
    _patch_stopping_criteria(vlm_utils)
    _patch_load_config_and_model(vlm_utils)


def _patch_skip_multimodal_module(vlm_utils: Any) -> None:
    original = getattr(vlm_utils, "skip_multimodal_module", None)
    if original is None or getattr(original, "_omlx_minimax_m3_compat", False):
        return

    def patched_skip_multimodal_module(path: str) -> bool:
        return "patch_merge_mlp" in path or original(path)

    patched_skip_multimodal_module._omlx_minimax_m3_compat = True
    patched_skip_multimodal_module._omlx_original = original
    vlm_utils.skip_multimodal_module = patched_skip_multimodal_module


def _patch_get_model_and_args(vlm_utils: Any) -> None:
    original = getattr(vlm_utils, "get_model_and_args", None)
    if original is None or getattr(original, "_omlx_minimax_m3_compat", False):
        return

    def patched_get_model_and_args(config: dict):
        raw_model_type = (
            config.get("model_type") if isinstance(config, dict) else None
        )
        if raw_model_type == "minimax_m3_vl":
            module, model_type = original(config)
            if model_type != "minimax_m3_vl":
                module = importlib.import_module("mlx_vlm.models.minimax_m3_vl")
                return module, "minimax_m3_vl"
            return module, model_type

        if (
            isinstance(config, dict)
            and not _is_minimax_model_type(raw_model_type)
            and _has_minimax_architecture(config)
        ):
            patched_config = dict(config)
            patched_config["model_type"] = "minimax_m3"
            return original(patched_config)
        return original(config)

    patched_get_model_and_args._omlx_minimax_m3_compat = True
    patched_get_model_and_args._omlx_original = original
    vlm_utils.get_model_and_args = patched_get_model_and_args


def _has_minimax_architecture(config: dict[str, Any]) -> bool:
    architectures = config.get("architectures") or []
    if any(str(arch) in _MINIMAX_ARCHITECTURES for arch in architectures):
        return True

    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_architectures = text_config.get("architectures") or []
        return any(str(arch) in _MINIMAX_ARCHITECTURES for arch in text_architectures)
    return False


def _patch_process_inputs(vlm_utils: Any) -> None:
    original = getattr(vlm_utils, "process_inputs", None)
    if original is None or getattr(original, "_omlx_minimax_m3_compat", False):
        return

    def patched_process_inputs(
        processor,
        prompts,
        images=None,
        audio=None,
        add_special_tokens=False,
        padding=True,
        padding_side="left",
        return_tensors="mlx",
        **kwargs,
    ):
        process_method = getattr(processor, "process", processor)
        parameters = inspect.signature(process_method).parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in parameters.values()
        )

        args = {
            "text": prompts,
            "images": images,
            "padding": padding,
            "return_tensors": return_tensors,
        }
        if "padding_side" in parameters:
            args["padding_side"] = padding_side
        if "add_special_tokens" in parameters:
            args["add_special_tokens"] = add_special_tokens

        for param in parameters:
            if param in kwargs:
                args[param] = kwargs[param]
        if accepts_kwargs:
            for key, value in kwargs.items():
                args.setdefault(key, value)

        if audio is not None and len(audio) > 0:
            if "audio" in parameters:
                args["audio"] = audio
            elif "audios" in parameters:
                args["audios"] = audio
            else:
                raise ValueError(
                    f"Processor {processor.__class__.__name__} does not support "
                    "audio parameter"
                )

        return process_method(**args)

    patched_process_inputs._omlx_minimax_m3_compat = True
    patched_process_inputs._omlx_original = original
    vlm_utils.process_inputs = patched_process_inputs


def _patch_stopping_criteria(vlm_utils: Any) -> None:
    original_cls = getattr(vlm_utils, "StoppingCriteria", None)
    if original_cls is None or getattr(original_cls, "_omlx_minimax_m3_compat", False):
        return

    class PatchedStoppingCriteria(original_cls):
        def __init__(self, eos_token_ids, tokenizer=None):
            if eos_token_ids is None:
                self.eos_token_ids = []
                self.tokenizer = tokenizer
                return
            super().__init__(eos_token_ids, tokenizer)

    PatchedStoppingCriteria.__name__ = original_cls.__name__
    PatchedStoppingCriteria.__qualname__ = original_cls.__qualname__
    PatchedStoppingCriteria._omlx_minimax_m3_compat = True
    PatchedStoppingCriteria._omlx_original = original_cls
    vlm_utils.StoppingCriteria = PatchedStoppingCriteria


def _patch_load_config_and_model(vlm_utils: Any) -> None:
    original_load_config = getattr(vlm_utils, "load_config", None)
    if original_load_config is not None and not getattr(
        original_load_config, "_omlx_minimax_m3_compat", False
    ):

        def patched_load_config(model_path, **kwargs):
            config = original_load_config(model_path, **kwargs)
            if isinstance(config, dict):
                _normalize_quantization_config(config)
            return config

        patched_load_config._omlx_minimax_m3_compat = True
        patched_load_config._omlx_original = original_load_config
        vlm_utils.load_config = patched_load_config

    _patch_nn_quantize_for_ignored_layers()

    original_load_model = getattr(vlm_utils, "load_model", None)
    if original_load_model is None or getattr(
        original_load_model, "_omlx_minimax_m3_compat", False
    ):
        return

    def patched_load_model(model_path, lazy=False, **kwargs):
        ignored_layers: tuple[str, ...] = ()
        try:
            load_kwargs = dict(kwargs)
            load_kwargs.pop("strict", None)
            config = vlm_utils.load_config(model_path, **load_kwargs)
            ignored_layers = _quantization_ignored_layers(config)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not inspect quantization ignored_layers: %s", exc)

        with _ignored_layers_context(ignored_layers):
            return original_load_model(model_path, lazy=lazy, **kwargs)

    patched_load_model._omlx_minimax_m3_compat = True
    patched_load_model._omlx_original = original_load_model
    vlm_utils.load_model = patched_load_model


def _normalize_quantization_config(config: dict[str, Any]) -> None:
    quantization_config = config.get("quantization_config")
    text_config = config.get("text_config")
    if quantization_config is None and isinstance(text_config, dict):
        quantization_config = text_config.get("quantization_config")
        if isinstance(quantization_config, dict):
            config["quantization_config"] = quantization_config

    if not isinstance(quantization_config, dict):
        return

    quant_method = str(quantization_config.get("quant_method") or "").lower()
    if quant_method != "mxfp8" or "quantization" in config:
        return

    config["quantization"] = {
        "group_size": int(quantization_config.get("group_size") or 32),
        "bits": int(quantization_config.get("bits") or 8),
        "mode": "mxfp8",
    }


def _quantization_ignored_layers(config: Any) -> tuple[str, ...]:
    if not isinstance(config, dict):
        return ()

    ignored: list[str] = []
    for key in ("quantization_config", "quantization"):
        value = config.get(key)
        if isinstance(value, dict):
            layers = value.get("ignored_layers") or ()
            ignored.extend(str(layer) for layer in layers)

    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        value = text_config.get("quantization_config")
        if isinstance(value, dict):
            ignored.extend(str(layer) for layer in value.get("ignored_layers") or ())

    return tuple(dict.fromkeys(layer for layer in ignored if layer))


def _patch_nn_quantize_for_ignored_layers() -> None:
    import mlx.nn as nn

    original = nn.quantize
    if getattr(original, "_omlx_minimax_m3_compat", False):
        return

    def patched_quantize(module, *args, **kwargs):
        ignored_layers = _IGNORED_LAYER_STACK[-1] if _IGNORED_LAYER_STACK else ()
        class_predicate = kwargs.get("class_predicate")
        if ignored_layers and class_predicate is not None:

            def combined_predicate(path, submodule):
                if _is_ignored_layer(path, ignored_layers):
                    return False
                return class_predicate(path, submodule)

            kwargs["class_predicate"] = combined_predicate
        return original(module, *args, **kwargs)

    patched_quantize._omlx_minimax_m3_compat = True
    patched_quantize._omlx_original = original
    nn.quantize = patched_quantize


def _is_ignored_layer(path: str, ignored_layers: tuple[str, ...]) -> bool:
    return any(path == layer or path.startswith(f"{layer}.") for layer in ignored_layers)


@contextlib.contextmanager
def _ignored_layers_context(ignored_layers: tuple[str, ...]):
    _IGNORED_LAYER_STACK.append(ignored_layers)
    try:
        yield
    finally:
        _IGNORED_LAYER_STACK.pop()


def _patch_prompt_utils(prompt_utils: Any) -> None:
    original_get_message_json = getattr(prompt_utils, "get_message_json", None)
    if original_get_message_json is not None and not getattr(
        original_get_message_json, "_omlx_minimax_m3_compat", False
    ):

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
            if _is_minimax_model_type(model_name):
                return _format_minimax_message(
                    prompt,
                    role,
                    skip_image_token=skip_image_token,
                    num_images=num_images,
                )
            return original_get_message_json(
                model_name,
                prompt,
                role=role,
                skip_image_token=skip_image_token,
                skip_audio_token=skip_audio_token,
                num_images=num_images,
                num_audios=num_audios,
                **kwargs,
            )

        patched_get_message_json._omlx_minimax_m3_compat = True
        patched_get_message_json._omlx_original = original_get_message_json
        prompt_utils.get_message_json = patched_get_message_json

    original_apply_chat_template = getattr(prompt_utils, "apply_chat_template", None)
    if original_apply_chat_template is None or getattr(
        original_apply_chat_template, "_omlx_minimax_m3_compat", False
    ):
        return

    def patched_apply_chat_template(
        processor,
        config,
        prompt,
        add_generation_prompt: bool = True,
        return_messages: bool = False,
        num_images: int = 0,
        num_audios: int = 0,
        **kwargs,
    ):
        model_type = _config_model_type(config)
        if not _is_minimax_model_type(model_type):
            return original_apply_chat_template(
                processor,
                config,
                prompt,
                add_generation_prompt=add_generation_prompt,
                return_messages=return_messages,
                num_images=num_images,
                num_audios=num_audios,
                **kwargs,
            )

        template_kwargs = dict(kwargs)
        _apply_minimax_thinking_kwargs(template_kwargs)
        messages = _build_minimax_messages(prompt, num_images=num_images)
        if return_messages:
            return messages
        return prompt_utils.get_chat_template(
            processor,
            messages,
            add_generation_prompt,
            **template_kwargs,
        )

    patched_apply_chat_template._omlx_minimax_m3_compat = True
    patched_apply_chat_template._omlx_original = original_apply_chat_template
    prompt_utils.apply_chat_template = patched_apply_chat_template


def _is_minimax_model_type(model_type: Any) -> bool:
    return isinstance(model_type, str) and model_type.lower() in _MINIMAX_MODEL_TYPES


def _config_model_type(config: Any) -> str | None:
    if isinstance(config, dict):
        value = config.get("model_type")
    else:
        value = getattr(config, "model_type", None)
    return value if isinstance(value, str) else None


def _apply_minimax_thinking_kwargs(kwargs: dict[str, Any]) -> None:
    enable_thinking = kwargs.pop("enable_thinking", None)
    if "thinking_mode" in kwargs:
        return
    if enable_thinking is True:
        kwargs["thinking_mode"] = "enabled"
    elif enable_thinking is False:
        kwargs["thinking_mode"] = "disabled"


def _build_minimax_messages(prompt: Any, num_images: int = 0) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [_format_minimax_message(prompt, "user", num_images=num_images)]

    if isinstance(prompt, dict):
        return [_minimax_message_from_dict(prompt, fallback_num_images=num_images)]

    if isinstance(prompt, list):
        last_user_idx = _last_user_message_index(prompt)
        messages = []
        for idx, item in enumerate(prompt):
            if isinstance(item, str):
                messages.append(
                    _format_minimax_message(
                        item,
                        "user",
                        skip_image_token=idx != last_user_idx,
                        num_images=num_images if idx == last_user_idx else 0,
                    )
                )
            elif isinstance(item, dict):
                messages.append(
                    _minimax_message_from_dict(
                        item,
                        fallback_num_images=num_images if idx == last_user_idx else 0,
                        skip_image_token=idx != last_user_idx,
                    )
                )
            else:
                messages.append(
                    _format_minimax_message(str(item), "user", num_images=0)
                )
        return messages

    return [_format_minimax_message(str(prompt), "user", num_images=num_images)]


def _last_user_message_index(items: list[Any]) -> int:
    last_user_idx = -1
    for idx, item in enumerate(items):
        if isinstance(item, str):
            last_user_idx = idx
        elif isinstance(item, dict):
            role = item.get("role", "user")
            if role not in ("system", "assistant", "tool"):
                last_user_idx = idx
    return last_user_idx


def _minimax_message_from_dict(
    message: dict[str, Any],
    fallback_num_images: int = 0,
    skip_image_token: bool = False,
) -> dict[str, Any]:
    role = message.get("role", "user")
    if role == "tool" or "tool_calls" in message or "tool_call_id" in message:
        return dict(message)

    raw_content = message.get("content", "")
    num_images = _count_image_parts(raw_content) or fallback_num_images
    content = _extract_text_from_content(raw_content)
    return _format_minimax_message(
        content,
        role,
        skip_image_token=skip_image_token,
        num_images=num_images,
    )


def _format_minimax_message(
    prompt: Any,
    role: str,
    skip_image_token: bool = False,
    num_images: int = 0,
) -> dict[str, Any]:
    content = "" if prompt is None else str(prompt)
    if role == "user" and not skip_image_token and num_images > 0:
        content = ("]<]image[>[" * num_images) + content
    return {"role": role, "content": content}


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "input_text"):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return "" if content is None else str(content)


def _count_image_parts(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    count = 0
    for item in content:
        if isinstance(item, dict) and item.get("type") in (
            "image",
            "image_url",
            "input_image",
        ):
            count += 1
    return count


__all__ = ["apply_mlx_vlm_minimax_m3_compat_patch", "is_applied"]
