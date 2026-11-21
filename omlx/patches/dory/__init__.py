# SPDX-License-Identifier: Apache-2.0 AND MIT
# Tokenizer wrapper construction adapted from mlx-lm, Copyright © 2024 Apple Inc.
# See LICENSE for the mlx-lm MIT license.
"""Dory support using the mlx-lm PR model and a local tokenizer fallback."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

PR_HEAD = "6fb10761eb7fd6814210a0122e6f4d3b4037eee1"


def register_model() -> None:
    """Register the pinned implementation under mlx-lm's architecture namespace."""
    name = "mlx_lm.models.dory"
    models = importlib.import_module("mlx_lm.models")
    # Use Python's per-module import lock, not just a private registrar lock:
    # ordinary importers must also wait until execution completes. The model's
    # dataclass needs its module in sys.modules while its body executes.
    with importlib._bootstrap._ModuleLockManager(name):
        if name in sys.modules:
            return
        path = Path(__file__).with_name("dory_model.py")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        spec._initializing = True
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            models.dory = module
        except BaseException:
            if sys.modules.get(name) is module:
                del sys.modules[name]
            raise
        finally:
            spec._initializing = False


def _known_validation_error(exc: Exception, model_path: Path) -> bool:
    try:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    try:
        from huggingface_hub.errors import StrictDataclassClassValidationError
    except ImportError:
        error_types = (ValueError,)
    else:
        error_types = (ValueError, StrictDataclassClassValidationError)
    text = str(exc)
    return (
        isinstance(config, dict)
        and config.get("model_type") == "dory"
        and isinstance(exc, error_types)
        and "layer_types" in text
        and "entries must be in" in text
        and "'mlp'" in text
        and (model_path / "tokenizer.json").is_file()
    )


def load_tokenizer(model_path, tokenizer_config_extra=None, eos_token_ids=None):
    """Use normal mlx-lm tokenization, retrying only the known Dory failure.

    On fallback, reproduce mlx-lm's wrapper construction so detokenization,
    template/tool inference, and the model's EOS IDs remain intact. No global
    tokenizer classes or dependency files are modified.
    """
    from mlx_lm import tokenizer_utils as tu
    from mlx_lm import utils
    from transformers import PreTrainedTokenizerFast

    model_path = Path(model_path)
    try:
        return utils.load_tokenizer(model_path, tokenizer_config_extra, eos_token_ids)
    except Exception as exc:
        if not _known_validation_error(exc, model_path):
            raise

    # Match mlx-lm's decoder selection and parsing, including malformed JSON errors.
    from functools import partial

    with (model_path / "tokenizer.json").open(encoding="utf-8") as stream:
        content = json.load(stream)
    detokenizer = tu.NaiveStreamingDetokenizer
    decoder = content.get("decoder")
    if decoder is not None:
        if tu._is_spm_decoder(decoder):
            detokenizer = tu.SPMStreamingDetokenizer
        elif tu._is_spm_decoder_no_space(decoder):
            detokenizer = partial(tu.SPMStreamingDetokenizer, trim_space=False)
        elif tu._is_bpe_decoder(decoder):
            detokenizer = tu.BPEStreamingDetokenizer

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path, **(tokenizer_config_extra or {})
    )
    defaults = utils._tokenizer_defaults(model_path) or {}
    if eos_token_ids is None:
        eos_token_ids = defaults.get("eos_token_ids")
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    config = tokenizer.init_kwargs
    template = None
    response_format = None
    template_type = config.get("chat_template_type", defaults.get("chat_template_type"))
    if template_type:
        mod = importlib.import_module(f"mlx_lm.chat_templates.{template_type}")
        if "chat_template_type" in config or tokenizer.chat_template is None:
            template = mod.apply_chat_template
        response_format = getattr(mod, "RESPONSE_FORMAT", None)
    parser_type = config.get("tool_parser_type", defaults.get("tool_parser_type"))
    if parser_type is None:
        parser_type = tu._infer_tool_parser(tokenizer)
    parser = start = end = None
    if parser_type is not None:
        mod = importlib.import_module(f"mlx_lm.tool_parsers.{parser_type}")
        parser, start, end = mod.parse_tool_call, mod.tool_call_start, mod.tool_call_end
        config["tool_parser_type"] = parser_type
    return tu.TokenizerWrapper(
        tokenizer, detokenizer, eos_token_ids=eos_token_ids, chat_template=template,
        tool_parser=parser, tool_call_start=start, tool_call_end=end,
        response_format=response_format,
    )


def load(
    path_or_repo, *, tokenizer_config=None, model_config=None, adapter_path=None,
    lazy=False, return_config=False, revision=None, trust_remote_code=False,
):
    """Mirror mlx-lm's text load for Dory, with only its tokenizer substituted."""
    from mlx_lm import utils

    model_path = utils._download(path_or_repo, revision=revision)
    model, config = utils.load_model(
        model_path, lazy, model_config=model_config, trust_remote_code=trust_remote_code
    )
    if adapter_path is not None:
        model = utils.load_adapters(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(model_path, tokenizer_config, config.get("eos_token_id"))
    return (model, tokenizer, config) if return_config else (model, tokenizer)
