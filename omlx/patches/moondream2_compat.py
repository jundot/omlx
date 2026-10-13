# SPDX-License-Identifier: Apache-2.0
"""Preserve Moondream2 checkpoint and local tokenizer compatibility."""

import json
from functools import wraps
from pathlib import Path

import numpy as np


def _legacy_weight_keys(weights):
    remapped = {}
    for key, value in weights.items():
        if key.startswith("region_model."):
            continue
        if key.startswith("vision_encoder.encoder.model.visual."):
            key = "vision.encoder." + key[len("vision_encoder.encoder.model.visual.") :]
            key = key.replace("patch_embed.linear.", "patch_emb.")
            key = key.replace("pos_embed", "pos_emb")
            key = key.replace(".norm1.", ".ln1.").replace(".norm2.", ".ln2.")
            key = key.replace("norm.", "post_ln.")
        elif key.startswith("vision_encoder.projection.mlp."):
            key = "vision.proj_mlp." + key[len("vision_encoder.projection.mlp.") :]
        elif key == "text_model.transformer.embd.wte.weight":
            key = "text.model.embed_tokens.weight"
        elif key.startswith("text_model.transformer.h."):
            key = "text.model.layers." + key[len("text_model.transformer.h.") :]
            key = key.replace(".mixer.Wqkv.", ".attn.qkv.")
            key = key.replace(".mixer.out_proj.", ".attn.proj.")
        elif key.startswith("text_model.lm_head.ln."):
            key = "text.model.post_ln." + key[len("text_model.lm_head.ln.") :]
        elif key.startswith("text_model.lm_head.linear."):
            key = "text.lm_head." + key[len("text_model.lm_head.linear.") :]
        remapped[key] = value
    return remapped


def _is_legacy_phi_checkpoint(model_path):
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config") or {}
    # The 2024 revisions keep the Phi text model and use their bundled tokenizer.
    return (
        config.get("architectures") == ["Moondream"]
        or text_config.get("model_type") == "phi"
    )


def _has_local_starmie(model_path):
    from tokenizers import Tokenizer

    tokenizer_path = Path(model_path) / "tokenizer.json"
    if not tokenizer_path.is_file():
        return False
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # Current checkpoints can bundle a stale GPT-2 tokenizer.
    return all(
        tokenizer.token_to_id(token) == token_id
        for token, token_id in (
            ("<|endoftext|>", 0),
            ("<|md_reserved_2|>", 3),
            ("<|md_reserved_3|>", 4),
        )
    )


def apply_moondream2_compat_patch() -> bool:
    from mlx_vlm.models.base import load_chat_template
    from mlx_vlm.models.moondream2 import Model
    from mlx_vlm.models.moondream2.processing_moondream2 import (
        TOKENIZER_REPO,
        Moondream2Processor,
    )
    from mlx_vlm.models.moondream3.processing_moondream3 import NUM_VISION_TOKENS

    original_sanitize = Model.sanitize
    if getattr(original_sanitize, "_omlx_moondream2_compat", False):
        return False

    @wraps(original_sanitize)
    def sanitize(self, weights):
        return original_sanitize(self, _legacy_weight_keys(weights))

    class LegacyMoondream2Processor(Moondream2Processor):
        """Prompt encoder for the 2024 Phi-based revisions.

        Those checkpoints prompt with ``<image>\\n\\nQuestion: ...\\n\\nAnswer:``
        and have no answer marker token.
        """

        def __call__(
            self,
            text=None,
            images=None,
            return_tensors="np",
            padding=True,
            add_special_tokens=True,
            **kwargs,
        ):
            result = {}
            has_images = images is not None and (
                not hasattr(images, "__len__") or len(images) > 0
            )
            if has_images:
                result.update(
                    super().__call__(
                        images=images, return_tensors=return_tensors, **kwargs
                    )
                )
            if text is None:
                return result
            if isinstance(text, str):
                text = [text]
            bos_id = self.tokenizer.bos_token_id
            sequences = []
            for prompt in text:
                if has_images:
                    tokens = self.tokenizer.encode(
                        f"\n\nQuestion: {prompt}\n\nAnswer:", add_special_tokens=False
                    )
                    sequences.append([bos_id] + [0] * NUM_VISION_TOKENS + tokens)
                else:
                    tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                    sequences.append(([bos_id] if add_special_tokens else []) + tokens)
            width = max(len(ids) for ids in sequences)
            pad_id = self.tokenizer.pad_token_id or 0
            result["input_ids"] = np.array(
                [[pad_id] * (width - len(ids)) + ids for ids in sequences],
                dtype=np.int32,
            )
            result["attention_mask"] = np.array(
                [[0] * (width - len(ids)) + [1] * len(ids) for ids in sequences],
                dtype=np.int32,
            )
            return result

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        from transformers import AutoTokenizer

        tokenizer_kwargs = {
            key: kwargs[key]
            for key in (
                "cache_dir",
                "force_download",
                "local_files_only",
                "token",
                "trust_remote_code",
            )
            if key in kwargs
        }
        legacy = _is_legacy_phi_checkpoint(model_path)
        tokenizer_source = TOKENIZER_REPO
        if legacy or _has_local_starmie(model_path):
            tokenizer_source = model_path
            if "revision" in kwargs:
                tokenizer_kwargs["revision"] = kwargs["revision"]
        # A model revision does not identify a revision of the tokenizer repo.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        load_chat_template(tokenizer, model_path)
        if legacy:
            return LegacyMoondream2Processor(tokenizer=tokenizer)
        return cls(tokenizer=tokenizer)

    sanitize._omlx_moondream2_compat = True
    Model.sanitize = sanitize
    Moondream2Processor.from_pretrained = from_pretrained
    return True
