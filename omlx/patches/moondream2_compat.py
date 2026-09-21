# SPDX-License-Identifier: Apache-2.0
"""Preserve Moondream2 checkpoint and local tokenizer compatibility."""

from functools import wraps
from pathlib import Path


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

    original_sanitize = Model.sanitize
    if getattr(original_sanitize, "_omlx_moondream2_compat", False):
        return False

    @wraps(original_sanitize)
    def sanitize(self, weights):
        return original_sanitize(self, _legacy_weight_keys(weights))

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
        tokenizer_source = TOKENIZER_REPO
        if _has_local_starmie(model_path):
            tokenizer_source = model_path
            if "revision" in kwargs:
                tokenizer_kwargs["revision"] = kwargs["revision"]
        # A model revision does not identify a revision of the tokenizer repo.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        load_chat_template(tokenizer, model_path)
        return cls(tokenizer=tokenizer)

    sanitize._omlx_moondream2_compat = True
    Model.sanitize = sanitize
    Moondream2Processor.from_pretrained = from_pretrained
    return True
