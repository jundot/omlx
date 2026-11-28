from __future__ import annotations

from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from ..qwen3_5.language import LanguageModel as Qwen3_5DenseLanguageModel
from ..qwen3_5_moe.language import LanguageModel as Qwen3_5MoeLanguageModel
from ..qwen3_5_moe.qwen3_5_moe import Model as Qwen3_5MoeModel
from ..qwen3_5_moe.vision import VisionModel as Qwen3_5MoeVisionModel
from ..qwen3_vl.qwen3_vl import masked_scatter
from .audio import WeMMAudioTower
from .config import ModelConfig

QFORMER_KEY_REPLACEMENTS = (
    (".attention.self.query.", ".attention.query."),
    (".attention.self.key.", ".attention.key."),
    (".attention.self.value.", ".attention.value."),
    (".attention.output.dense.", ".attention.dense."),
    (".attention.output.LayerNorm.gamma", ".attention.LayerNorm.weight"),
    (".attention.output.LayerNorm.beta", ".attention.LayerNorm.bias"),
    (".attention.output.LayerNorm.weight", ".attention.LayerNorm.weight"),
    (".attention.output.LayerNorm.bias", ".attention.LayerNorm.bias"),
    (".crossattention.self.query.", ".crossattention.query."),
    (".crossattention.self.key.", ".crossattention.key."),
    (".crossattention.self.value.", ".crossattention.value."),
    (".crossattention.output.dense.", ".crossattention.dense."),
    (".crossattention.output.LayerNorm.gamma", ".crossattention.LayerNorm.weight"),
    (".crossattention.output.LayerNorm.beta", ".crossattention.LayerNorm.bias"),
    (".crossattention.output.LayerNorm.weight", ".crossattention.LayerNorm.weight"),
    (".crossattention.output.LayerNorm.bias", ".crossattention.LayerNorm.bias"),
    (".intermediate_query.dense.", ".intermediate_dense."),
    (".output_query.dense.", ".output_dense."),
    (".output_query.LayerNorm.gamma", ".output_LayerNorm.weight"),
    (".output_query.LayerNorm.beta", ".output_LayerNorm.bias"),
    (".output_query.LayerNorm.weight", ".output_LayerNorm.weight"),
    (".output_query.LayerNorm.bias", ".output_LayerNorm.bias"),
)

NORM_SHIFT_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "language_model.model.norm.weight",
    "model.language_model.norm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


class VisionModel(Qwen3_5MoeVisionModel):
    pass


def _compute_wemm_rope_index(
    config: ModelConfig,
    input_ids: mx.array,
    image_grid_thw: Optional[mx.array] = None,
    video_grid_thw: Optional[mx.array] = None,
    attention_mask: Optional[mx.array] = None,
):
    if input_ids is None:
        raise ValueError("input_ids is required for get_rope_index")
    batch_size, seq_length = input_ids.shape
    if image_grid_thw is None and video_grid_thw is None:
        if attention_mask is not None:
            pos = mx.cumsum(attention_mask.astype(mx.int32), axis=-1) - 1
            pos = mx.where(attention_mask == 0, mx.ones_like(pos), pos)
            position_ids = mx.broadcast_to(pos[None, :, :], (3, batch_size, seq_length))
            max_pos = pos.max(axis=-1, keepdims=True)
            deltas = max_pos + 1 - seq_length
        else:
            pos = mx.arange(seq_length, dtype=mx.int32)[None, :]
            pos = mx.broadcast_to(pos, (batch_size, seq_length))
            position_ids = mx.broadcast_to(pos[None, :, :], (3, batch_size, seq_length))
            deltas = mx.zeros((batch_size, 1), dtype=mx.int32)
        return position_ids, deltas

    expanded_video_grids = None
    if video_grid_thw is not None:
        rows = []
        for thw in video_grid_thw.tolist():
            rows.extend([[1, int(thw[1]), int(thw[2])]] * int(thw[0]))
        expanded_video_grids = rows
    image_grids = image_grid_thw.tolist() if image_grid_thw is not None else None

    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    video_token_id = config.video_token_id
    vision_start_token_id = config.vision_start_token_id

    if attention_mask is None:
        attention_mask = mx.ones_like(input_ids)

    all_positions = []
    mrope_position_deltas = []
    image_index = 0
    video_index = 0

    for b in range(batch_size):
        row_mask = attention_mask[b].tolist()
        row_tokens = input_ids[b].tolist()
        active_tokens = [tok for tok, keep in zip(row_tokens, row_mask) if keep == 1]

        vision_next_tokens = [
            active_tokens[idx + 1]
            for idx in range(len(active_tokens) - 1)
            if active_tokens[idx] == vision_start_token_id
        ]
        remain_images = sum(t == image_token_id for t in vision_next_tokens)
        remain_videos = sum(t == video_token_id for t in vision_next_tokens)

        pos_chunks = []
        st = 0
        for _ in range(remain_images + remain_videos):
            ed_image = (
                active_tokens.index(image_token_id, st)
                if remain_images > 0 and image_token_id in active_tokens[st:]
                else len(active_tokens) + 1
            )
            ed_video = (
                active_tokens.index(video_token_id, st)
                if remain_videos > 0 and video_token_id in active_tokens[st:]
                else len(active_tokens) + 1
            )
            if ed_image < ed_video:
                t, h, w = image_grids[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = expanded_video_grids[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = int(t)
            llm_grid_h = int(h) // spatial_merge_size
            llm_grid_w = int(w) // spatial_merge_size

            text_len = ed - st
            st_idx = int(pos_chunks[-1].max().item()) + 1 if pos_chunks else 0
            if text_len > 0:
                txt = mx.arange(text_len, dtype=mx.int32)[None, :] + st_idx
                pos_chunks.append(mx.broadcast_to(txt, (3, text_len)))

            t_idx = mx.broadcast_to(
                mx.arange(llm_grid_t, dtype=mx.int32)[:, None],
                (llm_grid_t, llm_grid_h * llm_grid_w),
            ).reshape(-1)
            h_idx = mx.broadcast_to(
                mx.arange(llm_grid_h, dtype=mx.int32)[None, :, None],
                (llm_grid_t, llm_grid_h, llm_grid_w),
            ).reshape(-1)
            w_idx = mx.broadcast_to(
                mx.arange(llm_grid_w, dtype=mx.int32)[None, None, :],
                (llm_grid_t, llm_grid_h, llm_grid_w),
            ).reshape(-1)
            grid_pos = mx.stack([t_idx, h_idx, w_idx], axis=0) + (text_len + st_idx)
            pos_chunks.append(grid_pos)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(active_tokens):
            st_idx = int(pos_chunks[-1].max().item()) + 1 if pos_chunks else 0
            tail_len = len(active_tokens) - st
            tail = mx.arange(tail_len, dtype=mx.int32)[None, :] + st_idx
            pos_chunks.append(mx.broadcast_to(tail, (3, tail_len)))

        compact = mx.concatenate(pos_chunks, axis=1) if pos_chunks else mx.zeros((3, 0), dtype=mx.int32)
        max_pos = int(compact.max().item()) if compact.size > 0 else -1
        mrope_position_deltas.append(max_pos + 1 - len(active_tokens))

        if len(active_tokens) == seq_length:
            all_positions.append(compact)
        else:
            padded = [[1] * seq_length for _ in range(3)]
            compact_list = compact.tolist()
            c_idx = 0
            for col, keep in enumerate(row_mask):
                if keep == 1:
                    for dim in range(3):
                        padded[dim][col] = compact_list[dim][c_idx]
                    c_idx += 1
            all_positions.append(mx.array(padded, dtype=mx.int32))

    position_ids = mx.stack(all_positions, axis=1)
    deltas = mx.array(mrope_position_deltas, dtype=mx.int32).reshape(batch_size, 1)
    return position_ids, deltas


class LanguageModel(Qwen3_5MoeLanguageModel):
    """Qwen3.5-MoE language model with WeMM timestamped video+audio 3D M-RoPE."""

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        return _compute_wemm_rope_index(
            self.config, input_ids, image_grid_thw, video_grid_thw, attention_mask
        )


class DenseLanguageModel(Qwen3_5DenseLanguageModel):
    """Qwen3.5 dense language model with WeMM timestamped video+audio 3D M-RoPE."""

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        return _compute_wemm_rope_index(
            self.config, input_ids, image_grid_thw, video_grid_thw, attention_mask
        )


def sanitize_wemm_key(key: str) -> str:
    if key.startswith("model.language_model."):
        return key.replace("model.language_model.", "language_model.model.", 1)
    if key.startswith("lm_head."):
        return key.replace("lm_head.", "language_model.lm_head.", 1)
    if key.startswith("model.visual."):
        return key.replace("model.visual.", "vision_tower.", 1)
    if key.startswith("model.audio.") or key.startswith("audio_tower."):
        if key.startswith("model.audio."):
            key = key.replace("model.audio.", "audio_tower.", 1)
        key = key.replace(
            "audio_tower.qformer.bert.embeddings.LayerNorm.gamma",
            "audio_tower.qformer_embeddings_ln.weight",
        )
        key = key.replace(
            "audio_tower.qformer.bert.embeddings.LayerNorm.beta",
            "audio_tower.qformer_embeddings_ln.bias",
        )
        key = key.replace(
            "audio_tower.qformer.bert.embeddings.LayerNorm.weight",
            "audio_tower.qformer_embeddings_ln.weight",
        )
        key = key.replace(
            "audio_tower.qformer.bert.embeddings.LayerNorm.bias",
            "audio_tower.qformer_embeddings_ln.bias",
        )
        key = key.replace(
            "audio_tower.qformer.bert.encoder.layer.",
            "audio_tower.qformer_layers.",
        )
        for old_s, new_s in QFORMER_KEY_REPLACEMENTS:
            key = key.replace(old_s, new_s)
    return key


class Model(Qwen3_5MoeModel):
    """WeMM-Video Omni-Modal Model (Vision + Audio + Qwen3.5 Dense or MoE Backbone)."""

    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.audio_tower = (
            WeMMAudioTower(config.audio_config)
            if config.audio_config is not None
            else None
        )
        is_moe = int(getattr(config.text_config, "num_experts", 0) or 0) > 0
        self.language_model = (
            LanguageModel(config.text_config, config)
            if is_moe
            else DenseLanguageModel(config.text_config, config)
        )

    def _scatter_modality(
        self,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        features: mx.array,
        token_id: int,
        modality_name: str,
    ) -> mx.array:
        token_mask = input_ids == token_id
        n_tokens = int(token_mask.sum().item())
        n_features = int(features.shape[0])
        if n_tokens != n_features:
            raise ValueError(
                f"{modality_name} features and placeholder tokens do not match: "
                f"tokens={n_tokens}, features={n_features}"
            )
        expanded_mask = mx.broadcast_to(token_mask[..., None], inputs_embeds.shape)
        return masked_scatter(inputs_embeds, expanded_mask, features.astype(inputs_embeds.dtype))

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        pixel_values_videos = kwargs.get("pixel_values_videos", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        audio_feature = kwargs.get("audio_feature", None)
        if audio_feature is None:
            audio_feature = kwargs.get("input_features", None)
        mask = kwargs.get("mask", None)
        if mask is None:
            mask = kwargs.get("attention_mask", None)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        has_images = pixel_values is not None and image_grid_thw is not None
        has_videos = pixel_values_videos is not None and video_grid_thw is not None
        if pixel_values is not None and image_grid_thw is None and video_grid_thw is not None:
            pixel_values_videos = pixel_values
            pixel_values = None
            has_videos = True

        if has_images:
            cached = kwargs.get("cached_image_features", None)
            if cached is not None:
                img_hidden = cached
            else:
                vis_dtype = self.vision_tower.patch_embed.proj.weight.dtype
                img_hidden, _ = self.vision_tower(pixel_values.astype(vis_dtype), image_grid_thw)
            inputs_embeds = self._scatter_modality(
                inputs_embeds,
                input_ids,
                img_hidden,
                self.config.image_token_id,
                "Image",
            )

        if has_videos:
            cached_vid = kwargs.get("cached_image_features", None) if not has_images else None
            if cached_vid is not None:
                vid_hidden = cached_vid
            else:
                vis_dtype = self.vision_tower.patch_embed.proj.weight.dtype
                vid_hidden, _ = self.vision_tower(
                    pixel_values_videos.astype(vis_dtype), video_grid_thw
                )
            inputs_embeds = self._scatter_modality(
                inputs_embeds,
                input_ids,
                vid_hidden,
                self.config.video_token_id,
                "Video",
            )

        if audio_feature is not None:
            if self.audio_tower is None:
                raise ValueError("audio_feature provided but model has no audio_tower")
            audio_hidden = self.audio_tower(audio_feature)
            audio_flat = audio_hidden.reshape(-1, audio_hidden.shape[-1])
            inputs_embeds = self._scatter_modality(
                inputs_embeds,
                input_ids,
                audio_flat,
                self.config.audio_token_id,
                "Audio",
            )

        position_ids, rope_deltas = self.language_model.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=mask,
        )
        self.language_model._position_ids = position_ids
        self.language_model._rope_deltas = rope_deltas

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        weights = {
            k: v
            for k, v in weights.items()
            if "mtp." not in k and not k.endswith("embeddings.position_ids")
        }

        has_raw_hf_keys = any(
            k.startswith("model.language_model.")
            or k.startswith("model.audio.")
            or ("conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1)
            for k, v in weights.items()
        )

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
            weights.pop("language_model.lm_head.weight", None)

        num_layers = self.config.text_config.num_hidden_layers
        num_experts = int(getattr(self.config.text_config, "num_experts", 0) or 0)
        if num_experts > 0:
            for layer_idx in range(num_layers):
                for base_prefix in (
                    f"model.language_model.layers.{layer_idx}.mlp",
                    f"language_model.model.layers.{layer_idx}.mlp",
                ):
                    if f"{base_prefix}.switch_mlp.gate_proj.weight" in weights:
                        continue
                    if f"{base_prefix}.experts.gate_up_proj" in weights:
                        gate_up = weights.pop(f"{base_prefix}.experts.gate_up_proj")
                        gate_w, up_w = mx.split(gate_up, 2, axis=-2)
                        weights[f"{base_prefix}.switch_mlp.gate_proj.weight"] = gate_w
                        weights[f"{base_prefix}.switch_mlp.up_proj.weight"] = up_w
                        if f"{base_prefix}.experts.down_proj" in weights:
                            weights[f"{base_prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                                f"{base_prefix}.experts.down_proj"
                            )
                    elif f"{base_prefix}.experts.0.gate_proj.weight" in weights:
                        for proj in ("gate_proj", "up_proj", "down_proj"):
                            for suffix in ("weight", "scales", "biases"):
                                if f"{base_prefix}.experts.0.{proj}.{suffix}" not in weights:
                                    continue
                                weights[f"{base_prefix}.switch_mlp.{proj}.{suffix}"] = mx.stack(
                                    [
                                        weights.pop(f"{base_prefix}.experts.{e}.{proj}.{suffix}")
                                        for e in range(num_experts)
                                    ],
                                    axis=0,
                                )

        sanitized: Dict[str, mx.array] = {}
        for orig_key, value in weights.items():
            key = sanitize_wemm_key(orig_key)

            if key.endswith("linear_attn.conv1d.weight") and value.ndim == 3 and value.shape[-1] != 1:
                value = mx.moveaxis(value, 2, 1)
            elif key in ("audio_tower.conv1.weight", "audio_tower.conv2.weight") and value.ndim == 3:
                if value.shape[-1] == 3 and value.shape[1] != 3:
                    value = mx.moveaxis(value, 2, 1)
            elif key.endswith("patch_embed.proj.weight") and value.ndim == 5:
                in_ch = self.config.vision_config.in_channels
                if value.shape[-1] != in_ch and value.shape[1] == in_ch:
                    value = value.transpose(0, 2, 3, 4, 1)

            if has_raw_hf_keys and orig_key.startswith("model.language_model."):
                if any(key.endswith(sfx) for sfx in NORM_SHIFT_SUFFIXES) and value.ndim == 1:
                    value = value + 1.0

            sanitized[key] = value

        return sanitized

    @property
    def quant_predicate(self):
        def predicate(path: str, module: nn.Module):
            if not hasattr(module, "to_quantized"):
                return False
            if path.startswith("vision_tower") or path.startswith("audio_tower"):
                return False
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate
