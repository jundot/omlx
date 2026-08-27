import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config, config)
        self.vision_model = (
            VisionModel(config.vision_config)
            if config.vision_config is not None
            else None
        )

    def encode_image(
        self,
        pixel_values: mx.array,
        image_grid_thw: mx.array | None = None,
        **kwargs,
    ) -> mx.array:
        if self.vision_model is None:
            raise ValueError("Vision inputs were provided, but vision_config is None.")
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required to encode GLM-5.3 images.")
        dtype = self.vision_model.patch_embed.proj.weight.dtype
        return self.vision_model(pixel_values.astype(dtype), image_grid_thw)

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id,
        video_token_id,
        image_features,
        inputs_embeds,
        input_ids,
    ):
        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

        batch_size, seq_len = input_ids.shape
        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(batch_size):
            image_mask = image_positions[batch_idx]
            num_positions = mx.sum(image_mask).item()

            if num_positions > 0:
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]
                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        f"Number of image token positions ({num_positions}) does not match "
                        f"number of image features ({batch_features.shape[0]}) for batch {batch_idx}"
                    )
                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(image_mask, cumsum - 1, 0)
                gathered_features = batch_features[feature_indices]
                image_mask_expanded = mx.expand_dims(image_mask, axis=-1)
                batch_output = mx.where(
                    image_mask_expanded, gathered_features, inputs_embeds[batch_idx]
                )
                feature_start_idx += num_positions
            else:
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

        return mx.stack(batch_outputs, axis=0)

    def get_input_embeddings(
        self,
        input_ids: mx.array | None = None,
        pixel_values: mx.array | None = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        if self.vision_model is None:
            raise ValueError("Vision inputs were provided, but vision_config is None.")

        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        hidden_states = kwargs.get("cached_image_features")
        if hidden_states is None:
            hidden_states = self.encode_image(
                pixel_values,
                image_grid_thw=grid_thw,
            )

        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            mask=mask,
            cache=cache,
        )

    def sanitize(self, weights):
        # HF container: Glm5NextForConditionalGeneration -> model.{visual,language_model} + lm_head
        remapped = {}
        for k, v in weights.items():
            nk = k
            if nk.startswith("model.visual."):
                nk = "vision_model." + nk[len("model.visual.") :]
            elif nk.startswith("visual."):
                nk = "vision_model." + nk[len("visual.") :]
            elif nk.startswith("model.language_model."):
                nk = "language_model.model." + nk[len("model.language_model.") :]
            elif nk.startswith("lm_head."):
                nk = "language_model." + nk
            remapped[nk] = v

        lang, vis, other = {}, {}, {}
        for k, v in remapped.items():
            if k.startswith("language_model."):
                lang[k[len("language_model.") :]] = v
            elif k.startswith("vision_model."):
                vis[k[len("vision_model.") :]] = v
            else:
                other[k] = v

        lang = self.language_model.sanitize(lang)
        if self.vision_model is not None:
            vis = self.vision_model.sanitize(vis)
        else:
            vis = VisionModel.sanitize(vis)

        out = {f"language_model.{k}": v for k, v in lang.items()}
        out.update({f"vision_model.{k}": v for k, v in vis.items()})
        out.update(other)
        return out

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    def make_cache(self):
        return self.language_model.make_cache()
