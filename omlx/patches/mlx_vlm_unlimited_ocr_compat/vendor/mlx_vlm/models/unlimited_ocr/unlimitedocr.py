from typing import Optional

import mlx.core as mx
import numpy as np
from transformers import AutoProcessor

from mlx_vlm.models.base import InputEmbeddingsFeatures

from ..deepseekocr.deepseekocr import MlpProjector
from ..deepseekocr.deepseekocr import Model as DeepseekOCRModel
from ..deepseekocr.sam import SAMEncoder
from ..deepseekocr.vision import VisionModel
from .config import ModelConfig
from .language import LanguageModel
from .processing_unlimitedocr import UnlimitedOCRHFProcessor, UnlimitedOCRProcessor

AutoProcessor.register("unlimited-ocr", UnlimitedOCRProcessor)


class Model(DeepseekOCRModel):
    config_class = ModelConfig

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.language_model = LanguageModel(config.text_config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        images_spatial_crop: Optional[mx.array] = None,
        images_seq_mask: Optional[mx.array] = None,
        **kwargs,
    ):
        input_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=input_embeds)

        if (
            self.sam_model is not None
            and input_ids.shape[1] != 1
            and mx.sum(pixel_values[1]).item() != 0
        ):
            idx = 0
            patch_idx = 0
            all_patches = pixel_values[0]
            all_image_ori = pixel_values[1]
            batch_size = input_embeds.shape[0]
            image_token_positions = [
                np.where(images_seq_mask[batch_idx])[0].tolist()
                for batch_idx in range(batch_size)
            ]
            image_token_offsets = [0] * batch_size
            single_prompt_multi_image = (
                batch_size == 1 and len(images_spatial_crop) != batch_size
            )

            for crop_shape in images_spatial_crop.tolist():
                images_in_this_batch = []
                width_crop_num, height_crop_num = int(crop_shape[0]), int(crop_shape[1])

                has_crops = width_crop_num > 1 or height_crop_num > 1
                num_patches = width_crop_num * height_crop_num if has_crops else 0

                if has_crops and num_patches > 0:
                    patches = all_patches[patch_idx : patch_idx + num_patches]
                    patch_idx += num_patches
                else:
                    patches = None

                image_ori = all_image_ori[idx : idx + 1]

                if patches is not None and mx.sum(patches).item() != 0:
                    local_features_1 = self.sam_model(patches.transpose(0, 2, 3, 1))
                    local_features_2 = self.vision_model(
                        patches.transpose(0, 2, 3, 1), patch_embeds=local_features_1
                    )
                    local_features = mx.concatenate(
                        (
                            local_features_2[:, 1:],
                            local_features_1.flatten(start_axis=1, end_axis=2),
                        ),
                        axis=-1,
                    )
                    local_features = self.projector(local_features)

                    global_features_1 = self.sam_model(image_ori.transpose(0, 2, 3, 1))
                    global_features_2 = self.vision_model(
                        image_ori.transpose(0, 2, 3, 1), global_features_1
                    )
                    global_features = mx.concatenate(
                        (
                            global_features_2[:, 1:],
                            global_features_1.flatten(start_axis=1, end_axis=2),
                        ),
                        axis=-1,
                    )
                    global_features = self.projector(global_features)[0]
                    hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)
                    global_features = global_features.reshape(h, w, n_dim)
                    global_features = mx.concatenate(
                        [
                            global_features,
                            mx.broadcast_to(
                                self.image_newline[None, None, :], (h, 1, n_dim)
                            ),
                        ],
                        axis=1,
                    ).reshape(-1, n_dim)

                    _, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2**0.5)
                    local_features = (
                        local_features.reshape(
                            height_crop_num, width_crop_num, h2, w2, n_dim2
                        )
                        .transpose(0, 2, 1, 3, 4)
                        .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                    )
                    local_features = mx.concatenate(
                        [
                            local_features,
                            mx.broadcast_to(
                                self.image_newline[None, None, :],
                                (height_crop_num * h2, 1, n_dim2),
                            ),
                        ],
                        axis=1,
                    ).reshape(-1, n_dim2)

                    global_local_features = mx.concatenate(
                        [local_features, global_features, self.view_separator[None, :]],
                        axis=0,
                    )
                else:
                    global_features_1 = self.sam_model(image_ori.transpose(0, 2, 3, 1))
                    global_features_2 = self.vision_model(
                        image_ori.transpose(0, 2, 3, 1), global_features_1
                    )
                    global_features = mx.concatenate(
                        (
                            global_features_2[:, 1:],
                            global_features_1.flatten(start_axis=1, end_axis=2),
                        ),
                        axis=-1,
                    )
                    global_features = self.projector(global_features)[0]
                    hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)
                    global_features = global_features.reshape(h, w, n_dim)
                    global_features = mx.concatenate(
                        [
                            global_features,
                            mx.broadcast_to(
                                self.image_newline[None, None, :], (h, 1, n_dim)
                            ),
                        ],
                        axis=1,
                    ).reshape(-1, n_dim)
                    global_local_features = mx.concatenate(
                        [global_features, self.view_separator[None, :]], axis=0
                    )

                images_in_this_batch.append(global_local_features)

                if images_in_this_batch:
                    images_in_this_batch = mx.concatenate(images_in_this_batch, axis=0)
                    batch_idx = 0 if single_prompt_multi_image else idx
                    image_indices = image_token_positions[batch_idx]
                    start = image_token_offsets[batch_idx]
                    end = start + images_in_this_batch.shape[0]
                    if end > len(image_indices):
                        raise ValueError(
                            "More image features were produced than image token "
                            "positions in the prompt."
                        )
                    input_embeds[batch_idx, image_indices[start:end]] = (
                        images_in_this_batch
                    )
                    image_token_offsets[batch_idx] = end

                idx += 1

        return InputEmbeddingsFeatures(inputs_embeds=input_embeds)

    def make_cache(self):
        return self.language_model.make_cache()


__all__ = [
    "LanguageModel",
    "MlpProjector",
    "Model",
    "SAMEncoder",
    "UnlimitedOCRHFProcessor",
    "UnlimitedOCRProcessor",
    "VisionModel",
]
