import json
import math
from pathlib import Path
from typing import List

import mlx.core as mx
from PIL import Image, ImageOps
from transformers import PreTrainedTokenizerFast

from ..base import install_auto_processor_patch
from ..deepseekocr.processing_deepseekocr import (
    DeepseekOCRProcessor,
    dynamic_preprocess,
)


class UnlimitedOCRProcessor(DeepseekOCRProcessor):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("revision", None)
        kwargs.pop("force_download", None)
        kwargs.pop("quantize_activations", None)
        kwargs.pop("strict", None)
        path = Path(pretrained_model_name_or_path)
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(pretrained_model_name_or_path))

        processor_config = {}
        processor_config_path = path / "processor_config.json"
        if processor_config_path.exists():
            with open(processor_config_path, encoding="utf-8") as f:
                processor_config = json.load(f)

        processor_config.pop("processor_class", None)
        processor_config.update(kwargs)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
        return cls(tokenizer=tokenizer, **processor_config)

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        candidate_resolutions=((1024, 1024),),
        patch_size: int = 16,
        downsample_ratio: int = 4,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜▁pad▁｜>",
        add_special_token: bool = False,
        sft_format: str = "unlimitedocr",
        mask_prompt: bool = False,
        ignore_id: int = -100,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            candidate_resolutions=candidate_resolutions,
            patch_size=patch_size,
            downsample_ratio=downsample_ratio,
            image_mean=image_mean,
            image_std=image_std,
            normalize=normalize,
            image_token=image_token,
            pad_token=pad_token,
            add_special_token=add_special_token,
            sft_format=sft_format,
            mask_prompt=mask_prompt,
            ignore_id=ignore_id,
            **kwargs,
        )

        # Unlimited-OCR's released tokenizer already contains <image> at 128815.
        self.image_token_id = tokenizer.vocab.get(image_token, 128815)

    def process_one(
        self,
        prompt: str = None,
        images: List[Image.Image] = None,
        inference_mode: bool = True,
        base_size: int = 1024,
        image_size: int = 640,
        cropping: bool = True,
    ):
        (
            tokenized_str,
            images_list,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
        ) = self.tokenize_with_images(
            prompt,
            images,
            base_size=base_size,
            image_size=image_size,
            cropping=cropping,
        )

        masked_tokenized_str = [
            token_index if token_index != self.image_token_id else self.ignore_id
            for token_index in tokenized_str
        ]
        input_ids = mx.array(tokenized_str)
        target_ids = mx.array(masked_tokenized_str)
        images_seq_mask = mx.array(images_seq_mask)

        target_ids = mx.where(
            (input_ids < 0) | (input_ids == self.image_token_id),
            self.ignore_id,
            target_ids,
        )
        input_ids = mx.where(input_ids < 0, self.pad_id, input_ids)

        return {
            "input_ids": input_ids[None, :],
            "attention_mask": input_ids != self.pad_id,
            "labels": target_ids,
            "images": images_list,
            "images_seq_mask": images_seq_mask[None, ...],
            "images_spatial_crop": images_spatial_crop,
            "num_image_tokens": num_image_tokens,
        }

    def tokenize_with_images(
        self,
        conversation: str,
        images: List[Image.Image],
        base_size: int = 1024,
        image_size: int = 640,
        cropping: bool = True,
    ):
        patch_size = self.patch_size
        downsample_ratio = self.downsample_ratio
        valid_img_tokens = 0
        ratio = 1

        image_draw = images[0].copy()

        w, h = image_draw.size

        ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))

        image_token_count = conversation.count(self.image_token)
        assert (
            conversation.count(self.image_token) == len(images)
            or image_token_count == 1
        ), f"The number of image tokens in the prompt does not match the number of images: {image_token_count} != {len(images)}"
        text_splits = conversation.split(self.image_token)

        images_list, images_crop_list, images_seq_mask = [], [], []
        tokenized_str = []
        images_spatial_crop = []
        multi_image_single_token = image_token_count == 1 and len(images) > 1
        image_items = (
            [(text_splits[0], image) for image in images]
            if multi_image_single_token
            else list(zip(text_splits, images))
        )
        for image_idx, (text_sep, image) in enumerate(image_items):
            if multi_image_single_token and image_idx > 0:
                text_sep = ""
            tokenized_sep = self.encode(text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            if cropping:
                if image.size[0] <= 640 and image.size[1] <= 640:
                    crop_ratio = [1, 1]
                    images_crop_raw = []
                else:
                    images_crop_raw, crop_ratio = dynamic_preprocess(
                        image, min_num=2, max_num=32, image_size=image_size
                    )

                global_view = ImageOps.pad(
                    image,
                    (base_size, base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean),
                )

                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                elif base_size == 640:
                    valid_img_tokens += int(100 * ratio)

                images_list.append(
                    self.image_transform(global_view).astype(mx.bfloat16)
                )

                width_crop_num, height_crop_num = crop_ratio
                images_spatial_crop.append([width_crop_num, height_crop_num])

                num_image_crops = 0
                if width_crop_num > 1 or height_crop_num > 1:
                    num_image_crops = len(images_crop_raw)
                    for image_crop in images_crop_raw:
                        images_crop_list.append(
                            self.image_transform(image_crop).astype(mx.bfloat16)
                        )

                if image_size == 640:
                    valid_img_tokens += num_image_crops * 100

                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                num_queries_base = math.ceil(
                    (base_size // patch_size) / downsample_ratio
                )

                tokenized_image = (
                    [self.image_token_id] * num_queries_base + [self.image_token_id]
                ) * num_queries_base
                tokenized_image += [self.image_token_id]
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += (
                        [self.image_token_id] * (num_queries * width_crop_num)
                        + [self.image_token_id]
                    ) * (num_queries * height_crop_num)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)

            else:
                if image_size <= 640:
                    image = image.resize((image_size, image_size))

                global_view = ImageOps.pad(
                    image,
                    (image_size, image_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean),
                )
                images_list.append(
                    self.image_transform(global_view).astype(mx.bfloat16)
                )

                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                elif base_size == 640:
                    valid_img_tokens += int(100 * 1)
                elif base_size == 512:
                    valid_img_tokens += int(64 * 1)

                width_crop_num, height_crop_num = 1, 1
                images_spatial_crop.append([width_crop_num, height_crop_num])

                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                tokenized_image = (
                    [self.image_token_id] * num_queries + [self.image_token_id]
                ) * num_queries
                tokenized_image += [self.image_token_id]

                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)

        tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        bos_id = 0
        tokenized_str = [bos_id] + tokenized_str
        images_seq_mask = [False] + images_seq_mask

        images_seq_mask = mx.array(images_seq_mask)

        if len(images_list) == 0:
            images_ori = mx.zeros((1, 3, image_size, image_size))
            images_spatial_crop = mx.zeros((1, 2))
            images_crop = mx.zeros((1, 3, base_size, base_size))
        else:
            images_ori = mx.stack(images_list, axis=0)
            images_spatial_crop = mx.array(images_spatial_crop)
            if images_crop_list:
                images_crop = mx.stack(images_crop_list, axis=0)
            else:
                images_crop = mx.zeros((1, 3, base_size, base_size))

        assert len(tokenized_str) == len(
            images_seq_mask
        ), f"tokenize_with_images func: tokenized_str's length {len(tokenized_str)} is not equal to images_seq_mask's length {len(images_seq_mask)}"

        return (
            tokenized_str,
            [images_crop, images_ori],
            images_seq_mask,
            images_spatial_crop,
            valid_img_tokens,
        )


UnlimitedOCRHFProcessor = UnlimitedOCRProcessor

install_auto_processor_patch("unlimited-ocr", UnlimitedOCRProcessor)
