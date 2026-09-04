"""
Processor class for Mistral3.

Mistral3 uses the same processor architecture as Pixtral.
Adapted from HuggingFace Transformers.
"""

from typing import List, Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image, load_image
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

# omlx: vendored from mlx-vlm e3fec79; relative imports rewritten so this
# module lives outside the mlx_vlm package. Re-verify against upstream on
# every mlx-vlm pin bump.
from mlx_vlm.models.base import load_chat_template, to_mlx

from .image_processing_pixtral import (
    PixtralImageProcessor,
    split_image_sizes_by_sample,
)


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _load_mistral3_image_processor(pretrained_model_name_or_path, **kwargs):
    return PixtralImageProcessor.from_pretrained(
        pretrained_model_name_or_path, **kwargs
    )


class Mistral3Processor(ProcessorMixin):
    """Mistral3 processor, based on PixtralProcessor."""

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "patch_size", "spatial_merge_size"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 16,
        spatial_merge_size: int = 1,
        image_token: str = "[IMG]",
        image_break_token: str = "[IMG_BREAK]",
        image_end_token: str = "[IMG_END]",
        chat_template=None,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_token = image_token
        self.image_break_token = image_break_token
        self.image_end_token = image_end_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_break_token_id = tokenizer.convert_tokens_to_ids(
            self.image_break_token
        )
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)

        super().__init__(
            image_processor, tokenizer, chat_template=chat_template, **kwargs
        )

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                List[TextInput],
                List[PreTokenizedInput],
            ]
        ] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("You must provide either text or images.")

        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        image_inputs = {}
        if images is not None:
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
                images = [images]

            images = [
                [load_image(im) if is_url(im) else im for im in sample]
                for sample in images
            ]

            image_inputs = self.image_processor(
                images, patch_size=self.patch_size * self.spatial_merge_size
            )

            if text is not None:
                image_sizes_by_sample = split_image_sizes_by_sample(
                    image_inputs.get("image_sizes", []), images
                )
                prompt_strings = []
                for batch_idx, sample in enumerate(text):
                    if self.image_token in sample:
                        sample_sizes = (
                            image_sizes_by_sample[batch_idx]
                            if batch_idx < len(image_sizes_by_sample)
                            else []
                        )
                        parts = sample.split(self.image_token)
                        new_sample = parts[0]
                        for img_idx in range(len(parts) - 1):
                            if img_idx < len(sample_sizes):
                                h, w = sample_sizes[img_idx]
                                num_h = h // (self.patch_size * self.spatial_merge_size)
                                num_w = w // (self.patch_size * self.spatial_merge_size)
                                img_tokens = ""
                                for row in range(num_h):
                                    img_tokens += self.image_token * num_w
                                    if row < num_h - 1:
                                        img_tokens += self.image_break_token
                                img_tokens += self.image_end_token
                                new_sample += img_tokens
                            else:
                                new_sample += self.image_token
                            new_sample += parts[img_idx + 1]
                        prompt_strings.append(new_sample)
                    else:
                        prompt_strings.append(sample)
                text = prompt_strings

        kwargs.pop("return_tensors", None)

        if text is not None:
            text_inputs = self.tokenizer(text, **kwargs)
            data = {**text_inputs, **image_inputs}
        else:
            data = image_inputs

        return BatchFeature(data=to_mlx(data))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names + image_processor_input_names + ["image_sizes"]
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        kwargs.pop("use_fast", None)
        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()
        resolved_model_path = (
            str(model_path) if is_local else pretrained_model_name_or_path
        )

        # omlx: transformers 5.12 prefers the vocab-less MistralCommonBackend
        # for repos shipping tekken.json; the kwarg's presence opts back into
        # TokenizersBackend, which the streaming detokenizers require. Applied
        # to the tokenizer only so it never leaks into the image processor.
        tokenizer_kwargs = dict(kwargs)
        tokenizer_kwargs.setdefault("fix_mistral_regex", True)
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_path, **tokenizer_kwargs
        )
        load_chat_template(tokenizer, resolved_model_path)

        def _load_json_config(filename):
            try:
                path = (
                    model_path / filename
                    if is_local
                    else Path(hf_hub_download(pretrained_model_name_or_path, filename))
                )
            except Exception:
                return {}

            if not path.exists():
                return {}

            with open(path, encoding="utf-8") as f:
                return json.load(f)

        proc_cfg = _load_json_config("processor_config.json")
        model_cfg = _load_json_config("config.json")
        preproc_cfg = _load_json_config("preprocessor_config.json")

        proc_kwargs = {}
        for k in ("image_token", "image_break_token", "image_end_token"):
            if k in proc_cfg:
                proc_kwargs[k] = proc_cfg[k]

        patch_size = model_cfg.get("vision_config", {}).get("patch_size")
        if patch_size is None:
            patch_size = preproc_cfg.get("patch_size")
            if isinstance(patch_size, dict):
                patch_size = patch_size.get("height") or patch_size.get("width")
        if patch_size is None:
            patch_size = proc_cfg.get("patch_size")
        if patch_size is not None:
            proc_kwargs["patch_size"] = patch_size

        spatial_merge_size = model_cfg.get("spatial_merge_size")
        if spatial_merge_size is None:
            spatial_merge_size = proc_cfg.get("spatial_merge_size")
        if spatial_merge_size is not None:
            proc_kwargs["spatial_merge_size"] = spatial_merge_size

        image_processor = _load_mistral3_image_processor(
            resolved_model_path,
            **kwargs,
        )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **proc_kwargs,
        )


# omlx: upstream registers this class via install_auto_processor_patch here.
# The omlx installer transplants methods onto the pinned classes instead, so
# the upstream registration (already installed at model-module import) keeps
# routing while the patched bodies take effect regardless of import order.
__all__ = ["Mistral3Processor"]
