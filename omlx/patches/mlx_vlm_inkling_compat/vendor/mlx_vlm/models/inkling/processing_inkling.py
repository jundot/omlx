# Torch-free port of transformers' Inkling processors (transformers
# 5.14+, src/transformers/models/inkling/{image_processing,processing}_
# inkling.py, Apache-2.0). oMLX pins transformers <5.13, so the classes
# are vendored here on numpy/PIL following the mlx-vlm custom-processor
# pattern (see unlimited_ocr's processing_unlimitedocr.py).
#
# Differences from the transformers reference, on purpose:
# - PIL resampling instead of torchvision (bit-parity with torchvision
#   interpolation is not achievable either way; the checkpoint config
#   selects resample=3 / bicubic).
# - Audio (dMel feature extraction) is not implemented yet: oMLX serves
#   inkling text+vision first. Passing audio raises a clear error.
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizerFast


class InklingImageProcessor:
    """Patchifying image processor: divides an image into 40x40 patches
    (padding partial edges with -1.0 before rescale/normalize), duplicates
    each patch along a temporal axis of 2, and reports per-image patch
    counts for placeholder-token expansion.

    Output layout matches the reference exactly: ``pixel_values``
    [num_patches, T=2, H, W, C] float32, ``num_patches`` [num_images].
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size=None,
        image_mean=None,
        image_std=None,
        resample=3,
        rescale_factor=1.0 / 255.0,
        do_convert_rgb=True,
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        rescale_image_frac=None,
        rescale_image_max_upscaled_long_edge=2048,
        **kwargs,
    ):
        size = size or {"height": 40, "width": 40}
        if size["height"] != size["width"]:
            raise ValueError(f"Can resize only to square size but got {size}")
        self.size = size
        self.patch_size = int(size["height"])
        self.image_mean = np.asarray(
            image_mean or (0.48145466, 0.4578275, 0.40821073), dtype=np.float32
        )
        self.image_std = np.asarray(
            image_std or (0.26862954, 0.26130258, 0.27577711), dtype=np.float32
        )
        self.resample = int(resample)
        self.rescale_factor = float(rescale_factor)
        self.do_convert_rgb = bool(do_convert_rgb)
        self.do_rescale = bool(do_rescale)
        self.do_normalize = bool(do_normalize)
        self.rescale_image_frac = rescale_image_frac
        self.rescale_image_max_upscaled_long_edge = rescale_image_max_upscaled_long_edge

    def _pre_scale(self, image: Image.Image) -> Image.Image:
        if self.rescale_image_frac is None:
            return image
        width, height = image.size
        long_edge = max(height, width)
        target_long_edge = long_edge * self.rescale_image_frac
        if self.rescale_image_max_upscaled_long_edge is not None:
            target_long_edge = min(
                target_long_edge,
                max(self.rescale_image_max_upscaled_long_edge, long_edge),
            )
        ratio = target_long_edge / long_edge
        if ratio == 1.0:
            return image
        # Half-up rounding to match the reference.
        new_h = math.floor(height * ratio + 0.5)
        new_w = math.floor(width * ratio + 0.5)
        return image.resize((new_w, new_h), resample=self.resample)

    def _patchify(self, image: Image.Image) -> np.ndarray:
        """[H, W, C] -> [num_patches, P, P, C] float32 with -1.0 padding.

        Grid mirrors the reference ``divide_to_patches``: rows =
        ceil(H / P), cols = W // P + 1 (the extra column is intentional
        upstream behavior).
        """
        p = self.patch_size
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        height, width = arr.shape[:2]
        num_rows = (height + p - 1) // p
        num_cols = width // p + 1

        patches = []
        for i in range(num_rows):
            for j in range(num_cols):
                patch = arr[i * p : i * p + p, j * p : j * p + p]
                if patch.shape[0] != p or patch.shape[1] != p:
                    padded = np.full((p, p, arr.shape[-1]), -1.0, dtype=np.float32)
                    padded[: patch.shape[0], : patch.shape[1]] = patch
                    patch = padded
                patches.append(patch)
        return np.stack(patches, axis=0)

    def preprocess(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        per_image_patches = []
        num_patches = []
        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image))
            if self.do_convert_rgb:
                image = image.convert("RGB")
            image = self._pre_scale(image)
            patches = self._patchify(image)
            if self.do_rescale:
                patches = patches * self.rescale_factor
            if self.do_normalize:
                patches = (patches - self.image_mean) / self.image_std
            # Temporal duplication: [P, H, W, C] -> [P, T=2, H, W, C].
            patches = np.repeat(patches[:, None], 2, axis=1)
            per_image_patches.append(patches.astype(np.float32))
            num_patches.append(patches.shape[0])
        return {
            "pixel_values": np.concatenate(per_image_patches, axis=0),
            "num_patches": np.asarray(num_patches, dtype=np.int32),
        }

    def __call__(self, images, **kwargs):
        return self.preprocess(images)


class InklingProcessor:
    """Tokenizer + image-processor front end for Inkling checkpoints.

    The chat template renders one ``<|unused_200054|>`` placeholder per
    image; ``__call__`` expands the i-th occurrence to ``num_patches[i]``
    copies so ``masked_scatter`` in the model lines up with the vision
    tower's per-patch soft tokens.
    """

    def __init__(
        self,
        tokenizer,
        image_processor=None,
        image_token="<|unused_200054|>",
        audio_token="<|unused_200053|>",
        image_bos_token="<|content_image|>",
        audio_bos_token="<|content_audio_input|>",
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor or InklingImageProcessor()
        # No dMel feature extractor yet (text+vision first).
        self.feature_extractor = None
        self.image_token = image_token
        self.audio_token = audio_token
        self.image_bos_token = image_bos_token
        self.audio_bos_token = audio_bos_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        for ignored in (
            "trust_remote_code",
            "revision",
            "force_download",
            "quantize_activations",
            "strict",
            "use_fast",
        ):
            kwargs.pop(ignored, None)
        path = Path(pretrained_model_name_or_path)
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(pretrained_model_name_or_path)))

        processor_config = {}
        processor_config_path = path / "processor_config.json"
        if processor_config_path.exists():
            with open(processor_config_path, encoding="utf-8") as f:
                processor_config = json.load(f)

        image_processor = InklingImageProcessor(
            **processor_config.get("image_processor", {})
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
        _ensure_special_tokens(tokenizer, path)

        init_kwargs = {
            key: processor_config[key]
            for key in (
                "image_token",
                "audio_token",
                "image_bos_token",
                "audio_bos_token",
            )
            if key in processor_config
        }
        init_kwargs.update(kwargs)
        return cls(tokenizer=tokenizer, image_processor=image_processor, **init_kwargs)

    # --- transformers ProcessorMixin surface used by mlx-vlm / omlx ---

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        names = list(self.image_processor.model_input_names)
        names += list(self.tokenizer.model_input_names)
        return list(dict.fromkeys(names))

    def __call__(
        self,
        text=None,
        images=None,
        audio=None,
        audios=None,
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        return_tensors=None,
        **kwargs,
    ):
        if audio or audios:
            raise ValueError(
                "the omlx inkling processor does not support audio input yet "
                "(text+vision only)"
            )
        if text is None:
            raise ValueError("text is required")
        texts = [text] if isinstance(text, str) else list(text)

        outputs = {}
        if images is not None and (
            not hasattr(images, "__len__") or len(images) > 0
        ):
            image_outputs = self.image_processor.preprocess(images)
            outputs["pixel_values"] = image_outputs["pixel_values"]
            num_patches = [int(n) for n in image_outputs["num_patches"]]

            expanded = []
            image_cursor = 0
            for prompt in texts:
                pieces = prompt.split(self.image_token)
                needed = len(pieces) - 1
                if image_cursor + needed > len(num_patches):
                    raise ValueError(
                        f"prompt references {needed} images at offset "
                        f"{image_cursor} but only {len(num_patches)} were "
                        "provided"
                    )
                rebuilt = pieces[0]
                for piece in pieces[1:]:
                    rebuilt += (
                        self.image_token * num_patches[image_cursor] + piece
                    )
                    image_cursor += 1
                expanded.append(rebuilt)
            if image_cursor != len(num_patches):
                raise ValueError(
                    f"{len(num_patches)} images provided but prompts contain "
                    f"{image_cursor} image placeholders"
                )
            texts = expanded

        encoded = self.tokenizer(
            texts,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
            return_tensors="np",
        )
        outputs["input_ids"] = encoded["input_ids"]
        outputs["attention_mask"] = encoded["attention_mask"]
        return outputs


def _ensure_special_tokens(tokenizer, model_path: Path) -> None:
    """Give the tokenizer usable pad/eos handles.

    Inkling checkpoints expose neither a pad nor an eos token *string*
    (only ``eos_token_id`` in config.json), which breaks every
    ``tokenizer.pad_token`` fallback in mlx-vlm. Mirrors the fix in
    mlx-vlm PR #1756's ``load_processor``.
    """
    if tokenizer.pad_token is not None:
        return
    eos_ids = None
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                eos_ids = json.load(f).get("eos_token_id")
        except (OSError, ValueError):
            eos_ids = None
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if eos_ids is None:
        return
    eos_id = eos_ids[0] if isinstance(eos_ids, (list, tuple)) else eos_ids
    try:
        token = tokenizer.convert_ids_to_tokens(int(eos_id))
        if tokenizer.eos_token is None:
            tokenizer.eos_token = token
        tokenizer.pad_token = token
    except Exception:
        tokenizer.pad_token_id = int(eos_id)


__all__ = ["InklingImageProcessor", "InklingProcessor"]
