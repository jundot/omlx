# SPDX-License-Identifier: MIT
"""DeepSeek V4.1 image geometry, packing and prompt encoding."""

import math

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(
        math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height
    )
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def safe_resize(
    height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token
):
    """Shrink the pixel size until the image costs at most max_n_token LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token
        )
        n_llm_h, n_llm_w = llm_grid(
            best_height, best_width, patch_size, downsample_ratio
        )
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, args):
    """Resize plan for an image of the given original size; a pure function of its arguments."""
    p = args.vision_patch_size
    if (
        args.vision_max_wh_ratio is not None
        and width > height * args.vision_max_wh_ratio
    ):
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        args.vision_downsample_ratio,
        args.vision_max_n_token,
    )


def image_patches(image, config):
    image = image.convert("RGB")
    nh, nw, height, width = plan_image_grid(image.width, image.height, config)
    if (
        config.vision_max_wh_ratio is not None
        and image.width >= config.vision_max_wh_ratio * image.height
    ):
        image = image.resize((width, height))
    else:
        scale = min(width / image.width, height / image.height)
        contained = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        if round(image.width * scale) == 0 or round(image.height * scale) == 0:
            resized = image.resize(contained, Image.Resampling.BICUBIC)
            image = Image.new("RGB", (width, height), (127, 127, 127))
            image.paste(
                resized, ((width - contained[0]) // 2, (height - contained[1]) // 2)
            )
        else:
            image = ImageOps.pad(image, (width, height), color=(127, 127, 127))
    p = config.vision_patch_size
    x = (np.asarray(image, np.float32) / 255 - 0.5) / 0.5
    patches = (
        x.reshape(height // p, p, width // p, p, 3)
        .transpose(0, 2, 4, 1, 3)
        .reshape(-1, 3, p, p)
    )
    types = [0] + ([1] * nw + [2]) * nh + [3]
    return mx.array(patches).astype(mx.bfloat16), height // p, width // p, types


class Processor:
    def __init__(self, tokenizer, config):
        self.tokenizer, self.config = tokenizer, config
        self.chat_template = "deepseek_v41_python_reference"
        # Both oMLX's text and image paths must use the same encoder.
        tokenizer.apply_chat_template = self.apply_chat_template

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        thinking_mode=None,
        reasoning_effort=None,
        tools=None,
        continue_final_message=False,
        **kwargs,
    ):
        from copy import deepcopy

        from .encoding import encode_messages

        messages = deepcopy(messages)
        if tools:
            if not messages or messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})
            messages[0]["tools"] = tools
        mode = thinking_mode or ("thinking" if enable_thinking else "chat")
        prompt = encode_messages(
            messages, thinking_mode=mode, reasoning_effort=reasoning_effort
        )
        header = "<｜Assistant｜>" + ("<think>" if mode == "thinking" else "</think>")
        if not add_generation_prompt and prompt.endswith(header):
            prompt = prompt[: -len(header)]
        if continue_final_message:
            raise ValueError(
                "DeepSeek V4.1 partial assistant continuation is not yet supported"
            )
        return (
            self.tokenizer.encode(prompt, add_special_tokens=False)
            if tokenize
            else prompt
        )

    def __call__(self, text=None, images=None, audio=None, videos=None, **kwargs):
        if audio is not None or videos is not None:
            raise ValueError("DeepSeek V4.1 processor supports text and images")
        texts = [text] if isinstance(text, str) else text
        if len(texts) != 1:
            raise ValueError("Prepare each V4.1 prompt independently before batching")
        ids = self.tokenizer.encode(texts[0], add_special_tokens=False)
        images = [] if images is None else images
        if ids.count(self.config.image_token_id) != len(images):
            raise ValueError("Image placeholder count does not match image count")
        expanded, pixels, grids, spans, types = [], [], [], [], []
        image_index = 0
        for token in ids:
            if token != self.config.image_token_id:
                expanded.append(token)
                continue
            patch, h, w, kinds = image_patches(images[image_index], self.config)
            image_index += 1
            spans.append((len(expanded), len(kinds)))
            expanded.extend([token] * len(kinds))
            pixels.append(patch)
            grids.append((h, w))
            types.append(kinds)
        result = {
            "input_ids": mx.array([expanded]),
            "attention_mask": mx.ones((1, len(expanded)), mx.int32),
        }
        if pixels:
            result.update(
                pixel_values=mx.concatenate(pixels),
                image_grids=grids,
                image_spans=spans,
                image_types=types,
            )
        return result


def format_messages(messages, num_images):
    """Preserve text/image interleaving, retaining placeholders for decoded PIL inputs."""
    from copy import deepcopy

    output, ranges, seen = deepcopy(messages), [], 0
    for i, message in enumerate(output):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        count = 0
        for part in content:
            kind = part.get("type")
            if kind in ("image", "image_url", "input_image"):
                part.clear()
                part.update(type="image", url="omlx-prepared-image")
                count += 1
            elif kind in ("text", "input_text"):
                part["type"] = "text"
            else:
                raise ValueError(f"Unsupported DeepSeek V4.1 content: {kind}")
        if count:
            ranges.append((i, count))
            seen += count
    if seen == 0 and num_images:
        target = next(
            (i for i in range(len(output) - 1, -1, -1) if output[i]["role"] == "user"),
            None,
        )
        if target is None:
            raise ValueError("Images require a user message")
        content = output[target].get("content") or ""
        parts = (
            content
            if isinstance(content, list)
            else [{"type": "text", "text": content}]
        )
        output[target]["content"] = [
            {"type": "image", "url": "omlx-prepared-image"} for _ in range(num_images)
        ] + parts
        ranges = [(target, num_images)]
        seen = num_images
    if seen != num_images:
        raise ValueError("Message image count does not match supplied images")
    return output, ranges
