# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for the Embeddings API.

Provides:
- Base64 encoding for embeddings
- Dimension truncation with renormalization
- Token counting for usage statistics
"""

import base64
import math
import struct
from typing import Any, Dict, List, Optional, Sequence, Union

from ..exceptions import InvalidRequestError
from .embedding_models import EmbeddingInputItem


def encode_embedding_base64(embedding: List[float]) -> str:
    """
    Encode embedding vector as base64 string.

    OpenAI uses little-endian single-precision floats (float32).

    Args:
        embedding: List of float values

    Returns:
        Base64-encoded string of little-endian floats
    """
    packed = struct.pack(f"<{len(embedding)}f", *embedding)
    return base64.b64encode(packed).decode("ascii")


def find_non_finite_embeddings(embeddings: Sequence[Sequence[float]]) -> List[int]:
    """
    Return the indices of embeddings that contain NaN or infinite values.

    A model that overflows in its attention mask (see modernbert in
    mlx-embeddings) can hand back NaN vectors for some items of a padded
    batch. Those must not be serialized as a successful response: JSON has
    no NaN, so the client would receive ``null``-filled vectors with a 200.

    Args:
        embeddings: One vector per input item.

    Returns:
        Indices of the items whose vector is not entirely finite.
    """
    bad: List[int] = []
    for index, embedding in enumerate(embeddings):
        try:
            if not all(math.isfinite(float(value)) for value in embedding):
                bad.append(index)
        except (TypeError, ValueError):
            bad.append(index)
    return bad


def truncate_embedding(embedding: List[float], dimensions: int) -> List[float]:
    """
    Truncate embedding to specified dimensions and renormalize.

    When truncating embeddings, we need to renormalize to maintain
    unit length (L2 norm = 1) for cosine similarity calculations.

    Args:
        embedding: Original embedding vector
        dimensions: Target number of dimensions

    Returns:
        Truncated and renormalized embedding
    """
    if dimensions >= len(embedding):
        return embedding

    truncated = embedding[:dimensions]

    # Calculate L2 norm
    norm = math.sqrt(sum(x * x for x in truncated))

    # Renormalize to unit length
    if norm > 0:
        return [x / norm for x in truncated]
    return truncated


def count_tokens(processor: Any, texts: List[str]) -> int:
    """
    Count total tokens in input texts.

    Handles different tokenizer/processor types from mlx-embeddings.

    Args:
        processor: Tokenizer or processor from mlx-embeddings
        texts: List of input texts

    Returns:
        Total number of tokens across all texts
    """
    total = 0

    for text in texts:
        # Try different encoding methods based on processor type
        if hasattr(processor, "encode"):
            # Standard tokenizer
            tokens = processor.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                total += len(tokens)
            elif hasattr(tokens, "shape"):
                # MLX array
                total += tokens.shape[-1] if tokens.ndim > 0 else 1
            else:
                total += len(tokens)
        elif hasattr(processor, "tokenizer"):
            # Processor with nested tokenizer
            tokens = processor.tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                total += len(tokens)
            else:
                total += len(tokens)
        else:
            # Fallback: estimate based on whitespace
            total += len(text.split()) + 2  # +2 for special tokens

    return total


def normalize_input(
    input_data: Union[str, List[Any]],
) -> List[Union[str, Dict[str, str]]]:
    """
    Normalize the request ``input`` field into engine-ready items.

    Supported shapes:
    - A single string -> one text input.
    - A list of strings -> a batch of text inputs (OpenAI batch shape).
    - OpenAI-style multimodal content parts, e.g.
      ``[{"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:..."}}]``
      Any dict inside the list marks the whole list as the content parts of
      a *single* embedding input — the batch shape is list-of-strings only.
      Parts collapse into one structured item of the same shape ``items``
      produces, so content-part clients and ``items`` clients reach the
      engine through one path.

    Args:
        input_data: Single string, list of strings, or list of content parts

    Returns:
        List of strings (text batch) or a single structured item dict

    Raises:
        InvalidRequestError: When content parts are malformed or use
            unsupported features (e.g. several images in one input).
    """
    if isinstance(input_data, str):
        return [input_data]
    items = list(input_data)
    if all(isinstance(item, str) for item in items):
        return items
    if not items:
        return items
    return [_item_from_content_parts(items)]


def _content_part_url(part: Dict[str, Any], field: str) -> str:
    """Extract a media reference from a part whose value is a str or {"url": str}."""
    payload = part.get(field)
    if isinstance(payload, str) and payload:
        return payload
    if isinstance(payload, dict):
        url = payload.get("url")
        if isinstance(url, str) and url:
            return url
    raise InvalidRequestError(f"Embedding content part '{field}' requires a non-empty url")


def _input_audio_part_to_data_uri(part: Dict[str, Any]) -> str:
    """Fold an OpenAI ``input_audio`` part into the data-URI form engines take."""
    payload = part.get("input_audio")
    if isinstance(payload, str) and payload:
        return payload
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, str) and data:
            if data.startswith("data:"):
                return data
            fmt = payload.get("format") or "wav"
            return f"data:audio/{fmt};base64,{data}"
    raise InvalidRequestError("Embedding content part 'input_audio' requires base64 data")


def _item_from_content_parts(parts: List[Any]) -> Dict[str, str]:
    """Collapse a list of OpenAI-style content parts into one structured item."""
    texts: List[str] = []
    image: Optional[str] = None
    audio: Optional[str] = None

    for part in parts:
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, dict):
            raise InvalidRequestError(
                "Embedding input parts must be strings or content-part objects"
            )
        part_type = part.get("type")
        if part_type == "text":
            texts.append(str(part.get("text", "")))
        elif part_type in ("image_url", "image", "input_image"):
            if image is not None:
                raise InvalidRequestError(
                    "Embedding input supports at most one image per item"
                )
            field = "image_url" if part_type == "image_url" else part_type
            if part_type == "input_image" and isinstance(part.get("image_url"), str):
                image = part["image_url"]
            else:
                image = _content_part_url(part, field)
        elif part_type == "input_audio":
            if audio is not None:
                raise InvalidRequestError(
                    "Embedding input supports at most one audio per item"
                )
            audio = _input_audio_part_to_data_uri(part)
        else:
            raise InvalidRequestError(
                f"Unsupported embedding content part type: {part_type!r}"
            )

    item: Dict[str, str] = {}
    joined = "\n".join(t for t in texts if t)
    if joined:
        item["text"] = joined
    if image is not None:
        item["image"] = image
    if audio is not None:
        item["audio"] = audio
    if not item:
        raise InvalidRequestError(
            "Embedding content parts contain no text, image, or audio"
        )
    return item


def normalize_embedding_items(
    items: List[Union[EmbeddingInputItem, Dict[str, Any]]]
) -> List[Dict[str, str]]:
    """
    Normalize structured embedding items into plain dicts.

    Args:
        items: Structured embedding input items

    Returns:
        List of normalized item dicts with only supported keys
    """
    normalized: List[Dict[str, str]] = []

    for item in items:
        if hasattr(item, "model_dump"):
            payload = item.model_dump(exclude_none=True)
        else:
            payload = {
                key: value for key, value in dict(item).items() if value is not None
            }

        text = payload.get("text")
        image = payload.get("image")
        audio = payload.get("audio")

        normalized_item: Dict[str, str] = {}
        if text is not None:
            normalized_item["text"] = text
        if image is not None:
            normalized_item["image"] = image
        if audio is not None:
            normalized_item["audio"] = audio

        normalized.append(normalized_item)

    return normalized


