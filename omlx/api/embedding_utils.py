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
from typing import Any, Dict, List, Union

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


def normalize_input(input_data: Union[str, List[str]]) -> List[str]:
    """
    Normalize input to a list of strings.

    Args:
        input_data: Single string or list of strings

    Returns:
        List of strings
    """
    if isinstance(input_data, str):
        return [input_data]
    return list(input_data)


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

        normalized_item: Dict[str, str] = {}
        if text is not None:
            normalized_item["text"] = text
        if image is not None:
            normalized_item["image"] = image

        normalized.append(normalized_item)

    return normalized


