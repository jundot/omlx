# SPDX-License-Identifier: Apache-2.0
"""
Base model utilities for omlx custom model implementations.

This module provides common utilities for implementing custom models
that are not yet supported by mlx-embeddings.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class BaseModelArgs:
    """Base class for model configuration arguments."""

    pass


@dataclass
class BaseModelOutput:
    """Base output class for model forward pass."""

    last_hidden_state: mx.array
    """Hidden states from the last layer."""

    text_embeds: Optional[mx.array] = None
    """Normalized text embeddings."""

    pooler_output: Optional[mx.array] = None
    """Pooled output (e.g., CLS token or mean pooling)."""

    hidden_states: Optional[tuple] = None
    """All hidden states if output_hidden_states=True."""


def mean_pooling(hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    """
    Perform mean pooling over sequence with attention mask.

    Args:
        hidden_states: Shape (batch_size, seq_len, hidden_size)
        attention_mask: Shape (batch_size, seq_len)

    Returns:
        Pooled output of shape (batch_size, hidden_size)
    """
    # Expand mask to match hidden states shape
    mask_expanded = attention_mask[:, :, None].astype(hidden_states.dtype)

    # Sum embeddings weighted by mask
    sum_embeddings = mx.sum(hidden_states * mask_expanded, axis=1)

    # Sum mask values (clip to avoid division by zero)
    sum_mask = mx.clip(mx.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)

    return sum_embeddings / sum_mask


def last_token_pool(
    hidden_states: mx.array, attention_mask: Optional[mx.array] = None
) -> mx.array:
    """
    Pool the last *non-pad* token of each sequence (mask-aware).

    Decoder embedding models (Qwen2/Qwen3, gte-Qwen2, jina-code) represent a
    sequence by its final token's hidden state. The pool must be
    mask-aware because these tokenizers may pad on either side: a hardcoded
    ``[:, -1]`` is only correct under left padding and silently corrupts vectors
    under right padding.

    Args:
        hidden_states: Shape (batch_size, seq_len, hidden_size)
        attention_mask: Shape (batch_size, seq_len). If None, uses the last
            position.

    Returns:
        Pooled output of shape (batch_size, hidden_size)
    """
    if attention_mask is None:
        return hidden_states[:, -1]

    # Left padding: every sequence ends on a real token, so the last position
    # is always valid and we can index it directly.
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return hidden_states[:, -1]

    # Right (or mixed) padding: the last valid token is at sum(mask) - 1.
    sequence_lengths = attention_mask.sum(axis=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[mx.arange(batch_size), sequence_lengths]


def normalize_embeddings(embeddings: mx.array) -> mx.array:
    """
    L2 normalize embeddings.

    Args:
        embeddings: Shape (..., hidden_size)

    Returns:
        Normalized embeddings with same shape
    """
    return embeddings / mx.linalg.norm(embeddings, axis=-1, keepdims=True)
