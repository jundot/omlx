# SPDX-License-Identifier: Apache-2.0
"""
Llama bidirectional reranker for omlx.

Implements LlamaBidirectionalForSequenceClassification (for example
nvidia/llama-nemotron-rerank-1b-v2): a Llama backbone run with full
bidirectional attention, masked mean pooling over the real tokens, and a
bias-free scalar classification head.

The head returns raw signed logits. Unlike the other SequenceClassification
rerankers in this package, no sigmoid or softmax is applied here: the
reference implementation documents each score as a raw logit and leaves the
optional probability transform to the caller.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import ModelArgs as LlamaBackboneArgs
from mlx_lm.models.llama import TransformerBlock

from ..model_discovery import LLAMA_BIDIRECTIONAL_RERANKER_ARCHITECTURE
from .base_model import BaseModelArgs, BaseModelOutput, mean_pooling


@dataclass
class ModelArgs(BaseModelArgs):
    """LlamaBidirectionalForSequenceClassification configuration."""

    model_type: str = "llama_bidirec"
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    intermediate_size: int = 8192
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 131072
    vocab_size: int = 128256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_traditional: bool = False
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False

    # Classification head.
    architectures: List[str] = field(
        default_factory=lambda: [LLAMA_BIDIRECTIONAL_RERANKER_ARCHITECTURE]
    )
    pooling: str = "avg"
    temperature: float = 1.0
    num_labels: Optional[int] = None
    id2label: Optional[Dict[str, str]] = None
    label2id: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_labels is None:
            # These checkpoints ship id2label and no explicit num_labels, which
            # is how HuggingFace's PretrainedConfig derives the head width.
            self.num_labels = len(self.id2label) if self.id2label else 1
        if self.architectures != [LLAMA_BIDIRECTIONAL_RERANKER_ARCHITECTURE]:
            raise ValueError(
                f"Expected architecture {LLAMA_BIDIRECTIONAL_RERANKER_ARCHITECTURE}; "
                f"got {self.architectures}"
            )
        if self.pooling != "avg":
            # Every other pooling mode would score a different vector through
            # the same head, so fail closed instead of silently mis-ranking.
            raise ValueError(
                f"Unsupported pooling {self.pooling!r}; only 'avg' is implemented"
            )
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive; got {self.temperature}")

    def backbone_args(self) -> LlamaBackboneArgs:
        """Arguments for the shared mlx-lm Llama transformer blocks."""
        return LlamaBackboneArgs(
            model_type="llama",
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            rms_norm_eps=self.rms_norm_eps,
            vocab_size=self.vocab_size,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            num_key_value_heads=self.num_key_value_heads,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            rope_theta=self.rope_theta,
            rope_traditional=self.rope_traditional,
            rope_scaling=self.rope_scaling,
            tie_word_embeddings=self.tie_word_embeddings,
        )


def bidirectional_attention_mask(
    attention_mask: mx.array,
    dtype: mx.Dtype,
) -> mx.array:
    """Build an additive mask where every token attends to every real token.

    The mask is keyed on columns only, so a padding row still sees the real
    tokens. That keeps every softmax row finite; masking padding rows as well
    would make them all ``-inf`` and produce NaNs that spread through the
    batch, even though those rows are later dropped by the pooling mask.
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape (batch, sequence); "
            f"got {attention_mask.shape}"
        )
    batch_size, sequence_length = attention_mask.shape
    key_mask = attention_mask.astype(mx.bool_)[:, None, None, :]
    additive = mx.where(key_mask, 0.0, -mx.inf).astype(dtype)
    return mx.broadcast_to(
        additive,
        (batch_size, 1, sequence_length, sequence_length),
    )


class LlamaBidirectionalModel(nn.Module):
    """Llama backbone without the causal mask."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        backbone = args.backbone_args()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=backbone) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        mask = bidirectional_attention_mask(attention_mask, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, cache=None)
        return self.norm(hidden_states)


class Model(nn.Module):
    """Bidirectional Llama with a scalar relevance head."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = LlamaBidirectionalModel(config)
        # Root-level attribute so the parameter path is exactly `score.weight`,
        # matching the checkpoint. The reference head carries no bias.
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> BaseModelOutput:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(hidden_states, attention_mask)
        # Raw signed logits: the reference leaves the sigmoid to the caller, so
        # applying one here would silently change the score domain.
        logits = self.score(pooled) / self.config.temperature
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            text_embeds=None,
            pooler_output=logits,
        )

    def sanitize(self, weights):
        """Drop tensors this port does not use and keep every other key as-is.

        ``score.weight`` sits at the checkpoint root beside ``model.*``, which
        is exactly where this module places it, so no key is rewritten. A
        blanket ``model.`` prefix rule would move the head out of reach and
        break weight loading.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and key != "lm_head.weight"
        }

    @property
    def layers(self):
        return self.model.layers
