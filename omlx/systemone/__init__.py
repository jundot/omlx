# SPDX-License-Identifier: Apache-2.0
"""Jev-compatible "System One" structured reads.

One read-only denoise step over a seeded answer template yields a calibrated
distribution over each question's allowed labels. See ``types`` for the data
model, ``schema`` for template construction, and
``VLMBatchedEngine.structured_read`` for the forward pass.
"""

from .read import (
    confidence,
    entropy,
    mean_probabilities,
    slot_distribution,
    softmax,
    to_answer,
)
from .schema import (
    SCAFFOLD_TEXT,
    Schema,
    SchemaError,
    answer_text,
    build_prompt_ids,
    build_schema,
    canvas_width,
    choice_labels,
    groups,
    read_groups,
    resolve_template,
    scaffold_ids,
    system_text,
    template_emits_scaffold,
    text_of,
)
from .types import (
    ReadGroup,
    ReadSlot,
    SlotDistribution,
    StructuredReadResult,
    normalized,
)

__all__ = [
    "SCAFFOLD_TEXT",
    "ReadGroup",
    "ReadSlot",
    "Schema",
    "SchemaError",
    "SlotDistribution",
    "StructuredReadResult",
    "answer_text",
    "build_prompt_ids",
    "build_schema",
    "canvas_width",
    "choice_labels",
    "confidence",
    "entropy",
    "groups",
    "mean_probabilities",
    "normalized",
    "read_groups",
    "resolve_template",
    "scaffold_ids",
    "slot_distribution",
    "softmax",
    "system_text",
    "template_emits_scaffold",
    "text_of",
    "to_answer",
]
