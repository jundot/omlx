# SPDX-License-Identifier: Apache-2.0
"""MMLU (Massive Multitask Language Understanding) benchmark.

Tests knowledge across 57 subjects using 5-shot multiple choice.
Dataset bundled from cais/mmlu on HuggingFace.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .base import BaseBenchmark
from .datasets import load_jsonl, stratified_sample

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def _format_subject_name(subject: str) -> str:
    """Convert subject slug to readable name."""
    return subject.replace("_", " ").title()


def _format_question(item: dict) -> str:
    """Format a single MMLU question with choices."""
    question = item["question"]
    choices = item["choices"]
    parts = [question]
    for i, choice in enumerate(choices):
        parts.append(f"{ANSWER_MAP[i]}. {choice}")
    return "\n".join(parts)


def _parse_choices(choices_field):
    """Parse choices field which may be a list or a string repr of a list."""
    if isinstance(choices_field, list):
        return choices_field
    if isinstance(choices_field, str):
        try:
            parsed = json.loads(choices_field.replace("'", '"'))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return []


class MMLUBenchmark(BaseBenchmark):
    """MMLU: 5-shot multiple choice across 57 academic subjects."""

    name = "mmlu"
    quick_size = 300

    def __init__(self):
        self._few_shot_examples: dict[str, list[dict]] = {}

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load MMLU from bundled data files."""
        test_items = load_jsonl(DATA_DIR / "mmlu_test.jsonl")
        all_items = []
        for item in test_items:
            choices = _parse_choices(item.get("choices", []))
            answer_idx = item.get("answer", 0)
            answer_letter = ANSWER_MAP.get(answer_idx, str(answer_idx))
            all_items.append({
                "question": item["question"],
                "choices": choices,
                "answer": answer_letter,
                "subject": item.get("subject", "unknown"),
            })

        # Load dev examples for few-shot
        dev_items = load_jsonl(DATA_DIR / "mmlu_dev.jsonl")
        for item in dev_items:
            subject = item.get("subject", "unknown")
            choices = _parse_choices(item.get("choices", []))
            answer_idx = item.get("answer", 0)
            answer_letter = ANSWER_MAP.get(answer_idx, str(answer_idx))
            if subject not in self._few_shot_examples:
                self._few_shot_examples[subject] = []
            if len(self._few_shot_examples[subject]) < 5:
                self._few_shot_examples[subject].append({
                    "question": item["question"],
                    "choices": choices,
                    "answer": answer_letter,
                })

        logger.info(f"MMLU: loaded {len(all_items)} questions")

        if sample_size == 0:
            return all_items

        return stratified_sample(all_items, sample_size, key="subject")

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format with 5-shot examples from the same subject."""
        subject = item["subject"]
        subject_name = _format_subject_name(subject)

        parts = [
            f"The following are multiple choice questions about {subject_name}. "
            f"Answer with just the letter (A, B, C, or D).\n"
        ]

        # Add few-shot examples
        examples = self._few_shot_examples.get(subject, [])
        for ex in examples:
            parts.append(_format_question(ex))
            parts.append(f"Answer: {ex['answer']}\n")

        # Add the actual question
        parts.append(_format_question(item))
        parts.append("Answer:")

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_mc_answer(response, ["A", "B", "C", "D"])

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_category(self, item: dict) -> Optional[str]:
        return item.get("subject")
