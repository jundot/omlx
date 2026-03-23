# SPDX-License-Identifier: Apache-2.0
"""ARC-Challenge benchmark.

Tests science reasoning with multiple choice questions.
Dataset bundled from allenai/ai2_arc (Challenge split) on HuggingFace.
1,172 questions requiring scientific knowledge and deduction.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from .base import BaseBenchmark
from .datasets import deterministic_sample, load_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class ARCChallengeBenchmark(BaseBenchmark):
    """ARC-Challenge: 0-shot science reasoning multiple choice."""

    name = "arc_challenge"
    quick_size = 300

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load ARC-Challenge from bundled data."""
        items = load_jsonl(DATA_DIR / "arc_challenge.jsonl")

        normalized = []
        for item in items:
            choices = item.get("choices", [])
            labels = item.get("labels", [])
            if not choices or not labels:
                continue
            normalized.append({
                "id": item.get("id", ""),
                "question": item["question"],
                "choices": choices,
                "labels": labels,
                "answer": item["answer"],
            })

        logger.info(f"ARC-Challenge: loaded {len(normalized)} questions")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as multiple choice with lettered options."""
        question = item["question"]
        choices = item["choices"]
        labels = item["labels"]

        parts = [
            "Answer the following science question. "
            "Answer with just the letter.\n",
            f"Question: {question}\n",
        ]
        for label, choice in zip(labels, choices):
            parts.append(f"{label}. {choice}")

        parts.append("\nAnswer:")

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        valid = item.get("labels", ["A", "B", "C", "D"])
        return self._extract_mc_answer(response, valid)

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_category(self, item: dict) -> Optional[str]:
        return None
