# SPDX-License-Identifier: Apache-2.0
"""Winogrande benchmark.

Tests commonsense reasoning via pronoun/coreference resolution.
Given a sentence with a blank (_), choose which of two options fits.
Dataset bundled from winogrande (winogrande_xl) on HuggingFace.
1,267 validation questions.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from .base import BaseBenchmark
from .datasets import deterministic_sample, load_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class WinograndeBenchmark(BaseBenchmark):
    """Winogrande: 0-shot coreference resolution with 2 choices."""

    name = "winogrande"
    quick_size = 300

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load Winogrande from bundled data."""
        items = load_jsonl(DATA_DIR / "winogrande_val.jsonl")

        normalized = []
        for item in items:
            normalized.append({
                "id": item.get("id", ""),
                "sentence": item["sentence"],
                "option1": item["option1"],
                "option2": item["option2"],
                "answer": item["answer"],  # "1" or "2"
            })

        logger.info(f"Winogrande: loaded {len(normalized)} questions")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as a fill-in-the-blank choice."""
        sentence = item["sentence"]
        option1 = item["option1"]
        option2 = item["option2"]

        parts = [
            "Choose the correct option to fill in the blank (_). "
            "Answer with just the number (1 or 2).\n",
            f"Sentence: {sentence}\n",
            f"1. {option1}",
            f"2. {option2}",
            "\nAnswer:",
        ]

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_mc_answer(response, ["1", "2"])

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_question_text(self, item: dict) -> str:
        return item.get("sentence", "")

    def get_category(self, item: dict) -> Optional[str]:
        return None
