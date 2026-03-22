# SPDX-License-Identifier: Apache-2.0
"""HellaSwag benchmark.

Tests commonsense reasoning by choosing the most plausible
continuation of a scenario. 0-shot multiple choice.
Dataset bundled from Rowan/hellaswag on HuggingFace.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from .base import BaseBenchmark
from .datasets import deterministic_sample, load_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


class HellaSwagBenchmark(BaseBenchmark):
    """HellaSwag: 0-shot commonsense reasoning with 4 choices."""

    name = "hellaswag"
    quick_size = 200

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load HellaSwag from bundled data."""
        items = load_jsonl(DATA_DIR / "hellaswag_val.jsonl")

        normalized = []
        for item in items:
            label = item.get("label", "0")
            normalized.append({
                "id": item.get("ind", ""),
                "context": item.get("ctx", ""),
                "endings": item.get("endings", []),
                "answer": int(label) if isinstance(label, (int, float)) else int(label),
                "activity_label": item.get("activity_label", ""),
            })

        logger.info(f"HellaSwag: loaded {len(normalized)} questions")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format context + 4 endings as multiple choice."""
        context = item["context"]
        endings = item["endings"]

        parts = [
            "Choose the most plausible continuation. "
            "Answer with just the letter (A, B, C, or D).\n",
            f"Context: {context}\n",
        ]
        for i, ending in enumerate(endings[:4]):
            parts.append(f"{ANSWER_MAP[i]}. {ending}")

        parts.append("\nAnswer:")

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_mc_answer(response, ["A", "B", "C", "D"])

    def check_answer(self, predicted: str, item: dict) -> bool:
        expected_letter = ANSWER_MAP.get(item["answer"], "")
        return predicted == expected_letter

    def get_category(self, item: dict) -> Optional[str]:
        return item.get("activity_label")
