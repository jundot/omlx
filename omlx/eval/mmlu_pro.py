# SPDX-License-Identifier: Apache-2.0
"""MMLU-Pro benchmark.

A harder version of MMLU with 10 answer choices instead of 4,
requiring deeper reasoning. Covers 14 academic subjects.
Dataset bundled from TIGER-Lab/MMLU-Pro on HuggingFace.
"""

import logging
from pathlib import Path
from typing import Optional

from .base import BaseBenchmark
from .datasets import deterministic_sample, stratified_sample, load_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class MMLUProBenchmark(BaseBenchmark):
    """MMLU-Pro: 0-shot hard knowledge MC with 10 choices."""

    name = "mmlu_pro"
    quick_size = 300

    def get_max_tokens(self) -> int:
        return 2048

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load MMLU-Pro from bundled data."""
        items = load_jsonl(DATA_DIR / "mmlu_pro_test.jsonl")

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
                "subject": item.get("subject", "general"),
            })

        logger.info(f"MMLU-Pro: loaded {len(normalized)} questions")

        if sample_size == 0:
            return normalized

        return stratified_sample(normalized, sample_size, "subject")

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as multiple choice with lettered options (A-J)."""
        question = item["question"]
        choices = item["choices"]
        labels = item["labels"]

        parts = [
            "Answer the following question. "
            "Answer with just the letter.\n",
            f"Question: {question}\n",
        ]
        for label, choice in zip(labels, choices):
            parts.append(f"{label}. {choice}")

        parts.append("\nAnswer:")

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        valid = item.get("labels", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
        return self._extract_mc_answer(response, valid)

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_category(self, item: dict) -> Optional[str]:
        return item.get("subject")
