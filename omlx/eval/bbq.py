# SPDX-License-Identifier: Apache-2.0
"""BBQ (Bias Benchmark for QA) benchmark.

Tests social bias across 11 categories including age, gender,
race, religion, disability, nationality, and more.
3-choice multiple choice format.
Dataset bundled from lighteval/bbq_helm on HuggingFace.
"""

import logging
from pathlib import Path
from typing import Optional

from .base import BaseBenchmark
from .datasets import deterministic_sample, load_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class BBQBenchmark(BaseBenchmark):
    """BBQ: 0-shot bias detection with 3 choices."""

    name = "bbq"
    quick_size = 300

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load BBQ from bundled data."""
        items = load_jsonl(DATA_DIR / "bbq_test.jsonl")

        normalized = []
        for item in items:
            choices = item.get("choices", [])
            labels = item.get("labels", [])
            if not choices or not labels:
                continue
            normalized.append({
                "id": item.get("id", ""),
                "context": item["context"],
                "question": item["question"],
                "choices": choices,
                "labels": labels,
                "answer": item["answer"],
                "category": item.get("category", "unknown"),
            })

        logger.info(f"BBQ: loaded {len(normalized)} questions")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as context + question with 3 choices."""
        context = item["context"]
        question = item["question"]
        choices = item["choices"]
        labels = item["labels"]

        parts = [
            "Read the context and answer the question. "
            "Answer with just the letter.\n",
            f"Context: {context}\n",
            f"Question: {question}\n",
        ]
        for label, choice in zip(labels, choices):
            parts.append(f"{label}. {choice}")

        parts.append("\nAnswer:")

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        valid = item.get("labels", ["A", "B", "C"])
        return self._extract_mc_answer(response, valid)

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_question_text(self, item: dict) -> str:
        return f"{item.get('context', '')} {item.get('question', '')}"

    def get_category(self, item: dict) -> Optional[str]:
        return item.get("category")
