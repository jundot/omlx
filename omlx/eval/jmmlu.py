# SPDX-License-Identifier: Apache-2.0
"""JMMLU (Japanese MMLU) benchmark.

Tests knowledge across 112 Japanese subjects using 0-shot multiple choice.
Includes Japan-specific topics like Japanese history and culture.
Dataset bundled from nlp-waseda/JMMLU on HuggingFace.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from .base import BaseBenchmark
from .datasets import load_jsonl, stratified_sample

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def _format_subject_name(subject: str) -> str:
    return subject.replace("_", " ").replace("-", " ").title()


def _format_question(item: dict) -> str:
    question = item["question"]
    choices = item["choices"]
    parts = [question]
    for letter, choice in zip(["A", "B", "C", "D"], choices):
        parts.append(f"{letter}. {choice}")
    return "\n".join(parts)


class JMMLUBenchmark(BaseBenchmark):
    """JMMLU: 0-shot Japanese multiple choice across 112 subjects."""

    name = "jmmlu"
    quick_size = 300

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        test_items = load_jsonl(DATA_DIR / "jmmlu_test.jsonl")
        all_items = []
        for item in test_items:
            answer = item.get("answer", "A")
            all_items.append({
                "question": item["question"],
                "choices": item["choices"],
                "answer": answer,  # Already A/B/C/D
                "subject": item.get("subject", "unknown"),
            })

        logger.info(f"JMMLU: loaded {len(all_items)} questions")

        if sample_size == 0:
            return all_items
        return stratified_sample(all_items, sample_size, key="subject")

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        subject = item["subject"]
        subject_name = _format_subject_name(subject)

        parts = [
            f"以下は{subject_name}に関する選択問題です。"
            f"正解のアルファベット（A、B、C、D）だけを答えてください。\n"
        ]

        parts.append(_format_question(item))
        parts.append("答え:")

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_mc_answer(response, ["A", "B", "C", "D"])

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_category(self, item: dict) -> Optional[str]:
        return item.get("subject")
