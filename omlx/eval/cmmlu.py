# SPDX-License-Identifier: Apache-2.0
"""CMMLU (Chinese MMLU) benchmark.

Tests knowledge across 67 Chinese subjects using 5-shot multiple choice.
Includes China-specific topics like Chinese history, law, and culture.
Dataset bundled from haonan-li/cmmlu on HuggingFace.
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


class CMMLUBenchmark(BaseBenchmark):
    """CMMLU: 5-shot Chinese multiple choice across 67 subjects."""

    name = "cmmlu"
    quick_size = 300

    def __init__(self):
        self._few_shot_examples: dict[str, list[dict]] = {}

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        test_items = load_jsonl(DATA_DIR / "cmmlu_test.jsonl")
        all_items = []
        for item in test_items:
            answer = item.get("answer", "A")
            all_items.append({
                "question": item["question"],
                "choices": item["choices"],
                "answer": answer,  # Already A/B/C/D
                "subject": item.get("subject", "unknown"),
            })

        dev_items = load_jsonl(DATA_DIR / "cmmlu_dev.jsonl")
        for item in dev_items:
            subject = item.get("subject", "unknown")
            if subject not in self._few_shot_examples:
                self._few_shot_examples[subject] = []
            if len(self._few_shot_examples[subject]) < 5:
                self._few_shot_examples[subject].append({
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item.get("answer", "A"),
                })

        logger.info(f"CMMLU: loaded {len(all_items)} questions")

        if sample_size == 0:
            return all_items
        return stratified_sample(all_items, sample_size, key="subject")

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        subject = item["subject"]
        subject_name = _format_subject_name(subject)

        parts = [
            f"以下是关于{subject_name}的单选题，请直接回答正确选项的字母（A、B、C或D）。\n"
        ]

        examples = self._few_shot_examples.get(subject, [])
        for ex in examples:
            parts.append(_format_question(ex))
            parts.append(f"答案: {ex['answer']}\n")

        parts.append(_format_question(item))
        parts.append("答案:")

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_mc_answer(response, ["A", "B", "C", "D"])

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_category(self, item: dict) -> Optional[str]:
        return item.get("subject")
