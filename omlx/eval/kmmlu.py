# SPDX-License-Identifier: Apache-2.0
"""KMMLU (Korean MMLU) benchmark.

Tests knowledge across 45 Korean subjects using 5-shot multiple choice.
Includes Korean-specific topics like Korean history, law, and culture.
Dataset bundled from HAERAE-HUB/KMMLU on HuggingFace.
"""

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
    return subject.replace("-", " ").replace("_", " ").title()


def _format_question(item: dict) -> str:
    question = item["question"]
    choices = item["choices"]
    parts = [question]
    for i, choice in enumerate(choices):
        parts.append(f"{ANSWER_MAP[i]}. {choice}")
    return "\n".join(parts)


class KMMLUBenchmark(BaseBenchmark):
    """KMMLU: 5-shot Korean multiple choice across 45 subjects."""

    name = "kmmlu"
    quick_size = 300

    def __init__(self):
        self._few_shot_examples: dict[str, list[dict]] = {}

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        test_items = load_jsonl(DATA_DIR / "kmmlu_test.jsonl")
        all_items = []
        for item in test_items:
            answer_idx = item.get("answer", 0)
            answer_letter = ANSWER_MAP.get(answer_idx, str(answer_idx))
            all_items.append({
                "question": item["question"],
                "choices": item["choices"],
                "answer": answer_letter,
                "subject": item.get("subject", "unknown"),
            })

        dev_items = load_jsonl(DATA_DIR / "kmmlu_dev.jsonl")
        for item in dev_items:
            subject = item.get("subject", "unknown")
            answer_idx = item.get("answer", 0)
            answer_letter = ANSWER_MAP.get(answer_idx, str(answer_idx))
            if subject not in self._few_shot_examples:
                self._few_shot_examples[subject] = []
            if len(self._few_shot_examples[subject]) < 5:
                self._few_shot_examples[subject].append({
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": answer_letter,
                })

        logger.info(f"KMMLU: loaded {len(all_items)} questions")

        if sample_size == 0:
            return all_items
        return stratified_sample(all_items, sample_size, key="subject")

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        subject = item["subject"]
        subject_name = _format_subject_name(subject)

        parts = [
            f"다음은 {subject_name}에 대한 객관식 문제입니다. "
            f"정답의 알파벳(A, B, C, D)만 답하세요.\n"
        ]

        examples = self._few_shot_examples.get(subject, [])
        for ex in examples:
            parts.append(_format_question(ex))
            parts.append(f"정답: {ex['answer']}\n")

        parts.append(_format_question(item))
        parts.append("정답:")

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_mc_answer(response, ["A", "B", "C", "D"])

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_category(self, item: dict) -> Optional[str]:
        return item.get("subject")
