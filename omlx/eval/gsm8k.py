# SPDX-License-Identifier: Apache-2.0
"""GSM8K benchmark.

Tests mathematical reasoning using grade school math word problems.
5-shot chain-of-thought prompting, answer extraction from "#### N" pattern.
Dataset bundled from openai/gsm8k on HuggingFace.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from .base import BaseBenchmark
from .datasets import deterministic_sample, load_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# Standard 5-shot examples for GSM8K
FEW_SHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. #### 6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. Then 2 more arrive. So there are 3 + 2 = 5 cars. #### 5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops. #### 8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. He got 2 from mom and 2 from dad, so 2 + 2 = 4 more toys. Now he has 5 + 4 = 9 toys. #### 9",
    },
]


def _extract_numeric_answer(text: str) -> str:
    """Extract the final numeric answer from a GSM8K-style response.

    Looks for #### pattern first, then falls back to the last number.
    """
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def _normalize_number(s: str) -> str:
    """Normalize a number string for comparison."""
    s = s.strip().replace(",", "")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return s


class GSM8KBenchmark(BaseBenchmark):
    """GSM8K: 5-shot chain-of-thought math reasoning."""

    name = "gsm8k"
    quick_size = 100

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load GSM8K from bundled data."""
        items = load_jsonl(DATA_DIR / "gsm8k_test.jsonl")

        normalized = []
        for i, item in enumerate(items):
            answer_text = item.get("answer", "")
            numeric = _extract_numeric_answer(answer_text)
            normalized.append({
                "id": str(i),
                "question": item.get("question", ""),
                "answer_text": answer_text,
                "answer": numeric,
            })

        logger.info(f"GSM8K: loaded {len(normalized)} questions")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def get_max_tokens(self) -> int:
        return 512

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format with 5-shot chain-of-thought examples."""
        parts = [
            "Solve the following math problem step by step. "
            "End your answer with #### followed by the final numeric answer.\n"
        ]

        for ex in FEW_SHOT_EXAMPLES:
            parts.append(f"Question: {ex['question']}")
            parts.append(f"Answer: {ex['answer']}\n")

        parts.append(f"Question: {item['question']}")
        parts.append("Answer:")

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        return _extract_numeric_answer(response)

    def check_answer(self, predicted: str, item: dict) -> bool:
        if not predicted:
            return False
        return _normalize_number(predicted) == _normalize_number(item["answer"])

    def get_category(self, item: dict) -> Optional[str]:
        return None
