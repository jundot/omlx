# SPDX-License-Identifier: Apache-2.0
"""Base classes and data models for accuracy benchmarks."""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    """Result for a single benchmark question."""

    question_id: str
    correct: bool
    expected: str
    predicted: str
    time_seconds: float
    question_text: str = ""
    raw_response: str = ""


@dataclass
class BenchmarkResult:
    """Aggregated result for a complete benchmark run."""

    benchmark_name: str
    accuracy: float
    total_questions: int
    correct_count: int
    time_seconds: float
    question_results: list[QuestionResult] = field(default_factory=list)
    category_scores: Optional[dict[str, float]] = None


class BaseBenchmark(ABC):
    """Abstract base class for accuracy benchmarks."""

    name: str = ""
    quick_size: int = 100

    @abstractmethod
    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load dataset items.

        Args:
            sample_size: Number of questions to sample. 0 = full dataset.

        Returns:
            List of dataset items (format varies by benchmark).
        """
        pass

    @abstractmethod
    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format a dataset item into chat messages for the engine.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        pass

    @abstractmethod
    def extract_answer(self, response: str, item: dict) -> str:
        """Extract the predicted answer from model response text."""
        pass

    @abstractmethod
    def check_answer(self, predicted: str, item: dict) -> bool:
        """Check if the predicted answer is correct."""
        pass

    def get_max_tokens(self) -> int:
        """Max tokens to generate per question. Override for longer answers."""
        return 128

    def get_category(self, item: dict) -> Optional[str]:
        """Return category/subject for per-category scoring. None if N/A."""
        return None

    def get_question_text(self, item: dict) -> str:
        """Return a human-readable question text for result export."""
        return item.get("question", item.get("description", item.get("context", "")))

    @staticmethod
    def _extract_mc_answer(response: str, valid_letters: list[str]) -> str:
        """Extract multiple choice answer from response.

        Strategy:
        1. Look for explicit "answer is X" / "answer: X" patterns (last match)
        2. Fall back to last valid letter in response
        3. Case-insensitive
        """
        response_upper = response.strip().upper()
        pattern_letters = "".join(valid_letters)

        # 1. Look for "answer is X", "answer: X", "answer X" patterns — use LAST match
        answer_patterns = re.findall(
            r"(?:answer\s*(?:is|:)\s*)([" + pattern_letters + r"])\b",
            response_upper,
        )
        if answer_patterns:
            return answer_patterns[-1]

        # 2. Fall back to last valid letter with word boundary
        all_matches = re.findall(
            r"\b([" + pattern_letters + r"])\b",
            response_upper,
        )
        if all_matches:
            return all_matches[-1]

        # 3. Check first character
        if response.strip() and response.strip()[0].upper() in valid_letters:
            return response.strip()[0].upper()

        return ""

    @staticmethod
    def _extract_last_code_block(response: str) -> str:
        """Extract the LAST code block from model response.

        Uses last match to avoid picking up drafts/examples.
        Falls back to line-by-line detection if no code blocks found.
        """
        response = response.strip()

        # Find ALL python code blocks, use LAST
        blocks = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if blocks:
            return blocks[-1].strip()

        # Generic code blocks
        blocks = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
        if blocks:
            return blocks[-1].strip()

        # Line-by-line fallback
        lines = response.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if not in_code and (
                line.startswith("def ")
                or line.startswith("class ")
                or line.startswith("import ")
                or line.startswith("from ")
                or line.startswith("#")
            ):
                in_code = True
            if in_code:
                code_lines.append(line)

        return "\n".join(code_lines) if code_lines else response

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks from model output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    async def _eval_single(
        self, engine: Any, item: dict, index: int,
        sampling_kwargs: Optional[dict] = None,
    ) -> tuple[int, dict, str, str]:
        """Evaluate a single item. Returns (index, item, response_text, prompt_text)."""
        messages = self.format_prompt(item)
        # Extract the full prompt text for logging
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        kwargs = dict(sampling_kwargs or {})
        # Force benchmark-controlled params (override model settings)
        kwargs["max_tokens"] = self.get_max_tokens()
        kwargs["temperature"] = 0.0
        kwargs["presence_penalty"] = 0.0
        kwargs["repetition_penalty"] = 1.0
        try:
            output = await engine.chat(
                messages=messages,
                chat_template_kwargs={"enable_thinking": False},
                **kwargs,
            )
            text = self._strip_think_tags(output.text)
            return index, item, text, prompt_text
        except Exception as e:
            logger.warning(f"Engine error on question {index}: {e}")
            return index, item, "", prompt_text

    async def run(
        self,
        engine: Any,
        items: list[dict],
        on_progress: Optional[Callable[[int, int], Any]] = None,
        batch_size: int = 1,
        sampling_kwargs: Optional[dict] = None,
    ) -> BenchmarkResult:
        """Run the benchmark on all items.

        Args:
            engine: oMLX engine instance with chat() method.
            items: Dataset items to evaluate.
            on_progress: Callback(current, total) for progress reporting.
            batch_size: Number of concurrent requests (1 = sequential).

        Returns:
            BenchmarkResult with accuracy and per-question details.
        """
        results: list[QuestionResult] = []
        correct = 0
        category_correct: dict[str, int] = {}
        category_total: dict[str, int] = {}
        start_time = time.time()
        completed = 0

        # Process in batches
        for batch_start in range(0, len(items), batch_size):
            batch_end = min(batch_start + batch_size, len(items))
            batch = items[batch_start:batch_end]
            batch_start_time = time.time()

            # Launch concurrent requests
            tasks = [
                self._eval_single(engine, item, batch_start + j, sampling_kwargs)
                for j, item in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks)
            batch_elapsed = time.time() - batch_start_time

            # Process results in order
            for idx, item, response_text, prompt_text in sorted(batch_results, key=lambda x: x[0]):
                predicted = self.extract_answer(response_text, item)
                is_correct = self.check_answer(predicted, item)

                if is_correct:
                    correct += 1

                cat = self.get_category(item)
                if cat is not None:
                    category_total[cat] = category_total.get(cat, 0) + 1
                    if is_correct:
                        category_correct[cat] = category_correct.get(cat, 0) + 1

                q_id = item.get("id", str(idx))
                expected = item.get("answer", "")
                results.append(
                    QuestionResult(
                        question_id=str(q_id),
                        correct=is_correct,
                        expected=str(expected),
                        predicted=predicted,
                        time_seconds=batch_elapsed / len(batch),
                        question_text=prompt_text,
                        raw_response=response_text,
                    )
                )

            completed += len(batch)
            if on_progress:
                await on_progress(completed, len(items))

        total_time = time.time() - start_time
        total = len(items)
        accuracy = correct / total if total > 0 else 0.0

        cat_scores = None
        if category_total:
            cat_scores = {}
            for cat in sorted(category_total.keys()):
                cat_scores[cat] = (
                    category_correct.get(cat, 0) / category_total[cat]
                    if category_total[cat] > 0
                    else 0.0
                )

        return BenchmarkResult(
            benchmark_name=self.name,
            accuracy=accuracy,
            total_questions=total,
            correct_count=correct,
            time_seconds=total_time,
            question_results=results,
            category_scores=cat_scores,
        )
