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
        return 32

    def get_category(self, item: dict) -> Optional[str]:
        """Return category/subject for per-category scoring. None if N/A."""
        return None

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks from model output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    async def _eval_single(
        self, engine: Any, item: dict, index: int
    ) -> tuple[int, dict, str]:
        """Evaluate a single item. Returns (index, item, response_text)."""
        messages = self.format_prompt(item)
        try:
            output = await engine.chat(
                messages=messages,
                max_tokens=self.get_max_tokens(),
                temperature=0.0,
            )
            text = self._strip_think_tags(output.text)
            return index, item, text
        except Exception as e:
            logger.warning(f"Engine error on question {index}: {e}")
            return index, item, ""

    async def run(
        self,
        engine: Any,
        items: list[dict],
        on_progress: Optional[Callable[[int, int], Any]] = None,
        batch_size: int = 1,
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
                self._eval_single(engine, item, batch_start + j)
                for j, item in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks)
            batch_elapsed = time.time() - batch_start_time

            # Process results in order
            for idx, item, response_text in sorted(batch_results, key=lambda x: x[0]):
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
