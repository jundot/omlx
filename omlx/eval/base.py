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

# Token budget for thinking/reasoning models (industry reference: OpenCompass 8K~32K)
THINKING_MIN_TOKENS = 8192
THINKING_MAX_TOKENS = 32768


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
    category: Optional[str] = None
    pass_mode: Optional[str] = None
    failure_type: Optional[str] = None
    error: str = ""


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
    thinking_used: bool = False
    benchmark_variant: Optional[str] = None


@dataclass
class EvalGenerated:
    """Generated output for one indexed benchmark item."""

    index: int
    item: dict
    response_text: str
    prompt_text: str
    raw_text: str
    generation_seconds: float


class _EvalProgressCancelledError(Exception):
    """Internal bridge so worker-task cancellation reaches the caller."""


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

    def resolve_max_tokens(self, engine: Any, enable_thinking: bool = False) -> int:
        """Resolve benchmark-controlled generation budget for one question."""
        max_tokens = self.get_max_tokens()
        # Harmony models (gpt_oss) use analysis + final channels;
        # analysis can consume the entire budget before final is emitted
        if getattr(engine, "model_type", None) == "gpt_oss":
            return max(max_tokens * 4, 8192)
        if enable_thinking:
            return min(max(max_tokens, THINKING_MIN_TOKENS), THINKING_MAX_TOKENS)
        return max_tokens

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
        if "<think>" not in text:
            return text
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    async def _eval_single(
        self,
        engine: Any,
        item: dict,
        index: int,
        sampling_kwargs: Optional[dict] = None,
        enable_thinking: bool = False,
    ) -> tuple[int, dict, str, str, str]:
        """Evaluate a single item.

        Returns (index, item, response_text, prompt_text, raw_text).
        raw_text is the unstripped output for auto-detection of thinking tags.
        """
        messages = self.format_prompt(item)
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        kwargs = dict(sampling_kwargs or {})
        # Benchmark owns answer budget; sampling policy is supplied by caller.
        kwargs["max_tokens"] = self.resolve_max_tokens(engine, enable_thinking)
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("presence_penalty", 0.0)
        kwargs.setdefault("repetition_penalty", 1.0)
        # Merge enable_thinking into any existing chat_template_kwargs
        ct_kwargs = kwargs.pop("chat_template_kwargs", {}) or {}
        ct_kwargs["enable_thinking"] = enable_thinking
        kwargs["chat_template_kwargs"] = ct_kwargs
        try:
            output = await engine.chat(
                messages=messages,
                **kwargs,
            )
            raw_text = output.text
            text = self._strip_think_tags(raw_text)
            return index, item, text, prompt_text, raw_text
        except Exception as e:
            logger.warning(f"Engine error on question {index}: {e}")
            return index, item, "", prompt_text, ""

    async def _eval_single_timed(
        self,
        engine: Any,
        item: dict,
        index: int,
        sampling_kwargs: Optional[dict] = None,
        enable_thinking: bool = False,
    ) -> EvalGenerated:
        start_time = time.time()
        idx, eval_item, response_text, prompt_text, raw_text = await self._eval_single(
            engine, item, index, sampling_kwargs, enable_thinking
        )
        return EvalGenerated(
            index=idx,
            item=eval_item,
            response_text=response_text,
            prompt_text=prompt_text,
            raw_text=raw_text,
            generation_seconds=time.time() - start_time,
        )

    async def _run_probe_batch(
        self,
        engine: Any,
        indexed_items: list[tuple[int, dict]],
        sampling_kwargs: Optional[dict],
        enable_thinking: bool,
    ) -> list[EvalGenerated]:
        return await asyncio.gather(
            *[
                self._eval_single_timed(
                    engine, item, index, sampling_kwargs, enable_thinking
                )
                for index, item in indexed_items
            ]
        )

    async def _run_refill_queue(
        self,
        engine: Any,
        indexed_items: list[tuple[int, dict]],
        *,
        batch_size: int,
        sampling_kwargs: Optional[dict],
        enable_thinking: bool,
        score_generated: Callable[[EvalGenerated], QuestionResult],
        on_progress: Optional[Callable[[int, int], Any]],
        total_items: int,
        completed: int = 0,
        score_concurrency: int = 1,
    ) -> tuple[list[QuestionResult], int]:
        if not indexed_items:
            return [], completed

        max_in_flight = max(1, int(batch_size or 1))
        scorer_count = max(1, int(score_concurrency or 1))
        pending: asyncio.Queue[tuple[int, dict] | None] = asyncio.Queue()
        generated: asyncio.Queue[EvalGenerated | None] = asyncio.Queue(
            maxsize=max_in_flight * 2
        )
        scored: list[tuple[int, QuestionResult]] = []

        for index, item in indexed_items:
            pending.put_nowait((index, item))
        for _ in range(max_in_flight):
            pending.put_nowait(None)

        async def generation_worker() -> None:
            while True:
                job = await pending.get()
                try:
                    if job is None:
                        return
                    index, item = job
                    output = await self._eval_single_timed(
                        engine, item, index, sampling_kwargs, enable_thinking
                    )
                    await generated.put(output)
                finally:
                    pending.task_done()

        async def generation_coordinator() -> None:
            await asyncio.gather(*[generation_worker() for _ in range(max_in_flight)])
            for _ in range(scorer_count):
                await generated.put(None)

        async def scoring_worker() -> None:
            nonlocal completed
            while True:
                output = await generated.get()
                try:
                    if output is None:
                        return
                    # Code benchmarks run subprocesses during scoring. Keep that
                    # off the event loop so generation slots can refill promptly.
                    question_result = await asyncio.to_thread(score_generated, output)
                    scored.append((output.index, question_result))
                    completed += 1
                    if on_progress:
                        try:
                            await on_progress(completed, total_items)
                        except asyncio.CancelledError as exc:
                            raise _EvalProgressCancelledError from exc
                finally:
                    generated.task_done()

        tasks = [
            asyncio.create_task(generation_coordinator()),
            *[asyncio.create_task(scoring_worker()) for _ in range(scorer_count)],
        ]
        try:
            done, pending_tasks = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
            if pending_tasks:
                await asyncio.gather(*pending_tasks)
        except _EvalProgressCancelledError as exc:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise asyncio.CancelledError from exc.__cause__
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        scored.sort(key=lambda result: result[0])
        return [result for _, result in scored], completed

    def _score_generic_generated(self, generated: EvalGenerated) -> QuestionResult:
        predicted = self.extract_answer(generated.response_text, generated.item)
        is_correct = self.check_answer(predicted, generated.item)
        cat = self.get_category(generated.item)
        q_id = generated.item.get("id", str(generated.index))
        expected = generated.item.get("answer", "")
        return QuestionResult(
            question_id=str(q_id),
            correct=is_correct,
            expected=str(expected),
            predicted=predicted,
            time_seconds=generated.generation_seconds,
            question_text=generated.prompt_text,
            raw_response=generated.response_text,
            category=cat,
        )

    async def run(
        self,
        engine: Any,
        items: list[dict],
        on_progress: Optional[Callable[[int, int], Any]] = None,
        batch_size: int = 1,
        sampling_kwargs: Optional[dict] = None,
        enable_thinking: bool = False,
    ) -> BenchmarkResult:
        """Run the benchmark on all items.

        Args:
            engine: oMLX engine instance with chat() method.
            items: Dataset items to evaluate.
            on_progress: Callback(current, total) for progress reporting.
            batch_size: Number of concurrent requests (1 = sequential).
            enable_thinking: Enable thinking mode for reasoning models.
                When False, auto-detects if the model outputs <think> tags
                and re-runs the first batch with thinking enabled.

        Returns:
            BenchmarkResult with accuracy and per-question details.
        """
        start_time = time.time()
        completed = 0

        thinking_used = enable_thinking
        results: list[QuestionResult] = []

        indexed_items = list(enumerate(items))
        remaining_items = indexed_items

        # Keep current first-batch thinking auto-detection behavior before the
        # queue starts. If thinking is detected, discard and rerun the probe.
        if indexed_items and not thinking_used:
            probe_size = min(max(1, int(batch_size or 1)), len(indexed_items))
            probe_items = indexed_items[:probe_size]
            probe_results = await self._run_probe_batch(
                engine, probe_items, sampling_kwargs, False
            )
            has_think_tags = any(
                "<think>" in result.raw_text for result in probe_results
            )
            if has_think_tags:
                logger.warning(
                    f"{self.name}: model outputs <think> tags with "
                    "enable_thinking=False, auto-switching to thinking mode"
                )
                thinking_used = True
            else:
                for question_result in await asyncio.gather(
                    *[
                        asyncio.to_thread(self._score_generic_generated, result)
                        for result in sorted(
                            probe_results, key=lambda output: output.index
                        )
                    ]
                ):
                    results.append(question_result)
                    completed += 1
                    if on_progress:
                        await on_progress(completed, len(items))
                remaining_items = indexed_items[probe_size:]

        queued_results, completed = await self._run_refill_queue(
            engine,
            remaining_items,
            batch_size=batch_size,
            sampling_kwargs=sampling_kwargs,
            enable_thinking=thinking_used,
            score_generated=self._score_generic_generated,
            on_progress=on_progress,
            total_items=len(items),
            completed=completed,
        )
        results.extend(queued_results)

        correct = sum(1 for result in results if result.correct)
        category_correct: dict[str, int] = {}
        category_total: dict[str, int] = {}
        for result in results:
            if result.category is not None:
                category_total[result.category] = (
                    category_total.get(result.category, 0) + 1
                )
                if result.correct:
                    category_correct[result.category] = (
                        category_correct.get(result.category, 0) + 1
                    )

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
            thinking_used=thinking_used,
        )
