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
    # Populated only for external API evaluations.
    status: Optional[str] = None
    finish_reason: Optional[str] = None
    reasoning_fields_present: list[str] = field(default_factory=list)
    reasoning_fields_nonempty: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error_message: str = ""


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


class BaseBenchmark(ABC):
    """Abstract base class for accuracy benchmarks."""

    name: str = ""
    quick_size: int = 100
    # Full dataset size, recorded by load_dataset before sampling. Uploaded to
    # omlx.ai so "300 of 14,042" reads correctly on the community leaderboard.
    dataset_total: Optional[int] = None
    # run() probes the first questions for <think> tags and switches thinking
    # on when it finds them; code benchmarks keep the caller's setting.
    auto_detect_thinking: bool = True
    # Result rendering: fixed "expected" label and a cap on the predicted text.
    expected_label: Optional[str] = None
    predicted_max_chars: Optional[int] = None
    # Code execution blocks; keep it off the engine event loop.
    blocking_scoring: bool = False

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
        # response_upper is upper-cased, so the lower-case cue needs IGNORECASE.
        answer_patterns = re.findall(
            r"(?:answer\s*(?:is|:)\s*)([" + pattern_letters + r"])\b",
            response_upper,
            re.IGNORECASE,
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
        """Remove completed thinking spans from model output."""
        # Some chat templates open <think> in the prompt, so non-streaming
        # generation contains only ``reasoning</think>answer``.  Handle that
        # shape before the complete-block regex so reasoning drafts cannot
        # leak into answer extraction.
        if "<think>" not in text and "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _classify_response(
        self,
        response_text: str,
        item: dict,
        diagnostics: dict[str, Any],
    ) -> tuple[str, bool, Optional[str]]:
        """Score a response while preserving legacy local-evaluation behavior."""
        external_status = diagnostics.get("status")
        if external_status is None:
            predicted = self.extract_answer(response_text, item)
            return predicted, self.check_answer(predicted, item), None

        if external_status != "ok":
            return "", False, external_status

        predicted = self.extract_answer(response_text, item)
        if not predicted:
            return "", False, "parse_error"
        is_correct = self.check_answer(predicted, item)
        return predicted, is_correct, "correct" if is_correct else "wrong"

    @staticmethod
    def _diagnostic_result_fields(diagnostics: dict[str, Any]) -> dict[str, Any]:
        """Map internal external diagnostics onto QuestionResult fields."""
        return {
            "status": diagnostics.get("status"),
            "finish_reason": diagnostics.get("finish_reason"),
            "reasoning_fields_present": diagnostics.get(
                "reasoning_fields_present", []
            ),
            "reasoning_fields_nonempty": diagnostics.get(
                "reasoning_fields_nonempty", []
            ),
            "prompt_tokens": diagnostics.get("prompt_tokens", 0),
            "completion_tokens": diagnostics.get("completion_tokens", 0),
            "error_message": diagnostics.get("error_message", ""),
        }

    async def _eval_single(
        self, engine: Any, item: dict, index: int,
        sampling_kwargs: Optional[dict] = None,
        enable_thinking: bool = False,
    ) -> tuple[int, dict, str, str, str, dict[str, Any]]:
        """Evaluate a single item.

        Returns (index, item, response_text, prompt_text, raw_text, diagnostics).
        raw_text is the unstripped output for auto-detection of thinking tags.
        diagnostics is empty for local engines and contains external response
        metadata for remote evaluations.
        """
        messages = self.format_prompt(item)
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        kwargs = dict(sampling_kwargs or {})
        # max_tokens is always benchmark-controlled — a model's small configured
        # limit must not truncate long answers and corrupt scores.
        max_tokens = self.get_max_tokens()
        # Harmony models (gpt_oss) use analysis + final channels;
        # analysis can consume the entire budget before final is emitted
        if getattr(engine, "model_type", None) == "gpt_oss":
            max_tokens = max(max_tokens * 4, 8192)
        elif enable_thinking:
            max_tokens = min(
                max(max_tokens, THINKING_MIN_TOKENS), THINKING_MAX_TOKENS
            )
        kwargs["max_tokens"] = max_tokens
        # Greedy/neutral defaults keep scores reproducible. setdefault (not
        # force-set) lets the caller's "model_settings" sampling profile supply
        # its own temperature/penalties; the default profile passes none, so
        # these fall through to deterministic values.
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
            diagnostics: dict[str, Any] = {}
            if getattr(engine, "is_external_api", False):
                diagnostics = {
                    "status": getattr(output, "external_status", "invalid_response"),
                    "finish_reason": getattr(output, "finish_reason", None),
                    "reasoning_fields_present": list(
                        getattr(output, "reasoning_fields_present", ())
                    ),
                    "reasoning_fields_nonempty": list(
                        getattr(output, "reasoning_fields_nonempty", ())
                    ),
                    "prompt_tokens": int(getattr(output, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        getattr(output, "completion_tokens", 0) or 0
                    ),
                    "error_message": getattr(output, "error_message", ""),
                }
            return index, item, text, prompt_text, raw_text, diagnostics
        except Exception as e:
            logger.warning(f"Engine error on question {index}: {e}")
            diagnostics = {}
            if getattr(engine, "is_external_api", False):
                diagnostics = {
                    "status": getattr(e, "status", "invalid_response"),
                    "finish_reason": None,
                    "reasoning_fields_present": [],
                    "reasoning_fields_nonempty": [],
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "error_message": str(e),
                }
            return index, item, "", prompt_text, "", diagnostics

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
            on_progress: Callback(current, total), awaited after every
                answered question.
            batch_size: Questions in flight at once. A worker pool keeps that
                many requests running and starts the next question as soon
                as one finishes (1 = sequential).
            enable_thinking: Enable thinking mode for reasoning models.
                When False, the first batch_size questions act as a probe:
                if any of them outputs <think> tags, thinking is switched on
                and every question answered so far is re-run, so no
                mixed-mode results remain.

        Returns:
            BenchmarkResult with accuracy and per-question details.
        """
        start_time = time.time()
        total = len(items)
        thinking_used = enable_thinking
        probe_count = 0
        if (
            self.auto_detect_thinking
            and not enable_thinking
            and not getattr(engine, "is_external_api", False)
        ):
            probe_count = batch_size

        todo: asyncio.Queue[int] = asyncio.Queue()
        for idx in range(total):
            todo.put_nowait(idx)
        done: asyncio.Queue[Any] = asyncio.Queue()

        async def worker() -> None:
            while True:
                idx = await todo.get()
                thinking = thinking_used
                started = time.time()
                try:
                    outcome = await self._eval_single(
                        engine, items[idx], idx, sampling_kwargs, thinking
                    )
                except Exception as exc:
                    done.put_nowait(exc)
                    return
                done.put_nowait((thinking, time.time() - started, outcome))

        results: dict[int, QuestionResult] = {}
        workers = [
            asyncio.create_task(worker()) for _ in range(min(batch_size, total))
        ]
        scoring_task: asyncio.Task[QuestionResult] | None = None
        try:
            while len(results) < total:
                message = await done.get()
                if isinstance(message, Exception):
                    raise message
                thinking, elapsed, outcome = message
                idx, item, response_text, prompt_text, raw_text, diagnostics = outcome
                if thinking != thinking_used:
                    # Answered under the mode that was current when it
                    # started; the switch below made that answer stale.
                    todo.put_nowait(idx)
                    continue
                if idx < probe_count and not thinking_used and "<think>" in raw_text:
                    logger.warning(
                        f"{self.name}: model outputs <think> tags with "
                        "enable_thinking=False, auto-switching to thinking mode"
                    )
                    thinking_used = True
                    for stale in sorted([*results, idx]):
                        todo.put_nowait(stale)
                    results.clear()
                    continue
                args = (idx, item, response_text, prompt_text, diagnostics, elapsed)
                if self.blocking_scoring:
                    scoring_task = asyncio.create_task(
                        asyncio.to_thread(self._question_result, *args)
                    )
                    results[idx] = await asyncio.shield(scoring_task)
                    scoring_task = None
                else:
                    results[idx] = self._question_result(*args)
                if on_progress:
                    await on_progress(len(results), total)
        finally:
            for task in workers:
                task.cancel()

            async def drain() -> None:
                await asyncio.gather(*workers, return_exceptions=True)
                # Cancelling to_thread does not stop its subprocess. Drain the
                # current score before leaving; never start another score here.
                if scoring_task is not None:
                    await asyncio.gather(scoring_task, return_exceptions=True)

            cleanup = asyncio.create_task(drain())
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    # Repeated cancellation must not detach cleanup work.
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError()

        ordered = [results[idx] for idx in range(total)]
        correct = sum(1 for r in ordered if r.correct)
        category_correct: dict[str, int] = {}
        category_total: dict[str, int] = {}
        for r in ordered:
            if r.category is None:
                continue
            category_total[r.category] = category_total.get(r.category, 0) + 1
            if r.correct:
                category_correct[r.category] = (
                    category_correct.get(r.category, 0) + 1
                )

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
            accuracy=correct / total if total > 0 else 0.0,
            total_questions=total,
            correct_count=correct,
            time_seconds=time.time() - start_time,
            question_results=ordered,
            category_scores=cat_scores,
            thinking_used=thinking_used,
        )

    def _question_result(
        self,
        idx: int,
        item: dict,
        response_text: str,
        prompt_text: str,
        diagnostics: dict[str, Any],
        elapsed: float,
    ) -> QuestionResult:
        predicted, is_correct, question_status = self._classify_response(
            response_text, item, diagnostics
        )
        result_diagnostics = self._diagnostic_result_fields(diagnostics)
        result_diagnostics["status"] = question_status

        expected = self.expected_label
        if expected is None:
            expected = str(item.get("answer", ""))
        limit = self.predicted_max_chars
        if limit is not None and len(predicted) > limit:
            predicted = predicted[:limit] + "..."

        return QuestionResult(
            question_id=str(item.get("id", idx)),
            correct=is_correct,
            expected=expected,
            predicted=predicted,
            time_seconds=elapsed,
            question_text=prompt_text,
            raw_response=response_text,
            category=self.get_category(item),
            **result_diagnostics,
        )
