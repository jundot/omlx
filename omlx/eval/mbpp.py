# SPDX-License-Identifier: Apache-2.0
"""MBPP (Mostly Basic Python Problems) benchmark.

Tests code generation with natural language descriptions and assertion tests.
Dataset bundled from google-research-datasets/mbpp (full test) on HuggingFace.
500 problems with assert-based test cases.

SECURITY NOTE: This benchmark executes model-generated code on the local
machine. Mitigations: subprocess with timeout, memory limits, temp file cleanup.
"""

import ast
import logging
import os
import re
import resource
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .base import BaseBenchmark, BenchmarkResult, EvalGenerated, QuestionResult
from .datasets import deterministic_sample, load_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

EXEC_TIMEOUT_SECONDS = 15
EXEC_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024


@dataclass
class CodeCheckResult:
    passed: bool
    failure_type: str = ""
    error: str = ""
    pass_mode: str = ""


def _extract_code(response: str) -> str:
    """Extract Python code from model response."""
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = response.strip().split("\n")
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

    return "\n".join(code_lines) if code_lines else response.strip()


def _set_resource_limits():
    try:
        resource.setrlimit(
            resource.RLIMIT_AS, (EXEC_MEMORY_LIMIT_BYTES, EXEC_MEMORY_LIMIT_BYTES)
        )
    except (ValueError, resource.error):
        pass
    try:
        resource.setrlimit(
            resource.RLIMIT_CPU, (EXEC_TIMEOUT_SECONDS + 5, EXEC_TIMEOUT_SECONDS + 5)
        )
    except (ValueError, resource.error):
        pass


def _classify_error(error: str) -> str:
    if not error:
        return "wrong_answer"
    if "IndentationError" in error:
        return "indentation_error"
    if "SyntaxError" in error:
        return "syntax_error"
    if "NameError" in error and "is not defined" in error:
        return "missing_entry_point"
    if "AssertionError" in error:
        return "wrong_answer"
    if "timed out" in error.lower():
        return "timeout"
    return "runtime_error"


_TOLERANT_ASSERT_HELPER = r"""
import math as _omlx_math

def _omlx_close_equal(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _omlx_math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_omlx_close_equal(x, y) for x, y in zip(a, b))
    return a == b
"""


def _tolerant_assert_tests(test_list: list[str]) -> list[str]:
    """Rewrite simple equality asserts to tolerate tiny numeric drift."""
    rewritten = []
    for test in test_list:
        try:
            module = ast.parse(test)
        except SyntaxError:
            rewritten.append(test)
            continue
        if (
            len(module.body) == 1
            and isinstance(module.body[0], ast.Assert)
            and isinstance(module.body[0].test, ast.Compare)
            and len(module.body[0].test.ops) == 1
            and isinstance(module.body[0].test.ops[0], ast.Eq)
            and len(module.body[0].test.comparators) == 1
        ):
            compare = module.body[0].test
            left = ast.unparse(compare.left)
            right = ast.unparse(compare.comparators[0])
            rewritten.append(f"assert _omlx_close_equal({left}, {right})")
        else:
            rewritten.append(test)
    return rewritten


def _execute_with_tests(
    code: str,
    test_list: list[str],
    setup_code: str = "",
    tolerant_numeric_asserts: bool = False,
    setup_after_code: bool = False,
) -> tuple[bool, str]:
    """Execute generated code with assertion-based test cases."""
    setup = setup_code or ""
    if tolerant_numeric_asserts:
        test_code = "\n".join(_tolerant_assert_tests(test_list))
        helper = _TOLERANT_ASSERT_HELPER
    else:
        test_code = "\n".join(test_list)
        helper = ""

    if setup_after_code:
        script = f"{code}\n{setup}\n{helper}\n{test_code}\n"
    else:
        script = f"{setup}\n{code}\n{helper}\n{test_code}\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
            preexec_fn=_set_resource_limits,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
                "HOME": os.environ.get("HOME", "/tmp"),
                "LANG": "en_US.UTF-8",
            },
        )
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr[:500]
    except subprocess.TimeoutExpired:
        return False, "Execution timed out"
    except Exception as e:
        return False, str(e)[:500]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _run_with_tests(
    code: str, test_list: list[str], setup_code: str = ""
) -> CodeCheckResult:
    passed, error = _execute_with_tests(code, test_list, setup_code)
    if passed:
        return CodeCheckResult(
            passed=True,
            failure_type="passed",
            pass_mode="standalone_code",
        )

    best_error = error
    best_failure = _classify_error(error)

    if setup_code.strip():
        reordered_passed, reordered_error = _execute_with_tests(
            code,
            test_list,
            setup_code,
            setup_after_code=True,
        )
        if reordered_passed:
            return CodeCheckResult(
                passed=True,
                failure_type="passed",
                pass_mode="setup_after_code",
            )
        reordered_failure = _classify_error(reordered_error)
        if best_failure in (
            "syntax_error",
            "indentation_error",
            "missing_entry_point",
        ) and (
            reordered_failure
            not in ("syntax_error", "indentation_error", "missing_entry_point")
        ):
            best_error = reordered_error
            best_failure = reordered_failure

    if best_failure == "wrong_answer":
        for setup_after_code, pass_mode in (
            (False, "tolerant_numeric_asserts"),
            (True, "setup_after_code_tolerant_numeric_asserts"),
        ):
            if setup_after_code and not setup_code.strip():
                continue
            tolerant_passed, tolerant_error = _execute_with_tests(
                code,
                test_list,
                setup_code,
                tolerant_numeric_asserts=True,
                setup_after_code=setup_after_code,
            )
            if tolerant_passed:
                return CodeCheckResult(
                    passed=True,
                    failure_type="passed",
                    pass_mode=pass_mode,
                )
            best_error = tolerant_error or best_error
    return CodeCheckResult(
        passed=False,
        failure_type=_classify_error(best_error),
        error=best_error,
    )


class MBPPBenchmark(BaseBenchmark):
    """MBPP: code generation with assertion-based test verification."""

    name = "mbpp"
    quick_size = 200

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load MBPP from bundled data."""
        items = load_jsonl(DATA_DIR / "mbpp.jsonl")

        normalized = []
        for item in items:
            test_list = item.get("test_list", [])
            if not test_list:
                continue
            normalized.append(
                {
                    "id": str(item["task_id"]),
                    "prompt": item["prompt"],
                    "test_list": test_list,
                    "test_setup_code": item.get("test_setup_code", ""),
                    "question": item["prompt"],
                }
            )

        logger.info(f"MBPP: loaded {len(normalized)} problems")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def get_max_tokens(self) -> int:
        return 2048

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as a code generation prompt with test cases for function name."""
        prompt = item["prompt"]
        tests = item.get("test_list", [])
        test_str = "\n".join(tests[:3])
        content = (
            "Write a Python function to solve the following problem. "
            "Provide only the complete function implementation, no explanations.\n\n"
            f"Problem: {prompt}\n\n"
            f"Test cases:\n{test_str}\n\n"
            "Solution:"
        )
        return [{"role": "user", "content": content}]

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_last_code_block(response)

    def check_answer(self, predicted: str, item: dict) -> bool:
        if not predicted.strip():
            return False

        check = _run_with_tests(
            predicted,
            item["test_list"],
            item.get("test_setup_code", ""),
        )
        return check.passed

    async def run(
        self,
        engine: Any,
        items: list[dict],
        on_progress: Optional[Callable[[int, int], Any]] = None,
        batch_size: int = 1,
        sampling_kwargs: Optional[dict] = None,
        enable_thinking: bool = False,
    ) -> BenchmarkResult:
        """Override run: generation is queued, code execution is sequential."""
        start_time = time.time()

        def score_generated(generated: EvalGenerated) -> QuestionResult:
            code = self.extract_answer(generated.response_text, generated.item)
            check = _run_with_tests(
                code,
                generated.item["test_list"],
                generated.item.get("test_setup_code", ""),
            )
            is_correct = check.passed
            return QuestionResult(
                question_id=str(generated.item.get("id", generated.index)),
                correct=is_correct,
                expected="(test cases)",
                predicted=code[:200] + "..." if len(code) > 200 else code,
                time_seconds=generated.generation_seconds,
                question_text=generated.prompt_text,
                raw_response=generated.response_text,
                category=self.get_category(generated.item),
                pass_mode=check.pass_mode if is_correct else None,
                failure_type=check.failure_type,
                error=check.error,
            )

        results, _ = await self._run_refill_queue(
            engine,
            list(enumerate(items)),
            batch_size=batch_size,
            sampling_kwargs=sampling_kwargs,
            enable_thinking=enable_thinking,
            score_generated=score_generated,
            on_progress=on_progress,
            total_items=len(items),
            score_concurrency=1,
        )

        total_time = time.time() - start_time
        total = len(items)
        correct = sum(1 for result in results if result.correct)

        return BenchmarkResult(
            benchmark_name=self.name,
            accuracy=correct / total if total > 0 else 0.0,
            total_questions=total,
            correct_count=correct,
            time_seconds=total_time,
            question_results=results,
            thinking_used=enable_thinking,
        )
