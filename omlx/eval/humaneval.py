# SPDX-License-Identifier: Apache-2.0
"""HumanEval benchmark.

Tests code generation ability using function completion problems.
Model receives a function signature + docstring and must complete the body.
Verification: generated code + unit tests run in sandboxed subprocess.
Dataset bundled from openai/openai_humaneval on HuggingFace (164 problems).

SECURITY NOTE: This benchmark executes model-generated code on the local
machine. Mitigations: subprocess with timeout, memory limits, temp file cleanup.
"""

import logging
import os
import re
import resource
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .base import BaseBenchmark, BenchmarkResult, EvalGenerated, QuestionResult
from .datasets import deterministic_sample, load_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

EXEC_TIMEOUT_SECONDS = 15
EXEC_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024  # 256 MB


@dataclass
class CodeCheckResult:
    passed: bool
    failure_type: str = ""
    error: str = ""
    pass_mode: str = ""
    code: str = ""


def _get_imports(prompt: str) -> str:
    """Extract import lines from the prompt."""
    lines = []
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            lines.append(line)
    return "\n".join(lines)


def _extract_code(response: str, prompt: str) -> str:
    """Extract the function body from model response.

    The model may return the full function (including signature) or just the body.
    We need to combine it with the original prompt to form a complete function.
    Always prepends imports from the prompt to avoid NameError.
    """
    response = response.strip()
    imports = _get_imports(prompt)

    # If response contains a code block, extract it
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "def " in code:
            # Model included full function — prepend imports if missing
            if imports and not any(
                line.strip().startswith(("import ", "from "))
                for line in code.split("\n")
            ):
                return imports + "\n\n" + code
            return code
        return prompt + code

    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "def " in code:
            if imports and not any(
                line.strip().startswith(("import ", "from "))
                for line in code.split("\n")
            ):
                return imports + "\n\n" + code
            return code
        return prompt + code

    # No code block — response is the continuation of the prompt
    if response.startswith("def "):
        # Model repeated the function def — prepend imports
        if imports:
            return imports + "\n\n" + response
        return response
    if response.startswith("from ") or response.startswith("import "):
        return response

    # Response is just the function body — combine with prompt
    return prompt + response


def _classify_error(error: str, entry_point: str | None = None) -> str:
    if not error:
        return "wrong_answer"
    if "IndentationError" in error:
        return "indentation_error"
    if "SyntaxError" in error:
        return "syntax_error"
    if "NameError" in error and "is not defined" in error:
        if entry_point and f"name '{entry_point}' is not defined" in error:
            return "missing_entry_point"
        return "runtime_error"
    if "AssertionError" in error:
        return "wrong_answer"
    if "timed out" in error.lower():
        return "timeout"
    return "runtime_error"


def _has_imports(code: str) -> bool:
    return any(
        line.strip().startswith(("import ", "from ")) for line in code.split("\n")
    )


def _prepend_imports_if_needed(code: str, prompt: str) -> str:
    imports = _get_imports(prompt)
    if imports and "def " in code and not _has_imports(code):
        return imports + "\n\n" + code
    return code


def _extract_response_code(response: str) -> str:
    """Extract code while preserving body indentation where possible."""
    response = response.strip("\n")
    blocks = re.findall(
        r"(?ms)^[ \t]*`{3,}[ \t]*(?:python|py)[^\n]*\n(.*?)[ \t]*`{3,}[ \t]*$",
        response,
    )
    if blocks:
        return blocks[-1].strip("\n")
    blocks = re.findall(
        r"(?ms)^[ \t]*`{3,}[^\n]*\n(.*?)[ \t]*`{3,}[ \t]*$",
        response,
    )
    if blocks:
        return blocks[-1].strip("\n")
    return response


def _indent_body_if_needed(body: str) -> str:
    """Indent body-only HumanEval completions that arrived as top-level code."""
    body = body.strip("\n")
    lines = body.split("\n")
    first_code = next((line for line in lines if line.strip()), "")
    if not first_code or first_code[:1].isspace():
        return body
    return textwrap.indent(body, "    ")


def _indent_top_level_lines(body: str) -> str:
    """Repair completions where only top-level body lines lost indentation."""
    body = body.strip("\n")
    fixed = []
    for line in body.split("\n"):
        if line.strip() and not line[:1].isspace():
            fixed.append("    " + line)
        else:
            fixed.append(line)
    return "\n".join(fixed)


def _normalize_body_indentation(body: str) -> str:
    """Normalize completions with first-line-zero/rest-overindented shape.

    Some chat completions arrive as function bodies where top-level statements
    have inconsistent bases, e.g. the first line has 0 spaces while subsequent
    top-level lines have 8. Map that family to a normal function body:
    0 -> 4, 8 -> 4, 12 -> 8, preserving relative nested blocks.
    """
    body = body.strip("\n")
    lines = body.split("\n")
    nonzero_indents = []
    for line in lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent > 0:
            nonzero_indents.append(indent)

    if not nonzero_indents:
        return textwrap.indent(body, "    ")

    dedent_by = max(0, min(nonzero_indents) - 4)
    fixed = []
    for line in lines:
        if not line.strip():
            fixed.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            fixed.append("    " + line)
        elif dedent_by > 0 and indent >= dedent_by:
            fixed.append(line[dedent_by:])
        else:
            fixed.append(line)
    return "\n".join(fixed)


def _set_resource_limits():
    """Set resource limits for subprocess."""
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


def _execute_with_tests(
    code: str, test_code: str, entry_point: str
) -> tuple[bool, str]:
    """Execute generated code with test cases.

    Combines the generated function with test assertions and runs in subprocess.

    Returns:
        (passed, error_message)
    """
    # Build the complete test script
    script = f"""{code}

{test_code}

check({entry_point})
"""
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


def _run_with_tests(code: str, test_code: str, entry_point: str) -> CodeCheckResult:
    passed, error = _execute_with_tests(code, test_code, entry_point)
    if passed:
        return CodeCheckResult(passed=True, failure_type="passed", code=code)
    return CodeCheckResult(
        passed=False,
        failure_type=_classify_error(error, entry_point),
        error=error,
        code=code,
    )


def _candidate_codes(response: str, prompt: str) -> list[tuple[str, str]]:
    """Build deterministic HumanEval candidates from canonical and chat output."""
    extracted = _extract_response_code(response)
    candidates: list[tuple[str, str]] = []

    if "def " in extracted:
        candidates.append(
            ("standalone_function", _prepend_imports_if_needed(extracted, prompt))
        )
        candidates.append(("canonical_raw", prompt + response))
        if extracted != response:
            candidates.append(("canonical_extracted", prompt + extracted))
        candidates.append(
            (
                "canonical_normalize_indent",
                prompt + _normalize_body_indentation(extracted),
            )
        )
    else:
        candidates.append(("canonical_raw", prompt + response))
        if extracted != response:
            candidates.append(("canonical_extracted", prompt + extracted))
        candidates.append(
            ("canonical_indented", prompt + _indent_body_if_needed(extracted))
        )
        candidates.append(
            ("canonical_top_level_indent", prompt + _indent_top_level_lines(extracted))
        )
        candidates.append(
            (
                "canonical_normalize_indent",
                prompt + _normalize_body_indentation(extracted),
            )
        )

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for mode, code in candidates:
        if code not in seen:
            seen.add(code)
            unique.append((mode, code))
    return unique


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval: function completion with unit test verification."""

    name = "humaneval"
    quick_size = 100

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load HumanEval from bundled data."""
        items = load_jsonl(DATA_DIR / "humaneval.jsonl")

        normalized = []
        for item in items:
            normalized.append(
                {
                    "id": item["task_id"],
                    "prompt": item["prompt"],
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                    "question": item["prompt"],  # for get_question_text
                }
            )

        logger.info(f"HumanEval: loaded {len(normalized)} problems")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def get_max_tokens(self) -> int:
        return 2048

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as a function completion prompt."""
        prompt = item["prompt"]
        content = (
            "Complete the following Python function. "
            "Return only the missing indented function body, no explanations.\n\n"
            f"{prompt}"
        )
        return [{"role": "user", "content": content}]

    def extract_answer(self, response: str, item: dict) -> str:
        """Extract the complete function from model response."""
        result = self.evaluate_response(response, item)
        return result.code

    def check_answer(self, predicted: str, item: dict) -> bool:
        """Execute the generated code with test cases."""
        if not predicted.strip():
            return False

        passed, error = _execute_with_tests(
            predicted, item["test"], item["entry_point"]
        )
        return passed

    def evaluate_response(self, response: str, item: dict) -> CodeCheckResult:
        """Try canonical HumanEval completion plus deterministic chat repairs."""
        best_failure: CodeCheckResult | None = None
        for mode, code in _candidate_codes(response, item["prompt"]):
            result = _run_with_tests(code, item["test"], item["entry_point"])
            result.pass_mode = mode
            if result.passed:
                return result
            if best_failure is None or (
                best_failure.failure_type in ("syntax_error", "indentation_error")
                and result.failure_type not in ("syntax_error", "indentation_error")
            ):
                best_failure = result

        return best_failure or CodeCheckResult(
            passed=False,
            failure_type="runtime_error",
            error="No candidate code produced",
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
        """Override run: generation is queued, code execution is sequential."""
        start_time = time.time()

        def score_generated(generated: EvalGenerated) -> QuestionResult:
            check = self.evaluate_response(generated.response_text, generated.item)
            code = check.code
            is_correct = check.passed
            return QuestionResult(
                question_id=str(generated.item.get("id", generated.index)),
                correct=is_correct,
                expected="(unit tests)",
                predicted=code[:200] + "..." if len(code) > 200 else code,
                time_seconds=generated.generation_seconds,
                question_text=generated.prompt_text,
                raw_response=generated.response_text,
                category=self.get_category(generated.item),
                pass_mode=check.pass_mode,
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
