# SPDX-License-Identifier: Apache-2.0
"""LiveCodeBench benchmark.

Tests code generation ability using competitive programming problems.
Generates code, executes it in a sandboxed subprocess, and checks output.
Dataset bundled from livecodebench/code_generation_lite on HuggingFace.

SECURITY NOTE: This benchmark executes model-generated code on the local
machine. Mitigations: subprocess with timeout, memory limits via resource
module, temp file cleanup. Users are warned in the UI before running.
"""

import asyncio
import json
import logging
import os
import re
import resource
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .base import BaseBenchmark, BenchmarkResult, QuestionResult
from .datasets import deterministic_sample, load_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# Execution limits
EXEC_TIMEOUT_SECONDS = 30
EXEC_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024  # 256 MB


def _extract_code(response: str) -> str:
    """Extract Python code from model response.

    Looks for ```python...``` blocks first, then ```...``` blocks,
    then falls back to the entire response.
    """
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
    """Set resource limits for subprocess. Called via preexec_fn."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (EXEC_MEMORY_LIMIT_BYTES, EXEC_MEMORY_LIMIT_BYTES))
    except (ValueError, resource.error):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (EXEC_TIMEOUT_SECONDS + 5, EXEC_TIMEOUT_SECONDS + 5))
    except (ValueError, resource.error):
        pass


def _execute_code(code: str, stdin_input: str = "") -> tuple[str, bool, str]:
    """Execute Python code in a subprocess with safety limits.

    Returns:
        (stdout, success, error_message)
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            input=stdin_input,
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
            return result.stdout, True, ""
        else:
            return result.stdout, False, result.stderr[:500]
    except subprocess.TimeoutExpired:
        return "", False, "Execution timed out"
    except Exception as e:
        return "", False, str(e)[:500]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class LiveCodeBenchBenchmark(BaseBenchmark):
    """LiveCodeBench: code generation with sandboxed execution."""

    name = "livecodebench"
    quick_size = 100

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load LiveCodeBench from bundled data."""
        items = load_jsonl(DATA_DIR / "livecodebench.jsonl")

        normalized = []
        for i, item in enumerate(items):
            test_cases_str = item.get("public_test_cases", "[]")
            if isinstance(test_cases_str, str):
                try:
                    test_cases = json.loads(test_cases_str)
                except (json.JSONDecodeError, TypeError):
                    test_cases = []
            else:
                test_cases = test_cases_str

            if not isinstance(test_cases, list) or not test_cases:
                continue

            inputs = [tc.get("input", "") for tc in test_cases]
            outputs = [tc.get("output", "") for tc in test_cases]

            if not inputs or not outputs:
                continue

            normalized.append({
                "id": item.get("question_id", str(i)),
                "title": item.get("question_title", f"Problem {i}"),
                "description": item.get("question_content", ""),
                "inputs": inputs,
                "outputs": outputs,
                "difficulty": item.get("difficulty", ""),
                "starter_code": item.get("starter_code", ""),
            })

        logger.info(f"LiveCodeBench: loaded {len(normalized)} problems")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def get_max_tokens(self) -> int:
        return 16384

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as a coding problem prompt."""
        description = item["description"]
        prompt = (
            "Solve the following programming problem in Python. "
            "Read input from stdin and print the output to stdout. "
            "Provide only the complete Python code, no explanations.\n\n"
            f"Problem:\n{description}\n\n"
            "Solution:"
        )
        return [{"role": "user", "content": prompt}]

    def extract_answer(self, response: str, item: dict) -> str:
        """Extract code from the response (last code block to skip drafts)."""
        return self._extract_last_code_block(response)

    def check_answer(self, predicted: str, item: dict) -> bool:
        """Execute code and check against test cases.

        Runs the first 3 test cases to keep execution time reasonable.
        """
        if not predicted.strip():
            return False

        inputs = item["inputs"][:3]
        outputs = item["outputs"][:3]

        for inp, expected_out in zip(inputs, outputs):
            stdin_input = inp if isinstance(inp, str) else str(inp)
            expected = expected_out.strip() if isinstance(expected_out, str) else str(expected_out).strip()

            stdout, success, error = _execute_code(predicted, stdin_input)
            if not success:
                return False

            actual = stdout.strip()
            if actual != expected:
                return False

        return True

    async def run(
        self,
        engine: Any,
        items: list[dict],
        on_progress: Optional[Callable[[int, int], Any]] = None,
        batch_size: int = 1,
        sampling_kwargs: Optional[dict] = None,
        enable_thinking: bool = False,
    ) -> BenchmarkResult:
        """Override run: generation is batched, code execution is sequential."""
        results: list[QuestionResult] = []
        correct = 0
        start_time = time.time()
        completed = 0

        for batch_start in range(0, len(items), batch_size):
            batch_end = min(batch_start + batch_size, len(items))
            batch = items[batch_start:batch_end]
            batch_time = time.time()

            # Batch the generation phase
            gen_tasks = [
                self._eval_single(engine, item, batch_start + j, sampling_kwargs, enable_thinking)
                for j, item in enumerate(batch)
            ]
            gen_results = await asyncio.gather(*gen_tasks)
            gen_elapsed = time.time() - batch_time

            # Code execution is sequential (subprocess safety)
            for idx, item, response_text, prompt_text, _raw in sorted(gen_results, key=lambda x: x[0]):
                code = self.extract_answer(response_text, item)
                is_correct = self.check_answer(code, item)

                if is_correct:
                    correct += 1

                results.append(
                    QuestionResult(
                        question_id=str(item.get("id", idx)),
                        correct=is_correct,
                        expected="(test cases)",
                        predicted=code[:200] + "..." if len(code) > 200 else code,
                        time_seconds=gen_elapsed / len(batch),
                        question_text=prompt_text,
                        raw_response=response_text,
                        category=self.get_category(item),
                    )
                )

            completed += len(batch)
            if on_progress:
                await on_progress(completed, len(items))

        total_time = time.time() - start_time
        total = len(items)

        return BenchmarkResult(
            benchmark_name=self.name,
            accuracy=correct / total if total > 0 else 0.0,
            total_questions=total,
            correct_count=correct,
            time_seconds=total_time,
            question_results=results,
            thinking_used=enable_thinking,
        )
