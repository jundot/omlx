# SPDX-License-Identifier: Apache-2.0
"""MBPP (Mostly Basic Python Problems) benchmark.

Tests code generation with natural language descriptions and assertion tests.
Dataset bundled from google-research-datasets/mbpp (full test) on HuggingFace.
500 problems with assert-based test cases.

SECURITY NOTE: This benchmark executes model-generated code on the local
machine. Mitigations: subprocess with timeout, memory limits, temp file cleanup.
"""

import asyncio
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

EXEC_TIMEOUT_SECONDS = 15
EXEC_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024


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
        resource.setrlimit(resource.RLIMIT_AS, (EXEC_MEMORY_LIMIT_BYTES, EXEC_MEMORY_LIMIT_BYTES))
    except (ValueError, resource.error):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (EXEC_TIMEOUT_SECONDS + 5, EXEC_TIMEOUT_SECONDS + 5))
    except (ValueError, resource.error):
        pass


def _execute_with_tests(code: str, test_list: list[str], setup_code: str = "") -> tuple[bool, str]:
    """Execute generated code with assertion-based test cases."""
    test_code = "\n".join(test_list)
    script = f"{setup_code}\n{code}\n{test_code}\n"

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
            normalized.append({
                "id": str(item["task_id"]),
                "prompt": item["prompt"],
                "test_list": test_list,
                "test_setup_code": item.get("test_setup_code", ""),
                "question": item["prompt"],
            })

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

        passed, error = _execute_with_tests(
            predicted,
            item["test_list"],
            item.get("test_setup_code", ""),
        )
        return passed

    async def run(
        self,
        engine: Any,
        items: list[dict],
        on_progress: Optional[Callable[[int, int], Any]] = None,
        batch_size: int = 1,
        sampling_kwargs: Optional[dict] = None,
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

            gen_tasks = [
                self._eval_single(engine, item, batch_start + j, sampling_kwargs)
                for j, item in enumerate(batch)
            ]
            gen_results = await asyncio.gather(*gen_tasks)
            gen_elapsed = time.time() - batch_time

            for idx, item, response_text, prompt_text in sorted(gen_results, key=lambda x: x[0]):
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
        )
