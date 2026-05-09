# SPDX-License-Identifier: Apache-2.0
"""HumanEval benchmark.

Tests code generation ability using function completion problems.
Model receives a function signature + docstring and must complete the body.
Verification: generated code + unit tests run in sandboxed subprocess.
Dataset bundled from openai/openai_humaneval on HuggingFace (164 problems).

SECURITY NOTE: This benchmark executes model-generated code on the local
machine. Mitigations: subprocess with timeout, memory limits, temp file cleanup.
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

EXEC_TIMEOUT_SECONDS = 15
EXEC_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024  # 256 MB


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
            if imports and not any(line.strip().startswith(("import ", "from ")) for line in code.split("\n")):
                return imports + "\n\n" + code
            return code
        return prompt + code

    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "def " in code:
            if imports and not any(line.strip().startswith(("import ", "from ")) for line in code.split("\n")):
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


def _set_resource_limits():
    """Set resource limits for subprocess."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (EXEC_MEMORY_LIMIT_BYTES, EXEC_MEMORY_LIMIT_BYTES))
    except (ValueError, resource.error):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (EXEC_TIMEOUT_SECONDS + 5, EXEC_TIMEOUT_SECONDS + 5))
    except (ValueError, resource.error):
        pass


def _execute_with_tests(code: str, test_code: str, entry_point: str) -> tuple[bool, str]:
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


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval: function completion with unit test verification."""

    name = "humaneval"
    quick_size = 100

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load HumanEval from bundled data."""
        items = load_jsonl(DATA_DIR / "humaneval.jsonl")

        normalized = []
        for item in items:
            normalized.append({
                "id": item["task_id"],
                "prompt": item["prompt"],
                "test": item["test"],
                "entry_point": item["entry_point"],
                "question": item["prompt"],  # for get_question_text
            })

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
            "Provide only the complete function implementation, no explanations.\n\n"
            f"{prompt}"
        )
        return [{"role": "user", "content": content}]

    def extract_answer(self, response: str, item: dict) -> str:
        """Extract the complete function from model response."""
        # Use last code block to avoid picking drafts/examples
        code = self._extract_last_code_block(response)
        imports = _get_imports(item["prompt"])

        # If extracted code has function def but no imports, prepend from prompt
        if "def " in code and imports:
            if not any(line.strip().startswith(("import ", "from ")) for line in code.split("\n")):
                return imports + "\n\n" + code

        # If no function def found, combine prompt + response body
        if "def " not in code:
            return item["prompt"] + code

        return code

    def check_answer(self, predicted: str, item: dict) -> bool:
        """Execute the generated code with test cases."""
        if not predicted.strip():
            return False

        passed, error = _execute_with_tests(
            predicted, item["test"], item["entry_point"]
        )
        return passed

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

            gen_tasks = [
                self._eval_single(engine, item, batch_start + j, sampling_kwargs, enable_thinking)
                for j, item in enumerate(batch)
            ]
            gen_results = await asyncio.gather(*gen_tasks)
            gen_elapsed = time.time() - batch_time

            for idx, item, response_text, prompt_text, _raw in sorted(gen_results, key=lambda x: x[0]):
                code = self.extract_answer(response_text, item)
                is_correct = self.check_answer(code, item)

                if is_correct:
                    correct += 1

                results.append(
                    QuestionResult(
                        question_id=str(item.get("id", idx)),
                        correct=is_correct,
                        expected="(unit tests)",
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
