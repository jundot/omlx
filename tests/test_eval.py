# SPDX-License-Identifier: Apache-2.0
"""Unit tests for accuracy evaluation modules."""

import pytest

from omlx.eval.datasets import deterministic_sample, stratified_sample
from omlx.eval.gsm8k import GSM8KBenchmark, _extract_numeric_answer, _normalize_number
from omlx.eval.hellaswag import HellaSwagBenchmark
from omlx.eval.livecodebench import _extract_code
from omlx.eval.mmlu import MMLUBenchmark
from omlx.eval.truthfulqa import TruthfulQABenchmark

# --- MMLU Tests ---


class TestMMLU:
    def setup_method(self):
        self.bench = MMLUBenchmark()

    def test_extract_answer_simple_letter(self):
        assert self.bench.extract_answer("A", {}) == "A"
        assert self.bench.extract_answer("B", {}) == "B"
        assert self.bench.extract_answer("C", {}) == "C"
        assert self.bench.extract_answer("D", {}) == "D"

    def test_extract_answer_with_text(self):
        assert self.bench.extract_answer("The answer is B", {}) == "B"
        assert self.bench.extract_answer("A. Abstract algebra", {}) == "A"

    def test_extract_answer_verbose(self):
        assert (
            self.bench.extract_answer("I think the correct answer is C because...", {})
            == "C"
        )

    def test_extract_answer_empty(self):
        assert self.bench.extract_answer("", {}) == ""

    def test_extract_answer_no_match(self):
        assert self.bench.extract_answer("I don't know", {}) == ""

    def test_extract_answer_lowercase(self):
        assert self.bench.extract_answer("a", {}) == "A"
        assert self.bench.extract_answer("the answer is b", {}) == "B"

    def test_extract_answer_explanation_before_answer(self):
        """Model explains with wrong letters first, then gives correct answer."""
        assert (
            self.bench.extract_answer("B is wrong because... The answer is A", {})
            == "A"
        )
        assert (
            self.bench.extract_answer("I initially thought C but answer is D", {})
            == "D"
        )

    def test_extract_answer_last_letter(self):
        """When no 'answer is' pattern, use last valid letter."""
        assert self.bench.extract_answer("Looking at A and B, B is correct", {}) == "B"

    def test_check_answer_correct(self):
        assert self.bench.check_answer("A", {"answer": "A"}) is True

    def test_check_answer_incorrect(self):
        assert self.bench.check_answer("B", {"answer": "A"}) is False

    def test_check_answer_empty(self):
        assert self.bench.check_answer("", {"answer": "A"}) is False

    def test_format_prompt(self):
        self.bench._few_shot_examples = {
            "test_subject": [
                {
                    "question": "What is 2+2?",
                    "choices": ["3", "4", "5", "6"],
                    "answer": "B",
                }
            ]
        }
        item = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "3", "4"],
            "answer": "B",
            "subject": "test_subject",
        }
        messages = self.bench.format_prompt(item)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert "What is 1+1?" in content
        assert "A." in content
        assert "B." in content
        assert "Answer:" in content

    def test_get_category(self):
        assert self.bench.get_category({"subject": "math"}) == "math"
        assert self.bench.get_category({}) is None


# --- HellaSwag Tests ---


class TestHellaSwag:
    def setup_method(self):
        self.bench = HellaSwagBenchmark()

    def test_extract_answer(self):
        assert self.bench.extract_answer("A", {}) == "A"
        assert self.bench.extract_answer("B is correct", {}) == "B"
        assert self.bench.extract_answer("", {}) == ""

    def test_check_answer(self):
        # answer is 0-based index, expected letter is A
        assert self.bench.check_answer("A", {"answer": 0}) is True
        assert self.bench.check_answer("B", {"answer": 1}) is True
        assert self.bench.check_answer("A", {"answer": 1}) is False

    def test_format_prompt(self):
        item = {
            "context": "A man walks into a bar.",
            "endings": [
                "He orders a drink.",
                "He flies away.",
                "He disappears.",
                "He sings.",
            ],
            "answer": 0,
        }
        messages = self.bench.format_prompt(item)
        assert len(messages) == 1
        content = messages[0]["content"]
        assert "A man walks into a bar." in content
        assert "A." in content
        assert "He orders a drink." in content


# --- TruthfulQA Tests ---


class TestTruthfulQA:
    def setup_method(self):
        self.bench = TruthfulQABenchmark()

    def test_extract_answer(self):
        assert self.bench.extract_answer("A", {"choices": ["a", "b"]}) == "A"
        assert self.bench.extract_answer("B", {"choices": ["a", "b"]}) == "B"

    def test_check_answer(self):
        assert self.bench.check_answer("A", {"answer": 0}) is True
        assert self.bench.check_answer("B", {"answer": 0}) is False
        assert self.bench.check_answer("C", {"answer": 2}) is True


# --- GSM8K Tests ---


class TestGSM8K:
    def setup_method(self):
        self.bench = GSM8KBenchmark()

    def test_extract_numeric_answer_hash_pattern(self):
        assert _extract_numeric_answer("The answer is #### 42") == "42"
        assert _extract_numeric_answer("#### 1,234") == "1234"
        assert _extract_numeric_answer("So the answer is #### -5") == "-5"

    def test_extract_numeric_answer_fallback(self):
        assert _extract_numeric_answer("The answer is 42.") == "42"
        assert (
            _extract_numeric_answer("She has 15 apples and 20 oranges, so 35 total.")
            == "35"
        )

    def test_extract_numeric_answer_empty(self):
        assert _extract_numeric_answer("I don't know") == ""
        assert _extract_numeric_answer("") == ""

    def test_extract_numeric_answer_decimal(self):
        assert _extract_numeric_answer("#### 3.14") == "3.14"

    def test_normalize_number(self):
        assert _normalize_number("42") == "42"
        assert _normalize_number("42.0") == "42"
        assert _normalize_number("1,234") == "1234"
        assert _normalize_number("3.14") == "3.14"

    def test_check_answer(self):
        assert self.bench.check_answer("42", {"answer": "42"}) is True
        assert self.bench.check_answer("42.0", {"answer": "42"}) is True
        assert self.bench.check_answer("1234", {"answer": "1,234"}) is True
        assert self.bench.check_answer("43", {"answer": "42"}) is False
        assert self.bench.check_answer("", {"answer": "42"}) is False

    def test_format_prompt(self):
        item = {"question": "What is 2+2?", "answer": "4"}
        messages = self.bench.format_prompt(item)
        assert len(messages) == 1
        content = messages[0]["content"]
        assert "What is 2+2?" in content
        assert "####" in content  # Few-shot examples contain ####

    def test_get_max_tokens(self):
        assert self.bench.get_max_tokens() == 512


# --- LiveCodeBench Tests ---


class TestLiveCodeBench:
    def test_extract_code_python_block(self):
        response = (
            "Here's my solution:\n```python\ndef solve():\n    print(42)\n```\nDone."
        )
        code = _extract_code(response)
        assert "def solve():" in code
        assert "print(42)" in code

    def test_extract_code_generic_block(self):
        response = "```\nx = 1\nprint(x)\n```"
        code = _extract_code(response)
        assert "x = 1" in code

    def test_extract_code_no_block(self):
        response = "def solve():\n    n = int(input())\n    print(n * 2)"
        code = _extract_code(response)
        assert "def solve():" in code

    def test_extract_code_empty(self):
        code = _extract_code("")
        assert code == ""


# --- HumanEval Tests ---


class TestHumanEval:
    def test_extract_code_with_block(self):
        from omlx.eval.humaneval import _extract_code

        prompt = "def add(a, b):\n    "
        response = "```python\ndef add(a, b):\n    return a + b\n```"
        code = _extract_code(response, prompt)
        assert "return a + b" in code

    def test_extract_code_body_only(self):
        from omlx.eval.humaneval import _extract_code

        prompt = "def add(a, b):\n    "
        response = "return a + b"
        code = _extract_code(response, prompt)
        assert "def add(a, b):" in code
        assert "return a + b" in code

    def test_extract_code_preserves_imports(self):
        """Model returns def only — imports from prompt must be prepended."""
        from omlx.eval.humaneval import _extract_code

        prompt = "from typing import List\n\ndef foo(x: List[int]) -> int:\n    "
        response = "def foo(x: List[int]) -> int:\n    return sum(x)"
        code = _extract_code(response, prompt)
        assert "from typing import List" in code
        assert "return sum(x)" in code

    def test_execute_with_tests(self):
        from omlx.eval.humaneval import _execute_with_tests

        code = "def add(a, b):\n    return a + b"
        test = "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0"
        passed, error = _execute_with_tests(code, test, "add")
        assert passed is True

    def test_execute_with_tests_fail(self):
        from omlx.eval.humaneval import _execute_with_tests

        code = "def add(a, b):\n    return a - b"  # wrong
        test = "def check(candidate):\n    assert candidate(1, 2) == 3"
        passed, error = _execute_with_tests(code, test, "add")
        assert passed is False

    def test_body_only_without_indent_passes_after_repair(self):
        from omlx.eval.humaneval import HumanEvalBenchmark

        item = {
            "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
            "entry_point": "add",
        }
        result = HumanEvalBenchmark().evaluate_response("return a + b", item)
        assert result.passed is True
        assert result.pass_mode == "canonical_indented"

    def test_body_only_with_indent_passes_canonical_raw(self):
        from omlx.eval.humaneval import HumanEvalBenchmark

        item = {
            "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
            "entry_point": "add",
        }
        result = HumanEvalBenchmark().evaluate_response("    return a + b", item)
        assert result.passed is True
        assert result.pass_mode == "canonical_raw"

    def test_partially_stripped_body_indent_passes(self):
        from omlx.eval.humaneval import HumanEvalBenchmark

        item = {
            "prompt": (
                "def has_close_elements(numbers, threshold):\n"
                '    """Check close elements."""\n'
            ),
            "test": (
                "def check(candidate):\n"
                "    assert candidate([1.0, 2.8, 3.0], 0.3) is True\n"
                "    assert candidate([1.0, 2.0, 3.0], 0.5) is False"
            ),
            "entry_point": "has_close_elements",
        }
        response = (
            "for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False"
        )
        result = HumanEvalBenchmark().evaluate_response(response, item)
        assert result.passed is True
        assert result.pass_mode == "canonical_top_level_indent"

    def test_overindented_body_base_is_normalized(self):
        from omlx.eval.humaneval import HumanEvalBenchmark

        item = {
            "prompt": (
                "def sort_even(l: list):\n" '    """Sort values at even indexes."""\n'
            ),
            "test": (
                "def check(candidate):\n"
                "    assert candidate([5, 6, 3, 4]) == [3, 6, 5, 4]\n"
                "    assert candidate([1, 2, 3]) == [1, 2, 3]"
            ),
            "entry_point": "sort_even",
        }
        response = (
            "even_indices = [l[i] for i in range(0, len(l), 2)]\n"
            "        even_indices.sort()\n"
            "        result = l[:]\n"
            "        for i, val in enumerate(even_indices):\n"
            "            result[i * 2] = val\n"
            "        return result"
        )
        result = HumanEvalBenchmark().evaluate_response(response, item)
        assert result.passed is True
        assert result.pass_mode == "canonical_normalize_indent"

    def test_overindented_nested_helper_is_normalized(self):
        from omlx.eval.humaneval import HumanEvalBenchmark

        item = {
            "prompt": (
                "def prime_length(string):\n"
                '    """Return whether the string length is prime."""\n'
            ),
            "test": (
                "def check(candidate):\n"
                "    assert candidate('abc') is True\n"
                "    assert candidate('abcd') is False"
            ),
            "entry_point": "prime_length",
        }
        response = (
            "def is_prime(n):\n"
            "        if n < 2:\n"
            "            return False\n"
            "        for i in range(2, int(n ** 0.5) + 1):\n"
            "            if n % i == 0:\n"
            "                return False\n"
            "        return True\n"
            "    return is_prime(len(string))"
        )
        result = HumanEvalBenchmark().evaluate_response(response, item)
        assert result.passed is True
        assert result.pass_mode == "canonical_normalize_indent"

    def test_overindented_import_body_is_normalized(self):
        from omlx.eval.humaneval import HumanEvalBenchmark

        item = {
            "prompt": (
                "def fruit_distribution(s, n):\n" '    """Return missing fruits."""\n'
            ),
            "test": (
                "def check(candidate):\n"
                "    assert candidate('5 apples and 6 oranges', 19) == 8"
            ),
            "entry_point": "fruit_distribution",
        }
        response = (
            "import re\n"
            "        nums = re.findall(r'\\d+', s)\n"
            "        apples = int(nums[0])\n"
            "        oranges = int(nums[1])\n"
            "        return n - apples - oranges"
        )
        result = HumanEvalBenchmark().evaluate_response(response, item)
        assert result.passed is True
        assert result.pass_mode == "canonical_normalize_indent"

    def test_normalized_wrong_logic_still_fails(self):
        from omlx.eval.humaneval import HumanEvalBenchmark

        item = {
            "prompt": (
                "def maximum(arr, k):\n" '    """Return k largest elements."""\n'
            ),
            "test": (
                "def check(candidate):\n" "    assert candidate([1, 2, 3], 0) == []"
            ),
            "entry_point": "maximum",
        }
        result = HumanEvalBenchmark().evaluate_response("return sorted(arr)[-k:]", item)
        assert result.passed is False
        assert result.failure_type == "wrong_answer"

    def test_full_function_response_gets_prompt_imports(self):
        from omlx.eval.humaneval import HumanEvalBenchmark

        item = {
            "prompt": 'from typing import List\n\n\ndef first(xs: List[int]) -> int:\n    """Return first."""\n',
            "test": "def check(candidate):\n    assert candidate([3, 4]) == 3",
            "entry_point": "first",
        }
        response = "def first(xs: List[int]) -> int:\n    return xs[0]"
        result = HumanEvalBenchmark().evaluate_response(response, item)
        assert result.passed is True
        assert result.pass_mode == "standalone_function"

    def test_malformed_long_python_fence_is_extracted(self):
        from omlx.eval.humaneval import _extract_response_code

        response = "``````python\n    return 1\n```"
        assert _extract_response_code(response) == "    return 1"

    def test_internal_name_error_is_not_missing_entry_point(self):
        from omlx.eval.humaneval import _run_with_tests

        code = "def f():\n    python\n"
        test = "def check(candidate):\n    candidate()"
        result = _run_with_tests(code, test, "f")
        assert result.passed is False
        assert result.failure_type == "runtime_error"

    def test_syntax_error_is_categorized(self):
        from omlx.eval.humaneval import HumanEvalBenchmark

        item = {
            "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
            "entry_point": "add",
        }
        result = HumanEvalBenchmark().evaluate_response("return a +", item)
        assert result.passed is False
        assert result.failure_type in {"syntax_error", "indentation_error"}


class TestMBPPDiagnostics:
    def test_failing_assertion_reports_wrong_answer(self):
        from omlx.eval.mbpp import _run_with_tests

        result = _run_with_tests(
            "def add(a, b):\n    return a - b",
            ["assert add(1, 2) == 3"],
        )
        assert result.passed is False
        assert result.failure_type == "wrong_answer"

    def test_tolerates_tiny_numeric_assert_drift(self):
        from omlx.eval.mbpp import _run_with_tests

        result = _run_with_tests(
            "import math\n\ndef volume_cone(radius, height):\n    return (1/3) * math.pi * radius**2 * height",
            ["assert volume_cone(19,17)==6426.651371693521"],
        )
        assert result.passed is True
        assert result.pass_mode == "tolerant_numeric_asserts"

    def test_setup_can_run_after_generated_code(self):
        from omlx.eval.mbpp import _run_with_tests

        code = (
            "class Node:\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n"
            "        self.left = None\n"
            "\n"
            "def root_value(root):\n"
            "    return root.value\n"
        )
        result = _run_with_tests(
            code,
            ["assert root_value(root) == 7"],
            "root = Node(7)",
        )
        assert result.passed is True
        assert result.pass_mode == "setup_after_code"


class TestLiveCodeBenchDiagnostics:
    def test_variant_evaluator_reports_wrong_answer(self):
        from omlx.eval.livecodebench import LiveCodeBenchBenchmark

        item = {"inputs": [""], "outputs": ["42"]}
        result = LiveCodeBenchBenchmark().evaluate_code("print(41)", item)
        assert result.passed is False
        assert result.failure_type == "wrong_answer"


# --- Think Tag Stripping Tests ---


class TestStripThinkTags:
    def test_strip_think_block(self):
        from omlx.eval.base import BaseBenchmark

        text = (
            "<think>\nLet me think about this...\nThe answer should be A.\n</think>\nA"
        )
        assert BaseBenchmark._strip_think_tags(text) == "A"

    def test_strip_empty_think(self):
        from omlx.eval.base import BaseBenchmark

        assert BaseBenchmark._strip_think_tags("<think></think>B") == "B"

    def test_no_think_tags(self):
        from omlx.eval.base import BaseBenchmark

        assert BaseBenchmark._strip_think_tags("A") == "A"

    def test_incomplete_think_tag(self):
        from omlx.eval.base import BaseBenchmark

        # Incomplete think tag (no closing) — should be left as-is
        assert (
            BaseBenchmark._strip_think_tags("<think>still thinking")
            == "<think>still thinking"
        )


# --- Thinking Mode Tests ---


class TestThinkingMode:
    def test_benchmark_result_thinking_used_default(self):
        from omlx.eval.base import BenchmarkResult

        result = BenchmarkResult(
            benchmark_name="test",
            accuracy=0.5,
            total_questions=2,
            correct_count=1,
            time_seconds=1.0,
        )
        assert result.thinking_used is False

    def test_benchmark_result_thinking_used_true(self):
        from omlx.eval.base import BenchmarkResult

        result = BenchmarkResult(
            benchmark_name="test",
            accuracy=0.5,
            total_questions=2,
            correct_count=1,
            time_seconds=1.0,
            thinking_used=True,
        )
        assert result.thinking_used is True

    def test_thinking_token_constants(self):
        from omlx.eval.base import THINKING_MAX_TOKENS, THINKING_MIN_TOKENS

        assert THINKING_MIN_TOKENS == 8192
        assert THINKING_MAX_TOKENS == 32768
        assert THINKING_MIN_TOKENS < THINKING_MAX_TOKENS

    def test_strip_think_tags_with_answer(self):
        """Thinking content is stripped, leaving only the answer."""
        from omlx.eval.base import BaseBenchmark

        text = "<think>\nLet me analyze option A vs B.\nA seems correct.\n</think>\nThe answer is A"
        result = BaseBenchmark._strip_think_tags(text)
        assert "<think>" not in result
        assert "The answer is A" in result


# --- Dataset Sampling Tests ---


class TestSampling:
    def test_deterministic_sample_reproducible(self):
        """Same input always produces same output."""
        items = [{"id": i} for i in range(1000)]
        sample1 = deterministic_sample(items, 50)
        sample2 = deterministic_sample(items, 50)
        assert sample1 == sample2

    def test_deterministic_sample_correct_size(self):
        items = [{"id": i} for i in range(100)]
        sample = deterministic_sample(items, 30)
        assert len(sample) == 30

    def test_deterministic_sample_full_if_small(self):
        items = [{"id": i} for i in range(10)]
        sample = deterministic_sample(items, 50)
        assert len(sample) == 10

    def test_stratified_sample_reproducible(self):
        """Same input always produces same output."""
        items = [{"id": i, "cat": f"cat{i % 5}"} for i in range(500)]
        sample1 = stratified_sample(items, 50, "cat")
        sample2 = stratified_sample(items, 50, "cat")
        assert sample1 == sample2

    def test_stratified_sample_has_all_categories(self):
        items = [{"id": i, "cat": f"cat{i % 5}"} for i in range(500)]
        sample = stratified_sample(items, 50, "cat")
        cats = {item["cat"] for item in sample}
        assert len(cats) == 5

    def test_stratified_sample_proportional(self):
        """Categories should be roughly proportional."""
        items = []
        for i in range(100):
            items.append({"id": i, "cat": "big"})
        for i in range(10):
            items.append({"id": 100 + i, "cat": "small"})

        sample = stratified_sample(items, 22, "cat")
        big_count = sum(1 for item in sample if item["cat"] == "big")
        small_count = sum(1 for item in sample if item["cat"] == "small")
        # big should get ~20, small should get ~2
        assert big_count > small_count
        assert small_count >= 1


# --- Benchmark Registry Smoke Tests ---


class TestBenchmarkRegistry:
    """Cover every registered benchmark with cheap checks.

    Regression guard against silent bugs like registration drift or
    load_dataset() crashes on the sampling path.
    """

    def test_parity(self):
        """BENCHMARKS dict and VALID_BENCHMARKS list must be in sync."""
        from omlx.admin.accuracy_benchmark import VALID_BENCHMARKS
        from omlx.eval import BENCHMARKS

        assert set(BENCHMARKS.keys()) == set(VALID_BENCHMARKS)

    def test_instantiate_all(self):
        """Every registered class instantiates without error."""
        from omlx.eval import BENCHMARKS

        for cls in BENCHMARKS.values():
            cls()


def _registered_benchmark_names():
    from omlx.eval import BENCHMARKS

    return sorted(BENCHMARKS.keys())


@pytest.mark.parametrize("name", _registered_benchmark_names())
async def test_load_sample_per_benchmark(name):
    """Each registered benchmark loads a 10-row sample without crashing."""
    from omlx.eval import BENCHMARKS

    items = await BENCHMARKS[name]().load_dataset(sample_size=10)
    assert items, f"{name} returned empty list"
    assert len(items) <= 10, f"{name} returned {len(items)} items"
