# SPDX-License-Identifier: Apache-2.0
"""Unit tests for accuracy evaluation modules."""

import pytest

from omlx.eval.datasets import deterministic_sample, stratified_sample
from omlx.eval.gsm8k import GSM8KBenchmark, _extract_numeric_answer, _normalize_number
from omlx.eval.hellaswag import HellaSwagBenchmark
from omlx.eval.livecodebench import _extract_code
from omlx.eval.mmlu import MMLUBenchmark, _parse_choices
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
        assert self.bench.extract_answer("I think the correct answer is C because...", {}) == "C"

    def test_extract_answer_empty(self):
        assert self.bench.extract_answer("", {}) == ""

    def test_extract_answer_no_match(self):
        assert self.bench.extract_answer("I don't know", {}) == ""

    def test_extract_answer_lowercase_ignored(self):
        # Only uppercase A-D are valid
        assert self.bench.extract_answer("a", {}) == ""

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
            "endings": ["He orders a drink.", "He flies away.", "He disappears.", "He sings."],
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
        assert _extract_numeric_answer("She has 15 apples and 20 oranges, so 35 total.") == "35"

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
        response = "Here's my solution:\n```python\ndef solve():\n    print(42)\n```\nDone."
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


# --- Think Tag Stripping Tests ---


class TestStripThinkTags:
    def test_strip_think_block(self):
        from omlx.eval.base import BaseBenchmark
        text = "<think>\nLet me think about this...\nThe answer should be A.\n</think>\nA"
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
        assert BaseBenchmark._strip_think_tags("<think>still thinking") == "<think>still thinking"


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
