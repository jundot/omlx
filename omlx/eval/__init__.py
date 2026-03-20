# SPDX-License-Identifier: Apache-2.0
"""Accuracy evaluation benchmarks for LLMs.

Provides MMLU, HellaSwag, TruthfulQA, GSM8K, and LiveCodeBench
evaluators with deterministic sampling for fair model comparison.
"""

from .base import BaseBenchmark, BenchmarkResult, QuestionResult
from .gsm8k import GSM8KBenchmark
from .hellaswag import HellaSwagBenchmark
from .livecodebench import LiveCodeBenchBenchmark
from .mmlu import MMLUBenchmark
from .truthfulqa import TruthfulQABenchmark

BENCHMARKS: dict[str, type[BaseBenchmark]] = {
    "mmlu": MMLUBenchmark,
    "hellaswag": HellaSwagBenchmark,
    "truthfulqa": TruthfulQABenchmark,
    "gsm8k": GSM8KBenchmark,
    "livecodebench": LiveCodeBenchBenchmark,
}

__all__ = [
    "BENCHMARKS",
    "BaseBenchmark",
    "BenchmarkResult",
    "QuestionResult",
    "MMLUBenchmark",
    "HellaSwagBenchmark",
    "TruthfulQABenchmark",
    "GSM8KBenchmark",
    "LiveCodeBenchBenchmark",
]
