# SPDX-License-Identifier: Apache-2.0
"""Accuracy evaluation benchmarks for LLMs.

Provides MMLU, HellaSwag, TruthfulQA, GSM8K, and LiveCodeBench
evaluators with deterministic sampling for fair model comparison.
"""

from .arc import ARCChallengeBenchmark
from .base import BaseBenchmark, BenchmarkResult, QuestionResult
from .cmmlu import CMMLUBenchmark
from .gsm8k import GSM8KBenchmark
from .hellaswag import HellaSwagBenchmark
from .humaneval import HumanEvalBenchmark
from .jmmlu import JMMLUBenchmark
from .kmmlu import KMMLUBenchmark
from .livecodebench import LiveCodeBenchBenchmark
from .mbpp import MBPPBenchmark
from .mmlu import MMLUBenchmark
from .truthfulqa import TruthfulQABenchmark
from .winogrande import WinograndeBenchmark

BENCHMARKS: dict[str, type[BaseBenchmark]] = {
    "mmlu": MMLUBenchmark,
    "kmmlu": KMMLUBenchmark,
    "cmmlu": CMMLUBenchmark,
    "jmmlu": JMMLUBenchmark,
    "hellaswag": HellaSwagBenchmark,
    "truthfulqa": TruthfulQABenchmark,
    "arc_challenge": ARCChallengeBenchmark,
    "winogrande": WinograndeBenchmark,
    "gsm8k": GSM8KBenchmark,
    "humaneval": HumanEvalBenchmark,
    "mbpp": MBPPBenchmark,
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
    "ARCChallengeBenchmark",
    "WinograndeBenchmark",
    "GSM8KBenchmark",
    "HumanEvalBenchmark",
    "MBPPBenchmark",
    "LiveCodeBenchBenchmark",
]
