# SPDX-License-Identifier: Apache-2.0
"""Accuracy evaluation benchmarks for LLMs.

Provides benchmarks across knowledge, commonsense reasoning, math,
coding, safety, and bias categories with deterministic sampling
for fair model comparison.
"""

from .arc import ARCChallengeBenchmark
from .base import BaseBenchmark, BenchmarkResult, QuestionResult
from .bbq import BBQBenchmark
from .cmmlu import CMMLUBenchmark
from .gsm8k import GSM8KBenchmark
from .hellaswag import HellaSwagBenchmark
from .humaneval import HumanEvalBenchmark
from .jmmlu import JMMLUBenchmark
from .kmmlu import KMMLUBenchmark
from .livecodebench import LiveCodeBenchBenchmark
from .mathqa import MathQABenchmark
from .mbpp import MBPPBenchmark
from .mmlu import MMLUBenchmark
from .mmlu_pro import MMLUProBenchmark
from .safetybench import SafetyBenchBenchmark
from .truthfulqa import TruthfulQABenchmark
from .winogrande import WinograndeBenchmark

BENCHMARKS: dict[str, type[BaseBenchmark]] = {
    "mmlu": MMLUBenchmark,
    "mmlu_pro": MMLUProBenchmark,
    "kmmlu": KMMLUBenchmark,
    "cmmlu": CMMLUBenchmark,
    "jmmlu": JMMLUBenchmark,
    "hellaswag": HellaSwagBenchmark,
    "truthfulqa": TruthfulQABenchmark,
    "arc_challenge": ARCChallengeBenchmark,
    "winogrande": WinograndeBenchmark,
    "gsm8k": GSM8KBenchmark,
    "mathqa": MathQABenchmark,
    "humaneval": HumanEvalBenchmark,
    "mbpp": MBPPBenchmark,
    "livecodebench": LiveCodeBenchBenchmark,
    "bbq": BBQBenchmark,
    "safetybench": SafetyBenchBenchmark,
}

__all__ = [
    "BENCHMARKS",
    "BaseBenchmark",
    "BenchmarkResult",
    "QuestionResult",
    "MMLUBenchmark",
    "MMLUProBenchmark",
    "HellaSwagBenchmark",
    "TruthfulQABenchmark",
    "ARCChallengeBenchmark",
    "WinograndeBenchmark",
    "GSM8KBenchmark",
    "MathQABenchmark",
    "HumanEvalBenchmark",
    "MBPPBenchmark",
    "LiveCodeBenchBenchmark",
    "BBQBenchmark",
    "SafetyBenchBenchmark",
]
