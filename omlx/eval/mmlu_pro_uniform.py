# SPDX-License-Identifier: Apache-2.0
"""MMLU-Pro with Uniform Category Sampling."""

from pathlib import Path
from .datasets import load_jsonl
from .mmlu_pro import MMLUProBenchmark
from .sampling import uniform_stratified_sample


class MMLUProUniformBenchmark(MMLUProBenchmark):
    """MMLU-Pro with Uniform Stratified Sampling."""

    name = "mmlu_pro_uniform"

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load MMLU-Pro dataset and sample uniformly per category."""
        all_items = await super().load_dataset(sample_size=0)
        if sample_size == 0:
            return all_items
        return uniform_stratified_sample(all_items, sample_size, key="subject")
