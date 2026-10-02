# SPDX-License-Identifier: Apache-2.0
"""CMMLU with Uniform Category Sampling."""

from pathlib import Path
from .datasets import load_jsonl
from .cmmlu import CMMLUBenchmark
from .sampling import uniform_stratified_sample


class CMMLUUniformBenchmark(CMMLUBenchmark):
    """CMMLU with Uniform Stratified Sampling."""

    name = "cmmlu_uniform"

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load CMMLU dataset and sample uniformly per category."""
        all_items = await super().load_dataset(sample_size=0)
        if sample_size == 0:
            return all_items
        return uniform_stratified_sample(all_items, sample_size, key="subject")
