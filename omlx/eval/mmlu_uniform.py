# SPDX-License-Identifier: Apache-2.0
"""MMLU with Uniform Category Sampling."""

from pathlib import Path
from .datasets import load_jsonl
from .mmlu import MMLUBenchmark
from .sampling import uniform_stratified_sample


class MMLUUniformBenchmark(MMLUBenchmark):
    """MMLU with Uniform Stratified Sampling.
    
    Ensures each subject has equal representation in the sample,
    unlike the default proportional sampling.
    """
    
    name = "mmlu_uniform"
    
    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load MMLU dataset and sample uniformly per category."""
        # 1. Load base data (reusing parent logic to parse choices/answers)
        test_items = load_jsonl(Path(__file__).parent / "data" / "mmlu_test.jsonl")
        all_items = []
        
        # We need to replicate the item parsing from MMLUBenchmark.load_dataset
        # because that method returns a list[dict] with parsed answers.
        # Alternatively, we could call super().load_dataset(0) if it returns the full list.
        # Looking at the parent class, load_dataset(0) returns the full list of items.
        # However, the parent class logic includes loading few-shot examples from dev_items.
        # The safest way is to call the parent implementation and then re-sample.
        
        # Note: MMLUBenchmark.load_dataset handles both loading and sampling.
        # If sample_size == 0, it returns all items.
        # But we also need the few-shot examples to be loaded for self._few_shot_examples.
        
        # To ensure self._few_shot_examples is populated correctly, we must run the parent logic.
        # We can trick the parent by calling it with 0, then re-sampling the result.
        
        all_items = await super().load_dataset(sample_size=0)
        
        if sample_size == 0:
            return all_items

        # 2. Apply uniform stratified sampling
        return uniform_stratified_sample(all_items, sample_size, key="subject")
