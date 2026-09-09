# SPDX-License-Identifier: Apache-2.0
"""HellaSwag with Uniform Activity Sampling."""

from pathlib import Path
from .datasets import load_jsonl
from .hellaswag import HellaSwagBenchmark
from .sampling import uniform_stratified_sample


class HellaSwagUniformBenchmark(HellaSwagBenchmark):
    """HellaSwag with Uniform Stratified Sampling by activity_label."""

    name = "hellaswag_uniform"

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load HellaSwag dataset and sample uniformly per activity."""
        # HellaSwag uses activity_label as category
        # We need to replicate normalization or call super
        
        # Calling super().load_dataset(0) might not work if parent logic 
        # requires specific normalization steps that are hardcoded.
        # Looking at parent HellaSwagBenchmark:
        # It normalizes `label` to `answer` (int) and maps `ctx`, `endings`.
        # We should replicate this logic to be safe, then sample.
        
        items = load_jsonl(Path(__file__).parent / "data" / "hellaswag_val.jsonl")
        normalized = []
        for item in items:
            label = item.get("label", "0")
            normalized.append({
                "id": item.get("ind", ""),
                "context": item.get("ctx", ""),
                "endings": item.get("endings", []),
                "answer": int(label) if isinstance(label, (int, float)) else int(label),
                "activity_label": item.get("activity_label", ""),
            })
            
        if sample_size == 0:
            return normalized
            
        return uniform_stratified_sample(normalized, sample_size, key="activity_label")
