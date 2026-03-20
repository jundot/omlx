# SPDX-License-Identifier: Apache-2.0
"""Dataset loading and sampling utilities.

All benchmark datasets are bundled in eval/data/ as JSONL files.
All sampling uses a fixed seed for deterministic, reproducible results
so that different models are always evaluated on the same questions.
"""

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# Fixed seed for all sampling — ensures identical question sets across models
SAMPLE_SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def deterministic_sample(items: list[dict], n: int) -> list[dict]:
    """Sample n items with a fixed seed for reproducibility.

    Always returns the same subset for the same input data,
    enabling fair comparison across different models.
    """
    if n >= len(items):
        return items
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(items, n)


def stratified_sample(
    items: list[dict], n: int, key: str
) -> list[dict]:
    """Stratified sampling: proportional representation from each category.

    Uses a fixed seed so the same questions are always selected.

    Args:
        items: Full dataset.
        n: Target sample size.
        key: Dict key for the category field.

    Returns:
        Stratified sample of size <= n.
    """
    if n >= len(items):
        return items

    rng = random.Random(SAMPLE_SEED)

    # Group by category
    groups: dict[str, list[dict]] = {}
    for item in items:
        cat = item.get(key, "unknown")
        groups.setdefault(cat, []).append(item)

    # Calculate proportional allocation
    total = len(items)
    sampled: list[dict] = []
    remaining = n

    sorted_cats = sorted(groups.keys())
    for i, cat in enumerate(sorted_cats):
        group = groups[cat]
        if i == len(sorted_cats) - 1:
            count = remaining
        else:
            count = max(1, round(len(group) / total * n))
            count = min(count, remaining, len(group))

        selected = rng.sample(group, min(count, len(group)))
        sampled.extend(selected)
        remaining -= len(selected)

        if remaining <= 0:
            break

    return sampled
