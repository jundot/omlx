# SPDX-License-Identifier: Apache-2.0
"""Dataset sampling utilities with deterministic results.

Provides uniform and proportional stratified sampling strategies
to ensure fair representation across categories (e.g., subjects).
"""

import random
from .datasets import SAMPLE_SEED

def uniform_stratified_sample(
    items: list[dict], n: int, key: str, min_per_cat: int = 1
) -> list[dict]:
    """Uniform stratified sampling: equal representation from each category.
    
    Uses a fixed seed so the same questions are always selected.
    Ensures every category has at least `min_per_cat` items.

    Args:
        items: Full dataset.
        n: Target sample size.
        key: Dict key for the category field (e.g., 'subject').
        min_per_cat: Minimum number of items per category.

    Returns:
        Stratified sample of size <= n.
    """
    if not items:
        return []
    
    # Group by category
    groups: dict[str, list[dict]] = {}
    for item in items:
        cat = item.get(key, "unknown")
        groups.setdefault(cat, []).append(item)
        
    num_cats = len(groups)
    rng = random.Random(SAMPLE_SEED)
    
    # Edge Case Guard: If we can't satisfy the minimum constraint,
    # return all items to avoid sampling bias or crashes.
    if n < num_cats * min_per_cat:
        return items

    if n >= len(items):
        return items

    # Calculate base allocation
    base_count = max(min_per_cat, n // num_cats)
    remainder = n - (base_count * num_cats)
    
    sampled = []
    # Sort keys to ensure deterministic assignment of remainder
    sorted_cats = sorted(groups.keys())
    
    for i, cat in enumerate(sorted_cats):
        group = groups[cat]
        # First 'remainder' categories get one extra item
        count = base_count + (1 if i < remainder else 0)
        count = min(count, len(group))
        
        sampled.extend(rng.sample(group, count))
        
    return sampled

def proportional_stratified_sample(
    items: list[dict], n: int, key: str
) -> list[dict]:
    """Proportional stratified sampling (original oMLX behavior).
    
    Allocates samples based on category size.
    """
    if n >= len(items):
        return items

    rng = random.Random(SAMPLE_SEED)
    groups: dict[str, list[dict]] = {}
    for item in items:
        cat = item.get(key, "unknown")
        groups.setdefault(cat, []).append(item)

    total = len(items)
    sampled = []
    remaining = n
    sorted_cats = sorted(groups.keys())

    for i, cat in enumerate(sorted_cats):
        group = groups[cat]
        if i == len(sorted_cats) - 1:
            count = remaining
        else:
            count = max(1, round(len(group) / total * n))
            count = min(count, remaining, len(group))

        sampled.extend(rng.sample(group, min(count, len(group))))
        remaining -= len([x for x in sampled if x.get(key) == cat])
        if remaining <= 0:
            break
            
    return sampled
