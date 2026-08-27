# SPDX-License-Identifier: Apache-2.0
"""Coverage for the opt-in native benchmark controls."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_repeated_single_benchmark_uses_distinct_prompts(monkeypatch):
    from omlx.admin import benchmark as admin_benchmark
    from omlx.engine import vlm
    from scripts import bench

    generated = []
    measured = []

    def generate_prompt(_tokenizer, prompt_tokens, profile):
        prompt = f"{profile}-{prompt_tokens}-{len(generated)}"
        generated.append(prompt)
        return prompt

    async def run_single(_engine, prompt, _gen_tokens, prompt_tokens):
        measured.append(prompt)
        return {
            "prompt_tokens": prompt_tokens,
            "ttft_ms": 1.0,
            "tpot_ms": 2.0,
            "gen_tps": 3.0,
            "processing_tps": 4.0,
            "peak_memory_bytes": 5,
        }

    class FakeEngine:
        tokenizer = object()

        def __init__(self, _model_path, model_settings=None):
            self.model_settings = model_settings

        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(admin_benchmark, "_generate_prompt", generate_prompt)
    monkeypatch.setattr(admin_benchmark, "_run_single_test", run_single)
    monkeypatch.setattr(vlm, "VLMBatchedEngine", FakeEngine)

    single, batch = await bench._bench_model(
        "model",
        [1024],
        gen_tokens=128,
        batch_sizes=[],
        warmup=1,
        repeats=3,
        context_profile="code_python",
    )

    assert batch == []
    assert len(single) == 1
    assert len(measured) == 4  # one warmup plus three measured trials
    assert len(set(measured[1:])) == 3
    assert all(prompt.startswith("code_python-1024-") for prompt in measured)
