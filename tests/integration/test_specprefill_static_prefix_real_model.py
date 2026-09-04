# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-model validation for SpecPrefill static-prefix reuse (#2177).

Issue #2177 addresses a specific gap between the two halves of SpecPrefill:
the small draft model could avoid some repeated scoring work, but the target
model still prefetched the same system instructions and tool schemas on every
turn. Unit tests protect the orchestration contract, but they cannot prove that
real Qwen hybrid caches restore their recurrent state, rotating-cache metadata,
MLX stream ownership, and target positions correctly.

This test therefore loads an actual Qwen3.6 target and Qwen3.5 draft and runs
four deliberately distinct phases:

1. A cold request persists the exact target static prefix to SSD.
2. A new engine restores it after the original engine has shut down.
3. Hot-cache clearing and pressure reclaim leave the SSD prefix reusable.
4. Changed system material misses safely and creates a distinct exact chain.

The test never downloads checkpoints, never contacts a running oMLX server,
never reads or writes the user's persisted model settings, and stores its paged
cache under pytest's temporary directory. It is marked ``slow`` and requires
explicit model paths because the known validation pairs use a 27B or 35B target
plus a 4B draft and consequently need substantial Apple Silicon unified
memory. Timing is printed by oMLX for human inspection but is intentionally not
an assertion: target-prefill latency varies with hardware, thermals, and other
processes, while the phase-specific cache telemetry is deterministic.

Example using the locally available validation pair::

    OMLX_SPECPREFILL_TARGET_PATH="$HOME/.omlx/models/Jundot/Qwen3.6-27B-oQ4e-mtp" \
    OMLX_SPECPREFILL_DRAFT_PATH="$HOME/.omlx/models/lmstudio-community/Qwen3.5-4B-MLX-4bit" \
    uv run pytest tests/integration/test_specprefill_static_prefix_real_model.py \
      -o addopts="" -m slow -s -q
"""

from __future__ import annotations

import asyncio
import copy
import gc
import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason="Real SpecPrefill validation requires macOS on Apple Silicon.",
    ),
]

_TARGET_PATH_ENV = "OMLX_SPECPREFILL_TARGET_PATH"
_DRAFT_PATH_ENV = "OMLX_SPECPREFILL_DRAFT_PATH"
_PREFIX_CACHE_BLOCK_SIZE_TOKENS = 256
_STATIC_PREFIX_MINIMUM_TOKENS = 1024
_SPECPREFILL_THRESHOLD_TOKENS = 128
_CONVERSATION_MINIMUM_TOKENS = 384


def _load_explicit_model_config(
    environment_variable: str,
    expected_model_types: tuple[str, ...],
) -> tuple[Path, dict[str, Any]]:
    """Return one explicitly selected local checkpoint and validated config.

    Missing variables skip instead of discovering or downloading a convenient
    model. Once a contributor sets a variable, however, a wrong path is a test
    configuration error and must fail loudly rather than silently selecting a
    different architecture that does not exercise the production bug.
    """
    configured_path = os.environ.get(environment_variable)
    if not configured_path:
        pytest.skip(
            f"Set {environment_variable} to an existing local checkpoint to run "
            "the SpecPrefill real-model test."
        )
        raise AssertionError("pytest.skip unexpectedly returned")

    model_path = Path(configured_path).expanduser()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        pytest.fail(f"{environment_variable} has no config.json: {config_path}")

    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    actual_model_type = model_config.get("model_type")
    if actual_model_type not in expected_model_types:
        pytest.fail(
            f"{environment_variable} must identify one of model_type="
            f"{expected_model_types!r}, not {actual_model_type!r}: {model_path}"
        )
    return model_path, model_config


def _text_vocab_size(model_config: dict[str, Any]) -> int | None:
    """Read the shared text vocabulary from nested or flat model configs."""
    text_config = model_config.get("text_config")
    if isinstance(text_config, dict):
        vocab_size = text_config.get("vocab_size")
    else:
        vocab_size = model_config.get("vocab_size")
    return int(vocab_size) if isinstance(vocab_size, int) else None


@pytest.fixture(scope="module")
def specprefill_model_pair() -> tuple[Path, Path]:
    """Validate the exact target/draft architecture and tokenizer contract."""
    target_path, target_config = _load_explicit_model_config(
        _TARGET_PATH_ENV,
        ("qwen3_5", "qwen3_5_moe"),
    )
    draft_path, draft_config = _load_explicit_model_config(
        _DRAFT_PATH_ENV,
        ("qwen3_5",),
    )

    target_vocab_size = _text_vocab_size(target_config)
    draft_vocab_size = _text_vocab_size(draft_config)
    if target_vocab_size is None or draft_vocab_size is None:
        pytest.fail(
            "Both real-model configs must declare a text vocabulary so the "
            "SpecPrefill tokenizer compatibility check is meaningful."
        )
    if target_vocab_size != draft_vocab_size:
        pytest.fail(
            "SpecPrefill target and draft tokenizers are incompatible: "
            f"target vocab={target_vocab_size}, draft vocab={draft_vocab_size}."
        )
    return target_path, draft_path


def _repeat_until_token_count(
    tokenizer: Any,
    sentence: str,
    minimum_tokens: int,
) -> str:
    """Build natural repeated prose without hard-coding tokenizer ratios."""
    repetition_count = 1
    while repetition_count <= 4096:
        text = sentence * repetition_count
        if len(tokenizer.encode(text)) >= minimum_tokens:
            return text
        repetition_count *= 2
    raise AssertionError(
        f"Could not build {minimum_tokens} tokens of test prose for the tokenizer."
    )


def _specprefill_log_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return actionable SpecPrefill lines for diagnostics."""
    return [
        record.getMessage()
        for record in caplog.records
        if "SpecPrefill" in record.getMessage()
    ]


def _assert_log_contains(
    messages: list[str],
    expected_fragment: str,
    *,
    phase: str,
) -> None:
    """Fail with the complete phase telemetry instead of a bare substring error."""
    assert any(expected_fragment in message for message in messages), (
        f"{phase} phase did not emit {expected_fragment!r}. "
        f"Captured SpecPrefill telemetry: {messages}"
    )


def _assert_log_excludes(
    messages: list[str],
    forbidden_fragment: str,
    *,
    phase: str,
) -> None:
    """Explain unexpected cold/warm transitions with all relevant log lines."""
    assert all(forbidden_fragment not in message for message in messages), (
        f"{phase} phase unexpectedly emitted {forbidden_fragment!r}. "
        f"Captured SpecPrefill telemetry: {messages}"
    )


def test_issue_2177_static_prefix_reuse_with_real_qwen_models(
    specprefill_model_pair: tuple[Path, Path],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Validate exact SSD reuse across restart and memory reclamation."""
    target_model_path, draft_model_path = specprefill_model_pair

    # Heavy imports stay below the opt-in fixtures. A normal test collection or
    # missing-path skip therefore does not initialize MLX, import VLM runtimes,
    # or accidentally reserve unified memory.
    import mlx.core as mx

    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.model_settings import ModelSettings
    from omlx.scheduler import SchedulerConfig

    async def run_real_model_validation() -> None:
        caplog.set_level(logging.INFO)
        model_settings = ModelSettings(
            enable_thinking=False,
            specprefill_enabled=True,
            specprefill_draft_model=str(draft_model_path),
            specprefill_keep_pct=0.30,
            specprefill_threshold=_SPECPREFILL_THRESHOLD_TOKENS,
        )
        cache_directory = tmp_path / "specprefill-prefix-cache"

        def create_engine() -> VLMBatchedEngine:
            scheduler_config = SchedulerConfig(
                max_num_seqs=1,
                max_num_batched_tokens=512,
                completion_batch_size=1,
                prefill_step_size=512,
                paged_cache_block_size=_PREFIX_CACHE_BLOCK_SIZE_TOKENS,
                paged_ssd_cache_dir=str(cache_directory),
                paged_ssd_cache_max_size=4 * 1024**3,
                hot_cache_max_size=0,
                model_name=target_model_path.name,
                model_path=str(target_model_path),
            )
            return VLMBatchedEngine(
                model_name=str(target_model_path),
                scheduler_config=scheduler_config,
                model_settings=model_settings,
                enable_thinking=False,
            )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_repository_symbol",
                    "description": "Look up a repository symbol without changing files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"symbol": {"type": "string"}},
                        "required": ["symbol"],
                    },
                },
            }
        ]
        stable_system = ""

        def messages_for(system_text: str, user_text: str) -> list[dict[str, str]]:
            return [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]

        async def run_chat(
            selected_engine: VLMBatchedEngine,
            messages: list[dict[str, str]],
        ) -> Any:
            return await selected_engine.chat(
                messages,
                tools=tools,
                max_tokens=2,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False},
                specprefill=True,
                specprefill_keep_pct=0.30,
                specprefill_threshold=_SPECPREFILL_THRESHOLD_TOKENS,
            )

        cold_engine = create_engine()
        try:
            await cold_engine.start()
            assert cold_engine._engine is not None
            cold_scheduler = cold_engine._engine.engine.scheduler
            assert cold_scheduler._specprefill_draft_model is not None
            stable_system = _repeat_until_token_count(
                cold_engine.tokenizer,
                "Follow repository rules and inspect evidence before answering. ",
                _STATIC_PREFIX_MINIMUM_TOKENS,
            )
            cold_user_text = _repeat_until_token_count(
                cold_engine.tokenizer,
                "Explain how a scheduler preserves cache correctness. ",
                _CONVERSATION_MINIMUM_TOKENS,
            )

            boundary_attempt = 0
            while True:
                cold_messages = messages_for(stable_system, cold_user_text)
                prompt_tokens, vlm_embeds, *_ = cold_engine._process_chat_messages(
                    copy.deepcopy(cold_messages), copy.deepcopy(tools), {}
                )
                non_system_prompt = cold_engine.tokenizer.apply_chat_template(
                    [copy.deepcopy(cold_messages[-1])],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                system_end = len(prompt_tokens) - len(
                    cold_engine.tokenizer.encode(non_system_prompt)
                )
                if system_end % _PREFIX_CACHE_BLOCK_SIZE_TOKENS != 0:
                    break
                boundary_attempt += 1
                stable_system += f" Deterministic boundary padding {boundary_attempt}."
            assert vlm_embeds is None
            assert len(prompt_tokens) - system_end > _SPECPREFILL_THRESHOLD_TOKENS

            with caplog.at_level(logging.INFO):
                caplog.clear()
                cold_response = await run_chat(cold_engine, cold_messages)
                cold_logs = _specprefill_log_messages(caplog)
            assert cold_response.completion_tokens > 0
            _assert_log_contains(cold_logs, "tokens full prefill", phase="cold")
            _assert_log_excludes(cold_logs, "static system-prefix tokens", phase="cold")
            cold_stats = cold_scheduler.block_aware_cache.get_stats()
            assert cold_stats.exact_prefix_stores == 1
            assert cold_scheduler.paged_ssd_cache_manager.get_stats().saves > 0
        finally:
            await cold_engine.stop()
            del cold_engine
            gc.collect()
            mx.clear_cache()

        warm_engine = create_engine()
        try:
            await warm_engine.start()
            assert warm_engine._engine is not None
            warm_scheduler = warm_engine._engine.engine.scheduler
            warm_user_text = _repeat_until_token_count(
                warm_engine.tokenizer,
                "Describe why cache metadata must be restored atomically. ",
                _CONVERSATION_MINIMUM_TOKENS,
            )

            with caplog.at_level(logging.INFO):
                caplog.clear()
                warm_response = await run_chat(
                    warm_engine,
                    messages_for(stable_system, warm_user_text),
                )
                warm_logs = _specprefill_log_messages(caplog)
            assert warm_response.completion_tokens > 0
            _assert_log_contains(
                warm_logs,
                "static system-prefix tokens from tiered cache",
                phase="restart",
            )
            _assert_log_excludes(warm_logs, "tokens full prefill", phase="restart")

            warm_scheduler.paged_ssd_cache_manager.clear_hot_cache()
            warm_scheduler.request_pressure_reclaim()
            warm_engine._engine.engine._wake_engine_loop()
            for _ in range(5000):
                if not warm_scheduler._pending_pressure_clear:
                    break
                await asyncio.sleep(0.001)
            else:
                raise AssertionError("Engine loop did not consume pressure reclaim.")

            pressure_user_text = _repeat_until_token_count(
                warm_engine.tokenizer,
                "Summarize why SSD cache survives memory reclamation. ",
                _CONVERSATION_MINIMUM_TOKENS,
            )
            caplog.clear()
            pressure_response = await run_chat(
                warm_engine,
                messages_for(stable_system, pressure_user_text),
            )
            pressure_logs = _specprefill_log_messages(caplog)
            assert pressure_response.completion_tokens > 0
            _assert_log_contains(
                pressure_logs,
                "static system-prefix tokens from tiered cache",
                phase="pressure",
            )
            _assert_log_excludes(pressure_logs, "tokens full prefill", phase="pressure")

            changed_system = stable_system + " This instruction changed."
            caplog.clear()
            changed_response = await run_chat(
                warm_engine,
                messages_for(changed_system, pressure_user_text),
            )
            changed_logs = _specprefill_log_messages(caplog)
            assert changed_response.completion_tokens > 0
            _assert_log_excludes(
                changed_logs, "static system-prefix tokens", phase="changed-prefix"
            )
            assert warm_scheduler.block_aware_cache.get_stats().exact_prefix_hits >= 2
            assert warm_scheduler.block_aware_cache.get_stats().exact_prefix_misses >= 1
        finally:
            await warm_engine.stop()
            gc.collect()
            mx.clear_cache()

    asyncio.run(run_real_model_validation())
