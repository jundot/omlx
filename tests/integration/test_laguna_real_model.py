# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-model integration coverage for Laguna XS.2.

This test never downloads a checkpoint or contacts a running oMLX server. It
loads the caller-supplied local model through ``BatchedEngine``, then verifies
that a deterministic repeated prompt reuses a temporary paged-SSD prefix cache.

Run explicitly after downloading a supported checkpoint:

    OMLX_LAGUNA_MODEL_PATH=/absolute/path/to/Laguna-XS.2-4bit \
        uv run pytest tests/integration/test_laguna_real_model.py -m slow -k 4bit -s -q

    OMLX_LAGUNA_5BIT_MODEL_PATH=/absolute/path/to/Laguna-XS.2-5bit \
        uv run pytest tests/integration/test_laguna_real_model.py -m slow -k 5bit -s -q

    OMLX_LAGUNA_NVFP4_MODEL_PATH=/absolute/path/to/Laguna-XS.2-nvfp4 \
        uv run pytest tests/integration/test_laguna_real_model.py -m slow -k nvfp4 -s -q

The ``slow`` marker excludes these tests from default pytest runs and repository
CI. The environment variables prevent accidental use of arbitrary local models
when contributors intentionally run the slow suite.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import platform
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason="Laguna MLX integration requires macOS on Apple Silicon.",
    ),
]

_MINIMUM_SHARED_PROMPT_TOKENS = 8192
_REQUESTED_PREFIX_CACHE_BLOCK_SIZE_TOKENS = 256


def _model_path_from_environment(
    environment_variable: str,
    expected_quantization_bits: int,
    expected_quantization_mode: str,
) -> Path:
    """Validate and return an explicitly requested downloaded checkpoint."""
    configured_model_path = os.environ.get(environment_variable)
    if not configured_model_path:
        pytest.skip(f"Set {environment_variable} to run this Laguna real-model test.")

    model_path = Path(configured_model_path).expanduser()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        pytest.skip(f"Laguna config.json not found at {config_path}")

    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert model_config.get("model_type") == "laguna", (
        f"{environment_variable} must point to a Laguna checkpoint, "
        f"not model_type={model_config.get('model_type')!r}."
    )
    quantization_config = model_config.get("quantization") or {}
    assert quantization_config.get("bits") == expected_quantization_bits, (
        f"{environment_variable} must point to a {expected_quantization_bits}-bit "
        f"checkpoint, not bits={quantization_config.get('bits')!r}."
    )
    quantization_mode = quantization_config.get("mode", "affine")
    assert quantization_mode == expected_quantization_mode, (
        f"{environment_variable} must point to a {expected_quantization_mode} "
        f"checkpoint, not mode={quantization_mode!r}."
    )
    return model_path


@pytest.fixture(scope="module")
def laguna_4bit_model_path() -> Path:
    """Return the explicitly requested downloaded 4-bit checkpoint."""
    return _model_path_from_environment("OMLX_LAGUNA_MODEL_PATH", 4, "affine")


@pytest.fixture(scope="module")
def laguna_5bit_model_path() -> Path:
    """Return the exact 5-bit checkpoint reported in issue #2073."""
    return _model_path_from_environment("OMLX_LAGUNA_5BIT_MODEL_PATH", 5, "affine")


@pytest.fixture(scope="module")
def laguna_nvfp4_model_path() -> Path:
    """Return the primary NVFP4 checkpoint reported in issue #2073."""
    return _model_path_from_environment(
        "OMLX_LAGUNA_NVFP4_MODEL_PATH",
        4,
        "nvfp4",
    )


def _build_cacheable_prompt(tokenizer) -> str:
    """Build a shared prompt large enough for an observable prefix-cache hit."""
    stable_context = (
        "This is stable shared context for a deterministic cache integration test. "
    )
    repetition_count = 1

    while repetition_count <= 16384:
        prompt_text = stable_context * repetition_count
        if len(tokenizer.encode(prompt_text)) >= _MINIMUM_SHARED_PROMPT_TOKENS:
            return prompt_text
        repetition_count *= 2

    raise AssertionError("Could not construct a cacheable prompt for the tokenizer.")


def _run_laguna_real_model_validation(
    laguna_model_path: Path,
    tmp_path: Path,
) -> None:
    """Laguna loads natively, selects its parser, and reuses cached prompt KV."""
    import httpx
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from omlx.engine.batched import BatchedEngine
    from omlx.engine_pool import EngineEntry
    from omlx.model_discovery import detect_thinking_default
    from omlx.scheduler import SchedulerConfig
    from omlx.server import _server_state, app

    async def run_cache_integration() -> None:
        scheduler_config = SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=2048,
            completion_batch_size=1,
            prefill_step_size=2048,
            paged_cache_block_size=_REQUESTED_PREFIX_CACHE_BLOCK_SIZE_TOKENS,
            paged_ssd_cache_dir=str(tmp_path / "laguna-prefix-cache"),
            paged_ssd_cache_max_size=2 * 1024**3,
            model_name=laguna_model_path.name,
            model_path=str(laguna_model_path),
        )
        engine = BatchedEngine(
            model_name=str(laguna_model_path),
            scheduler_config=scheduler_config,
        )

        try:
            await engine.start()

            assert engine.model_type == "laguna"
            assert engine.prefix_cache_enabled is True
            assert (
                engine.tokenizer.tool_parser.__module__ == "mlx_lm.tool_parsers.laguna"
            )
            assert engine.tokenizer._tokenizer.init_kwargs["fix_mistral_regex"] is True

            model_caches = engine._model.make_cache()
            assert (
                sum(type(layer_cache) is KVCache for layer_cache in model_caches) == 10
            )
            assert (
                sum(
                    type(layer_cache) is RotatingKVCache for layer_cache in model_caches
                )
                == 30
            )
            assert {
                layer_cache.max_size
                for layer_cache in model_caches
                if type(layer_cache) is RotatingKVCache
            } == {512}
            effective_cache_block_size_tokens = (
                engine._engine.engine.scheduler.config.paged_cache_block_size
            )
            assert effective_cache_block_size_tokens == 512

            thinking_enabled_prompt = engine.tokenizer.apply_chat_template(
                [{"role": "user", "content": "What is one plus one?"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            assert thinking_enabled_prompt.rstrip().endswith("<think>")

            chat_response = await engine.chat(
                [{"role": "user", "content": "What is one plus one?"}],
                max_tokens=8,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": True},
            )
            assert chat_response.completion_tokens > 0

            model_id = "laguna-real-http-test"
            thinking_default = detect_thinking_default(laguna_model_path)
            assert thinking_default is True
            model_config = json.loads(
                (laguna_model_path / "config.json").read_text(encoding="utf-8")
            )
            model_context_length = model_config.get("max_position_embeddings")
            assert isinstance(model_context_length, int)
            assert model_context_length > 0
            engine_entry = EngineEntry(
                model_id=model_id,
                model_path=str(laguna_model_path),
                model_type="llm",
                engine_type="batched",
                estimated_size=0,
                config_model_type="laguna",
                thinking_default=thinking_default,
                preserve_thinking_default=None,
                model_context_length=model_context_length,
                engine=engine,
            )

            class SingleEnginePool:
                """Lease-compatible pool exposing only the loaded Laguna engine."""

                def resolve_model_id(self, requested_model_id, settings_manager=None):
                    return requested_model_id

                async def get_engine(self, requested_model_id, _lease=False):
                    assert requested_model_id == model_id
                    return engine

                async def release_engine(self, requested_model_id):
                    assert requested_model_id == model_id

                def get_entry(self, requested_model_id):
                    assert requested_model_id == model_id
                    return engine_entry

                def is_abort_requested(self, requested_model_id):
                    return False

            original_apply_chat_template = engine._apply_chat_template
            applied_template_kwargs: list[dict[str, object]] = []

            def record_apply_chat_template(*args, **kwargs):
                applied_template_kwargs.append(
                    dict(kwargs.get("chat_template_kwargs") or {})
                )
                return original_apply_chat_template(*args, **kwargs)

            original_pool = _server_state.engine_pool
            original_default_model = _server_state.default_model
            original_api_key = _server_state.api_key
            try:
                engine._apply_chat_template = record_apply_chat_template
                _server_state.engine_pool = SingleEnginePool()
                _server_state.default_model = model_id
                _server_state.api_key = None

                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://laguna-test",
                    timeout=120.0,
                ) as http_client:
                    http_response = await http_client.post(
                        "/v1/chat/completions",
                        json={
                            "model": model_id,
                            "messages": [
                                {"role": "user", "content": "What is one plus one?"}
                            ],
                            "max_tokens": 8,
                            "temperature": 0.0,
                        },
                    )

                assert http_response.status_code == 200, http_response.text
                response_body = http_response.json()
                assert response_body["choices"][0]["message"]["role"] == "assistant"
                assert response_body["usage"]["completion_tokens"] > 0
                assert applied_template_kwargs
                assert all(
                    template_kwargs.get("enable_thinking") is True
                    for template_kwargs in applied_template_kwargs
                )
            finally:
                engine._apply_chat_template = original_apply_chat_template
                _server_state.engine_pool = original_pool
                _server_state.default_model = original_default_model
                _server_state.api_key = original_api_key

            shared_prompt = _build_cacheable_prompt(engine.tokenizer)
            first_response = await engine.generate(
                shared_prompt,
                max_tokens=16,
                temperature=0.0,
            )
            second_response = await engine.generate(
                shared_prompt,
                max_tokens=16,
                temperature=0.0,
            )

            assert first_response.completion_tokens > 0
            assert second_response.completion_tokens > 0
            assert first_response.text == second_response.text
            assert first_response.cached_tokens == 0
            assert second_response.cached_tokens >= effective_cache_block_size_tokens, (
                "Expected the repeated Laguna prompt to reuse at least one "
                "paged prefix-cache block."
            )
        finally:
            await engine.stop()
            gc.collect()
            mx.clear_cache()

    asyncio.run(run_cache_integration())


def test_laguna_4bit_real_model(
    laguna_4bit_model_path: Path,
    tmp_path: Path,
) -> None:
    """Validate the readily available 4-bit Laguna checkpoint."""
    _run_laguna_real_model_validation(laguna_4bit_model_path, tmp_path)


def test_laguna_5bit_issue_2073_real_model(
    laguna_5bit_model_path: Path,
    tmp_path: Path,
) -> None:
    """Validate the exact 5-bit checkpoint reported in GitHub issue #2073."""
    _run_laguna_real_model_validation(laguna_5bit_model_path, tmp_path)


def test_laguna_nvfp4_issue_2073_real_model(
    laguna_nvfp4_model_path: Path,
    tmp_path: Path,
) -> None:
    """Validate the primary NVFP4 checkpoint reported in GitHub issue #2073."""
    _run_laguna_real_model_validation(laguna_nvfp4_model_path, tmp_path)
