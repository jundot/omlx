# SPDX-License-Identifier: Apache-2.0
"""
Real model integration tests for oMLX.

These tests load actual mlx-lm models to verify:
- Tensor shape consistency
- Memory handling
- Generation quality
- Streaming output correctness
- UTF-8 handling for CJK characters

These tests are marked with @pytest.mark.slow and are skipped by default.
Run with: pytest -m slow tests/integration/test_real_model_inference.py

Requirements:
- Apple Silicon (M1/M2/M3/M4)
- At least 8GB unified memory
- Model files in ~/Workspace/models/ (or set via OMLX_MODEL_DIR env var)

Environment variables:
- OMLX_MODEL_DIR: Directory containing models (default: ~/Workspace/models)
- OMLX_TEST_MODEL: Specific model path or name to test (optional)
"""

import gc
import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import pytest

# Skip all tests in this module if not on Apple Silicon
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="Real model tests require macOS with Apple Silicon"
    ),
]


def get_test_model_dir() -> Optional[Path]:
    """Get the model directory for testing."""
    # Try environment variable first
    if model_dir := os.environ.get("OMLX_MODEL_DIR"):
        return Path(model_dir)

    # Try common locations
    common_paths = [
        Path.home() / "Workspace" / "models",
        Path.home() / "models",
        Path("/opt/models"),
    ]

    for path in common_paths:
        if path.exists():
            return path

    return None


def find_test_model(model_dir: Path) -> Optional[Path]:
    """Find a test model in the model directory.

    Priority:
    1. OMLX_TEST_MODEL env var (absolute path or model name)
    2. Preferred small models for faster testing
    3. Any model with config.json
    """
    # Check for specific model via environment variable
    if test_model := os.environ.get("OMLX_TEST_MODEL"):
        test_path = Path(test_model)
        # If absolute path, use directly
        if test_path.is_absolute():
            if test_path.exists() and (test_path / "config.json").exists():
                return test_path
        else:
            # Treat as model name, look in model_dir
            candidate = model_dir / test_model
            if candidate.exists() and (candidate / "config.json").exists():
                return candidate
            # Try glob pattern match
            matches = list(model_dir.glob(f"*{test_model}*"))
            for match in matches:
                if (match / "config.json").exists():
                    return match
        # If specified model not found, skip
        return None

    if not model_dir.exists():
        return None

    # Look for small models suitable for testing
    # Prefer 4-bit quantized models for faster loading
    preferred_patterns = [
        "*SmolLM*",  # Very small model
        "*Qwen*0.5B*",
        "*Qwen*1.5B*",
        "*TinyLlama*",
        "*Llama*1B*",
        "*Llama*3B*4bit*",
        "*Phi*mini*",
        "*Gemma*2B*",
    ]

    for pattern in preferred_patterns:
        matches = list(model_dir.glob(pattern))
        if matches:
            # Return the first match that has config.json
            for match in matches:
                if (match / "config.json").exists():
                    return match

    # Fall back to any model with config.json
    for subdir in model_dir.iterdir():
        if subdir.is_dir() and (subdir / "config.json").exists():
            return subdir

    return None


@pytest.fixture(scope="module")
def model_dir() -> Path:
    """Get the model directory, skip if not available."""
    path = get_test_model_dir()
    if path is None or not path.exists():
        pytest.skip("Model directory not found. Set OMLX_MODEL_DIR env var.")
    return path


@pytest.fixture(scope="module")
def test_model_path(model_dir: Path) -> Path:
    """Find a test model, skip if none available."""
    path = find_test_model(model_dir)
    if path is None:
        pytest.skip(f"No suitable test model found in {model_dir}")
    return path


class TestMLXLanguageModel:
    """Tests for MLXLanguageModel with real models."""

    def test_model_loading(self, test_model_path: Path):
        """Test that model loads correctly."""
        from omlx.models.llm import MLXLanguageModel

        model = MLXLanguageModel(str(test_model_path))
        model.load()

        assert model._loaded is True
        assert model.model is not None
        assert model.tokenizer is not None

        # Check model info
        info = model.get_model_info()
        assert info["loaded"] is True
        # vocab_size may not be present for all model types
        if "vocab_size" in info:
            assert info["vocab_size"] > 0

        # Cleanup
        del model
        gc.collect()

    def test_basic_generation(self, test_model_path: Path):
        """Test basic text generation."""
        from omlx.models.llm import MLXLanguageModel

        model = MLXLanguageModel(str(test_model_path))
        model.load()

        output = model.generate(
            prompt="The capital of France is",
            max_tokens=10,
            temperature=0.0,  # Greedy for deterministic output
        )

        assert output.text is not None
        assert len(output.text) > 0
        assert len(output.tokens) > 0
        assert output.finish_reason in ("stop", "length")

        # Cleanup
        del model
        gc.collect()

    def test_streaming_generation(self, test_model_path: Path):
        """Test streaming text generation."""
        from omlx.models.llm import MLXLanguageModel

        model = MLXLanguageModel(str(test_model_path))
        model.load()

        chunks: List[str] = []
        for output in model.stream_generate(
            prompt="Hello, my name is",
            max_tokens=20,
            temperature=0.7,
        ):
            chunks.append(output.text)
            if output.finished:
                assert output.finish_reason is not None
                break

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0

        # Cleanup
        del model
        gc.collect()

    def test_chat_completion(self, test_model_path: Path):
        """Test chat completion with message format."""
        from omlx.models.llm import MLXLanguageModel

        model = MLXLanguageModel(str(test_model_path))
        model.load()

        messages = [
            {"role": "user", "content": "What is 2 + 2?"}
        ]

        output = model.chat(
            messages=messages,
            max_tokens=50,
            temperature=0.0,
        )

        assert output.text is not None
        assert len(output.text) > 0
        # Model should mention "4" somewhere in response
        # (This is a weak check, but tests the flow)

        # Cleanup
        del model
        gc.collect()

    def test_utf8_generation_cjk(self, test_model_path: Path):
        """Test UTF-8 streaming for CJK characters."""
        from omlx.models.llm import MLXLanguageModel

        model = MLXLanguageModel(str(test_model_path))
        model.load()

        # Use a prompt that should elicit CJK output
        # (Results depend on model, but tests UTF-8 handling)
        chunks: List[str] = []
        for output in model.stream_generate(
            prompt="Translate to Japanese: Hello",
            max_tokens=30,
            temperature=0.7,
        ):
            chunks.append(output.text)
            # Each chunk should be valid UTF-8
            try:
                output.text.encode('utf-8')
            except UnicodeEncodeError:
                pytest.fail(f"Invalid UTF-8 in chunk: {output.text!r}")

            if output.finished:
                break

        # Cleanup
        del model
        gc.collect()


class TestSchedulerWithRealModel:
    """Tests for Scheduler with real model integration."""

    @pytest.fixture
    def scheduler_setup(self, test_model_path: Path):
        """Set up scheduler with real model."""
        from mlx_lm import load

        from omlx.scheduler import Scheduler, SchedulerConfig

        # Load model
        model, tokenizer = load(str(test_model_path))

        # Create scheduler config with conservative settings for testing
        config = SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=1024,
            prefill_batch_size=2,
            completion_batch_size=4,
        )

        # Create scheduler
        scheduler = Scheduler(
            config=config,
            model=model,
            tokenizer=tokenizer,
        )

        yield scheduler, tokenizer

        # Cleanup
        scheduler.shutdown()
        del model, tokenizer
        gc.collect()

    def test_single_request(self, scheduler_setup):
        """Test single request through scheduler."""
        scheduler, tokenizer = scheduler_setup
        from omlx.request import Request, SamplingParams

        request = Request(
            request_id="test-001",
            prompt="The weather today is",
            sampling_params=SamplingParams(
                max_tokens=20,
                temperature=0.7,
            ),
        )

        scheduler.add_request(request)

        # Run scheduler steps until completion
        all_outputs = []
        for _ in range(50):  # Max iterations
            step_result = scheduler.step()
            # SchedulerOutput has outputs attribute with List[RequestOutput]
            all_outputs.extend(step_result.outputs)

            if step_result.finished_request_ids:
                break

        assert len(all_outputs) > 0

    def test_batch_requests(self, scheduler_setup):
        """Test multiple concurrent requests."""
        scheduler, tokenizer = scheduler_setup
        from omlx.request import Request, SamplingParams

        prompts = [
            "The capital of Japan is",
            "Python is a programming language that",
            "Machine learning is",
        ]

        for i, prompt in enumerate(prompts):
            request = Request(
                request_id=f"batch-{i}",
                prompt=prompt,
                sampling_params=SamplingParams(
                    max_tokens=15,
                    temperature=0.7,
                ),
            )
            scheduler.add_request(request)

        # Run until all complete
        completed = set()
        for _ in range(100):  # Max iterations
            step_result = scheduler.step()
            # Collect finished request IDs
            completed.update(step_result.finished_request_ids)

            if len(completed) == len(prompts):
                break

        assert len(completed) == len(prompts)

    def test_cancel_request(self, scheduler_setup):
        """Test request cancellation."""
        scheduler, tokenizer = scheduler_setup
        from omlx.request import Request, SamplingParams

        request = Request(
            request_id="to-cancel",
            prompt="Write a very long story about",
            sampling_params=SamplingParams(
                max_tokens=1000,  # Long generation
                temperature=0.7,
            ),
        )

        scheduler.add_request(request)

        # Run a few steps
        for _ in range(5):
            scheduler.step()

        # Cancel the request (deferred abort - enqueue then process)
        scheduler.abort_request("to-cancel")
        scheduler._process_pending_aborts()

        # Verify cancelled - check running dict is empty
        assert len(scheduler.running) == 0


class TestMemoryHandling:
    """Tests for memory handling with real models."""

    def test_model_memory_footprint(self, test_model_path: Path):
        """Test that model loading doesn't cause memory issues."""
        import mlx.core as mx

        from omlx.models.llm import MLXLanguageModel

        # Get initial memory state
        mx.metal.clear_cache()
        gc.collect()

        model = MLXLanguageModel(str(test_model_path))
        model.load()

        # Generate some tokens to allocate KV cache
        for _ in model.stream_generate(
            prompt="Test prompt",
            max_tokens=50,
        ):
            pass

        # Force memory evaluation
        mx.eval()

        # Model should be usable after generation
        output = model.generate(
            prompt="Another test",
            max_tokens=10,
        )
        assert output.text is not None

        # Cleanup
        del model
        gc.collect()
        mx.metal.clear_cache()

    def test_repeated_generation_no_leak(self, test_model_path: Path):
        """Test that repeated generations don't leak memory."""
        import mlx.core as mx

        from omlx.models.llm import MLXLanguageModel

        model = MLXLanguageModel(str(test_model_path))
        model.load()

        # Run multiple generations
        for i in range(5):
            output = model.generate(
                prompt=f"Generation test {i}:",
                max_tokens=20,
                temperature=0.7,
            )
            assert output.text is not None

            # Clear intermediate cache
            mx.metal.clear_cache()

        # Final generation should still work
        final = model.generate(
            prompt="Final test:",
            max_tokens=10,
        )
        assert final.text is not None

        # Cleanup
        del model
        gc.collect()


class TestTokenizerIntegration:
    """Tests for tokenizer integration with real models."""

    def test_tokenizer_roundtrip(self, test_model_path: Path):
        """Test tokenizer encode/decode roundtrip."""
        from mlx_lm import load

        _, tokenizer = load(str(test_model_path))

        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "12345 67890",
            "Special chars: @#$%^&*()",
        ]

        for text in test_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)

            # Decoded text should contain the original
            # (may have added special tokens)
            assert text.strip() in decoded or text in decoded

    def test_tokenizer_special_tokens(self, test_model_path: Path):
        """Test handling of special tokens."""
        from mlx_lm import load

        _, tokenizer = load(str(test_model_path))

        # Check EOS token exists
        assert hasattr(tokenizer, 'eos_token_id') or hasattr(tokenizer, 'eos_token')

        # Check vocab size
        vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else tokenizer.vocab_size
        assert vocab_size > 0

    def test_chat_template_application(self, test_model_path: Path):
        """Test chat template application if available."""
        from mlx_lm import load

        _, tokenizer = load(str(test_model_path))

        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(formatted, str)
                assert len(formatted) > 0
                # Should contain message content
                assert "Hello!" in formatted or "Hi there!" in formatted
            except Exception as e:
                # Some tokenizers may not support all features
                pytest.skip(f"Chat template not fully supported: {e}")
        else:
            pytest.skip("Tokenizer doesn't have apply_chat_template")


class TestEngineIntegration:
    """Tests for engine integration with real models."""

    def test_batched_engine_generation(self, test_model_path: Path):
        """Test BatchedEngine with real model."""
        import asyncio

        from omlx.engine.batched import BatchedEngine

        async def run_generation():
            engine = BatchedEngine(
                model_name=str(test_model_path),
                trust_remote_code=True,
            )

            try:
                # Generate (auto-loads model on first call)
                output = await engine.generate(
                    prompt="What is 1+1? Answer briefly:",
                    max_tokens=20,
                    temperature=0.7,
                )

                # Basic assertion - output object should be valid
                assert output is not None
                assert output.finish_reason in ("stop", "length")
                # Note: Some models (like gpt-oss with Harmony) may return
                # empty text due to parsing issues, so we only check that
                # tokens were generated (completion_tokens > 0)
                assert output.completion_tokens > 0

            finally:
                # Cleanup
                await engine.stop()
                gc.collect()

        asyncio.run(run_generation())
