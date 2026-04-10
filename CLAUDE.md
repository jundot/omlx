# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run unit tests (excludes slow/integration tests by default)
pytest

# Run specific test file
pytest tests/test_scheduler.py

# Run single test with keyword
pytest tests/test_engine_pool.py -k "test_load"

# Run slow tests (require model loading)
pytest -m "slow"

# Run integration tests (require running server)
pytest -m "integration"

# Lint/format
ruff check omlx/
black omlx/
mypy omlx/
```

## Architecture Overview

oMLX is an LLM inference server optimized for Apple Silicon, providing OpenAI/Anthropic API compatibility with continuous batching and tiered KV caching.

### Core Components

- **FastAPI Server** (`omlx/server.py`): Entry point exposing `/v1/chat/completions`, `/v1/messages`, `/v1/embeddings`, `/v1/rerank`, `/v1/models` endpoints
- **EnginePool** (`omlx/engine_pool.py`): Multi-model manager with LRU eviction, model pinning, TTL, and pre-load memory checking
- **Scheduler** (`omlx/scheduler.py`): Continuous batching via mlx-lm's BatchGenerator; manages waiting/running queues
- **Cache Stack** (`omlx/cache/`): Block-based KV cache with prefix sharing, hot (RAM) tier, and cold (SSD) tier persistence

### Engine Types

| Engine | File | Purpose |
|--------|------|---------|
| BatchedEngine | `engine/batched.py` | LLMs with continuous batching |
| VLMBatchedEngine | `engine/vlm.py` | Vision-language models |
| EmbeddingEngine | `engine/embedding.py` | Text embeddings |
| RerankerEngine | `engine/reranker.py` | Document reranking |
| STT/TTS/STS | `engine/stt.py`, `tts.py`, `sts.py` | Audio models |

### API Layer

- `omlx/api/adapters/`: OpenAI and Anthropic request/response adapters
- `omlx/api/openai_models.py`, `anthropic_models.py`: Pydantic request models
- `omlx/api/tool_calling.py`: Tool call parsing for various model families (Qwen XML, Llama JSON, Gemma, GLM, etc.)
- `omlx/api/thinking.py`: Extended thinking/reasoning support

### Model Discovery

- `omlx/model_discovery.py`: Auto-detects models from directories, determines type (LLM/VLM/embedding/reranker/audio)
- `omlx/model_registry.py`: Model metadata and capabilities registry

### Settings

- CLI flags → `~/.omlx/settings.json` (persisted)
- Per-model settings in `~/.omlx/model_settings.json` (sampling params, aliases, TTL)

## Key Patterns

### Cache Flow

Request → Scheduler checks prefix cache → Hot cache (RAM) → Cold cache (SSD) → Restore blocks if prefix match → Prefill remaining tokens → Stream generation

### Memory Management

`ProcessMemoryEnforcer` monitors total process memory, triggers TTL-based unloads or aborts in-progress model loads when exceeding limit.

### Streaming

SSE streaming with keep-alive for long prefill; usage stats in final chunk via `stream_options.include_usage`.

## Testing Notes

- Tests use `MockTokenizer`, `MockModel`, `MockModelConfig` fixtures in `conftest.py`
- Slow tests marked with `@pytest.mark.slow` require real model files
- Integration tests marked with `@pytest.mark.integration` require running server