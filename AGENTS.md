# AGENTS.md

## Platform Scope (Read First)

This project is ARM-first and ARM-only for Apple Silicon CPUs: **M1, M2, M3, M4, M5**.
Assume macOS + Apple ARM architecture for runtime, performance tuning, and compatibility decisions.

## Mission Context

oMLX is a local-first, Apple Silicon–optimized multi-model inference platform.
It serves OpenAI-compatible and Anthropic-compatible APIs with continuous batching,
tiered KV caching (RAM + SSD), model lifecycle management, and an admin dashboard.

Primary goals for changes:
- Keep inference correctness and API compatibility stable.
- Protect memory safety and graceful degradation under load.
- Preserve low-latency streaming behavior.
- Avoid regressions in cache coherence and model routing.

## Tech Stack

- Language: Python (project metadata requires `>=3.11`; classifiers include 3.10–3.13)
- Server: FastAPI + Uvicorn
- Core runtime: `mlx`, `mlx-lm`, `mlx-vlm`, `mlx-embeddings`, optional `mlx-audio`
- Packaging: setuptools (`pyproject.toml`)
- App wrapper: PyObjC menubar app under `packaging/omlx_app`
- Tests: Pytest (unit + integration + slow/model tests)

## Repo Map

- `omlx/server.py`: FastAPI app, endpoint wiring, auth checks, streaming and lifecycle.
- `omlx/cli.py`: `omlx serve` entrypoint, settings init, runtime env/bootstrap.
- `omlx/settings.py`: hierarchical config and persistence (`CLI > env > file > defaults`).
- `omlx/engine_core.py`, `omlx/scheduler.py`, `omlx/engine_pool.py`: request scheduling,
  batching, model loading/eviction, execution orchestration.
- `omlx/cache/`: paged/prefix/hybrid/SSD cache implementations + observability/stats.
- `omlx/api/`: OpenAI/Anthropic models, adapters, response conversion, tool calling,
  structured output, embeddings/rerank/audio/mcp routes.
- `omlx/models/`: model wrappers and model-type abstractions.
- `omlx/patches/`: model-family/runtime patches (DeepSeek/Qwen/Gemma/specprefill/etc).
- `omlx/admin/`: dashboard routes, templates, static assets, auth/downloader tools.
- `omlx/mcp/`: MCP client/manager/executor/types and tool bridge.
- `omlx/integrations/`: setup helpers for Codex/Copilot/Claude/OpenCode/OpenClaw/etc.
- `tests/`: broad unit/integration coverage.
- `docs/`: contributor docs and feature docs.

## Local Setup

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e ".[dev]"
# Optional extras:
# pip install -e ".[mcp]"
# pip install -e ".[audio]"
```

## Common Commands

```bash
# Start server
omlx serve --model-dir ~/models

# Fast test loop
pytest -m "not slow"

# Single-file test
pytest tests/test_config.py -v

# Slow/model tests
pytest -m slow

# Integration tests
pytest -m integration
```

## Coding Rules for Agents

1. Keep API contracts stable unless explicitly changing versioned behavior.
2. Preserve streaming semantics and usage accounting for chat/completions/responses.
3. Any scheduler/cache/memory change must include focused tests.
4. Prefer minimal, surgical edits over broad refactors.
5. Follow existing style (Black/Ruff, line length 88, SPDX header on source files).
6. Do not silently change default thresholds (memory/cache/concurrency) without tests and rationale.
7. Maintain offline/admin-dashboard compatibility (vendored frontend deps).
8. For model-family-specific changes, isolate logic in `omlx/patches` or adapter layers.

## High-Risk Areas

Treat these as sensitive and regression-prone:
- `omlx/scheduler.py` (admission, chunked prefill, queue/full behavior)
- `omlx/cache/*` (prefix sharing, block identity, persistence/recovery)
- `omlx/engine_core.py` and `omlx/engine/batched.py` (streaming + batching)
- `omlx/server.py` response streaming / API conversion / auth paths
- `omlx/process_memory_enforcer.py` + memory monitor interactions
- Tool-calling parsing (`omlx/api/tool_calling.py`, harmony/output parsers)

For these areas, add or update targeted tests before concluding.

## Test Strategy Expectations

When touching code, choose the narrowest meaningful test subset first, then widen.

Recommended mapping:
- API schemas/serialization/adapters -> `tests/test_openai_models.py`,
  `tests/test_anthropic_models.py`, `tests/test_responses.py`, `tests/test_*adapter*.py`
- Scheduler/engine behavior -> `tests/test_scheduler*.py`, `tests/test_engine*.py`
- Cache behavior -> `tests/test_cache_*.py`, `tests/test_prefix_cache*.py`,
  `tests/test_paged_cache*.py`, `tests/test_hybrid_cache.py`
- MCP/tooling -> `tests/test_mcp_*.py`, `tests/test_tool_calling.py`
- Admin/auth/settings -> `tests/test_admin_*.py`, `tests/test_settings.py`

Always run at least one relevant test for each modified subsystem.

## Configuration & Persistence Notes

- Runtime base path defaults to `~/.omlx`.
- Important persisted artifacts include settings, cache, and logs.
- Settings are initialized early and CLI overrides may be persisted.
- Ensure config precedence remains: `CLI > env > settings file > defaults`.

## Performance & Reliability Guardrails

- Avoid adding per-token Python overhead in hot paths.
- Keep allocations low in streaming loops.
- Don’t block event loop with heavy sync work on request path.
- Preserve graceful failures (`HTTPException`/typed errors) over crashes.
- Keep memory guard behavior deterministic and observable via logs/metrics.

## Integration Compatibility

oMLX is used as a backend for coding agents and clients expecting OpenAI/Anthropic behavior.
For changes in compatibility layers:
- Validate endpoint payload shape and finish reasons.
- Preserve handling for tool-call formats and structured output.
- Keep `stream_options.include_usage` and SSE behavior consistent.

## Definition of Done for Agent PRs

1. Code compiles and imports cleanly.
2. Relevant tests pass locally.
3. New behavior is covered by tests or justified as no-test-needed.
4. No unintended API/setting default drift.
5. Logging/error messages remain actionable.
6. Documentation/comments updated when behavior changes materially.

## Quick File Targets

- New endpoint behavior: start at `omlx/server.py` and `omlx/api/*`.
- Model load/evict logic: `omlx/engine_pool.py`, `omlx/model_registry.py`.
- Cache correctness: `omlx/cache/*` and matching cache tests.
- Admin UX/API: `omlx/admin/routes.py`, `omlx/admin/templates/*`, static JS/CSS.
- Packaging/app startup: `packaging/omlx_app/*`.

## Non-Goals

- Do not introduce non-macOS assumptions in core runtime paths.
- Do not remove compatibility shims without migration path.
- Do not bypass settings/auth checks for convenience during feature work.
- Do not introduce support assumptions for non-ARM or non-Apple-Silicon CPU targets.
