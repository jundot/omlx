# oMLX Repository Discovery Guide

Agent-maintenance map for quickly orienting in oMLX; this is not user-facing documentation.

## Maintenance mandate

- Read this guide before substantive repository work, then spot-check two or three facts.
- Update it when documented paths, commands, conventions, or costly gotchas change; re-verify it if older than 90 days.
- Before committing, update affected sections without adding session history or duplicating the README.
- Last verified: 2026-07-12.

## Project overview

oMLX is a macOS/Apple Silicon inference server built around MLX, MLX-LM, and FastAPI. The core architectural boundary is engine selection: text models use the batched MLX-LM path, while VLM, embedding, reranking, and audio models use dedicated engines; model metadata discovered from local files drives that selection.

## Known gotchas

- MLX-LM is Git-pinned in `pyproject.toml`; unsupported architectures are normally added as narrow, upstream-first modules in `omlx/patches/`, not by modifying installed packages.
- `pytest.ini` excludes `slow` and `integration` markers by default, but a test under `tests/integration/` runs unless it actually carries the marker.
- `omlx/server.py` owns OpenAI Chat, Anthropic Messages, and OpenAI Responses request preparation; cross-API template policy must be applied to all three routes.
- Model discovery uses both `config.json` and tokenizer/chat-template metadata; retain explicit request and model-setting overrides when adding inferred defaults.
- Hybrid-attention model patches must expose accurate `make_cache()` types; the scheduler inspects them for rotating-window SSD block alignment and cache restoration.

## Conventions

- Run Python tools through `uv run`.
- Start behavior changes with focused RED tests, then make the smallest GREEN change and run affected suites.
- Preserve upstream module names when vendoring compatibility patches so MLX-LM dispatch continues unchanged.
- Keep model-weight tests opt-in or availability-gated; unit tests must not download checkpoints.

## Structure map

- `omlx/server.py` — API routing, request policy, response streaming, shared server state.
- `omlx/engine/` and `omlx/engine_pool.py` — engine implementations and model lifecycle.
- `omlx/model_discovery.py` — local model detection, capabilities, and discovered metadata.
- `omlx/utils/model_loading.py` — pre-load compatibility patch dispatcher.
- `omlx/patches/` — scoped upstream-first MLX-LM/MLX-VLM compatibility modules.
- `tests/` — unit, integration, model-gated, and marker-controlled verification.
- `tests/integration/test_laguna_real_model.py` — opt-in 18–23 GB Laguna 4-bit, 5-bit, and NVFP4 HTTP/cache verification; never downloads a model.
- `.github/workflows/ci.yml` — CI Python matrix and canonical non-slow/non-integration test selection.

## Entry points

- Focused tests: `uv run pytest tests/<module>.py -q`.
- CI selection: `uv run pytest tests/ -m "not slow and not integration"`.
- Lint: `uv run ruff check <changed paths>`.
- Type check: `uv run mypy <changed package or module>`.
- Laguna 4-bit live test: `OMLX_LAGUNA_MODEL_PATH=/absolute/model/path uv run pytest tests/integration/test_laguna_real_model.py -o addopts="" -m slow -k 4bit -s -q`.
- Laguna 5-bit issue test: `OMLX_LAGUNA_5BIT_MODEL_PATH=/absolute/model/path uv run pytest tests/integration/test_laguna_real_model.py -o addopts="" -m slow -k 5bit -s -q`.
- Laguna NVFP4 issue test: `OMLX_LAGUNA_NVFP4_MODEL_PATH=/absolute/model/path uv run pytest tests/integration/test_laguna_real_model.py -o addopts="" -m slow -k nvfp4 -s -q`.
- Run server: `uv run omlx` (inspect CLI help/config before changing invocation).

## What to verify

- Dependency pins and upstream module availability before adding a compatibility patch.
- Loader dispatch placement before `mlx_lm.load()` and correct engine classification.
- Chat-template defaults across Chat, Messages, and Responses, including explicit caller overrides.
- Marker selection, optional dependencies, model availability, and existing repository-wide static-analysis baselines.
- Patch module registration in `sys.modules` and its MLX-LM parent package.

## Maintenance snapshot

- Verified repository entry points, model-discovery path, server request-policy ownership, patch conventions, and CI test selection on 2026-07-12.
