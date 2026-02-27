<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-rounded-dark.svg" width="140">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/icon-rounded-light.svg" width="140">
    <img alt="oMLX" src="docs/images/icon-rounded-light.svg" width="140">
  </picture>
</p>

<h1 align="center">oMLX</h1>
<p align="center"><b>LLM inference, optimized for your Mac</b><br>Continuous batching and infinite SSD caching, managed directly from your menu bar.</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <a href="https://buymeacoffee.com/jundot"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="https://github.com/jundot/omlx">GitHub</a>
</p>

---

<p align="center">
  <img src="docs/images/omlx_dashboard.png" alt="oMLX Admin Dashboard" width="800">
</p>

> *Every LLM server I tried made me choose between convenience and control. I wanted to pin everyday models in memory, auto-swap heavier ones on demand, set context limits - and manage it all from a menu bar.*
>
> *oMLX persists KV cache to SSD - even when context changes mid-conversation, all past context stays cached and reusable across requests, making local LLMs practical for real coding work with tools like Claude Code. That's why I built it.*

## Install

### macOS App

Download the `.dmg` from [Releases](https://github.com/jundot/omlx/releases), drag to Applications, done.

### Homebrew

```bash
brew tap jundot/omlx https://github.com/jundot/omlx
brew install omlx

# Run as a background service (auto-restarts on crash)
brew services start omlx
```

### From Source

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e .
```

Requires Python 3.10+ and Apple Silicon (M1/M2/M3/M4).

## Quickstart

### macOS App

Launch oMLX from your Applications folder. The Welcome screen guides you through three steps - model directory, server start, and first model download. That's it.

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.36.32.png" alt="oMLX Welcome Screen" width="500">
  <img src="docs/images/Screenshot 2026-02-10 at 00.51.54.png" alt="oMLX Welcome Screen" width="500">
</p>

### CLI

```bash
omlx serve --model-dir ~/models
```

The server discovers models from subdirectories automatically. Any OpenAI-compatible client can connect to `http://localhost:8000/v1`. A built-in chat UI is also available at `http://localhost:8000/admin/chat`.

### Homebrew Service

If you installed via Homebrew, you can run oMLX as a managed background service:

```bash
brew services start omlx    # Start (auto-restarts on crash)
brew services stop omlx     # Stop
brew services restart omlx  # Restart
brew services info omlx     # Check status
```

The service runs `omlx serve` with zero-config defaults (`~/.omlx/models`, port 8000). To customize, either set environment variables (`OMLX_MODEL_DIR`, `OMLX_PORT`, etc.) or run `omlx serve --model-dir /your/path` once to persist settings to `~/.omlx/settings.json`.

Logs are written to two locations:
- **Service log**: `$(brew --prefix)/var/log/omlx.log` (stdout/stderr)
- **Server log**: `~/.omlx/logs/server.log` (structured application log)

## Features

oMLX is built on top of [vllm-mlx](https://github.com/waybarrios/vllm-mlx), extending it with paged SSD caching, multi-model serving, an admin dashboard, Claude Code optimization, and Anthropic API support. Currently supports text-based LLMs - VLM and OCR model support is planned for upcoming milestones.

**macOS menubar app** - Native menubar app to start, stop, and monitor the server without opening a terminal.

**Admin dashboard** - Web UI at `/admin` for model management, chat, real-time monitoring, and per-model settings.

**Built-in model downloader** - Search and download MLX models from HuggingFace directly in the admin dashboard. No CLI or `git clone` needed.

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.35.01.png" alt="oMLX Model Downloader" width="720">
</p>

**Claude Code optimization** - Context scaling support for running smaller context models with Claude Code. Scales reported token counts so that auto-compact triggers at the right timing, and SSE keep-alive prevents read timeouts during long prefill.

**Paged KV cache with SSD tiering** - Block-based cache management inspired by vLLM, with prefix sharing and Copy-on-Write. When GPU memory fills up, blocks are offloaded to SSD. On the next request with a matching prefix, they're restored from disk instead of recomputed from scratch - even after a server restart.

**Continuous batching** - Handles concurrent requests through mlx-lm's BatchGenerator. Prefill and completion batch sizes are configurable.

**Multi-model serving** - Load LLMs, embedding models, and rerankers within the same server. Least-recently-used models are evicted automatically when memory runs low. Pin frequently used models to keep them loaded.

**API compatibility** - Drop-in replacement for OpenAI and Anthropic APIs.

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (streaming) |
| `POST /v1/completions` | Text completions (streaming) |
| `POST /v1/messages` | Anthropic Messages API |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/rerank` | Document reranking |
| `GET /v1/models` | List available models |

**Tool calling & structured output** - Supports all function calling formats available in mlx-lm, JSON schema validation, and MCP tool integration. Tool calling requires the model's chat template to support the `tools` parameter. The following model families are auto-detected via mlx-lm's built-in tool parsers:

| Model Family | Format |
|---|---|
| Llama, Qwen, DeepSeek, etc. | JSON `<tool_call>` |
| Qwen3 Coder | XML `<function=...>` |
| Gemma | `<start_function_call>` |
| GLM (4.7, 5) | `<arg_key>/<arg_value>` XML |
| MiniMax | Namespaced `<minimax:tool_call>` |
| Mistral | `[TOOL_CALLS]` |
| Kimi K2 | `<\|tool_calls_section_begin\|>` |
| Longcat | `<longcat_tool_call>` |

Models not listed above may still work if their chat template accepts `tools` and their output uses a recognized `<tool_call>` XML format. Streaming requests with tool calls buffer all content and emit results at completion.

## Models

Point `--model-dir` at a directory containing MLX-format model subdirectories:

```
~/models/
├── Step-3.5-Flash-8bit/
├── Qwen3-Coder-Next-8bit/
├── gpt-oss-120b-MXFP4-Q8/
└── bge-m3/
```

Models are auto-detected by type. You can also download models directly from the admin dashboard.

| Type | Models |
|------|--------|
| LLM | Any model supported by [mlx-lm](https://github.com/ml-explore/mlx-lm) |
| Embedding | BERT, BGE-M3, ModernBERT |
| Reranker | ModernBERT, XLM-RoBERTa |

## CLI Configuration

```bash
# Memory limit for loaded models
omlx serve --model-dir ~/models --max-model-memory 32GB

# Enable SSD cache for KV blocks
omlx serve --model-dir ~/models --paged-ssd-cache-dir ~/.omlx/cache

# Adjust batch sizes
omlx serve --model-dir ~/models --prefill-batch-size 8 --completion-batch-size 32

# With MCP tools
omlx serve --model-dir ~/models --mcp-config mcp.json

# API key authentication
omlx serve --model-dir ~/models --api-key your-secret-key
```

All settings can also be configured from the web admin panel at `/admin`. Settings are persisted to `~/.omlx/settings.json`, and CLI flags take precedence.

<details>
<summary>Architecture</summary>

```
FastAPI Server (OpenAI / Anthropic API)
    │
    ├── EnginePool (multi-model, LRU eviction)
    │   ├── BatchedEngine (LLMs, continuous batching)
    │   ├── EmbeddingEngine
    │   └── RerankerEngine
    │
    ├── Scheduler (FCFS, configurable batch sizes)
    │   └── mlx-lm BatchGenerator
    │
    └── Cache Stack
        ├── PagedCacheManager (GPU, block-based, CoW, prefix sharing)
        └── PagedSSDCacheManager (SSD tier, safetensors format)
```

</details>

## Development

### CLI Server

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e ".[dev]"
pytest -m "not slow"
```

### macOS App

Requires Python 3.11+ and [venvstacks](https://venvstacks.lmstudio.ai) (`pip install venvstacks`).

```bash
cd packaging

# Full build (venvstacks + app bundle + DMG)
python build.py

# Skip venvstacks (code changes only)
python build.py --skip-venv

# DMG only
python build.py --dmg-only
```

See [packaging/README.md](packaging/README.md) for details on the app bundle structure and layer configuration.

## Contributing

We welcome contributions! See [Contributing Guide](docs/CONTRIBUTING.md) for details.

- Bug fixes and improvements
- Performance optimizations
- Documentation improvements

## License

[Apache 2.0](LICENSE)

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [vllm-mlx](https://github.com/vllm-project/vllm-mlx) - oMLX originated as a fork of vllm-mlx v0.1.0, since re-architected with multi-model serving, paged SSD caching, an admin panel, and a standalone macOS menu bar app
- [venvstacks](https://venvstacks.lmstudio.ai) - Portable Python environment layering for the macOS app bundle
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) - Embedding model support for Apple Silicon

