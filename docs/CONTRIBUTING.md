# Contributing to oMLX

Thank you for your interest in contributing to oMLX! This guide will help you get started.

## Getting Started

1. Fork the repository on [GitHub](https://github.com/jundot/omlx)
2. Clone your fork:

```bash
git clone https://github.com/<your-username>/omlx.git
cd omlx
```

3. Install development dependencies:

```bash
pip install -e ".[dev]"
```

> **Note**: oMLX requires Apple Silicon (M1/M2/M3/M4) and Python 3.10+.

## Development Workflow

### Testing

Run tests with:

```bash
# Fast tests only (recommended during development)
pytest -m "not slow"

# Run a specific test file
pytest tests/test_config.py -v

# Run slow tests (requires model files)
pytest -m slow
```

**Test markers:**

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Tests that require loading models |
| `@pytest.mark.integration` | Tests that require a running server |

**Test file naming:** For a source file `omlx/<module>.py`, the test file should be `tests/test_<module>.py`.

When modifying source code, always check if existing tests are affected and update them accordingly. New code should include corresponding tests.

### License Header

All source files should include the Apache 2.0 license identifier:

```python
# SPDX-License-Identifier: Apache-2.0
```

## Project Structure

```
omlx/
├── omlx/                 # Main package
│   ├── api/              # API models and adapters (OpenAI, Anthropic)
│   ├── cache/            # KV cache management (paged, prefix, SSD)
│   ├── engine/           # Inference engines (simple, batched, embedding)
│   ├── mcp/              # Model Context Protocol integration
│   ├── models/           # Model wrappers (LLM, embedding)
│   ├── utils/            # Utilities
│   ├── server.py         # FastAPI server
│   ├── scheduler.py      # Request scheduling with mlx-lm BatchGenerator
│   ├── engine_core.py    # Core async inference engine
│   ├── paged_cache.py    # Block-based KV cache with LRU eviction
│   └── cli.py            # CLI entry point
├── apps/omlx-mac/        # Native SwiftUI macOS app (menubar + admin UI)
├── packaging/            # macOS .app bundle build pipeline (venvstacks)
├── tests/                # Test suite
└── docs/                 # Documentation
```

For a more detailed architecture overview, see the [Architecture](../README.md#architecture) section in the README.

## Areas for Contribution

- **Bug fixes** — Check [open issues](https://github.com/jundot/omlx/issues) for reported bugs
- **Performance** — Inference speed, memory efficiency, cache hit rates
- **New features** — API endpoints, model format support, admin dashboard improvements
- **Documentation** — Guides, examples, API references
- **Model support** — Testing and fixing compatibility with new MLX models
- **macOS app** — UI improvements, new settings, system integration

## Pull Request Process

1. **Fork & branch** — Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make changes** — Write code and tests following the style guidelines above.

3. **Test** — Ensure all related tests pass:
   ```bash
   pytest tests/test_<affected_module>.py -v
   ```

4. **Submit** — Push your branch and open a pull request against `main`. Describe what changed and why.

## Support

If you have questions or run into issues, please open an issue on [GitHub Issues](https://github.com/jundot/omlx/issues).
