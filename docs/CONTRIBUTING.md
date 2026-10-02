# Contributing to oMLX

Thank you for contributing! Bug fixes, performance improvements, model support, UI improvements, translations, and documentation are welcome.

## Getting Started

oMLX requires Apple Silicon, macOS 15.0+, and Python 3.11–3.13. Fork the repository and create a branch from current `main`:

```bash
git clone https://github.com/<your-username>/omlx.git
cd omlx
git checkout -b fix/describe-the-change
python -m pip install -e ".[dev]"
```

See [Development](../README.md#development) for app builds and [installation instructions](../README.md#from-source) for native kernel requirements.

## Keep Changes Focused

- Check existing issues and PRs first. Discuss large features, new dependencies, and changes to default behavior before starting a substantial implementation.
- Keep each PR focused on one problem or coherent feature. Include the settings, API, and UI integration it needs, but separate unrelated changes. For stacked PRs, explain the dependency and merge order.
- Fix the underlying problem and reuse existing code paths. Preserve supported behavior; explain any compatibility changes or performance tradeoffs.
- Follow the existing style, Black formatting, and Ruff configuration. Keep unrelated formatting and development artifacts out of the diff. Preserve third-party license notices and use Apache-2.0 SPDX headers for new original code.

## Test Your Change

For bug fixes, provide a reproduction and a regression test where practical. The test should fail before the fix and pass afterward through the affected production code, rather than a copy of the implementation inside the test.

Run the relevant tests, then the default suite for code changes:

```bash
python -m pytest tests/test_config.py  # Replace with the affected test files
python -m pytest                      # Excludes slow and integration tests
```

See [TESTING.md](TESTING.md) for additional checks. For inference or cache changes, include a representative real-model check when possible, covering affected features such as prefix reuse, streaming, or concurrent requests. State what you ran and what remains untested; CI or mocked tests do not replace hardware validation.

For visible UI changes, include screenshots and check the actual screen. Run relevant JavaScript tests and build the macOS app when those components change. After editing admin templates or JavaScript, rebuild CSS:

```bash
python omlx/admin/build_css.py
```

Use the existing translation catalogs and preserve placeholders. When web translation keys change, update `omlx/admin/i18n/en.json` and run:

```bash
python scripts/normalize_i18n.py
```

## Performance PRs

Include a before-and-after benchmark against the relevant implementation on `main`, using the same hardware, model, input, and settings. Provide:

- **Setup:** commit IDs, Mac chip/RAM, model and quantization, input/output lengths, and relevant sampling, concurrency, cache, and acceleration settings.
- **Results:** relevant throughput, time to first token, and memory measurements, including regressions or tradeoffs. State the run count and warm-up/cache conditions.
- **Reproduction:** commands or a small harness. Verify that the changed path runs and rebuild native kernels when they change.
- **Correctness:** appropriate output or quality checks when numerical behavior changes. A speedup alone is not enough; exact token equality is not required for every optimization.

Hardware-specific improvements are welcome. If you cannot run representative benchmarks, explain the limitation and what you verified. Keep concise results in the PR description; link or attach large logs and one-off experiment files instead of committing them.

## Submit for Review

Describe the problem, your approach, related issues, and validation results. Update documentation for user-facing changes. Keep unfinished work in Draft; mark it ready when you want review, or ask a specific question for early feedback.

Address review comments or explain your disagreement, then request another review. After resolving conflicts, preserve fixes already on `main` and rerun relevant checks. CI must pass before merge.

I maintain oMLX independently and normally squash-merge PRs. Review timing depends on scope and the validation needed; focused, reproducible changes help me review faster.

For suspected vulnerabilities, follow [SECURITY.md](../SECURITY.md) and report privately.
