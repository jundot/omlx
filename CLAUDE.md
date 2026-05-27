# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project: Flyto MLX

Flyto MLX is a brand-independent fork of **oMLX** — an MLX-based LLM inference
engine and OpenAI-compatible server for Apple Silicon. It serves LLM / VLM /
audio models, with speculative decoding (DFlash), KV-cache tiering, oQ
quantization, grammar-constrained structured output (xgrammar), and an admin UI.

- GitHub `panwudi/flyto-mlx` (Gitee mirror, hourly). PyPI `flyto-mlx`.
- CLI main name `fmlx`; `omlx` kept as an alias for upstream-script compat.
- Renamed from `panwudi/omlx` on 2026-05-16.

### Upstream relationship — soft fork

Upstream is `jundot/omlx` (`upstream` git remote). We **soft-fork**: cherry-pick
upstream bug fixes and new-model support; we do **not** PR our own features back.
Sync workflow: `git fetch upstream && git log upstream/main..main`, cherry-pick
selectively (`-x`), and record every import/skip decision in
`docs/upstream-sync.md`. To check whether an upstream commit is already in,
actually `git cherry-pick` it (a "nothing to commit" means it is) — `git cherry`
and grep both mis-report.

## Repo layout

```
omlx/          engine package
  server.py            OpenAI-compatible API server
  engine/ engine_pool.py engine_core.py    model engines + pooling
  cache/               KV cache (paged, tiered, observability)
  speculative/         DFlash speculative decoding (Path A double-engine)
  api/                 request models, tool calling, grammar, responses
  oq.py                oQ quantization
  admin/               admin dashboard + routes
  ablations/ eval/ patches/ integrations/ mcp/
docs/          engine docs — upstream-sync.md, dflash-pathA-spec.md,
               reasoning-api.md, oQ_Quantization.md, roadmap.md, ...
packaging/     omlx_app — GUI + server_manager
Formula/       Homebrew formula
tests/         pytest suite
```

## Conventions

- **Commit messages are bilingual.** Subject in English; body = English text,
  then a `---` line on its own, then the Chinese version. No exceptions.
- **No AI trailers in commits** — no "Generated with…", no `Co-Authored-By`
  AI lines, no tool attribution.
- Chinese docs / release notes: plain prose — no emoji, no stacked sub-headers,
  no marketing phrasing, no decorative parallel bullet lists.
- Single `main` trunk = sole deploy + release branch. Develop on `feat/*`,
  rebase onto `main`, then merge. No long-lived half-finished branches.
- Commit / push only when asked.

## Standing authorizations (AI workflow)

These are durable, repo-scoped grants. They override the default "ask before
destructive / shared-state operation" rule for this repo only.

- **AI may self-merge own PRs from `sync/*` branches into `main`** when ALL of:
  1. Branch name matches `sync/*` (AI-opened upstream-sync or AI-driven fix
     workflow, not human-authored feature branches).
  2. Full test suite was run on the branch; only failures are the known
     pre-existing set (see `docs/upstream-sync.md` baseline). Zero regression.
  3. PR body has a Test plan section with the targeted-test list AND the
     full-suite numbers (pass / fail / skip) AND an explicit "zero regression"
     statement tied to the baseline.
  4. Use `merge_method: "merge"` (preserve atomic cherry-pick history) — NOT
     squash, NOT rebase. flyto's `git log --grep="cherry picked from"` workflow
     depends on atomic commits surviving into main.
  5. Merge commit subject + body follow the bilingual convention.

- **AI may force-push to its own `sync/*` branches** when rebasing AI-authored
  commits before the PR is merged. Once a PR is merged the branch is read-only
  history.

- **AI may NOT self-merge feature branches** (`feat/*`, `fix/*`, anything not
  matching `sync/*`) or any branch opened by a human.

- **AI may NOT push directly to `main`.** Always via PR, even with the
  self-merge grant above.

## Machines

- oMLX servers run on **m2max** and **m5max** at `~/Code/omlx`, launched by
  `/Applications/oMLX.app` running fork code (GUI bundle hack — do not upgrade
  the app). The local clone dir is named `omlx`; it is this `flyto-mlx` repo.
- `omlx serve` binds `settings.server.host` at startup (`~/.omlx/settings.json`);
  a config change needs a genuine server-process restart to take effect.
