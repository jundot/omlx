# Slim app — external Python runtime instead of an embedded one

Python and the MLX layer are **not** shipped inside the DMG. The app bundle
carries only the Swift shell and the small `omlx`/`omlx_web` source tree, plus
`uv` and the lockfile for the first-run installer. A provisioned runtime lives
under:

```text
~/Library/Application Support/oMLX/runtime/venv
```

The pinned interpreter comes from Homebrew when available, otherwise from a
tagged `uv`-managed CPython. The dependency set comes from `uv.lock`.

Measured on an Apple Silicon Mac:

| Artifact | Size |
|---|---|
| embedded app (old) | 1.6 GB |
| embedded DMG (old) | 884 MB |
| slim app | 146 MB |
| **slim DMG** | **~72 MB** (with `uv`/installer + helper; 58 MB before those) |
| external runtime venv | ~1.0 GB (outside the DMG) |

## Provision the runtime

```sh
apps/omlx-mac/Scripts/provision_runtime.sh
```

- `PYTHON_VERSION` defaults to the repo's `.python-version`.
- Homebrew `python<version>` is preferred; otherwise `uv python install` fetches
  the tagged build.
- `UV_PROJECT_ENVIRONMENT=<venv> uv sync --frozen --no-dev
  --no-install-project --extra cluster` installs the locked dependencies only.
- The `omlx`/`omlx_web` source is deliberately **not** installed into the venv;
  the app ships it in `Contents/Resources` and puts that path on `PYTHONPATH`.

## Build the slim app

```sh
OMLX_TEAM_ID=<TEAM_ID> \
OMLX_NEXT_OUT="$PWD/apps/omlx-mac/build/SlimStage" \
  apps/omlx-mac/Scripts/build.sh release --slim
```

`--slim` copies the Swift app plus `omlx`/`omlx_web`, writes wrappers that use
the external venv, compiles and stages the SMJobBless helper, and skips
`cpython-3.11` and `framework-mlx-base` entirely. `OMLX_TEAM_ID` generates the
helper's trust requirement; without it the helper builds but cannot be blessed
(see [Privileged helper](HELPER.md)).

## Package the DMG

```sh
APP_PATH="$PWD/apps/omlx-mac/build/SlimStage/oMLX.app" \
OUT_DIR="$PWD/build/slim" \
  apps/omlx-mac/Scripts/package_dmg.sh
```

The final image is `build/slim/oMLX-<version>.dmg`. See
[Distribution](DISTRIBUTION.md) for signing, notarization and the expected
Gatekeeper verdicts.

## Runtime resolution

`PythonRuntime.resolve()` order:

1. `OMLX_PYTHON_OVERRIDE`
2. `~/Library/Application Support/oMLX/runtime/venv/bin/python3`
3. the legacy embedded bundle (`Contents/Resources/Python/cpython-3.11`)

The external venv gets `PYTHONPATH=Contents/Resources`; the embedded runtime
keeps `PYTHONHOME` and the MLX site path.

## First-run GUI installer

The slim app bundles `uv` (~27 MB), `pyproject.toml`, `uv.lock`,
`.python-version` and the fallback provision script under
`Contents/Resources/runtime{,-project}`.

The Welcome setup shows a **Python runtime required** card with an explicit
**Install Runtime…** button and a live log. `RuntimeInstaller` runs:

```text
uv python install <pin>
uv venv --clear --python <pin> …/oMLX/runtime/venv
uv sync --frozen --no-dev --no-install-project --extra cluster …
```

The download never starts without the click. A successful install makes
`PythonRuntime.resolve()` find the venv and the normal Start Server path
continues.

### Testable shape

`RuntimeInstaller` is not a hard-wired enum. It is a struct with injected
`runtimeRoot`, `projectDirectory` and `uvCandidates`; `RuntimeInstaller.shared`
keeps the production locations and the static `isInstalled` /
`install(progress:)` members the Welcome view model calls. That lets the
clean-first-run path be exercised against a temporary directory with no
network:

- `RuntimeInstaller.plan(pythonVersion:venv:)` is the pure command plan
  (`RuntimeInstallStep`), so a test pins the exact `uv` surface, the
  `UV_PROJECT_ENVIRONMENT` target and `--frozen --no-dev --no-install-project`.
- `pinnedPythonVersion()` follows the bundled `.python-version` and falls back
  to the baked-in pin.
- `isInstalled` requires an **executable** `venv/bin/python3` (a stray file does
  not count).
- `install()` validates `uv` **and** `uv.lock` before creating anything, so a
  broken bundle fails with `missingUV`/`missingProject` and leaves no half-built
  venv.

`apps/omlx-mac/Tests/oMLXTests/RuntimeInstallerTests.swift` covers those cases;
all run offline (the network install itself is exercised by hand, see below).

### What a real clean first-run test still needs

- The notarized DMG (or the `SlimStage` app) on a fresh user, or with
  `~/Library/Application Support/oMLX/runtime` moved aside.
- Network for the ~1 GB download; the pinned interpreter comes from `uv` or
  Homebrew.
- A click on **Install Runtime…**, then: the card disappears,
  `PythonRuntime.resolve()` reports the external venv, and **Start Server**
  reaches the dashboard. A failed/offline attempt should leave the card visible
  with the error text and no partial venv — the preflight covers the
  missing-bundle half, not a dropped connection.

## Open

- Exercise the first-run card on a clean user/runtime; add repair/remove and
  offline/error messaging.
- Runtime updates after a lock bump.
- The SMJobBless install and a clean first-run runtime install still need the
  click-through test.