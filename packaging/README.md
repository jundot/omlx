# oMLX macOS App Packaging

Produces the venvstacks Python layers that the Swift macOS bundle
embeds. Building the user-facing `.app` itself is owned by
[`apps/omlx-mac/Scripts/build.sh`](../apps/omlx-mac/Scripts/build.sh);
this directory only hands it a `_export/` tree of Python layers.

> **PyObjC menubar retired.** The earlier Python / PyObjC menubar
> (`packaging/omlx_app/`) and the `packaging/build.py` `.app` + DMG
> pipeline that wrapped it have been removed. The Swift app under
> [`apps/omlx-mac/`](../apps/omlx-mac/) is now the only macOS bundle.

## Requirements

- macOS 15.0+ (Sequoia) — required by MLX ≥ 0.29.2
- Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.11–3.13 on the host
- venvstacks (installed via `pip install -e ".[dev]"` from the repo
  root, or any of `uv`, `pipx run`)
- Full Xcode and the downloadable Metal Toolchain for custom-kernel builds;
  Command Line Tools alone do not provide `xcrun metal`

## Build

```bash
# Re-export the venvstacks layers (cold ~10-20 min, warm ~4 min)
python packaging/build.py --venvstacks-only

# Stable fingerprint of the inputs that drive the export shape — used
# by build.sh to decide whether to re-export
python packaging/build.py --print-fingerprint
```

Then the Swift bundle:

```bash
apps/omlx-mac/Scripts/build.sh release             # full bundle
apps/omlx-mac/Scripts/build.sh release --no-rebuild-donor   # reuse _export/
apps/omlx-mac/Scripts/build.sh release --with-custom-kernel  # bundle Bonsai / GLM / MiniMax / Qwen kernels
```

## Donor and Custom-Kernel Build

The app embeds two venvstacks donor layers verbatim:

- `cpython-3.11`, the bundled interpreter and ABI target;
- `framework-mlx-base`, the bundled MLX and server dependencies.

By default `build.sh` uses the fingerprinted `packaging/_export/`. An explicit
`OMLX_DONOR_APP` uses that app's Python layers as-is, while
`--no-rebuild-donor` permits a stale local export or `/Applications/oMLX.app`
fallback. `swift` and `swift-fast` require and reuse an existing export.

With `--with-custom-kernel`, the script creates a build-only virtualenv from
the donor's CPython 3.11. It installs `[build-system].requires` from
`pyproject.toml`, then builds an oMLX wheel through PEP 517. This automatically
uses the pinned `mlx==0.32.0` and `nanobind==2.13.0` ABI pair without adding
build tools to the runtime layers. The script extracts only the expected
Bonsai, GLM, MiniMax, and Qwen native artifacts from that wheel; Qwen's NAX
metallib is required when the active SDK supports it.

The app-build modes and environment overrides are documented at the top of
[`apps/omlx-mac/Scripts/build.sh`](../apps/omlx-mac/Scripts/build.sh).

## Output

```
packaging/
├── _build/         # venvstacks intermediate layers
├── _export/        # venvstacks export — embedded into the .app
└── _wheels/        # cached local wheels (e.g. mlx + mlx-metal pins)
```

## Layer Configuration

| Layer | Contents |
|-------|----------|
| Runtime (`cpython-3.11`) | Python 3.11 |
| Framework (`mlx-base`) | MLX, mlx-lm, mlx-vlm, FastAPI, transformers, mlx-audio, paroquant, spaCy |

No application layer — the Swift app is the application surface.

## Installation

The Swift build (`build.sh release`) produces
`apps/omlx-mac/build/Stage/oMLX.app` directly — no DMG step. To install:

1. Drag `apps/omlx-mac/build/Stage/oMLX.app` to `/Applications`, or
   `open` it in-place to launch from `apps/omlx-mac/build/Stage/`.
2. Launch the app (appears in the menubar).
3. Walk through the first-run wizard (Storage + API key), then Start
   Server.

> The DMGs on the [Releases](https://github.com/jundot/omlx/releases)
> page are produced by an off-tree maintainer pipeline, not by anything
> in this repo. End users follow the Releases install path; this
> section is for developers building from source.
