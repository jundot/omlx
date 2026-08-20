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
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11+ on the host
- venvstacks (installed via `pip install -e ".[dev]"` from the repo
  root, or any of `uv`, `pipx run`)

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
apps/omlx-mac/Scripts/build.sh release --with-custom-kernel  # bundle GLM-5.2 / MiniMax M3 native kernels
```

DS4/GGUF backend releases are built from the pinned source commit recorded in
`omlx/vendor/ds4/darwin-arm64/manifest.json`. Git tracks only the manifest plus
`LICENSE` and `README.md`; the `ds4-server` Mach-O binary and `metal/*.metal`
runtime files are staged by build jobs and are not present in source archives.
To prepare the app-bundle support tree, run:

```bash
scripts/build-ds4-support.sh
apps/omlx-mac/Scripts/build.sh release
```

The helper shallow-fetches the pinned DS4 commit, runs the manifest
`build_command`, validates `ds4-server --help`, and writes
`packaging/DS4Support/`. The bundle step copies that same validated tree into
`Contents/Resources/DS4Support`, so release and auto-update artifacts carry the
prebuilt support files and end users do not need Xcode or a runtime network
fetch. App-bundle builds require DS4 support by default; if no staged tree
exists, `build.sh` invokes the helper automatically. Local/dev bundles can opt
out with `OMLX_REQUIRE_DS4_BUNDLE=0`.

Custom maintainers can override the source pin without adding binaries to git:

```bash
scripts/build-ds4-support.sh \
  --source https://github.com/example/ds4.git \
  --commit <sha>
OMLX_DS4_BUNDLE_SOURCE=/path/to/ds4-support apps/omlx-mac/Scripts/build.sh release
```

Homebrew builds DS4 from the formula's pinned `resource "ds4"` during
`def install` and installs `ds4-server` plus `metal/` into the package support
directory. Source-clone users can run `omlx ds4 install` to build the same
support tree into `~/.omlx/support/ds4`; if they skip that step, the first DS4
model launch attempts the same pinned-source build when `ds4.auto_build` is
enabled. Missing `make`/Apple Command Line Tools produce a hint to run
`xcode-select --install`, and failures are remembered for the process so every
request does not retry the compile. Air-gapped or custom prebuilt deployments
keep using `ds4.support_dir` / `OMLX_DS4_SUPPORT_DIR` and `ds4.binary_path` /
`OMLX_DS4_BINARY_PATH`; set `ds4.auto_build=false` or
`OMLX_DS4_AUTO_BUILD=false` to require explicit provisioning.

AC #7 (#22): DS4 support provisioning remains configurable from admin
settings. The support directory, prebuilt binary path, auto-build toggle,
source repository, and source commit all persist through
`/admin/api/global-settings`.

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
