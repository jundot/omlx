# oMLX macOS App Packaging

Packages oMLX as a macOS menubar app using venvstacks.

> **Two-track build during the Swift rewrite.** The `oMLX.app` (Python /
> PyObjC) described below is the current shipping target. A SwiftUI rewrite
> is landing under [`apps/omlx-mac/`](../apps/omlx-mac/) and builds to a
> separate `oMLX-next.app` until the Phase 12 cutover. See
> [`plan.md`](../plan.md) and
> [`docs/native_app_architecture.md`](../docs/native_app_architecture.md) for
> the roadmap.

## Requirements

- macOS 15.0+ (Sequoia) — required by MLX >= 0.29.2
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- venvstacks: `pip install venvstacks`

## Build

```bash
cd packaging

# Full build (venvstacks + app bundle + DMG)
python build.py

# Skip venvstacks build (use existing environment)
python build.py --skip-venv

# DMG only (use existing build)
python build.py --dmg-only
```

## Output

```
packaging/
├── build/
│   ├── venvstacks/     # venvstacks build cache
│   ├── envs/           # Exported environments
│   └── oMLX.app/       # App bundle
└── dist/
    └── oMLX-<version>.dmg  # Distribution DMG
```

## Structure

```
oMLX.app/
├── Contents/
│   ├── Info.plist
│   ├── MacOS/
│   │   └── oMLX           # Launcher script
│   ├── Resources/
│   │   ├── omlx_app/      # Menubar app
│   │   ├── omlx/          # oMLX server
│   │   └── AppIcon.icns
│   └── Frameworks/
│       ├── cpython3.11/   # Python runtime
│       ├── mlx-framework/ # MLX + dependencies
│       └── omlx-app/      # App layer
```

## Layer Configuration

| Layer | Contents |
|-------|----------|
| Runtime | Python 3.11 |
| Framework | MLX, mlx-lm, mlx-vlm, FastAPI, transformers |
| Application | rumps, PyObjC |

## Installation

1. Open the DMG file
2. Drag oMLX.app to the Applications folder
3. Launch the app (appears in the menubar)
4. Set the model directory in Settings
5. Click Start Server
