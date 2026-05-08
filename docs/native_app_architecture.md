# Native oMLX.app Architecture

Deep-dive on the Swift menubar + AppView rewrite. The high-level roadmap lives
in [`plan.md`](../plan.md); this document covers the architectural detail a new
contributor needs before touching code.

---

## Process model

`oMLX.app` is a single bundle that runs **two cooperating processes**:

| Process | Language | Lifetime | Responsibility |
|---|---|---|---|
| **Parent** — the `.app` itself | Swift / SwiftUI / AppKit | User session (LSUIElement) | NSStatusItem + menu, AppView (Settings), Welcome wizard, server lifecycle, auto-updater |
| **Child** — `omlx serve` | Python (FastAPI) | Spawned/killed by parent | OpenAI/Anthropic-compatible API on `127.0.0.1:{port}` and the existing `/admin/api/*` endpoints |

```
┌─ oMLX.app (parent, Swift) ──────────────────────────────┐
│  NSStatusItem ─── menu (Start/Stop/Admin/Quit/…)        │
│  AppView (NavigationSplitView, 9 screens)               │
│  WelcomeWindow (5-step wizard)                          │
│  ServerProcess ─┬─ spawns ─► python -m omlx serve …     │
│                 │                                       │
│                 └─ HTTP localhost:{port} ◄──────────────┼──► /admin/api/* (FastAPI child)
│  SparkleUpdater                                         │
└─────────────────────────────────────────────────────────┘
```

### Why parent-child instead of XPC

Earlier sketches considered an XPC service to bridge a separate AppView and
menubar. The single-`.app` consolidation removes that need: lifecycle control
is direct (`Process.terminate()`), and everything else runs over plain HTTP
to a localhost port the parent already controls. No new IPC surface.

### Lifecycle invariants

1. **The parent always owns the child.** If the parent crashes, the child
   must be reaped within ≤5 s. We register `atexit` + signal handlers in the
   child (Python side, in `omlx serve`'s existing process group) and a
   `Process.terminationHandler` + SIGCHLD watcher in the parent (Swift).
2. **Child crashes are recoverable.** `ServerProcess` runs a 2 s health-check
   loop against `:{port}/health`. Three consecutive failures → auto-restart
   (mirrors `packaging/omlx_app/server_manager.py`).
3. **Sparkle relaunch terminates the child first.** SIGTERM → wait ≤5 s →
   SIGKILL → Sparkle relaunches the parent → parent re-spawns the child.
   See "Auto-updater" below.

---

## Networking

### Auth flow

Identical to today's browser flow, simplified by being in-process.

1. On parent launch, read API key from `AppConfig` (Keychain-backed; falls
   back to `~/Library/Application Support/oMLX/config.json` for migration).
2. `OMLXClient` POSTs `/admin/api/login` with the API key once at startup.
3. Subsequent calls share `URLSession.shared.httpCookieStorage`. The session
   cookie is the only thing carried.
4. On 401, retry the login once, then surface "Re-authenticate…" if it
   continues to fail.

### Endpoint coverage

The 76 `/admin/api/*` endpoints already cover every screen we ship. **Zero
server changes are required.** See `plan.md` §6 for the full mapping.

`OMLXClient` exposes typed methods (`getGlobalSettings()`,
`updateGlobalSettings(_:)`, `listModels()`, `loadModel(id:)`, …) backed by
`Endpoints.swift` + `DTO/` Codable structs. Endpoints are typed; the client
itself is dumb (no caching, no offline mode).

---

## Bundling Python inside the `.app`

`venvstacks` produces three layers today; we keep two and drop one:

| Layer | Source | Lands at |
|---|---|---|
| `cpython-3.11` runtime | `packaging/venvstacks.toml [[runtimes]]` | `oMLX.app/Contents/Frameworks/cpython3.11/` |
| `mlx-framework` (server deps) | `packaging/venvstacks.toml [[frameworks]]` | `oMLX.app/Contents/Frameworks/mlx-framework/` |
| ~~`omlx-app` (Python menubar)~~ | ~~`packaging/venvstacks.toml [[applications]]`~~ | dropped at PR 12 |

### Locating Python from Swift

`PythonRuntime.swift` resolves the bundled interpreter relative to
`Bundle.main.bundleURL`:

```
Contents/Frameworks/cpython3.11/bin/python3.11
```

with `PYTHONPATH` pointing into `Contents/Frameworks/mlx-framework/lib/python3.11/site-packages`.
The `omlx` package itself is part of the `mlx-framework` layer (installed
during the venvstacks build via `pyproject.toml`).

### Spawn invocation

```swift
let proc = Process()
proc.executableURL = PythonRuntime.shared.python3
proc.arguments = ["-m", "omlx", "serve", "--host", "127.0.0.1", "--port", "\(port)"]
proc.environment = ServerProcess.environment(port: port)
proc.terminationHandler = { … }
try proc.run()
```

Logs are tee'd to `~/Library/Application Support/oMLX/logs/server.log` via a
`Pipe()` plumbed into both `standardOutput` and `standardError`.

### Codesign

`codesign --deep --options runtime --sign "$DEVELOPER_ID_APPLICATION"` over the
whole `.app` works **provided** we preserve the dylib hashes venvstacks records
in its shipped `_codesign/` metadata. PR 2 establishes the canonical command in
`packaging/build.py` and adds a CI check that asserts:

- `codesign --verify --deep --strict oMLX.app` succeeds.
- `spctl --assess --type execute --verbose=4 oMLX.app` returns "accepted".

---

## Status item / Menubar

`MenubarController` owns:

- **Icon**: `menubar-outline` template image (from `omlx/admin/static/menubar-outline.svg`)
  when the server is stopped; `menubar-filled` when running. NSImage
  `isTemplate = true` so the icon auto-inverts under the system menubar
  background.
- **Menu**: dynamic. Built once on creation, refreshed in `menuWillOpen`. Items:
  - Status header (server status + uptime + port)
  - Stop / Restart / Force Restart / Start (only one shown at a time)
  - Admin Panel (opens AppView)
  - Chat with oMLX (opens `/admin/chat` in default browser via the existing
    auto-login redirect)
  - Settings… (Cmd-,) — opens AppView (same window as Admin Panel)
  - Check for Updates… (delegates to Sparkle; same flow as Status →
    Updates → Check Now)
  - Quit oMLX (Cmd-Q)
- **Stats poller**: 1 Hz `URLSession` GET to `/admin/api/stats`, mirroring
  `app.py:1745+`. Posts to `MenubarController` via Combine; menu items render
  the freshest values whenever the menu opens.
- **Visibility watcher**: ports the Bartender / hidden-icon detection logic
  in `app.py:355-444` line-for-line. Same alert copy, same recovery flow.

---

## AppView

`AppView` is a single SwiftUI scene presented as both:

- **Settings** (Cmd-,)
- **Admin Panel** (menubar item)

Same window. The two entry points just choose a default selection
(`activeNav: .server` for Settings; `activeNav: .status` for Admin Panel).

### Layout

```
NavigationSplitView {
  Sidebar — squircle gradient icons, 3 grouped sections
} content: {
  the selected screen
}
.frame(minWidth: 1140, minHeight: 760)
```

Sidebar items (per the design):

| Group | Items |
|---|---|
| Server | Server · Status · Logs |
| AI | Models · Downloads · Integrations |
| General | Security · About |

### Drill-down

`Models → {model} → Settings` is implemented as a navigation push within the
content pane (not a new sidebar entry). State is held by the parent
`AppViewModel`.

### Theme

`Theme.swift` ports the JSX `themeFor()` tokens verbatim
(`omlx-components.jsx:10-94`). Three layers:

1. **Colors** — `accent`, `windowBg`, `sidebarBg`, etc. Resolved from a
   `dark: Bool` derived from `@Environment(\.colorScheme)`.
2. **Materials** — `Glass.swift` exposes `.appGlass()` modifier:
   - `if #available(macOS 26.0, *) { .glassEffect(...) }`
   - else `.background(.regularMaterial)`.
3. **Primitives** — `ListGroup`, `Row`, `SectionHeader`, `StatusPill`,
   `CodeChip`, `Squircle`, `Popup`, `TextInput`, `Segmented`, `Button`. All
   live in `Sources/Theme/Components/` with `#Preview` in light + dark.

---

## Auto-updater (Sparkle)

We use [Sparkle](https://sparkle-project.org/) with three GitHub-releases-backed
appcasts.

### Channel mapping

| Channel | Source | Notes |
|---|---|---|
| Stable | `gh release list` non-prerelease tags | Default. Matches today's `select_latest_stable_release`. |
| Beta | Releases with `prerelease: true` and tags matching `*-beta*` | |
| Nightly | A single floating `nightly` tag, updated by a CI cron job | One DMG, overwritten on each successful main-branch build. |

A small CI job runs `Scripts/build_appcast.py` after every release to
generate three `appcast-*.xml` files committed to `gh-pages`. Sparkle is
configured with `SUFeedURL` per channel; switching channels changes the feed
URL and re-checks immediately.

### Install & Restart sequencing

1. User clicks **Install & Restart** in Status → Updates.
2. `SparkleUpdater.installUpdate()` is invoked.
3. Sparkle's installer requests app termination via
   `applicationShouldTerminate`.
4. `AppDelegate.applicationShouldTerminate` calls
   `ServerProcess.terminate()`:
   - Send SIGTERM to the child PID.
   - Wait up to 5 s for graceful shutdown (FastAPI handles this; releases
     port `:{port}`).
   - If still alive, SIGKILL.
5. Return `.terminateNow`.
6. Sparkle replaces the `.app` and relaunches; new parent re-spawns the child.

If the user clicks **Check for Updates…** from the menubar instead, the same
controller is used; the only difference is `installUpdate(deferred: true)` —
Sparkle remembers the update for next launch (Status row keeps showing "{ver}
available" until consumed).

### Manual UI (Status → Updates section)

Three render states (per design v2, `omlx-screens.jsx:862-923`):

| State | Headline | Subtitle | Button |
|---|---|---|---|
| Idle | "oMLX is up to date" | "Last checked {ts}" | "Check Now" (plain) |
| Checking | "Checking for updates…" | "Checking GitHub releases…" | "Checking…" (disabled) |
| Available | "oMLX {ver} is available" | "You have {current} · {size} download" | "Install & Restart" (primary) |

Plus three rows below:

- **Update Channel** — Stable / Beta / Nightly
- **Automatically Check** — toggle (on by default)
- **Auto-download Updates** — toggle (off by default)

`UpdateController` exposes `@Published` properties bound to those views and
delegates `checkForUpdates()` / `installUpdate()` to `SparkleUpdater`. PR 7
ships `UpdateController` with a stub backend (1.4 s simulated check); PR 11
swaps in real Sparkle.

---

## Welcome wizard

Five steps, ported from `packaging/omlx_app/welcome.py`:

1. Welcome (project intro, hardware check)
2. Hugging Face mirror (default `huggingface.co`; users in CN can pick
   `hf-mirror.com` etc.)
3. Pick a starter model (downloads via `/admin/api/hf/download`)
4. Set or generate API key
5. Ready (summary + finish)

PR 10 captures the Python wizard's exact state machine in
`docs/welcome-spec.md` first, then ports against the spec. The wizard runs
inside its own `NSWindow` (not a sheet) so the AppView remains independently
usable.

---

## Configuration & secrets

| | |
|---|---|
| **Where** | `~/Library/Application Support/oMLX/config.json` (kept compatible with today's Python format for one release; new Swift config dir is `oMLX-next` during the rewrite) |
| **Schema** | Mirrors `packaging/omlx_app/config.py`'s `ServerConfig` + adds `update.{channel,autoCheck,autoDownload,lastChecked}` |
| **API keys** | Keychain item `oMLX server API key` (legacy plaintext key in `config.json` migrated on first run) |
| **Logs** | `~/Library/Application Support/oMLX/logs/server.log` (tee'd from server process), `…/menubar.log` (Swift log handler) |

---

## Build pipeline (post-rewrite)

```
                ┌──────────────────────────────────────────┐
                │ packaging/build.py                       │
                ├──────────────────────────────────────────┤
                │ 1. venvstacks build                      │
                │    └─ produces _build/Frameworks/        │
                │       ├─ cpython3.11/                    │
                │       └─ mlx-framework/                  │
                │                                          │
                │ 2. xcodebuild -scheme oMLX-next archive  │
                │    └─ produces oMLX-next.app             │
                │                                          │
                │ 3. cp -R venvstacks/Frameworks/* into    │
                │    oMLX-next.app/Contents/Frameworks/    │
                │                                          │
                │ 4. codesign --deep                       │
                │ 5. notarytool submit + staple            │
                │ 6. hdiutil create oMLX-next.dmg          │
                └──────────────────────────────────────────┘
```

CI (GitHub Actions) runs steps 1+2 on every PR; 3+ only on tags.

---

## Out of scope (browser-only)

| Surface | Today | After rewrite |
|---|---|---|
| Benchmarks (`/admin/api/bench/*`) | HTML in `/admin/dashboard` | Same. Linked from Status as "Open Benchmarks…" → browser. |
| ModelScope downloader (`/admin/api/ms/*`) | HTML | Same. Linked from Downloads → browser. |
| HF uploader, oQ manager | HTML | Same. Linked from About / Models → browser. |
| Chat (`/admin/chat`) | HTML | Same. Menubar "Chat with oMLX" opens via auto-login redirect. |

The full HTML admin panel keeps working (it shares the API). Power users can
still hit `http://127.0.0.1:{port}/admin/dashboard` directly.

---

## References

- [`plan.md`](../plan.md) — phase-by-phase roadmap and progress tracker.
- [`packaging/README.md`](../packaging/README.md) — build pipeline.
- [`packaging/venvstacks.toml`](../packaging/venvstacks.toml) — Python layer
  configuration.
- Design source bundles: `/tmp/omlx-design/` (v1) and `/tmp/omlx-design-v2/`
  (v2 — adds Updates section).
