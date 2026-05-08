# Native oMLX.app — Swift Rewrite Roadmap

Single source of truth for the menubar+AppView Swift rewrite. Updated as each PR
ships.

- **Branch:** `feature/native-app-swift-rewrite`
- **Worktree:** `/Users/Fido/workspace/omlx.worktrees/native-app-swift-rewrite`
- **Strategy:** side-by-side `oMLX-next.app` during the rewrite; promote to
  `oMLX.app` at Phase 12 cutover.

---

## Progress

| PR | Title | Status | Notes |
|----|-------|--------|-------|
| 0  | Land structure + plan doc | ✅ Merged | a19437c |
| 1  | Xcode project + minimal launchable shell | ✅ Merged | 65dabfe |
| 2  | Bundled Python runtime + ServerProcess stub | ✅ Merged | 75a5f69 |
| 3  | Theme + design system + previews | ✅ Merged | 8262581 |
| 4  | Menubar parity (status item, menu, stats poll, visibility watcher) | ✅ Merged | d6da035 |
| 5  | ServerProcess full lifecycle (port `server_manager.py`) | ✅ Merged | 91cf0c2 |
| 6  | AppView shell — NavigationSplitView + sidebar | ✅ Merged | bf90a74 |
| 7  | Configuration core: Server / Status / Logs (incl. Updates UI shell) | ✅ Merged | 63dbd71 |
| 8  | Models + Downloads + per-model Settings | ✅ Merged | 2597210 |
| 9  | Integrations + Security + About | ✅ Merged | a436b6c · About has no inline updater (lives on Status) |
| 10 | Welcome wizard | ✅ Merged | abb0f73 |
| 11 | Sparkle auto-updater + i18n | ✅ Merged | c63c015 · Sparkle 2.x SPM dep, real feed at build.py time |
| 12 | Cutover & deprecation | ⬜ Not started | rename `-next.app` → `oMLX.app`, drop `packaging/omlx_app/` |

Status legend: ⬜ Not started · 🟡 In progress · ✅ Merged · 🔁 Re-opened.

---

## 1. Decisions (locked)

| | |
|---|---|
| Toolkit | **SwiftUI** in an **Xcode project** |
| macOS floor | **15.0+**, with `if #available(macOS 26)` Tahoe-glass branches |
| Process model | **Single `oMLX.app`**, two processes inside: Swift UI/menubar (parent) + `omlx serve` Python (background subprocess) |
| Settings/Prefs | **Merged into the AppView** — old native Preferences window is removed |
| Rollout | **Side-by-side** as `oMLX-next.app` until Phase 12 cutover |
| Codesign | Same Developer ID as today's app |
| Chat | Menubar "Chat with oMLX" still opens `/admin/chat` in the default browser |
| Server lifecycle | **In-process** parent-child (no XPC); HTTP everywhere else |
| Auto-updater | Sparkle, GitHub releases as the appcast source of truth |
| Update channels | Stable = non-prerelease tags · Beta = prereleases tagged `*-beta*` · Nightly = floating `nightly` tag updated by CI cron |
| Update feed copy | "Checking GitHub releases…" (reword from prototype's `releases.omlx.app`) |
| Restart sequencing | SIGTERM child → wait ≤5s → SIGKILL → Sparkle relaunch |

### Out of scope (browser-only after rewrite)
| Surface | After |
|---|---|
| Benchmarks (`/admin/api/bench/*`) | Linked from Status as "Open Benchmarks…" → browser |
| ModelScope downloader (`/admin/api/ms/*`) | Linked from Downloads → browser |
| HF uploader, oQ manager | Linked from About / Models → browser |
| Chat (`/admin/chat`) | Menubar "Chat with oMLX" → browser via existing auto-login flow |
| `omlx serve` CLI / Python package | Untouched |
| `Formula/omlx.rb` | Untouched (the brew formula serves the CLI, not the `.app`) |

---

## 2. Architecture

```
oMLX.app (single bundle)
├── Contents/
│   ├── MacOS/oMLX                          ← Swift binary (LSUIElement, NSStatusItem)
│   ├── Resources/                          ← assets, Localizable.xcstrings, AppIcon
│   ├── Frameworks/
│   │   ├── cpython3.11/                    ← venvstacks runtime layer
│   │   └── mlx-framework/                  ← venvstacks server-deps layer
│   └── Info.plist
│
└── runtime: two processes
    ┌────────────────────────────────────┐    spawn (Process API)    ┌──────────────────────────┐
    │ Parent: Swift app                  │ ─────────────────────────►│ Child: python -m omlx    │
    │  • NSStatusItem + dynamic menu     │                            │   serve --host 127.0.0.1 │
    │  • SwiftUI AppView (Settings)      │  HTTP localhost:{port}     │   --port {port}          │
    │  • SwiftUI Welcome wizard          │ ◄─────────────────────────┤  (FastAPI; /admin/api/…) │
    │  • ServerProcess (lifecycle)       │                            └──────────────────────────┘
    │  • Sparkle auto-updater            │
    └────────────────────────────────────┘
```

- **No XPC.** Server lifecycle is parent→child via `Process`. AppView ↔ server
  is plain HTTP using `URLSession.shared.httpCookieStorage` after one-time
  `/admin/api/login`.
- **Python responsibility shrinks** to the FastAPI server only. Everything
  menubar/UI is Swift.
- **Two-track during transition:** the Python `oMLX.app` keeps building until
  PR 12. Swift target builds to `oMLX-next.app`/`oMLX-next.dmg` so users can
  run both side-by-side.

For deeper architectural detail, see
[`docs/native_app_architecture.md`](docs/native_app_architecture.md).

---

## 3. Repository layout (target end-state)

```
apps/
  omlx-mac/                              # NEW — Xcode project
    oMLX.xcodeproj/
    Sources/
      App/
        oMLXApp.swift                    # @main
        AppDelegate.swift                # NSApplicationDelegate, LSUIElement
      Menubar/
        MenubarController.swift          # NSStatusItem + menu
        MenubarStatsPoller.swift         # /admin/api/stats poll @1Hz
        MenubarVisibilityWatcher.swift   # Bartender / hidden-icon detection
      Server/
        ServerProcess.swift              # Process spawn + health check
        PythonRuntime.swift              # locates bundled cpython3.11
        PortConflictResolver.swift
      AppView/
        AppView.swift                    # NavigationSplitView shell
        Sidebar.swift
        Screens/
          ServerScreen.swift
          StatusScreen.swift             # incl. Updates section (design v2)
          LogsScreen.swift
          ModelsScreen.swift
          ModelSettingsScreen.swift      # Profiles/Basic/Advanced/Aliases
          DownloadsScreen.swift
          IntegrationsScreen.swift
          SecurityScreen.swift
          AboutScreen.swift
      Welcome/
        WelcomeWindow.swift              # SwiftUI port of welcome.py (5 steps)
      Theme/
        Theme.swift                      # ports themeFor() tokens verbatim
        Squircle.swift                   # gradient rounded-square icons
        Glass.swift                      # @available(macOS 26) glass; else material
        Components/                      # ListGroup, Row, SectionHeader, StatusPill,
                                         # CodeChip, Popup, TextInput, Segmented, Button
      Net/
        OMLXClient.swift                 # URLSession async; auto-login on first call
        Endpoints.swift                  # typed wrappers for /admin/api/*
        DTO/                             # Codable structs mirroring server JSON
      Updater/
        SparkleUpdater.swift             # SUUpdaterDelegate, feed = GitHub releases
        UpdateController.swift           # state for Status → Updates section
      Config/
        AppConfig.swift                  # ~/Library/Application Support/oMLX/config.json
        Keychain.swift                   # API key storage
      Localization/
        Localizable.xcstrings            # generated from omlx/admin/i18n/*.json
    Resources/
      Assets.xcassets/                   # app icon, menubar-outline, menubar-filled
    Tests/
      DTOTests/                          # JSON round-trip vs captured fixtures
      ScreenSnapshotTests/               # SwiftUI Preview snapshots (light/dark)
    Scripts/
      i18n_sync.py                       # generate xcstrings from omlx/admin/i18n
      build.sh                           # xcodebuild + codesign + embed Frameworks
      capture_fixtures.sh                # save /admin/api responses for tests

packaging/
  build.py                               # MODIFIED — orchestrates venvstacks + xcodebuild,
                                         #   embeds Frameworks/, codesigns, builds DMG
  venvstacks.toml                        # MODIFIED — drops [[applications]] omlx-app block at PR 12
  omlx_app/                              # KEPT during transition; deleted in PR 12
  README.md                              # rewritten to describe two-track build

docs/
  native_app_architecture.md             # NEW — process model, IPC, build pipeline
  welcome-spec.md                        # NEW (PR 10) — captures Python wizard before port
```

The `omlx/` Python package is **untouched** (the FastAPI server stays as-is).
Only `packaging/omlx_app/` (the Python menubar) is replaced.

---

## 4. PR / phase split

Each PR is shippable independently because `oMLX-next.app` builds alongside the
still-working `oMLX.app` until PR 12.

### PR 0 — Land structure + plan doc
**Touches:** `apps/omlx-mac/` empty dirs (with `.gitkeep`), `plan.md`,
`docs/native_app_architecture.md`, `packaging/README.md`.

**Outcome:** Branch alive; layout committed; no functional change.

### PR 1 — Xcode project + minimal launchable shell
**Touches:** `apps/omlx-mac/oMLX.xcodeproj`, `oMLXApp.swift`, `AppDelegate.swift`,
stub `MenubarController` (placeholder icon, single Quit menu item), `Info.plist`,
entitlements, `Scripts/build.sh` initial version.

**Outcome:** `xcodebuild -scheme oMLX-next` produces a launchable
`oMLX-next.app` — empty status item, no server, no AppView.

### PR 2 — Bundled Python runtime + ServerProcess stub
**Touches:** `packaging/build.py` (embed `Frameworks/cpython3.11` +
`mlx-framework` into the `.app`), `Sources/Server/PythonRuntime.swift`,
`Sources/Server/ServerProcess.swift` (spawn-only, no health check yet).

**Outcome:** Launching `oMLX-next.app` spawns `omlx serve` from the bundled
runtime. Verified by curling `:8080/health`.

### PR 3 — Theme + design system + previews
**Touches:** `Sources/Theme/`, `Sources/Theme/Components/*`,
`Sources/Theme/Glass.swift`, `Sources/Theme/Squircle.swift`, `#Preview` blocks.

**Outcome:** Every primitive renders at design spec in light + dark; snapshot
baseline captured. No screens yet.

### PR 4 — Menubar parity (Swift port of menubar UX)
**Touches:** `Sources/Menubar/MenubarController.swift`,
`Sources/Menubar/MenubarStatsPoller.swift`,
`Sources/Menubar/MenubarVisibilityWatcher.swift`, port menubar SVG icons to
`Resources/Assets.xcassets`.

**Outcome:** Status item + dynamic menu match today's Python behavior: Status
header, Start/Stop/Restart/Force Restart, Admin Panel (opens AppView), Chat
(opens browser), Settings…, Quit. Live tok/s + req in flight. Bartender /
hidden-icon detection with the same alert copy.

### PR 5 — ServerProcess full lifecycle
**Touches:** `Sources/Server/ServerProcess.swift` (full version),
`Sources/Server/PortConflictResolver.swift`, banner UI hook in AppView shell.

**Outcome:** Health-check loop, crash detection + auto-restart, port-conflict
alert, force-restart. Equivalent to `packaging/omlx_app/server_manager.py:1-460`.

### PR 6 — AppView shell + sidebar
**Touches:** `Sources/AppView/AppView.swift`, `Sources/AppView/Sidebar.swift`,
navigation state, opens via `Cmd-,` and "Admin Panel" menu item.

**Outcome:** `NavigationSplitView` with the 9 sidebar items + squircle gradient
icons. Empty placeholder content per item. Search wired but inert.

### PR 7 — Configuration core: Server / Status / Logs
**Touches:** `Sources/AppView/Screens/ServerScreen.swift`,
`Sources/AppView/Screens/StatusScreen.swift`,
`Sources/AppView/Screens/LogsScreen.swift`,
`Sources/Net/OMLXClient.swift`, `Sources/Net/Endpoints.swift`,
`Sources/Net/DTO/` (server, status, logs subset),
`Sources/Updater/UpdateController.swift` (stub).

**Outcome:** Reads `/admin/api/global-settings`, `/api/server-info`,
`/api/stats`, `/api/logs`. Writes `/api/global-settings`. ServerHero
Start/Stop/Restart wired through `ServerProcess`.

**Status screen v2:** also renders the Updates `ListGroup` (three-state status
row, Channel popup, AutoCheck + AutoDownload toggles) bound to a stub
`UpdateController` that simulates a 1.4s `checkForUpdates()`. Real Sparkle wiring
lands in PR 11.

### PR 8 — Models + Downloads + per-model Settings
**Touches:** `Sources/AppView/Screens/ModelsScreen.swift`,
`Sources/AppView/Screens/ModelSettingsScreen.swift`,
`Sources/AppView/Screens/DownloadsScreen.swift`, additional Endpoints + DTOs.

**Outcome:** Active models, library, load/unload via
`/admin/api/models/{id}/{load,unload}`. HF downloader via `/admin/api/hf/*`
with 1Hz polling. Full Profiles UI (Preset/Global/Model + chips + new/delete)
bound to `/admin/api/models/{id}/profiles` and `/admin/api/profile-templates`.

### PR 9 — Integrations + Security + About
**Touches:** `Sources/AppView/Screens/IntegrationsScreen.swift`,
`Sources/AppView/Screens/SecurityScreen.swift`,
`Sources/AppView/Screens/AboutScreen.swift`, additional Endpoints + DTOs.

**Outcome:** Claude Code routing, OpenAI compat defaults, sub-keys CRUD via
`/admin/api/sub-keys`, version + license + credits + links on About. The
manual update button does **not** live here (it's on Status, per design v2).

### PR 10 — Welcome wizard
**Touches:** `Sources/Welcome/WelcomeWindow.swift`, `docs/welcome-spec.md`,
config-first-run trigger in `AppDelegate`.

**Outcome:** Native onboarding: welcome → HF mirror → model pick → API key
→ ready. Same data flow as `packaging/omlx_app/welcome.py` today.

### PR 11 — Sparkle auto-updater + i18n
**Touches:** Sparkle SwiftPM dep, `Sources/Updater/SparkleUpdater.swift`,
`Scripts/i18n_sync.py`, `Sources/Localization/Localizable.xcstrings`.

**Outcome:** Auto-update via GitHub releases as appcast (Stable / Beta /
Nightly). "Install & Restart" calls Sparkle's update flow with a
`terminationHandler` that gracefully stops `ServerProcess` (SIGTERM →
≤5s → SIGKILL) before Sparkle relaunches the parent. Localizable strings
synced from `omlx/admin/i18n/*.json` (en, zh, zh-TW, ja, ko). CI fails on i18n
drift.

### PR 12 — Cutover & deprecation
**Touches:** rename target → `oMLX.app`, remove `packaging/omlx_app/`, remove
`[[applications]] omlx-app` from `venvstacks.toml`, update DMG branding,
README, install docs.

**Outcome:** Single Swift `.app` is the only macOS distribution. Python menubar
deleted (kept in git history). Smoke-test on macOS 15.x and 26.x. Release
notes call out migration.

### Critical-path order
0 → 1 → 2 → 3 → 4 ↔ 5 (parallel) → 6 → 7 → 8 ↔ 9 (parallel) → 10 → 11 → 12.

---

## 5. Build pipeline (`packaging/build.py` after rewrite)

1. `venvstacks` builds **runtime** + **framework** layers (drops the `omlx-app`
   Python application layer at PR 12).
2. `xcodebuild -scheme oMLX-next archive` produces the Swift `.app`.
3. Embed venvstacks `Frameworks/` into `.app/Contents/Frameworks/`.
4. `codesign --deep --sign "$DEVELOPER_ID"` (same identity as today).
5. `xcrun notarytool` for notarization.
6. `hdiutil` builds the DMG.

The `[[runtimes]]` and `[[frameworks]]` blocks of `venvstacks.toml` stay; only
`[[applications]] omlx-app` is removed at Phase 12.

---

## 6. API contract (server unchanged)

The server's 76 `/admin/api/*` endpoints already cover every screen we're
shipping. **Zero server changes required.** Mapping:

| Screen | Endpoints |
|---|---|
| Server | `GET/POST /api/global-settings`, `GET /api/server-info` |
| Status | `GET /api/stats`, `GET /api/server-info`, `GET /api/device-info`; Updates section calls `Sparkle.checkForUpdates` (out-of-band, not via the server API) |
| Logs | `GET /api/logs` |
| Models | `GET /api/models`, `POST /api/models/{id}/load`, `POST /api/models/{id}/unload`, `POST /api/reload` |
| Model Settings | `GET/POST/PUT/DELETE /api/models/{id}/profiles`, `POST /api/models/{id}/profiles/{name}/apply`, `GET /api/profile-fields`, `GET/POST/PUT/DELETE /api/profile-templates`, `GET /api/models/{id}/generation_config` |
| Downloads | `POST /api/hf/download`, `GET /api/hf/tasks`, cancel/retry/delete, `GET /api/hf/recommended`, `/api/hf/search` |
| Integrations | `GET/POST /api/global-settings` (claude.* + openai.*) |
| Security | `POST /api/setup-api-key`, `POST/DELETE /api/sub-keys` |
| About | `GET /api/server-info`, `GET /api/update-check` (legacy fallback; Sparkle is canonical post-PR 11) |

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Two-track build (Python + Swift) for 11 PRs. | `oMLX-next.app` is fully separate target; never breaks the Python `oMLX.app` build. CI runs both. |
| Bundled Python integrity post-codesign. | `codesign --deep` over the embedded `Frameworks/cpython3.11/` is fragile. PR 2 establishes the working incantation (preserving venvstacks' shipped `_codesign/` metadata) and adds a CI check. |
| Server-process orphaning on parent crash. | Spawn with `setpgid` + handle SIGCHLD; on parent termination, send SIGTERM via `atexit` + `signal` handlers. Swift: `Process.terminationHandler` + signal masks. |
| Pixel parity vs design. | Theme tokens lifted verbatim from `omlx-components.jsx:10-94`. Snapshot tests per screen at 1140×760 in light + dark. Manual visual diff vs design canvas before each PR merges. |
| Tahoe API gating. | All Tahoe-only calls behind `if #available(macOS 26.0, *)`; fall back to `.regularMaterial` and stock `Picker`/`Form`. Snapshot CI runs on 15 and 26 simulators. |
| Profiles state-machine complexity (Preset/Global/Model). | PR 8 lands DTOs + state machine + unit tests against fixtures captured by `Scripts/capture_fixtures.sh` from a running server. |
| Bartender / hidden-icon detection fragile in Swift. | Port `app.py:355-444` line-by-line. Same `NSWindowOcclusionState` + frame inspection. Same alert copy + recovery flow. |
| i18n drift between Python and Swift. | `Scripts/i18n_sync.py` runs in CI and fails the build if any key in `omlx/admin/i18n/*.json` is missing from `Localizable.xcstrings`. |
| Welcome wizard regressions during port. | PR 10 captures the Python wizard's state-machine in `docs/welcome-spec.md` first, ports against the spec. |
| Sparkle "Install & Restart" orphans the Python child. | `terminationHandler` on `ServerProcess`: SIGTERM → wait ≤5s → SIGKILL → Sparkle relaunches. Confirmed contract. |
| User has both old `oMLX.app` and new `oMLX-next.app` installed. | Different bundle IDs (`app.omlx` vs `app.omlx-next`); separate config dirs (`oMLX` vs `oMLX-next`). No collision. Cutover at Phase 12 renames + migrates config. |

---

## 8. Design source

- v1 bundle (initial design): `/tmp/omlx-design/omlx-macos-app-frontend/`
- v2 bundle (added Updates section to Status): `/tmp/omlx-design-v2/omlx-macos-app-frontend/`
- Variant accepted: **Variant A — Classic System Settings (Sequoia/Tahoe)**.
  Window 1140×760. Sidebar groups: Server (Server, Status, Logs), AI (Models,
  Downloads, Integrations), General (Security, About).
- Theme tokens verbatim from `omlx-components.jsx:10-94`.

---

## 9. How to update this file

When a PR ships:
1. Bump its row in the **Progress** table to ✅ Merged.
2. Add a one-line note in the Notes column with the merge commit short SHA.
3. If the PR scope changed during review, update the matching PR section in §4
   so future readers see the actual delivered shape.
4. If a new risk surfaced, append it to §7.
