# Welcome Wizard — Spec

Captures the first-run wizard's data flow so the SwiftUI port (PR 10) can
verify behavior parity against `packaging/omlx_app/welcome.py` without
diff-by-diff porting.

## Trigger

- **First run** = `~/Library/Application Support/oMLX-next/config.json`
  does not yet exist (matches the Python heuristic in
  `packaging/omlx_app/config.py:53`).
- `AppDelegate.applicationDidFinishLaunching` checks this **before**
  spawning `ServerProcess`. If first-run:
  1. Skip the auto-start.
  2. Construct `MenubarController` (server stays `nil` until onboarding
     completes — same as a port-conflict early-out today).
  3. Open the welcome window.
  4. Flip activation policy to `.regular` for as long as the wizard is
     visible so the user can see + close it; switch back to `.accessory`
     when the window closes (preserves the menubar-only ergonomics).
- The user can also reach the wizard later via the menubar's
  "Welcome…" item (PR 11 reuses the same window).

## Flow

Single window, four pages, Next/Back/Skip on the footer. Step bar at the
top.

1. **Welcome** — logo, tagline, "Get started" → Page 2.
2. **Storage** — captures `AppConfig.basePath`, an optional override for
   the model directory, and the port.
3. **API Key** — captures the initial admin API key. POSTs
   `/admin/api/setup-api-key` after the server is up.
4. **Ready** — recap, "Start Server" (spawns `ServerProcess` + applies
   API key + opens AppView). "Open Admin Panel & Close" performs the
   same start, then opens the admin URL in the browser.

The Python wizard's "Hugging Face Mirror" choice is exposed on Page 2 as
an optional dropdown, written via `hf_endpoint` on the
`/admin/api/global-settings` patch after the server is up.

## Data writes (in order, after user clicks Start Server on Page 4)

| Order | Target | Field(s) |
|---|---|---|
| 1 | `AppConfig` (local file) | `host`, `port`, `apiKey` (yes — also kept here for the Welcome resync) |
| 2 | `ServerProcess.start()` | spawn |
| 3 | `POST /admin/api/setup-api-key` (if key not yet set on server) | `api_key`, `api_key_confirm` |
| 4 | `POST /admin/api/global-settings` | `hf_endpoint` (only when user picked a non-default) |

If step 3 fails (e.g. server already has a key), we fall back to
`POST /admin/api/login` so the cookie jar is populated and the user
isn't immediately bumped to the login redirect on first AppView open.

## Validation

| Field | Rule | Source |
|---|---|---|
| Base directory | non-empty | `welcome.py:460-462` |
| Port | 1024 ≤ p ≤ 65535 (Python) — relaxed to 1 ≤ p ≤ 65535 in Swift to match `ServerScreen` | `welcome.py:466-472` |
| API key | length ≥ 4, no whitespace, printable only | `welcome.py:487-492` (matches `validate_api_key` server-side) |
| HF endpoint | optional URL; trailing-slash-tolerant | new in Swift wizard |

API key + confirm must match on the API Key page (per
`SetupApiKeyRequest` server-side — `omlx/admin/routes.py:1248-1249`).

## State machine

```
Welcome → Storage → APIKey → Ready
            ↑   ↓    ↑   ↓     ↓
            └───┴────┴───┘   Start Server (writes + spawn)
                              └─→ Close (the AppView opens automatically
                                  via the existing showSettingsWindow:
                                  hook in PR 1; menubar drops policy
                                  back to .accessory).
```

Skipping the wizard (top-right close box on macOS title bar) writes the
current Storage values + retains an empty API key. The user lands on
the AppView's Server screen showing the API-key-not-configured banner
(PR 9 SecurityScreen handles this state).

## Out of scope (PR 10)

- Downloading a starter model. Plan §4's "model pick" step is replaced
  by a hint on the Ready page pointing to Downloads — keeps the wizard
  fast and avoids a long-running task before the menubar appears.
- Sparkle pre-flight (PR 11 lands the appcast + auto-update wiring).
- i18n. Strings are hard-coded in English; PR 11 syncs from
  `omlx/admin/i18n/*.json` via `Scripts/i18n_sync.py`.

## Re-entry

The user can re-open the wizard from the menubar's overflow at any
time. Re-entry treats the existing `config.json` as a starting point
and skips the Welcome page. `applicationDidFinishLaunching` always
respects the saved config and never re-shows the wizard once
config.json exists.
