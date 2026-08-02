# Native Codex Local Interceptor Plan

## Goal

Make the proven Codex local-inference interceptor a first-class OMLX feature:
select an OMLX model, click **Start Interceptor**, launch a fresh Codex desktop
session through a process-scoped proxy, and monitor local and OpenAI traffic in
the OMLX macOS UI.

The feature must preserve Codex's normal signed-in product surface. It must not
replace the OpenAI provider or edit Codex configuration.

## Non-negotiable invariants

1. `~/.codex/config.toml` remains byte-for-byte unchanged. Record its SHA-256
   before startup, while running, and after shutdown.
2. No system proxy setting, system keychain certificate, Codex app bundle, or
   Codex on-disk model cache is changed.
3. Proxy and CA configuration exists only in the environment of the OMLX-launched
   Codex process tree.
4. Only exact Codex Responses inference paths are eligible for local routing.
   Projects, automations, plugins, account state, MCP, model catalogue requests,
   and all unrelated traffic keep their normal OpenAI/ChatGPT destination.
5. Only the selected Codex model slot routes locally. Other Codex models continue
   to use OpenAI.
6. The local upstream is the running OMLX server on loopback. This PR does not
   add Claude, Kimi, Pi, OpenCode, public providers, or arbitrary remote servers.
7. OpenAI cookies, bearer tokens, account identifiers, and `x-openai-*` headers
   are stripped before a locally routed request is sent to OMLX.
8. Logs and UI state never contain prompts, response bodies, headers, cookies,
   query strings, or credentials.

## Branch implementation status

The branch implements the native proxy package, server-owned lifecycle manager,
authenticated admin control plane, process-scoped Codex CLI/desktop launchers,
bundled mitmproxy dependency, and SwiftUI controls described below. The runtime
also filters status fields at the writer, so diagnostics cannot become a traffic
capture through an accidental future call site.

Automated verification on the branch includes 140 focused Python tests, a
successful macOS app-and-test build, two passing Swift DTO tests, Python 3.11
bundle dependency resolution, and an actual `mitmdump` routing smoke test that
proved local model rewrite, credential replacement, response delivery, config
immutability, and absence of prompt/response/credential content from the status
file. The full upstream Python suite completed with 7,734 passes, 63 skips, and
12 failures in unrelated numerical-tolerance, ambient API-key, and legacy test
environment cases.

The staged-app desktop acceptance checklist remains a release gate. It cannot be
run from the Codex session creating this branch because the required fresh-app
launch would first quit that active Codex process.

## Current-state findings

- Upstream OMLX already exposes `/v1/responses`, model load/unload APIs, a native
  SwiftUI Integrations screen, and Codex launch commands.
- The existing OMLX Codex integration is not suitable for this feature. It writes
  `model_provider = "omlx"` and an OMLX provider section into
  `~/.codex/config.toml`, which crosses the feature boundary this work must keep.
- The working interceptor under
  `Harness/codex-agent-harness/scripts/` already implements the required data
  plane, request and stream compatibility, local model warm-up, privacy-safe
  receipts, and desktop launch environment.
- The live harness working tree is ahead of its last commit: the five relevant
  interceptor files contain roughly 2,161 inserted and 213 removed lines. The
  implementation port must snapshot and hash the live working files rather than
  copy the older repository `HEAD` versions.

  Source snapshot used for this branch (live working files, not Harness `HEAD`):

  - `agent_harness/interceptor.py`: `5e19e39a919e96f6ca5fd6edfe06bceb7fd530109420f6685a52d21563bf9e27`
  - `agent_harness/websocket_bridge.py`: `22eb15f8aa649ce81e9b7adf1cb6e717e1028d7f6436606ddb3399eb1c901d8d`
  - `interceptor_addon.py`: `4e6508f28e5d41d55274200e5c72af1fa714aa531ed9ad45de10945817937268`
- OMLX's app bundle embeds Python 3.11 through venvstacks. Bundle requirements
  come from `pyproject.toml`, and the app is not sandboxed, so a packaged
  mitmproxy helper can bind loopback and launch Codex without changing app
  entitlements.

## Scope

### Port from the working interceptor

- Exact host/path routing for Responses HTTP and WebSocket traffic.
- Wire-only model catalogue adaptation for the selected local slot: local label,
  real context window, native tools, and non-lite Responses mode.
- Responses request normalization and validation for local chat templates.
- Codex tool schema exposure, namespace flattening/restoration, client-tool
  mapping, and conservative textual tool-call recovery.
- SSE and WebSocket streaming adaptation, usage observation, tool-name repair,
  and local output sanitization.
- Local compaction conversion, readable summaries, and forced-local compaction
  where hosted encrypted compaction would make later local turns unreadable.
- Duplicate-response replay, repeated-failure suppression, doom-loop guard,
  prefix prefill, model-residency keepalive, and connection reuse.
- Owner-only session receipts, bounded event storage, timing metrics, and
  privacy-safe errors.
- Process-scoped CA/proxy environment and fresh Codex desktop launch.

### Exclude from this PR

- Claude CLI and Kimi CLI bridges and their effort controls.
- Pi and OpenCode catalogue discovery or credentials.
- Arbitrary LAN/public inference endpoints.
- Agent Harness MCP tools and Harness-specific desktop attestation prompts.
- Cloud audit capture, GPT Pro aliases, Codex lab-mode overrides, and the
  separately compiled `CodexLocalMenu.swift` helper.
- A system-wide proxy, system keychain trust change, or attachment to an already
  running Codex process.

## Target architecture

```text
OMLX SwiftUI Integrations screen
        |
        | start / stop / restart + sanitized status
        v
OMLX admin API + CodexInterceptorManager (lifecycle owner)
        |
        | bundled mitmproxy module
        v
omlx.codex_interceptor addon
        |-- starts loopback mitmproxy + addon
        |-- warms selected OMLX model
        |-- launches a fresh Codex app with process-only proxy/CA env
        |-- writes owner-only, bounded status events
        |
        +---------------- Codex non-local traffic ----------------> OpenAI/ChatGPT
        |
        `-- selected Responses traffic --> http://127.0.0.1:<omlx>/v1/responses
```

The server-owned manager gives the Swift app and existing admin surface one
lifecycle authority. It starts and stops the proxy and Codex process group,
serves sanitized status through authenticated admin endpoints, and is stopped by
the FastAPI lifespan before inference exits. SwiftUI polls that small metadata
endpoint while Integrations is visible; there is no manual helper script or
second control protocol.

## Package and file layout

Add a focused package rather than folding thousands of protocol lines into the
existing config-file integration:

```text
omlx/codex_interceptor/
  __init__.py
  protocol.py          # pure request/response transforms and validation
  websocket.py         # incremental Responses WebSocket state and SSE decoding
  addon.py             # mitmproxy HTTP/WebSocket hooks
  manager.py           # validation, CA env, child lifecycle, safe aggregation
  privacy.py           # hard-disabled body/header audit compatibility boundary
  mitmdump_runner.py   # bundled mitmproxy module entry point

apps/omlx-mac/Sources/
  Net/DTO/CodexInterceptorDTO.swift
  AppView/ViewModels/IntegrationsScreenVM.swift
  AppView/Screens/IntegrationsScreen.swift

tests/
  test_codex_interceptor_addon.py
  test_codex_interceptor_manager.py
```

Existing files to update:

- `omlx/integrations/codex.py` and `codex_app.py`: stop writing Codex config;
  make `omlx launch codex` and `omlx launch codex_app` use the transparent runner.
- `omlx/admin/routes.py` and `omlx/server.py`: expose authenticated lifecycle
  endpoints and stop managed children during server shutdown.
- `omlx/settings.py`: retain `integrations.codex_model`; the project path stays
  in local app preferences.
- `pyproject.toml` and `packaging`: include a tested mitmproxy dependency in the
  macOS bundle and a `codex-interceptor` optional extra for non-bundled installs.
- `IntegrationsScreen.swift` and `IntegrationsScreenVM.swift`: replace the current
  Codex command row with the native control and monitoring section.
- `Localizable.xcstrings` and Swift tests: localize and verify every new state.
- README/integration docs: make transparent mode the sole recommended Codex path.

## Backend/helper implementation

### 1. Freeze the known-good behavior

- Record SHA-256 hashes for the live harness source files used by the port.
- Copy no Harness configuration, model catalogue, runtime receipts, evidence, or
  credentials into OMLX.
- Port pure transforms first and bring across the relevant existing unit tests
  before changing names or structure.
- Keep a temporary parity test that feeds identical fixtures into the harness and
  OMLX transform functions and compares normalized output.

### 2. Extract the local-only protocol core

- Preserve the current transformations for special Codex input items, developer
  instruction placement, function-call/result adjacency, deferred tools,
  namespace tools, and client tools.
- Preserve validation and repair as separate stages. A request still invalid
  after repair fails before spending a local inference turn.
- Preserve compaction prompt stability so the tool array remains present with
  `tool_choice = "none"`, protecting OMLX KV-prefix reuse.
- Preserve deterministic digests, response replay, repeated-failure limits, and
  calibrated loop/churn guards.
- Remove Harness plugin-skill injection and external-provider selection cleanly;
  do not leave dead branches controlled by hidden environment flags.

### 3. Port the proxy addon

- Bind only `127.0.0.1` on an automatically allocated free port and keep
  `block_global=true`.
- Match only `chatgpt.com` and `api.openai.com` Responses paths. Remove
  `harness.local` from production matching.
- Pass non-inference requests through untouched, but observe only allowlisted
  metadata needed for the live UI.
- Keep the authenticated OpenAI Responses WebSocket open. Consume and answer
  local-slot `response.create` frames through OMLX; pass other models' frames to
  OpenAI unchanged.
- Rewrite local HTTP requests to the running OMLX `/v1/responses` URL, scrub
  sensitive headers, and inject the OMLX API key in memory.
- Support streaming and non-streaming responses, compaction adaptation, native
  tool streaming, conservative buffered recovery, and safe error mapping.

### 4. Build the managed runner

The helper accepts machine-local launch inputs from the native app:

- selected OMLX model ID;
- OMLX loopback port and API key via environment, never command-line arguments;
- project directory;
- runtime directory and optional explicit proxy port for tests.

Startup state machine:

```text
stopped -> validating -> warming -> proxyStarting -> proxyReady
        -> launchingCodex -> running -> stopping -> stopped
                                  `-> failed
```

Startup performs these gates in order:

1. OMLX server health, `/v1/models`, selected model, and `/v1/responses` reachability.
2. Codex/ChatGPT app bundle detection by bundle identifier.
3. Refusal if another Codex app instance is already running; the UI offers a
   separate, explicit **Quit and Relaunch** action.
4. Owner-only runtime directory (`0700`) and files (`0600`).
5. Proxy launch and CA readiness.
6. Fresh Codex app launch with `HTTP_PROXY`, `HTTPS_PROXY`, combined CA variables,
   preserved `NO_PROXY`, and the selected project deep link.
7. Non-blocking OMLX model load, reported live as warm-up state in the UI.

Shutdown handles SIGTERM/SIGINT, terminates the OMLX-launched Codex instance
gracefully, stops the proxy process group, drains final events, verifies the
Codex config hash, and leaves a stopped receipt. OMLX app quit and crash cleanup
must call the same path.

### 5. Package mitmproxy inside OMLX

- Add a compatible pinned mitmproxy Python dependency to the bundle dependency
  set and a non-default `codex-interceptor` extra for pip/Homebrew development.
- Launch it through an OMLX-owned Python entry point instead of relying on a
  Homebrew `mitmdump` executable being on `PATH`.
- Exercise dependency resolution against OMLX's embedded Python 3.11 and existing
  cryptography/network stack.
- Verify all native wheels are included and signed by the existing bundle build.
- Add a bundle smoke test that runs interceptor `doctor` using the staged app's
  embedded interpreter.

## Native UI plan

Replace the current Codex row in **Integrations** with a dedicated **Codex Local
Interceptor** section.

### Controls

- OMLX model picker populated from the server's actual local model list.
- Project folder picker, defaulting to the last successful folder.
- Routing display: **Automatic local slot** before catalogue discovery, then the
  actual wire slot and local label once Codex fetches its catalogue.
- Primary **Start Interceptor** button. Once the proxy is ready it opens a fresh
  Codex app automatically.
- **Stop** and **Restart** actions while active, with the private diagnostics path
  visible for troubleshooting.
- If Codex is already open, replace Start with an explicit **Quit Codex & Start**
  action so the destructive part is never hidden.

### Live status

Show, without exposing content:

- lifecycle status pill and active model/slot route;
- current activity: ready, receiving, prefilling, generating, delivered, or error;
- local request/response count and OpenAI pass-through request/response count;
- first-byte, first-visible, total latency, output tokens/second, cache hit rate,
  and connection reuse;
- model resident and prefix-cached indicators;
- controlled performance warnings and the last safe error;
- a short sanitized event list;
- invariant checks: config unchanged, process-only proxy, local upstream, and
  non-inference pass-through observed.

The model and project controls are disabled while active. Changing either offers
a managed restart rather than mutating a live session underneath Codex.

## Settings and persistence

- Continue using `integrations.codex_model` for the preferred OMLX model.
- Store the last project path and UI-only choices in machine-local OMLX app
  preferences unless the CLI also needs them; do not add server settings only to
  satisfy SwiftUI state restoration.
- Do not persist the OMLX API key, CA material, proxy port, Codex slot, or runtime
  PID in preferences. Those are per-session values.
- Cache only capability results and prefix digests/expiry, never prompt text.

## Verification strategy

### Pure Python and proxy tests

- Port the harness fixtures for request transforms, validation/repair,
  compaction, namespace/client tools, tool-name normalization, replay, loop
  guards, SSE, and WebSocket state.
- Assert exact-match interception and pass-through for every known host/path and
  near-miss path.
- Assert sensitive headers never reach the local upstream.
- Assert prompt/body/header values cannot appear in status JSON, logs, or errors.
- Assert event files rotate/truncate safely and retain owner-only permissions.
- Assert startup failure leaves no proxy/helper process and shutdown reaps the
  whole process group.

### OMLX local-model smoke gate

Run against a real local OMLX model and require:

1. a tool-requiring turn emits a native function call;
2. function call -> result -> answer completes without template rejection;
3. compaction returns a readable handoff summary;
4. a follow-up turn reports cached prompt tokens;
5. HTTP and WebSocket streaming both produce visible output and usage;
6. cancellation/retry replays a completed response rather than rerunning it.

Any change to prompt shape, item order, roles, instructions, or tools must pass
this gate; unit tests alone are insufficient.

### Swift tests

- Decode every helper state and event fixture.
- Verify lifecycle transitions, start-button enablement, running-state control
  lockout, warnings, counters, and error recovery.
- Verify an existing Codex process produces the explicit relaunch sheet.
- Verify OMLX termination reaps the helper and a helper crash becomes a visible
  failed state rather than silently restarting Codex.
- Extend localization smoke tests for every added string.

### Desktop acceptance

From a staged OMLX app bundle:

1. Hash `~/.codex/config.toml`.
2. Start OMLX, select a local model, choose a project, and click Start.
3. Confirm the relabelled local slot appears and returns a local response.
4. Select another Codex model and confirm it uses OpenAI.
5. Confirm existing Projects, Automations, plugins/MCP, and account state remain
   available.
6. Exercise a Codex tool call and Computer Use/browser capability.
7. Watch local/remote counters and latency update in OMLX.
8. Stop from OMLX and confirm Codex/proxy children exit cleanly.
9. Re-hash the Codex config and confirm it is identical.
10. Inspect diagnostics and confirm no prompt, response, header, cookie, or key was
    recorded.

## Failure behavior to design explicitly

- OMLX server stopped or unavailable.
- Selected model removed, renamed, unloaded, or rejected for context size.
- Codex/ChatGPT app missing or already running.
- Proxy port conflict or mitmproxy startup failure.
- CA generation/read failure.
- OMLX restart during a Codex turn.
- Proxy/helper crash while Codex is open.
- WebSocket protocol change in a future Codex build.
- Local model emits malformed/ambiguous tools or an empty response.
- User stops OMLX or the interceptor during generation.

Every failure must produce a safe, actionable UI message and leave no orphaned
proxy. Automatic retry is appropriate for temporary OMLX reachability, but not
for malformed requests, deterministic local-model failures, or a changed Codex
transport.

## Commit and PR sequence

1. **Protocol port + parity tests** — pure transforms, WebSocket state, privacy
   schemas, and source snapshot record.
2. **Proxy + managed runtime** — addon, helper lifecycle, warm-up, receipts,
   transparent CLI launches, and removal of config mutation from Codex paths.
3. **Packaging** — Python-3.11-compatible bundled mitmproxy dependency,
   staged-app doctor, and signing verification.
4. **Native lifecycle + UI** — admin lifecycle, Integrations controls, live
   status, localization, and Swift tests.
5. **End-to-end verification + docs** — real OMLX smoke gate, staged desktop
   acceptance, screenshots, and user documentation.

Keep the PR reviewable by preserving these commit boundaries. Do not mix the
distributed-cluster branch into this work.

## Definition of done

- One click in OMLX starts the helper, proxy, warm-up, and a fresh Codex app.
- The chosen local slot completes real Codex tool-using work through OMLX.
- Other models and all non-inference Codex features continue to use OpenAI.
- OMLX shows trustworthy live lifecycle, routing, timing, cache, and error state.
- Stop/restart/app quit leave no proxy or helper processes behind.
- The staged app works without a separate Python script or Homebrew mitmproxy.
- Automated and desktop acceptance prove `~/.codex/config.toml` is unchanged and
  diagnostics contain no sensitive content.
