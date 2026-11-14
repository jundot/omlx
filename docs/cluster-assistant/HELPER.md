# Privileged helper — first slice and open points

Status: **helper implemented and packaging repaired (2026-11-14)**. The
persistent helper builds, is Developer ID signed and its trust requirements are
verified offline; the **one-prompt SMJobBless install is still not exercised**
because it needs a click in the signed app. Privileged network setup is not
implemented.

## Recorded decisions

- Team: **Andreas Kolb — `YB894QRG4W`** (Developer ID).
- One **standard macOS administrator dialog**, not a Terminal `sudo` step.
- The helper scope includes the wired limit and network setup.
- Distribution is a direct signed/notarized **DMG**, not the App Store.

## Implemented: wired limit with one admin dialog

The Performance screen shows the existing `iogpu.wired_limit_mb` warning and
now has **Apply with Admin…**:

- `PerformanceScreenVM.wiredLimitMB` reads the server's requested limit.
- `PrivilegedActions.setWiredLimit(mb:)` runs exactly
  `sysctl iogpu.wired_limit_mb=<validated integer>` through
  `NSAppleScript`'s `do shell script … with administrator privileges`.
- The integer is validated (`1…1_048_576`); no UI text reaches the shell.
- The standard macOS admin window is the only prompt; after it succeeds the
  settings are reloaded. The user is told to restart oMLX if it is already
  serving.
- `PerformanceScreenVMTests.testPrivilegedWiredLimitValidation` pins the
  validation without opening a dialog.

This is a **one-shot privileged action**, not a launchd daemon. It is the
smallest slice that gives the requested one-dialog behaviour without a separate
signed target.

## Persistent SMJobBless helper — implemented 2026-11-08, packaging repaired 2026-11-14

`apps/omlx-mac/Helper/` contains the helper sources, its embedded Info.plist and
two launchd plists. `build.sh release --slim` compiles it with `swiftc` and
embeds both plists into the binary; `package_dmg.sh` signs it before the app.

| File | Role |
|---|---|
| `Helper/main.swift` | XPC listener; one command: validated `sysctl iogpu.wired_limit_mb` |
| `Helper/Info.plist` | embedded `__TEXT,__info_plist`, carries `SMAuthorizedClients` |
| `Helper/app.omlx.helper.bless.plist` | embedded `__TEXT,__launchd_plist`: `Label`, `MachServices`, no `ProgramArguments` |
| `Helper/app.omlx.helper.plist` | `Contents/Library/LaunchDaemons`, `BundleProgram` for the later `SMAppService` path |

### What the 2026-11-14 audit found and fixed

`ServiceManagement.h` is explicit, and the previous slice broke three of its
requirements. With the old layout **SMJobBless could not have installed the
helper at all**; the AppleScript fallback was the only path that ever worked.

1. **The launchd plist was not embedded.** SMJobBless reads it from
   `__TEXT,__launchd_plist`, not from `Contents/Library/LaunchDaemons`. The
   build now emits both sections (`-sectcreate __TEXT __launchd_plist`). The
   on-disk copy is kept for the future `SMAppService.daemon(plistName:)` path.
2. **The embedded launchd plist carried `ProgramArguments`.** SMJobBless fills
   those in itself and documents that the tool takes no custom command line; the
   embedded plist now declares `Label`, `MachServices` (without it launchd never
   creates the bootstrap name the helper's `NSXPCListener` checks in with) and
   `RunAtLoad=false`, and nothing else. The bundle plist uses `BundleProgram`
   (`Contents/Library/LaunchServices/app.omlx.helper`) instead of a
   `/Library/PrivilegedHelperTools/…` path that only SMJobBless creates.
3. **`CFBundleInfoDictionaryVersion` was missing** from the helper's
   Info.plist; it must be exactly `6.0`.

Two behavioural fixes came with it:

- **A changed helper is now replaced.** SMJobBless answers “already installed,
  success” and never swaps the tool, so `PrivilegedHelper` records the version
  it blessed in `UserDefaults` (`omlx.helper.installedVersion`) and runs
  `SMJobRemove` + `SMJobBless` when that differs from
  `PrivilegedHelper.helperBundleVersion` (== `Helper/Info.plist`
  `CFBundleVersion`). An install that predates the ledger counts as unknown and
  is replaced.
- **A cancelled prompt no longer opens a second prompt.** Only an installation
  or XPC failure falls back to the one-shot `NSAppleScript`; a cancelled admin
  dialog or a refused value stops there.

The trust requirements were tightened as well. They previously pinned only the
team (`certificate leaf[subject.OU] = "YB894QRG4W"`), which any certificate in
that team satisfies — including an Apple Development certificate. Both sides now
also pin the Developer ID Certification Authority and the Developer ID
Application leaf marker:

```text
anchor apple generic and identifier "app.omlx.helper" and certificate 1[field.1.2.840.113635.100.6.2.6] and certificate leaf[field.1.2.840.113635.100.6.1.13] and certificate leaf[subject.OU] = "YB894QRG4W"
```

`SMPrivilegedExecutables` in the app names the helper; `SMAuthorizedClients` in
the helper names the app (`identifier "app.omlx"`).

### Verified offline on .11 (2026-11-14)

- `swiftc` embeds `__launchd_plist` (1292 bytes = the source plist, which parses
  back to `Label` + `MachServices` + `RunAtLoad=false`) and `__info_plist`
  (1328 bytes = the source plist).
- `build.sh release --slim` stages the helper and both plists.
- `package_dmg.sh` with the Developer ID identity signs the helper, then the
  app, then the image; `codesign --verify --deep --strict` passes.
- Both directions of the trust pair pass, which is what SMJobBless evaluates:

  ```sh
  codesign --verify -R='<SMAuthorizedClients entry>'  <app>
  codesign --verify -R='<SMPrivilegedExecutables entry>' <helper>
  ```

  Use the `=` form: `codesign -R <text>` treats the argument as a file path.
- `codesign -dv` confirms `Authority=Developer ID Application: Andreas Kolb
  (YB894QRG4W)` on app and helper.
- The embedded launchd plist reads back as `Label` + `MachServices` +
  `RunAtLoad=false`.

### Not verified — needs the signed install

The admin-prompt install itself (`SMJobBless` → `/Library/PrivilegedHelperTools/app.omlx.helper`
→ `launchctl print system/app.omlx.helper` → one XPC call) still needs a click
on the notarized DMG in **Performance → Apply with Admin…**. Nothing in this
audit simulated or claimed that step.

## Read-only network preflight — implemented 2026-11-08

The chosen scope starts with a read-only preflight. `GET
/admin/api/cluster/assistant/diagnostics` is the support bundle; the separate
`GET /admin/api/cluster/assistant/network-preflight` reads the local
interfaces through the existing `transport.probe_host_interfaces` and returns:

- Thunderbolt/RDMA interface names, IPv4 addresses, self-assigned 169.254
  interfaces, link-local addresses and media speeds;
- exact `ifconfig bridge…` command lines for the detected interfaces;
- warnings for no Thunderbolt interface and for a ring/STP loop;
- `executed: false` and `confirmation_required: true`.

The card shows the commands with a **read-only** label and a Copy button. It
never runs them, never opens an admin dialog and never touches a peer. The
privileged apply is the separate helper step and is not implemented.

## Open: privileged network apply

Automatically creating a Thunderbolt bridge can create a layer-2 loop; the
validation cluster runs as a **line** and macOS does not run STP on its bridges
by default (see `PHASE-2B-RUN.md` §10). The preflight deliberately does not
execute. A later helper step needs:

- the user's explicit interface/topology confirmation;
- a safe way to add members to an existing bridge without disrupting the
  current connection;
- an STP policy or a line-only restriction.

The interface selection and STP policy remain open product decisions.

## Tests and build (verified 2026-11-14 on .11)

- `swiftc -parse` passes for the changed sources.
- `xcodebuild build -scheme oMLX CODE_SIGNING_ALLOWED=NO` succeeds.
- `xcodebuild test -scheme oMLX CODE_SIGNING_ALLOWED=NO`: **391 passed, 3
  failed, 5 skipped**. The three failures are outside this surface and their
  files are untouched: `ModelSettingsScreenVMTests` (two percent-formatting
  assertions render `42 %` with a narrow no-break space on this host) and
  `ThemeTests` (one light-appearance assertion depends on the host appearance).
- Focused: `PerformanceScreenVMTests` (wired-limit validation, helper install
  state bless/reuse/replace, “a cancelled prompt never falls back”, helper
  identity) and the new `RuntimeInstallerTests`.
- Offline packaging tests: `uv run pytest apps/omlx-mac/Tests/python` — 27
  passed, including `test_privileged_helper_packaging.py` (plist consistency,
  both embedded sections in the build, Developer ID pinning, one fixed sysctl
  command, no shell in the helper, signing order).
- **`xcodebuild test` needs `CODE_SIGNING_ALLOWED=NO` on this Mac**: the project
  signs automatically for team `YB894QRG4W` and asks for a **Mac Development**
  certificate, while only the Developer ID Application certificate exists.
  That is unrelated to the release build, which signs explicitly with the
  Developer ID identity in `package_dmg.sh`.

## P2 status — privileged helper

| Item | Doable now — done | Signing-/admin-gated | Next step |
|---|---|---|---|
| Helper sources, plists, both embedded `__TEXT` sections | Done 2026-11-14 (`build.sh`) | – | keep in sync via `test_privileged_helper_packaging.py` |
| Developer ID trust pinning on both sides | Done, verified offline with `codesign --verify -R=…` | – | re-verify if the certificate changes |
| Helper upgrade path (version ledger + `SMJobRemove`) | Done, unit-tested | – | confirm on the first real install |
| One-dialog `SMJobBless` install end to end | – | Needs the notarized DMG and one admin click on .11 | Performance → Apply with Admin…; then `launchctl print system/app.omlx.helper` |
| XPC client authentication | – | Needs the install to verify, and a decision on the private audit-token API vs `SMAppService` | decide `SMJobBless` vs `SMAppService` first |
| Privileged network apply (bridge/STP) | – | Product decision (interface selection, STP/line policy) | record the policy, then extend the helper surface |

## Distribution

See [`DISTRIBUTION.md`](DISTRIBUTION.md). No key, certificate or notarization
credential is committed.
