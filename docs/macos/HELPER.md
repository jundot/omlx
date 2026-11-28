# Privileged helper (SMJobBless)

The macOS app ships a small privileged helper that sets
`iogpu.wired_limit_mb`. It is installed with **one standard macOS administrator
dialog**, not a Terminal `sudo` step, through `SMJobBless`.

This document describes the implementation and the packaging invariants. It
contains no signing identity: the team id is supplied at build time (see
[Signing and notarization](SIGNING.md)).

## Scope

- One prompt: the standard macOS admin window, raised by `SMJobBless`.
- The helper runs exactly one command,
  `sysctl iogpu.wired_limit_mb=<validated integer>`, and nothing else.
- Distribution is a direct signed/notarized DMG, not the App Store (see
  [Distribution](DISTRIBUTION.md)).
- Privileged **network** setup (Thunderbolt bridge/STP) is **not** implemented.

## One-shot fallback: `Apply with Admin…`

The Performance screen shows the existing `iogpu.wired_limit_mb` warning and
offers **Apply with Admin…**:

- `PerformanceScreenVM.wiredLimitMB` reads the limit the server asks for.
- `PrivilegedActions.setWiredLimit(mb:)` runs exactly
  `sysctl iogpu.wired_limit_mb=<validated integer>` through `NSAppleScript`'s
  `do shell script … with administrator privileges`.
- The integer is validated (`1…1_048_576`); no UI text reaches the shell.
- The standard macOS admin window is the only prompt; after it succeeds the
  settings are reloaded and the user is told to restart oMLX if it is already
  serving.
- `PerformanceScreenVMTests.testPrivilegedWiredLimitValidation` pins the
  validation without opening a dialog.

The one-shot action is the fallback path. It is also what runs when the helper
cannot be installed (no signed app, or a cancelled install that is then retried
explicitly); a cancelled admin dialog stops there instead of opening a second
prompt.

## The helper as a capability

Installing the helper is not hidden behind the low-limit warning. One seam is
the single source of truth:

```
PrivilegedHelper.status → { installed, version,
                            wiredLimitCurrentMB, wiredLimitRequestedMB }
```

- **Welcome** has a *System helper* step: one admin prompt while the person is
  already clicking through first run.
- **Performance** has a permanent *System Helper* row with **Install /
  Reinstall**, independent of any warning, and a separate wired-limit row with
  **Apply** whenever the current limit differs from the requested one. A raised
  limit therefore never hides the capability.
- `ensureInstalled` is idempotent and versioned: it blesses, reuses, or replaces
  the tool (see the version ledger below), and the five status states are
  covered by `PerformanceScreenVMTests` without opening a prompt.

## Uninstall

The app uninstalls itself from **Settings → Advanced → Uninstall…**: one sheet
with a keep switch per scope (app, helper, runtime, settings, cluster state) and
a byte summary before anything is removed. Models and the cluster identity are
kept by default; model deletion is opt-in per directory. Removing the helper is
the one admin dialog; cancelling it leaves the rest done and is reported.
`Uninstall.swift` separates locator, plan and runner so the tests drive temporary
directories and delete nothing real.

## Persistent helper

`apps/omlx-mac/Helper/` contains the helper sources, its embedded Info.plist and
two launchd plists. `build.sh release --slim` compiles it with `swiftc` and
embeds both plists into the binary; `package_dmg.sh` signs it before the app.

| File | Role |
|---|---|
| `Helper/main.swift` | XPC listener; one command: validated `sysctl iogpu.wired_limit_mb` |
| `Helper/Info.plist` | embedded `__TEXT,__info_plist`, carries `SMAuthorizedClients` |
| `Helper/app.omlx.helper.bless.plist` | embedded `__TEXT,__launchd_plist`: `Label`, `MachServices`, no `ProgramArguments` |
| `Helper/app.omlx.helper.plist` | `Contents/Library/LaunchDaemons`, `BundleProgram` for the later `SMAppService` path |

### What `ServiceManagement.h` requires

`ServiceManagement.h` is explicit, and all three points are easy to break:

1. **The launchd plist must be embedded.** SMJobBless reads it from
   `__TEXT,__launchd_plist`, not from `Contents/Library/LaunchDaemons`. The
   build emits both sections (`-sectcreate __TEXT __launchd_plist`); the on-disk
   copy is kept for the future `SMAppService.daemon(plistName:)` path.
2. **The embedded launchd plist carries no `ProgramArguments`.** SMJobBless
   fills those in itself and documents that the tool takes no custom command
   line. The embedded plist declares `Label`, `MachServices` (without it launchd
   never creates the bootstrap name the helper's `NSXPCListener` checks in with)
   and `RunAtLoad=false`, and nothing else. The bundle plist uses
   `BundleProgram` (`Contents/Library/LaunchServices/app.omlx.helper`) instead
   of a `/Library/PrivilegedHelperTools/…` path that only SMJobBless creates.
3. **`CFBundleInfoDictionaryVersion` must be exactly `6.0`**, and
   `CFBundleIdentifier` must equal the tool's file name.

Two behavioural details:

- **A changed helper is replaced.** SMJobBless answers “already installed,
  success” and never swaps the tool, so `PrivilegedHelper` records the version
  it blessed in `UserDefaults` (`omlx.helper.installedVersion`) and runs
  `SMJobRemove` + `SMJobBless` when that differs from
  `PrivilegedHelper.helperBundleVersion` (== `Helper/Info.plist`
  `CFBundleVersion`). An install that predates the ledger counts as unknown and
  is replaced.
- **A cancelled prompt no longer opens a second prompt.** Only an installation
  or XPC failure falls back to the one-shot AppleScript path.

### Trust requirements, generated from the team id

Both sides pin a full code requirement, and they mirror each other:

- `SMPrivilegedExecutables` in the app names the helper
  (`identifier "app.omlx.helper"`);
- `SMAuthorizedClients` in the helper names the app (`identifier "app.omlx"`).

A team-only requirement (`certificate leaf[subject.OU] = "<team>"`) is **not**
enough: any certificate in that team satisfies it, including a plain
**Apple Development** certificate. Both requirements therefore also pin the
Developer ID Certification Authority and the Developer ID Application leaf
marker:

```text
anchor apple generic and identifier "app.omlx.helper" and certificate 1[field.1.2.840.113635.100.6.2.6] and certificate leaf[field.1.2.840.113635.100.6.1.13] and certificate leaf[subject.OU] = "$(OMLX_TEAM_ID)"
```

`$(OMLX_TEAM_ID)` is the **only** variable part. The repository carries no Apple
team id, so no contributor has to edit a checked-in file:

- The app's `Resources/Info.plist` is processed by Xcode, which substitutes the
  build setting in place.
- The helper plists are embedded with `-sectcreate`, so Xcode never sees them:
  `build.sh` renders the same `$(OMLX_TEAM_ID)` marker into a build copy
  (`build/helper-plists/`) and embeds that.
- `DEVELOPMENT_TEAM` in the Xcode project is `$(OMLX_TEAM_ID)` as well, so the
  same variable drives Xcode's signing.
- Unset means empty. The requirement then names no team and matches no
  certificate, which is exactly why an **unsigned** build cannot bless the
  helper: it builds, it just cannot install the daemon. `build.sh` warns about
  it, and `package_dmg.sh` refuses to sign a stage whose requirement does not
  name the identity's team.

`apps/omlx-mac/Tests/python/test_privileged_helper_packaging.py` pins this: it
validates the rendered requirement with `csreq` when `OMLX_TEAM_ID` is set (and
skips when it is not), and it checks the Developer ID markers so a plain Apple
Development certificate would not satisfy the designator.

### Verifying the trust pair offline

Both directions of the trust pair can be verified without a bless, which is what
SMJobBless evaluates:

```sh
codesign --verify -R='<SMAuthorizedClients entry>'  <app>
codesign --verify -R='<SMPrivilegedExecutables entry>' <helper>
```

Use the `=` form: `codesign -R <text>` treats the argument as a file path. The
same check works on the embedded plists:

```sh
otool -X -s __TEXT __info_plist <helper>    # dump the embedded Info.plist
strings -a <helper> | grep 'subject.OU'     # the rendered team
```

### Not verifiable without a signed install

The admin-prompt install itself (`SMJobBless` →
`/Library/PrivilegedHelperTools/app.omlx.helper` →
`launchctl print system/app.omlx.helper` → one XPC call) needs a click on a
signed, notarized app in **Performance → Apply with Admin…**. No offline test
simulates that step.

## Tests and build

- `swiftc -parse` passes for the helper sources.
- `xcodebuild build -scheme oMLX CODE_SIGNING_ALLOWED=NO` succeeds with no team
  configured.
- `xcodebuild test -scheme oMLX CODE_SIGNING_ALLOWED=NO` runs the Swift suite;
  `PerformanceScreenVMTests` covers wired-limit validation, helper install state
  (bless/reuse/replace), “a cancelled prompt never falls back”, and the helper
  identity.
- Offline packaging tests: `uv run pytest apps/omlx-mac/Tests/python` (plist
  consistency, both embedded sections, Developer ID pinning, one fixed sysctl
  command, no shell in the helper, signing order). The team-dependent assertions
  run with `OMLX_TEAM_ID=<team>` and skip without it.
- On a Mac that has only a Developer ID Application certificate and no **Mac
  Development** certificate, Xcode's automatic signing cannot build the test
  bundle: run tests with `CODE_SIGNING_ALLOWED=NO`. Release packaging signs
  explicitly with the Developer ID identity and is unaffected.

## Open

| Item | Status | Next step |
|---|---|---|
| Helper sources, plists, both embedded `__TEXT` sections | Done | keep in sync via `test_privileged_helper_packaging.py` |
| Developer ID trust pinning on both sides | Done, verified offline with `codesign --verify -R=…` | re-verify after a certificate change |
| Helper upgrade path (version ledger + `SMJobRemove`) | Done, unit-tested | confirm on the first real install |
| One-dialog `SMJobBless` install end to end | Gated: needs the notarized DMG and one admin click | Performance → Apply with Admin…; then `launchctl print system/app.omlx.helper` |
| XPC client authentication | Gated: needs a working install | decide `SMJobBless` vs `SMAppService` |
| Privileged network apply (bridge/STP) | Not implemented; product decision (interface selection, line/STP policy) | record the policy, then extend the helper surface |

No key, certificate or notarization credential is committed.