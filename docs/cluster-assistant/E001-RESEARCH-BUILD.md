# E001 research build — a second app beside the cluster app

**Not part of the product line.** This is a lab artefact for the parallel
research project (LOCAL-MoE 27, E001 gated sidecar). It is built from a pinned
revision plus a third-party patch, installed next to the normal oMLX app on
`.11`, and nothing in this repository's product code depends on it.

The hook is **inert by default**: with no sidecar attached, `layer.sidecar` is
`None` and behaviour, memory and throughput are unchanged.

## What was built

| Item | Value |
|---|---|
| Base revision | `94b96ca6abfa1efa50f95d77ec51b79c40378566` (`rebase/upstream-main`) |
| Patch | `e001-p2-sidecar.patch`, 4 hunks in `qwen4_exp/language.py` |
| Worktree | `~/omlx-e001-build` (a git worktree at the pinned revision, not a branch of this tree) |
| Version | **`0.8.0rc2`** (`omlx/_version.py`, so app and Python package agree) |
| Applied result | 36 insertions, 6 deletions in one file |
| Patched file sha256 | `740b1beb48e973cf76e2e2ab2d420c11ef8f8ee93dc743737f319506afb9b589` |
| Base file sha256 | `f2d40e456e25a7140691cd1b340c8a9bf92b3a0cc6d317e78d1a7e5e090a340a` |
| DMG | `oMLX-E001-0.8.0rc2.dmg`, SHA256 `55f7be312eb539452015f7f03752c5d936307d8f38594e99a71888ad431aa4a9` |
| App binary | `b1eea987a7687349566b4696b70a02dcca0b8b865ab20fac24b29d827b306ce8` |
| Notarization | Accepted, stapled; `spctl` → `accepted, source=Notarized Developer ID` |
| Bundled `language.py` | `740b1beb…` — byte-identical to the patched file above, verified inside the DMG |
| Installed on | `.11` and `.14`, as `/Applications/oMLX-E001.app` next to the normal app (not started) |

## How it differs from the product app

Everything here is a worktree-only edit; this repository's own sources are
unchanged.

| | Product app | E001 research app |
|---|---|---|
| File in `/Applications` | `oMLX.app` | **`oMLX-E001.app`** |
| Display name | oMLX | **oMLX E001** |
| Bundle id | `app.omlx` | **`app.omlx.e001`** |
| Version | `0.8.0rc1` | **`0.8.0rc2`** |
| State | `~/.omlx` | **`~/.omlx-e001`** |
| Default port | 8000 | **8001** |
| DMG | `oMLX-0.8.0rc1.dmg` | **`oMLX-E001-<version>.dmg`**, volume „oMLX E001" |

A different bundle id is what lets both apps exist at once: LaunchServices,
Dock, the privileged-helper binding and the update path all key on it. The
separate base path means the research build can never rewrite the cluster app's
settings, pairing or operation state — and both can run side by side.

### Cleanup must never touch it

`scripts/cleanup-cluster-macs.sh` carries an explicit guard
(`PROTECTED_APP=oMLX-E001.app`, `PROTECTED_STATE=~/.omlx-e001`): it skips the app
in its removal loop (the `/Applications/oMLX-*.app` glob would otherwise match
it), keeps its LaunchServices record, leaves `~/.omlx-e001` alone and does not
delete a trashed copy. Its closing report prints `E001 kept: app, state kept` so
a wiped Mac is not mistaken for a bare one. The same rule applies to
`scripts/build-release-dmg.sh`, which only unregisters paths inside this
repository.

`.14` carries the E001 app at the owner's request even though `.14` is otherwise
out of scope: it is **not** part of the validation cluster, is never cleaned,
never paired, never measured, and its own `/Applications/oMLX.app` stays
untouched.

## Verification (in the patched worktree)

```text
tests/test_qwen4_hc_fused.py                396 passed
tests -k "qwen4 or qwen35"                  2033 passed, 350 skipped, 0 failed
tests/test_e001_sidecar.py                  4 passed
```

`tests/test_e001_sidecar.py` is the §6 set from the integration request, pinned
against the real `Qwen4ExpDecoderLayer`/`Qwen4ExpModel` (no checkpoint needed):

1. `sidecar` defaults to `None` and two runs are bit-identical,
2. a neutral sidecar (`active=False`) keeps the deferred write **and** the output
   is bit-identical (same number of eager `_hc_write` calls),
3. an active sidecar completes the write instead of dropping it — and equals the
   run with deferral forced off, bit for bit,
4. an active sidecar really disables deferral (more eager writes than the
   baseline), so the term is never silently dropped.

## Two things the requester should know

1. **The patch's own comment contradicts its contract.** Hunk 1 says the object
   "must expose `.enabled` and `.streams(x)`", while the code reads `sidecar.active`.
   A sidecar implementing only `.enabled` would raise `AttributeError` on the
   first forward.
2. **§2 of the request says hunk 4 "passes the fallback flag"; it does not.** The
   patch disables deferral at the `Qwen4ExpModel` level while any sidecar is
   active, so `sidecar_immediate_fallback` is never passed by the model. The
   `RuntimeError` guard therefore only fires for a direct layer call with
   `defer_write=True` (which is what test 4 exercises, at the layer boundary).

## Deferred writes and quantisation (answers to §9)

- `defer_write = hc_fused.write_enabled()` in both `Qwen4ExpModel.__call__`
  entries, and `write_enabled()` is `not (_DISABLED or _WRITE_DISABLED)` — write
  deferral is **on by default**. So with an active sidecar the model turns it off
  for the whole stack: every layer completes its own write. Numerically identical
  to the deferred form, but the fused-deferred path is bypassed for those layers.
- The sidecar is attached at runtime, after the base load, so it is in no
  quantisation plan and stays in the dtype the owner builds it in (BF16).

## Rebuilding it

```sh
git worktree add ~/omlx-e001-build 94b96ca6
cd ~/omlx-e001-build
git apply /path/to/e001-p2-sidecar.patch          # expect 1 file, +36 −6
# variant edits (applied in the worktree, not in this tree):
#   omlx/_version.py                     0.8.0rc1 -> 0.8.0rc2
#   Sources/Config/AppConfig.swift       .omlx -> .omlx-e001, port 8000 -> 8001
#   Sources/Server/ServerProcess.swift   .omlx -> .omlx-e001
#   oMLX.xcodeproj                       PRODUCT_BUNDLE_IDENTIFIER app.omlx.e001
#   Resources/Info.plist                 CFBundleDisplayName "oMLX E001"
#   Scripts/package_dmg.sh               DMG/volname "oMLX-E001-…"/"oMLX E001"
scripts/build-release-dmg.sh                       # build + sign + notarize
```

Install (never start it programmatically — that stays a person's click):

```sh
hdiutil attach -nobrowse -readonly build/oMLX-E001-0.8.0rc2.dmg
ditto "/Volumes/oMLX E001/oMLX.app" /Applications/oMLX-E001.app
hdiutil detach "/Volumes/oMLX E001"
```

`xattr -dr com.apple.quarantine /Applications/oMLX-E001.app` is unnecessary for a
locally built DMG but harmless.