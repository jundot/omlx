# Distribution — direct DMG, outside the App Store

Status: **signed and notarized (2026-11-08, re-notarized 2026-11-13);
packaging audited 2026-11-14**. The 2026-11-13 slim DMG is ~76 MB (72.8 MiB)
and the app inside it is accepted by Gatekeeper as `Notarized Developer ID`. The
rebuild from 2026-11-14 (helper packaging fix) is signed but **not** notarized
yet — see §Remaining release prerequisites.

## Decision

- Distribution is a **direct download DMG**, not the App Store.
- Team: **Andreas Kolb — `YB894QRG4W`**.
- The app and the helper are signed with a Developer ID Application
  certificate, hardened runtime, and the DMG is notarized with `notarytool`.
- Updates use the existing GitHub Releases poller (`ReleasesChecker` +
  `AppUpdater`); the DMG is the release asset. No appcast or EdDSA key is
  introduced.

## Build and package

The app is **slim**: Python and MLX are not embedded. The staged app contains
the Swift shell, the `omlx`/`omlx_web` source, the first-run runtime installer
(`uv` + lock) and the signed SMJobBless helper. The runtime is provisioned
once from the Welcome card (or `provision_runtime.sh`); see
[`SLIM-RUNTIME.md`](SLIM-RUNTIME.md).
`apps/omlx-mac/Scripts/package_dmg.sh` signs the helper first, then the app, then
the `.dmg` image itself; it notarizes, staples and assesses the image only when
`SIGN_IDENTITY` **and** `NOTARY_PROFILE` are supplied.

Signing the image is not cosmetic. Until 2026-11-14 the script left the
container unsigned, so `spctl --type open` reported `no usable signature` even
for an image that had been notarized and stapled — the ticket was there, the
container was not assessable. With the image signed, a signed-but-unnotarized
build reports `Unnotarized Developer ID` (measured); a notarized one is then
expected to report `Notarized Developer ID` (that verdict was measured for the
app inside the 2026-11-13 image, not yet for a signed image).

1. Provision the external runtime and build the slim app:

   ```sh
   apps/omlx-mac/Scripts/provision_runtime.sh
   OMLX_NEXT_OUT="$PWD/apps/omlx-mac/build/SlimStage" \
     apps/omlx-mac/Scripts/build.sh release --slim
   ```

2. Sign the staged app with the Developer ID identity, hardened runtime and the
   app entitlements. `package_dmg.sh` does this (helper first, then the app), but
   the equivalent manual commands are:

   ```sh
   codesign --force --options runtime --timestamp \
     --identifier app.omlx.helper \
     --sign "Developer ID Application: Andreas Kolb (YB894QRG4W)" \
     apps/omlx-mac/build/SlimStage/oMLX.app/Contents/Library/LaunchServices/app.omlx.helper
   codesign --force --deep --options runtime \
     --entitlements apps/omlx-mac/Resources/oMLX.entitlements \
     --sign "Developer ID Application: Andreas Kolb (YB894QRG4W)" \
     apps/omlx-mac/build/SlimStage/oMLX.app
   codesign --verify --deep --strict --verbose=2 apps/omlx-mac/build/SlimStage/oMLX.app
   ```

3. Create, sign and (when the credentials are set) notarize the DMG. The app
   path must be the **slim stage** (`SlimStage`), not `Stage`: only the slim
   build contains the helper and the runtime installer.

   ```sh
   APP_PATH="$PWD/apps/omlx-mac/build/SlimStage/oMLX.app" \
   OUT_DIR="$PWD/build/slim" \
   SIGN_IDENTITY="Developer ID Application: Andreas Kolb (YB894QRG4W)" \
   NOTARY_PROFILE=omlx-notary \
   apps/omlx-mac/Scripts/package_dmg.sh
   ```

   Without `NOTARY_PROFILE` it produces a Developer ID-signed app in an
   **unnotarized** DMG for local inspection and says so. That is what
   `build/slim-audit/` holds after the 2026-11-14 audit.

4. The script already runs these; keep them as the manual fallback:

   ```sh
   codesign --force --timestamp \
     --sign "Developer ID Application: Andreas Kolb (YB894QRG4W)" \
     build/oMLX-<version>.dmg
   xcrun notarytool submit build/oMLX-<version>.dmg \
     --keychain-profile omlx-notary --wait
   xcrun stapler staple build/oMLX-<version>.dmg
   spctl --assess --type open --context context:primary-signature \
     --verbose=2 build/oMLX-<version>.dmg
   ```

5. Upload the DMG as a GitHub release asset; the app's updater uses that asset.

The repository contains only the commands and no certificate, key, notary
profile or credential.

## Current artifacts

Recorded 2026-11-08 (unverified here beyond the notary history):

```text
build/slim-helper2/oMLX-0.7.4.dev1.dmg   72 MB
SHA-256 b44aaa686d36bfe7ab27cc84375170c64b0e4cd94f8dafd45bc4f35bbdeb03f6
```

That directory no longer exists; the images on .11 are:

| Path | Bytes | State |
|---|---|---|
| `build/slim/oMLX-0.7.4.dev1.dmg` | 76,382,657 (72.8 MiB) | 2026-11-13, notarized + stapled |
| `build/slim-helper6/oMLX-0.7.4.dev1.dmg` | 76,106,509 | 2026-11-13, earlier helper iteration |
| `build/slim-audit/oMLX-0.7.4.dev1.dmg` | 77,289,103 | 2026-11-14 audit: helper+app+image Developer ID signed, **not notarized**; SHA-256 `31fb73e8…cfb3a87` |

The “72 MB” figures in older notes are MiB; the decimal size is ~76 MB.

- Signed with Developer ID Application (team `YB894QRG4W`), hardened runtime,
  timestamp on app and helper; the `.dmg` is signed too.
- The 2026-11-13 images have a stapled ticket (`xcrun stapler validate` passes).
  The 2026-11-14 image has none.
- Contains the external-runtime installer (`uv` + lock) and the SMJobBless
  helper — after 2026-11-14 the helper carries the embedded
  `__TEXT,__launchd_plist` section SMJobBless requires.
- `codesign --verify --deep --strict` passes on the app inside `build/slim`
  (mounted) and on the staged app; `hdiutil verify` reports a valid checksum for
  the 2026-11-14 image. The two 2026-11-13 images came from the same script.

### What Gatekeeper reports (measured 2026-11-14)

| Check | Result |
|---|---|
| `spctl -a -t open --context context:primary-signature` on the **2026-11-13 DMG** | `rejected`, `source=no usable signature` — the container was never signed |
| `spctl -a -t execute` on the **app inside** that image (mounted) | `accepted`, `source=Notarized Developer ID` |
| `spctl -a -t open` on the **2026-11-14 audit DMG** (now signed, unnotarized) | `rejected`, `source=Unnotarized Developer ID` — correct and expected |

So the earlier “`spctl` accepted (`source=Notarized Developer ID`)” claim was
true for the app inside the image, not for the image. After the script fix both
lines line up; a notarized image is expected to be `accepted`.

## Remaining release prerequisites

- **Re-notarize the 2026-11-14 stage.** The accepted submission predates the
  helper packaging fix, so the shipped helper would still be the broken layout.
  Run `package_dmg.sh` with `NOTARY_PROFILE=omlx-notary` and capture the new
  SHA-256.
- The SMJobBless installation itself needs one click on the signed app
  (`Performance → Apply with Admin…`) to prove the one-prompt install; the
  packaging is now correct, the click is not.
- A clean first-run test of the runtime installer (see
  [`SLIM-RUNTIME.md`](SLIM-RUNTIME.md)).
- A version/tag and release notes.

## P2 status — DMG and release

| Item | Doable now — done | Gated | Next step |
|---|---|---|---|
| Slim stage build with helper + installer | Done 2026-11-14 (`build.sh --slim`) | – | rebuild from the exact release commit |
| Helper → app signing order, hardened runtime | Done, `codesign --verify --deep --strict` passes | – | – |
| Helper trust pair (`SMPrivilegedExecutables` ↔ `SMAuthorizedClients`) | Done, verified both directions with `codesign --verify -R=…` | – | re-verify after a certificate change |
| DMG creation and image signing | Done (`build/slim-audit`, + SHA-256) | – | delete the audit image after review |
| Notarize + staple | – | Needs `NOTARY_PROFILE` (present on .11; not run in this audit) | one script run on the rebuilt stage |
| Gatekeeper assessment | Verified: app inside the notarized image `accepted`; signed-unnotarized image `Unnotarized Developer ID` | – | re-check after notarization |
| Release asset, tag, notes | – | Release decision | after the click-through test |
| Xcode test bundle signing | – | “Mac Development” certificate absent; Developer ID only | tests run with `CODE_SIGNING_ALLOWED=NO` |

## Verification set

- `codesign --verify --deep --strict` on the app; `codesign --verify` on the
  signed image.
- `spctl --assess --type execute` on the app inside the mounted image (the
  check that reflects what a user's Mac evaluates).
- `spctl --assess --type open --context context:primary-signature` on the DMG —
  only meaningful once the image itself is signed.
- `xcrun stapler validate` on the DMG.
- `hdiutil verify` on the DMG.
- The updater's DMG download path remains the existing `AppUpdater` flow.
