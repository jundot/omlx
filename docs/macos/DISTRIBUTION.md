# Distribution — direct DMG, outside the App Store

## Decision

- Distribution is a **direct download DMG**, not the App Store.
- The app and the helper are signed with a **Developer ID Application**
  certificate (team supplied at build time as `OMLX_TEAM_ID`, see
  [Signing](SIGNING.md)), hardened runtime, and the DMG is notarized with
  `notarytool`.
- Updates use the existing GitHub Releases poller (`ReleasesChecker` +
  `AppUpdater`); the DMG is the release asset. No appcast or EdDSA key is
  introduced.

## Build and package

The app is **slim**: Python and MLX are not embedded. The staged app contains
the Swift shell, the `omlx`/`omlx_web` source, the first-run runtime installer
(`uv` + lock) and the SMJobBless helper. The runtime is provisioned once from the
Welcome card (or `provision_runtime.sh`); see
[Slim runtime](SLIM-RUNTIME.md).

`apps/omlx-mac/Scripts/package_dmg.sh` signs the helper first, then the app,
then the `.dmg` image itself; it notarizes, staples and assesses the image only
when `SIGN_IDENTITY` **and** `NOTARY_PROFILE` are supplied.

Signing the image is not cosmetic. Without it the script leaves the container
unsigned, so `spctl --type open` reports `no usable signature` even for an image
that was notarized and stapled — the ticket is there, the container is not
assessable. With the image signed, a signed-but-unnotarized build reports
`Unnotarized Developer ID`; a notarized one reports `Notarized Developer ID`.

1. Provision the external runtime and build the slim app:

   ```sh
   apps/omlx-mac/Scripts/provision_runtime.sh
   OMLX_TEAM_ID=<TEAM_ID> \
   OMLX_NEXT_OUT="$PWD/apps/omlx-mac/build/SlimStage" \
     apps/omlx-mac/Scripts/build.sh release --slim
   ```

   `OMLX_TEAM_ID` is what the helper's and the app's SMJobBless requirements are
   generated from. Without it the stage builds but cannot bless the helper.

2. Sign the staged app with the Developer ID identity, hardened runtime and the
   app entitlements. `package_dmg.sh` does this (helper first, then the app), but
   the equivalent manual commands are:

   ```sh
   codesign --force --options runtime --timestamp \
     --identifier app.omlx.helper \
     --sign "Developer ID Application: <Your Name> (<TEAM_ID>)" \
     apps/omlx-mac/build/SlimStage/oMLX.app/Contents/Library/LaunchServices/app.omlx.helper
   codesign --force --deep --options runtime \
     --entitlements apps/omlx-mac/Resources/oMLX.entitlements \
     --sign "Developer ID Application: <Your Name> (<TEAM_ID>)" \
     apps/omlx-mac/build/SlimStage/oMLX.app
   codesign --verify --deep --strict --verbose=2 apps/omlx-mac/build/SlimStage/oMLX.app
   ```

3. Create, sign and (when the credentials are set) notarize the DMG. The app
   path must be the **slim stage** (`SlimStage`), not `Stage`: only the slim
   build contains the helper and the runtime installer.

   ```sh
   APP_PATH="$PWD/apps/omlx-mac/build/SlimStage/oMLX.app" \
   OUT_DIR="$PWD/build/slim" \
   SIGN_IDENTITY="Developer ID Application: <Your Name> (<TEAM_ID>)" \
   NOTARY_PROFILE=omlx-notary \
   apps/omlx-mac/Scripts/package_dmg.sh
   ```

   Without `NOTARY_PROFILE` it produces a Developer ID-signed app in an
   **unnotarized** DMG for local inspection and says so.

4. The script already runs these; keep them as the manual fallback:

   ```sh
   codesign --force --timestamp \
     --sign "Developer ID Application: <Your Name> (<TEAM_ID>)" \
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

## What a good artifact looks like

- Slim DMG is roughly 70–80 MB (decimal) with the runtime installer and the
  helper; the app inside is not the multi-hundred-MB embedded-runtime build
  (see [Slim runtime](SLIM-RUNTIME.md) for the measured comparison).
- Signed with Developer ID Application, hardened runtime and timestamp on app
  and helper; the `.dmg` is signed too.
- The notarized image has a stapled ticket (`xcrun stapler validate` passes).
- Contains the external-runtime installer (`uv` + lock) and the SMJobBless
  helper with both embedded `__TEXT` plist sections.
- `codesign --verify --deep --strict` passes on the app inside the mounted image
  and on the staged app; `hdiutil verify` reports a valid checksum.

### What Gatekeeper reports

| Check | Expected |
|---|---|
| `spctl -a -t open --context context:primary-signature` on an **unsigned** DMG | `rejected`, `source=no usable signature` — the container was never signed |
| `spctl -a -t execute` on the **app inside** a notarized image | `accepted`, `source=Notarized Developer ID` |
| `spctl -a -t open` on a **signed, unnotarized** DMG | `rejected`, `source=Unnotarized Developer ID` — correct and expected |
| `spctl -a -t open` on a **signed, notarized, stapled** DMG | `accepted` |

The second row is why a notarized image can still be misreported as “accepted”
when only the app inside was assessed: check the container and the app
separately.

## Remaining release prerequisites

- **Notarize the current stage.** Any re-package after a helper or runtime
  change needs a fresh `package_dmg.sh` run with `NOTARY_PROFILE`, and the new
  SHA-256 recorded.
- The SMJobBless installation itself needs one click on the signed app
  (`Performance → Apply with Admin…`) to prove the one-prompt install. The
  packaging can be correct while the click is still untested.
- A clean first-run test of the runtime installer (see
  [Slim runtime](SLIM-RUNTIME.md)).
- A version/tag and release notes.

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