# Developer ID signing and notarization — setup checklist

Status on .11 (2026-11-14): **already set up**. The Developer ID Application
identity for team `YB894QRG4W` and the `omlx-notary` keychain profile exist, and
a DMG was notarized and accepted on 2026-11-13. Steps 1–2 below are therefore a
record of how the machine was prepared, not work that is still missing; what
remains is the SMJobBless click-through and a fresh notarization of the rebuilt
stage (§4).

This is the external prerequisite for the signed helper, the signed app and the
DMG. The repository cannot create the certificate: it belongs to the Apple
Developer account and requires the account holder's sign-in and two-factor
authentication.

Recorded team: **Andreas Kolb — `YB894QRG4W`**.

## 1. Add the Apple account to Xcode

1. Open **Xcode → Settings → Accounts**.
2. Add the Apple ID that belongs to team `YB894QRG4W`.
3. Select the team and open **Manage Certificates…**.
4. Click **+** and create **Developer ID Application**.
   - This requires a paid Apple Developer Program membership with the
     Developer ID capability.
   - A free/personal team cannot create a Developer ID certificate.
5. Verify from a terminal:

   ```sh
   security find-identity -v -p codesigning
   ```

   Expected: a line containing
   `Developer ID Application: Andreas Kolb (YB894QRG4W)`.

   **State on .11, 2026-11-14:** one valid identity is installed —
   `E8BDDA9205AA188B20CE35DC5AD7E5BD8E60DD8B "Developer ID Application: Andreas
   Kolb (YB894QRG4W)"`. This step is done. (Earlier revisions of this file
   said the machine reported zero identities; that is stale.)

   There is deliberately **no Mac Development certificate** for the team, so
   Xcode's automatic signing cannot build the test bundle: run local tests with
   `-scheme oMLX CODE_SIGNING_ALLOWED=NO`. Release packaging signs explicitly
   with the Developer ID identity and is unaffected.

## 2. Notarization credentials

Choose one of these, then store it in the login keychain:

```sh
# App-specific password:
xcrun notarytool store-credentials omlx-notary \
  --apple-id "<apple-id>" \
  --team-id YB894QRG4W \
  --password "<app-specific-password>"

# Or an App Store Connect API key:
xcrun notarytool store-credentials omlx-notary \
  --key "<AuthKey_XXXX.p8>" \
  --key-id "<key-id>" \
  --issuer "<issuer-uuid>"
```

Verify:

```sh
xcrun notarytool history --keychain-profile omlx-notary
```

**State on .11, 2026-11-14:** the `omlx-notary` keychain profile exists and its
history contains accepted submissions (`oMLX-0.7.4.dev1.dmg`, last accepted
2026-11-13). No credential is stored in the repository, and this audit did not
create or modify one.

Do not put the password, the `.p8` key or the keychain profile in the repo.

## 3. Sign, notarize and package

Once the identity and profile exist:

```sh
SIGN_IDENTITY="Developer ID Application: Andreas Kolb (YB894QRG4W)" \
NOTARY_PROFILE=omlx-notary \
apps/omlx-mac/Scripts/package_dmg.sh
```

The script signs the **helper first**, then the app with hardened runtime and
the app entitlements, then creates the `.dmg` image and signs it, then
notarizes and staples it: `notarytool --wait`, `stapler staple`,
`spctl --assess`, and the SHA-256. Signing the image matters — without it
`spctl --type open` cannot assess a notarized image (`no usable signature`);
the 2026-11-14 audit fixed that in the script. See
[`DISTRIBUTION.md`](DISTRIBUTION.md) for the full bundle build.

Verified on .11 on 2026-11-14 with `SIGN_IDENTITY` set and `NOTARY_PROFILE`
empty: helper, app and image sign and verify, the DMG builds, and notarization
is skipped as documented. No notarization was run in that audit.

## 4. State of the signing work

Done:

- Identity and team verified (`security find-identity`, `codesign -dv`).
- App and helper signed with the Developer ID identity, hardened runtime,
  timestamp; `codesign --verify --deep --strict` passes.
- `.dmg` image signed as well (fixed 2026-11-14); `codesign --verify` passes and
  `spctl --type open` reports `Unnotarized Developer ID` instead of
  `no usable signature`.
- `SMPrivilegedExecutables` / `SMAuthorizedClients` requirements pinned to the
  Developer ID CA and leaf marker, and verified against the signed binaries
  with `codesign --verify -R=…`.
- A DMG was notarized, stapled and accepted on 2026-11-13 (app inside accepted
  by Gatekeeper as `Notarized Developer ID`).

Still gated:

- The **SMJobBless install** itself: it needs one admin click in the signed,
  notarized app on the target Mac, then `launchctl print system/app.omlx.helper`
  to confirm the job. See [`HELPER.md`](HELPER.md).
- A **fresh notarization** of the rebuilt slim stage (the 2026-11-13 submission
  predates the helper packaging fix; the notary profile is present, so this is
  one script run, not a credential problem).
- **Mac Development certificate** (optional): only needed to run the Xcode test
  bundle with signing enabled; tests run fine with `CODE_SIGNING_ALLOWED=NO`.

Without a Developer ID certificate the app builds only with
`CODE_SIGNING_ALLOWED=NO`, the helper cannot be blessed on another Mac, and the
DMG cannot be notarized.
