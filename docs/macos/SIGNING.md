# Developer ID signing and notarization

This is the external prerequisite for the signed helper, the signed app and the
DMG. The repository cannot create the certificate: it belongs to an Apple
Developer account and requires the account holder's sign-in and two-factor
authentication. **No identity, team id, key or notary profile is committed.**

## 0. Configure your team

The project carries no signing team. The Apple team id lives in one variable,
`OMLX_TEAM_ID`, which drives three things:

- `DEVELOPMENT_TEAM` in `apps/omlx-mac/oMLX.xcodeproj` (`$(OMLX_TEAM_ID)`),
- the app's `SMPrivilegedExecutables` requirement,
- the helper's `SMAuthorizedClients` requirement (rendered by `build.sh`).

Set it for a build:

```sh
OMLX_TEAM_ID=ABCDE12345 xcodebuild build -scheme oMLX ...
OMLX_TEAM_ID=ABCDE12345 apps/omlx-mac/Scripts/build.sh release --slim
OMLX_TEAM_ID=ABCDE12345 uv run pytest apps/omlx-mac/Tests/python
```

In Xcode, add `OMLX_TEAM_ID` as a **User-Defined** build setting (project or
target level) or choose a team in **Signing & Capabilities**; the build setting
wins. Leave it unset when you have no Apple team: the app still builds with
`CODE_SIGNING_ALLOWED=NO`, but the helper cannot be blessed and a DMG cannot be
notarized. See [Privileged helper](HELPER.md) for the requirement structure.

## 1. Add the Apple account to Xcode

1. Open **Xcode → Settings → Accounts**.
2. Add the Apple ID that belongs to your team.
3. Select the team and open **Manage Certificates…**.
4. Click **+** and create **Developer ID Application**.
   - This requires a paid Apple Developer Program membership with the
     Developer ID capability.
   - A free/personal team cannot create a Developer ID certificate.
5. Verify from a terminal:

   ```sh
   security find-identity -v -p codesigning
   ```

   Expected: one line containing
   `Developer ID Application: <Your Name> (<TEAM_ID>)`.

   There is deliberately **no Mac Development certificate** for the team in the
   usual setup, so Xcode's automatic signing cannot build the test bundle: run
   local tests with `-scheme oMLX CODE_SIGNING_ALLOWED=NO`. Release packaging
   signs explicitly with the Developer ID identity and is unaffected.

## 2. Notarization credentials

Choose one of these, then store it in the login keychain:

```sh
# App-specific password:
xcrun notarytool store-credentials omlx-notary \
  --apple-id "<apple-id>" \
  --team-id <TEAM_ID> \
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

Do not put the password, the `.p8` key or the keychain profile in the repo.

## 3. Sign, notarize and package

Once the identity and profile exist:

```sh
OMLX_TEAM_ID=<TEAM_ID> \
SIGN_IDENTITY="Developer ID Application: <Your Name> (<TEAM_ID>)" \
NOTARY_PROFILE=omlx-notary \
apps/omlx-mac/Scripts/package_dmg.sh
```

The script signs the **helper first**, then the app with hardened runtime and
the app entitlements, then creates the `.dmg` image and signs it, then notarizes
and staples it: `notarytool --wait`, `stapler staple`, `spctl --assess`, and the
SHA-256.

Signing the image matters: without it `spctl --type open` cannot assess a
notarized image (`no usable signature`) even though the app inside it is fine.
See [Distribution](DISTRIBUTION.md) for the full bundle build.

Before signing, the script reads the app's `SMPrivilegedExecutables` requirement
and refuses to continue when it does not name the team of `SIGN_IDENTITY`: a
mismatch would only surface at bless time, after the user already approved the
admin dialog. The fix is always to rebuild the stage with the right
`OMLX_TEAM_ID`.

## 4. What can be verified offline

- `security find-identity -v -p codesigning` lists the Developer ID identity.
- `codesign -dv --verbose=4 <app>` reports the expected `Authority=` and
  `TeamIdentifier=`.
- `codesign --verify --deep --strict` passes on the helper, the app and the
  image.
- `codesign --verify -R='<requirement>' <binary>` proves both directions of the
  `SMPrivilegedExecutables` ↔ `SMAuthorizedClients` pair (see
  [Privileged helper](HELPER.md)).
- `spctl --type open` on a signed-but-unnotarized image reports
  `Unnotarized Developer ID`; after notarization and stapling it is expected to
  report `Notarized Developer ID`.

## 5. What still needs a real machine

- The **SMJobBless install** itself: one admin click in the signed, notarized
  app, then `launchctl print system/app.omlx.helper` to confirm the job.
- A **fresh notarization** after any change to the helper packaging — the
  notary profile lives outside the repository, so it is one script run.
- **Mac Development certificate** (optional): only needed to run the Xcode test
  bundle with signing enabled; tests run fine with `CODE_SIGNING_ALLOWED=NO`.

Without a Developer ID certificate the app builds only with
`CODE_SIGNING_ALLOWED=NO`, the helper cannot be blessed on another Mac, and the
DMG cannot be notarized.