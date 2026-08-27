# Public fork and signed macOS release checklist

The `Signed macOS DMG` workflow builds an Apple Silicon app in a job that has
no Apple credentials. A separate `macos-release` environment gates signing,
notarization, verification, and upload to a **draft** GitHub Release. It never
publishes a release or changes repository visibility.

## One-time Apple setup

1. Keep the Apple Developer Program membership and agreements current. Create
   a **Developer ID Application** certificate (not Apple Development, Mac
   Distribution, or Developer ID Installer) and export only that identity plus
   its private key to a password-protected PKCS#12 file.
2. In App Store Connect, create a **Team API key** with the least role that can
   notarize software (Developer is sufficient for this workflow). Apple's
   [API-key documentation](https://developer.apple.com/documentation/appstoreconnectapi/creating-api-keys-for-app-store-connect-api)
   says individual API keys cannot use `notarytool`. Record the key ID and
   issuer ID; the `.p8` private key can be downloaded only once.
3. Revoke and replace either credential immediately if it is exposed. Never
   commit a certificate, private key, API key, password, or encoded copy.

Apple requires Developer ID signing, hardened runtime, secure timestamps, and
valid signatures for distributed executables. The workflow uses `notarytool`,
staples both the app and DMG, then checks them with `codesign`, `stapler`,
`syspolicy_check`, and `spctl`. See Apple's [notarization requirements](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution),
[custom workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow),
and [common issue checks](https://developer.apple.com/documentation/security/resolving-common-notarization-issues).

## One-time GitHub setup

Create a GitHub Actions environment named `macos-release`. Restrict it to the
protected default branch and require at least one reviewer who can compare the
requested tag/commit before secrets are released. Environment protection rules
delay access to environment secrets until approval; see [GitHub's environment documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments).

Add these environment secrets:

| Name | Value |
| --- | --- |
| `APPLE_DEVELOPER_ID_APPLICATION_P12_BASE64` | Base64 of the password-protected Developer ID Application `.p12` |
| `APPLE_DEVELOPER_ID_APPLICATION_P12_PASSWORD` | PKCS#12 export password |
| `APPLE_NOTARY_API_KEY_P8_BASE64` | Base64 of the App Store Connect Team API `.p8` |
| `APPLE_NOTARY_KEY_ID` | Team API key ID |
| `APPLE_NOTARY_ISSUER_ID` | Team API issuer UUID |

Add the environment variable `APPLE_TEAM_ID` containing the 10-character Apple
Developer Team ID. It is an identifier, not a private credential.

GitHub secrets are limited to 48 KB and should not contain JSON wrappers. Encode
the two binary/text credential files as one-line base64 values before storing
them. Do not paste their decoded contents into an issue, workflow log, command
argument, or repository file. GitHub's [secret handling reference](https://docs.github.com/en/actions/reference/security/secrets)
describes scope, precedence, and redaction limits.

Protect the default branch, require CI and review for workflow/release-script
changes, enable Dependabot and GitHub secret scanning with push protection, and
leave Actions restricted to reviewed, SHA-pinned actions. The workflow uses
only SHA-pinned `actions/*` dependencies.

As of 2026-08-23, GitHub's official runner catalog maps `macos-26` to the
Apple Silicon macOS 26 image, and that image includes Xcode 26.6 at the path
selected by the workflow. Recheck the [runner image inventory](https://github.com/actions/runner-images/blob/main/images/macos/macos-26-arm64-Readme.md)
before future Xcode pin changes instead of assuming `macos-latest` is stable.

## Cutting a release

1. Update `omlx/_version.py`, merge the reviewed release commit, and create and
   push a matching tag such as `v0.6.3`. Prefer a protected, signed tag.
2. Run `Signed macOS DMG` manually and enter that existing tag. Approve the
   `macos-release` environment only after confirming the tag resolves to the
   intended reviewed commit.
3. Inspect the workflow's signed DMG artifact and its SHA-256 file. Confirm the
   draft release notes and assets. On a separate Mac, download through a
   quarantine-setting browser, verify the checksum, mount the image, and launch
   the app. Test a full in-app update from the preceding release as well.
4. Publish the draft manually only after those checks pass. Publishing remains
   an explicit maintainer action.

The DMG is named `oMLX-VERSION-macos15-26-arm64.dmg`, matching the updater's
macOS-range selection. `OMLX_RELEASE_REPOSITORY` is embedded at build time and
defaults to `jonathan308/omlx`; the workflow sets it to the repository running
the workflow. The updater validates the DMG with Gatekeeper, mounts it with
image verification enabled, and requires the app to be notarized Developer ID
code with the same bundle ID and Apple Team ID as the running app. Quarantine
metadata is preserved. If relaunching the new app fails, the worker swaps the
previous app back before relaunching it.

An installation signed by the upstream project's Apple team cannot update
itself directly to a build signed by this fork's different team. That is the
intended result of the same-team trust policy. Existing users must download,
verify, and manually install the first fork-signed DMG once; subsequent fork
releases can use the in-app updater. Do not weaken the team check to automate
that one-time migration.

## Public-history and licensing audit (2026-08-23)

The fork is already public. A filename/history scan found no tracked `.p12`,
`.p8`, private-key, provisioning-profile, `.env`, provider-specific GitHub
token, AWS access-key, or PEM private-key markers. Gitleaks 8.30.1 also scanned
2,652 commits (about 202 MB) with values fully redacted. Its generic API-key
detector reported 14 candidates; metadata-only review mapped them to SHA-256
benchmark fingerprints, test/synthetic `server_key` values, corpus examples,
and long code/prose identifiers rather than provider-specific credentials.
That classification is not proof that arbitrary prose or binary blobs contain
no sensitive data. Independently review the redacted findings and keep GitHub
secret scanning with push protection enabled. If any candidate is real, revoke
it first; deleting a file in a new commit does not remove it from Git history.

The repository declares Apache-2.0 in `LICENSE` and `pyproject.toml`. MLX-derived
kernel sources retain an MIT license that the DMG workflow copies alongside the
Apache license. Public source and binary distribution still need a complete
dependency/data audit: the bundle includes many Python packages, vendored
JavaScript/CSS/fonts, model/evaluation datasets, and native kernels with their
own notices and dataset terms. Apple copyright headers in MLX-derived/common
kernel and parser files do not all carry an adjacent SPDX identifier, some
vendored web assets identify permissive licenses only in
`omlx/admin/vendor_deps.py`, and evaluation corpora contain third-party
material. Treat confirmation of provenance, exact notice consolidation, and
dataset redistribution rights as a release-blocking legal review, not as
something code signing or this checklist resolves. Preserve all upstream
copyright, patent, trademark, and attribution notices, and do not imply that
the upstream project endorses the fork.
