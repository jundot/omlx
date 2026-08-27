<!-- markdownlint-disable MD013 -->

# oMLX 0.6.4 Beta 1 release-candidate notes

> This is a readiness record, not a published release announcement. The
> recommended prerelease tag is `v0.6.4b1`, but the source version has not been
> changed to `0.6.4b1`, no tag has been created, and no signed artifact exists
> yet.

Benchmark source commit: `6ebc22a822e3b50fe3c3d59acf1da62e8694f5dc`

Expected artifact after the release gates pass:
`oMLX-0.6.4b1-macos15-26-arm64.dmg`

## Readiness summary

| Gate | State | Evidence or remaining action |
| --- | --- | --- |
| Text-only TP on a second model family | Passed | Qwen3-30B-A3B-4bit loaded with an equal two-rank split and completed parity, cache, cancel, and concurrency qualification. |
| Cold-prefix prefill and decode | Passed | The Qwen and DS4 measurements below were captured with no reusable prompt prefix. |
| Exact-prompt cache reuse | Passed | Qwen reported 30,004 cached tokens out of 30,005 and reduced TTFT from 19.92 seconds to 0.67 seconds. |
| In-flight prefill cancellation | Passed | Both Qwen ranks stopped at 26,624 processed tokens and returned to ready. |
| Concurrent generation | Passed | Qwen completed four 512-token streams at 216.3 aggregate tok/s; DS4 completed its qualified B4 run at 69.3 aggregate tok/s. |
| Functional greedy parity | Passed with limitation | With thinking disabled, four of six Qwen cases were token-exact and two of six were semantically equivalent. This is not a bit-exact or token-exact parity claim. |
| Version and tag | Pending | After review and green release CI, change `omlx/_version.py` from `0.6.4.dev1` to `0.6.4b1`, then create a matching protected tag `v0.6.4b1`. |
| Signed, notarized DMG | Blocked | Install the five missing Apple secrets, configure `APPLE_TEAM_ID`, and approve the protected `macos-release` environment. |

Do not advertise this candidate as signed, notarized, or downloadable until the
gated workflow has produced and verified the DMG and checksum. Publishing the
draft GitHub Release remains a separate maintainer action.

## Validated Qwen TP2 results

Model: `mlx-community/Qwen3-30B-A3B-4bit`

Topology: Apple M3 Ultra with 256 GB unified memory plus Apple M5 Max with
128 GB unified memory, equal `1:1` tensor-parallel split, direct Thunderbolt 5,
and JACCL RDMA.

Here, **cold prefix** means `prefix-miss` on an already loaded deployment; it
does not assert a cold process or include model-load time. API and rank-marker
rates are listed separately instead of being blended.

| Scenario | Workload and cache state | Observed result |
| --- | --- | --- |
| Cold-prefix prefill | Approximately 30K input tokens, `prefix-miss`, single stream | 1,533.73 API tok/s; 1,537.15 rank-marker tok/s |
| Decode during that run | `prefix-miss`, single stream, non-MTP | 56.83 API tok/s; 56.46 rank-marker tok/s |
| Exact-prompt repeat | 30,005-token fixed prompt, `hot-prefix-hit` in the rank prompt LRU | 30,004 cached tokens; TTFT 19.92 s -> 0.67 s |
| In-flight cancel | Cancellation during cold-prefix prefill | Both ranks stopped at 26,624 processed tokens and returned to ready |
| Concurrent decode | Four independent streams, 512 output tokens each, non-MTP | 2,048 tokens in 9.469 s; 216.3 aggregate tok/s |
| Greedy functional parity | Six cases, thinking disabled | 4/6 token-exact; 2/6 semantically equivalent |
| MTP decode | Not qualified | No Qwen MTP performance or correctness claim is made for Beta 1 |

These results qualify the general text-only TP path on Qwen; they do not claim
that every model architecture is supported. In particular, Qwen3.8 models with
the vision tower are VLMs and are excluded from this distributed beta. Do not
weaken the text-only guard to make them appear compatible.

## Validated DS4 reference results

Model: DeepSeek-V4-Flash-0731-MXFP4-MLX (DS4 MXFP4).

TP2 topology: the same two Macs and link, using the qualified `3:5` split
(M3 Ultra:M5 Max). All prefill rows are cold-prefix, single-stream observations
and exclude model-load time.

| Scenario | Workload | API tok/s | Rank-marker tok/s |
| --- | --- | ---: | ---: |
| TP2 prefill | 30K, `prefix-miss`, single stream | 841.51 | 846.90 |
| TP2 prefill | 100K, `prefix-miss`, single stream | 771.77 | 773.86 |
| TP2 prefill | 250K, `prefix-miss`, single stream | 621.79 | 622.74 |
| TP2 decode | `prefix-miss`, single stream, non-MTP | approximately 30-32 | not recorded here |
| TP2 decode | `prefix-miss`, single stream, MTP depth 5 | 75.93 | not recorded here |
| TP2 aggregate decode | four concurrent streams, non-MTP | 69.3 aggregate | not recorded here |

Single-node reference topology: Apple M3 Ultra with 256 GB unified memory,
with no distributed transport. The accepted DS4 kernel stack measured 481.02
tok/s at 30K, cold-prefix and single-stream.

The values above are reference observations, not cross-hardware guarantees or
medians over an undisclosed run count. They must not be presented as proof of a
1,000-1,300 tok/s target. See
[`docs/benchmark-provenance.md`](benchmark-provenance.md) for labels and
publication requirements.

## Beta tester focus

- Load, unload, restart, and reconnect both ranks; confirm that an unloaded
  model never remains stale in the cluster dashboard.
- Cancel in-flight requests during both prefill and decode; capture the request
  ID, processed-token count, cancel-to-stop latency, and both rank states.
- Exercise the distributed rank-local prompt LRU and durable SSD prompt-snapshot
  tier, including restore and process restart; report cached-token counts and
  cache source. This is distinct from the local engine's paged hot/SSD KV tier.
- Run independent concurrent prompts and report each request's throughput as
  well as aggregate throughput.
- Qualify additional text-only model families without model-specific split or
  kernel overrides leaking into other deployments.

Please attach diagnostics, exact reproduction steps, source commit, model
revision, and whether each run was prefix-miss or prefix-hit. Do not attach API
keys, private prompts, model credentials, or signing material.

## Signing blocker

The protected `macos-release` job cannot sign or notarize Beta 1 until its
environment contains all five secrets:

- `APPLE_DEVELOPER_ID_APPLICATION_P12_BASE64`
- `APPLE_DEVELOPER_ID_APPLICATION_P12_PASSWORD`
- `APPLE_NOTARY_API_KEY_P8_BASE64`
- `APPLE_NOTARY_KEY_ID`
- `APPLE_NOTARY_ISSUER_ID`

It also requires the 10-character environment variable `APPLE_TEAM_ID` and an
approved, protected GitHub Actions environment named `macos-release`. Follow
[`docs/release-public-checklist.md`](release-public-checklist.md); never commit,
log, or paste the credential values into release notes.

## Artifact verification after the gate passes

After downloading the DMG and its generated `.sha256` file:

```sh
shasum -a 256 -c oMLX-0.6.4b1-macos15-26-arm64.dmg.sha256
xcrun stapler validate oMLX-0.6.4b1-macos15-26-arm64.dmg
spctl --assess --type open --verbose=2 \
  --context context:primary-signature oMLX-0.6.4b1-macos15-26-arm64.dmg
```

The checksum proves byte identity. Gatekeeper and stapler checks confirm the
signed and notarized distribution. Do not bypass Gatekeeper for this build.
