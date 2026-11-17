# Peer versions — keep every cluster Mac on the same build

**Why this file exists.** The Cluster Assistant drives the elected coordinator's
**own** admin routes over SSH (`/plan`, `/stage`, `/deployments`) and, for the
cluster-wide operation lock, `/admin/api/cluster/assistant/operations/*`. A Mac
that runs an **older build** answers `405 Method Not Allowed` for a route that
exists on the origin. Live, that showed up as:

> `staging_refused` / `the elected coordinator did not accept the request:
> {"detail":"Method Not Allowed"}`

measured 2026-11-14 when `.12` (M2 Ultra) was elected coordinator while running a
build without the new assistant routes (see `PHASE-2B-RUN.md` §27).

## What the code does now

- The operation-lock mirror **tolerates** a missing lock route (`404`/`405`):
  the cluster-wide lock is unavailable on that Mac, which is logged, and the
  apply/start continues instead of failing. A mixed-version cluster can deploy;
  it just cannot enforce "one apply/start at a time" across origins while the
  coordinator is older (`start._mirror_running`, `_MIRROR_ROUTE_ABSENT`).
- Every other coordinator call (`plan`, `stage`, `deployments`, `models`,
  `observe`) is a **core** route that even older builds expose, so staging,
  activation and measurement work.
- The tuning routes (`/tune`, `/tune-redeploy`) live on the **origin** and only
  the origin records the preset, so tuning works regardless of peer builds.

Tolerance is a safety net, not the target. The target is that every Mac answers
the same route surface, so the cluster-wide lock is real everywhere.

## Read-only check

```sh
uv run python scripts/check_peer_versions.py
```

It probes each Mac (no key, no writes) and prints `OK` or `OLDER BUILD` plus the
missing routes. A route is "present" when it answers `401`/`403`/`422` (it
answered before auth/validation) and "missing" on `404`/`405`. Exit code `1`
means skew was found.

## How to align the Macs

The peers run the app's runtime, not this checkout, so aligning means putting the
**same build** on every Mac:

1. Build/notarize the slim app from this branch (`apps/omlx-mac/Scripts/build.sh`
   + `package_dmg.sh`; see `SIGNING.md`, `DISTRIBUTION.md`, `SLIM-RUNTIME.md`).
2. Install it on `.12`, `.13`, `.14` (and `.11`, this Mac) and let the app's
   first-run runtime installer provision the same oMLX build on each.
3. Re-run `scripts/check_peer_versions.py` — every reachable Mac must print
   `OK` before relying on the cluster-wide lock.

Until that is done, a live run works (the lock is the only degraded piece), but
record it: a second origin driving the same older coordinator could start a
second apply/start, because that coordinator cannot hold the mirror.

## Notes

- **Measured 2026-11-14** with the origin stopped: all three peers (`.13`, `.12`,
  `.14`) answered `OLDER BUILD` — they expose the core `/plan` and `/stage`
  routes but not `/admin/api/cluster/assistant/operations/active` (nor
  `/tuning-preset`, which was being added the same day). So the cluster-wide lock
  is currently enforced only when the coordinator is this Mac.
- Do not "fix" skew by lowering the origin to the peers; the assistant, tuning
  and free-disk rules depend on this branch.
- The helper that raises the wired limit is Phase 5 and unrelated to skew.
- This check never sends a key and never stops or changes a deployment.