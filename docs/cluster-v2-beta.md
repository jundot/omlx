<!-- markdownlint-disable MD013 -->

# Cluster v2 Beta contract and feature-PR evidence

Cluster v2 replaces the previous manual, pipeline-first distributed surface
with one capability-driven flow. This document is the review contract for the
major feature PR: a code path is not considered complete merely because its
UI exists, and a benchmark is not a compatibility certificate.

## Product boundary

The Beta supports automatic planned scaling. oMLX discovers or accepts Macs,
pairs identities, verifies their software and link, inventories and stages a
checkpoint, recommends an executable topology, signs the exact placement, and
loads isolated ranks through the normal model lifecycle.

Adding or removing a device from a resident model performs a signed replan and
reload. Live elastic tensor migration is not a Beta claim. A failed distributed
launch never falls back to loading a full local model.

## End-to-end operator flow

| Step | Cluster v2 contract | Failure behavior |
| --- | --- | --- |
| Detect | Stable node identity; Bonjour, IPv6 multicast, persisted-pair, manual-IP and optional Tailscale control discovery | Discovery failure keeps Add by IP available and never grants trust |
| Pair | Human approval, expiring authenticated exchange, 0600 device/enrollment records, pinned SSH identity | Wrong, expired, replayed or changed identities fail closed |
| Qualify link | Local and peer Thunderbolt, `rdma_ctl`, `ibv_devices`, route, JACCL/runtime and kernel fingerprints | The wizard explains Recovery `rdma_ctl enable`; TCP ring fallback is explicit |
| Select model | Union the selected nodes' downloaded-model inventory and retain its source node/path | Unreadable or incompatible models remain visible with a reason |
| Stage | Copy only required checkpoint/sidecar files, resume partial jobs, preserve per-node absolute paths | Partial or mismatched manifests cannot activate |
| Recommend | Evaluate architecture-valid TP, pipeline and capability-gated hybrid factors against memory, context and measured fabric | Unsupported TP/pipeline dimensions fail before model loading |
| Approve | Render layer ranges, tensor shares, context/cache reservation and backend; sign the placement | Any changed node, model, runtime, tensor vector or setting invalidates approval |
| Load | Progressive rank-local materialization, post-shard memory guard, residency verification and all-rank readiness canary | One rank failure tears down the entire group in bounded time |
| Serve | Normal OpenAI/Anthropic/Responses routes, continuous batching, per-request metrics, targeted disconnect cancellation and tiered rank cache | Media or request features without a distributed contract return an actionable error |
| Recover | Launcher/peer watchdogs, JACCL progress guard, communicator lease, orphan reaper and stale-row reconciliation | No local fallback, detached launcher or stale loaded model is reported as healthy |

## Universal planning and loading

The planner is model-layout driven, not DS4 hard-coded. It reads safetensors
headers and architecture dimensions, accounts for fixed and per-layer weights,
KV scope and context reservation, and emits only strategies implemented by the
loaded adapter/runtime.

- Tensor parallelism gives every TP rank the selected stage's layers but only
  its tensor slices. Progressive loading drops full materializations before the
  next layer and verifies measured resident bytes after load.
- Pipeline parallelism gives ranks different contiguous layer ranges and keeps
  KV state with the owning range.
- Hybrid TP x pipeline is exposed only when the backend advertises logical
  subgroup support. The code path exists, but promotion still requires a real
  four-device execution gate.
- Unequal tensor recommendations are advisory until an exact, runtime-keyed
  parity qualification is persisted. Drift falls back to equal safe shards.
- Model files may live at different absolute paths. A signed `path_map` and
  complete sidecar staging replace the former same-path requirement.
- Phase split is a separate, default-off two-rank full-replica strategy for
  compatible text models. The planner independently proves full weights plus
  KV fit on both Macs, signs prefill/decode ownership, stages a complete model
  manifest to each owner and launches the persistent phase server. It is not a
  fallback for a model that fails TP/pipeline fit.

## Full-replica Phase split Beta contract

The coordinator UI covers selection, role display, context reservation,
serving profile, hot/SSD prompt reuse, staging/synchronization, signed
activation, live per-request/phase metrics, all-rank cache clear, unload/reload
and change-model flow. Replans preserve phase ownership, and the active card
follows the runtime-loaded deployment rather than stale registry order.

The backend uses a model-independent cache manifest carrying model identity,
cache class, tree path, shape, dtype and metadata. Tensor leaves transfer
directly over Ring/JACCL; rank 0 reconstructs each installed cache through its
`from_state` contract. Unknown/non-array cache state fails closed. Public chat,
streaming, reasoning, structured tool calls, sampling/logit processors,
chunk-boundary prefill cancellation, decode cancellation and exact/nearest hot
prefix reuse are wired. Persistent SSD snapshots use the normal cluster cache
root and clear workflow.

This is not yet phase-sharded universal execution. The present contract is two
full replicas, rank 1 prefill and rank 0 decode/API, text only, with MTP,
multimodal inputs and guided grammar disabled. DS4 Flash cannot enter this mode
on the reference 128 GB M5 because it does not fit twice; it remains a TP model
until a logical phase can itself be a shard group.

## Stability mechanisms replacing the old surface

- Isolated launchers and ranks with deterministic plan agreement.
- Rank-ready/release barrier before any model collective.
- Load-time admission and live memory guards that include existing rank
  residency.
- Parent, peer and control-plane watchdogs plus bounded JACCL no-progress
  teardown.
- One-live-communicator fence and kernel-released process lease while the
  current JACCL build cannot safely own multiple independent QP sets.
- Request-scoped cancellation consumed at shared batch boundaries, so one
  client disconnect does not sever another request's collective graph.
- Deployment/plan/rank-scoped SSD prompt snapshots, unanimous restore votes,
  bounded write-behind and authenticated all-rank cache clearing.
- Main-dashboard ownership derived from the live engine pool and current rank
  markers, never detached historical jobs.
- Independent active-request rows and separate per-request versus aggregate
  throughput semantics.

## Current physical qualification snapshot

Hardware: Apple M3 Ultra 256 GB plus Apple M5 Max 128 GB, direct Thunderbolt 5,
JACCL/RDMA. The active lossless DS4 plan is equal TP2 (`4:4`), about 80.03 GB
measured weights per rank, with a one-million-token context reservation.

| Gate | Accepted observation |
| --- | ---: |
| DS4 cold-prefix 30K, prompt A | 742.17 tok/s |
| DS4 cold-prefix 30K, prompt B | 722.80 tok/s |
| DS4 30K two-prompt average | 732.49 tok/s |
| DS4 cold-prefix 100K | 684.66 tok/s |
| DS4 30K-to-100K taper | about 6.5% |
| DS4 non-MTP B1 decode | about 29-31 tok/s |
| DS4 MTP fixed-depth-5, high acceptance | about 79.8-80.6 tok/s raw |
| DS4 non-MTP B1/B2/B4 aggregate, current gate | 31.22 / 47.35 / 75.22 tok/s |
| Qwen3-30B-A3B TP2 cold-prefix 30K | 1,533.73 tok/s |
| Qwen3-30B-A3B TP2 B1 decode | 56.83 tok/s |
| Qwen3-30B-A3B TP2 B4 aggregate | 216.3 tok/s |
| Qwen3.8-27B Phase split 9,410-token cold prefill compute average | 991.36 tok/s |
| Qwen3.8-27B Phase split decode | 29.59 tok/s |
| Phase cache handoff, 770.64 MB / 128 arrays | 104.97 ms / 7.34 GB/s |
| Exact 9,410-token Phase prefix repeat | 12.31 s -> 1.40 s |
| Phase B4 queued throughput gain | about 1.29x vs sequential stage time |

An outer DS4 `3:5` split reached about 826 tok/s at 30K but changed a greedy
completion on the parity corpus. It is persisted as rejected and is not an
accepted Beta result. See `benchmark-provenance.md` for cache, concurrency and
measurement labels.

## Automated evidence at the current feature head

At source `5575294f13fad2a2985ac2746ea5fcf6223c911b`:

- 1,768 cluster, distributed-engine, cache, lifecycle, route and wizard tests
  passed together on Python 3.13;
- the preceding full CI head passed Python 3.11, 3.12 and 3.13;
- live read-only autoconfigure against the paired M3/M5 returned
  `ready_to_activate=true`, JACCL with `fabric_ready=true`, a signed equal TP2
  plan, complete staging and no preflight issues;
- ten consecutive physical unload/reload cycles each reached zero rank
  processes and no launcher after unload, then exactly two ready ranks plus a
  successful real completion after reload;
- terminating the worker rank removed the launcher and both ranks in bounded
  time, revoked the model's loaded state, returned HTTP 503 instead of hanging,
  and reloaded to a successful canary without rebooting either Mac;
- a 200K-token prefill disconnect cancelled at the shared 4,096-token boundary
  with zero failures; a concurrent decode disconnect cancelled only its target
  while the other 256-token stream completed;
- a fresh distributed cache gate produced 0 cached tokens, then 8,000 reused
  tokens; authenticated hot/SSD clear reported both ranks and the third
  identical request returned to 0 cached tokens;
- physical all-rank cache clear deleted hot and SSD state on both Macs and the
  repeat request reported zero cached tokens;
- both active ranks report the same plan hash, measured shard bytes, request
  counters and completion state.

The upstream-rc3 integration branch `feat/cluster-v2-beta-pr` at `f10ad8cf`
then passed 1,770/1,770 Cluster v2 tests, 1,072/1,072 conflict-focused tests,
and the complete repository gate (**11,803 passed, 62 skipped, 77 deselected**).
The full native extension stack also rebuilt successfully from that tree on
both qualification Macs.

## Gates before calling a public build stable Beta

The feature PR can be reviewed before release packaging, but the stable-Beta
label requires all of these artifacts:

1. Visual click-through of every wizard state in the packaged app, including
   first-run Local Network and Remote Login permissions.
2. Simultaneous and staggered two-Mac reboot recovery, followed by a clean
   load without a manual discovery seed.
3. Ten load/unload cycles, forced-rank death/reload and an extended mixed
   prefill/decode soak with no orphan, stale dashboard row or poisoned JACCL
   communicator.
4. B1/B2/B4/B8 plus mixed-long-prefill concurrency on every advertised model
   family, with per-request latency separated from aggregate throughput.
5. Fresh hot-cache, SSD-cache, all-rank clear, prefill cancel, decode cancel and
   cancel-to-replacement evidence.
6. A dense text, sparse MoE, DS4 and text-backbone VLM matrix. Full multimodal
   TP remains outside Beta until its vision/audio towers are distributed.
7. A real four-device gate before hybrid TP x pipeline is enabled by default.
8. Signed/notarized DMG installation, upgrade, uninstall and rollback on a
   clean Mac.

No unchecked item may be converted into a positive release claim merely by a
unit test, synthetic benchmark, two-rank result or planner-only proof.
