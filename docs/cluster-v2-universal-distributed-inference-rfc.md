# RFC: cluster v2 universal distributed inference

Status: proposed

Date: 2026-08-24

Scope: Apple Silicon text-generation clusters; documentation only

## Summary

Cluster v2 should make distributed inference an oMLX runtime capability rather
than a collection of model-specific launch scripts. A coordinator admits a
known model, stages its immutable artifacts, produces a signed and
content-addressed execution plan, starts one rank per selected Mac, and exposes
that deployment through the existing oMLX API. Every rank must agree on the
model, plan, backend, and runtime epoch before the deployment becomes ready.

The first beta is intentionally narrower than the eventual design. It targets
two Apple Silicon Macs, qualified text-only model families, tensor parallelism,
and JACCL/RDMA with an explicitly planned fallback. It does not claim support
for arbitrary MLX models, multimodal models, live membership changes, or mixed
Metal/CUDA execution.

This RFC separates evidence from aspiration. The reference implementation has
been exercised with DeepSeek V4 Flash and Qwen3-30B-A3B-4bit, including Qwen
staging, cache reuse, cancellation, concurrent batching, per-request metrics,
and lifecycle reload. Those capabilities are not in upstream `main` merely
because they appear in this document. The proposed PR series below moves them
upstream in reviewable layers.

## Motivation

An inference engine is only generally useful when its control plane is as
reliable as its kernels. A fast two-rank run does not establish that operators
can safely discover peers, reproduce a plan, stage a model, cancel work, unload
every rank, or tell a live request from stale dashboard state. Conversely, a
polished setup flow does not establish that a model's tensor layout is safe to
shard.

Cluster v2 joins those concerns through explicit contracts:

- a model capability contract decides whether and how a checkpoint may be
  sharded;
- a deterministic plan binds the model identity, rank placement, backend, and
  cache geometry;
- a lifecycle state machine makes readiness, cancellation, and teardown
  observable and idempotent;
- authenticated pairing separates discovery from trust;
- content-addressed staging removes the same-absolute-path requirement;
- backend negotiation prefers verified RDMA without silently changing an
  active deployment; and
- scheduler and dashboard telemetry use the same request and runtime records.

The design should work for many models, but universal means capability-driven
admission and a safe refusal path. It does not mean optimistically sharding
every checkpoint.

## Goals

1. Run a qualified text model across two unequal Apple Silicon Macs through
   the normal oMLX API.
2. Keep rank-local weights and KV state within an admitted memory budget.
3. Give every rank an identical, authenticated execution plan while allowing
   different local model paths.
4. Prefer a measured JACCL/RDMA route and support an explicit, re-planned
   fallback when RDMA is unavailable.
5. Make load, ready, drain, unload, failure, and cancellation states accurate
   after client disconnects, server restarts, and partial rank failures.
6. Preserve prompt-cache reuse, tiered-cache policy, continuous batching, and
   per-request metrics in distributed mode.
7. Add new model families through small compatibility adapters and tests,
   without adding model names to the scheduler or UI.
8. Land the work as a sequence of bounded PRs with feature gates and fallback
   behavior, not as one product-and-engine change.

## Non-goals for the first beta

- Sharding vision towers, audio towers, cross-modal projectors, or other VLM
  components.
- Dynamic rank membership or re-sharding a live request.
- Continuing a request after a rank is lost.
- Pipeline parallelism, expert parallelism, or more than two ranks as a
  release claim.
- Mixed Metal/CUDA execution or a Ring/NCCL gateway.
- Treating CPU and ANE capacity as interchangeable with Metal tensor shards.
- Claiming linear or exponential speedup for every prompt, model, or phase.
- Automatically falling back to a local full-model load after a distributed
  request has been admitted.

These remain valid follow-up directions. They need their own correctness and
hardware gates.

## Current evidence

### Evidence boundary

The measurements below came from the Fusion reference branch, not upstream
`main`. They are point measurements on one two-Mac topology and are not product
guarantees. `API` is end-to-end timing observed by the client. `marker` is the
coordinator's engine marker. Cold prefill rows had zero cached prompt tokens.
Small API/marker differences are expected because their timing boundaries are
different.

Hardware and transport:

- rank 0: M3 Ultra Mac;
- rank 1: M5 Max Mac;
- two-rank JACCL over a directly connected Thunderbolt RDMA path;
- DeepSeek V4 Flash used an admitted asymmetric `3:5` tensor placement; and
- Qwen3-30B-A3B-4bit used equal `1:1` tensor placement with exactly
  `8,766,091,264` weight bytes admitted on each rank.

No number in these tables is an estimate unless the row explicitly says
`target` or uses an approximate sign.

### Measured reference results

<!-- markdownlint-disable MD013 -->

| Model | Path | Workload | API result | Engine marker | Qualification note |
| --- | --- | --- | ---: | ---: | --- |
| Qwen3-30B-A3B-4bit | non-MTP, B1 | 30K cold prefill | 1,533.73 tok/s | 1,537.15 tok/s | 0 cached tokens |
| Qwen3-30B-A3B-4bit | non-MTP, B1 | decode following 30K cold prefill | 56.83 tok/s | 56.46 tok/s | single request |
| Qwen3-30B-A3B-4bit | non-MTP, B1 | exact 30K repeat | TTFT 0.67 s | 30,004/30,005 prompt tokens cached | cold TTFT was 19.92 s; output hash matched |
| Qwen3-30B-A3B-4bit | non-MTP, B4 | 4 concurrent requests, 512 output tokens each | 216.3 tok/s aggregate | four request rows observed | 2,048 output tokens in 9.469 s |
| Qwen3-30B-A3B-4bit | MTP | not yet qualified | -- | -- | no result claimed |
| DeepSeek V4 Flash | non-MTP, B1 | 30K cold prefill | 841.51 tok/s | 846.90 tok/s | 0 cached tokens |
| DeepSeek V4 Flash | non-MTP, B1 | 100K cold prefill | 771.77 tok/s | 773.86 tok/s | 0 cached tokens |
| DeepSeek V4 Flash | non-MTP, B1 | 250K cold prefill | 621.79 tok/s | 622.74 tok/s | 0 cached tokens |
| DeepSeek V4 Flash | non-MTP, B1 | qualified decode runs | approximately 30-32 tok/s | approximately 30-32 tok/s | observed range, not one exact request |
| DeepSeek V4 Flash | fixed-depth-5 MTP, B1 | decode | 75.93 tok/s | approximately 70 tok/s | experimental MTP path |
| DeepSeek V4 Flash | non-MTP, B4 | qualified concurrent run | 69.3 tok/s aggregate | -- | aggregate result only |
| DeepSeek V4 Flash | prompt-cache repeat | cache reuse functionally qualified | -- | -- | no publishable cached-prefill throughput yet |

<!-- markdownlint-enable MD013 -->

`B1` means one active request and `B4` means four concurrent requests. The
Qwen B4 number is aggregate output throughput, not the speed of each request.
The Qwen exact-repeat row reports cache coverage and TTFT instead of inventing
a prefill throughput for work that was bypassed.

### Measured results versus performance targets

Targets guide optimization; they are not current results or Beta 1 correctness
requirements.

<!-- markdownlint-disable MD013 -->

| DeepSeek V4 Flash metric | Best qualified measurement | Optimization target | State |
| --- | ---: | ---: | --- |
| TP2 cold prefill at 30K | 841.51 API tok/s | at least 1,000 tok/s | below target |
| TP2 cold prefill at 100K | 771.77 API tok/s | at least 700 tok/s | target exceeded |
| TP2 cold prefill at 250K | 621.79 API tok/s | high 600s tok/s | below target |
| TP2 non-MTP B1 decode | approximately 30-32 tok/s | 50-80 tok/s | below target |
| TP2 fixed-depth-5 MTP B1 decode | 75.93 client tok/s | at least 130 tok/s | below target |
| TP2 B4 aggregate decode | 69.3 tok/s | not yet fixed | measured baseline |

<!-- markdownlint-enable MD013 -->

Qwen is included as a generality and product-path gate, not as evidence that
every model will have the same scaling curve. Its measured 1,533.73 tok/s cold
prefill and 56.83 tok/s single-request decode are therefore reference results,
not targets retroactively assigned to other models.

### Correctness and control-plane validation

The Qwen text-only run also exercised:

- automatic staging of 12 files (`17,190,783,781` bytes) in 13.5 seconds,
  measured at 1.27 GB/s;
- two-rank plan and placement agreement, readiness canary, unload, process
  restart, and reload without a stale loaded-model row;
- exact-repeat prompt-cache reuse with an identical output hash;
- client interruption during prefill, after which both ranks reported
  cancellation at 26,624 processed tokens and returned to zero active
  requests; and
- four simultaneous request IDs with distinct running/completed dashboard rows
  and aggregate throughput.

For six no-thinking, greedy Qwen prompts, distributed and standalone output was
token-identical on four. The arithmetic result and moving-average function in
the other two were semantically equivalent but worded differently. This is
functional parity, not bit-exact parity. Tensor reductions can change floating
point order, so release gates should test bounded numerical agreement and
same-plan repeatability in addition to text equality.

DeepSeek V4 planning and sharding prerequisites are already separated into
draft [PR #3106](https://github.com/jundot/omlx/pull/3106) and
[PR #3107](https://github.com/jundot/omlx/pull/3107). This RFC does not imply
that either PR has been accepted.

## Model support boundary

### Qualified text path

The first adapter matrix should include the exact checkpoint shapes and
quantization layouts used by:

- DeepSeek V4 Flash; and
- `mlx-community/Qwen3-30B-A3B-4bit`.

Qwen3-30B is important because it fits on one of the test machines. Running it
through TP2 demonstrates that staging, planning, execution, cache, cancel, and
telemetry are not tied only to a model that requires both machines. It does not
show that TP2 is the fastest decode strategy when the model fits locally; its
standalone decode was faster in this topology.

### Explicit Qwen3.8 VLM exclusion

The tested `Qwen3.8-27B-4bit` artifact is not a text-only checkpoint. Its config
uses `Qwen3_5ForConditionalGeneration` and its weights include a
`vision_tower`. Cluster v2 must reject it during distributed admission for the
first beta. Reclassifying it as text-only would risk incomplete loading or
incorrect multimodal outputs.

Qwen3.8 VLM support requires an explicit design for vision-tower ownership,
projector tensors, processor/config identity, multimodal cache state, request
routing, and parity tests with image inputs. It is future work, not a planner
allow-list change.

### Capability adapter contract

A model-family adapter should return either a complete capability description
or a structured refusal. The description includes:

- architecture and checkpoint-layout fingerprints;
- quantization method and group geometry;
- tensor axes that may be partitioned or must be replicated;
- head, expert, and vocabulary divisibility constraints;
- rank-local cache types and shapes;
- fixed weights and conservative working-memory estimates;
- supported backends, world sizes, and execution modes; and
- optional provider capabilities such as a qualified custom kernel.

The default result is unsupported. The engine, scheduler, and dashboard consume
the capability object; none should branch on a model display name.

## Architecture

### Control plane and data plane

The control plane owns trust, inventory, staging, planning, lifecycle, and
telemetry. Rank processes own tensors, KV state, collectives, and generation.
Separating them lets a failed rank terminate without leaving the API process or
dashboard convinced that a model is still resident.

```text
OpenAI-compatible client
          |
          v
oMLX coordinator and distributed scheduler
          |
          +-- trusted node registry
          +-- model manifest and path map
          +-- signed deployment plan
          +-- lifecycle and request journal
          |
          v
rank 0  <======= negotiated collective =======>  rank 1
weights + KV                                     weights + KV
```

Each runtime launch has a new `runtime_epoch`. Deployment intent may persist
across restarts, but a loaded state may not. Dashboard state comes from live
rank heartbeats for the current epoch, never solely from a saved deployment.

### Identity, discovery, and pairing

Discovery is an untrusted suggestion. Bonjour/mDNS and manual addresses may
provide candidates, but they cannot add a rank to a plan. Pairing establishes:

- a stable oMLX node ID;
- the peer's host/public-key identity;
- a short-lived bootstrap proof and a persistent trust record;
- a boot or process nonce used to reject stale status; and
- a versioned capability document for hardware, memory, oMLX, MLX, MLX-LM,
  protocol, transports, and model inventory.

The registry stores no login passwords or private keys. Identity changes and
protocol incompatibility fail closed. Removing a pairing invalidates future
plans but does not pretend that an active rank has stopped; teardown still
follows the lifecycle protocol.

### Model identity, path mapping, and staging

Absolute paths are rank-local details. The coordinator builds a bounded model
manifest from config, tokenizer/processor metadata, and weight headers. The
manifest has a content digest and records the files and tensor schema needed by
the selected adapter.

For each rank, staging then:

1. compares the requested manifest with its local inventory;
2. transfers only missing artifacts using a resumable temporary location;
3. validates file size and digest;
4. promotes the complete snapshot atomically; and
5. returns a rank-local `model_path` bound to the same model digest.

No rank may load a partial snapshot. Existing compatible files may be reused,
but path equality alone never proves model equality.

### Signed tensor-parallel plan

The planner consumes the model capability, requested context, cache policy,
usable memory, calibrated rank weights, and negotiated backend. Its canonical
payload includes:

- model and tokenizer/processor digests;
- adapter and protocol versions;
- rank IDs and runtime epoch;
- tensor assignments, replicated tensors, and rank-local paths;
- context, KV/cache geometry, concurrency limits, and memory reserves;
- backend, interfaces, addresses, and backend-specific options; and
- a plan digest and placement signature.

The coordinator signs the canonical envelope with its paired control-plane
identity. Each rank verifies that signature, recomputes the digest from the
payload, validates its local assignment against the model headers, and returns
an acknowledgement over the authenticated control channel. A plan becomes
loadable only after all ranks acknowledge the same digest.

The deterministic digest is also useful before signed envelopes land, but it
is an integrity fingerprint rather than proof of origin. Documentation and UI
must not call a bare digest a cryptographic signature.

### Backend selection and fallback

Backend selection is part of the plan, not a process-global guess.

1. Probe routes and transport capabilities from every rank.
2. Verify the actual RDMA interfaces and run a bounded collective canary.
3. Prefer JACCL only when every required rank passes the same route and backend
   contract.
4. Otherwise build and display a separate TCP Ring plan when it is supported.
5. Require all ranks to acknowledge the selected plan before starting.

A backend may not change under in-flight work. If JACCL fails after load, the
deployment enters failure/drain and tears down. A Ring retry requires a new
plan, new runtime epoch, and fresh rank acknowledgements. There is no silent
local-model fallback after request admission.

Backend diagnostics should report selected interfaces, route addresses,
collective canary results, and fallback reason without exposing credentials.

## Lifecycle, cancellation, and verified teardown

### Deployment state machine

```text
inactive
   |
   v
staging -> planned -> starting -> ready -> draining -> stopped
    |          |          |         |         |
    +----------+----------+---------+---------+--> failed
```

Only `ready` accepts generation. Transitions are monotonic within one runtime
epoch and idempotent when replayed. `failed` contains a rank-attributed reason;
it is not overwritten by the persisted deployment's desired state.

Unload is complete only after:

- request admission is closed;
- active requests finish or receive cancellation;
- every rank reports scheduler and KV release;
- child processes exit;
- the supervisor verifies no process for the runtime epoch remains; and
- live status contains zero loaded ranks.

A timeout produces a failed teardown with the unresolved rank or process IDs.
It must not produce a green `unloaded` row.

### In-flight cancellation

The coordinator assigns one request ID before dispatch. Client cancellation,
stream disconnect, stop from the dashboard, and administrative drain all enter
the same protocol:

1. mark the request `cancelling` and stop further token delivery;
2. broadcast an idempotent cancel message containing request ID, plan digest,
   and runtime epoch;
3. remove queued prefill/decode work and release rank-local transient state;
4. collect a terminal acknowledgement from every live rank; and
5. mark the request cancelled, or fail the deployment when a rank cannot
   acknowledge within the teardown policy.

Repeated cancellation is safe. A late token or acknowledgement from an older
runtime epoch is ignored. The dashboard reads this request record, so it cannot
show a request as running after the engine has accepted its cancellation.

## Cache and scheduling semantics

### Distributed prompt and tiered cache

KV remains rank-local for the layers owned by that rank. A reusable prefix is
identified by at least model digest, plan digest, tokenizer/chat-template
identity, adapter state, cache format, and token prefix. A cluster cache hit is
reported only when every rank confirms a compatible prefix. One-rank misses
fall back to the longest common prefix rather than reporting an impossible
full hit.

Hot-memory and cold/SSD tiers may use different local paths and capacities.
Promotion, eviction, and invalidation are coordinated by prefix identity, not
by copying all KV through the coordinator. Cache telemetry reports common
cached tokens, per-rank residency, bytes promoted/evicted, and the reason reuse
was refused.

### Concurrency and batching

Concurrency is the number of independently addressable client requests.
Batching is the scheduler's temporary coalescing of compatible work from those
requests. The API should expose normal concurrent requests; the scheduler may
form prefill or decode batches without requiring a client-side batching mode.

Fairness and latency limits remain explicit. A large prefill must yield at
bounded intervals so active decode streams continue to make progress. Cache
affinity, memory admission, and backend limits constrain batch formation.

Every request records:

- queue, staging, TTFT, prefill, and decode durations;
- prompt, cached, and generated token counts;
- per-request prefill and decode throughput;
- batch membership over time;
- plan digest, runtime epoch, and terminal reason; and
- optional rank-attributed stalls without treating rank-local rates as
  end-to-end throughput.

The dashboard derives aggregate throughput from these records while retaining
one row per request. Aggregate output tok/s must never be labelled as a single
request's decode speed.

## Dashboard and onboarding

The default flow should ask for intent, not distributed-systems terminology:

1. enable the experimental cluster feature;
2. discover or enter a second Mac;
3. authenticate pairing and verify runtime compatibility;
4. choose a downloaded text model and desired context;
5. stage missing artifacts;
6. let Automatic select a qualified strategy and backend;
7. review memory, rank placement, transport, cache, and compatibility warnings;
8. activate and wait for the readiness canary; and
9. use the existing oMLX model and API controls.

Advanced controls may expose tensor weights, reserves, backend, concurrency,
and cache budgets. Defaults come from measured capabilities and must always
show when they differ from a previously saved plan.

The live dashboard shows desired deployment separately from current runtime.
It includes runtime epoch, plan digest, model identity, rank liveness, loaded
bytes, cache state, active request rows, per-request rates, aggregate rates,
and the last failure or teardown result. A saved deployment with no live ranks
is inactive, not loaded.

## Failure policy

Cluster v2 follows four rules:

1. Refuse unsupported models before staging or allocation.
2. Fail the whole deployment when any required tensor rank is lost.
3. Never advertise ready, cancelled, or unloaded until the corresponding
   all-rank condition is verified.
4. Never hide a backend or distributed failure by loading a different local
   execution path for an already admitted request.

Retries create a new runtime epoch. Persisted intent may offer a one-click
retry, but old heartbeats, metrics, cancellation acknowledgements, and process
records cannot satisfy the new launch.

## Staged upstream PR series

Each implementation PR should keep distributed inference behind the existing
experimental gate and include unit tests for its refusal/fallback behavior.

<!-- markdownlint-disable MD013 -->

| Stage | Review scope | Required proof |
| --- | --- | --- |
| 0 | This RFC plus the narrow provider and DS4 planner prerequisites in PRs #3106 and #3107 | documentation review; existing focused prerequisite tests |
| 1 | Lifecycle supervisor, runtime epochs, request IDs, cancellation broadcast/ack, drain, and verified teardown | simulated rank loss, disconnect, repeated cancel, restart, orphan process, and stale-state tests |
| 2 | Stable node identity, discovery suggestions, authenticated pairing, boot nonce, and versioned capabilities | replay, expiry, identity-change, unpaired-discovery, and protocol-mismatch tests |
| 3 | Generic model capability adapter, bounded manifest, signed TP plan, path-independent assignments, and Qwen3 text adapter | exact-header fixtures, signature/digest disagreement, memory refusal, Qwen standalone/TP numerical parity, and Qwen3.8 VLM rejection |
| 4 | Resumable content-addressed staging and per-rank path mapping | partial transfer, corruption, existing-file reuse, atomic promotion, and different-local-path tests |
| 5 | JACCL/RDMA auto-configuration, collective canary, explicit Ring re-plan, and failure policy | route/interface fixtures, backend mismatch, canary failure, new-plan fallback, and no silent local fallback tests |
| 6 | Distributed prompt/tiered cache, common-prefix reuse, concurrent scheduler integration, and per-request telemetry | exact repeat, partial-rank miss, eviction, B1/B4 fairness, cache isolation, and aggregate-versus-request metric tests |
| 7 | Onboarding wizard, deployment/runtime dashboard, compatibility explanations, and Beta 1 release gates | two-Mac clean install, restart/reload, stale-unload, cancel, cache, batching, accessibility, and signed-build smoke matrix |

<!-- markdownlint-enable MD013 -->

Stages may be split further when a diff mixes protocol, engine, and UI changes.
No stage should depend on copying the full Fusion branch into upstream.

## Beta 1 qualification matrix

The first beta may be called cluster-ready only when all applicable rows pass
on the release commit.

<!-- markdownlint-disable MD013 -->

| Area | Required gate |
| --- | --- |
| Build/runtime | same released oMLX, MLX, MLX-LM, and cluster protocol on both Macs |
| Model admission | exact qualified text checkpoint accepted; unknown and VLM checkpoints refused with an actionable reason |
| Planning | identical signed plan and model digest acknowledged by both ranks; memory reserve maintained |
| Transport | JACCL/RDMA route and collective canary pass, or an explicitly displayed Ring re-plan passes |
| Correctness | rank-local shard fixtures match reference numerically; same-plan repeat is deterministic; standalone/TP model outputs pass the declared tolerance suite |
| Lifecycle | load, ready, cancel, drain, unload, server restart, and reload pass with no stale rank or process state |
| Cache | cold request reports zero cached tokens; exact repeat reports the common reused prefix and matching output |
| Concurrency | B1 and B4 complete without request identity loss; per-request and aggregate rates remain distinct |
| Observability | both dashboards agree on epoch, plan, ranks, cache, requests, throughput, and terminal state |
| Packaging | signed/notarized build installs on a clean Mac and the two-node smoke test uses that artifact |

<!-- markdownlint-enable MD013 -->

Performance targets are tracked alongside this matrix, but missing a stretch
target must not be disguised as a correctness failure or a measured result.

## Future work

- Additional text-family adapters and a public compatibility conformance kit.
- Qwen3.8 and other VLM tensor plans with image-input parity.
- More than two ranks, dynamic topology selection, and heterogeneous tensor
  weights qualified per family.
- Pipeline, expert, and hybrid parallel strategies chosen by measured workload.
- Mixed Metal/CUDA Ring and a hierarchical Ring/NCCL gateway.
- MTP/speculative heads generalized beyond the qualified DeepSeek path.
- Recovery from failed workers through checkpointed request state, if it can be
  proven correct and faster than an explicit retry.
- Safe CPU/ANE auxiliary kernels that preserve the same capability and fallback
  contracts.

## Open questions

1. Should the coordinator use its paired SSH identity to sign plan envelopes,
   or should cluster v2 generate a separate scoped Ed25519 control-plane key?
2. Which numerical/logit tolerances should the public adapter conformance kit
   require when collective reduction order prevents token-exact parity?
3. Is TCP Ring part of Beta 1, or should the initial release require verified
   JACCL and leave Ring behind a separate experimental gate?
4. Which model families beyond DeepSeek V4 Flash and Qwen3 should be mandatory
   before the feature is described as broadly available rather than beta?
5. Should tiered-cache metadata be coordinated by the deployment supervisor or
   by a smaller rank-owned cache protocol with the supervisor observing only
   common-prefix results?
