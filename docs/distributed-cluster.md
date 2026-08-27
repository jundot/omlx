# Distributed inference across Macs

Status: Cluster v2 Beta candidate (source build; release qualification remains
mandatory)

oMLX can run one downloaded MLX model across a capability-matched group of
Macs while preserving its existing OpenAI-compatible API. Automatic planning
chooses tensor parallelism for supported models and fast links, contiguous
pipeline stages for capacity-oriented or non-TP models, or a capability-gated
TP x pipeline factorization. An explicit, default-off **Phase split** mode can
instead load two complete replicas and overlap prefill on rank 1 with decode
and API work on rank 0. Rank zero remains the API coordinator; it never
silently loads a second full local model when a distributed launch fails.

Experimental Mac + NVIDIA execution is available through the outer MLX Ring
compatibility path. See the [heterogeneous model-pool guide](heterogeneous-cluster.md)
for the logical Metal/CUDA memory pool, automatic placement, GUI worker
enrollment, hardware gates, and the still-pending hierarchical Ring/NCCL
gateway.

The implementation currently provides:

- stable node identities plus untrusted Bonjour, IPv6 multicast, manual-IP,
  persisted-pair and opportunistic Tailscale control-plane discovery;
- read-only Thunderbolt, RDMA interface, IP, route, memory, power, runtime,
  kernel and JACCL fingerprint probes;
- consent-based pairing, persistent 0600 device records, signed short-lived
  enrollment, automatic key exchange and changed-host-key refusal;
- GUI-generated, ten-minute, single-use CUDA worker enrollment with pinned
  bootstrap/source digests and pinned SSH identities;
- prompt-free SSH trust-on-first-use: new peer aliases are recorded in the
  user's `known_hosts`, while changed keys are still refused;
- exact oMLX, MLX, MLX-LM, cluster-protocol, per-node model-path, import and
  bounded model-manifest preflight (config/tokenizer/processor metadata plus
  weight headers);
- unioned model inventory, resumable staging, per-node path mapping, complete
  sidecar copying and safetensors-header planning across unequal budgets;
- architecture-aware tensor sharding, progressive post-slice materialization,
  load-time memory guards and signed per-rank residency verification, so a TP
  rank retains its shard rather than a full in-memory checkpoint;
- bounded per-rank MLX compute and collective calibration, followed by
  performance-aware shard rebalancing when every rank reports valid results;
- Ring, JACCL, and JACCL Ring launch through MLX's official launcher;
- isolated rank processes, deterministic plan agreement, and hard group teardown;
- rank-local shard loading and KV caches, including Nemotron-H hybrid-cache support;
- interactive, balanced, and throughput execution profiles with headroom-aware
  concurrency, prefill, coalesced-batch, prompt-cache, and KV limits;
- native MLX-LM asynchronous next-token dispatch, multi-connection Ring tuning,
  cache affinity, and a capability-gated experimental token-only output path;
- completion, chat, Responses, Anthropic Messages, streaming, usage, targeted
  disconnect cancellation, bounded failure propagation and orphan recovery
  through the normal oMLX engine interface;
- rank-unanimous hot/SSD prompt snapshot reuse and all-rank cache clearing;
- model-independent full-replica prefill/decode cache handoff for compatible
  text models, with signed ownership, exact/nearest-prefix reuse, optional
  persistent snapshots, chunk-boundary cancellation and measured RDMA handoff;
- a Cluster dashboard on every Mac with a full live shard map, local-rank
  highlighting, memory headroom, rank-local KV ownership, TTFT, prefill tok/s,
  per-request and aggregate decode tok/s, pipeline utilization, prompt-cache hit
  rate, measured collective bandwidth, predicted stage time, active requests,
  and cumulative token counts.

oMLX does not enable RDMA without approval, overwrite changed SSH host keys, or
install login credentials without pairing. Those remain explicit administrator
actions. A new hostname or link address is recorded using OpenSSH's
``accept-new`` policy so setup never pauses for a terminal prompt.

## Architecture

```text
OpenAI client
     |
     v
oMLX API + tokenizer/chat template (coordinator Mac)
     |
     v
DistributedBatchedEngine
     |
     v
private rank-0 MLX-LM HTTP endpoint (127.0.0.1, random port)
     |
     v
rank 0 tensor shard ───────── Thunderbolt RDMA / JACCL ───────── rank 1 tensor shard
 every layer + local KV                                         every layer + local KV
```

For a pipeline plan, replace each tensor shard above with its signed contiguous
layer range. For a hybrid plan, each stage owns a layer range and each rank
inside that stage owns one tensor shard of those layers.

For **Phase split**, both Macs load the complete model. Rank 1 processes the
prompt and sends the lossless MLX-LM cache state and final prefill logits over
the selected fabric; rank 0 reconstructs the cache and owns sampling, decode and
the private API. While rank 0 decodes request N, rank 1 can prefill request N+1.
This improves queued throughput rather than making two Macs compute one prompt.
The mode is admitted only when weights plus the requested KV reservation fit on
both Macs and every cache state has a validated transfer contract.

The cluster runtime lives outside oMLX's main MLX scheduler process. This keeps
the existing API adapters and model lifecycle intact while allowing MLX-LM to
own its distributed batch generator and prompt cache. A launcher or rank
failure tears down the job; oMLX never silently falls back to a local full-model
load.

KV cache stays on the rank that owns the corresponding layers. Centralizing KV
on one Mac would add a network read/write to every layer and generated token,
so it is not the default.

Automatic topology selection considers every executable factorization of the
selected Macs. With four compatible Macs, for example, it can choose pure TP4,
hybrid TP2 × pipeline-2, or pipeline-4. In a hybrid plan each pipeline stage
owns a dynamically balanced contiguous layer range, and every Mac inside that
stage holds only its tensor shard of those layers. The signed plan and active
dashboard show both coordinates; no rank loads a full checkpoint copy.

Hybrid Beta currently requires one verified collective backend across the
whole world. A future nested-fabric mode can use JACCL inside fast TP groups
and a TCP pipeline edge between groups, but oMLX does not silently claim that
mixed-backend topology today. Automatic hybrid selection also remains hidden
unless the loaded MLX runtime explicitly reports backend subgroup support;
stock Ring and stock JACCL builds do not implement `Group.split`.

oMLX currently admits one live JACCL communicator per Mac. Starting a second
distributed model or an auto-tuning probe while another communicator owns the
same Thunderbolt RDMA device is rejected with an actionable 409. This fence is
temporary but intentional: physical testing reproduced lost RDMA completions
with concurrent communicators, while the same subgroup operations passed
cleanly after the resident communicator was stopped.
Each inference rank and synthetic performance worker also holds a kernel-owned
`flock` lease for the device. The lease releases automatically on crash or
SIGKILL, so it prevents cross-process races without creating a stale-lock
recovery problem.

## Requirements

On every Mac:

1. Run the same oMLX build and matching MLX/MLX-LM versions.
2. Make the model available on each selected Mac. Paths may differ; the signed
   deployment carries a per-node path map and staging copies missing files.
3. Enable Remote Login and use key-based SSH for the coordinator account.
4. Pair the Macs in oMLX. The first connection records a new hostname or
   Thunderbolt address without prompting; an identity change is still refused.
5. For JACCL, configure Thunderbolt RDMA outside oMLX and confirm `rdma_ctl
   status` and `ibv_devices` report the link.

The device card reports the **control/discovery route**. Seeing “Tailscale
control” there does not mean tensor traffic is using Tailscale. For an active
fast-path deployment, the dashboard must separately report **Inference: JACCL
over Thunderbolt RDMA**; otherwise it reports the TCP ring fallback.

If you are not on macOS 27 and `rdma_ctl status` reports RDMA disabled, enable
it once on every Mac from macOS Recovery: shut down, hold the power button,
choose Options, open **Utilities > Terminal**, run `rdma_ctl enable`, and
restart. Reconnect the Thunderbolt cable and verify both `rdma_ctl status` and
`ibv_devices` before starting the cluster. oMLX does not attempt this
Recovery-only change remotely.

Rank zero is the Mac whose dashboard activates the deployment and owns the
private inference coordinator. Its layer/tensor ownership comes from the
signed plan; do not infer ownership from RAM size or rank number.

For an Ubuntu/Debian CUDA worker, no oMLX desktop installation is required.
Use **Cluster > Add a CUDA worker** on the coordinator and paste its generated
command into the Linux account the worker should use. The installer creates a
minimal environment at `/opt/omlx-cluster-worker/venv`, verifies it, and adds
the worker to the pool. Use one newly generated command per physical box.

## Use the GUI

Start this source build on both Macs. In **Settings > Advanced**, enable
**Distributed Inference**, save, and restart oMLX. The **Cluster** tab, cluster
API routes, and Bonjour advertisement remain off until this explicit opt-in is
enabled.

### Automatic Peer Discovery


oMLX combines Bonjour/mDNS with an IPv6 multicast fallback designed for
multi-interface and Thunderbolt Macs. Persisted paired addresses are probed at
the fast heartbeat cadence, and Tailscale addresses may recover the reliable
control channel when macOS denies ordinary application TCP on the direct
Thunderbolt subnet. Tailscale never substitutes for the signed JACCL/RDMA data
path. Discovery is untrusted and never implies pairing; **Add by IP** remains
available when local-network permission or multicast is unavailable.

Pairing presents the joining Mac and a short-lived approval code. Approval
exchanges the cluster key and SSH identity through authenticated endpoints;
wrong, expired, replayed or altered tokens are rejected. The legacy manual key
exchange remains available only as an advanced recovery path.

### CUDA Worker Enrollment

The CUDA card is the normal Linux path; the older two-dashboard key exchange is
only for peer Macs. The coordinator must listen on a LAN-reachable address. If
the dashboard URL uses localhost, set **Settings > Server host** to `0.0.0.0`,
restart oMLX, and enter the Studio's private IPv4 address in the card.

Select **Generate join command**, copy it, and paste it into one CUDA worker. The
command expires after thirty minutes and can be claimed only once. It may ask for
`sudo`; package installation, the worker-only virtual environment, SSH key
exchange, source verification, live imports, and pool selection happen
automatically. Generate a fresh command for the second CUDA worker. No join
credential is stored
in browser storage or in the completed node registry.

The current execution mode still launches both CUDA boxes as physical ranks in
the outer Ring. A successful ConnectX/NCCL verification keeps the CUDA pair
adjacent and uses its direct-link addressing, but does not yet turn it into the
future one-gateway hierarchical Ring-to-NCCL supernode.

### Setup Flow


On the coordinator:

1. Connect the Thunderbolt cable and open **Cluster**. Nearby Macs appear
   automatically; otherwise use **Add by IP**.
2. Pair the Mac and approve its displayed identity. The wizard establishes the
   enrolled SSH control path without asking for an inference password.
3. Let the automatic checks verify reachability, exact versions, model/import
   readiness, Thunderbolt and RDMA. If RDMA is disabled, follow the displayed
   Recovery instructions; choosing TCP ring remains explicit.
4. Select any model in the unioned cluster inventory. Missing checkpoint and
   sidecar files are staged before activation, including different per-node
   paths.
5. Leave strategy on **Automatic** or choose Tensor/Pipeline. For a compatible
   two-Mac text model, **Phase split** is also available as an explicit
   experimental choice. Automatic inspects
   model capabilities, per-node memory and measured fabric, then returns a
   signed shard recommendation. Unequal tensor vectors require exact persisted
   parity qualification; otherwise the plan safely uses equal tensor shards.
6. Review every rank's layer range, tensor share, memory, context reservation,
   backend and cache policy, then activate. The normal dashboard and APIs use
   the clustered model; no separate inference endpoint is required.



Both dashboards show the complete rank-to-layer map while highlighting the
shard resident on that Mac. A 256 GiB Mac can therefore own more layers than a
128 GiB Mac; no 50/50 split is required. The planner uses the actual byte size
of each transformer layer and each Mac's usable capacity, so layer counts can
also differ when the model cannot be divided evenly.

TTFT, prefill tok/s, and decode tok/s are end-to-end pipeline measurements.
They describe the cooperating cluster, not independent per-rank speeds.
Layer range, planned weights, headroom, and KV ownership remain rank-specific.
Activation records the approved deployment. Loading from either the normal
model control or the first request starts its signed ranks; a one-action replan
quiesces, unloads, replaces the signed placement and reloads.

The active card follows the live runtime owner when several signed deployments
exist. It can unload resident weights while preserving the setup, reload them
with a readiness canary, or drain/remove the setup and reopen the model picker.
Model staging/synchronization runs before activation for Phase split exactly as
for TP and pipeline, so each full-replica owner must hold a complete manifest.
The ordinary **Clear SSD Cache** action clears active rank stores through their
cache managers. If the deployment is already unloaded, it uses the existing
enrolled SSH policy to remove the same validated cluster snapshot and legacy
roots from every configured peer; an unreachable peer makes the clear fail
visibly instead of leaving a silent restore behind.

### Phase split boundaries

The current Beta contract is deliberately narrow and fail-closed:

- exactly two Macs, with rank 1 prefill and rank 0 decode/API;
- text requests only; VLM media inputs are rejected;
- no MTP/speculative decode or guided grammar state across the handoff;
- both Macs must fit complete weights and the full requested KV reservation;
- cache types are universal by serialization contract, not by model name: every
  state leaf must be an MLX array and the installed cache class must implement
  `from_state`; unfamiliar state fails the readiness canary;
- DS4 Flash does not fit as two full replicas on the 128 GB reference M5 and
  therefore stays on TP. A future phase-shard-group topology is separate work.

Deactivation prevents future distributed loads. An already-loaded engine
continues until the normal unload lifecycle so an admin click cannot interrupt
an in-flight request.

## Performance system

The activation benchmark runs a small, bounded MLX matrix workload on every
rank and measures a small-message collective plus a 1 MiB collective over the
selected backend. Compute results are stored as relative calibration signals,
not advertised model throughput. The planner combines decode and prefill
signals according to the selected profile, estimates each contiguous stage's
compute and activation-send time, and minimizes the slowest stage without ever
crossing a node's memory budget.

The benchmark is fail-soft. Missing ranks, non-finite values, timeouts, or
launcher failures discard all measurements and retain the already-verified
memory-only plan. A partial or stale measurement can never produce a
performance plan. The final plan hash includes the selected workload,
microbatch target, measurements, and exact layer ranges, and every worker
validates it before loading.

The three profiles provide conservative starting points:

| Profile | Decode concurrency | Prompt concurrency | Prefill step | Coalesced target | Ring connections/IP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Interactive | 4 | 2 | 1,024 | 2 | 1 |
| Balanced | 8 | 4 | 2,048 | 4 | 2 |
| Throughput | 16 | 8 | 4,096 | 8 | 4 |

Auto-tuning reduces these values when the smallest stage has limited
headroom and bounds the MLX-LM prompt-cache budget. The coalesced target caps
MLX-LM's continuous prefill and completion batches; it is not a claim of a
new 1F1B pipeline scheduler. A rotating-KV token limit is optional and remains
blank by default so full context is preserved.

MLX-LM's pinned generation path already dispatches the next token with
`mx.async_eval`; oMLX capability-checks and reports that path in the live view.
Prompt-cache affinity keeps requests for the deployed model on the same
persistent rank set, allowing each rank's local cache to be reused.
Multi-connection tuning applies only to the TCP Ring backend. JACCL owns its
Thunderbolt RDMA connection strategy and never receives Ring-only flags.

Every distributed rank has a bounded in-memory MLX-LM prompt cache. The
cluster planner's **Persistent prompt reuse** option adds a second,
rank-local SSD snapshot tier: each rank saves its own layer state at aligned
prefill boundaries, and a later request restores a boundary only when every
rank reports the matching snapshot. Snapshot directories are scoped by
deployment, signed plan, and rank, so a changed layer or tensor split cannot
restore incompatible state. Writes run in a bounded FIFO behind prefill; they
never block on a saturated queue, and only committed files participate in
cross-rank restore votes. Detached pending state is capped at two writes and
512 MiB per rank, so larger long-context snapshots are safely skipped rather
than risking an OOM. Leaving it off does not disable the in-memory prompt
cache. The active-cluster card and runtime-cache row report which mode is
actually running.

**Experimental token-only output** is opt-in. For a model whose pipeline
forward path matches the pinned, validated contract, oMLX removes the final
hidden-state all-gather, samples on rank zero, and all-sums only the selected
token IDs so every rank advances the same local KV state. If source inspection
does not prove that exact contract, normal all-gather remains active. Seeded
single-request generation is rejected while this experimental mode is selected
because MLX-LM routes seeded requests outside its continuous-batch path.

These controls improve scheduling and remove avoidable communication, but the
decode latency of a pipeline still includes every stage and inter-stage send.
The live view therefore shows both predicted stage time and observed
end-to-end measurements so a poor cut, cache miss, or slow link is visible.

## Diagnostics

```bash
omlx cluster status --json
omlx cluster status --route-to 169.254.42.2
omlx cluster worker-smoke
omlx cluster collective-smoke
omlx cluster pipeline-smoke
```

`collective-smoke` is a two-rank loopback all-sum. `pipeline-smoke` executes a
small, real, unequal two-rank hybrid Nemotron-H graph and verifies both ranks
produce the same result. Neither proves the physical Thunderbolt path.

Plan a model before activating it:

```bash
omlx cluster plan \
  --model /absolute/path/to/model \
  --node studio=256GiB \
  --node mobile=128GiB \
  --reserve 8GiB \
  --json
```

Only safetensors headers are read. Fixed weights such as embeddings and the
language-model head are conservatively accounted on every rank. The plan
contains a SHA-256 digest checked by every worker before loading.

## Admin API

All cluster endpoints use the existing oMLX admin authentication:

```text
GET    /admin/api/cluster/status
GET    /admin/api/cluster/runtime
GET    /admin/api/cluster/discover
POST   /admin/api/cluster/peer-probe
POST   /admin/api/cluster/worker-smoke
POST   /admin/api/cluster/collective-smoke
POST   /admin/api/cluster/pipeline-smoke
POST   /admin/api/cluster/plan
POST   /admin/api/cluster/join-keys
GET    /admin/api/cluster/join-status
DELETE /admin/api/cluster/join-keys/{join_id}
GET    /admin/api/cluster/deployments
POST   /admin/api/cluster/deployments
DELETE /admin/api/cluster/deployments/{deployment_id}
```

Deployment records contain hostnames, communication IPs, RDMA device names,
assignments, and the plan hash. They never contain passwords, private keys, or
SSH options. The registry is written atomically with mode `0600`.

The bootstrap transport endpoints live under `/cluster/join`. They do not use
the browser admin cookie: `/claim` consumes the one-time bearer key, while
`/source` and `/complete` require the resulting short-lived session. The
bootstrap program itself is public only while Distributed Inference is enabled
and is sent with no-store headers; its exact digest is embedded in the
authenticated admin command before it is executed.

## Current compatibility

- Pipeline-compatible text models use the upstream MLX-LM pipeline loader.
  Every worker verifies after loading that its exact approved unequal range is
  resident and fails closed if a model-specific pipeline hook ignored the
  plan.
- Nemotron-H receives a worker-local compatibility hook for its hybrid
  Mamba/attention cache layout and is covered by the real two-rank pipeline
  smoke test.
- Distributed MTP is qualified only for pure-TP DeepSeek V4/DSpark deployments.
  DFlash, SpecPrefill, VLM MTP, TurboQuant KV, guided grammar and `logit_bias`
  are rejected rather than ignored when their distributed contract is absent.
- The inventory, wizard, schema, planner, launcher and runtime accept multiple
  peers. Adding or removing a Mac while a model is resident performs a signed
  replan and reload; live elastic resharding is not a Beta claim.
- Qwen3 text, the Qwen3.8 text backbone and DeepSeek V4 have physical TP2
  evidence. A unified VLM may explicitly expose a text-backbone-only contract;
  image, video and audio input then fails with an actionable 400 rather than
  being silently discarded.
- Performance calibration is synthetic and intended for relative partitioning.
  Validate final throughput with the real model and workload.
- Cross-host JACCL/TB5 TP2 is physically validated on an M3 Ultra plus M5 Max.
  Repository tests still do not make a new hardware topology proven: every
  additional device count, model and hybrid TP x pipeline layout needs its own
  physical gate.

## Verification

Run the cluster suite with the same Python environment used by oMLX:

```bash
python -m pytest \
  tests/test_cluster_*.py \
  tests/test_distributed_engine.py \
  -q

ruff check \
  omlx/cluster \
  omlx/engine/distributed.py \
  tests/test_cluster_*.py \
  tests/test_distributed_engine.py
```

Before describing JACCL as hardware-validated, record all of the following on
the two target Macs:

1. exact oMLX, MLX, MLX-LM, Python, and macOS versions;
2. Thunderbolt port/speed and `rdma_ctl`/`ibv_devices` output on both nodes;
3. route-to-peer interface on both nodes;
4. JACCL collective smoke over the direct link;
5. small pipeline model load, streamed generation, disconnect cancellation,
   forced rank failure, teardown, and restart;
6. performance-probe repeatability, measured shard cut, collective bandwidth,
   and comparison against the memory-only cut;
7. the target large model's per-rank resident memory, TTFT, prefill throughput,
   single-stream decode, concurrent aggregate decode, cache hit rate, pipeline
   utilization, and long-context KV growth.
