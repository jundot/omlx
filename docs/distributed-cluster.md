# Distributed inference across Macs

Status: experimental, source-build preview

oMLX can run one downloaded MLX model across two unequal-memory Macs while
preserving its existing OpenAI-compatible API. The first implementation uses
contiguous pipeline stages: each rank loads only its assigned transformer
layers, while rank zero remains the API coordinator.

Experimental Mac + NVIDIA execution is available through the outer MLX Ring
compatibility path. See the [heterogeneous model-pool guide](heterogeneous-cluster.md)
for the logical Metal/CUDA memory pool, automatic placement, GUI worker
enrollment, hardware gates, and the still-pending hierarchical Ring/NCCL
gateway.

The implementation currently provides:

- read-only Thunderbolt, RDMA interface, IP, route, memory, and runtime probes;
- untrusted Bonjour suggestions for Macs advertising SSH;
- GUI-generated, ten-minute, single-use CUDA worker enrollment with pinned
  bootstrap/source digests and pinned SSH identities;
- prompt-free SSH trust-on-first-use: new peer aliases are recorded in the
  user's `known_hosts`, while changed keys are still refused;
- exact oMLX, MLX, MLX-LM, cluster-protocol, remote model-path, and bounded
  model-manifest preflight (config/tokenizer metadata plus weight headers);
- safetensors-header planning across unequal memory budgets;
- bounded per-rank MLX compute and collective calibration, followed by
  performance-aware shard rebalancing when every rank reports valid results;
- Ring, JACCL, and JACCL Ring launch through MLX's official launcher;
- isolated rank processes, deterministic plan agreement, and hard group teardown;
- rank-local shard loading and KV caches, including Nemotron-H hybrid-cache support;
- interactive, balanced, and throughput execution profiles with headroom-aware
  concurrency, prefill, coalesced-batch, prompt-cache, and KV limits;
- native MLX-LM asynchronous next-token dispatch, multi-connection Ring tuning,
  cache affinity, and a capability-gated experimental token-only output path;
- completion, streaming, usage, disconnect cancellation, and error propagation
  through the normal oMLX engine interface;
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
MLX pipeline group ───────── Thunderbolt RDMA / JACCL ───────── rank 1
  late layers + KV                                             early layers + KV
```

The cluster runtime lives outside oMLX's main MLX scheduler process. This keeps
the existing API adapters and model lifecycle intact while allowing MLX-LM to
own its distributed batch generator and prompt cache. A launcher or rank
failure tears down the job; oMLX never silently falls back to a local full-model
load.

KV cache stays on the rank that owns the corresponding layers. Centralizing KV
on one Mac would add a network read/write to every layer and generated token,
so it is not the default.

## Requirements

On every Mac:

1. Run the same oMLX build and matching MLX/MLX-LM versions.
2. Keep the downloaded model at the same absolute path.
3. Enable Remote Login and use key-based SSH for the coordinator account.
4. Pair the Macs in oMLX. The first connection records a new hostname or
   Thunderbolt address without prompting; an identity change is still refused.
5. For JACCL, configure Thunderbolt RDMA outside oMLX and confirm `rdma_ctl
   status` and `ibv_devices` report the link.

Rank zero is the Mac whose dashboard activates the deployment. It owns the
late pipeline layers and the private inference coordinator. For a 256 GiB Mac
paired with a 128 GiB Mac, rank zero should normally be the larger machine.

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


oMLX uses Bonjour/mDNS to discover nearby Macs advertising SSH or the oMLX
specific `_omlx._tcp` service. The discovery is read-only and never implies
trust. Discovered peers appear under **Detected nearby** with their hostname
and service type.

For manual pairing, generate a shared pairing secret on one Mac and copy it to
the other. Enter the same secret on both dashboards before generating and
exchanging the short-lived SSH key tokens. The secret authenticates the token
with HMAC-SHA256; an unkeyed or altered token is rejected.

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

1. Connect the Thunderbolt cable. A nearby Mac should appear under
   **Detected nearby** or via the QR code pairing.
2. Select the peer or enter its SSH hostname. **Check peer** records a new host
   alias automatically, refuses a changed key, and requires a non-interactive
   SSH login key.
3. Select **Downloaded model**, choose its local directory, set a reserve for
   KV/activations, and build the unequal plan.
4. Select JACCL, JACCL Ring, or the TCP Ring fallback and choose an execution
   profile. Leave **Auto benchmark & tune** enabled to calibrate both Macs and
   the selected link before the final shard plan is stored.
5. Review the final measured shard map, then activate.
6. Load or request that model through the normal oMLX API. If it was already
   loaded locally, unload it first so the new deployment applies.



Both dashboards show the complete rank-to-layer map while highlighting the
shard resident on that Mac. A 256 GiB Mac can therefore own more layers than a
128 GiB Mac; no 50/50 split is required. The planner uses the actual byte size
of each transformer layer and each Mac's usable capacity, so layer counts can
also differ when the model cannot be divided evenly.

TTFT, prefill tok/s, and decode tok/s are end-to-end pipeline measurements.
They describe the cooperating cluster, not independent per-rank speeds.
Layer range, planned weights, headroom, and KV ownership remain rank-specific.
Activation is lazy: it records an approved deployment and starts ranks when
oMLX next loads that model.

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
- DFlash, SpecPrefill, MTP, VLM MTP, TurboQuant KV, thinking budgets, guided
  grammar, and `logit_bias` are rejected for a distributed deployment rather
  than ignored.
- The first GUI activation flow intentionally supports two Macs. The schema,
  planner, launcher, and runtime support up to 64 ranks; a multi-peer GUI is
  follow-up work.
- Performance calibration is synthetic and intended for relative partitioning.
  Validate final throughput with the real model and workload.
- Cross-host JACCL/TB5 execution must still be validated on physical hardware.
  The repository tests do not claim that a loopback run proves RDMA.

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
