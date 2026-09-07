# One heterogeneous MLX + CUDA model pool

Status: outer-Ring compatibility path implemented; hierarchical gateway planned

Date: 2026-08-10

## Decision

oMLX should support one model sharded across a mixed pool of Apple Silicon and
NVIDIA CUDA nodes. The Mac runs MLX on Metal, the DGX Spark runs MLX on CUDA,
and the deployment exposes one logical model pipeline. Ordinary workers join a
common MLX TCP Ring. A verified group of CUDA workers may instead sit behind
one Ring gateway as a composite stage and use NCCL over ConnectX internally.
From the operator's perspective, the coordinator exposes one cluster, one model
catalogue, one aggregate memory budget, and one OpenAI-compatible endpoint.

This is the primary heterogeneous architecture. Prefill/decode disaggregation
is an optional optimization for spare capacity, not the mechanism that creates
the large pool.

The first implementation should extend the current oMLX unequal pipeline:

1. discover Metal and CUDA nodes;
2. verify that they have compatible MLX, MLX-LM, model, tokenizer, and cache
   contracts;
3. benchmark compute, memory headroom, and the links between them;
4. collapse verified high-speed CUDA groups into logical planner units;
5. allocate contiguous model layers to physical ranks or composite stages; and
6. launch the outer Ring plus any approved CUDA-local NCCL groups.

The existing planner, rank-local loading, KV ownership, memory guards, engine
proxy, and lifecycle supervision are the right foundations. The platform and
packaging assumptions around them must be generalized.

## What “one unified pool” means

The machines do not become hardware-coherent unified memory. Each machine keeps
its own physical memory, and oMLX presents a **logical model pool** by placing a
different part of the model in each rank's local memory.

```text
one model
  embeddings / first layers       -> rank 0, Metal or CUDA
  next contiguous layer range     -> rank 1, Metal or CUDA
  next contiguous layer range     -> rank 2, Metal or CUDA
  ...
  final layers / head              -> coordinator rank
```

Weights are not duplicated across the whole pool, apart from model components
that the MLX pipeline loader must replicate on each rank. KV cache remains with
the rank that owns its layers. Activations cross the network as inference moves
through the pipeline.

For the screenshot topology, one 256 GB M3 Ultra plus five 128 GB DGX Sparks is
896 GB of installed memory. The usable model working set will be lower because
every node needs operating-system, runtime, activation, and KV headroom, and
some fixed model weights may be replicated. The dashboard should therefore
show all three values:

- installed aggregate memory;
- aggregate memory admitted to oMLX; and
- the planner's actual maximum model working set.

That last value is the honest answer to “what size model can this pool load?”

For a model such as GLM 5.2, the catalogue must read the downloaded manifest
and safetensor headers instead of relying on a product-page parameter count or
a hard-coded 300-400 GB estimate. If the measured resident requirement is, for
example, 380 GiB and the pool admits 850 GiB, Automatic chooses the smallest
fast measured set that fits weights, the requested KV cache, activations, and
load headroom. **Use all eligible memory** keeps every compatible unit in the
placement. Advanced shard targets let the operator request more layers on a
particular Mac, CUDA worker, or CUDA supernode; they are soft targets that the
contiguous-layer and physical-memory checks may adjust or refuse.

## Current platform reality

This design no longer depends on treating CUDA support as an unofficial MLX
experiment. [MLX 0.32.2 officially supports CUDA 12 and CUDA 13](https://ml-explore.github.io/mlx/build/html/install.html),
including Linux ARM wheels relevant to DGX Spark. oMLX already pins
`mlx==0.32.2` in this branch.

MLX provides several distributed transports:

- **Ring** uses TCP sockets and is the common backend available to Metal and
  CUDA ranks. It is the first mixed-cluster transport.
- **JACCL** uses Thunderbolt RDMA and remains specific to compatible Macs.
- **NCCL** is the high-performance CUDA transport and remains specific to CUDA
  groups.

The [MLX distributed documentation](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
states that Ring is always available and describes NCCL as the CUDA backend.
An all-Mac group can still use JACCL and an all-CUDA group can later use NCCL;
a flat group containing both must start with Ring. MLX also permits separate
backends to be initialized in one program, which is the basis of the composite
Ring/NCCL gateway described below; it still requires a purpose-built bridge and
must not be inferred merely from NCCL being installed.

Exo is useful corroborating evidence. Its current placement code represents
`MlxMetal`, `MlxCuda`, and `MlxCpu` as eligible backends for one `MlxRing`
instance, and its CUDA/DGX packaging landed in
[`93a2474`](https://github.com/exo-explore/exo/commit/93a24748e60f356d472859c7da991dfadd2d8107).
The supplied screenshot is evidence that mixed hardware can be assembled, but
it is not a reproducible correctness or performance result. oMLX still needs a
hardware gate for the exact model and topology it advertises.

## The non-negotiable execution rule

If a model only fits by using the entire 800-900 GB pool, every forward pass
needs every rank. CUDA nodes cannot be “prefill only” and then disappear during
decode, because they own model layers that are needed for every generated
token. Metal and CUDA ranks both participate in prompt prefill and token decode.

The automatic planner can still exploit their different strengths:

- prefill calibration gives more layers to ranks with stronger matrix and
  attention throughput;
- decode calibration gives more layers to ranks with stronger memory-bound
  token throughput;
- the selected workload profile weights those two measurements; and
- link calibration penalizes cuts that send large activations over slow paths.

Specialized prefill and decode pools require two complete placements of the
model. They are possible only when there is enough spare memory for both model
copies. They increase throughput or reduce interference; they do not increase
the maximum model size.

## ConnectX CUDA pairs and the future supernode gateway

Two directly connected DGX Sparks should be one **logical supernode** in the
planner and fabric diagram while remaining two physical workers in inventory,
health, memory, and diagnostics. NVIDIA documents two QSFP ports per Spark,
each capped at 200 Gb/s, connected through the ConnectX-7 NIC. That is the path
for traffic between the two CUDA workers; it does not make the Mac's 10 GbE
edge any faster.

The target execution shape is:

```text
Metal ranks -- outer MLX Ring / 10 GbE -- CUDA gateway on Spark A
                                            |
                                  NCCL / ConnectX-7
                                            |
                                      Spark B executor
```

The gateway is the only CUDA process visible in the outer mixed Ring. It
receives one stage activation, executes the composite layer range together with
Spark B, and returns one stage activation. The large per-layer tensor traffic
stays inside the pair. The external control plane therefore schedules one
logical unit and maintains one slow cross-platform stage boundary, rather than
placing both physical CUDA workers independently on the 10 GbE Ring.

MLX supports initializing more than one distributed backend in one program,
but its global rank environment and model sharding path do not automatically
create this hierarchy. oMLX must add an explicit gateway runner with:

- an outer Ring group containing the Macs and one gateway process;
- an inner NCCL group containing both CUDA processes;
- a composite-stage contract with a shared contiguous layer range;
- equal or model-supported tensor shards inside that stage;
- activation transfer between the Ring and NCCL groups without a hidden full
  model or CPU round-trip; and
- fail-closed lifecycle handling: loss of either Spark removes the whole
  supernode and invalidates the approved plan.

Before the gateway exists, the safe compatibility mode keeps both Sparks as
adjacent outer Ring ranks and binds their internal neighbour hop to the
ConnectX address. That uses the fast cable for their direct hop but is not yet
the true one-node abstraction, because Ring-wide control collectives still see
both ranks.

A supernode is admitted only after all of these are true:

- every member is CUDA and exposes the same compatible MLX/NCCL contract;
- the members are connected through a verified ConnectX route, not merely an
  installed NIC;
- an isolated two-worker NCCL bandwidth test, bound to the detected ConnectX
  interfaces, passes the configured floor;
- the model supports the proposed inner sharding operation; and
- each member independently passes memory admission for its physical share.

Detection may suggest a pair, but automatic activation requires a measured
link. Grouping and verification are dashboard actions; an environment variable
or stale process setting cannot turn a candidate into a verified topology.

Until that gateway exists, the dashboard renders the result as one **visual
CUDA pair** with an NVIDIA identity, an aggregate safe memory figure, and two
expandable physical members. It also states that the pair remains two adjacent
execution ranks. This avoids presenting a verified NCCL link as proof that the
hierarchical execution path has been implemented. Manual shard controls still
apply to the physical ranks in compatibility mode.

## User-facing automatic modes

The coordinator should make the distinction automatically rather than asking
the operator to understand parallelism terminology.

| Situation | Automatic plan |
| --- | --- |
| Model needs the combined capacity | One mixed Metal/CUDA pipeline using every required node |
| Two CUDA workers pass the ConnectX gate | One composite CUDA stage behind a Ring/NCCL gateway |
| Model fits on a faster subset | Use the fastest measured subset with safe headroom |
| Model fits in separate prefill and decode pools | Benchmark optional disaggregation and use it only when end-to-end performance improves |
| Mac-only high-speed mesh wins | Use JACCL with the Mac subset |
| CUDA-only group wins | Use NCCL with the CUDA subset once supported by oMLX |
| A node is incompatible or makes the plan slower | Leave it available but out of this deployment |

An explicit **Use all eligible memory** control can force a capacity-oriented
plan. The default **Automatic** mode should optimize the selected workload while
still using enough nodes to fit the model safely.

## Target architecture

```text
OpenAI client
    |
    v
oMLX coordinator on any elected node
    |
    +-- discovery, trust, inventory, model catalogue
    +-- capability and link benchmarks
    +-- model fit and layer planner
    +-- launch/liveness/metrics
    |
    v
one heterogeneous logical deployment
    |
    +-- Metal rank: local layer shard + local KV
    +-- CUDA gateway / composite stage
    |      +-- Spark A shard -- NCCL / ConnectX-7 -- Spark B shard
    +-- CUDA rank: local layer shard + local KV
    +-- ...
    |
    v
normal oMLX streaming API response
```

The control plane may run on the Mac for convenience, but rank zero is an
execution role, not a permanent hardware assumption. The planner should place
the tokenizer/API coordinator and final model head where fixed-weight memory,
network reachability, and output latency are best.

## Automatic discovery and trust

### GUI-managed CUDA worker enrollment

The Cluster dashboard now has an **Add a CUDA worker** card. Enter the
coordinator Studio's private LAN IPv4 address and select **Generate join
command**. Paste that command into one Ubuntu/Debian CUDA box. Generate a fresh
command for every additional box; each credential is single-use and expires
after thirty minutes.

The generated command is intentionally not `curl | sudo`. It downloads a
standalone standard-library bootstrap to a temporary file, verifies its
SHA-256 digest from the admin-generated command, and only then runs it through
`sudo`. The command separately pins the SHA-256 of the exact oMLX worker source
bundle and the coordinator's SSH public-key fingerprint. The bootstrap then:

1. installs Python, Git, and OpenSSH prerequisites on Ubuntu/Debian;
2. adds the pinned coordinator key to the invoking Linux user's
   `authorized_keys`, restricted to the selected coordinator IP;
3. creates `/opt/omlx-cluster-worker/venv` with MLX 0.32 CUDA 13, the pinned
   MLX-LM revision, NumPy `<2.4`, and the worker-only parser/runtime set;
4. runs `pip check` and imports the real inference worker before enrollment can
   complete;
5. reports the worker's Ed25519 host key and installs that exact identity in
   the coordinator's `known_hosts`; and
6. adds the credential-free node record to the dashboard and active pool.

Join keys and post-claim sessions exist only in coordinator memory. Restarting
oMLX invalidates them. Completed records contain addresses, runtime path, and
public SSH fingerprint but no join key, session, password, or private key; the
registry is written atomically with mode `0600`. Replayed, expired, revoked,
identity-mutated, source-mismatched, or host-fingerprint-mismatched requests
fail closed.

The coordinator web port must be reachable from the CUDA LAN. If the dashboard
is open as `localhost`, set the oMLX **Server host** to `0.0.0.0` in Settings,
restart, and enter the Studio's LAN address in the enrollment card. Plain HTTP
is appropriate only on a trusted private LAN; use the dashboard's HTTPS origin
when the network is not trusted. The one-time secret is present in the pasted
shell command and may therefore remain in that worker user's shell history
until the short expiry passes.

Every headless Linux worker should advertise the existing `_omlx._tcp` service
through Avahi/mDNS and expose the same bounded capability endpoint as a Mac.
Bonjour suggestions remain untrusted until pairing succeeds.

The onboarding sequence is:

1. discover a node or enter its address manually;
2. verify SSH host identity and complete oMLX pairing;
3. collect OS, architecture, accelerator, memory, model inventory, and route
   facts;
4. run a small accelerator and Ring compatibility probe;
5. retain the node in the cluster inventory even when it is not selected for a
   particular model; and
6. invalidate old approval if the SSH identity, oMLX build, MLX build, model
   manifest, or accelerator contract changes.

Discovery failure must not make the feature unusable. Manual addresses and a
saved, host-key-verified inventory remain first-class paths; Exo has had public
reports of DGX nodes not appearing through discovery on otherwise reachable
10 GbE networks.

## Capability contract

The current probe reports Mac-specific facts and version strings. A mixed pool
needs a platform-neutral contract containing at least:

- operating system and version;
- machine architecture;
- accelerator kind (`metal`, `cuda`, or `cpu`), device identity, and usable
  accelerator/system memory;
- oMLX version and build digest;
- Python ABI, MLX version and platform build fingerprint, and MLX-LM revision;
- available distributed transports;
- supported model operations, quantization formats, cache types, and wire
  dtypes;
- model-manifest, config, tokenizer, and chat-template digests; and
- current admission ceiling after platform-specific reserves.

Metal and CUDA binaries are expected to have different build fingerprints.
Compatibility means the same declared MLX semantic contract plus successful
cross-rank parity probes, not identical binary hashes.

Model support must be capability-gated. MLX CUDA 0.32 is broad, but a model
using a Metal-only custom kernel or an operation missing from CUDA must be
rejected before launch. oMLX's optional Metal custom kernels must never be
imported as a required Linux worker dependency.

## Worker environment

The Mac application remains a Metal distribution. The GUI bootstrap creates a
headless Linux ARM64 environment using the official CUDA wheel set:

```text
mlx[cuda13]==0.32.2
same pinned MLX-LM revision
same oMLX cluster/runtime code
no macOS app, Metal-only extension, or Mac authorization dependency
```

The environment uses the controller's digest-pinned Python source rather than
installing the full desktop/server package metadata. This keeps `pip check`
clean and avoids pulling macOS UI, Metal-only extensions, downloaders, and web
server dependencies onto an inference-only CUDA worker. Re-enrollment is
idempotent and reapplies the declarative dependency set if `pip check` detects
that a later manual package operation damaged it.

A Linux worker should provide:

```text
omlx worker start
omlx cluster status --json
omlx cluster worker-smoke
omlx cluster collective-smoke
omlx cluster pipeline-smoke
```

It does not need the Mac menu-bar application or full local admin experience in
the first release.

## Model identity and staging

The current preview expects one absolute model path on every Mac. That is too
strict across macOS and Linux. A heterogeneous deployment should identify a
model by immutable manifest digest and maintain a per-node local path mapping.

The planner should:

1. resolve the selected catalogue entry to one manifest;
2. determine the exact safetensor files needed by every assigned layer range;
3. reuse already verified files on each node;
4. stage only missing files through the existing bounded, encrypted staging
   path or download them from the approved source; and
5. make every rank prove its assigned shard before launch.

The model can therefore live under different filesystem roots without weakening
identity or plan agreement.

## Heterogeneous planner

The planner should operate on a graph rather than summing memory alone.

### 1. Eligibility

Reject a node from a candidate deployment when any of these fail:

- paired and live;
- model/backend operation support;
- exact model and tokenizer identity;
- compatible MLX/MLX-LM and cluster protocols;
- a common Ring route to the candidate topology; or
- a positive memory budget after reserve and current pressure.

### 2. Capacity

Use safetensor headers exactly as the current planner does. Assign each layer's
real bytes once, charge fixed replicated weights to every applicable rank, and
reserve rank-local KV/activation memory for the requested context and
concurrency. A model fits only if every rank's assignment fits.

### 3. Calibration

Synthetic matrix calibration is useful but not sufficient for mixed devices.
Collect separate signals for:

- quantized matrix kernels used by the model;
- BF16 attention/prefill;
- single-token memory-bound decode;
- small decode activation transfers;
- large prompt/prefill activation transfers; and
- end-to-end pipeline timing for a tiny real model graph.

### 4. Placement

Enumerate viable node subsets and Ring orders. For each, find a contiguous
layer partition that minimizes the slowest predicted stage without crossing a
memory limit. The objective should combine prefill and decode according to the
selected execution profile.

The planner should prefer a plan with fewer slow cross-platform cuts, but it
must not pretend the DGX-to-DGX 200 Gb/s fabric makes the Mac edge faster. The
[DGX Spark hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
specifies 128 GB, 273 GB/s memory bandwidth, and ConnectX-7, while
[Apple specifies 10 GbE](https://www.apple.com/uk/mac-studio/specs/) for the
Mac Studio's built-in Ethernet. Unless another measured Mac route is present,
that 10 GbE boundary is part of every mixed plan.

### 5. Approval

Show the operator the selected nodes, local layer ranges, memory headroom,
transport, predicted bottleneck, and excluded-node reasons. Hash the complete
plan and make every rank agree before loading.

## Performance expectations

The large memory pool is achievable before a speedup is guaranteed.

Pipeline parallelism sends activations rather than model weights. During decode
the activation for one token is relatively small, so a 10 GbE boundary can be
workable; every token still waits for every pipeline stage. During long-prompt
prefill, activation transfers are larger and overlapping compute with transport
matters more.

The first success criterion is therefore:

```text
the model loads once across the combined memory and generates correctly
```

Only then should automatic planning claim:

```text
adding this CUDA node improves the selected workload
```

A slower node can increase maximum model capacity while reducing token rate.
The dashboard must present both effects rather than reducing cluster quality to
one aggregate-memory number.

## Hardware feasibility probe

The normal path is entirely in the Cluster dashboard. When two CUDA workers
advertise ConnectX and NCCL, **Verify ConnectX** launches an isolated two-rank
NCCL check through the authenticated admin API. **Start Cluster** runs that
verification automatically when the pair is still unverified. The result is
accepted only when both ranks answer and the measured large-payload rate clears
the configured floor.

[`benchmarks/heterogeneous_pool_probe.py`](../benchmarks/heterogeneous_pool_probe.py)
remains a developer and recovery diagnostic. It has no oMLX server dependency;
ordinary setup must not require it or a hand-written hostfile.

Run it locally on each node:

```bash
python3 benchmarks/heterogeneous_pool_probe.py
```

Then make a Ring hostfile containing the Mac and every Spark and run:

```bash
mlx.launch \
  --backend ring \
  --hostfile /absolute/path/to/heterogeneous-hosts.json \
  -- \
  python3 benchmarks/heterogeneous_pool_probe.py \
  --distributed \
  --expect-ranks 6 \
  --require-accelerators metal,cuda \
  --cuda-supernode-ranks 4,5 \
  --collective-mib 1,64
```

The developer probe verifies:

- Metal and CUDA are both present in one Ring;
- nominal MLX versions agree;
- every rank completes representative 4-bit quantized matrix and BF16 attention
  work;
- cross-rank result spread remains within a declared tolerance;
- small and large all-sums complete with measured timing;
- proposed CUDA pair members are adjacent in the outer Ring and expose
  NCCL.

It deliberately does not call `RingGroup.split()` or report that measurement as
NCCL. Ring subgroup support is version-dependent and, even where supported,
would still measure Ring. Direct ConnectX admission belongs to the dashboard's
separate NCCL verifier.

Passing this diagnostic is necessary but not sufficient. The dashboard must
then pass the real unequal pipeline smoke, a small downloaded model, and the
intended large model before hierarchical gateway execution is enabled.

## Implementation sequence

### Phase 0: prove the common data plane

- Run the new probe on one Mac plus one Spark, then the full topology.
- Run a two-rank real pipeline graph with one Metal and one CUDA rank.
- Compare seeded local and cross-rank continuation output.
- Record failures by MLX operation and quantization type.

Do not build UI around a topology that cannot pass this gate repeatedly.

### Phase 1: Linux worker distribution

- Add official MLX CUDA 13 packaging for Linux ARM64.
- Isolate Mac-only dependencies, custom kernels, memory APIs, and authorization
  flows.
- Generalize hardware and admission probes.
- Make the headless worker pass the existing cluster CLI and protocol tests.

### Phase 2: heterogeneous inventory and discovery

- Extend node capability schemas with platform, accelerator, build, and model
  operation facts.
- Advertise and discover Linux workers through `_omlx._tcp` plus manual pairing.
- Store trusted nodes independently from one deployment.
- Show Metal/CUDA identity, aggregate installed/admitted/model-usable memory,
  CUDA-pair membership, and exact exclusion reasons.

### Phase 3: mixed Ring planning and launch

- Separate model identity from node-local path.
- Extend preflight and staging to Linux workers.
- Feed Metal/CUDA calibration into the existing unequal layer planner.
- Launch the approved mixed Ring through the existing supervisor.
- Validate local shard residency, KV ownership, cancellation, and peer-loss
  teardown on every platform.

### Phase 3b: ConnectX composite CUDA stage

- Detect a two-Spark fabric candidate in the dashboard, then verify its route
  with the isolated NCCL direct-link probe.
- Add the Ring gateway plus CUDA-local NCCL executor group.
- Extend planning and model loading for a non-uniform stage: singleton Metal
  ranks plus a two-member tensor-sharded CUDA stage.
- Keep both Sparks visible under one logical dashboard unit and expose their
  independent health, memory, and shard residency.
- Fall back to adjacent flat Ring ranks if gateway validation is unavailable;
  never silently fall back after an approved hierarchical plan starts loading.

### Phase 4: Automatic strategy selection

- Compare mixed pipeline, Mac-only JACCL, and CUDA-only candidates.
- Add NCCL for homogeneous CUDA groups.
- Select subsets and layer maps from measured workload performance.
- Add explicit **Automatic**, **Use all eligible memory**, and advanced manual
  per-unit shard controls. Manual targets remain soft until the planner proves
  a feasible contiguous layer cut.

### Phase 5: optional specialization

- Add prefill/decode disaggregation only when two complete placements fit.
- Generalize the proven two-Spark gateway to additional JACCL/NCCL composite
  groups where a non-uniform model stage is supported.
- Add model-specific hybrid/SSM/MLA and quantized-cache support behind parity
  tests.

## Prefill/decode as a secondary strategy

Exo's public prefill/decode path creates separate model instances, computes KV
on the prefill side, and transfers it to the decode side. It is disabled by
default, uses a raw TCP cache service, silently falls back to local prefill, and
its public benchmark defaults to one node for each role. That is useful design
evidence, but not the foundation of the requested memory pool.

For oMLX, disaggregation should be considered only after the one-copy mixed
pipeline works. Both role pools must fit the full model, the cache protocol must
be authenticated and bounded, and measured transfer plus injection time must be
lower than local prefill time.

## Acceptance criteria

The heterogeneous pool is ready for an experimental UI when:

- a Metal rank and CUDA rank repeatedly pass the Ring compute/collective probe;
- a real unequal pipeline graph produces matching output across the mixed Ring;
- a declared ConnectX pair passes direct-link and NCCL capability gates before
  it is shown as a verified visual pair;
- a composite CUDA stage keeps its per-layer communication off the 10 GbE
  boundary and produces the same output as a flat placement;
- the Linux worker is installable from a pinned, reproducible package set;
- model and tokenizer identity are exact even with different local paths;
- every assigned shard passes memory admission and post-load residency checks;
- a peer failure cancels the whole deployment without a hidden local full-model
  load;
- the planner explains included and excluded nodes;
- the dashboard distinguishes installed, admitted, and actual model capacity,
  shows NVIDIA/Metal identities, and exposes safe automatic/manual shard maps;
- the model catalogue hides formats unsupported by any selected rank; and
- real TTFT, prefill, decode, and link measurements replace generic claims that
  CUDA always makes the cluster faster.

## Bottom line

The requested system is feasible and aligns with the current distributed oMLX
architecture. The key is one sharded model copy: a common MLX Ring for ordinary
ranks, with verified NCCL/ConnectX composite stages where they reduce external
traffic—not a KV handoff between replicas. Official MLX CUDA support removes
the largest software-stack uncertainty. The remaining hard work is Linux
packaging, the Ring/NCCL gateway, platform-neutral discovery and memory
admission, model-operation parity, and a planner honest about the 10 GbE Mac
boundary.

The immediate next action is to run the new probe on actual DGX Spark hardware.
That result determines whether implementation should move directly into the
Linux worker and mixed launcher or first isolate an MLX CUDA/model compatibility
gap.
