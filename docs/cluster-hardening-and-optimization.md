# Cluster / tensor-parallel hardening and optimization plan

Design doc + phased implementation checklist from the clustering /
tensor-parallel subsystem review. Every file:line reference below was
**verified against HEAD `2718845b` (branch `deploy/session-fixes-v2`) on
2026-08-25**. Line numbers will drift as the tree moves — treat them as anchors
(the quoted identifiers are the stable handles), and re-locate rather than
trust a stale number if a reference doesn't land on the described code.

Paths are relative to the repo root; the package is nested one level
(`omlx/omlx/...` on disk, written `omlx/...` here). References into the
**pinned runtime packages** are written `site-packages/mlx_lm/...` and
`site-packages/mlx/...` (mlx-lm 0.31.3, mlx 0.32.0 under
`.venv/lib/python3.11/site-packages/`) — those are not repo files, and a pin
bump invalidates them wholesale.

Findings are labeled **CONFIRMED** (the full causal chain was traced in code)
or **PLAUSIBLE** (mechanism traced, final runtime behavior needs a repro).

---

## 1. Context

### What the subsystem is

The clustering path is `omlx/cluster/` (48 modules, ~28k lines) plus
`omlx/engine/distributed.py`. Architecture, as actually wired:

- **Coordinator**: `DistributedBatchedEngine` (`engine/distributed.py`) keeps
  oMLX's API/tokenizer layer and proxies model work over HTTP to a **private
  mlx-lm server on rank 0** (`127.0.0.1:<port>`). It owns a
  `DistributedJobSupervisor` (`cluster/launch.py:1717`) that runs preflight,
  builds a hostfile + one shared argv, and launches every rank through MLX's
  `mlx._distributed_utils.launch` over SSH.
- **Rank process**: `cluster/inference_worker.py` — decodes the signed plan
  (`deployment.decode_worker_contract`), admits its stage against the local
  memory guard, loads progressively (`progressive_loading.py`), TP-shards via
  `tensor_strategies.py` or pipelines via `pipeline_compat.py`, then runs the
  pinned `mlx_lm.server.run()` on every rank (rank 0 serves HTTP; workers run
  the generation thread in lockstep, receiving requests via
  `_share_object` pickle-over-`all_sum` broadcast in
  `site-packages/mlx_lm/server.py:485-502`).
- **Collectives**: `mx.distributed` ring (TCP) or jaccl (RDMA over
  Thunderbolt); backend chosen by `transport.py` detection
  (`select_backend`, transport.py:359). Hostfile IPs come from live
  two-ended probes (`resolve_link_addresses`), never cached.
- **TP sharding**: explicit adapters for `qwen3_next`/`nemotron_h`
  (`tensor_strategies.py:331,474`) + an AST-gated native `shard()` fallback
  (`native_shard_is_layer_local`, tensor_strategies.py:165). One layer
  materialized/sharded/evaluated at a time.
- **Liveness**: per-rank JSON markers with fsync'd atomic writes
  (`RuntimeMarker`, inference_worker.py:288) refreshed by heartbeat threads;
  `PeerWatchdog` (rank 0 only, `liveness.py:435`) reads peer markers over SSH;
  a launcher-parent watchdog on every rank
  (`_watch_launcher_parent`, inference_worker.py:492); per-request coordinator
  preflight (`_require_healthy_cluster`, engine/distributed.py:1290, cached
  `_PEER_HEALTH_TTL = 10.0`).
- **Guards**: `memory_guard.py` (admission + `LoadMemoryWatchdog` during
  load), `prefill_guard.py` (per-prompt rank-voted admission).
- **Benchmarking**: `strategy_benchmarks.py` (durable per-TP-degree medians at
  `~/.omlx/cluster/strategy-benchmarks.json`, fed by streaming requests via
  `_record_strategy_benchmark`, engine/distributed.py:1216) and the synthetic
  pre-launch performance probe (`run_cluster_performance_probe`,
  launch.py:470).

### What triggered the review, and deployment facts

A 2-Mac Thunderbolt cluster serving **NVIDIA-Nemotron-3.5-Lightning-30B-A3B
(oQ4, nemotron_h)** at **TP=2**, made functional by PR #2844 ("make quantized
Nemotron-H run tensor-parallel at TP=2", `d85c0d9f` + follow-up `15ed4841`).
The staged checkpoint's config was read during this review:
`mlp_bias=false`, `moe_latent_size=null`, `n_shared_experts=1`, `n_groups=8`,
`mamba_num_heads=64`, heads 32/2, oQ per-module quant overrides — this matters
for finding B1's severity (latent, not live).

Registry state at review time: `~/.omlx/cluster/deployments.json` is
**empty** (`{"deployments": [], "schema_version": 1}`) — the cluster is
currently deactivated, consistent with the pending peer-restart state. The
Nemotron TP=2 deployment above is the most recent activation, and the
`cluster_bench_Qwen3.8-27B…_tp2` / `Qwen3.6-35B…_tp2` filenames show Qwen
checkpoints have also been cluster-activated at TP=2 (via the MTP branch) —
i.e. the **native-shard fallback path** of `tensor_strategies.py` has been
live too, not only the two registered adapters.

**Reconnection: there is none, by design.** No rank reconnects and no rank is
relaunched: any rank loss (or watchdog verdict) ends the whole job —
launcher-group SIGTERM/SIGKILL plus SSH reaping of remote ranks
(`_terminate`/`_reap_remote_ranks`, launch.py:1932-2124) — and recovery is a
manual reactivation. A failed distributed *teardown* keeps the engine
registered so stop can be retried (`engine_pool.py:2048`). Every finding
below about wedges is therefore about *detection*, not recovery; recovery is
already fail-stop-and-relaunch.

MTP status on this branch: **fail-closed**. `_validate_model_settings`
(engine/distributed.py:258-277) refuses `mtp_enabled`/`vlm_mtp_enabled` (and
dflash/specprefill/turboquant-KV) on distributed engines, and the Nemotron
pipeline hook raises on a non-zero `n_confirmed`
(`pipeline_compat.py:181-196`). MTP-over-TP lives on the local branch
`feat/mtp-tensor-parallel-clustering-v2` (`709bf57d` "support native MTP on
pure tensor-parallel deployments", `a832e9e8` CPU-stream fix) and is **not**
in this review's scope beyond noting the seams it will touch (the
`coordinator_generation_step`/synchronized-sampler paths in
`runtime_optimizations.py` and the `n_confirmed` guard above).

The `~/.omlx/cluster_bench_*.json` files (`..._Qwen3.8-27B-oQ4e-mtp_tp2.json`
etc.) are **ad-hoc MTP on/off A/B results** (prefill/decode timings per
condition); no producer script exists in this branch — they came from the MTP
branch or scratch tooling. The durable, product-consumed benchmark store is
`~/.omlx/cluster/strategy-benchmarks.json` (`strategy_benchmarks.py`). Don't
confuse the two.

### Review coverage (honesty note)

Read in full: `engine/distributed.py`, `inference_worker.py`,
`tensor_strategies.py`, `runtime_optimizations.py`, `telemetry.py`,
`prompt_snapshot_cache.py`, `liveness.py`, `launch.py`, `memory_guard.py`,
`prefill_guard.py`, `node_role.py`, `progressive_loading.py`,
`pipeline_compat.py`, `deployment.py`, `runtime.py`, `discovery.py`,
`transport.py`, `performance.py`, `strategy_benchmarks.py`, `collective.py`,
plus the pinned `mlx_lm/server.py` distributed paths, `mlx_lm/generate.py`
batch surface, `mlx/nn/layers/distributed.py`, and
`mlx_lm/models/{nemotron_h,qwen3_next,switch_layers}.py`. Skimmed/targeted
only: `routes.py` (role/plan flow), `planner.py` (TP gates ~584-850),
`autoconfigure.py` (~30-230), `staging.py`, `enrollment.py`, `ssh_*`,
`catalogue.py`, `incidents.py`, `guidance.py`, `probe.py`, CUDA paths. Claims
about the skimmed files are correspondingly weaker. Test coverage is unusually
dense (50+ `tests/test_cluster_*.py` files) and several findings below
explicitly name the test file to extend.

---

## 2. Theme A — Lockstep execution and partial failure

The distributed design is SPMD-lockstep: every rank runs the same
`mlx_lm.server` generation loop, requests are broadcast
(`_share_object`, `site-packages/mlx_lm/server.py:485-502`), and every
divergence between ranks must be resolved by an explicit collective
(cancellation votes, prefill-guard votes, SSD-boundary votes). The failure
class that matters is a rank leaving lockstep **unilaterally**: the survivors
block inside a collective that has no timeout.

### A1. Unilateral mid-generation exception desyncs the collective — and liveness cannot see it — HIGH (CONFIRMED mechanism; wedge endgame PLAUSIBLE)

**What's wrong.** Two exception surfaces on the generation thread, with
different endgames:

- **Sequential path** (used for any non-batchable request — notably every
  request with `seed` set, `_is_batchable`,
  `site-packages/mlx_lm/server.py:685-686`): `_serve_single` wraps everything
  in `except Exception as e: rqueue.put(e)`
  (`site-packages/mlx_lm/server.py:1023-1024`). A unilateral failure on one
  rank (Metal OOM in a layer, a kernel error, an I/O error in the SSD
  snapshot path) is caught **locally**; that rank returns to the
  `_next_request` loop and issues its next collective —
  `_share_object`'s scalar `all_sum(0)` — while the other ranks are still
  blocked inside a *model-layer* collective of a completely different shape.
  Mismatched ring messages then either wedge both sides or corrupt/crash;
  nothing agrees the failure across ranks.
- **Batched path**: `batch_generator.next()` at
  `site-packages/mlx_lm/server.py:853` is **not** wrapped in try/except; an
  escaping exception kills the whole generation thread on that rank.
  Asymmetric outcome: on a **worker** rank, `run()` is just
  `response_generator.join()` (`site-packages/mlx_lm/server.py:1749`), so the
  process exits, the ring socket closes, and the survivors' collectives error
  out — fail-stop, detected, acceptable. On **rank 0**, the HTTP server keeps
  serving and the process stays alive; requests queue forever.

**Why liveness cannot detect the wedge.** Every liveness signal is
thread-independent of the generation thread: `RuntimeMarker.start_heartbeat`
(inference_worker.py:357-380) and `RuntimeTelemetry._heartbeat_loop`
(telemetry.py:145-153) are daemon threads that refresh the marker every 10 s
regardless of whether generation is making progress. The telemetry docstring
("'stale' has to mean *stalled*", telemetry.py:112-118) is aspirational — a
rank whose generation thread is dead or blocked in a mismatched collective
publishes fresh, healthy markers forever. `PeerWatchdog` reads exactly those
markers (`liveness.py:485-491`), and the coordinator preflight
(`_require_healthy_cluster`) reads the same markers plus supervisor state. The
only backstop is the per-request inactivity timeout
(`OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT`, default **300 s**,
engine/distributed.py:133) — which fails the *request* but leaves the wedged
deployment marked healthy until a human deactivates it.

**Failure scenario.** A seeded request (sequential path) hits a transient
Metal OOM on rank 1 mid-decode. Rank 1 catches it, loops, and answers the next
broadcast with a scalar all_sum; rank 0 is mid-layer in a tensor all_sum.
Both ranks block (or the ring errors non-deterministically). Markers stay
"ready"; the dashboard is green; every subsequent request 500s after 300 s of
silence. The cluster stays in this state until manual deactivation.

**Fix.** Two parts, effort M total:

1. **Fail-stop on unilateral generation failure.** In the worker, distinguish
   *agreed* failures (the prefill-guard vote and the coordinated cancel — both
   raise on every rank together) from *unilateral* ones, and make unilateral
   generation-thread failures terminate the rank process (marker phase
   `failed` + `os._exit(1)`), converting the undetectable wedge into the
   already-well-handled process-death path (launcher teardown,
   `_runtime_failure_reason` reads the marker, launch.py:2137). Cheapest hook:
   oMLX already intercepts every exception surfaced to the response queue —
   `_TelemetryQueue.put`'s `isinstance(item, BaseException)` branch
   (telemetry.py:609-610) — plus a try/except wrapper around
   `TelemetryBatchGenerator.next()` (telemetry.py:793) for the escaping
   batched case on rank 0. Mark agreed-rejection exception types
   (`PrefillMemoryExceededError`, the cancel path) as safe-to-continue;
   everything else on a distributed rank is fail-stop.
2. **Make heartbeats carry generation progress** so a wedge that slips through
   is still visible: publish a monotonically increasing step counter
   (`_batch_steps` already exists, telemetry.py:339) plus
   `active_requests > 0` in the marker, and teach `PeerHealth`/the dashboard
   to flag "active request, no step progress for N × interval" as stalled
   rather than healthy. This is observability, not control — no automatic
   kill from this signal at first.

Regression test: extend `tests/test_cluster_telemetry.py` with a fake queue
receiving a `RuntimeError` item and assert the fail-stop hook fires (and does
NOT fire for `PrefillMemoryExceededError`); a two-process loopback repro
belongs in Phase 0 (0.3) before trusting the fix end-to-end.

### A2. Prefill-guard vote can be skipped by a non-voted exception — MEDIUM (CONFIRMED, small fix)

**What's wrong.** `RankPrefillGuard.check_collective`
(`prefill_guard.py:165-231`) catches only `PrefillMemoryExceededError` from
the local check (185-194); any other exception from `check()` — which
re-raises non-PMEE exceptions explicitly (prefill_guard.py:154-158) —
propagates **before this rank contributes its vote**, while the peer ranks
proceed into `all_sum` (212-214) and block. Same lockstep-violation class as
A1, but in a code path oMLX fully owns.

**Failure scenario.** `raise_if_prefill_exceeds` or the usage measurement
raises anything unexpected (`TypeError` from a malformed monitor field, an
import error on a stripped worker install) on one rank only → that rank
raises out of `_tokenize` while the others sit in the vote collective.

**Fix (S).** Convert *any* local exception into a rejection vote: catch
`Exception`, vote 1, complete the collective, then re-raise the original
error locally (peers raise the generic "rejected by rank N" error). This
keeps the vote count matched on every rank under every local failure. One
test in `tests/test_cluster_prefill_guard.py` with a monitor that raises
`TypeError`.

### A3. Cross-rank determinism of `all_sum` is a stated assumption — LOW (assumption, partially evidenced)

MLX-LM's synchronized sampler (pure-TP path) has every rank sample
independently from a shared seed (`site-packages/mlx_lm/server.py:713-714`)
against logits produced by collectives. Token agreement therefore requires
`all_sum` to be **bitwise identical on every rank**. For TP=2 this is exact
(two-operand fp addition is commutative); for ring reduce-scatter at larger
world sizes it holds only if every rank applies the same reduction order. The
MiniMax decode smoke test asserts cross-rank token equality
(`collective.py:361-363`), which is the right check but only runs world=2.
Not a finding to fix — a constraint to keep: **any future >2-rank TP
deployment must re-run a cross-rank token-equality check first**, and the
smoke harness should grow a `world_size` parameter when that day comes.

---

## 3. Theme B — Tensor-parallel sharding correctness

The #2844 adapters were re-derived during this review against the pinned
model code and found correct (see §7 for the verified-non-issues list: qkvz
packing vs. contiguous split, conv1d gather indices, Mamba in_proj layout,
uneven quant-group ranges). Two residual gaps:

### B1. `shard_inplace` + wrapper `all_sum` counts biases world_size times — MEDIUM (CONFIRMED mechanism; latent for the deployed checkpoint)

**What's wrong.** MLX's `_sharded_to_all` predicate returns `None` for any
parameter path ending in `bias`
(`site-packages/mlx/nn/layers/distributed.py:96-104`) — i.e.
`shard_inplace(module, "sharded-to-all")` keeps the **full additive bias on
every rank** and provides no collective of its own. That is correct for
`shard_linear`, whose `ShardedToAllLinear.__call__` adds the bias **after**
its internal `all_sum` (`distributed.py:330-338`). But oMLX's MoE strategy
slices weights in place and recombines with one *external*
`mx.distributed.all_sum` around the whole block (`_wrap_sharded_moe`,
`tensor_strategies.py:237-253`) — so any per-rank bias added *inside* the
wrapped module is summed `world_size` times. Exposed sites in the pinned
`nemotron_h`:

- `shared_experts` (`NemotronHMLP` with `bias=args.mlp_bias`,
  `site-packages/mlx_lm/models/nemotron_h.py:302-307`), sharded via
  `shard_inplace` at `tensor_strategies.py:600-610`: with `mlp_bias=true`,
  `down_proj.bias` is added per rank → doubled at TP=2.
- `fc2_latent_proj` (when `moe_latent_size` is set,
  `nemotron_h.py:401-405,419`): left entirely unsharded/replicated. Without
  bias that is algebraically fine (`all_sum(W·partial) = W·sum`), **with**
  `mlp_bias=true` the bias is again summed per rank.
- Not exposed: the routed `switch_mlp` (`SwitchMLP` defaults `bias=False`,
  `site-packages/mlx_lm/models/switch_layers.py:200-213`) and the qwen3_next
  MoE (all projections bias-free; its `shared_expert_gate` multiplies the
  partial by a replicated scalar, which distributes over the sum correctly).

**Verified safe for the checkpoint staged on this coordinator**: the
Nemotron-3.5-Lightning config has `mlp_bias=false` and `moe_latent_size=null`
(read 2026-08-25; note the deployment registry is currently empty, §1), so
this is **latent** — it fires the day someone activates a nemotron_h
checkpoint with biases or latent-MoE projections, and it fires *silently*
(outputs are wrong by a bias offset per MoE layer; nothing crashes).

**Fix.** Two options, do the first regardless:

1. **(S) Fail-closed planner gate**, same idiom as the #2844 follow-up's
   quant-group constraints (`_tensor_parallel_divisors`,
   `planner.py:612-674`): refuse TP for `nemotron_h` configs with
   `mlp_bias=true` or `moe_latent_size` set, with a reason string naming
   this doc. Cheap, honest, keeps the wrong answer unreachable.
2. **(M) Actual support**, if such checkpoints matter: at shard time, zero
   the retained bias on every rank except one (or pre-divide by
   `world_size`) for modules recombined by `_wrap_sharded_moe`, and shard
   `fc1/fc2_latent_proj` properly. Needs a numerics test with a synthetic
   biased config in `tests/test_cluster_tensor_strategies.py`.

### B2. Rank prefill guard divides replicated (MLA) KV by the TP degree — LOW (CONFIRMED code asymmetry; conditional on a future model)

**What's wrong.** `rank_monitor` (`prefill_guard.py:75-80`) divides
`kv_heads`, `heads`, and `kv_override` by `tensor_parallel_size`
unconditionally. The planner knows better: `_kv_cache_replicated_across_tp`
(`planner.py:788-800`) documents that MLA-style latent KV
(`kv_lora_rank`/`qk_rope_head_dim`) is **not** per-head and must not be
divided. `kv_heads` survives by accident (`max(1, 1 // tp) == 1`), but a
`kv_bytes_per_token` override for an MLA model would be halved → the rank
guard under-charges prefill by 2× and admits prompts that OOM. No currently
TP-shardable architecture is MLA, so this is latent. **Fix (S):** replicate
the planner's predicate in `rank_monitor` (skip the `/ tp` scaling when the
model config is MLA-shaped); one test beside the existing TP cases in
`tests/test_cluster_prefill_guard.py`.

---

## 4. Theme C — Liveness measures the wrong channels

### C1. Health probes ride SSH routes; collectives ride hostfile IPs — MEDIUM-HIGH (CONFIRMED by construction; hang endgame PLAUSIBLE)

**What's wrong.** Every liveness mechanism reaches peers via `host.ssh` (the
SSH hostname — typically the LAN/mDNS route): `probe_peer` and
`read_remote_marker` (`liveness.py:126-150,185-248`), the `PeerWatchdog`, the
coordinator preflight. The collectives ride the **hostfile IPs** — the
Thunderbolt/RDMA point-to-point addresses selected by
`resolve_link_addresses` (`transport.py:1407`). These are different physical
links. A Thunderbolt cable pull (or an RDMA path failure) with the LAN/Wi-Fi
still up leaves every SSH probe green, every marker fresh (heartbeat threads,
see A1), and every rank blocked in a collective with no timeout.

**Partial mitigations that exist**: on the TCP ring, the peer's interface
going down usually errors the socket (`ENETDOWN`/reset) and crashes the rank
→ process death → detected. The jaccl/RDMA failure mode is not established.
And the 300 s request-inactivity timeout eventually fails requests without
un-wedging the deployment (same shape as A1's endgame). The module docstring
(`liveness.py:1-24`) frames the watchdog as the answer to exactly this cable
pull — it is, but only when the pull also severs SSH (single shared link) or
kills the process.

**Failure scenario.** Two Macs connected by both Thunderbolt (172.16.99.x
hostfile IPs) and Wi-Fi (SSH). TB cable unplugged mid-generation with jaccl:
ranks block in the collective, markers stay fresh over Wi-Fi SSH, watchdog
never fires, dashboard green, requests time out at 300 s each. Recovery is a
manual deactivate (which works — `_reap_remote_ranks`, launch.py:1978).

**Fix (M).** Add a **data-plane check** to the watchdog and the preflight:
the deployment already carries the hostfile IPs (`deployment.hosts[*].ips`);
`verify_link_reachability` (`transport.py:1482`) already implements a bounded
route+ICMP probe over an explicit interface/address pair, and rank 0 has SSH
to every peer to run the reverse direction. On data-plane failure with
control-plane success, fail the deployment with a message naming the cable
("Thunderbolt path to <node> is down; SSH still answers"). Combine with A1's
progress-epoch heartbeat so either signal catches a wedge.

### C2. `status()` reads supervisor deques without the lock — LOW-MED severity, MEDIUM confidence (PLAUSIBLE race)

**What's wrong.** `DistributedJobSupervisor.status()`
(`launch.py:2194-2210`) does `tuple(self._stderr)[-20:]` and iterates
`self.rank_ready_events` while the `_drain` reader threads append/mutate them
under `self._condition` (launch.py:1832-1856). Iterating a `collections.deque`
during a concurrent append raises `RuntimeError: deque mutated during
iteration`. `status()` sits on the per-request path
(`_ensure_available`, `_read_timeout_error`, `_transport_failure_error`,
engine/distributed.py:293-373), so a burst of launcher stderr during serving
can fail an unrelated request with a spurious exception.

**Fix (S).** Take `self._condition` (or a dedicated lock) around the snapshot
in `status()`; both structures are tiny. A deterministic test can hammer
`status()` against a thread appending to `_stderr`.

### C3. Healthy-path peer refresh pays a redundant serial SSH round trip — LOW (CONFIRMED)

`check_peers` (`liveness.py:339-353`) runs `probe_peer` (SSH `true`) and
*then* `read_remote_marker` (SSH python one-liner) **serially per peer**. A
successful marker read already proves reachability; the separate probe adds a
full SSH handshake per peer per refresh (there is no ControlMaster
multiplexing in `cluster_ssh_options` — verify in `ssh_policy.py` before
assuming). **Fix (S):** attempt the marker read first when
`require_heartbeat` and `deployment_id` are set; fall back to `probe_peer`
only to distinguish "host down" from "marker problem" on failure. Halves the
health-refresh latency that D1 puts on the request path.

---

## 5. Theme D — Request-path and prefill performance

### D1. Peer-health refresh blocks TTFT at every TTL expiry — MEDIUM (CONFIRMED cost structure; magnitude needs Phase 0 numbers)

**What's wrong.** `_require_healthy_cluster`
(`engine/distributed.py:1310-1341`) refreshes peer health at most every 10 s
(`_PEER_HEALTH_TTL`), single-flighted behind `_peer_health_lock` — good — but
the refresh is **synchronous on the request path**: the first request after
expiry (i.e., roughly one request per 10 s under steady traffic, or *every*
first request of an interactive turn after an idle gap) waits for
`check_peers` = per-peer serial (probe + marker read) SSH, each with a 5 s
connect timeout (7 s subprocess cap). Typical cost is a few hundred ms per
peer; worst case ~14 s per peer before the request even starts.

**Fix (S/M).** Serve-stale-while-revalidate: return the cached verdict
immediately and kick the refresh to a background task; only *block* when
there is no verdict at all (first request after load) or the cached verdict
is unhealthy. Combine with C3 (drop the redundant probe) and parallelize the
per-peer reads (`asyncio.gather`/thread pool — they are independent SSH
subprocesses). Keep the supervisor-state checks (returncode/failure_reason)
synchronous — they are free and catch hard failures.

### D2. SSD prompt-snapshot writes are synchronous inside prefill — MEDIUM (PLAUSIBLE; measure before fixing; default ON)

**What's wrong.** `prompt_cache_ssd` defaults **on**
(`ExecutionSettings.prompt_cache_ssd = True`, `performance.py:168`). On every
aligned 2048-token prefill boundary, the generation thread synchronously
serializes and writes a boundary file: sequential path via the chained
progress callback (`snapshotting_stream_generate` → `ssd_store.put`,
`telemetry.py:994-1000`), batched path via `_omlx_snapshot_boundary` →
`extract_cache` + `put` (telemetry.py:762-791). For sliceable KV the file
holds only the newest 2048-token slab (`KVCacheSegment`,
`prompt_snapshot_cache.py:176`), but for the models this feature exists for —
rotating windows, Mamba/GDN recurrent state — each boundary stores the **full
non-sliceable state** (`_wrap_for_save`, prompt_snapshot_cache.py:269-310),
which for a large hybrid model can be hundreds of MB per boundary, written
with `save_prompt_cache` + `os.replace` while the prefill waits. A 100k-token
prompt takes ~49 boundaries of this on every rank.

**Fix.** Phase 0 measures first (offline: construct an
`SSDPromptSnapshotStore` and time `put()` against realistic nemotron_h cache
states — no cluster needed). If material: move serialization+write to a
single background writer thread per store with a bounded queue —
correctness-safe *by design* because the restore path already requires
all-rank agreement per boundary (`agree_ssd_boundary`, telemetry.py:687-712;
`agreed_boundary`, prompt_snapshot_cache.py:336-352): a rank whose async
write hasn't landed simply doesn't vote that boundary, exactly like today's
failed-write path. Note the store as wired also has **no byte bound**
(`ssd_max_entries=64`, `max_bytes=None`, telemetry.py:634-635/679-681):
64 entries × full recurrent state can be tens of GB on disk; add a
`max_bytes` while in there (S).

### D3. Per-token cancellation collective on the sequential path — LOW (CONFIRMED cost; the correctness it buys is worth it)

`CoordinatedGenerationContext._should_stop` (telemetry.py:896-916) issues one
scalar `all_sum` + `.item()` sync **per generated token** on the sequential
(seeded / non-batchable) path. On a TCP ring that is an extra RTT per token
on a path that already pays per-layer collectives. Only worth touching if
seeded distributed requests are common: cheap variants are voting every k
tokens (bounded extra tokens after a cancel) or piggybacking the stop bit as
one extra element on an existing per-token collective. Do nothing until D0
instrumentation shows the sequential path is actually exercised.

### D4. Only streaming requests feed the strategy-benchmark store — LOW (CONFIRMED)

`_record_strategy_benchmark` is called from `stream_chat` and
`stream_generate` only (engine/distributed.py:944-951,1194-1201); the
non-streaming `chat()`/`generate()` return without recording. Automatic
strategy selection (`choose_parallelism` `measurements`,
`autoconfigure.py:222+`) therefore learns nothing from batch/agent workloads
that use non-streaming completions. **Fix (S):** compute the same
tps/ttft numbers in the non-streaming paths (server timings suffice:
`generated_at`/`generated_until` are already tracked) and record when they
pass the same validity gates.

---

## 6. Theme E — Launch and teardown robustness

### E1. Collective port span is reserved only on the coordinator — LOW-MED (CONFIRMED; fails loud at launch)

`_available_launch_ports` (`launch.py:126-168`) binds the API port and the
whole collective span on the **coordinator's loopback**, then closes the
listeners and hands the numbers to `mlx.launch`. Two gaps: (a) the same span
must be free on **every peer** (ring listeners bind there on each host) —
never checked, so a busy port on the peer fails the launch late with a bind
error; (b) classic TOCTOU between close and rebind. Both fail loudly at
launch rather than corrupting anything, hence LOW-MED. **Fix (M, low
priority):** probe the span on peers during preflight (one SSH python
one-liner alongside `_PREFLIGHT_SCRIPT`), and/or retry `start()` once with a
fresh span on a bind-failure signature in the launcher stderr.

### E2. `_drain`'s failure-event predicate is a fragile contract — LOW (CONFIRMED; benign with today's emitters)

`_drain` latches `self.failure_event` for **any** stdout event carrying a
`reason` or `error` key (`launch.py:1854-1855`), and
`_require_healthy_cluster` hard-fails serving on `status.failure_reason`
(engine/distributed.py:1306-1309). Today's emitters
(`launcher_lost`, `peer_lost`, worker `failed`) are all genuinely fatal, so
this is correct — but the first person to emit an *informational* event with
an `error` field bricks a healthy deployment. **Fix (S):** match on an
explicit set of fatal `type` values instead of key presence; assert the set
in `tests/test_cluster_launch.py`.

### E3. Accepted-as-is (documented, no action)

- **Unreachable-peer reap leaves a resident rank** — `_reap_remote_ranks`
  logs and continues (`launch.py:2079-2089`); nothing better is possible
  without the peer.
- **300 s default request-read timeout** (engine/distributed.py:133) is a
  deliberate inactivity bound (SSE keepalives make it per-chunk); revisit the
  default only alongside A1/C1, which shrink how long a wedge can matter.
- **`plan_hash` is consistency, not security** — the worker compares the
  argv-supplied hash to the argv-supplied plan (inference_worker.py:970-971);
  both ride the same SSH channel, which is the actual trust boundary.

---

## 7. Phased implementation checklist

Ordering: correctness before optimization; low-risk/high-confidence first.
Effort tags: S(<~1h) / M(half-day) / L(multi-day). Risk is conveyed by phase
grouping. All line refs verified 2026-08-25 @ `2718845b`; re-grep the quoted
identifiers if executing later. **Nothing in Phase 0 touches the live
cluster** — the two Macs are serving; every Phase 0 item is offline or
log-reading.

### Phase 0 — Instrumentation and repro (unblocks Phase 2/3 decisions)

- [ ] **0.1** [S] Measure SSD snapshot cost offline: construct
  `SSDPromptSnapshotStore` + realistic nemotron_h cache states (Mamba
  recurrent + KV slabs at deployed shapes), time `put()` and record per-file
  bytes across a 2048-step boundary ladder. Decides D2's fix-vs-accept.
  (§D2)
- [ ] **0.2** [S] Time the peer-health refresh: log elapsed around
  `check_peers` in `_require_healthy_cluster` (one log line), collect a day
  of samples from normal serving. Decides D1's priority and validates C3's
  expected halving. (§D1, §C3)
- [ ] **0.3** [M] Two-process loopback wedge repro: extend the
  `collective.py` smoke pattern with a worker pair where one rank raises
  mid-"generation" (a) inside a caught sequential-style loop and (b) killing
  its loop thread, and observe the survivor + marker/watchdog behavior.
  This is the ground truth for A1's endgame (wedge vs. crash, per backend)
  and the regression harness for the 2.1 fix. (§A1, §C1)

### Phase 1 — Correctness, low-risk / high-confidence

- [x] **1.1** [S] Prefill-guard vote-on-any-exception: catch `Exception` in
  `check_collective`'s local check, vote 1, re-raise locally after the
  collective (`prefill_guard.py:185-231`). (§A2)
  — Done (commit `b97b87f1`). `check_collective` now catches any
  `Exception` (not just `PrefillMemoryExceededError`) as `local_exc`,
  casts this rank's vote regardless of type, and re-raises `local_exc`
  after the collective on both the `world_size<=1` and normal paths.
  Tests: `test_unexpected_local_exception_still_casts_a_vote_before_reraising`,
  `test_unexpected_local_exception_on_single_node_reraises_without_a_collective`.
- [x] **1.2** [S] Planner fail-closed TP gate for biased/latent nemotron_h
  configs: refuse TP when `mlp_bias` or `moe_latent_size` is set, in
  `_tensor_parallel_divisors`/`_supports_tensor_parallel`
  (`planner.py:612-674`). (§B1 option 1)
  — Done (commit `084a02eb`). Added `_config_truthy` (mirrors
  `_config_int`'s nested `text_config`/`language_config`/`llm_config`
  walk) and `_nemotron_h_tp_bias_unsafe`; checked before the
  `supports_model_type` short-circuit in `_supports_tensor_parallel`.
  4 new tests cover both fields plus the nested-config case.
- [x] **1.3** [S] Snapshot supervisor state under the condition lock in
  `status()` (`launch.py:2194-2210`). (§C2)
  — Done (commit `59987a47`). `status()` now reads `_stderr` and
  `_rank_ready_events` under `with self._condition:`; `_failure_reason()`
  stays outside the lock (atomic reference read). Verified with a
  deterministic forced-interleaving test (a `deque` subclass that
  pauses mid-iteration on a `threading.Event`) rather than a
  probabilistic hammer test — the first version of this test passed
  5/5 without the fix and was discarded as non-discriminating.
- [x] **1.4** [S] Marker-read-first health check: drop the redundant
  `probe_peer` round trip on the healthy path (`liveness.py:339-353`);
  verify `cluster_ssh_options` has no ControlMaster before assuming the
  cost, and consider adding multiplexing while there (`ssh_policy.py`).
  (§C3)
  — Done (marker-read-first only; multiplexing deferred, see below).
  `check_peers` now reads the marker first whenever a heartbeat is
  required; a successful read (`marker is not None`) already proves
  reachability and `probe_peer` is skipped entirely. `probe` only runs
  as a fallback when the marker read fails, to tell "host down" (probe
  also fails → `reachable=False`) from "marker problem" (probe
  succeeds → `reachable=True`, heartbeat-missing detail). Confirmed
  `cluster_ssh_options` (`ssh_policy.py:24-`) sets no
  ControlMaster/ControlPersist. **Not done: adding multiplexing.**
  ControlMaster would change the behavior of every SSH call in the
  cluster (staging, preflight, launch), not just this refresh path —
  out of scope for an [S] fix; left for a dedicated item if the
  0.1/0.2 Phase-0 numbers show it's worth the added failure surface
  (stale control sockets, `-O exit` cleanup on stale-pairing errors).
- [ ] **1.5** [S] Restrict `_drain`'s failure-event latch to an explicit
  fatal-type set (`launch.py:1854-1855`). (§E2)
- [ ] **1.6** [S/M] MLA-aware `rank_monitor`: skip TP division of
  `kv_override` for replicated-KV configs, mirroring
  `_kv_cache_replicated_across_tp` (`prefill_guard.py:75-80`;
  `planner.py:788-800`). (§B2)

### Phase 2 — Robustness, needs design judgment (gate 2.1 on 0.3)

- [ ] **2.1** [M] Fail-stop on unilateral generation failure: hook
  `_TelemetryQueue.put`'s `BaseException` branch (telemetry.py:609-610) +
  wrap `TelemetryBatchGenerator.next()` (telemetry.py:793); allowlist the
  agreed-rejection types; marker `failed` + `os._exit(1)` on distributed
  ranks. Validate against the 0.3 harness on both paths and both roles
  (worker / rank 0). (§A1 part 1)
- [ ] **2.2** [M] Generation-progress heartbeat: publish step counter +
  active-request flag in the marker (telemetry already tracks both);
  surface "active but not progressing" in `PeerHealth`/dashboard as
  stalled. Observability only — no auto-kill in this phase. (§A1 part 2)
- [ ] **2.3** [M] Data-plane liveness: bounded route+reachability check of
  the hostfile IPs (reuse `verify_link_reachability`,
  `transport.py:1482`) in `PeerWatchdog`/preflight; distinct
  failure message for "SSH up, fabric down". Decide kill semantics from 0.3
  evidence per backend (ring may already fail-fast; jaccl is the open
  case). (§C1)
- [ ] **2.4** [M] (Only if biased/latent nemotron_h checkpoints become a
  target) real TP support for per-rank biases: zero retained biases on
  rank≠0 under `_wrap_sharded_moe` recombination + shard the latent
  projections, with a synthetic-config numerics test. Supersedes the 1.2
  refusal for those configs. (§B1 option 2)

### Phase 3 — Performance (3.2 gated on 0.1; 3.1 informed by 0.2)

- [ ] **3.1** [S/M] Stale-while-revalidate peer health: background refresh,
  parallel per-peer reads; block only with no verdict or unhealthy verdict
  (`engine/distributed.py:1310-1341`). (§D1)
- [ ] **3.2** [M] Async SSD snapshot writer (single writer thread, bounded
  queue) — only if 0.1 shows material prefill stalls; the boundary vote
  already tolerates missing writes. Add `max_bytes` to the store wiring
  regardless [S] (telemetry.py:679-681). (§D2)
- [ ] **3.3** [S] Record strategy benchmarks from non-streaming
  `chat()`/`generate()` (engine/distributed.py:966-1041). (§D4)
- [ ] **3.4** [S] (Only if instrumentation shows seeded/sequential
  distributed traffic) batch the per-token stop vote every k tokens
  (telemetry.py:896-916). (§D3)
- [ ] **3.5** [M, low priority] Peer-side collective-port-span preflight or
  bind-failure retry (`launch.py:126-168`). (§E1)

### Explicitly not doing (this review)

- **MTP-over-cluster**: lives on `feat/mtp-tensor-parallel-clustering-v2`;
  when it merges, re-review the seams it crosses — the synchronized sampler
  assumption (§A3), `_validate_model_settings`'s gate, and the
  `n_confirmed` guard in `pipeline_compat.py` — rather than this whole doc.
- **>2-rank TP** determinism validation (§A3) until such a deployment is
  planned.
- **CUDA/heterogeneous paths** (`cuda_worker_bootstrap.py`,
  `nccl_fabric_worker.py`): skimmed only; out of scope for this pass.
- **SSH multiplexing / ControlMaster** beyond the 1.4 check — a broader
  change to `ssh_policy.py` with its own failure modes.

---

## 8. Already resolved / verified non-issues — do not re-investigate

- **#2844 Nemotron TP=2 (`d85c0d9f`) + follow-up (`15ed4841`) — verified
  present and correct at HEAD.** The uneven quant-group MoE split
  (`_shard_switch_mlp_uneven`, `tensor_strategies.py:275-328`;
  `_uneven_group_ranges` math re-derived: contiguous group ranges, low ranks
  absorb the remainder, fc1 neuron slice `groups → neurons_per_group` and
  fc2 packed-axis slice `packed_per_group` are mutually consistent), the
  quantized-in_proj row gather with matching scales/biases
  (tensor_strategies.py:546-560), and the planner's fail-closed divisor
  gates including per-module oQ quant-group overrides
  (`_tensor_parallel_divisors`, planner.py:612-674). Do not re-derive.
- **qwen3_next TP adapter sharding layout — verified correct** against the
  pinned model: the fused `in_proj_qkvz`/`in_proj_ba` are packed
  **per-k-head-group** (`fix_query_key_value_ordering` reshapes to
  `(..., nk, -1)`, `site-packages/mlx_lm/models/qwen3_next.py:214-234`), so
  `shard_linear`'s contiguous row split hands each rank whole k-head groups —
  consistent with the halved head counts; the conv1d gather indices
  (tensor_strategies.py:371-384) match the component-contiguous
  `concat(q,k,v)` conv input built at qwen3_next.py:254-256. The block's own
  `sharding_group` mechanism (qwen3_next.py:351-352) is unused by the oMLX
  adapter (which wraps externally instead) — one all_sum, not two.
- **Sequential-cancel `NotImplementedError` wedge — FIXED by
  `CoordinatedGenerationContext`** (telemetry.py:896-916,951-964): mlx-lm's
  distributed `_serve_single` raises `NotImplementedError` on cancel
  (`site-packages/mlx_lm/server.py:1008-1010`); oMLX suppresses that guard
  (`self._is_distributed = False` inside `_serve_single` only) and replaces
  it with a per-token all-rank stop vote, so every rank leaves together at
  the same token boundary. Verified: the vote count is symmetric (the check
  precedes the `finish_reason` break on every rank; batched reads stay local
  via the thread-local `sequential` flag). Residual cost is D3.
- **Batched-path cancellation** was already rank-agreed upstream
  (`uids_to_remove = self._share_object(uids_to_remove)`,
  `site-packages/mlx_lm/server.py:908`).
- **0.6.2 role-flap / Workstation-reserve fixes — verified in code.** The
  role now always contributes its reserve unless a manual memory limit
  explicitly replaces it (`_reserve_bytes_for`, routes.py:516-538); all plan
  paths build budgets at one construction site (`_node_budgets`,
  routes.py:541-590, closing the auto-tune re-plan that dropped role+cap);
  plan-approval mismatches are refused with "This is not the plan you
  approved" (routes.py:3128-3140); dashboard interpreter-fallback flapping
  (#2680) fixed in `probe_remote_admission_ceiling` (launch.py:1141-1263).
  Tests: `test_cluster_status_flap.py`, `test_cluster_plan_approval.py`,
  `test_cluster_split_control.py`.
- **PeerWatchdog self-watch suicide pact — fixed.** `_peer_hosts_by_rank`
  excludes the watching rank with the incident documented in its docstring
  (inference_worker.py:529-550); only rank 0 runs the SSH watchdog
  (remote ranks use the launcher-parent watchdog, inference_worker.py:553-582);
  idle self-kill fixed by the telemetry idle heartbeat (telemetry.py:21-32).
- **Marker lifecycle hardening — verified.** Dead-owner markers treated as
  absent (`marker_owner_is_live`, liveness.py:269-286); remote marker ages
  computed against the **peer's own clock** returned by the same query
  (`_REMOTE_MARKER_SCRIPT` `peer_now`, liveness.py:164-248), eliminating the
  cross-Mac clock-skew false-stale; failure markers preserved as crash
  evidence and read back by `_runtime_failure_reason` (launch.py:2137-2178);
  remote reap validates deployment/plan/rank/cmdline identity before
  signaling (#2722, launch.py:2004-2070).
- **Rank memory admission — deliberate, coherent design** (memory_guard.py +
  node_role.py): planner reserve and guard admission derive from one
  `admission_fraction`; headless ranks admit exactly what single-node would;
  the load peak is watched (`LoadMemoryWatchdog`) rather than pre-charged;
  the effective (unpinned) stage is guarded, not the planned one
  (`_guard_effective_stage`, inference_worker.py:797-861);
  `kv_cache_bytes` round-trips through the plan decode (the 40+20 GiB
  under-charge is fixed, deployment.py:197-204).
- **Prompt-cache coherence invariant** — `tune_execution_settings` forces
  `prompt_cache_size=1` + no byte eviction on every path including
  auto-tune-off (`performance.py:293-364`), because byte-based LRU diverges
  across unequal ranks and desyncs the next request's collectives. The SSD
  tier is exempted from this constraint by design: its per-boundary all-rank
  vote (`agree_ssd_boundary`) makes divergent rank-local disk state safe, and
  the collective is taken on every request, hit or miss, so vote counts stay
  matched (telemetry.py:855-861).
- **SSD snapshot store correctness properties — verified**: chain files with
  per-boundary full state for non-sliceable members, newest-slab-only for
  plain KV; wire stand-ins for unserialisable states; self-disable on
  unserialisable types; deterministic same-decision LRU across ranks;
  `extract_cache` is non-destructive (`site-packages/mlx_lm/generate.py:1684`),
  so the batched boundary snapshot does not perturb the batch. Residuals are
  perf-only (D2).
- **Pure-TP vs pipeline seams — verified**: `pipeline=tensor_parallel_size == 1`
  in `_server_arguments` (inference_worker.py:402), sampling-rank-only and
  prefill-overlap disabled under pure TP
  (`runtime_optimizations.py:215-217`), double-sharding prevented (TP applied
  layer-by-layer in the progressive loader only,
  inference_worker.py:1142-1143), `_validate_loaded_stage` accepts the
  complete-stage TP shape (inference_worker.py:660-676).
- **Ring all_sum smoke** (`collective.py`) validates the collective path
  and cross-rank checksum/token agreement at world=2 — see A3 for the >2
  constraint.
