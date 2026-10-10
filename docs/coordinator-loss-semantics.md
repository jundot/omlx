# Coordinator-Loss Semantics

When a coordinator node becomes unavailable in a distributed deployment, the
deployment ends. No rebalancing, no failover, no rank migration. This document
describes why.

## Failure Sequence

1. The MLX launcher process (rank 0) dies with the coordinator node.  It is
   the SSH parent of every remote rank process, so all remote ranks are
   immediately reparented to `init`.

2. Every remote rank's parent-PID watchdog (`_watch_launcher_parent` in
   `inference_worker.py:492`) detects the reparenting within one poll interval
   (200 ms), writes a `launcher_lost` runtime marker, emits a
   `launcher_lost` event, and calls `os._exit(1)`.

3. Each rank holds its shard's weights in unified memory.  When a rank exits,
   that shard is freed.  No node has ever held the full model — each rank only
   held its pipeline stage — so no surviving node can serve the model alone.

4. The deployment record persists in `~/.omlx/cluster/deployments.json` (the
   `ClusterRegistry`).  Nothing is running, but the record still exists.

5. Each node's own standalone HTTP server (the oMLX server started on that Mac)
   is unaffected and keeps serving whatever models it can load independently.
   It does not participate in distributed inference without an explicit
   activation.

## Why No Automatic Failover

### mx.distributed has no membership-change protocol

MLX's distributed collective backend (`mx.distributed`) is built on fixed
world-size collectives: every rank must call every collective operation.  There
is no membership-change protocol, no dynamic rejoin, and no way to tell an
existing collective that a rank has left.  A collective with a missing rank
deadlocks — every surviving rank blocks inside the all-reduce waiting for a
peer that will never respond.  The parent-PID watchdog prevents this deadlock:
instead of blocking, each remote rank exits immediately when it detects the
coordinator is gone.

### The plan is content-addressed and immutable

A deployment's `plan_hash` is a SHA-256 digest of the full shard placement
(`planner.py:1548`).  The activation endpoint (`routes.py:3125`) compares the
caller's `approved_placement` signature against the computed plan; if they
differ, activation returns HTTP 409.  This means a recomputed placement — the
kind an automatic failover would need — would fail the signature check.  The
system cannot silently change who holds what layers without operator approval.

### Shards are bound to rank index

Each rank's `PipelineAssignment` (`deployment.py:182`) specifies a fixed layer
range (`start_layer`, `end_layer`) tied to the rank's index in the host list.
There is no mechanism to reassign a layer range to a different node at runtime.
The assignment was validated at deployment time, signed into the plan hash, and
encoded into the worker plan the launcher distributes.

## Recovery Is Manual Re-activation

After a coordinator loss:

1. The operator re-activates the deployment through the oMLX dashboard or API.
   The activation endpoint (`POST /admin/api/cluster/deployments`) recomputes
   the shard plan, checks peer health, and launches new rank processes.

2. The deployment record in `deployments.json` persists across the failure.
   If the same model and nodes are used, the plan hash may be identical.
   Otherwise, a new plan is computed, a new placement is signed, and the
   operator must approve it again.

3. Model shards are already staged on each node's filesystem from the original
   activation (the staging step copied them).  Re-activation does not need to
   re-stage unless the model path or node set has changed.

### Deactivation

The `DELETE /admin/api/cluster/deployments/{deployment_id}` endpoint
(`routes.py:3382`) removes the deployment record from `deployments.json` and
unregisters the cluster model from the engine pool.  This is the explicit
cleanup path when the operator does not intend to re-activate.

## Standalone Fallback

Each node runs its own HTTP server independently of the distributed deployment.
When the coordinator dies:

- The standalone oMLX server on each Mac continues serving whatever single-node
  models it can load (the catalogue probes each node for standalone capacity in
  `catalogue.py:433`).

- A distributed deployment does not prevent standalone serving.  The catalogue
  reports a `standalone_node_id` and `standalone_max_context_tokens` for any
  node that fits the model alone, so the operator knows which nodes can serve
  independently after a coordinator loss.

- Standalone serving is the default path.  Distributed inference requires an
  explicit activation that loads shard weights across the cluster.  Losing the
  coordinator removes the distributed path but does not touch the standalone
  path.
