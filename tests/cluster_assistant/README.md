# Cluster Assistant tests

Tests for `omlx/cluster/assistant`, runnable off-cluster: no Mac, model, GPU or
SSH is involved. The election is a pure function over measured numbers, and
these tests prove the *decisions*.

- `test_election.py` — coordinator election, scoring order, no local fallback.
- `test_any_to_any.py` — the app origin is not an input to the plan.
- `test_pp_plan.py` — 3/4-Mac pipeline plans: contiguous stages, rank 0 on the
  tail, every stage inside its admission, the fewest Macs that fit.
- `test_measurement.py` — oMLX measurements → `NodeFacts`; a measurement that
  did not happen stays missing; endpoint host/port mapping.
- `test_measurement_state.py` — persisted `MeasurementSet` records: restart
  round trip, corrupt/atomic/bounded file, `measurement_id`s, staleness TTL and
  node-set/model/MTP mismatch.
- `test_measure_api.py` — `POST …/measure`: returns a handle before the first
  probe finishes, reaches `ready`, fails with exactly one action, one job at a
  time, and no staging/launch/registry call.
- `test_measure_calibration.py`, `test_calibration.py`,
  `test_calibration_runner.py` — the real-model calibration the measurement
  folds into a node's compute, and its published-spec ordering.
- `test_plan_api.py` — `POST …/plan`: elected coordinator through a stored
  measurement, the notes plan object, real endpoint port, explicit TP rejection,
  distinct stale/infeasible 409s, no side effects, and the exact route surface.
- `test_recommendation.py` — model recommendation: the 40 GiB model on the
  M1+M5 pair names M5 as rank 0, a model that needs TP is refused, the largest
  context that still plans wins, staging bytes and unknown disk, no invented
  speed, stable ordering, one shared budget builder.
- `test_recommend_api.py` — `POST …/recommend`: happy path through a real
  `/measure`, the same stale 409 as `/plan`, the `no_feasible_candidate` 409
  with near misses, an unreadable candidate, and no probe/settings/write side
  effects.
- `test_policy.py`, `test_policy_api.py` — the resource policy: the subset
  solver, the levers, and what the plan reports for them.
- `test_topology.py`, `test_topology_api.py` — the measured line, the
  `topology_incomplete` refusal and the signed 2-Mac consent.
- `test_delete_model_api.py` — the free-space rule and its one relay route.
- `test_probe_spec_expectations.py` — the compute probe's published-spec
  ordering and the plausibility guard.
- `test_performance_store.py` — the stored compute profile the next
  measurement uses.
- `test_bootstrap.py` — workspace and package wiring.

Run:

```sh
uv run pytest tests/cluster_assistant
```