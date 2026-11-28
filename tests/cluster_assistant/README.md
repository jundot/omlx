# Cluster Assistant tests

Tests for `omlx/cluster/assistant`, runnable off-cluster: no Mac, model, GPU or
SSH is involved. The election is a pure function over measured numbers, and
these tests prove the *decisions*.

- `test_election.py` — coordinator election, scoring order, no local fallback.
- `test_measurement.py` — oMLX measurements → `NodeFacts`; a measurement that
  did not happen stays missing; endpoint host/port mapping.
- `test_measurement_state.py` — persisted `MeasurementSet` records: restart
  round trip, corrupt/atomic/bounded file, `measurement_id`s, staleness TTL and
  node-set/model/MTP mismatch.
- `test_measure_api.py` — `POST /admin/api/cluster/assistant/measure`: returns a
  handle before the first probe finishes, reaches `ready`, fails with exactly
  one action, one job at a time, and no staging/launch/registry call.
- `test_plan_api.py` — `POST /admin/api/cluster/assistant/plan`: elected
  coordinator through a stored measurement, §13.3 plan object, real endpoint
  port, explicit TP rejection, distinct stale/infeasible 409s, no side effects,
  and the exact seven-route surface.
- `test_recommendation.py` — model recommendation: the 40 GiB model on the
  M1+M5 pair names M5 as rank 0, a model that needs TP is refused, the largest
  context that still plans wins, staging bytes and unknown disk, no invented
  speed, stable ordering, and one shared budget builder.
- `test_recommend_api.py` — `POST /admin/api/cluster/assistant/recommend`:
  happy path through a real `/measure`, the same stale 409 as `/plan`, the
  `no_feasible_candidate` 409 with near misses, an unreadable candidate, and no
  probe/settings/write side effects.
- `test_operations.py` — the persisted operation: forward-only states,
  terminal `applied`/`ready`, one apply/start at a time, restart recovery as
  `failed`, atomic/bounded/mode-0600 file, fail-closed load, and no secrets.
- `test_apply_api.py` — `POST /admin/api/cluster/assistant/apply`: the same flat
  stale 409 as `/plan`, model/MTP/node-set/signature mismatches, the
  single-Mac and unpaired-Mac refusals, staging that never activates, the
  coordinator's recomputed-placement warning, and the operation handle.
- `test_start_api.py` — `POST /admin/api/cluster/assistant/start` and
  `GET .../operations/{id}`: only an applied plan starts, the elected
  coordinator is the one that launches (never the origin), the remote path goes
  through the existing SSH seam, a failed start has one action and starts
  nothing locally, `endpoint.ready` needs health plus one smoke completion,
  states are monotonic, and no rate appears anywhere.
- `test_bootstrap.py` — workspace and package wiring.

Later phases: `test_any_to_any.py`, `test_pp_plan.py`.

Run:

```sh
uv run pytest tests/cluster_assistant
```