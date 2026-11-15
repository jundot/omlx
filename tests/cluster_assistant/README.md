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
  coordinator through a stored measurement, the notes plan object, real endpoint
  port, explicit TP rejection, distinct stale/infeasible 409s, no side effects,
  and the exact route surface.
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
- `test_any_to_any.py` — the app origin is not an input to the plan.
- `test_pp_plan.py` — 3/4-Mac pipeline plans: contiguous stages, rank 0 on the
  tail, every stage inside its admission, the fewest Macs that fit.
- `test_election.py`, `test_measurement*.py`, `test_calibration*.py`,
  `test_policy*.py`, `test_topology*.py`, `test_delete_model_api.py`,
  `test_performance_store.py`, `test_probe_spec_expectations.py`,
  `test_diverse_candidates.py` — the decision half (also shipped first).
- `test_search.py`, `test_search_api.py`, `test_search_wiring.py`,
  `test_evaluation.py` — the automatic constellation search: the loop, its
  route wiring, the apply/start/measure adapter and the measured ranking.
- `test_tuning.py`, `test_tuning_store.py`, `test_tuning_wiring.py`,
  `test_tune_api.py`, `test_redeploy_tuning.py`, `test_tune_redeploy_api.py` —
  fine-tuning with real completions, the saved per-(model, hardware) preset and
  the MTP redeploy sweep.
- `test_session_api.py`, `test_stop_api.py`, `test_observed_api.py`,
  `test_diagnostics_api.py`, `test_network_preflight.py`,
  `test_probe_toggle_api.py`, `test_two_origin_lock.py`,
  `test_start_robustness.py`, `test_store_failures.py`,
  `test_provisioning_settings.py`, `test_chat_profile_contract.py`,
  `test_auto_free.py`, `test_copy_model.py` — the run backend's edges: reload
  resume, stop, the measured rate block, the support bundle, the one-operation
  lock, store failures and the free-space retry.
- `test_bootstrap.py` — workspace and package wiring.

Run:

```sh
uv run pytest tests/cluster_assistant
```