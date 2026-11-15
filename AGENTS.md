# AGENTS.md — oMLX Cluster Assistant

This repository is a focused oMLX fork for the Cluster Assistant project.

The single source of product decisions is:

**`docs/cluster-assistant/PROJECT-BRIEF.md`**

Read it before changing code. In particular read:

- Section 7 — Coordinator election
- Section 8 — PP planning
- Section 9 — Any-to-any install and start
- Appendix A — Cold-start handoff for the next agent
- `docs/cluster-assistant/CHAPTER-5.md` — current apply/start chapter
- `docs/cluster-assistant/SCREEN-3-MINIMAL.md` — minimal first UI slice

If this file and the brief disagree, the brief wins. If the code and the brief disagree, inspect the code and update the brief only with an explicit decision.

---

## 1. Frozen decisions

Do not re-open or silently change these:

1. This is **not** exo.
2. This is **not** a second LocalInferenz app or control plane.
3. Build inside the existing oMLX app/web UI/cluster control plane.
4. MVP is **PP only**. No tensor-parallel behavior or UI.
5. Product UI is **English only**. No German localization.
6. Support **2–4 Apple Silicon Macs over Thunderbolt**.
7. The app origin must not determine rank 0.
8. Rank 0, coordinator, and endpoint are co-located in the MVP.
9. The coordinator supervisor runs on the elected Mac; rank 0 remains local there.
10. Feasibility wins over performance.
11. Free admission is a strong rank-0 preference, not the only criterion.
12. Never silently fall back to the local Mac as rank 0.
13. Reuse oMLX discovery, pairing, staging, activation, chat, and client integrations.
14. The current implementation milestone is Chapter 5 apply/start backend; UI stays out until that backend is stable.

If a task appears to require changing one of these, stop and ask for an explicit decision.

---

## 2. Repository baseline

- Upstream base: `upstream/main` `624cff0e` (rebased 2026-11-14; `upstream/main` is
  an ancestor of HEAD).
- Branch: **`rebase/upstream-main`** — current working branch, the fork replayed
  onto `upstream/main`. Pre-rebase history is kept untouched on
  `feat/cluster-assistant` (`28245931`) and `backup/pre-rebase-2026-11-14`.
- Historical base: oMLX `1e27576af5e10c8f4459e0fdcb05c5cf59245d44` (tag
  `cluster-assistant-base-2026-11-07`). It is no longer the pin; see
  `docs/cluster-assistant/REBASE-2026-11-14.md`.
- Origin: <https://github.com/digitalfactory-ako/omlx>
- Upstream: <https://github.com/jundot/omlx>
- Patch/PR status: `docs/cluster-assistant/UPSTREAM.md`
- Reference prototype: `~/Documents/PycharmProjects/localinferenz`
- Reference installed oMLX checkout: `~/Library/Application Support/LocalInferenzMachine/src/omlx`

Do not assume every local patch branch is included. Check `UPSTREAM.md` and
`git merge-base --is-ancestor` before relying on a commit.

Resolved implementation decisions:

- Measurement is an asynchronous job that persists a `MeasurementSet` and
  returns a `measurement_id`.
- `plan` is synchronous, pure, and requires `measurement_id`; it never probes.
- `endpoint.port` comes from the elected coordinator's settings (default 8099
  only when unreadable). Never return port 0.
- The plan exposes `predicted.critical_stage_seconds` only; measured token
  rates belong in `observed` after the smoke test.
- Measurements persist in `<base_path>/cluster/assistant-measurements.json`,
  are stale 15 minutes after `measured_at`, and one job runs at a time (a
  second `measure` is refused with 409, not joined).
- Recommendation uses the hardware-only part of a ready measurement and may
  evaluate any catalogue model (Option A). It never relaxes `plan`: a chosen
  model must be measured model-bound again before `plan`, and plan keeps the
  strict node-set/model-identity/`mtp` match.
- Hardware-only measurement (`model: null`) is allowed for recommendation and
  is never accepted by `plan`.
- Recommendation is implemented in `omlx/cluster/assistant/recommendation.py`
  and exposed as `POST /admin/api/cluster/assistant/recommend`: `measurement_id`
  plus 1–32 candidates (`model_path`, optional `friendly_name`, optional
  `declared_context_tokens`), an optional target `context_tokens` and an
  `execution_profile`. Freshness/TTL are `state.measurement_refusal`'s rule; the
  model identity is not compared. Feasibility is `elect()` per candidate, PP
  only, and the context is the largest ladder value at or below the target and
  the declared limit that still plans. No feasible candidate is
  `409 no_feasible_candidate` with the near misses and one action.
- `election.budget_for(facts, rank=…)` is the single conversion from measured
  admission to a `NodeBudget`. Do not build budgets anywhere else.
- **The plan uses the fewest Macs that fit.** `elect` enumerates subsets from
  2 upward; the smallest feasible subset wins and must not grow to 3/4 Macs
  just because they are paired (three hops instead of one). Extra Macs are
  reported as `excluded_nodes` with the reason. Within the chosen size, rank by
  measured critical stage, then headroom, then fewer hops, then the existing
  coordinator preference; without measured compute, prefer headroom and record
  `unmeasured_compute` rather than inventing a performance ranking. **The
  compute probe must be bandwidth-bound, and an implausible probe counts as
  unknown compute** (decision E4=B, 2026-11-13): the old latency-bound
  `rows=1` GEMV reported hardware-impossible rates, so a probe that fails a
  plausibility check is treated exactly like missing compute (headroom
  preference + `unmeasured_compute`); only a trustworthy probe ranks by
  critical stage. Never estimate a rate to fill a gap. The
  measurement still covers every paired Mac; plan/apply/start bind to the
  selected subset, and the plan signature includes it. An advanced
  `node_selection=all` may force every Mac; it is not the default. A single-Mac
  plan remains unapplicable (`ClusterDeployment` needs 2+ hosts).
- **Tuning is a saved preset, not a re-tune on every Start (2026-11-14).**
  `TuningPresetStore` persists the winning knobs per (model, node config) at
  `<base_path>/cluster/assistant-tuning-presets.json`, keyed by
  `tuning.preset_key(model_signature, hardware_key)`, atomic and fail-closed.
  `/tune` saves the sampler preset; Start replays it through
  `start.apply_tuning_preset` **before activation** (so `mtp_adaptive_max_depth`
  is the launched depth). The tuning preset is **authoritative**: it writes its
  measured values over a leftover setting (only `None` fields are skipped),
  because it is the user's tuned winner — a sweep's last config must not win over
  it. (The chat profile keeps the opposite "fill only empty" rule.) Each outcome
  is `tuning_preset_applied`/`tuning_preset_skipped`. The MTP switch and depth are
  per-model settings, so they are tuned by redeploy: `redeploy_tuning.py` plans
  configs and ranks real measured rates, and `POST /tune-redeploy` (background
  job) stops the running cluster first, reuses the proven `diverse_candidates` +
  `evaluate_constellation` body, stops each config, and merges the winner into the
  preset. MTP on/off can flip the optimal rank order (measured 2026-11-14: rank 0
  moved from the M2 Ultra to the M5 Max when MTP was off), so `choose_candidate`
  measures the best fewest-Mac split at the target context for that config rather
  than a fabricated infeasible row. A model without an MTP head offers only
  `mtp_off`; a failed deploy is an infeasible row, never a rate; the
  `prefill_chunk_size` field does not exist in this build, so the chunk stays
  honestly "not applicable". Live result (PHASE-2B-RUN.md §27): GLM-5.3 depth 3 =
  31.6 tok/s vs MTP off 22.07 (+43 %) on the `.12`+`.11` line.
- **Freeing disk is automatic when a model must be distributed (2026-11-14).**
  An `insufficient_disk` apply no longer stops at `free_disk_space`: it frees the
  safest unused model(s) on the short Mac per the frozen rule — only a model that
  Mac listed, never the checkpoint the cluster is about to run, never the last
  complete copy across the measured Macs, largest first — retries once with the
  freed bytes counted, and records one `space_freed` warning. `reclaim.py` is the
  pure selection; the route does the relay delete and the retry. The delete route
  (`DELETE …/assistant/models/{node}/{model}`) stays the manual affordance. The
  search/tuning loop never auto-deletes.
- **Context vs experts is the user's choice, at plan time (2026-11-14).** The
  assistant decides what costs no speed — a checkpoint's lookup table
  (Engram/PLE) is offloaded by default (`_levers` base lever), which frees RAM for
  routed experts and is not silent (the plan shows it, Start writes it). What
  costs something is the user's call: reducing context loses quality but keeps
  speed, while MoE expert offload keeps quality but costs speed. The plan returns
  the **quality-first** result (largest fitting context, expert offload allowed,
  even a worst case like 8k + 75 % if that is the only way it runs) and, when
  experts are offloaded, a `tradeoffs` list with the best **experts-resident**
  alternative (`label: speed`, smaller context, its own `resource_policy` +
  `plan_signature`). The user picks; `apply` uses the chosen `resource_policy` and
  signature. There is no upfront context-ceiling oracle and no silent context
  drop.
- **Routed-expert (MoE) offload is priced again after the 2026-11-14 rebase.**
  Upstream removed it from the planner; it is restored on upstream's refactored
  pipeline (`plan_hybrid`/`plan_unequal_pipeline`/`_finish_pipeline_plan`/
  `_held_layer_bytes` + `PipelineAssignment.moe_offload_*`; a `()` default equals
  upstream) and threaded through the coordinator plan path
  (`routes._saved_moe_offload`/`_moe_offload_layer_bytes`/`_moe_offload_for_plan`)
  and `deployment._assignment_from_dict`. DeepSeek-V4.1 only; other families keep
  upstream's resident plan; TP > 1 is refused; never invent bytes.
- **A stale cluster deployment is never auto-unloaded across coordinators
  (2026-11-14).** `local_only` on a model load (`POST /api/models/{id}/load
  ?local_only=true`) already prevents the real bug: a single-Mac load adopting a
  stale cluster deployment of the checkpoint and comparing its per-rank shards.
  Beyond that, a cross-coordinator auto-unload is refused by decision — it could
  stop another origin's or a user's serving deployment. The supported cleanup is
  the manual unload on the owning coordinator (`/deployments/{id}/unload`, Screen
  6) plus the `diagnostics` action a memory-pressure start reports; each
  coordinator stays the authority on its own registry.
- **The resource policy is one object across the whole flow (Phase A,
  2026-11-07).** `policy.ResourcePolicy` carries the context, table offload
  (`engram`/`ple`/`none`), MoE expert offload and resident fraction, memory-guard
  tier, microbatch and admission percentile through measure → recommend → plan →
  apply → start. `solve_cluster_plan` is a loop around `elect`, not a second
  planner: every feasibility verdict is still `plan_hybrid`. Defaults come from
  the saved settings; the solver's lever order is context first (quality), then
  resident/saved, table offload, MoE 0.75/0.5/0.25, a higher tier, and finally
  the p95 admission with explicit consent. Every lever is shown in the plan and
  applied on Start setup; a drifted policy is `plan_mismatch`. The apply step
  writes the chosen per-model switches on every **selected** Mac before staging
  and fails with one action if a write fails; excluded Macs are never written,
  staged or started. The plan reports `selected_node_ids`, `excluded_nodes` and
  `resource_policy`; the card renders Macs/Tier/Context/Offload/Experts and an
  advanced "Use all Macs" override. `p05` is the floor, `p50`/live admission
  plans by default, and `p95` requires `allow_optimistic_admission`.
- **Table offload parity:** `_saved_engram_ssd_offload` honours either the
  DeepSeek V4.1 Engram switch or the Qwen4 PLE switch, and the planner reads
  Qwen4 `ple_layer_ids` tensors into the same per-layer SSD bytes as DeepSeek
  Engram. An explicit resident plan for a table-capable checkpoint writes the
  switch off, because the plan showed it and Start setup confirms it.
- **Pipeline capability is part of the plan.** `inspect_safetensors_layout`
  applies oMLX's pre-load architecture patches before capability detection, and
  `solve_cluster_plan` refuses a multi-Mac plan for a checkpoint whose
  `supports_pipeline` is false with `choose_pipeline_model`. **Qwen4-Exp /
  Qwen3.8 is PP-capable on this branch** (upstream PR #5033 is applied):
  `supports_pipeline` is true for that family and the assistant may plan it as a
  pipeline; the PLE table detection stays alongside. **Release rule (user
  decision, 2026-11-14): PP only in R1 — Qwen included; general tensor-parallel
  support comes in R2.** The earlier note that the Qwen3.8 PLE VLM is "never
  planned as PP" described the pre-#5033 state and is withdrawn.
- The synthetic distributed compute probe runs **after `apply`**, on the
  elected coordinator, through the coordinator's
  `POST /admin/api/cluster/performance-probe`; its profiles are stored on the
  origin and used by the **next** measurement/plan. `auto_tune` stays false and
  the probe never rewrites the current `predicted`, never gates apply or start;
  a failure is one `compute_probe_failed` warning. `compute` is `None` only when
  no profile exists; never estimate a rate instead.
- Chapter 5 is the only place that may wire staging, deployment, launch,
  supervisor start and health. `measure`, `recommend` and `plan` stay inert.
- Chapter 5 is blocked only by apply/start. The performance probe is not a
  gate; full Screen-3 UI comes after the backend. Model-less (hardware-only)
  measurement is implemented (Phase C3, see the Screen-3 entry bullet below).
- Apply/start must start the supervisor on the elected coordinator, use the
  existing oMLX launch/SSH path, and never silently fall back to the local Mac.
- Operation status is separate from measurement status. `endpoint.ready` stays
  false until health and one smoke completion succeed.
- Do not add token-rate estimates to fill a missing performance probe. Leave
  `predicted.measured=false`, `measured_speed=false`, and `observed=null`.
- The minimal Screen-3 UI is blocked only by the UI itself. It binds
  `selected_model`/`selected_context_tokens`, posts apply/start, polls
  operations, hides `endpoint.url` until `ready=true`, and renders
  `error.action`.
- `retry_coordinator` means: open/configure oMLX on the elected coordinator and
  retry. `node_not_paired`/`pair_mac` means: pair the origin Mac and retry.
  Neither path may start the local Mac as a fallback.
- Browser JS must never launch, deploy, SSH, stage or supervise. It only calls
  the assistant API.
- Screen-3 entry measurement is **Phase C3: hardware-only first** (recorded in
  `docs/cluster-assistant/ROADMAP-TO-CHAPTER-10.md` §6 C3; superseded the
  2026-11-03 Option A entry on 2026-11-13, see `SCREEN-3-MINIMAL.md` §4.1):
  the UI measures bare hardware (`model: null`), polls it ready, calls
  `recommend` over the model list, binds the recommended model/context, and
  measures that model **model-bound** before `plan`. `plan` keeps the strict
  node-set/model-identity/`mtp` match and never accepts a hardware-only
  measurement.
- One tab, one component. The assistant lives inside `clusterV2Wizard()` and
  the existing Cluster tab, with its own `assistantState()` machine and
  `data-assistant-state="…"` markers. It does not add entries to
  `WIZARD_STATES` or the existing `wizardState()` chain. Assistant states are
  `idle → measuring → recommending → measuring_model → planning → applying →
  starting → loading → ready`, plus `failed`; `applying/starting/loading` mirror
  the operation states one-for-one and no state exists that the backend cannot
  report. A computed plan rests in `planning` with `assistant.plan` set.
- Endpoint visibility: `endpoint.url` is rendered (link, copy, Open Chat,
  Connect agent) only when `assistantState() === 'ready'` **and**
  `endpoint.ready === true`; an unresolved endpoint stays `null` and is never a
  port-0 URL.
- No SSE/streaming. One single-flight poll loop per job (measurement and
  operation) with a generation counter, mirroring `pollStagingJob`; a late
  response from an old generation cannot overwrite a newer one, and polling
  stops at `ready`/`failed`. `Start setup` posts `apply`, waits for the apply
  operation to be `applied`, then posts `start`, and never posts `start` from
  an unapplied or failed operation.
- Screen-4 progress is **Option A** (recorded in
  `docs/cluster-assistant/SCREEN-4-PROGRESS.md`): map the existing assistant
  states, operation kind and `progress.{phase,message,nodes_ready,nodes_total}`
  onto the eight Screen-4 labels. No backend change, no SSE. Only six labels
  are independently observable: "Preparing runtime" is combined with "Copying
  model", and "Verifying endpoint" is combined with "Loading model" (the
  readiness canary runs inside the blocking activation call). The real order is
  Checking → Planning → Preparing/Copying → Starting → Loading/Verifying →
  Ready; the brief's label order was corrected because planning must precede
  staging. A row may never animate a phase the backend did not report, and no
  percentage, byte count or token rate is displayed. Implemented 2026-11-07:
  `assistantProgressRows()` renders the six rows from `assistantState()`, the
  operation kind/state and `progress.phase`, `Ready` needs `assistantReady()`, a
  failure keeps the row it stopped in, and an unknown phase keeps the previous
  row with its raw message (`SCREEN-4-PROGRESS.md` §8).
- The assistant plans **Lightning MTP when the checkpoint is MTP-capable**
  (decided 2026-11-06 during the live run): `assistantMtpFor(modelPath)` reads
  the catalogue's `mtp_compatible`, and the model-bound measurement and the
  frozen plan inputs both use it so measurement identity, plan and apply
  cannot disagree. The hardware-only entry measurement (Phase C3) has no
  model to ask about. A model without an MTP head is still planned
  with `mtp: false`; the draft head is never reserved for weights that do not
  exist.
- A started cluster gets a **documented chat profile** when the model has none
  (`start.CHAT_PROFILE`: temperature 0.6, top_p 0.95, top_k 0, `max_tokens`
  2048, `force_sampling`). It is written on the elected coordinator through the
  existing local/remote admin seam, **never overwrites a value the user set**, is
  recorded as the operation warning `chat_profile_applied` /
  `chat_profile_skipped`, and can never fail an already-ready cluster.
  `enable_thinking` is deliberately not written: forcing it at model level was
  measurably worse than leaving it to the model (PHASE-2B-RUN.md §5).
- The assistant's action vocabulary is the frozen twelve
  (`measure | plan | apply | start | status | add_mac | pair_mac | copy_model |
  configure_endpoint | retry_coordinator | diagnostics | adjust_model_or_context`)
  **plus the recommendation's four near-miss actions**
  (`choose_smaller_model`, `choose_pipeline_model`, `check_model_path`,
  `free_disk_space`), which the card renders with their own labels, **plus
  `raise_wired_limit`** (R5, decided 2026-11-15). A refusal's single action is
  taken from the first near miss whose failure is about the cluster, never from
  a candidate that is merely `model_unreadable`.
- **The wired limit is a preflight, not a guess (R5, 2026-11-15).**
  `election.wired_limit_binds(facts)` is true only when a Mac's admitted ceiling
  sits at its `iogpu.wired_limit_mb` — `capacity_bytes` is the minimum of the
  static, live and GPU ceilings, so being *at* the cap is the one case where
  raising the limit admits more. `None` (not measured) never reads as too low; a
  measured `0` is the macOS default nobody raised, which is the case this exists
  for. When no plan fits and a measured Mac's cap binds, the refusal names that
  Mac and its MB value, and its action is `raise_wired_limit`: the app applies
  the recommended limit from Performance → "Apply with Admin…" (one admin
  dialog); the browser has no privileged path and only names the step. A
  refusal that already chose a specific step (a checkpoint mlx-lm cannot split)
  keeps it — more memory does not make an architecture pipeline-capable.
- **Rank 0 can be pinned, and the pin is never overridden silently (user
  decision, 2026-11-15).** `preferred_coordinator` names the Mac that must host
  rank 0 — and with it the endpoint, because rank 0, coordinator and endpoint
  stay co-located. The election restricts candidate generation to that Mac, the
  solver keeps only subsets that contain it, and when no legal split exists with
  it in front the plan refuses with the Mac named (never a different coordinator).
  It is a plan-time input: the request, `PlanInputs`, the operation record and
  the card all carry it, so apply re-elects with it and the approved placement is
  the launched one. The pin cannot make an infeasible split feasible: oMLX's rank
  0 runs the sampler and must own the tail, so pinning a Mac that cannot hold the
  tail is a refusal, not a reordering. On the live 2-Mac GLM run that is why
  `.12` had rank 0: `.12` cannot hold the 66.1 GiB head stage, so the only
  feasible order is `.12` = tail. Pinning `.11` there needs a smaller
  (unbalanced, slower) tail — an explicit choice, not a silent one.
- The measurement records each Mac's own catalogue (`NodeFacts.model_paths`), so
  a recommendation answers "already present" for **any** candidate from measured
  facts. Empty `model_paths` means "not measured" and keeps the honest
  `staging_required` warning; it never means "holds nothing".
- Only a **paired** Mac can be a launch target. `NodeTarget.paired` distinguishes
  "never paired" (`node_not_paired` / `pair_mac`) from "paired but no dialable
  address" (`coordinator_unresolved` / `configure_endpoint`).
- **One apply/start at a time, cluster-wide.** Each origin's store only covers
  itself, so `apply`/`start` ask the elected coordinator through the admin seam
  (`GET /admin/api/cluster/assistant/operations/active`, the assistant router's
  only route the browser does not call) before creating an operation. A second
  origin gets the existing `409 operation_in_progress` + action `status` naming
  the running operation; an unanswered question is never treated as a lock.
- **The measurement owns the model layout.** plan/apply/start re-elect from
  `MeasurementRecord.model_layout()`, never from a fresh read of the checkpoint:
  a shard file that changes while the server runs must not make the same
  measurement elect a different coordinator and fail an applied plan with
  `plan_mismatch`.
- A Mac below **64 GB installed RAM is Headless**, not Workstation: the
  workstation reserve is at least 32 GiB and would leave ~1 GiB of admission.
  The card sends `headless` for it and says so; the backend enforces the same
  rule from the computed budget (warning `workstation_reserve_too_large`), so a
  client that asks for workstation cannot starve a small Mac either.
- A model that lives only on **another Mac** is still usable: when the origin
  cannot read the checkpoint, the measurement asks the Mac whose inventory lists
  it for its layout (`layout_read_on_peer`) and stores it. A copy that cannot fit
  is refused before staging with `insufficient_disk` + `free_disk_space`; the
  assistant never deletes a model itself, and the delete affordance is a recorded
  next slice (`PHASE-2B-RUN.md` §9).
- **Freeing space is a dedicated assistant route.** A copy that cannot fit is refused
  with `insufficient_disk` + `free_disk_space`, and the card may then delete an
  unused model on the Mac that is short:
  `DELETE /admin/api/cluster/assistant/models/{node_id}/{model_id}`, relayed over
  the same admin seam to that Mac's own delete route (worker or coordinator
  alike). Two-step confirm, never automatic, and never the model the cluster is
  about to run. The browser still never deletes anything itself.
- **Screen 5 preselects in the existing `/chat`.** `/chat?model=<id>&endpoint=<url>`
  preselects the model and, for a remote endpoint, uses it as the API base. No
  new page, no second chat. Implemented 2026-11-07 as **model-only**:
  `Open Chat` builds `/chat?model=<serving id>` when the coordinator is this Mac
  and `<endpoint.url>/chat?model=<serving id>` when it is another Mac (that
  Mac’s own page; its session and key apply, so the user may sign in there
  once). The serving id is the catalogue’s `modelOptions` id resolved by
  `model_path`; an unresolved id links without the parameter and never guesses.
  `Connect agent` shows `<endpoint>/v1`, the serving id and where the key lives
  (“Settings → API key on <coordinator>”), never the key. The chat page reads
  `model` once on load, selects it only if available, adds no history entry and
  reads no credential. `endpoint` is not read.
- **`observed` is one nullable operation field, filled only after `ready`.**
  `OperationRecord`/`AssistantOperationStatus` carry
  `observed: {prefill_tok_per_s, decode_tok_per_s, ttft_ms, measured_at} | null`.
  It is filled by one real non-streaming completion on the elected coordinator
  (oMLX’s own decode `usage`) plus one streaming completion whose final usage
  chunk supplies TTFT and prefill (`stream_options.include_usage`); both go
  through the same local/remote admin seam apply/start use, never the 2-token
  readiness canary, never the pre-launch synthetic weight-byte probe, never an
  estimate. Switch: `ServerSettings.assistant_performance_probe`, default
  **false**, read on the origin that posts `start` and toggled in-app via
  `GET/POST /assistant/performance-probe`; the ready card also has a bounded
  Refresh that re-reads the operation once so a late block appears without a
  reload. `predicted` is never rewritten, `plan.observed` stays null,
  `auto_tune` stays false, and a probe failure only adds the warning
  `performance_probe_failed` to the already-`ready` operation. The card renders
  a measured line only when the block is present; otherwise it says nothing
  about speed. Decided and recorded in `SCREEN-5-HANDOFF.md`; B1/B2/B3 live
  evidence is in the untracked `REBASE-VERIFICATION-2026-11-07.md`.
- **Screen 6 is state + stop/restart, nothing more.** The ready card carries
  `Stop cluster`, which calls the tenth assistant route
  `POST /admin/api/cluster/assistant/stop` with the ready start operation's ID.
  The backend relays to the elected coordinator's existing
  `POST /admin/api/cluster/deployments/{id}/unload` through the same local/SSH
  seam, using the deployment ID, coordinator and port recorded on the operation
  — it never re-elects, so a stale measurement cannot block a stop. The record
  gains `deployment_id`, `coordinator_local` and `stopped`; `state` stays
  `ready` (the frozen machine has no stopped value) and `stopped` is the
  authoritative boolean. A stopped operation hides the endpoint and offers
  `Start again`, which re-applies the frozen plan inputs and lets the
  existing poll start it. Re-planning, editing the model or context of a running
  cluster, and adding/removing a Mac stay out. **After a reload the card
  resumes** (Phase C1, recorded in `ROADMAP-TO-CHAPTER-10.md` §6 C1):
  `GET /assistant/session` returns the newest ready start (else applied apply)
  with its frozen plan inputs, coordinator, deployment ID and stopped flag —
  no key, SSH target or secret. The session strip shows Running/Stopped and
  offers `Start again`/`Stop`; apply re-proves the plan signature, so a drifted
  placement is still `plan_mismatch`. The existing deployments list remains a
  return surface, not the only one.
- **The performance probe runs after `ready`, asynchronously, on the elected
  coordinator.** It may fill one nullable `observed` block on the operation
  (`prefill_tok_per_s`, `decode_tok_per_s`, `ttft_ms`, `measured_at`) and a
  performance profile on the coordinator. `predicted` is never rewritten, and
  with the probe off (the default) `observed` stays `null` — the UI then says
  nothing about speed.
- **Screen 6 is state + stop/restart, nothing more.** Stop is the coordinator's
  existing `POST /admin/api/cluster/deployments/{id}/unload`; “Start again”
  re-applies the stored plan inputs and starts. Re-planning, editing the model or
  context of a running cluster, and adding/removing a Mac belong to a later
  slice. The cluster-wide operation lock already prevents a double start.
- **A 3- or 4-Mac plan requires the full topology the deployment needs.** If the
  selected ring/mesh is incomplete, the plan does not silently degrade: it shows
  the missing cable/link and offers “run with 2 Macs” only with explicit
  confirmation, as the brief §10.3 records. A 2-Mac plan is otherwise the normal
  minimal-subset choice when it fits. Nothing degrades silently. A **ring of
  macOS software bridges is a layer-2 loop and macOS does not run STP by
  default**: measured 65 % packet loss and >45 s for 64 MiB after the 2026-11-06
  re-cabling, against 0.2–1.6 ms and 26.9 Gbit/s on a line. The validation
  cluster therefore runs as a **line**; keeping a physical ring requires STP
  per bridge member (`sudo ifconfig bridge0 stp <member> …`). See
  `PHASE-2B-RUN.md` §10. A 4-Mac plan is feasible, but starting four Macs needs
  a loss-free fabric and stays Phase-3 work.
- **A peer-only model is ranked from measured metadata, marked as estimated.**
  The peer inventory carries each model's size and layer count, so
  `recommend` can order a model that only another Mac holds; the chosen model is
  still measured (layout read on its holder) before `plan`.
- **Live evidence lives in `docs/cluster-assistant/PHASE-2B-RUN.md`**, never in
  the test suite. Any new test stays fake-driven; a real Mac, a real model or a
  real endpoint appears only in that document, with measurement/plan/operation
  IDs.
- Remote start is **Option A: SSH control plane, same activation** (recorded in
  `docs/cluster-assistant/CHAPTER-5.md` §2.1). The origin resolves the elected
  coordinator's paired SSH target at request time and drives the coordinator's
  own admin routes (`/plan`, `/stage`, `/deployments`) over the existing SSH
  seam (`staging.run_remote_python`, `ssh_policy`). The coordinator's loopback
  admin call uses the key that Mac already stores locally; the origin never sees
  or stores it, pairing exchanges no new credential, and no Keychain entry is
  added. When the coordinator is this Mac, the same pipeline runs in-process.
  There is no branch that names the origin as a coordinator, and a refused
  coordinator start fails the operation with the one action
  `retry_coordinator`.
- A single-Mac plan cannot be applied **as a cluster**: `ClusterDeployment`
  requires 2–64 hosts, so cluster apply refuses `single_node_plan` with the one
  action `add_mac`. **But the recommendation may recommend single-node inference
  (decision E6, 2026-11-14)** when a model fits on one Mac: the local engine
  serves it without a pipeline. PHASE-2B-RUN.md §22 measured single-node M5 Max
  at 254 tok/s vs 53 tok/s for the 4-Mac cluster on dev-1.5b — for a model that
  fits on one Mac, single-node wins; the cluster is for models that do not fit
  (or for context one Mac cannot hold). The recommendation prefers single-node
  for small models and the cluster otherwise; it never silently falls back.
- **`run-local` starts that single-node model on this Mac (2026-11-14).** The
  cluster apply refuses a single-Mac plan, so a separate flow loads it locally:
  `POST /admin/api/cluster/assistant/run-local` (body `measurement_id`,
  `model_path`, `layer_count`, `context_tokens`, `mtp`) starts a background job
  (`GET …/run-local/{job_id}`; `queued → running → ready|failed`, one run at a
  time, `409 local_run_in_progress`). It reuses `measurement_refusal` and the
  recommendation's own `_single_node_fit` (the one `budget_for` rule) to prove the
  model fits **this** Mac at the requested context, replays the tuning preset for
  `(model, hardware_key([this_node]))`, then asks this Mac's own engine pool to
  load it and verifies the pool reports `loaded`. Everything is loopback with this
  Mac's key — no SSH, never another Mac, never a silent smaller context. A too
  large context is `context_too_large`/`adjust_model_or_context`; nothing fits is
  `insufficient_memory`/`choose_smaller_model`. The card offers "Run on this Mac"
  only when the recommendation's `single_node` answer names this Mac.
  **Live 2026-11-14:** `Qwen3.8-Flash-Next-oQ5e-mtp` (117 GiB VLM) loaded on `.11`
  alone with the PLE table on SSD, 32k context, and served a real completion at
  **40.3 tok/s, TTFT 0.34 s** — the "Qwen3.8 must fit `.11`" goal is met. It has
  `POST …/run-local/{job_id}/stop` and a readiness canary (one tiny completion)
  before `ready`. A ready run is tunable on this Mac: `POST /assistant/tune-local
  {job_id}` runs the sampler sweep + the MTP on/off/depth configs locally
  (write settings, unload, load `local_only`, one real completion each), leaves
  the winner loaded, and saves one preset for `(model, hardware_key([this Mac]))`
  so a later Run on this Mac replays it. Tests: 15 backend + 2 UI.
- The plan is not persisted. `apply` and `start` take the plan-time inputs again
  (`context_tokens`, `rank0_extra_bytes`, `mtp`, `execution_profile`,
  `microbatch_size`), re-elect from the stored measurement and refuse a
  `plan_signature` mismatch (`409 plan_mismatch`, one action `plan`). A client
  stage table is never trusted.
- Operations persist in `<base_path>/cluster/assistant-operations.json` (schema
  1, mode 0600, atomic, newest 8, fail-closed, interrupted → failed). One
  apply/start at a time: a second is `409 operation_in_progress` with the running
  operation ID and the action `status`. States are forward-only
  (`queued → applying → applied`, `queued → starting → loading → ready`,
  `failed`), `applied`/`ready` are terminal, and `start` re-prepares and refuses
  a placement that changed since apply (`placement_changed`, action `plan`).
- The synthetic compute probe is wired **after `apply`**, not into the launch
  path: the assistant request keeps `auto_tune=False`, and the probe's profiles
  are stored for the **next** plan, so the approved placement is still the
  launched one. A failed probe is a warning on the applied operation; `observed`
  may still be filled after `ready` by the real completion measurements
  (Screen 5), and that never rewrites `predicted`.

The LocalInferenz repo is reference material only. Do not copy its app, wizard, chat, profile system, or German UI into this project.

---

## 3. Read these oMLX files before implementation

| Area | Files |
|---|---|
| Cluster wizard template | `apps/omlx-web/omlx_web/templates/dashboard/_cluster_v2.html` |
| Cluster wizard JS | `apps/omlx-web/omlx_web/static/js/cluster_v2.js` |
| Cluster routes | `omlx/cluster/routes.py` |
| Pairing routes | `omlx/cluster/pairing_routes.py` |
| Discovery routes | `omlx/cluster/discovery_routes.py` |
| Deployment validation | `omlx/cluster/deployment.py` |
| Planner / strategies | `omlx/cluster/planner.py` |
| Launch / remote rank start | `omlx/cluster/launch.py` |
| Staging / model sync | `omlx/cluster/staging.py`, `omlx/cluster/modelsync.py` |
| macOS app | `apps/omlx-mac/Sources/` |
| Plan approval tests | `tests/test_cluster_plan_approval.py` |
| Cluster UI backend tests | `tests/test_cluster_ui_integration.py` |
| Cluster UI script tests | `apps/omlx-web/tests/test_cluster_ui_integration_ui.py` |

Existing oMLX routes to reuse, not duplicate:

```text
/api/cluster/devices
/api/cluster/pair/*
/api/cluster/discovery/*
/admin/api/cluster/models
/admin/api/cluster/catalogue
/admin/api/cluster/peer-probe
/admin/api/cluster/autoconfigure
/admin/api/cluster/stage
/admin/api/cluster/deployments
/admin/api/cluster/replan
/admin/api/cluster/diagnostics
```

---

## 4. Current important constraint

oMLX still assumes:

```text
rank 0 == coordinator == endpoint owner
```

The rank-0 **process** may be spawned as a local child
(`hosts[0].ssh == "127.0.0.1"`, the default) or over SSH to the **coordinator's
own** LAN address (the macOS Tahoe ring workaround in `fix/cluster-rank0-ssh`).
It is never another Mac.

Do not turn that into a local coordinator controlling a remote rank 0.

Instead:

1. Elect the coordinator Mac.
2. Start the supervisor on that Mac.
3. The elected Mac is local to its own rank 0.
4. The endpoint belongs to that Mac.

A remote start is a control-plane action, not a new rank-0 topology.

---

## 5. Project layout

The new implementation belongs in:

```text
omlx/cluster/assistant/
  __init__.py
  election.py
  measurement.py
  recommendation.py
  provisioning.py
  start.py
  routes.py
  models.py
  state.py
```

Planned tests (the list is kept current as chapters land):

```text
tests/cluster_assistant/
  test_election.py
  test_measurement.py
  test_measurement_state.py
  test_measure_api.py
  test_recommendation.py
  test_recommend_api.py
  test_plan_api.py
  test_operations.py
  test_apply_api.py
  test_start_api.py
  test_delete_model_api.py
```

Still later: `test_any_to_any.py`, `test_pp_plan.py` (multi-Mac ring and
end-to-end UI).

UI work belongs in the existing oMLX web UI as an Assistant/Simple mode of the Cluster tab. Do not add a second tab or wizard.

macOS app work belongs in `apps/omlx-mac` as a menu/Welcome entry that opens the existing Cluster Assistant mode.

---

## 6. First task

Build the election library before the UI.

Create:

```text
omlx/cluster/assistant/election.py
tests/cluster_assistant/test_election.py
```

Minimum data:

- `NodeFacts`
- `PlanCandidate`
- `ElectionResult`

Minimum tests:

1. App on small Mac; large Mac is elected coordinator.
2. M1 rank 0 cannot hold tail/head; M5 rank 0 fits; M5 is elected.
3. Both candidates feasible; best plan wins; free admission is a preference/tie-break.
4. No candidate feasible; explicit error; no local fallback.
5. Rank-0 extra bytes (LM head/MTP/HTTP) change feasibility.
6. 3/4-node permutations produce valid rank orders.
7. TP request is rejected or hidden.

Exit criterion:

> A standalone test proves that the app origin does not decide rank 0, and that an infeasible local rank 0 is replaced by a feasible remote rank 0.

---

## 7. Commands

Use the project venv and uv:

```sh
uv sync --extra cluster
uv run pytest tests/cluster_assistant
```

Useful targeted tests:

```sh
uv run pytest tests/test_cluster_plan_approval.py
uv run pytest tests/test_cluster_ui_integration.py
uv run pytest apps/omlx-web/tests/test_cluster_ui_integration_ui.py
```

Web CSS after template changes:

```sh
cd apps/omlx-web
python build_css.py
```

### PyCharm: inspections only on source, never the whole tree

A whole-project *Inspect Code* used to run for many minutes: `packaging/` (1.7 GB)
and `build/` were indexed, and the profile enables ~128 inspections, including
spell/grammar over 13k lines of docs, packaging/requirements resolution,
`PyCompatibility`, external JS linters over vendored/minified assets and 74 legacy
JS style rules that fire on the Alpine templates.

Setup that works (`.idea` is gitignored, so re-create it per clone):

- **Exclude** `packaging/`, `build/`, `omlx.egg-info/`, `apps/omlx-mac/build/`
  (`.venv`, `__pycache__`, `.pytest_cache` usually already are).
- **Custom scope `localDev`** (Settings → Appearance & Behavior → Scopes):
  ```
  (file:omlx//*||file:tests//*||file:apps/omlx-web/omlx_web//*.py
  ||file:apps/omlx-web/omlx_web/templates//*||file:apps/omlx-web/omlx_web/static/js/cluster_v2.js
  ||file:apps/omlx-web/tests//*||file:apps/omlx-mac/Sources//*||file:apps/omlx-mac/Tests//*
  ||file:scripts//*||file:Formula//*)
  &&!file:omlx/patches/mlx_vlm_<name>_compat/vendor//*   (seven vendored trees)
  ```
- **Curate the profile.** Even inside that scope a run reported **58 126**
  findings, ~57 000 of them unactionable here: `PyCompatibility` against a 3.9
  target (27 502 — the project runs 3.12 only), missing docstrings (6 159),
  MLX/native unresolved references, argument-equals-default (2 649), unused
  parameters, protected-member access in tests, name shadowing, PEP8 naming,
  broad-except (deliberate and `noqa`-ed), Alpine-template attribute checks and
  the 74 legacy JS style rules. Those are off in the project profile; the
  bug-level ones stay on (unused imports, unreachable code, calling
  non-callable, argument lists, inconsistent returns, comparison with `None`,
  redeclaration, unbound locals, string format, property access, unresolved
  references, `JSUnresolved*`/`JSUndefinedPropertyAssignment`).
  Of the first real run, three findings were genuine and are fixed: a duplicate
  `_is_ssd_table_tensor` in `planner.py`, and a missing `Sequence` import in
  `assistant/models.py` and `assistant/tuning.py`. Everything else in our files
  was narrowing/setter false positives that must not be "fixed" — e.g.
  `pipeline_compat.py` deliberately writes `seen = seen | {…}` instead of
  `|=`, because the parameter may be a caller-owned `set`.
- Then **Code → Inspect Code…** with scope `localDev` (PyCharm remembers it), and
  run `ruff check` / `node --check` from the CLI for the rest.

Not in the scope on purpose: `docs/**` (prose), `i18n/*.json`,
`static/js/*.min.js`, `static/css/tailwind.css` (generated/vendored) and the
vendored `mlx_vlm` model trees under `omlx/patches/*_compat/vendor/`.

Inspection exports land in the **repo root** as one `<Inspection>.xml` per class
(PyCharm's "Export" dialog) — 98 files, many MB. Move them out of the working
tree when done; they are untracked but they pollute the root and `git status`.

Do not install packages globally with pip.

---

## 8. Hard prohibitions

Do not:

- create a second wizard or control plane;
- add German strings or a German locale;
- add TP behavior or TP UI;
- assume the local Mac is rank 0;
- silently fall back to local rank 0;
- duplicate oMLX discovery, pairing, staging, chat, or client integrations;
- add a second macOS app;
- commit API keys, model files, private logs, or machine credentials;
- promise absolute throughput before measuring;
- edit upstream files outside the assistant surface without a small, documented patch;
- start UI work before the election tests pass.

---

## 9. Environment reference

The current validation cluster may be offline. Do not assume it is available.

| Node | Hardware | RAM | Test role |
|---|---|---|---|
| `172.16.1.11` | MacBook Pro M5 Max | 128 GiB | fast/large candidate |
| `172.16.1.13` | Mac Studio M2 Ultra | 64 GiB | medium candidate |
| `172.16.1.14` | Mac Studio M1 Max | 32 GiB | small candidate |
| `172.16.1.12` | MacBook Pro M1 Max | 32 GiB | known 2-node peer, often offline |

Never copy API keys or model data into this repository.

---

## 10. Product decisions

All roadmap-level product decisions were closed on 2026-11-07. Do not reopen
them without an explicit request; if a new question appears, ask once and record
the answer here.

| Decision | Recorded answer |
|---|---|
| Product/repo name | **oMLX Cluster Assistant** (oMLX brand, feature name). |
| Fork / upstream / pin | **Focused fork, now on `upstream/main` `624cff0e`** (rebased 2026-11-14, branch `rebase/upstream-main`; `feat/cluster-assistant` and `backup/pre-rebase-2026-11-14` keep the pre-rebase tree). Our net work from the old pin `1e27576a` was squash-reapplied. Upstream removed routed-expert (MoE) offload from the planner; it is restored on top of upstream's refactored pipeline **and threaded through the coordinator plan path and the deployment decode**, so the assistant MoE lever and the context-vs-experts tradeoff work (`()` default reproduces upstream). Both open PRs are rebased onto `624cff0e` and pushed clean with green CI (#5024 → `bbbf18d2`, #5033 → `25da70de`) and their content is applied on this branch (`054f92e4` #5033, `392aaccb` #5024 runtime), so Qwen4-Exp/Qwen3.8 is PP-capable here. `qwen4_exp_mlx_lm`, the qwen4 cluster/pipeline tests and the experimental doc are part of the #5033 content, not a drop (`UPSTREAM-PACKAGES.md` §7/§8). See `REBASE-2026-11-14.md` and `CLUSTER-TEST-RUNBOOK.md`. Tests: combined `tests/` + `apps/omlx-web/tests` **20 752 passed / 483 skipped / 0 failed** (native kernels built), macOS `xcodebuild test … -testLanguage en`. Prepared PR branches: `UPSTREAM-PACKAGES.md` §8. |
| Rank-0 rule | Best feasible plan; free admission is a strong preference, not the only criterion (already frozen). |
| Coordinator / rank 0 | Co-located in the MVP (already frozen). |
| RDMA default | **Ring over TCP only**; RDMA/JACCL is the later advanced path. |
| Default context / model | **Recommendation decides; 32k target; no fixed default model.** Advanced can raise the context. |
| 3/4-node topology | **A complete topology is a connected line**: every consecutive pair in the chosen rank order has a measured link. An incomplete line is refused with `topology_incomplete`, the missing pairs are named, and **“run with 2 Macs” is offered only after explicit confirmation**; the consent salts the plan signature so apply/start must repeat it. `fabric_measured=false` (no transport answer) keeps the nominal ring with a warning; an empty matrix is not “no links”. Implemented 2026-11-07 (`topology.py`, `test_topology.py`, `test_topology_api.py`). A 2-Mac plan is still the normal minimal-subset choice when it fits and is linked. |
| Helper timing | Full signed/notarized helper is **Phase 5 (after Chapter 10)**. Before the gate, a too-low wired limit is a preflight refusal with one explicit action. |
| Telemetry | **None**; opt-in diagnostics/local logs only, never model or prompt content. |
| Resource levers | Tier/context/offload/MoE changes are **shown in the plan and applied on Start setup**; never written silently, never overwrite an explicit user setting without showing it. |
| Admission rule | **p05 is the safety floor; p50/live admission plans by default.** p95 may be tried only for a huge model and only with an explicit risk confirmation; safe/balanced/aggressive are visible choices. |
| Measured compute | Run the existing **synthetic distributed probe after apply** (staged, before start); use its profiles for the **next** plan. Never rewrite the current `predicted`, never gate the start. |
| TTFT / prefill | **One streaming measurement after ready** reads the final usage chunk; the probe stays off by default. |
| Gate scope | **P0/P1 only.** macOS app/tray, full helper, telemetry, TP and >4 Macs are P2, after Chapter 10. |
| Parallelism per release | **PP only in R1 — Qwen4-Exp/Qwen3.8 included** (upstream PR #5033 applied: `supports_pipeline` true for that family). General tensor-parallel support is **R2**; no TP UI in R1. |
| P2 start (2026-11-07) | **macOS menu/tray entry first**, opening the **existing** Cluster tab in the existing web dashboard; no new window, app, wizard or control plane, and no status/stop in the tray in this slice. The menu item and the Welcome button deep-link `/admin/dashboard?tab=cluster`; both build and the focused Xcode tests pass. Helper: **Developer ID team `YB894QRG4W`**, **one standard macOS admin dialog**, wired limit + network setup scope. The first slice is a one-shot `sysctl iogpu.wired_limit_mb` action from Performance, a **read-only** network preflight (`GET /assistant/network-preflight`, exact commands, `executed: false`), a **SMJobBless** helper for the wired limit (one admin prompt, signed; helper install test pending), and the **first-run runtime installer** in Welcome (bundled `uv` + lock, explicit click, progress; clean-first-run test pending). The privileged network apply is open (bridge/STP policy). Distribution is a direct **slim** DMG (72 MB with helper/installer; no embedded Python/MLX; `SLIM-RUNTIME.md`/`DISTRIBUTION.md`); signing setup is `SIGNING.md`. Telemetry stays none; TP, >4 Macs, HA and multi-tenancy stay out. |

If blocked by an environmental unknown (paths, credentials, hardware), ask once
instead of inventing an answer.

The gap-closing plan that precedes Chapter 10 is
[`docs/cluster-assistant/ROADMAP-TO-CHAPTER-10.md`](docs/cluster-assistant/ROADMAP-TO-CHAPTER-10.md).
Do not start Chapter 10 until its gate is met. Its resource-planning phase is
the recorded response to the “too conservative planning / thin-layer parity”
defect; port LocalInferenz’s planning logic and measured policy, never its app
or UI.

---

## 11. Environment rules (settled — do not re-open these in a session)

These come back in every cluster session; the answers are decided:

- **The app runs interactively on every node.** Copying the DMG over SSH/scp and
  replacing `/Applications/oMLX.app` is fine — but the app is **started by a
  person at that Mac** (Spotlight/Dock), never with `open -a` or a CLI server
  over SSH as the way a node comes up. macOS ties the **Local Network** permission,
  the **SMJobBless helper** prompt and the **first-run runtime installer** to the
  logged-in GUI session, so an SSH launch produces a node that silently cannot
  probe the LAN or bless its helper. SSH stays the **control plane** for staging,
  rank start and probes — not for launching the app.
- **The helper is notarized and blessed by that dialog.** The wired-limit helper
  is installed through one admin prompt while the app is open at the Mac. A node
  where nobody accepted the prompt has no helper; do not fake it with `sudo`,
  `launchctl` or an SSH equivalent.
- **If a step needs a click, ask the user.** Local Network, helper blessing and
  the runtime installer are exactly those steps. Never work around a TCC prompt.
- **"Cleanup" means everything is gone — disk and processes.** Cleaning the
  cluster (or the nodes) means **both**:
  1. **Processes:** every oMLX process — the app, `omlx-server`, servers spawned by
     tests, `pytest`/`xcodebuild`/`XCTRunner`, and the `dns-sd` advertiser — is
     terminated, and the port (8099) is free.
  2. **Disk:** `~/.omlx/cluster/*` is removed (devices, deployments, measurements,
     operations, tuning presets, pairing keys, incidents, benchmarks) **and** the
     `runtime/` staging cache (multi-GB). `identity.json` is **kept** — deleting it
     only re-rolls the node id and invalidates every pairing.
  Back up the JSON state first (`~/.omlx/backup-cluster-<date>/`), then verify: no
  processes, port free, and the cluster directory contains only `identity.json`
  plus backups. This applies to `.11` and to every peer.
- **A real cleanup also removes every oMLX *copy* and its Launchpad entry.** Delete
  `/Applications/oMLX*.app`, the **Xcode DerivedData products**
  (`~/Library/Developer/Xcode/DerivedData/oMLX-*`), **Trash copies**, app-support
  and cache dirs, `~/omlx-pr` / `~/omlx-distributed`, mounted DMG volumes and
  `/tmp` images — then garbage-collect Launch Services
  (`lsregister -gc -f -r -domain user -domain local`) and `killall Dock`, or
  Launchpad keeps showing icons for apps that no longer exist. Keep the **models**
  (`~/models`, `~/Models`) and this repository (source + the notarized DMG under
  `build/`); the gitignored `apps/omlx-mac/build/` is a build product and goes.
- **Verify the version on every Mac before a cluster run.** A mixed-build cluster
  is not a test of this tree (`CLUSTER-TEST-RUNBOOK.md` §1.5): `GET
  /api/cluster/node_id` must report the same `version` on all nodes.
