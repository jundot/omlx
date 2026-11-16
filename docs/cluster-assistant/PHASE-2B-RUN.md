# Phase 2b — live run on real Macs

Status: in progress (2026-11-06). Evidence is filled in per scenario as it is run.

Parent: `PROJECT-BRIEF.md` §18 Phase 2b, `CHAPTER-5.md`, `SCREEN-3-MINIMAL.md`.

Scope: the first run of the committed Screen-3 slice on real Apple Silicon, plus
the recorded Screen-4 decision (`SCREEN-4-PROGRESS.md`).

Nothing in this document is a unit test. Every number below was produced by a
real Mac pair, the real oMLX server, the real models and the real assistant
routes; unit tests stay fake-driven and live in `tests/cluster_assistant/`.

---

## 1. Environment

| Node | Hardware | RAM | Address | Runs | Role in this run |
|---|---|---|---|---|---|
| `d7f9d455` | MacBook Pro M5 Max | 128 GiB | `172.16.1.11` | this fork, port 8099, web UI | origin for B/C/E, coordinator in B |
| `2cc19e81` | Mac Studio M1 Max | 32 GiB | `172.16.1.14` | this fork, port 8099, web UI | origin for A/D, worker/coordinator |
| `11618ac4` | Mac Studio M1 Max | 32 GiB | `172.16.1.12` | this fork, port 8099, web UI | third Mac for C |
| `ed36d44b` | Mac Studio M2 Ultra | 64 GiB | `172.16.1.13` | this fork, port 8099, web UI | unpaired on `.11` (scenario E) |

- Source: this branch at `ac059445` plus the live fixes recorded in §5, run from
  the checkout with the installed oMLX 0.7.3 interpreter
  (`~/Library/Application Support/LocalInferenzMachine/venv/bin/python`,
  `PYTHONPATH=<checkout>:<checkout>/apps/omlx-web`). That is the same code and
  the same activation oMLX uses; only the interpreter comes from the installed
  runtime because no wheel of this branch exists on the peers.
- Model set (all local, no downloads): `GLM-5.3-Flash-oQ2e-mtp` (107.9 GB,
  PP-capable), `Qwen3.8-Flash-Next-oQ5e-mtp` (125.7 GB, not PP-capable),
  `DeepSeek-V4.1-Flash-1017-oQ2e-ngram-mtp` (384 GB, does not fit).
- Admin keys are per-Mac, generated for this run and kept outside the repository
  in a temporary directory. No key, cookie or SSH credential appears here.
- Transport: **Thunderbolt bridge**, not Wi-Fi. Every Mac carries its
  `172.16.1.x` address on `bridge0` (macOS "Thunderbolt Bridge"), `en0` reports
  `media: autoselect (none)` on all four, and `route -n get 172.16.1.11` resolves
  to `bridge0`. `system_profiler SPThunderboltDataType` reports
  `Link Status: 0x100, Speed: Up to 40 Gb/s` on two buses of `.14` (one cable to
  each Studio). Measured between `.14` and `.11` with a 512 MiB socket transfer:
  **26.9 Gbit/s** each way.

### 1.1 Environment finding that shapes the runbook

mDNS/multicast discovery does not work for a server started over SSH on any of
these Macs:

```text
GET /api/cluster/discovery/health
{"multicast_rx_within_5s": false, "last_multicast_rx_at": null,
 "mdns_active": false, "transport": "multicast"}
```

and `.14`'s server log repeats
`HELLO sends failed on every joined interface for 3 rounds`. macOS Local Network
permission is granted per responsible app, and a process started from an SSH
session is not the app. The same server started from a Terminal window on that
Mac gets the permission: the manual probe to `.11` then returns
`{"ok":true,"verified":true,"peer":{"node_id":"d7f9d455…"}}` and
`multicast_rx_within_5s` becomes `true`. All servers for this run are therefore
started from a terminal window on their own Mac. No application code was changed
for this.

---

## 2. Runbook used for the manual pass

The operator sheet — machine-ID table, per-scenario intention, clicks, typed
values, expected result and what to record — is
[`PHASE-2B-RUNBOOK.md`](PHASE-2B-RUNBOOK.md). The summary below is the same
flow in short form.

Reset to the initial condition on every Mac (no pairing, no deployment, no
assistant records), then:

### 2.1 Pair the M1 Max (`.14`) into this Mac's cluster

| # | Where | Action | Value | Expected |
|---|---|---|---|---|
| 1 | `.14` UI | Cluster tab → **Add by IP** | `172.16.1.11:8099` | toast “Found MacBook-Pro-von-Andreas at 172.16.1.11.”, card “Joining MacBook-Pro-von-Andreas's cluster” with a 6-digit code and a countdown |
| 2 | `.11` UI | Cluster tab → device card “MacStudio-von-Andreas-2” shows **Wants to join** | the 6 digits from step 1 | **Approve & pair** → toast “MacStudio-von-Andreas-2 joined the cluster.” |
| 3 | `.14` UI | — | — | toast “This Mac joined MacBook-Pro-von-Andreas's cluster.”; both lists show the other Mac as **Paired** |

Pair `.12` the same way from `.12`'s UI (address `172.16.1.11:8099`) when the
3-Mac scenario is due.

### 2.2 Start the assistant (per scenario)

The card lives **inside the Plan step**: Cluster tab → walk the wizard forward
until the step heading says **Plan** → card **“Set up this cluster”**.

| # | Action | Expected |
|---|---|---|
| 1 | **Find and start a setup** | states: *Checking Macs* → *Choosing a model for these Macs* → *Preparing measurements for <model>* → plan summary (**Model / Context / Coordinator / Macs**) and **Start setup** |
| 2 | **Start setup** | *Preparing the model* → *Starting <coordinator>* → *Loading the model* → **Ready** with endpoint, **Open Chat**, **Connect agent** |
| 3 | Read the endpoint | `http://<coordinator>:8099`, `GET /v1/models` and one completion answer |

The card does **not** apply by itself: reaching the plan summary is not Ready.
That is the shipped behaviour (SCREEN-3-MINIMAL §5), not a defect.

### 2.3 What each scenario must prove

| # | Origin | Peers | Expected | Passes when |
|---|---|---|---|---|
| A | `.14` | `.11` | coordinator elected (either Mac) | a non-coordinator origin drives a remote coordinator, or `.14` is elected and serves; the origin never wins by being the origin |
| B | `.11` | `.13` | `.11` or `.13` | local-origin path, coordinator still elected |
| C | `.11` | `.14` + `.12` | election's choice | 3-Mac plan, staging and start without typing ranks or IPs |
| D | `.14` | `.11` with oMLX stopped | none (failure) | `error.action = retry_coordinator`, same coordinator, nothing starts locally |
| E | `.11` | unpaired `.13` in the plan | none (failure) | `error.code = node_not_paired`, `action = pair_mac`, nothing started |
| F | winner of A | — | ready | `GET /v1/models` and one real completion |

---

## 3. Evidence

### 3.0 Scenario B — origin `.11`, coordinator `.11` (local-origin path)

Before the browser pass, the same routes were driven from `.11` with `.14` as the
peer. The election made `.11` the coordinator, so this is the "app and
coordinator on the same Mac" case the Phase 2b exit criteria ask for:

| Step | Value |
|---|---|
| measurement | `ms-20261106T122021Z-0001dfcb`, ready, `.11` 96.9 GiB / `.14` 17.5 GiB |
| plan | `plan_signature 629bea726e92…`, coordinator `d7f9d455` (`.11`) |
| apply | `op-20261106T122024Z-apply-000176ce` → `applied` in 4 s |
| start | `op-20261106T122038Z-start-00022b13` → `starting` → `loading` → `ready` in ~30 s |
| deployment | rank 0 `.11` layers 8–45 (92.05 GB planned), rank 1 `.14` 0–8 (`andreaskolb@172.16.1.14`) |
| endpoint | `http://172.16.1.11:8099`, `ready: true` |
| completion | `"ready"`, 22 tokens, 1.19 s |

The literal matrix row ("origin M1 `.14`, coordinator M5 `.11`") did **not** occur:
every 2-Mac GLM plan over `.11`+`.14` elected `.14`, because the election scores
the tightest stage and the M1 Max's tail stage had more headroom. The
remote-coordinator path itself is proven the other way round in §3.3: origin
`.11`, coordinator `.14`, apply and start driven remotely, endpoint on `.14`.

### 3.1 Scenario A — origin `.14`, peers `.11` (browser, user-driven)

| Step | Value |
|---|---|
| measurement (entry, model-bound) | `ms-20261106T155054Z-00013594` — DeepSeek, `.14` 1.0 GiB / `.11` 96.9 GiB → **refused** (see 4.2.4) |
| measurement (after Headless) | `ms-20261106T155843Z-00033bf8`, `ms-20261106T155846Z-000407a8` — ready, `.14` 18.7 GiB, `.11` 96.9 GiB, warnings `unmeasured_compute` only |
| recommendation | feasible, `GLM-5.3-Flash-oQ2e-mtp` at 32 768 tokens |
| plan | `plan_signature ca5b6f47fa719998…`, coordinator `2cc19e81` (`.14`) |
| apply | `op-20261106T160556Z-apply-00012316` → `applied` |
| start | `op-20261106T160559Z-start-00021a6a` → `starting` → `loading` → `ready` |
| endpoint | `http://172.16.1.14:8099`, `ready=true` after one canary completion |
| ranks | `.14` rank 0 (coordinator, local to itself) and `.11` rank 1 (`omlx.cluster.inference_worker` running on both) |
| chat | a real completion answered on the endpoint (see 4.2.4/§6 for the generation-quality findings) |

**Pass against Phase 2b:** the origin completed measure → recommend → plan →
apply → start → ready and served the endpoint. The *election* put rank 0 on the
origin itself (`.14`), which is legal: the origin was the best feasible
coordinator for that split, not the app origin by default. The remote-coordinator
case therefore falls to B/C.

### 3.2 Scenario C — three Macs, origin `.14` (agent-driven through the same routes)

Driven through the assistant API from `.14` because the browser pass was spent on
A; every call is the same one the card makes (`measure → recommend → plan →
apply → start`), and the `operation_in_progress`/polling behaviour is identical.
Scope note: the card's rendering of these states is covered by A and by the
static component tests.

| Step | Value |
|---|---|
| measurement | `ms-20261106T181732Z-000172b4`, ready, model identity `mtp: true` |
| nodes | `.14` 19.7 GiB, `.11` 96.9 GiB, `.12` 18.6 GiB — warnings `unmeasured_compute ×3` |
| recommendation | feasible, `GLM-5.3-Flash-oQ2e-mtp` at 32 768 tokens |
| plan | `plan_signature 285c0525ebea…`, coordinator `2cc19e81` (`.14`), rank order `.14 → .12 → .11` |
| stages | rank 0 `.14` layers 41–45 (14.40 GiB, headroom 5.31, critical), rank 1 `.12` 36–41 (12.67), rank 2 `.11` 0–36 (77.48) |
| endpoint | `http://172.16.1.14:8099` (plan), `ready: true` after the canary |
| apply | `op-20261106T181739Z-apply-00019793` → `applied` in 4 s |
| start | `op-20261106T181743Z-start-000256fc` → `starting` 19:17:43 → `loading` 19:17:47 → `ready` 19:18:19 |
| warnings | `performance_unmeasured`, **`chat_profile_applied`** |
| engine | `Distributed inference enabled … ranks=3 backend=ring plan=d8b9b4e8e76389e5`, no draft-head warning |
| live refusal check | re-planning the same measurement with `mtp: false` is refused: `stale_measurement — the measurement was taken with Lightning MTP on, and this plan asks for it off` |

Completions on that cluster (three ranks, MTP planned):

| Request | Reasoning | Total | Time | tok/s |
|---|---|---|---|---|
| model defaults (now the assistant's profile) | 688 | 797 | 39.7 s | 20.1 |
| `chat_template_kwargs.enable_thinking=false` | 465 | 558 | 22.4 s | 24.9 |

No speed claim is made: adding the third rank costs more per token than the
draft head gains on this pair, and the reasoning length varies between runs.

### 3.3 Scenario D — the elected coordinator stops answering (API, agent-driven)

Origin `.11`, peer `.14`, measurement `ms-20261106T182759Z-0001766f` (`.11` 96.9
GiB, `.14` 20.1 GiB), plan feasible with coordinator `2cc19e81` (`.14`) and
endpoint `http://172.16.1.14:8099`. `.14`'s oMLX was then stopped **after** the
plan, and apply was posted from `.11`:

```json
{"code":"staging_refused",
 "message":"2cc19e81-… did not answer as the coordinator; open oMLX on it,
            make sure its admin API key is set, and retry the coordinator",
 "action":"retry_coordinator"}
```

`endpoint.ready` stayed `false`, no rank started on `.11`, and the coordinator was
still `2cc19e81` — the origin did not become a fallback.

Before the fix in §4.1.10 the same apply answered `apply_failed` /
`diagnostics`, i.e. it sent the user to a support bundle instead of "open oMLX on
that Mac".

Retry after opening `.14` again, with the same measurement and the same plan:
`op-20261106T182856Z-apply-0002a517` → `applied` (4 s), then
`op-20261106T182900Z-start-00034299` → `starting` 19:29:00 → `loading` 19:29:05 →
`ready` 19:29:35, endpoint `http://172.16.1.14:8099` `ready: true`, warnings
`performance_unmeasured`, `chat_profile_applied`.

A note recorded while preparing D: with the coordinator's app closed its
*measured admission* drops from 96.9 GiB to 30.0 GiB on `.11` (the probe reads the
memory guard's live ceiling), so a measurement taken while the app is closed can
refuse the model before apply is ever reached. That is why D stops the app after
the plan, which is also the honest user story ("it was running, then I closed
it").

### 3.4 Scenario F — the finished endpoint

On the endpoint from D's retry (`http://172.16.1.14:8099`):

```text
GET /v1/models
["DeepSeek-V4.1-Flash-1017-oQ2e-ngram-mtp", "GLM-5.3-Flash-MLX-4bit-MTP",
 "GLM-5.3-Flash-oQ2e-mtp", "Qwen3.8-Flash-Next-oQ5e-mtp", "dev-v41-15L-real"]

POST /v1/chat/completions {"max_tokens":512}
finish: stop | answer: "Earth" | 83 tokens (80 reasoning) | 2.9 s | 28.6 tok/s
```

The same request with `max_tokens: 64` returned `finish: length` and an empty
answer, because the reasoning consumed the whole budget — the reason the chat
profile caps at 2048 rather than tighter.

### 3.5 Scenario E — an unpaired Mac (API, agent-driven)

`.13` was discovered by `.11` but never paired. A three-Mac plan
(`ms-20261106T164244Z-0002d811`, coordinator `.11`, rank order `.11 → .13 → .14`)
refused on apply:

```json
{"code":"node_not_paired",
 "message":"ed36d44b-… has no paired SSH target on this Mac; pair it again and measure again",
 "action":"pair_mac"}
```

HTTP 409, no operation created, nothing started, no local fallback. Before the fix
in §4.1.8 the same apply returned `apply_failed — invalid SSH target:
'fe80::18b2:3e41:95e1:b30b'` (action `diagnostics`), i.e. it tried to launch on a
Mac the user never approved. When the unpaired Mac is elected *coordinator*
instead, apply refuses earlier with `coordinator_unresolved`
(action `configure_endpoint`) — see 4.2.7.

The UI cannot produce this case, because it only offers paired Macs; its
rendering of `pair_mac` is covered by the component tests.

---

## 4. Findings

### 4.1 Bugs found live and fixed

Each one has a fake-driven test; several were also re-checked on the Macs.

1. **Every measurement reported the model inventory as unreadable.**
   `_model_inventory` passed the caller's absent `python_executable` through, and
   the probe's absolute-path rule then raised. Fixed: the argument is only passed
   when known. Test `test_an_unknown_interpreter_lets_the_probe_discover_one`.
2. **A plan whose coordinator was this Mac carried no endpoint.** A Mac has no
   paired record of itself, so the measured endpoint host was empty. Fixed: the
   local node falls back to the addresses it announces (IPv4, private first).
   Tests `test_this_mac_resolves_a_dialable_endpoint_without_a_paired_record`,
   `test_local_endpoint_addresses_prefer_something_a_peer_can_dial`.
3. **Apply could never treat this Mac as the coordinator.** The address seam read
   `address` instead of `ip`, so `local` was always false. Test
   `test_the_default_seams_read_the_addresses_discovery_reports`.
4. **Every remote coordinator call failed at the coordinator.** The bare
   `python -c` read settings without loading them. Test
   `test_the_remote_admin_snippet_loads_settings_on_a_bare_interpreter`.
5. **The app origin could never be a worker.** Fixed: this Mac supplies its own
   measured address when another Mac is rank 0. Tests
   `test_the_origin_mac_is_dialable_when_another_mac_is_rank_zero`,
   `test_a_local_coordinator_still_runs_over_loopback`.
6. **A discovered Mac had no name, chip or dialable address.** Fixed: the
   fallback announcement carries `friendly_name`, caps and IPv4 addresses. Tests
   `test_wassup_carries_the_display_identity`,
   `test_wassup_names_the_peer_without_mdns`,
   `test_wassup_records_the_announced_ipv4_and_probes_it`.
7. **A Thunderbolt peer was labelled “Network”/“wifi”.** Fixed: `bridge0`
   addresses are kind `thunderbolt`, the kind is announced, and a peer keeps the
   best path it answered on. Tests
   `test_a_thunderbolt_address_is_classified_from_metadata_not_latency`,
   `test_a_peer_reachable_over_several_paths_keeps_the_best_one`.
8. **A refusal whose action the card did not know was a one-way door.** Fixed:
   the card knows the recommendation's four near-miss actions and returns to
   idle. Test `test_assistant_can_recover_from_a_recommendation_refusal`.
9. **A stopped coordinator was reported as `diagnostics`.** Fixed: "no answer" is
   status 0 and maps to `retry_coordinator`. Tests
   `test_a_closed_coordinator_asks_to_retry_the_coordinator`,
   `test_an_http_refusal_keeps_the_apply_action`.
10. **An unpaired but discovered Mac was used as a launch target.** Fixed: only
    a paired row may become a target, and IPv4 wins over link-local IPv6. Tests
    `test_an_unpaired_discovered_mac_is_never_a_launch_target`,
    `test_a_link_local_address_never_wins_over_ipv4`.
11. **A freshly started cluster had no usable generation settings.** Fixed: the
    start path fills `CHAT_PROFILE` where the model has none. Tests
    `test_start_gives_a_model_without_sampling_a_usable_chat_profile`,
    `test_start_never_overwrites_sampling_the_user_chose`,
    `test_a_failed_chat_profile_write_does_not_unready_the_cluster`.
12. **The same measurement could elect a different coordinator after a restart.**
    plan/apply/start each re-read the checkpoint, and that read can change while
    the server runs, so an applied plan failed to start with `plan_mismatch`.
    Fixed: the measurement records the `ModelLayout` it was taken against and
    every re-election uses it. Test
    `test_the_plan_is_built_from_the_layout_the_measurement_recorded`.
13. **A cluster launched by the assistant could not use Lightning MTP.** Fixed:
    the assistant plans MTP when the catalogue says the checkpoint supports it.
    Test `test_assistant_plans_lightning_mtp_from_the_model_catalogue`.

### 4.2 Gaps closed in the follow-up batch

1. **A recommendation could not say “this model is already on every Mac”.** The
   measurement records each Mac's catalogue (`NodeFacts.model_paths`) and
   `_presence` answers per candidate. Live: GLM recommended with
   `model_present: true`, `stage_bytes: 0` on both Macs.
2. **The single refusal action came from the smallest near miss.** It now comes
   from the first near miss whose failure is about the cluster. Live: action
   `choose_smaller_model` instead of `check_model_path`.
3. **An unpaired coordinator was reported as `configure_endpoint`.** Fixed:
   `node_not_paired` / `pair_mac` for a Mac that was never paired.
4. **“One apply/start at a time” is per origin, not cluster-wide.** Fixed with
   `GET /admin/api/cluster/assistant/operations/active` plus a check over the
   admin seam. Live: `.11` refused while `.14` loaded, allowed once ready.

### 4.3 Contract questions still open (need a decision, not a patch)

1. **The coordinator's endpoint port is also its admin port.** `endpoint.ready`
   is the only thing that separates "reachable" from "serving".
2. **The Mac in front of the user defaults to “Workstation”.** The role holds
   back at least 32 GiB, so a 32 GiB Mac contributes 1.0 GiB and no model fits;
   the card never names the role as the cause.
3. **A recommendation can only judge checkpoints present on the origin.**
   Planning a peer-only model needs its layer metadata over the network, which
   the inventory does not carry today.
4. **No measured performance.** `predicted.measured: false`, `observed: null`
   everywhere; the probe is a later chapter. *Closed 2026-11-07: the post-ready
   measurement and its nullable `observed` block are decided in
   `SCREEN-5-HANDOFF.md` §4–§6 and recorded live in §12.*

### 4.4 Environment

1. mDNS/multicast discovery was initially unavailable for SSH-started servers
   (§1.1): `mdns_active: false`, and `.14`'s “Add by IP” answered “No oMLX node
   answered at that address yet”. Starting the same server from a Terminal
   window on that Mac fixed both at once — the probe to `.11` then returned
   `{"ok":true,"verified":true}` and `multicast_rx_within_5s` became true. macOS
   grants Local Network access per responsible app, and a process started from an
   SSH session is not the app.
2. Before that permission existed, the joiner's own
   `POST /api/cluster/pair/request` also failed with `EHOSTUNREACH`, so pairing
   had to be completed through the same pairing API from a short-lived process.
3. `.13` (M2 Ultra) has 14 GiB free on a 926 GiB volume, so any staging there
   fails; it is used as the unpaired Mac for the pairing scenarios.
4. SSH to `.14`/`.12` intermittently timed out during set-up (one `rsync` and one
   measure refused with `admission_probe_failed`); every retry succeeded.
5. A cluster that is *loaded* drops the measured admission sharply (`.11`
   97 → 30 GiB). Unload before measuring, or the plan refuses for lack of memory
   even though the Mac is capable.
6. One measured fabric reported 0.8 Gbit/s because a member's only path to a peer
   was the 192.168.178.x Ethernet link. That is the honest weakest link, not an
   artefact; the Thunderbolt-only pair measures 26.9 Gbit/s.

---

## 5. Generation settings (why a simple question took minutes)

Measured on the cluster this chapter started (coordinator `.14`, 2 ranks, ring,
`.14`'s log `~/.omlx/logs/server.log`):

| Configuration | Reasoning tokens | Completion tokens | Time | Answer |
|---|---|---|---|---|
| model defaults (temperature 1.0, `max_tokens` 32768) | 1149 | 1286 | 47.7 s | correct |
| model defaults, `max_tokens` 512 | 387 | 482 | 25.5 s | correct |
| `chat_template_kwargs.enable_thinking=false`, `max_tokens` 2048 | 18 | 90 | **3.4 s** | correct |
| per-model `enable_thinking=false`, `max_tokens` 2048 | 339 | 443 | 16.4 s | correct |
| per-model `enable_thinking=false`, `max_tokens` 512 | 512 | 512 | — | **empty, truncated** |

Three conclusions:

1. The token cap, not the temperature, is what makes a thinking model ramble for
   minutes. The model's own `generation_config.json` ships
   `temperature 1.0 / top_p 0.95`, which oMLX copies as its defaults.
2. A cap small enough to stop the rambling truncates the answer instead, so the
   cap has to be generous (2048) rather than tight.
3. Forcing `enable_thinking` at the *model* level is unreliable and can be worse
   than leaving it alone; the per-request
   `chat_template_kwargs: {"enable_thinking": false}` is what actually shortens
   the reasoning (3.4 s). The assistant therefore does not write that key — it
   has no chat request to attach it to.

The chat UI's own aborts (`public client transport disconnected`, every 40–60 s)
are what made it look endless: the stream was dropped and retried, so no attempt
ever reached its `finish_reason`.

The profile the assistant now fills when a model has none
(`omlx/cluster/assistant/start.py::CHAT_PROFILE`): temperature 0.6, top_p 0.95,
top_k 0, `max_tokens` 2048, `force_sampling` true.

---

## 6. What still blocks Screen 4 / Screen 5

Evidence-backed, from this run:

| Blocker | Evidence | Consequence |
|---|---|---|
| ~~Screen-4 progress is not implemented~~ | §3.2 shows the real transition set is `applying → applied → starting → loading → ready` with `progress.phase ∈ {staging, applied, starting, loading, ready}` | **closed 2026-11-07**: the card renders six observable rows from exactly those values (`SCREEN-4-PROGRESS.md` §8, live walk here §11) |
| ~~The Workstation default starves a small Mac~~ | A refused with 1.0 GiB admission on `.14` while the same pair planned at 19.0 GiB as Headless (§4.2.4) | **closed 2026-11-06**: a Mac below 64 GB is measured as Headless and says so (§8.1) |
| ~~No cluster-wide operation lock~~ | `.11` and `.14` each own an operation store, so two origins can drive one coordinator (§4.2.1) | **closed 2026-11-06**: `operations/active` over the admin seam (§4.2.4) |
| ~~A recommendation cannot see models on other Macs~~ | `dev-1.5b` / `dev-v41-15L` are `model_unreadable` on the origin although they exist on a peer (§4.2.6) | **closed 2026-11-06**: the origin reads the layout on the holder (§8.2) |
| ~~No `observed` performance~~ | `predicted.measured: false`, `observed: null` everywhere in this run | **closed 2026-11-07**: `observed` is one nullable block on the operation, filled after `ready` by a real completion on the coordinator (server setting `assistant_performance_probe`, default off); `predicted` is never rewritten (`SCREEN-5-HANDOFF.md` §4–§6) |
| ~~Screen-5 agent handoff has no deep link~~ | `Open Chat` navigates to `/chat` without a model preselection (SCREEN-3 §11) | **closed 2026-11-07**: `Open Chat` opens the existing `/chat` with the resolved serving id (the coordinator’s own page when it is another Mac), and `Connect agent` shows `<endpoint>/v1`, the serving id and where the key lives (`SCREEN-5-HANDOFF.md` §1–§3) |

Not blocking: Thunderbolt transport (26.9 Gbit/s measured), discovery naming and
IPv4 addresses (fixed), Lightning MTP (now planned when the checkpoint supports
it), the chat sampling profile (now filled when the model has none).

---

## 7. Gaps closed after the run (2026-11-06, same branch)

Four gaps recorded during the run — plus two found while closing them — are now
closed with fake-driven tests and live re-checks; §4.1 and §4.2 carry the
per-item reason and test names. The two found along the way:

- **the election was not reproducible**: the same stored measurement elected a
  different coordinator after a restart, because plan/apply/start each re-read
  the checkpoint. The measurement now owns the layout (§4.1.12).
- **the cluster-wide lock**: a second origin was refused while the coordinator
  was busy, and allowed once it was ready (§4.2.4).

Correction recorded while checking the last item: the earlier
`link.bandwidth_gbps: 0.8` note was **not** an artefact — that group contained
`.13`, whose path to one peer is the Ethernet link, so the weakest link really
was ~0.8 Gbit/s (§4.4.6).

---

## 8. Decisions taken after the run (2026-11-06)

### 8.1 A small Mac is Headless, and says so

Decided: **a Mac below 64 GB installed RAM is set to Headless**, with a warning,
because the Workstation role holds back at least 32 GiB and on a 32 GiB Mac that
left **1.0 GiB** of admission — the run's “none of the 7 candidate models fits”
dead end.

- The card sends `headless` for such a Mac (`ASSISTANT_SMALL_MAC_RAM_GB = 64`)
  and renders one sentence per Mac (`cluster.assistant.small_mac_note`).
- The backend enforces the same rule from the **computed budget**, so it also
  covers this Mac (whose ceiling is not probed) and any other client: if a
  workstation budget would leave less than 4 GiB usable, the Mac is measured
  again as Headless and the measurement carries
  `workstation_reserve_too_large`.
- Live: `.14` asked as *workstation* now admits **19.7 GiB** (was 1.0) and the
  measurement warns. Tests:
  `test_a_small_mac_is_measured_as_headless_and_says_so`,
  `test_a_small_mac_is_forced_to_headless_and_says_so` (card).

### 8.2 A model on another Mac is usable, and a copy that cannot fit is refused early

Decided: the assistant detects and uses **models that live on other Macs**, and
refuses a staging copy that cannot fit with one action.

- The origin cannot read a peer's checkpoint, so when its own read fails the
  measurement asks the Mac whose inventory lists the model for its
  `inspect_safetensors_layout` and stores that layout
  (`layout_read_on_peer`). plan/apply/start then re-elect from it.
- Live: `.14` measured `dev-1.5b`, which it does **not** hold — layout read on
  `.11` (28 layers), warning `layout_read_on_peer`, and the plan is feasible
  across both Macs (`.11` rank 0 layers 2–28, `.14` 0–2).
- A copy that does not fit is refused before staging starts:
  `insufficient_disk` with the Mac, needed and free GiB, and the one action
  `free_disk_space`.
- **Deletion stays a user action.** `DELETE /admin/api/hf/models/{model_name}`
  already removes a model from a Mac's disk and refreshes its pool; the assistant
  names the Mac and the amount, and never deletes anything itself. Automating the
  deletion from the card (a per-Mac model list with sizes and a two-step confirm,
  reached through a new assistant route) is the next slice — see §9.
- Tests: `test_a_model_that_lives_on_another_mac_is_measured_from_that_mac`,
  `test_a_mac_without_room_for_the_copy_asks_to_free_space`.

### 8.3 Not a question

The “endpoint URL is also the admin port” note was an observation, not an open
decision; it is recorded in §4.3 for the Screen-5 copy only.

## 9. Decision: the delete affordance is a relay route

**Decided 2026-11-06: Option A — a new assistant route.** To finish 8.2's last
step the card needs, when a failure action is `free_disk_space`:

1. the measured model list **of that Mac** with sizes (the measurement stores
   `NodeFacts.model_paths`; the inventory snapshot must also carry each model's
   size),
2. a two-step confirm (“Delete” → “Delete <model>?”) — destructive, never
   automatic, and never the model the cluster is about to run,
3. a call the browser may make: **`DELETE /admin/api/cluster/assistant/models/
   {node_id}/{model_id}`**, relayed through the same admin seam to that Mac's own
   `DELETE /admin/api/hf/models/{model_name}`. It works for the worker that is
   actually short of space, keeps SSH in the backend, and reuses the coordinator
   credential story.

Consequences to keep honest: the route is the assistant router's ninth path, the
router-list test and the seam allowlist (`allowed_without_caller` is for
unreachable routes; this one *is* called by the card, so it gets a caller) both
follow, and `measure` is the action after a successful delete.

Rejected: **B** (coordinator-only reuse — cannot free a worker, which is the case
that fails) and **C** (record only — leaves a dead end the run already proved).

### 9.1 Implemented (2026-11-07)

- Route: `DELETE /admin/api/cluster/assistant/models/{node_id}/{model_id}` with a
  required `measurement_id` query parameter. `model_id` is the id the card showed
  from the measured inventory, so the route deletes only a model that Mac itself
  listed — a wrong node, a stale list or an unknown measurement is refused
  (`measurement_node_mismatch`, `unknown_model`, `unknown_measurement`, all with
  the one action `measure`).
- The checkpoint that measurement is about to run is refused with
  `model_in_use` and the action `adjust_model_or_context`; the card also does not
  offer it.
- Target resolution reuses apply/start's evidence (measured address, measurement
  endpoint, paired record): this Mac → loopback; a paired peer → its SSH target
  and measured admin port; never paired → `node_not_paired`/`pair_mac`; no
  dialable address or unknown port → `configure_endpoint`.
- The relay (`start.delete_model_on_mac`) is the same seam apply/start use: a
  bare interpreter on that Mac authenticates its own loopback call with the key
  it stores, so the origin never sees a key and pairing exchanges no credential.
  The Mac's own answer decides the one action: 404 → `unknown_model`/`measure`,
  no answer → `delete_unreachable`/`retry_coordinator`, anything else →
  `delete_failed`/`diagnostics`.
- The relay names the model by its **measured folder name** (the path's last
  segment), which is what oMLX's route resolves for a flat or
  org-folder layout. An HF-cache entry (`models--Org--Name/snapshots/<hash>`)
  cannot be addressed by a single name; that Mac answers 404 and the card reports
  it, so the user frees that one from the Mac's own Models page. Nothing deletes
  silently and nothing guesses a path.
- The measurement now stores each Mac's model list (`model_id`, `model_path`,
  `size_bytes`) in `MeasurementRecord.inventories` and exposes it on the status
  poll (`MeasuredNodeSummary.models`). It is optional on disk: a record written
  before this chapter still loads and simply offers no delete.
- Tests: `tests/cluster_assistant/test_delete_model_api.py` (14, fake-driven: the
  happy path per Mac kind, `model_in_use`, unknown model/node/measurement,
  unpaired Mac, the peer's three answers, the status shape, an old record and a
  malformed one), plus the card's static and Node-driven tests.

### 9.2 Live evidence (2026-11-07)

Run on `.11` (M5 Max 128 GB, this Mac, running this branch — the new route
answers 401 there while the three older servers answer 404) with `.14`
(M1 Max 32 GB, paired, Headless) as the peer. Precondition checked first: no
deployment, no live rank, no active assistant operation.

**Validation, all refused before any relay ran** (`measurement_id`
`ms-20261106T233135Z-00015848`):

| Request | Answer |
|---|---|
| delete the model the measurement plans (`dev-1.5b` on `.11`) | 409 `model_in_use`, action `adjust_model_or_context` |
| delete a name the Mac never listed | 409 `unknown_model`, action `measure` |
| delete on a Mac outside the measurement | 409 `measurement_node_mismatch`, action `measure` |
| delete with an unknown measurement | 404 `unknown_measurement`, action `measure` |
| the same call without an admin key | 401 |

**The relay itself, on the copy this run had staged.** After the Screen-4 walk
(§11) the deployment was unloaded, then `ms-20261106T233421Z-0002d921` was taken
of `dev-v41-15L`, so `dev-1.5b` was no longer the planned model and `.14`'s copy
became deletable:

```text
DELETE /admin/api/cluster/assistant/models/2cc19e81…/dev-1.5b?measurement_id=ms-20261106T233421Z-0002d921
HTTP 200 in 0.503 s
{"node_id":"2cc19e81-…","model_id":"dev-1.5b","deleted":true}
```

- On `.14` (`ssh andreaskolb@172.16.1.14`): `~/Models/dev-1.5b` (852 MB) is gone;
  `df -g` used 875 → 874 GiB (98 % → 97 %).
- On `.11`: the original `~/Models/dev-1.5b` (839 MB) is untouched.
- A fresh measurement `ms-20261106T233444Z-0003df12` lists 7 models on `.11`
  (with `dev-1.5b`) and 5 on `.14` (without it).

So the browser-called route deletes on a **worker** over the existing SSH seam,
one model, only a measured one, and never the planned checkpoint — the two halves
the run had left open (§8.2) are now both proven live. `.14` sits at 27 GiB free
(97 % full), so the next run that needs an `insufficient_disk` refusal has a
natural place to produce one.

---

## 10. Phase 3 readiness — cabling is a ring, and why that breaks the transport

The four Macs were re-cabled from a line into a **ring**: each Mac now has
exactly two active Thunderbolt links (`.11` en1+en6, `.12` en2+en5, `.13`
en2+en3, `.14` en2+en3) — eight ends, four cables, i.e. a real ring.

**A ring of macOS software bridges is a layer-2 loop, and macOS does not run
STP by default** (`ifconfig bridge0` reports `proto stp` with `hellotime 0` and a
zero root id). Measured immediately after the rewiring:

| Check | Line (before) | Ring (now) |
|---|---|---|
| ping, 20 packets | 0 % loss, 0.23–1.6 ms | **65 % loss**, 4.8–10.9 ms |
| TCP, 512 MiB | 0.16 s ≈ 26.9 Gbit/s | — |
| TCP, 64 MiB | — | **> 45 s (interrupted)** |

So the ring is not an IP-configuration problem and not simply “needs RDMA”:

- With `bridge0`, the whole fabric is one bridged segment. A ring closes a loop
  across four independent bridges; without STP, flooded frames circulate and
  source-address learning churns, which is exactly the loss and latency above.
- Keeping the physical ring needs **STP on every bridge**, e.g.
  `sudo ifconfig bridge0 stp en1 stp en6` on each Mac (one member then blocks,
  so the usable topology becomes a line with redundancy). Verify with
  `ifconfig bridge0` (one member not forwarding) and a loss-free `ping`.
- True ring transport without the bridge would mean giving each Thunderbolt port
  its own point-to-point subnet and letting an RDMA/JACCL fabric use them
  directly. That is the exo/JACCL model; this MVP plans the `ring` backend over
  TCP and builds no RDMA matrix, so it is out of scope.

**Recommendation: wire a line** (3 cables for 4 Macs — no loop, no STP
dependency), which is also what the measured good numbers came from. PP only
passes hidden states between adjacent ranks in the *logical* ring, and staging
plus health are TCP between Macs, so a line already provides every pair in one
L2 segment.

**Restored to a line and re-verified (2026-11-06).** Active Thunderbolt bridge
members are now `.11` en1 · `.12` en2 · `.13` en2+en3 · `.14` en2+en3 (six ends,
three cables — no loop). Every pair, both directions:

| | `.11` | `.12` | `.13` | `.14` |
|---|---|---|---|---|
| `.11` | — | 0.0 % · 1.59 ms | 0.0 % · 0.52 ms | 0.0 % · 1.00 ms |
| `.12` | 0.0 % · 1.56 ms | — | 0.0 % · 1.45 ms | 0.0 % · 1.01 ms |
| `.13` | 0.0 % · 0.57 ms | 0.0 % · 1.42 ms | — | 0.0 % · 0.88 ms |
| `.14` | 0.0 % · 1.15 ms | 0.0 % · 1.11 ms | 0.0 % · 0.98 ms | — |

and a 512 MiB socket transfer `.14 → .11` runs in 0.16 s = **26.9 Gbit/s**. The
line is the reference topology for every later run.

A 4-Mac plan is nevertheless feasible on the ring's measurements
(`ms-20261106T222236Z-000242ba`: coordinator `.11`, ranks `.11 → .12 → .14 →
`.13`, layers 13–45 / 9–13 / 5–9 / 0–5, warnings `unmeasured_compute ×4`,
`unmeasured_plan`). Planning does not depend on link quality; **starting** does,
and must wait for a loss-free fabric.

---

## 11. Screen-4 live walk (2026-11-07)

The progress surface was exercised on hardware, not only against fixtures. The
run is agent-driven through the same routes the card calls; the card's rendering
of these values is covered by the Node-driven tests
(`test_assistant_walks_the_screen_4_rows_in_the_backend_order`), because the
agent cannot click.

**Precondition.** No deployment, no live rank, no active assistant operation
(`operations/active` → `{"operation": null}`). `.11` ran this branch; `.14` ran
an older checkout, which is enough — the relay only needs that Mac's own
`/admin/api/hf/models` route, not the assistant router.

| Step | Value |
|---|---|
| measurement | `ms-20261106T233135Z-00015848`, ready in ~1 s |
| nodes | `.11` 53.9 GiB (workstation), `.14` 19.9 GiB (headless); warnings `unmeasured_compute ×2` |
| inventory | `.11` 7 models with sizes, `.14` 5 — the field the free-space panel lists |
| recommendation | `dev-1.5b` feasible at 32 768 tokens, rank 0 `.11` (53.7 GiB headroom); 6 candidates refused with `choose_smaller_model` ×4 and `choose_pipeline_model` ×1; warnings `staging_required`, `speed_unknown`, `multi_node_required` |
| plan | `65f0a121d30a2d0a…`, coordinator `.11`, rank order `.11 → .14`, layers 27–28 (0.18 GiB) / 0–27 (1.63 GiB), endpoint `http://172.16.1.11:8099` `ready:false` |
| prediction | `predicted.measured: false`, `critical_stage_rank: 1`, `observed: null` — no rate promised |
| apply | `op-20261106T233303Z-apply-000197a1` → `applied` 23:33:06.343 |
| start | `op-20261106T233313Z-start-000239d8`, created 23:33:13.954 → `ready` 23:33:21.335 |
| endpoint | `http://172.16.1.11:8099`, `ready: true` |
| warnings | `performance_unmeasured`, `chat_profile_applied` (`dev-1.5b`) |
| runtime | rank 0 live on `.11`, launcher `ready`, assignments `.11` rank 0 / `.14` rank 1 |
| completion | `"Ready"`, 2 tokens, 0.73 s, `finish_reason=stop` |

### 11.1 The phase sequence the six rows map onto

From `~/.omlx/logs/server.log` (local time = UTC+1) and the operation record:

```text
00:33:03.298  apply  created            → applying,  phase=staging
00:33:06.003  staged 0.8 GiB to .14 in 2 s (0.46 GB/s)
00:33:06.343  apply  applied            → phase=applied ("nothing has been started")
00:33:13.954  start  created            → starting ("checking that .11 and its Macs answer")
00:33:15.735  loading                   → phase=loading
00:33:20.262  distributed engine ready  (/Users/andreaskolb/Models/dev-1.5b, ranks=2, ring)
00:33:21.262  readiness canary: POST /v1/completions → 200
00:33:21.334  start  ready              → phase=ready, endpoint.ready=true
```

That is exactly the mapping `SCREEN-4-PROGRESS.md` §2 records:

- rows 1–2 come from the measurement/recommendation jobs (measured, then
  recommended, in that order — the card shows Checking Macs and then Planning);
- row 3 is one row for `applying` + `staging`: the 2 s copy above is the only
  event between "runtime prepared" and "staged 2 of 2 Macs", so the pair cannot
  be split;
- row 4 is `starting`: 1.8 s of placement re-check and health before the load;
- row 5 is one row for `loading` + the canary: the engine was ready at
  `00:33:20.262` and the operation only became `ready` after the canary at
  `00:33:21.262` — there is no "loaded but unverified" state to show;
- row 6 is `ready` with `endpoint.ready=true`, which is what the record reports
  at `00:33:21.334`.

No percentage, byte count or rate appears anywhere in the operation record; the
only progress numbers are `nodes_ready/nodes_total` (`2/2`).

**Cleanup.** Unloaded after the walk:
`{"ok":true,"deployment_id":"dev-1.5b-b84fc2d85d74","stopped":true,"local_only":false}`,
then no live jobs and no launchers. The staged copy on `.14` was deleted through
the free-space route (§9.2), so both Macs are back to their own models.

---

## 12. Screen-5 handoff and `observed` (2026-11-07)

The chapter's decision and its fake-driven proof are in
[`SCREEN-5-HANDOFF.md`](SCREEN-5-HANDOFF.md). This section records the live
preconditions checked in this session and what a live run must show.

### 12.1 Preconditions checked

All four Macs answered on port 8099 and required admin authentication:

```text
172.16.1.11 401  172.16.1.14 401  172.16.1.12 401  172.16.1.13 401
```

On `.11`, with its own key read from `~/.omlx/settings.json` and never printed:

```text
GET /admin/api/cluster/assistant/operations/active
{"operation": null}
GET /admin/api/cluster/deployments
{"deployments": []}
```

`.14` (`2cc19e81…`) was the paired peer, `.12` was discovered but unpaired.
The origin server was restarted from this branch with the installed runtime
interpreter (`PYTHONPATH=<checkout>:<checkout>/apps/omlx-web`, port 8099); the
probe was switched on for the run by temporarily setting
`server.assistant_performance_probe=true`. `.14` still ran its older checkout,
which is enough: it serves its own admin/staging routes and the endpoint.

### 12.2 The live run (2026-11-07)

Origin and elected coordinator `.11`, peer `.14`, `dev-1.5b`, 32,768 tokens:

| Step | Value |
|---|---|
| measurement | `ms-20261107T054326Z-00011a18`, ready; warnings `unmeasured_compute ×2` |
| recommendation | `dev-1.5b` feasible at 32,768 tokens, rank 0 `.11` |
| plan | `68d44100cf0253f9c506f36aaded3b0717d0de784fb8d3520f53987b89d0dd36`, coordinator `.11`, rank order `.11 → .14`, endpoint `http://172.16.1.11:8099` |
| prediction | `predicted.measured: false`, `observed: null` before start |
| apply | `op-20261107T054330Z-apply-00013340` → `applied`, staged 2 of 2 Macs |
| start | `op-20261107T054332Z-start-00027fb3` → `ready`, endpoint `ready: true` |
| **observed** | `decode_tok_per_s: 66.35`, `prefill_tok_per_s: null`, `ttft_ms: null`, `measured_at: 2026-11-07T05:43:39.783562+00:00` |
| warnings | `performance_unmeasured`, `chat_profile_applied`; **no probe warning** |
| replan | `observed: null`, `predicted` identical to the approved plan |
| deployment | `dev-1.5b-dc77897d365e`, unloaded afterwards (`stopped: true`, teardown verified on 2 ranks) |

The rate is a real end-to-end completion on the launched cluster, not the
readiness canary and not the synthetic probe. It is stamped with the
coordinator's time and shown only because `observed` is non-null.

### 12.3 Finding: the distributed non-streaming path reports decode only

The first probe attempt (before the fix) recorded
`performance_probe_failed`: “the measurement completion reported no timing”.
Inspecting the same `/v1/chat/completions` reply showed:

```json
"usage": {
  "prompt_tokens": 51, "completion_tokens": 25,
  "total_time": 0.35, "generation_duration": 0.35,
  "generation_tokens_per_second": 71.79
}
```

There is no `time_to_first_token` and no `prompt_tokens_per_second` on the
distributed non-streaming path (TTFT is populated on the chat streaming path).
The honest fix is to require only `generation_tokens_per_second` and record
TTFT/prefill **only when the coordinator reports them** — never derive them.
The card therefore shows the measured decode rate alone on this path.

### 12.4 What this proves, and what it does not

* The nullable `observed` block exists on the operation, is filled after
  `ready`, and is timestamped; `predicted` and a re-run plan are unchanged.
* A probe failure is a warning on a still-`ready` operation (seen in §12.3’s
  first attempt).
* With the setting restored to its original value (absent, i.e. default off),
  the same start records `observed: null`; the server was restarted with the
  restored settings so the user’s configuration is byte-for-byte unchanged.
* The remote-coordinator handoff (Open Chat pointing at the coordinator’s own
  page) is covered by the Node tests; this run elected `.11` and therefore
  exercised the local link and the local probe path.

---

## 13. Screen-6 stop / start-again live check (2026-11-07)

Same origin/coordinator `.11`, peer `.14`, `dev-1.5b` at 32,768 tokens, with the
post-ready probe left at its default (off). The origin ran this branch; `.14`
ran its older checkout.

| Step | Value |
|---|---|
| measurement | `ms-20261107T055806Z-00017132`, ready |
| plan | `425e1eff772e4287157caa186fef69e00ee9243cab2ca1481b7091ce58bffaab`, coordinator `.11` |
| first apply | `op-20261107T055810Z-apply-000179ef` → `applied` |
| first start | `op-20261107T055812Z-start-00029ddc` → `ready`, deployment `dev-1.5b-ce3d12488e01`, endpoint `http://172.16.1.11:8099` |
| **stop** | `POST /admin/api/cluster/assistant/stop` → 200, `stopped: true`; warning `deployment_stopped` |
| record after stop | `state: ready`, `stopped: true`, `deployment_id: dev-1.5b-ce3d12488e01` |
| second stop | 200, `stopped: true` (idempotent; no second unload) |
| **start again** | same frozen inputs: apply `op-20261107T055821Z-apply-00039062` → `applied`, start `op-20261107T055823Z-start-0004b58f` → `ready`, same deployment ID, `stopped: false` |
| final stop | 200, `stopped: true` |

The stop went through the assistant route, not the browser: for this run the
coordinator was `.11` itself, so the route used the local loopback unload with
the key `.11` stores. The remote-coordinator half (the `remote_admin` SSH seam)
is covered by `tests/cluster_assistant/test_stop_api.py`; the run did not force a
remote coordinator.

After the final stop the deployments list still shows the configured
`dev-1.5b-ce3d12488e01` (unloaded, not running), which is the state “Start
again” starts from and the surface a reloaded page already shows.

---

## 14. Phase A resource policy (2026-11-07)

This section is the evidence record for the roadmap's R1–R7. The fake-driven
acceptance is green and the live line was reachable on 2026-11-07; §14.2 records
what ran and §14.3 what remains.

### 14.1 Fake-driven evidence (this session)

The resource policy is now one object from measure to start
(`omlx/cluster/assistant/policy.py`). `solve_cluster_plan` loops over subsets
from two upward and, at the smallest feasible size, over the lever ladder
(largest context first, then saved/resident, table offload, MoE 0.75/0.5/0.25,
a higher tier, and p95 only with explicit consent). The election still produces
every verdict through `plan_hybrid`.

Tests that pin it (`uv run pytest tests/cluster_assistant`):

- `test_policy.py` — each lever turns a refusal into a plan (table offload, MoE
  offload, a raised tier, a shorter context), the model's own context limit is
  never exceeded, p95 needs consent, a 4-Mac measurement plans on the 2 Macs
  that fit and names the other 2, a model that needs 3 returns 3 with the 2-Mac
  refusal recorded, `node_selection=all` forces every Mac, the policy changes
  the plan signature, the settings patch only enables what the plan uses, and
  both the Qwen4 PLE and DeepSeek V4.1 Engram switches are honoured.
- `test_policy_api.py` — the plan reports `selected_node_ids`, `excluded_nodes`
  and `resource_policy`; a table-offload plan writes the switch on every
  selected Mac; a failed write fails apply with `policy_settings_failed` and
  the one action `diagnostics`; a drifted policy is `plan_mismatch`.
- The existing suite still passes: 270 assistant tests, 1600 cluster tests
  (2 skipped), 316 web tests (177 skipped), `ruff` clean on the touched files.

### 14.2 Live evidence (2026-11-07)

The validation Macs were reachable. `.11` was restarted from this branch with
`uv run omlx serve`; `.12` was paired through the existing join/approve API and
runs this branch from `~/omlx-cluster-assistant-branch`; `.14` was synced to the
same branch. Start a ring **coordinator** from a Terminal on that Mac: an
SSH-launched server has no Local Network permission and its rank 0 fails with
`[ring] Couldn't connect (error: 60)`. On `.12` the branch server is launched by
opening `~/start-omlx-branch.command` in Terminal (`open -a Terminal …`), which
fixes that. Nothing below is a fake.

**A. Full path, small model (`dev-1.5b`, `.11`+`.14`)** — the first live run of
the new policy fields.

| Step | Value |
|---|---|
| measurement | `ms-20261107T083815Z-0001f7c9`, ready, 2 Macs |
| plan | `b0276f8dbe59952525e9c5fdf06be1f9c3d6390a746aa707adbc33949998264c` |
| plan policy | context 8,192; table `none`; MoE off; admission `p50`; tier per Mac |
| apply | `op-20261107T083822Z-apply-0001d555` → `applied` |
| start | `op-20261107T083824Z-start-00029a2f` → `ready`, deployment `dev-1.5b-df749cd5c557`, endpoint `http://172.16.1.11:8099` |
| completion | 13 tokens, 1.12 s, 11.65 tok/s |
| stop | 200, `stopped: true` |

**B. Four measured Macs → the two that fit → start**
(`GLM-5.3-Flash-oQ2e-mtp`, 115.8 GB / 107.8 GiB; measurement
`ms-20261107T091444Z-00040cab`):

| Value | Result |
|---|---|
| selected | `.12` (rank 0, 7 layers, 17.19 GiB, 4.59 GiB headroom) + `.11` (38 layers, 81.82 GiB, 15.40 GiB headroom) |
| excluded | `.14` and `.13`, “not needed: the selected 2 Macs fit this model at 8,192 tokens” |
| plan | `b8540d200a55cd1956bea07e8701d9e2a4312b0e0b9c3ce6452579228dda7991` |
| plan policy | context 8,192; table `none`; MoE off; admission `p50`; tiers per Mac |
| apply | `op-20261107T091456Z-apply-0007172d` → `applied`, warning `macs_not_needed` naming `.14`/`.13` |
| start | `op-20261107T091459Z-start-000852ca` → `ready`, deployment `GLM-5.3-Flash-oQ2e-mtp-45b3432bae2b`, endpoint `http://172.16.1.12:8099` |
| completion | 61 tokens (42 reasoning), 2.72 s, 22.44 tok/s |
| stop | 200, `stopped: true` |

**C. Qwen4 PLE — the table is found, the architecture cannot PP.**
`Qwen3.8-Flash-Next-oQ5e-mtp` is 134,971,338,685 bytes (125.7 GiB) and its
checkpoint carries `ple_layer_ids`; `inspect_safetensors_layout` reports
`table_offload_kind=ple` and the measurement reads
`qwen4_ple_ssd_offload: true`. But `supports_pipeline` is **false**: this VLM
does not implement the MLX-LM pipeline forward path. The assistant now refuses
it at plan time:

| Value | Result |
|---|---|
| measurement | `ms-20261107T085806Z-000163b1` |
| plan | 409 `no_feasible_plan`, action **`choose_pipeline_model`** |
| message | “cannot be split into pipeline stages … the Cluster Assistant plans pipeline parallelism only. Choose a pipeline-capable model” |

The earlier “feasible” Qwen plan in this session was a real bug: `plan` did not
enforce `ModelLayout.supports_pipeline`, and `inspect_safetensors_layout` did
not apply oMLX's pre-load architecture patches, so capability detection
depended on call order. Both are fixed: `inspect_safetensors_layout` now calls
`maybe_apply_pre_load_patches`, and `solve_cluster_plan` refuses a
non-pipeline model with `choose_pipeline_model` before the coordinator is
asked. A Qwen4 PLE **start** under the PP-only MVP therefore needs a
pipeline-capable PLE checkpoint; tensor parallelism is P2 and out of scope.

**D. DeepSeek V4.1 Engram start (`dev-v41-15L`, 87.7 GB, `.11`+`.12`).**
Measurement `ms-20261107T090048Z-00021e30`; `.12` was paired during this
session, so the plan could be applied.

| Value | Result |
|---|---|
| plan (default policy) | `da16963ffeefed0ba5e51ccf32450b71a1023bd087ed1724432db1013c760bd0` |
| plan policy | **table `engram` / `table_kind` `engram`**; context **4,096, reduced from 8,192**; admission `p50` |
| saved setting | `deepseek_v41_engram_ssd_offload: true` |
| apply | `op-20261107T085156Z-apply-0006554c` → `applied` |
| start | `op-20261107T085158Z-start-0007f61f` → `ready`, deployment `dev-v41-15L-5e6701638324`, endpoint `http://172.16.1.11:8099` |
| completion | 256 tokens, 5.17 s, 49.47 tok/s (the dev checkpoint answers in reasoning tokens) |
| stop | 200, `stopped: true` |

**E. Routed-expert offload start.** The same model with the table resident and
routed experts streamed at 25 % resident (plan `2694469b38d90a97e80b30634562ee0fa64b924497f12cf2d399fe7230ffe08f`,
`moe_offload_enabled` warning):

| Value | Result |
|---|---|
| plan policy | `table_offload: none` (`table_kind: engram`), `moe_expert_offload: true`, `moe_resident_fraction: 0.25`, context 4,096 |
| apply | `op-20261107T090108Z-apply-000331e9` → `applied` |
| start | `op-20261107T090110Z-start-00042f22` → `ready`, deployment `dev-v41-15L-24ab7035d1c0`, endpoint `http://172.16.1.11:8099` |
| completion | 256 tokens, 7.39 s, 34.64 tok/s |
| stop | 200, `stopped: true` |

After the MoE run the model's saved settings were restored to the original
`deepseek_v41_engram_ssd_offload: true`, `moe_expert_offload_enabled: false` on
`.11` and `.12`.

### 14.3 What remains

1. **Qwen4 PLE start** is not a resource-policy gap: the checkpoint is not
   pipeline-capable, and the MVP is PP-only. A pipeline-capable Qwen4 PLE
   checkpoint would plan with `ple` tables on SSD; the assistant now says
   `choose_pipeline_model` instead of producing a plan that can never stage.
2. **A pipeline-capable 150 GiB-class checkpoint** is not on the validation
   cluster. The 125.7 GiB Qwen is not PP; the 177.7 GiB GLM does not fit the
   pair even at the aggressive ceiling. The largest PP-capable model
   (GLM, 107.8 GiB) proves the 4-Mac measurement → 2-Mac start with the other
   two named. The 150 GiB figure remains a model-availability gap, not a
   planner gap.
3. **4-Mac start** of a >100 GiB model remains Phase D environment work: `.13`
   reports a 8.5 GiB live admission and 0.2 GiB free disk; `.14` has 26.7 GiB
   free. The 4-Mac **plan** and the 2-Mac **start** are proven above.
4. **MoE and Engram live starts are closed** (§14.2 D/E); the resident-fraction
   ladder is also covered by the fake tests.

The new `python_executable` carry was also proven necessary: a worker whose
checkout lives elsewhere fails its SSH preflight if the coordinator's
`sys.executable` is assumed. The measurement now stores the interpreter each
Mac reported and the deployment uses it.

Run live work from a Terminal on each Mac (never over SSH), with
`GET /admin/api/cluster/assistant/operations/active` answering
`{"operation": null}` first and no deployment loaded (a loaded cluster drops the
measured admission).

The new `python_executable` carry was also proven necessary: a worker whose
checkout lives elsewhere fails its SSH preflight if the coordinator's
`sys.executable` is assumed. The measurement now stores the interpreter each
Mac reported and the deployment uses it.

Run live work from a Terminal on each Mac (never over SSH), with
`GET /admin/api/cluster/assistant/operations/active` answering
`{"operation": null}` first and no deployment loaded (a loaded cluster drops the
measured admission).

## 15. Prefill working set on the 32 GiB Mac (2026-11-12, #5033 review)

This is the measurement behind review item 1 (“what grew on the 32 GiB Mac when
cut 9 ran out of memory?”). It is a real two-Mac run of the PR branch
(`feat/qwen4-exp-cluster`, `01f8af0e` plus the wording commit `f7953597`), not a
unit test.

Setup: `.11` (128 GiB) rank 0, layers 4–48; `.14` (32 GiB) rank 1, layers 0–4;
`throughput` profile (4096-token prefill chunks); `engram`/PLE table on SSD;
context 32768. The worker pair was launched directly with the argv
`build_mlx_launch_argv` produces, the same one the supervisor runs, and released
through `~/.omlx/cluster/runtime/dev-qwen4-prefill-serve.json` on both Macs.

### What the plan reserved

| Run | Layers on `.14` | Planned on `.14` | Headroom | Prefill reserved |
|---|---|---|---|---|
| before the fix (coefficient forced to 0 in the driver) | 0–7 | 14.05 GB | 11.95 GB | 0 |
| after the fix | 0–4 | 16.27 GB | 11.65 GB | 7.65 GB |

The two headrooms look alike and mean different things: the pre-fix 11.95 GB was
claimed against a prefill the plan did not price, and `.14` really needs ≥16 GB
for it at a 4096-token chunk. After the fix the same slot is a reservation, and
the extra layer that would not have fit is gone.

The 4096-token chunk is what makes the two slots differ: `interactive` (1024)
reserves 1.78 GiB at the same context, `balanced` (2048) 3.56 GiB, `throughput`
(4096) 7.65 GiB — that is
`MemoryMonitor.estimate_prefill_peak_bytes` on the **generic SDPA path**. The
rank guard prices this model through its own Qwen4 profile instead (gathered
QSA core, see below), which is smaller, so the plan is conservative against
admission rather than identical to it; §15.1 measures how far both are from the
real peak.

### What actually grew

`footprint <pid>` (`phys_footprint`, which includes the Metal allocations;
`ps -o rss` does not) on the 32 GiB Mac, 26,100-token prompt, unique nonce so the
prompt cache could not short-circuit the prefill:

| Stage on `.14` | Footprint after load | Peak during prefill | Wall clock |
|---|---|---|---|
| layers 0–4 (fixed plan) | 8.75 GB | **25 GB** | 27.4 s |
| layers 0–7 (pre-fix plan) | 14.0 GB | **25 GB** | 32.4 s |

Both runs stop at the same 25 GB because that is the GPU ceiling, not the
demand: `iogpu.wired_limit_mb` is `0` (macOS default) on a 32 GiB Mac. A 9-layer
stage (the maintainer’s cut 9) starts at ~17 GB, so the same prefill working set
has ~8 GB left where it wants ≥16 GB — that is the OOM.

So the answer is: **the prefill working set of the head stage**, at least 16 GB
above the resident stage at a 4096-token chunk for this checkpoint, not the KV
cache and not the weights. It is chunk-sized transient work that is gone again
by the next chunk and that the plan used to price at zero. §15.1 gives the
scaling and §15.2 the fact that rules out the obvious explanation: it is not
proportional to attention layers. It is also not the expert dispatch — 1024
tokens × top-10 × 640 × 2 B is 13 MB of intermediate per layer.

`sampling_rank_only` was run both ways on the same split. Off, the answer stays
coherent (`We need answer user's request. User provided long text with nonce and
many repeated sentences, then asks: "Answer in…"`), which is the `lm_head` fix:
rank 1 now samples from the real head instead of a randomly initialised one.

### 15.1 What the working set scales with (2026-11-12)

Same stage (`.14`, layers 0–4, 8.75 GB after load), same fixed plan, same
26.1k-token prompt with a fresh nonce, only the prefill chunk changed. The plan
was not re-cut per chunk, so the reservation column is what the plan for a 26k
context would charge at that chunk; `PREFILL_CHUNK` overrides the worker's
`--prefill-step-size` without re-planning.

| Prefill chunk | Profile | Peak | Δ above load | Plan reserves | Wall clock |
|---|---|---|---|---|---|
| 1024 | `interactive` | 16 GB | +7.25 GB | 1.42 GiB | 34.8 s |
| 2048 | `balanced` | 22 GB | +13.25 GB | 2.84 GiB | 29.7 s |
| 4096 | `throughput` | 25 GB | ≥ +16.3 GB | 5.68 GiB | 27.4 s |
| 8192 | `throughput` + override | 25 GB | ≥ +16.3 GB | 11.36 GiB | 27.1 s |

The 4096 and 8192 rows stop at the same 25 GB for the reason above (wired
ceiling), so they are lower bounds; 1024 and 2048 stay below it and are real.
All four completed and answered coherently. Against the two real measurements
the reservation is 5.1× under (1.42 vs 7.25 GiB) and 4.6× under (2.84 vs 13.25
GiB).

Context sweep at a fixed 1024-token chunk, fresh process, increasing prompts so
the high-water mark is attributable:

| Prompt tokens | Peak | Δ above load | Wall clock |
|---|---|---|---|
| 4,612 | 14 GB | +5.25 GB | 8.8 s |
| 13,107 | 16 GB | +7.25 GB | 15.3 s |
| 26,116 | 16 GB | +7.25 GB | 31.3 s |

Two conclusions, both measurable:

- **The chunk sets the size.** Halving the chunk from 2048 to 1024 halves the
  working set (13.25 → 7.25 GB). That is why the plan reserves a chunk-scaled
  term at all.
- **The context hardly matters above ~13k.** At a 1024-token chunk the peak only
  moves from 5.25 GB (4.6k context) to 7.25 GB (26k). The estimator’s
  `per_token × context` shape is therefore the pessimistic direction, not the
  reason for the gap.

### 15.2 Where the growth sits

The same request ran on both stages at the same time, so `footprint` on the
128 GiB Mac is the same prefill seen from the other side:

| Stage | Layers | Full-attention layers | Δ over load (chunk 1024, 26.1k prompt) |
|---|---|---|---|
| `.14`, head | 0–4 | 1 | **+7.25 GB** (8.75 → 16 GB) |
| `.11`, tail | 4–48 | 11 | **+4.0 GB** (75 → 79 GB peak, fresh launch) |

The tail’s 4 GB spread over 44 layers is 0.09 GB per layer, which is the
attention/GDN transient the estimator reasons about. The head grows 1.8× as much
with 1/11 of the attention layers — 1.45 GB per layer — so the working set is
*not* proportional to attention layers. It is concentrated on the stage that
holds layer 1, the only layer on either side with the SSD-backed PLE n-gram
embedding (`ple_layer_ids: [2]`, 1-based; 128 shards, 38.4 GB of
`ngram_embedding.shards.*`). Per chunk token the head’s growth is 5.1 MB at a
4.6k context and 7.1 MB at 26k.

It is not the table being paged in either: one lookup over this prompt reads
26k tokens × 24 n-gram rows × ~1.3 KB ≈ 0.8 GB of file data, an order of
magnitude below the spike, and `vmmap --summary` on the idle worker lists
177 MB of mapped-file pages against 9 GB of footprint.

That is the accurate answer to “what grew”: **the per-chunk working set of the
stage that carries the PLE layer**, sized by the chunk and only weakly by the
context. The stage’s single full-attention layer cannot produce 7 GB: one fp32
score matrix for 1024 × 4608 keys × 24 heads is 453 MB.

### Residual gap (reported, not papered over)

At a 26k context and a 4096-token chunk three different numbers are in play:

| Source | Bytes | vs. measured (≥16.3 GB) |
|---|---|---|
| rank guard, Qwen4 gathered profile | 1.34 GiB | 12× under |
| plan, generic SDPA path (what §15 reserves) | 5.68 GiB | ≥ 2.9× under |
| measured peak on `.14` | ≥ 16.3 GB | — |

Neither price names the quantity §15.2 measured. The guard's profile
(`_Qwen4ExpPrefillMemoryProfile`) prices a gathered QSA core of
`indexer_budget` (2048) keys for **one** layer; the plan prices generic SDPA for
every layer. Both are attention terms, and §15.2 shows the growth is
concentrated where the attention layers are not: the 11-attention-layer stage
grows 4 GB where the 1-attention-layer stage grows 7.25 GB. The plan’s 5.68 GiB
landing at ~a third of the peak is a coincidence of magnitude on this
checkpoint, not the right mechanism.

On `.14` at 26k/1024 that is a 21× miss for the guard (0.34 GiB priced against
7.25 GB measured) and it is why cut 9 passed admission.

So closing the gap is not a constant in front of the attention term. The
planner has to charge the **table-holding stage** a per-chunk term for the layer
in `ple_layer_ids` (Engram has the same shape), sized by the chunk — the
`ModelLayout` already knows which layer holds the table and how many bytes it
is, so the placement decision can carry it; the coefficient per chunk token
does not exist in the config and would have to come from this measurement,
which is why this PR stops at the correct *scaling* and reports the rest. This
PR also does not touch admission, and the generic formula is why MLA layouts
stay at 0.

What that means for this split: the reservation moved the planner from 0–7 to
0–4 on `.14`, and the 0–4 stage completes a 26k prefill at chunk 4096 (25 GB of
a 26 GB budget, 27 s). A 9-layer stage (~17 GB resident) would still OOM at
chunk 4096 with this fix, because the real demand there is ~16 GB and it has
~8 GB; the lever for such a stage is the chunk (`interactive`: 7.25 GB measured)
or the context, and that is the follow-up above.

## 16. Three Macs on the planner's own split (2026-11-13)

Until now the PR's split had only been exercised on two Macs. This is the N=3
case the planner actually emits: `.12` Mac Studio M2 Ultra (64 GiB) rank 0,
layers 21–47; `.14` Mac Studio M1 Max (32 GiB) rank 1, layers 10–20; `.13` Mac
Studio M1 Max (32 GiB) rank 2, layers 0–9 with the PLE table on SSD. Plan hash
`1a7e9e22…`, world size 3, 1024-token chunks. The worker trio was launched
directly with the argv `build_mlx_launch_argv` produces — there is no supervisor
on that path, so the serve gate was written by hand — and released through
`dev-qwen4-3rank-serve.json` on all three Macs.

One operational detail decides whether a reproduction works at all: **rank 0
has to be started from that Mac's own login session.** An ssh-spawned rank-0
worker never completes the ring handshake — 7 of 7 attempts failed with
`[ring] Couldn't connect (error: 60)` on the peers and `EHOSTUNREACH (65)` from
rank 2, while the listener was bound, the firewalls were off and 120/120 pings
returned. A minimal ring probe over the same hosts and the same ports passes, so
the fault is the spawn context of rank 0, not the fabric.

Load, against the plan (`footprint` prints GiB under a `GB` label — the same
tool §15 used, so the measured column and the planned column are directly
comparable):

| Rank | Mac | Layers | Planned | Measured weights | Footprint after load | Guard ceiling |
|---|---|---|---|---|---|---|
| 0 | `.12`, 64 GiB | 21–47 | 46.8 GiB | 46.35 GiB | 47.00 GiB | 52.1 GiB |
| 1 | `.14`, 32 GiB | 10–20 | 19.8 GiB | 19.62 GiB | 20.00 GiB | 23.4 GiB |
| 2 | `.13`, 32 GiB | 0–9 | 18.1 GiB | 17.98 GiB | 19.00 GiB | 23.4 GiB |

All three ranks reported `rank_ready` about 20 s after launch and the ring came
up (four connections `.12`↔`.13`, one `.12`↔`.14`). One real completion through
`127.0.0.1:8099` on `.12` — 5,868 prompt tokens at a 1024-token chunk, 8 new
tokens — returned HTTP 200 in 31.03 s and answered. Its peaks:

| Rank | Peak during the completion | Guard ceiling that run |
|---|---|---|
| 0 | 51.00 GiB | 52.1 GiB |
| 1 | 25.00 GiB | 23.4 GiB |
| 2 | 24.00 GiB | 23.4 GiB |

Two readings of that table, both needed:

- **The three-Mac plan is real.** The split loads, the ring handshake completes,
  the measured weights sit at or below the plan's weight term on every rank (the
  resident footprint adds 0.2–0.9 GiB of fixed overhead), and the endpoint
  serves a real completion.
- **The small Macs finish at their Metal ceiling, not at the plan's headroom.**
  The guard's 23.4 GiB was a projection from the free memory at launch; both
  32 GiB stages ran at 24–25 GiB and were fine, while `.12` finished 1.1 GiB
  under its own cap (52.1 GiB). The guard ceiling is therefore conservative on
  the small Macs and biting on the big one, and roughly 1 GiB is what is left
  everywhere. What that leaves for a larger chunk or a longer context is a
  separate measurement: §15.1's chunk scaling was taken on the two-Mac split and
  is not extended to N=3 here.

Requests have to name the model the worker loaded. `--model` goes through
`Path(...).expanduser()`, so a `~/…` argument is fine; the request's `model` is
compared against that one string, and `default_model` is the alias that always
resolves to it.

Not taken in this PR: a 32 GiB Mac as rank 0. The stage that would move to the
server is the one already peaking at 24.00 GiB of a 32 GiB Mac, and the HTTP
server, the sampling head and the request's own prefill would all land on it.
That swap asks for its own measurement, and a separate PR if it is wanted.

### 16. Qwen3.8 PLE is pipeline-capable in the current build (2026-11-13)

The earlier §14.2 C observation that `Qwen3.8-Flash-Next-oQ5e-mtp` is not
pipeline-capable is **stale for the current checkpoint/build**. The current
`inspect_safetensors_layout` reports `supports_pipeline=True` (48 layers,
`table_offload_kind=ple`), and the assistant started a live 3-Mac, 32k, MTP-PLE
deployment:

| Value | Result |
|---|---|
| measurement | `ms-20261108T212638Z-000104b7` |
| plan | 3 Macs, local rank 0, context 32,768 |
| deployment | `Qwen3.8-Flash-Next-oQ5e-mtp-85060a3c7085` |
| endpoint | `http://172.16.1.11:8099` |
| observed | prefill 11.99 tok/s, decode 15.34 tok/s, TTFT 80 ms |

The non-PP rule itself is unchanged: a checkpoint whose layout really reports
`supports_pipeline=false` is still refused with `choose_pipeline_model`. Only
this checkpoint/version's capability claim is corrected.

## 17. Compute probe corrected against published specs (2026-11-13, E4=B)

The review (REVIEW-2026-11-13.md B10) found the synthetic probe was
latency-bound: a `rows=1`, 2048-wide (8 MiB) GEMV with 5 iterations reported
hardware-impossible rates that inverted the chip ordering, and the election
consumed them as stable throughput.

**Expectation baseline (researched, not speculated** — Apple newsroom,
Wikipedia, everymac, 2026-11-13):

| Chip | GPU cores | Bandwidth | FP32 | AI compute | RAM (cluster) |
|---|---|---|---|---|---|
| M1 Max (.13/.14) | 32 | 400 GB/s | 10.4 TF | baseline | 32 GB |
| M2 Ultra (.12) | 76 | **800 GB/s** | 27.2 TF | ~2× M1 Max | 64 GB |
| M5 Max (.11) | 40 + NPU/core | 614 GB/s | ~17 TF | **~6× M1 Max** | **128 GB** |

Two physics drive pipeline timing (`planner._predict_stage_seconds` weights
decode vs prefill by the execution profile):
- **Decode is memory-bound** → must order by bandwidth: M2 Ultra > M5 Max > M1 Max.
- **Prefill is compute-bound** → must order by AI compute: M5 Max > M2 Ultra >
  M1 Max (the M5 Max's per-GPU-core Neural Accelerators beat the M2 Ultra's
  raw core count for fp16 matmul).

**First attempt (insufficient):** widening the decode GEMV to 8192-wide raised
the rate but it stayed *kernel-limited* (~20–30 % of bandwidth): MLX's rows=1
GEMM efficiency varies by architecture and still placed the M5 Max *below* the
M1 Max (119 vs 146 GB/s) — wrong, since 614 > 400 GB/s. A small-M GEMV is unfit
for the decode rate.

**Final probe:**
- Decode = a streaming **bandwidth reduction** (`_bandwidth_rate`, 128 MiB 2-D
  buffer, `mx.sum`), which saturates the memory controllers and orders reliably.
- Prefill = a **compute-bound GEMM** (`_weight_rate`, rows=1024 × width=8192),
  whose weight-read rate tracks fp16 compute (rows=64 was not compute-bound).
- `ComputeProfile.plausible()` (election) treats a rate outside 50 GB/s..1.2 TB/s
  as unknown compute → headroom preference, never a fabricated ranking.

Measured live (`.11` repo checkout; peers via `~/.omlx/bin/omlx-cluster-python`
over SSH — a standalone MLX op has GPU access over SSH; the rank-0 login-session
constraint is about the distributed ring handshake, not basic Metal):

| Mac | Hardware | Decode GB/s (bandwidth) | Prefill GB/s (compute) |
|---|---|---|---|
| .12 | M2 Ultra | **281** | 17.1 |
| .11 | M5 Max | 233 | **54.2** |
| .13 | M1 Max | 131 | 6.8 |
| .14 | M1 Max | 110 | — |

Both orderings now match the specs: decode M2 Ultra > M5 Max > M1 Max
(800 > 614 > 400); prefill M5 Max > M2 Ultra > M1 Max. The compute-bound GEMM
measured 56 / 18.8 / 7.1 TFLOPS (M5 Max / M2 Ultra / M1 Max), i.e. M5 Max ≈ 7.9×
M1 Max — close to Apple's "6× AI compute vs M1 Max" claim. Absolute bandwidth is
~50–60 % of peak (typical for a reduction); the *relative* ordering is the
calibration signal, not a tokens/s promise.

Fake-driven tests pin these spec expectations through the real planner timing
(`tests/cluster_assistant/test_probe_spec_expectations.py`); the live numbers
above are the evidence, per AGENTS.md.

**Deployment caveat:** the peers run the *installed app* oMLX
(`/Applications/oMLX.app/…`), which does not yet contain the corrected probe;
the fix must be deployed to the peers (or they must run this checkout) before a
live cluster measurement uses it. The plausibility guard meanwhile rejects the
old stored latency-bound rates as unknown compute.

## 18. Live smoke of the inert assistant stack (2026-11-13, post-review)

After the review fixes (REVIEW-FIX-PLAN.md slices 0–11), the origin oMLX was
started on .11 (`omlx serve --port 8099 --headless`) and the inert assistant
flow exercised live against all four paired Macs (no deployment):

| Step | Result |
|---|---|
| hardware-only measure | `ms-20261113T201939Z-00012ba0`, ready; .11 admission p05 54.0 GiB, .12 17.0, .13 19.6, .14 20.7; `unmeasured_compute` (no compute probed, as designed) |
| recommend (4 candidates, 32k) | selected `dev-1.5b` @ 32,768; order dev-1.5b → GLM-5.3-oQ2e → Qwen3.8-oQ5e → dev-v41-15L, all feasible. With the plausibility guard treating the stored latency-bound rates as unknown compute, the tie-break is smallest size (B11 ranking) |
| plan on the hardware-only measurement | refused `409 stale_measurement`, action `measure` — the frozen C3 contract (plan needs a model-bound measurement) intact |
| model-bound measure of dev-1.5b | `ms-20261113T202127Z-00022b4d`, ready |
| plan (dev-1.5b, 32k) | **feasible**, 2-Mac minimal subset, coordinator `.14` (c84867fe), rank order [.14, .13]; the guard's headroom preference elects the two 32 GiB Macs that fit |

This validates live: hardware-only entry measurement (C3), the new
recommendation ranking (B11), the plan identity guard, the minimal-subset
election, and the compute plausibility guard (E4=B) rejecting the old stored
latency-bound rates. The apply/start deployment step was not exercised here:
the elected coordinator is remote, and a live activation needs rank 0 started
from the coordinator's own GUI login session (§16) — a guided next step.

## 19. Real per-node calibration validated live (2026-11-13)

The calibration runner (`calibration.calibrate_node`) was exercised on .11
(M5 Max) against a real `dev-1.5b` completion through the local oMLX server
(`omlx serve --port 8099`), streamed with `include_usage`:

| Value | Result |
|---|---|
| dev-1.5b weight bytes | 0.869 GB (28 layers) |
| real decode rate | **100.5 GB/s** (~116 tok/s) |
| real prefill rate | 4.9 GB/s |
| decode plausible (50..1200 GB/s) | yes |

The real decode rate (100.5 GB/s) is below the pure-bandwidth probe (233 GB/s,
§17) because a real model decode carries kernel/attention overhead — this is the
*actual* decode throughput the ranking should use, not a microbenchmark ceiling.
The conversion `decode_rate = decode_tok/s × model_weight_bytes` recovers a
plausible bandwidth-proportional rate from a real completion. The serving model
id is the directory name (`dev-1.5b`), not the absolute path; the completion
streams (SSE), and the final usage chunk carries the timing.

## 20. First live search-loop smoke (2026-11-13, partial)

The automatic search loop was exercised live on the 4-Mac cluster with dev-1.5b
(8k context, priority=speed, max_count=2):

| Constellation | Coordinator | Result |
|---|---|---|
| [.14, .13] (2-Mac) | .14 (remote) | **infeasible** — rank 0 spawned over SSH on .14 never completed the ring handshake (no GUI login session, §16) |
| [.11, .12, .13, .14] (4-Mac) | .11 (local) | **measured 38.7 tok/s decode**, deployment `dev-1.5b-2e3e31b4f121` |

The loop correctly ranked, deployed, measured, and picked the best *measured*
constellation (the 4-Mac local-coordinator one); the remote-coordinator 2-Mac
candidate failed and was recorded infeasible, never preferred. This confirms the
loop logic end-to-end, and re-confirms the §16 constraint: a constellation whose
coordinator is a remote Mac needs that Mac's oMLX app running in its own GUI
login session. The fix is to deploy the signed app (with helper) to the nodes and
launch it in their GUI sessions, rather than SSH-spawning rank 0.

## 21. Full live search loop on the signed app (2026-11-13, success)

After deploying the signed slim app (Developer ID YB894QRG4W, notarized) to the
nodes and launching it in their GUI login sessions, and fixing the search-loop
`plan_mismatch` (candidates now carry their resolved resource policy so
`plan_apply` re-proves the signature), the full loop ran live with dev-1.5b
(8k, priority=speed, max_count=2, **headless roles on all Macs** — oMLX is
conservative in memory management, so headless yields far more admission:
.11 p05 97.2 GiB vs 54.0 as workstation):

| Constellation | Coordinator | Measured decode |
|---|---|---|
| [.13, .12, .11] (3-Mac) | .13 (remote) | 8.38 tok/s |
| [.14, .13, .12, .11] (4-Mac) | .14 (remote) | **53.19 tok/s** |

Both remote-coordinator constellations deployed and measured (the §16/§20
rank-0-over-SSH failure is gone now the coordinator's app runs in its GUI
session). The loop picked the best measured (4-Mac, 53.19 tok/s, deployment
`dev-1.5b-2382cf1ad6d7`). Measure → rank → oMLX plans+starts → measure → pick
best, end-to-end on real hardware.

## 22. Single-node vs cluster benchmark (2026-11-14, dev-1.5b)

Same load on every configuration (dev-1.5b, observe prompt, 96 tokens, headless
roles, 8k context where clustered; single-node = the Mac's own server, no
pipeline):

| Configuration | Decode tok/s |
|---|---|
| **M5 Max (.11) single** | **254.2** |
| M2 Ultra (.12) single | 115.9 |
| M1 Max (.13) single | 82.2 |
| M5+M2 cluster (.11+.12), coord .11 | 102.8 |
| 3-Mac cluster [.13,.12,.11], coord .13 | 8.38 |
| 4-Mac cluster [.14,.13,.12,.11], coord .14 | 53.19 |

**Key finding: for a model this small (0.869 GB, fits on one Mac with room to
spare), single-node wins decisively.** The M5 Max single (254.2) is ~2.5× the
M5+M2 cluster (102.8) and ~5× the 4-Mac cluster (53.19): pipeline ring
communication and stage synchronisation cost more than they save when the whole
model already fits in one Mac's memory. The cluster pays off only for models
too large for one Mac (or to buy context length). The 3-Mac result (8.38) is
anomalously slow — a poor rank order / topology bottleneck (M1 Max coordinator)
— and shows the search loop's value: it *measures* each constellation rather
than assuming more Macs is faster.

Product consequence: the recommendation must recognise when a model fits on a
single Mac and prefer single-node inference; the distributed path (2+ hosts) is
for models that do not fit, or for context the single Mac cannot hold. The
search loop already picks the best *measured* constellation, but a single-node
option belongs in the candidate set for small models.

Note: the M5 Max single decode (254.2 tok/s) is higher than the §19 calibration
estimate (~116 tok/s from 100.5 GB/s); the calibration is a relative bandwidth
signal under its own prompt, not an absolute tok/s prediction for every load.

## 23. Full async search loop + tune on the signed app cluster (2026-11-14)

The complete real-measurement loop ran live on the signed app cluster (dev-1.5b,
8k, headless, priority=speed, max_count=2):

| Step | Result |
|---|---|
| measure (real calibration, headless) | ready |
| POST /search (async job) | handle immediately; progress "measuring constellation 1/2" → "2/2" → completed |
| 4-Mac [.11,.12,.13,.14] coord .14 | measured 84.3 tok/s |
| 3-Mac [.11,.12,.13] coord .11 | **measured 99.17 tok/s** (best) |
| tune the winner | **chunk 2048** (153.9 tok/s; 4096→148.5, 8192→145.7), **sampler balanced**, **MTP off** (146.7 vs 144.0) |

The async job returns a handle immediately (no minutes-long blocking request),
streams progress as each constellation is measured, and the UI polls it. The
tune then sweeps chunk/sampler/MTP with real completions and picks by measured
throughput + golden-prompt quality. Two live bugs found and fixed along the way:
the tune deadlocked because the blocking completion seam ran on the event loop
(now `asyncio.to_thread`), and the tuning bodies did not request a streamed
completion so no usage chunk came back (now `stream:True` +
`include_usage`).

This closes the end-to-end path: measure → rank → oMLX plans+starts → measure
(async job) → best → tune, all on real hardware with the signed app.

## 24. MTP sweep red→green: skipped when the model has no MTP head (2026-11-14)

dev-1.5b has no MTP heads (`mtp_compatible: (False, 'model has no MTP heads in
config')`), and MTP is a per-model setting, not a request field — so the tune's
MTP A/B (sending `mtp: true/false` in the chat body) measured two identical
no-ops and picked "off" on noise.

| | `mtp` | `mtp_rates` |
|---|---|---|
| **RED** (§23, before fix) | `False` | `{False: 146.73, True: 143.97}` |
| **GREEN** (after fix) | **`None`** (not applicable) | **`{}`** (nothing measured) |

The tune route now determines `mtp_capable` from the model
(`mtp_compatible_for_model`, fail-closed) and passes it to `fine_tune`;
`sweep_mtp` returns `{mtp: None, rates: {}, applicable: False}` without running
when the model has no head, and the response carries `mtp_applicable`. Live
search still ran alongside (3-Mac 95.24 tok/s best over 4-Mac 85.94).

A real MTP A/B for MTP-capable models (DeepSeek V4.1, Qwen3.8, GLM with heads)
remains a follow-up: it needs a redeploy with `mtp_enabled` on vs off, not a
request field. The chunk sweep may share the same per-request-vs-per-model
caveat (`prefill_chunk_size`) and should be verified against the engine's real
field before trusting its winner.

## 25. Chunk + MTP sweeps red→green: model settings, not request fields (2026-11-14)

`prefill_chunk_size` is not a field on the chat-completion request, and the MTP
knobs (`mtp_enabled`, `mtp_adaptive_max_depth`) are per-model settings too. The
tune's per-request chunk sweep therefore measured the same default chunk three
times (noise) and the MTP "A/B" two identical no-ops. Only the sampler
(`temperature`/`top_p`/`top_k`) is a real request field.

| | `chunk` | `chunk_applicable` | `mtp` | `mtp_applicable` |
|---|---|---|---|---|
| **RED** (before) | 8192 (noise) | — | False (noise) | — |
| **GREEN** (after) | **None** | **False** | **None** | **False** |

The tune route now passes `chunk_applicable=False` and reports it, so the card
says "not tuned via request" instead of a noise winner; MTP is skipped without a
head (or marked not-applicable). The sampler sweep stays real (balanced won).

Real chunk / MTP-on-off / MTP-depth tuning remains the redeploy-tuning feature:
each config is a per-model setting, so it needs an apply+start with the setting
written and a measurement per config (not a request field). The MTP-depth A/B
(`mtp_adaptive_max_depth`) and the one-off MTP-off-vs-on comparison need an
MTP-capable model (Qwen3.8/GLM) deployed — dev-1.5b has no heads.


## 26. Tuning presets + redeploy tuning (2026-11-14, software slice)

Tuning a large model is expensive and mtplx-like: the sampler can be swept on a
running cluster, but the MTP switch and the MTP adaptive depth are per-model
settings, so changing them needs a redeploy per config. The result should be a
**one-off**: tune once per (model, node config), save the winner, and a later
Start replays it instead of re-tuning.

### 26.1 What was built (fake-driven)

| Piece | Behaviour |
|---|---|
| `tuning_store.py` | `TuningPresetStore`: atomic JSON at `<base_path>/cluster/assistant-tuning-presets.json`, schema 1, keyed by `tuning.preset_key(model_signature, hardware_key)`; a corrupt file fails closed to "no preset"; never reused across hardware. |
| `POST /tune` | Saves the winning sampler preset and returns `preset_key`. |
| Start replay | `start.apply_tuning_preset()` writes the preset's real `ModelSettings` (`mtp_adaptive_max_depth`, `temperature`/`top_p`/`top_k`) on the elected coordinator **before** activation, so the launch uses the tuned MTP depth. It writes only fields still unset (never a hand-set value), and records `tuning_preset_applied` / `tuning_preset_skipped` — never silent, never able to fail the start. |
| `redeploy_tuning.py` | Pure config planning + ranking: `plan_configs`, `choose_best`, `sweep`. A model without an MTP head only measures `mtp_off`; a failed deploy is an infeasible row, never a rate; the winner is the highest measured decode, then prefill, then the lower depth (adaptive wins a tie). |
| `evaluate_constellation(model_settings_patch=…)` | Writes a config's model settings before the launch; a failed write drops the config rather than silently measuring the baseline. |
| `POST /tune-redeploy` + `GET /tune-redeploy/{job_id}` | A background job (a deploy per config) that reuses `diverse_candidates` + `evaluate_constellation` (the proven search body), stops each measured config, ranks the real rates, and merges the MTP winner into the model's preset (keeps the sampler preset). |

### 26.2 What is not live yet

No live sweep is recorded here. The MTP off-vs-on comparison and the
`mtp_adaptive_max_depth` depth sweep need an MTP-capable model (Qwen3.8/GLM,
~117 GiB); `dev-1.5b` and `dev-v41-15L` have no MTP heads, so `plan_configs`
correctly offers only `mtp_off` for them. The `prefill_chunk_size` knob is not a
`ModelSettings` field in this build, so the chunk stays honestly "not applicable"
(§25) rather than faked with a redeploy.

The software path is the same apply/start/observe body that is already live in
§21/§23; only the per-config settings write is new. The live red→green for the
MTP sweep is the next step once a large MTP checkpoint is available.


## 27. Live MTP redeploy sweep on GLM-5.3 (2026-11-14)

**Setup.** `GLM-5.3-Flash-oQ2e-mtp` (103 GiB, 45 layers, MTP head 4.3 GiB) on the
minimal 2-Mac line the election chose: `.12` (M2 Ultra, **coordinator/rank 0 —
not the origin**) + `.11` (M5 Max, this Mac), 32768 context. Measurement
`ms-20261114T102546Z-0001294e`; the plan excluded `.13`/`.14` ("not needed").
The app origin `.11` was rank 1, which is the intended proof that the origin does
not decide rank 0.

**The sweep** (`POST /tune-redeploy`, one real apply+start+measure per config,
~7 min total). Decode rates are oMLX's own `usage` from one real completion each:

| config | decode tok/s |
|---|---|
| `mtp_off` | 22.07 |
| `mtp_on_adaptive` | 29.04 |
| `mtp_on_depth_1` | 22.30 |
| `mtp_on_depth_2` | 30.72 |
| **`mtp_on_depth_3`** | **31.60** |
| `mtp_on_depth_4` | 30.90 |

Winner: **MTP on, depth 3 = 31.6 tok/s**, +43 % over MTP off (22.07). The preset
was saved (`e4824a5e7f644130`: model `GLM-5.3-Flash-oQ2e-mtp`, hardware
`.12,.11`, `mtp=true`, `mtp_depth=3`). Adaptive depth (14.9 on an earlier run,
29.0 here) is clearly worse than a fixed tuned depth.

**Five real blockers found and fixed (commit "auto-free space on distribute,
live-tuning fixes"):**

1. **Case-only path** — `~/models` vs the inventory's `~/Models` made
   `model_present` false for every Mac, so apply tried to copy a model that was
   already there and hit `insufficient_disk`. The canonical path fixes the run;
   `reclaim.model_key` lower-cases the folder name so a copy is recognised
   across case differences.
2. **Older coordinator build** — `.12` lacks the new `/assistant/operations/*`
   routes, so the cluster-wide lock mirror returned 405 and aborted every
   apply. A missing lock route (404/405) is now "no lock available", logged, not
   a failure: a mixed-version cluster must still deploy.
3. **Sweep on a running cluster** — redeploying while the winner held the model
   was refused "does not fit … right now". The sweep now unloads the running
   cluster once before the first config.
4. **MTP flips the optimal split** — with MTP off the election moved rank 0 from
   the M2 Ultra to the M5 Max, so the winner's exact rank order was infeasible.
   `choose_candidate` measures the best fewest-Mac split at the target context
   for that config, which gave the honest MTP-off baseline instead of a
   fabricated "infeasible" row.
5. **Preset must be authoritative** — the sweep's last config wrote depth 4, so
   a Start that only fills empty fields kept 4 instead of the winner 3. The
   tuning preset now writes its measured values over a leftover setting (the
   chat profile keeps the opposite "fill only empty" rule).

**Automatic free-disk (user decision 2026-11-14).** The delete rule is now
enforced automatically when a model must be distributed: an `insufficient_disk`
apply frees the safest unused model(s) on the short Mac (only a model that Mac
listed, never the planned checkpoint, never the last cluster copy, largest
first), retries once with the freed bytes counted, and records one `space_freed`
warning. The pure selection is `reclaim.py` (7 tests) + the route orchestration
(4 tests); the search/tuning loop never auto-deletes.

**Not applicable.** `prefill_chunk_size` is not a `ModelSettings` field in this
build, so the chunk stays honestly "not applicable" (§25).

**Preset replay verified live (2026-11-14).** After a fresh measure/plan/apply
on the same `.12`+`.11` set, `POST /start` returned `ready` and the operation
warning **`tuning_preset_applied` — "the saved tuning preset was applied before
launch: mtp_adaptive_max_depth"**. So the saved winner (depth 3) is written on the
coordinator before activation and the cluster "just starts" with it. Two fixes
made this reliable live, both committed:

- the preset is **authoritative** (it replaces a leftover sweep value; the earlier
  run kept depth 4 from the last config instead of the winner 3);
- the preset key is the **sorted node set**, not the rank order, so a later
  election that flips rank 0 (MTP off did) still finds the preset.

The first replay attempt failed with a real environment issue, not the preset: a
leftover cluster rank still held ~40 GiB on `.11`, so activation saw "needs 71.5
GiB but can admit 31.4 GiB right now". Unloading every registered deployment and
killing the stale `inference_worker` ranks freed it (`.11` back to 107.9 GiB) and
the next start loaded normally. Stale deployments from earlier sessions are the
practical hazard; the assistant's own apply/start never leave one running.

That failure now names its cause: a coordinator refusing activation with "does
not fit … right now" is reported with the action `diagnostics` and the remedy
("another deployment or app on a selected Mac is still holding memory; stop the
previous cluster (Screen 6) or close apps, then Start again"), instead of the
misleading `retry_coordinator`. A full cross-coordinator auto-cleanup of leftover
cluster deployments is a later slice; the manual path (unload registered
deployments, kill orphan `inference_worker` ranks) is the verified remedy.

### 27.1 Deferred live proofs (items 2 and 6 of the 2026-11-14 order)

Both have real, environmental blockers; nothing was fabricated to look green.

**Item 2 — automatic free-disk, live.** The rule is fake-driven (reclaim.py 7
tests + route orchestration 4 tests), but a natural live trigger needs a
*selected* Mac that lacks the model, is short of disk for the copy, and holds a
deletable model. On the current cluster none qualifies: `.12` has ~235 GiB free
(after the GLM-4bit deletion) and holds every large candidate; `.13` has 358 GiB
free and `.14` 3 TB free. The one candidate that would not fit (`.13` for the
366 GiB DeepSeek) is excluded on memory grounds before the disk check, so the
plan never reaches it. Unblocking it means either a genuinely short Mac or a
safe, non-destructive way to reserve space — filling a disk by hand is not that.
The rule stays unit-tested until such a Mac exists.

**Item 6 — a second MTP model, live.** Only `GLM-5.3-Flash-oQ2e-mtp` (103 GiB)
fits the 2-Mac admission (`.11` 108 + `.12` 47 GiB) and is live-swept (§27).
`GLM-5.3-Flash-MLX-4bit-MTP` (169 GiB) and `DeepSeek-V4.1-Flash` (366 GiB)
exceed that budget; `Qwen3.8-Flash-Next-oQ5e-mtp` was not pipeline-capable at
the time of this entry (2026-11-14 — superseded by #5033, which is applied on
this branch since 2026-11-16, so the family is PP-capable now). A
4-Mac `GLM-4bit` plan is memory-marginal (185 GiB summed admission for a 169 GiB
model + KV) and would need six ~169 GiB redeploys — hours of staging — so it is
an explicit opt-in, not a default validation step. The preset key is already the
sorted node set, so a different model/node config keys its own preset.

### 27.2 Context-vs-experts tradeoff is the user's choice (2026-11-14)

Two levers trade against each other: reducing context costs quality but keeps
speed, while routing experts to SSD keeps quality but costs speed. The assistant
only decides what costs nothing — a checkpoint's lookup table (Engram/PLE) is
offloaded by default — and leaves the real trade to the user at plan time.

- **Plan** returns the **quality-first** result: the largest fitting context,
  willing to offload routed experts (the worst case, e.g. 8k + 75 % resident, is
  surfaced in `resource_policy` when that is the only way it runs).
- When experts are offloaded, the plan also returns `tradeoffs`: the best
  **experts-resident** alternative (`label: speed`), a smaller context with its
  own `resource_policy` and `plan_signature`.
- The user picks; `apply` uses the chosen `resource_policy` + signature. No
  upfront context-ceiling question and no silent context drop.
- Implementation: `policy._levers(exclude_expert_offload=…)` +
  `policy.solve_cluster_plan(exclude_expert_offload=…)` compute the alternative;
  `routes._speed_alternative` embeds it; `models._plan_tradeoffs` /
  `AssistantPlanResponse.tradeoffs` expose it. Tests: the solver's quality plan
  keeps experts offloaded, the exclude pass yields the resident alternative, and
  the tradeoff builder drops a non-resident "alternative".
- The dashboard renders the choice in the plan card: a "Context vs speed"
  block (only when the plan returns tradeoffs) with "Best quality" and the
  experts-resident alternative(s); apply and start use the chosen option's
  `resource_policy`, `plan_signature`, `coordinator_node` and `rank_order`
  (`assistantPlanSource()`), and the policy labels follow the selection. A Node
  test proves selecting the alternative sends its signature/levers.

## 28. Single-node "run-local" flow (2026-11-14, fake-driven)

Without a cluster, a model that fits one Mac must still be runnable. Cluster
apply refuses a single-Mac plan (`single_node_plan`/`add_mac`) by design, so the
assistant gained a **separate** local-run flow (decision E6):

- `POST /admin/api/cluster/assistant/run-local` `{measurement_id, model_path,
  layer_count, context_tokens, mtp}` → a job handle; `GET …/run-local/{job_id}`
  polls `queued → running → ready | failed`. One run at a time
  (`409 local_run_in_progress`); a failed job is always terminal (a shell wraps
  unexpected errors), so the guard never stays closed.
- Fit is the recommendation's own rule (`recommendation._single_node_fit`, the one
  `budget_for` conversion), not a second estimate: a too-large context is
  `context_too_large`/`adjust_model_or_context` naming what fits; nothing fits is
  `insufficient_memory`/`choose_smaller_model`. It never silently lowers the
  context and never targets another Mac.
- It replays the tuning preset for `(model, hardware_key([this Mac]))` — the same
  merge Start uses — then asks this Mac's engine pool to load the model and
  verifies the pool reports `loaded` before `ready`. Everything is loopback with
  this Mac's key; no SSH.
- The card shows "Run on this Mac" only when the recommendation's `single_node`
  answer names this Mac; on ready it shows the local endpoint and Open Chat.

**Evidence:** fake-driven (assistant + engine + web green). The live run-local is
still not recorded, but the two loading blockers are now understood and fixed at
the code level (2026-11-14):

- **Distributed-vs-local load (the real cause).** `POST /api/models/{id}/load`
  had no way to say "this Mac": `EnginePool.get_engine` adopted a **stale cluster
  deployment** of the checkpoint (`DistributedBatchedEngine`) and compared its
  per-rank shards (`.11` 48.45 GiB, `.13` 19.83 GiB) against live admission — the
  exact "rank 0 needs 48.5 / rank 2 needs 19.8" error. Fixed with a `local_only`
  flag threaded through `load_model` → `EnginePool.get_engine` → the
  distributed-adoption/resident-size/load paths; `run-local` calls
  `…/load?local_only=true` and fails `local_engine_required` if the pool ever
  reports `distributed: true`.
- **Fit reconciliation.** `recommendation._single_node_fit` charged the whole
  table while the engine (and the plan lever) discounts it via PLE/Engram SSD
  offload. It now uses the resident layer bytes (total − table offload), the same
  bytes the engine prices (measured: 125.7 GiB resident → 88.15 GiB admitted with
  35.76 GiB PLE on SSD). `run-local` writes the offload switch before the load and
  reports `table_offload_enabled`; it caps context at the declared limit. Residual
  (documented, not hidden): the engine's 5 % PLE allowance leaves a narrow
  [88.15, ~92] GiB band where it may still refuse; that maps to
  `insufficient_memory`/`adjust_model_or_context` with exact bytes — no silent
  fallback.
- **False LLM fallback.** A failed *deployment* was rewritten as a local VLM→LLM
  fallback; `qwen4_exp` is not in `VLM_NATIVE_TEXT_MODEL_TYPES`, so mlx-lm's
  strict loader rejected the VLM-only tensors ("Received N parameters not in
  model"). A failed deployment is no longer rewritten as a local fallback. Whether
  `VLMBatchedEngine` alone loads Qwen3.8 on `.11` is now reachable but **untested**
  (a ~120 GiB whole-machine load); the one-command check is
  `POST /api/models/Qwen3.8-Flash-Next-oQ5e-mtp/load?local_only=true` with
  `qwen4_ple_ssd_offload=true`.
- **Stop / canary.** `POST /assistant/run-local/{job_id}/stop` aborts a
  queued/running job (failed, `local_run_stopped`) or unloads a ready one
  (`stopped=true`, endpoint hidden), own-Mac unload, idempotent. `ready` now also
  requires a **canary** (one `max_tokens=2` completion through the loopback seam;
  no rate claimed); failure is `local_canary_failed`/`diagnostics`. The card has a
  "Stop local run" button.
- **Note / discrepancy:** the stored measurement says Qwen3.8
  `supports_pipeline=True`, while PHASE-2B-RUN §27.6 (older note) says it is not
  pipeline-capable; the code is the authority (`supports_pipeline` comes from the
  inspected layout).

**Live evidence on `.11` (2026-11-14) — Qwen3.8 single-node works.** The real
`run-local` flow (measure → run-local) loaded `Qwen3.8-Flash-Next-oQ5e-mtp` on
`.11` alone and served it:
- measurement `ms-20261114T154133Z-0001bd7c` (this Mac only), job
  `local-13cf8aa0f615`; ceiling 107.9 GiB, nothing else loaded.
- state `ready` in ~30 s (PLE table streamed from SSD, not resident); warnings
  `table_offload_enabled` (PLE on SSD so the model fits this Mac) and
  `tuning_preset_skipped` (no saved preset for this model + Mac yet).
- endpoint `http://192.168.178.132:8099`, readiness canary passed.
- one real completion: coherent answer, **decode 40.3 tok/s, TTFT 0.34 s**.

So with PLE SSD offload the 117 GiB VLM **loads single-node on the M5 Max and
serves**. The earlier single-node failure was the load adopting a stale cluster
deployment (`local_only` now prevents it) plus a dead text fallback, not a local
memory limit. The cluster PP `rank 1` TypeError is separate and still needs a
cluster. Not yet wired: tuning a single-node run (the cluster `/tune` and
`/tune-redeploy` are cluster-only), so `tuning_preset_skipped` is expected until a
single-node tuning path exists.

**Single-node tuning (2026-11-14).** A ready `run-local` can now be tuned on this
Mac: `POST /assistant/tune-local {job_id}` starts a job (`GET …/{id}` polls
queued→running→completed|failed; one at a time; `run-local` refuses while a tune
holds the pool). It runs the sampler sweep on the loaded model, then the MTP
configs (`redeploy_tuning.plan_configs`) by writing `mtp_enabled`/depth,
unloading, loading `?local_only=true`, and measuring one real completion's
generation rate; then it writes the winning settings, reloads, and saves ONE
preset for `(model, hardware_key([this Mac]))` so a later Run on this Mac replays
it. The model is left loaded with the winner. Honest: a config that fails to
load/measure is an infeasible row, never a rate; a still-unloading engine is
waited out (never measured with the old settings); if the winner cannot be left
loaded the job fails with one action and saves no preset. The card has a "Tune on
this Mac" block (button + honest progress + per-config rows + winner + preset).
Tests: 15 backend + 2 UI.

## 29. First fully green 2-Mac cluster run, with any-to-any start (2026-11-15)

The 0.8.0rc1 rollout (runbook §1.5) paired the four Macs fresh and carried
measure → recommend → plan → apply → start to a running endpoint. It also found
and fixed the last plan/start disagreement (R23, §16 of
`UPSTREAM-PR-REVIEW-2026-11-15.md`).

Nodes: `.11` `c09cc74f-c164-4c62-80fb-49de94eba8ee` (M5 Max, 128 GiB), `.12`
`9078f858-d20b-43dc-b420-d54031bf2943` (M2 Ultra, 64 GiB), `.13` `2dd48599-…`
and `.14` `8092d5d1-…` (M1 Max, 32 GiB each). Origin `.11`; coordinator **`.12`**
— the origin is *not* rank 0.

Hardware-only measurement, then the model-bound one for
`/Users/andreaskolb/Models/GLM-5.3-Flash-oQ2e-mtp` with `mtp=true`:
`ms-20261115T190346Z-000131c5` (ready).

Plan `c587ce6071529eda…`: minimal 2-Mac set, topology `line` / complete /
measured, `resource_policy.memory_guard_tier = aggressive` (warning
`tier_raised`), `admission = p50`, context 32 768, warnings `tier_raised` +
`macs_not_needed` (`.13`/`.14` excluded). Predicted critical stage 0.929 s on
rank 1.

| Rank | Mac | Layers | resident | admission | headroom | predicted |
|---|---|---|---:|---:|---:|---:|
| 0 (coordinator, endpoint) | `.12` | 31–45 (14) | 37.2 GiB | 39.8 GiB | 2.6 GiB | 0.927 s |
| 1 | `.11` | 0–31 (31) | 66.1 GiB | 99.8 GiB | 33.8 GiB | **0.929 s** (critical) |

Rank 0 additionally holds 4.0 GiB of head/coordinator memory (LM head + MTP
draft head). The plan states it itself: `rank 0 holds layers 31-45 with 2.6 GiB
headroom inside 39.8 GiB admission`.

The split is forced, not preferred. oMLX spawns rank 0 as the sampler's process
and the planner requires rank 0 to own the tail, so the tail must fit the Mac
that becomes rank 0. `.12` cannot hold the 66.1 GiB head stage, so the only
feasible 2-Mac order is `.12` = tail/rank 0. Pinning `.11` as rank 0 would need
a smaller (unbalanced, slower) tail; that is a product decision, not a bug.

apply `op-20261115T190433Z-apply-000192aa` → `applied`; start
`op-20261115T190519Z-start-00026aff` → `starting` → `loading` → **`ready`** in
~50 s, endpoint `http://172.16.1.12:8099`, warnings `performance_unmeasured` +
`chat_profile_applied`. The origin drove a *remote* coordinator over the SSH
control plane (any-to-any) and the coordinator spawned its own rank 0 locally.

One real completion through the endpoint (`/v1/chat/completions`,
non-streaming):

```text
model  GLM-5.3-Flash-oQ2e-mtp
text   17 × 23 equals 391.
usage  prompt 25, completion 66 (55 reasoning), 3.52 s, generation 18.76 tok/s
```

`observed` stayed `null` (the performance probe is off by default), so the
operation claims no rate; the number above is the server's own `usage`. TTFT and
prefill therefore remain unmeasured for this build — the next run turns the
probe on so prefill can be attributed per stage.

Stop (Screen 6): `POST /admin/api/cluster/assistant/stop` with the ready start
operation → `stopped: true`, deployment `GLM-5.3-Flash-oQ2e-mtp-f8ef5eced870`;
`.12` then reported `loaded: 0`, `current_model_memory: 0` and no rank
processes. `GET /assistant/session` returned the same ready start with its
frozen inputs, coordinator `.12` and `stopped: false` before the stop — the
resume surface works for a remote coordinator too.

**The fix this run needed (R23).** The first three start attempts failed on a
cleanly applied plan:

```text
503: rank 1 (9078f858-…) needs 40.9 GiB but can admit 40.4 GiB right now
```

40.9 GiB did not move when the context dropped to 24 576 (weights dominate), and
the reported admission wandered between 40.4 and 40.8 GiB. The plan priced the
stage with `memory_guard_tier = aggressive` while the launch preflight read the
Mac's *live* ceiling from a server still guarding at `balanced`; the reserve
difference is exactly that gap. apply now writes the approved tier through
oMLX's own `/admin/api/global-settings` route on every selected Mac before
staging, and the refusal names the tier it read. `memory_guard_tier` was
`aggressive` on both `.12` and `.11` afterwards.

**Mixed builds.** The coordinator `.12` ran the installed 0.8.0rc1 while the
origin `.11` ran this tree, because only the origin needs the new apply code and
the rank processes are launched by the coordinator's own interpreter. That is
not a §1.5-clean cluster: the rebuilt DMG that contains this fix is installed on
all four Macs before the repeated run.

## 30. W1+W2 live run: the helper as a capability and the in-app uninstall (2026-11-15/16)

W1 (D1 uninstall) and W2 (D6 helper capability) implemented and live-checked on
`.11`, from a wiped state, with the interactive steps done by a person at the
Mac. Builds (all four Macs, identical app-binary hash each time):

| DMG | App binary | Content |
|---|---|---|
| `19a446d9…` | `01e7898c…` | session start (pre-W1/W2) |
| `9bb7dc89…` | `aa19aac5…` | W1+W2, first Welcome scroll fix |
| `6ad5bde7…` | `8b8c63ca…` | compact Welcome panel, runtime row always visible |
| `6dfaf640…` | `4194baa3…` | quit-after-sheet + real helper removal |

**W2 live (A5).** The Mac was wiped but the kernel limit stayed at 122 880 MB
and the install ledger still said `1` — exactly the D6 defect state in which the
old UI showed nothing. The new UI showed **Install** anyway; one admin dialog
blessed the helper (`/Library/PrivilegedHelperTools/app.omlx.helper` +
LaunchDaemon plist, Team `YB894QRG4W`); the status seam read the disk truth, not
the stale ledger. Welcome then showed „installed (v1)"; Performance showed the
permanent row plus the wired-limit row **122 880 MB of 122 880 MB requested**
with Apply hidden and Install/Reinstall still visible (state 4, live), and
Reinstall ran **without a second dialog** (state 2, live). States 3 (other
version) and 5 (limit below request) are not producible on this Mac right now
and stay XCTest-covered (five states, injected status, no prompt).

**W1 live (A6).** The first real uninstall (build `8b8c63ca…`) moved the app to
the Trash, removed runtime/settings/cluster state, stopped the server, and kept
the 902 GB of models untouched. The identity was removed via the sheet's opt-in
switch (user-confirmed: the switch was on; the keep-default is unit-proven).
Two defects surfaced and were fixed (`caa3901a`):

1. **Quit after the sheet.** `NSApp.terminate` is ignored inside a sheet button
   action and returns without quitting while a sheet is attached; the window's
   close button is disabled under a sheet, so the only way out was File → Close
   plus the menubar Quit. `quitAppNow` now ends every attached sheet and
   terminates deferred (the `UpdateController.performSwap` lesson).
2. **Helper leftovers.** `SMJobRemove` deregistered the launchd job but left
   the root-owned tool and plist (mtime unchanged) — the same leftover the
   cleanup script knows. The uninstall's one admin dialog now runs the
   lab-proven fixed command (bootout + `rm -f` both files). The leftover files
   from the pre-fix run are removed by the next uninstall of `4194baa3…` or by
   Welcome's Reinstall re-blessing them.

**Welcome UX (found live, fixed).** The fixed 680×620 window with a
non-scrollable card stack overflowed into the footer: rows (including the
**API key**) slid under it and Start Server's validation pointed at an
invisible field, with an overlay scrollbar macOS only reveals while scrolling.
Now: one compact panel with every row visible (Port, Model Directory, API Key,
Python runtime, System helper), runtime log/errors below the panel, binding
hint folded into one line, hero shrunk, window resizable (680×660 default,
620 minimum), ScrollView only as a safety net. The runtime row follows the D6
capability rule: **always present**, „Installed — Python and MLX are ready." /
„Not installed — …", install button only while missing (user: no
system-dependent appearance of buttons).

**Launchpad zombie (diagnosed, not a build defect).** „The bundled runtime
manifest is missing" + a wired-limit failure appeared after the uninstall: the
user had launched the **DerivedData Debug test build** registered by the Xcode
test host (no `runtime-project`, unsigned). The zombie was deregistered and
deleted; the notarized app carries the manifest (verified in the Trash copy).

**New product decision (recorded in AGENTS §10.7).** *Distributed inference is
on by default* (`omlx/settings.py`, `exposure.py`): a fresh install must reach
the Cluster Assistant without hunting for the switch; saved explicit values
still win; the settings hint was updated in all 11 locales.

**Suites.** `tests/cluster_assistant` 658 passed; macOS python packaging 27
passed; full Xcode 441 passed / 0 failed (`-scheme oMLX CODE_SIGNING_ALLOWED=NO
-testLanguage en`); full pytest `tests` + `apps/omlx-web/tests`
**20 765 passed / 0 failed** (483 skipped; the two wizard-test failures the
rank-0 pin's Mac picker had tripped were fixed by allowing exactly the one
coordinator select). `make web` is a no-op. The distributed-inference
default-on build is the final artefact: DMG `6958fda5…`, app binary
`09f51c2d…`, notarized + stapled, identical on all four Macs.

