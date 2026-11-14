# Phase 2b runbook — operator sheet

Companion to [`PHASE-2B-RUN.md`](PHASE-2B-RUN.md). This is the click list used on
the real Macs: intention, preconditions, exact clicks, values to type, expected
result, and what to write down so a defect can be reproduced.

Everything here is the shipped product surface. No step needs a terminal, an IP
choice beyond the one address written below, or a rank/layer field.

---

## 0. The four Macs and their machine IDs

Before the discovery fix a Mac could only be identified by its machine ID; the
fallback announcement now carries the friendly name, the chip/RAM and the IPv4
addresses, so the cards name themselves. The table stays as a cross-check.

| Machine ID starts with | Mac | Chip / RAM | Address | Admin UI |
|---|---|---|---|---|
| `d7f9d455` | MacBook-Pro-von-Andreas | M5 Max · 128 GB | `172.16.1.11:8099` | http://172.16.1.11:8099/admin |
| `2cc19e81` | MacStudio-von-Andreas-2 | M1 Max · 32 GB | `172.16.1.14:8099` | http://172.16.1.14:8099/admin |
| `11618ac4` | MacStudio-von-Andreas | M1 Max · 32 GB | `172.16.1.12:8099` | http://172.16.1.12:8099/admin |
| (fifth, M2 Ultra) | Mac-Studio-M2Ultra | M2 Ultra · 64 GB | `172.16.1.13` | none (scenario E) |

Why the ID: mDNS is unavailable on this network for a server started from a
terminal, so the fallback announcement carries only the machine ID and port —
not the name or the chip. That is recorded as a finding in
`PHASE-2B-RUN.md`; the name arrives with pairing.

---

## 1. Pair the M1 Max (`2cc19e81`, `.14`) into the M5's cluster

**Intention:** create the 2-Mac cluster used by scenarios A, B and D.

**Preconditions:** both UIs logged in, both Cluster tabs open, nothing paired.

| # | Mac | Clicks | Values | Expected |
|---|---|---|---|---|
| 1 | `.14` | Cluster tab → card **“MacBook-Pro-von-Andreas”** (Apple M5 Max · 128 GB, *Found nearby*) → **Show code instead** | — | card **“Joining MacBook-Pro-von-Andreas's cluster”**, six digits, “Code expires in …” |
| 1b | *(fallback)* `.14` | if the card is missing or reads *Unreachable*: **Add by IP** → **Add & show code** | `172.16.1.11:8099` | same six-digit card |
| 2 | `.11` | Cluster tab → card **“MacStudio-von-Andreas-2 — Wants to join”** → type the code → **Approve & pair** | the six digits | toast **“MacStudio-von-Andreas-2 joined the cluster.”** |
| 3 | `.14` | wait | — | toast **“This Mac joined MacBook-Pro-von-Andreas's cluster.”**, both cards switch to **Paired** with chip + RAM |

**Do not** type `.14`'s address into “Add & show code” on `.11`: on the
coordinator you only approve the incoming request.

**Record:** the two toasts, and whether the paired cards show name, chip and RAM.

**If it fails:** write down which step, the exact message, and whether the card
offered a button (and its label).

---

## 2. Scenario A — origin `.14`, non-coordinator origin

**Intention:** prove that the app origin does not decide rank 0, and that an
origin which is not the coordinator can still drive the setup.

**Preconditions:** step 1 done; `.11` and `.14` both serving; no deployment
running (Cluster tab shows no active cluster).

| # | Clicks | Expected |
|---|---|---|
| 1 | `.14` → Cluster → step forward until the heading is **Plan** | card **“Set up this cluster”** |
| 2 | **Find and start a setup** | *Checking Macs* → *Choosing a model for these Macs* → *Preparing measurements for GLM-5.3-Flash-oQ2e-mtp* → summary with **Model · Context · Coordinator · Macs** and **Start setup** |
| 3 | write down the **Coordinator** name | it must be an election result; `.14` or `.11` are both legal, but the origin must not win just by being the origin |
| 4 | **Start setup** | *Preparing the model* → *Starting <coordinator>* → *Loading the model* → **Ready** with endpoint `http://<coordinator>:8099`, **Open Chat**, **Connect agent** |
| 5 | **Open Chat** → send `hello` | a reply arrives |

**Record:** coordinator name, model, context, endpoint, the time from *Start setup*
to *Ready*, and whether the endpoint only appeared when Ready.

**If it fails:** the exact message, the single button label, and whether the card
had already shown a plan summary.

---

## 3. Scenario B — origin `.11`, local-origin path

**Intention:** the same flow where the origin is also a candidate coordinator.

**Preconditions:** pair the M2 Ultra (`.13`) from its own UI first (Add by IP
`172.16.1.11:8099`, approve on `.11`). `.13` has only 14 GiB free, so a staging
refusal there is an environment result, not a defect.

Repeat §2 on `.11`. Expected: coordinator `.11` or `.13`; the endpoint sits on
whichever was elected.

**Record:** as §2, plus whether the plan spans 2 Macs.

---

## 4. Scenario C — three Macs

**Intention:** no manual ranks, IPs or layer counts for 3 Macs.

**Preconditions:** pair the second Mac Studio (`.12`) as well, from its own UI.

Repeat §2 on `.11`. Expected: **Macs: 3 Macs** in the summary; no IP, SSH, rank
or layer field anywhere in the primary path.

**Record:** the three stage rows in **Details**, the coordinator, the endpoint,
and the time to Ready.

---

## 5. Scenario D — the coordinator refuses to start

**Intention:** one action, same coordinator, no local fallback.

**Preconditions:** scenario A done; the assistant is on **Ready** or on a fresh
plan; I stop oMLX on the elected coordinator (the agent does this and tells you).

| # | Clicks | Expected |
|---|---|---|
| 1 | `.14` → **Find and start a setup** → **Start setup** | a single failure card: message that the coordinator did not answer, **one** button; its label comes from `retry_coordinator` (“Open/configure oMLX on <coordinator>”) |
| 2 | do nothing else | nothing starts on `.14`; the endpoint stays hidden |

**Record:** the message, the button label, and whether the endpoint was hidden.

---

## 6. Scenario E — an unpaired Mac in the plan

**Intention:** `pair_mac`, and nothing started.

This one cannot be produced from the UI, because the UI only offers paired Macs.
The agent drives it through the same admin API with the unpaired M2 Ultra and
records the refusal; the UI's rendering of `pair_mac` is covered by the static
component tests.

**Record:** the status code, `error.code`, `error.action`, and the fact that no
operation exists afterwards.

---

## 7. Scenario F — the finished endpoint

**Intention:** the endpoint is real, not a claim.

| # | Do | Expected |
|---|---|---|
| 1 | with the cluster Ready, open the endpoint URL + `/v1/models` | a model list |
| 2 | one completion through the UI's chat or `curl` | a real answer |

**Record:** the model id, the completion text, and the time it took.