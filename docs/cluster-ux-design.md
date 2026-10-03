# oMLX Cluster UX Design: Pairing, Diagnostics, and Fabric Doctor

**Status:** Design proposal — ready for implementation review
**Scope:** macOS two-node (extensible to N) Thunderbolt/RDMA clusters
**Grounded in:** the alytaphoenix (Mac mini M4 Pro) ↔ aphoenix (work MacBook M4 Pro, Cloudflare WARP full-tunnel) debugging session, and the current code in `/Users/alytaphoenix/repos/omlx`.

---

## 0. Traceability: every observed failure → the mechanism that kills it

A reviewer should be able to check this document against the incident. Each numbered failure from the debugging session maps to a specific design mechanism below. If a failure has no row, the design is incomplete.

| # | Failure | Section | Mechanism |
|---|---------|---------|-----------|
| 1 | Bare-hostname SSH → wrong account; corporate `User` override; "Permission denied" with no username hint | A.3, A.4 | Identity exchange over HTTP *before* first SSH; destination always `user@host` (explicit user on the command line beats any `ssh_config User` line); preflight P3 "account check" with copy that names both accounts |
| 2 | Coordinator's absolute home path baked into deployment; peer re-staged 17 GB and preflight lied ("model directory is missing") | A.5, A.6 | `home_dir` is a stored field of the paired-peer record; all peer paths are declared `~`-relative by contract; preflight P7 verifies staging root exists *on the peer's own home*; cross-user pairing is a first-class UI state, not an anomaly |
| 3 | GSSAPI hangs 30–90 s per SSH attempt | A.4 | `GSSAPIAuthentication=no` added to the one SSH policy choke point (`ssh_policy.py` — absent today, see A.4); preflight timeout budget assumes it |
| 4 | Auto-assigned 10.0.1.x swallowed by WARP; fix was ephemeral 172.16.99.x | C.3, C.5, C.6 | Collision-checked subnet selection (never a hardcoded default); empirical VPN-swallow probe *before* claiming fabric ready; durable addressing via `networksetup -setmanual`; link-setup respects existing manual addressing instead of re-addressing |
| 5 | `Transport: peer_linked_config_pending`, `fabric_verified: false`, stale 169.254 address — none actionable | B.2, C.2 | TransportState extended past its current ceiling into a full readiness ladder; every state carries `reason` + `remedy` copy; stale-address detection is an explicit Fabric Doctor check |
| 6 | `[jaccl] Couldn't connect (error: 60)` with silent exponential backoff | B.3, C.4 | Errno-mapped structured errors; retry loops emit progress incidents ("attempt 3/6, still timing out — this usually means…") instead of silence; Fabric Doctor JACCL probe reproduces the connect *outside* the collective and diagnoses it |
| 7 | Memory-ceiling probe hit hardcoded dead `127.0.0.1:9000`, silently fell back to slow path | B.3, C.7 | "Silent fallback" is banned by the incident model — every fallback emits a WARN incident; the probe targets the peer's *actual* advertised admin port (stored in the peer record), and the dead-port case becomes a named check |
| 8 | GUI failed silently: "Preparing model…" flashed, poll wiped error state, dead button | B.3, B.4 | Start Cluster becomes a **server-owned job** with a persisted incident log; polls merge monotonically and may never clear errors; buttons render job state, never local component state; asset-version handshake detects stale cached JS |
| 9 | "Model does not fit (18.6 GB required)" — actually a stale Workstation-role 32 GB reserve | B.5 | The "Why can't I start?" memory row always shows the *live* `admission_ceiling_bytes` **and its breakdown** (physical − role reserve − other apps), with the role named and editable inline; the invariant is already stated in code (`models.py:122–126`) — the panel enforces it |
| 10 | VLM only supports tensor-parallel; UI offered pipeline and 400'd at runtime | B.5 | Parallelism modes are gated by a capability query at plan time (`pipeline_compat.pipeline_assignment_is_honored()`, `pipeline_compat.py:30`); incompatible modes render disabled with the reason inline |
| 11 | Support bundles existed but were undiscoverable; staging failures buried | B.6 | Every failure surface carries a "Get details" affordance that opens the relevant bundle slice; staging jobs are first-class rows in the readiness panel |

---

## A. Easier pairing

### A.1 Goals

1. A user with two Macs, different local accounts, different home directories, and a corporate-managed `~/.ssh/config` on one of them can pair in under two minutes without touching a terminal.
2. Pairing never claims success it hasn't verified. "Paired" means: bidirectional SSH as the correct users, bidirectional control-plane HTTP, and agreed identities — all proven, not assumed.
3. Every failure names the actual variable that's wrong (the username, the key, the port) and offers a one-click or copy-paste remedy.

### A.2 The core inversion: identity before SSH

Today's flow SSHes first and learns identity later — which is exactly backwards, because SSH is the step that *needs* the identity. The `ssh_user` field was added to `ClusterStatus` (`omlx/cluster/models.py:142–146`, populated by `probe.local_login_name()`, `probe.py:385–395`) precisely because "a bare hostname … OpenSSH resolves against the *caller's* account (or the caller's ssh_config `User`) — the wrong user." But status is served by an already-running peer — there's a chicken-and-egg problem if pairing needs SSH to read status.

The existing enrollment machinery resolves it. `ClusterEnrollmentStore.issue_join_key()` / `claim()` / `complete()` (`omlx/cluster/enrollment.py:195, 236, 281`) already define a join-key handshake over the control-plane HTTP port that requires **no SSH at all**. Design rule:

> **Everything SSH needs is exchanged over the join-key HTTP channel first.** SSH happens only after both sides know each other's login name, home directory, host-key fingerprint, and public key. The first SSH attempt is already fully specified — there is nothing left for OpenSSH defaults or a corporate ssh_config to guess wrong.

### A.3 Pairing flow — screens and states

Five screens. Each screen owns one question; auto-detection answers it when possible and the user confirms.

**Screen 1 — Discover ("Add a Mac")**
- Auto: Bonjour browse for oMLX instances + Thunderbolt peer-name detection (`ThunderboltPort.peer_names`, `models.py:80`). Each candidate row shows: device name, chip, "connected by Thunderbolt ✓/–", oMLX version.
- Asked: nothing, unless discovery finds nothing — then a manual field: "Enter the other Mac's address or name" with inline hint *"On the other Mac, open oMLX → Cluster → 'This Mac's address'."*
- State names: `discovering`, `candidates_found`, `manual_entry`.

**Screen 2 — Identify (join-key exchange)**
- The peer's oMLX shows a 6-digit join code (an `issue_join_key` invocation surfaced as UI). The coordinator's user types it (or scans a QR of the join URL).
- On `claim`, both sides exchange an **identity envelope** over HTTPS (see A.6 data model): login short name, home dir, hostnames/addresses, SSH host-key fingerprint, oMLX admin port, cluster public key, protocol version.
- Auto-detected and *displayed, not hidden*: the account situation. Copy when accounts differ:
  > **Different accounts — that's fine.** This Mac is `alytaphoenix`; "Aphoenix's MacBook Pro" is `aphoenix`. oMLX will always connect as the right user on each side. Model files will live under each Mac's own home folder.
- This is the moment cross-user/cross-home becomes a *first-class, understood case* rather than a surprise at deploy time.
- State names: `awaiting_code`, `claiming`, `identity_exchanged`, `identity_mismatch` (e.g., protocol version incompatible).

**Screen 3 — Trust (key installation)**
- Auto: each side generates/uses the managed identity `~/.ssh/omlx_cluster` (`ssh_policy.py:21`) and sends its public key inside the join session. Each side appends the peer's key to its own `authorized_keys` — locally, by the local oMLX process, as the local user. **No SSH-copy-id over the network, no password prompt, no dependence on Remote Login being pre-shared.**
- The only asked item: if Remote Login (sshd) is off on either Mac, show a one-click "Turn on Remote Login" (System Settings deep link `x-apple.systempreferences:com.apple.Sharing-Settings.extension`) plus copy-paste `sudo systemsetup -setremotelogin on`.
- State names: `keys_installed`, `sshd_disabled_local`, `sshd_disabled_peer`.

**Screen 4 — Verify (bidirectional preflight)**
- Runs the preflight ladder in A.5, **both directions**, live check-by-check with pass/fail rows. Nothing is claimed paired until every row passes or is explicitly waived.
- State names: `verifying`, `verified`, `verify_failed:<check_id>`.

**Screen 5 — Paired**
- Shows the durable summary card: `alytaphoenix@mini.local ⇄ aphoenix@aphoenix-mbp.local`, "verified both directions 12s ago", link type (Thunderbolt 5), and a "Run checks again" button.
- Pairing writes the `PairedPeer` record (A.6) via `ClusterEnrollmentStore.complete()`.

### A.4 SSH policy changes (the one choke point)

`apply_cluster_ssh_policy()` (`ssh_policy.py:66`) is already the single place every cluster ssh/scp goes through. Three mandated changes:

1. **Destination is always `user@host`.** The user comes from the stored `PairedPeer.ssh_user`, never from OpenSSH resolution. An explicit user on the command line **overrides any `User` line in the caller's or a corporate `~/.ssh/config`** — this is the property that defeats failure #1's `User alyta.phoenix` override, and the doc-comment in the code should say so. A bare-hostname destination in cluster code becomes a programming error (assert/lint in `apply_cluster_ssh_policy`).
2. **Add `GSSAPIAuthentication=no`** to `cluster_ssh_options()` (`ssh_policy.py:31–50` — it is absent today; this is failure #3's 30–90 s hang). Also add `PreferredAuthentications=publickey` so a misconfigured peer fails in ~1 s with a key error instead of walking the auth-method list.
3. **Add `IdentitiesOnly=yes` paired with the managed `IdentityFile`.** Today's comment says OpenSSH "can still fall back to an operator's existing keys" (`ssh_policy.py:43–46`) — in a corporate environment that fallback is how you end up authenticating as the wrong principal or triggering a smart-card prompt. Cluster SSH should present exactly one identity: the one pairing installed. (Escape hatch: a per-peer `allow_agent_keys` flag for users who deliberately pair with pre-existing keys.)

### A.5 Preflight checks — exact order, both directions

Ordered so each check's failure has a unique cause given everything before it passed. Timeout budget per check in parentheses (feasible only because GSSAPI is off).

| # | Check | Method | Proves |
|---|-------|--------|--------|
| P1 | Control-plane reachability | HTTPS GET peer status endpoint | Network path + oMLX running |
| P2 | Identity freshness | Compare live `ssh_user`/`hostname` from status against stored record | Record isn't stale (account renamed, hostname changed) |
| P3 | SSH account check (3 s) | `ssh user@host -- id -un`, compare output to `ssh_user` | Auth *and* that we landed in the right account — catches RecordName aliases and NFS-weirdness where auth succeeds as someone else |
| P4 | Reverse SSH (3 s) | Ask peer (over HTTP control channel) to run its own P3 back at us | Bidirectionality — coordinator-only success is failure #1's reverse direction |
| P5 | Non-interactive shell sanity (3 s) | `ssh user@host -- echo ok` byte-exact | No MOTD/banner/corporate wrapper corrupting stdout (cf. the banner handling in `autoconfigure.py:1258`) |
| P6 | Remote Python & venv | `ssh user@host -- ~/omlx-distributed/.venv/bin/python -c 'import omlx'` (path per `staging.py:36–37`, `~` expanded by the *peer's* shell) | Worker runtime exists in the peer's own home |
| P7 | Staging root, `~`-relative | Peer-side probe of `~/.omlx/models` writability + free space | Failure #2 cannot recur: the path is resolved in the peer's home (`staging.py:499–502`), and disk headroom is checked *before* a 17 GB copy, not after |
| P8 | Clock skew | Compare timestamps from P1 | TLS/token validity; ordering of merged logs (B.3) |

P3/P4 are the load-bearing novelties; everything after runs only if they pass, so "Permission denied" can never again masquerade as a capacity or model problem.

### A.6 Data model — `PairedPeer`

Extend `EnrolledNode` (`enrollment.py:63`) rather than invent a parallel store.

**Stored (survives reboot; written at pairing, updated on verified change):**

```
PairedPeer {
  node_id: stable UUID (from enrollment)
  display_name: "Aphoenix's MacBook Pro"
  ssh_user: "aphoenix"                  # from identity envelope, re-verified by P2/P3
  home_dir: "/Users/aphoenix"           # display + sanity only; contract paths stay ~-relative
  hostnames: ["aphoenix-mbp.local"]     # known_hosts identity, per ssh_policy CheckHostIP=no
  ssh_host_key_fp: "SHA256:…"           # pinned at pairing; mismatch = hard stop, not accept-new
  cluster_pubkey_installed: true        # our omlx_cluster.pub is in their authorized_keys
  admin_port: 9010                      # the peer's REAL admin/health port (kills the 9000 assumption)
  protocol_version, omlx_version (at pairing)
  paired_at, last_verified_at, last_verify_result
  allow_agent_keys: false               # A.4 escape hatch
  role: workstation | dedicated         # feeds reserve, surfaced in B.5
}
```

**Probed live (never stored as truth, only cached with timestamp):** addresses (LAN + fabric), transport state, RDMA devices/addresses, `admission_ceiling_bytes`, staged-model inventory, disk free, current omlx version. Anything in this list rendered in UI carries its age ("as of 8 s ago") and goes stale-gray after 30 s — the stale 169.254 display of failure #5 becomes visually impossible to mistake for live data.

### A.7 Failure copy (each with remedy)

- **P3 fails, exit 255 "Permission denied":**
  > **Wrong account or missing key.** This Mac tried to sign in to "Aphoenix's MacBook Pro" as **aphoenix** but was refused. Usually this means the pairing key isn't installed anymore. → [Re-install pairing key] (re-runs Screen 3) · [Show details]
- **P3 succeeds but `id -un` ≠ `ssh_user`:**
  > **Signed in as the wrong user.** We connected as **alyta.phoenix** but expected **aphoenix**. Something on the other Mac remaps this account name (often a corporate directory alias). oMLX will pin the exact account. → [Use `aphoenix` explicitly and re-verify]
- **P4 fails while P3 passed:**
  > **One-way connection.** This Mac can reach the MacBook, but the MacBook can't reach back. A firewall or VPN on this Mac may be blocking incoming connections. → [Run Fabric Doctor] (section C) · Copy-paste check: `ssh alytaphoenix@mini.local -- id -un` (run on the MacBook)
- **Host-key fingerprint changed:**
  > **This Mac's identity changed.** The MacBook is presenting a different SSH key than when you paired (reinstall? different machine on the same name?). oMLX won't connect until you confirm. → [Compare fingerprints] · [Re-pair]
- **P7 low disk:**
  > **Not enough space to stage the model.** "Aphoenix's MacBook Pro" has 11 GB free but the model needs 17.2 GB in `~/.omlx/models`. → [Choose a smaller quantization] · [Manage models on that Mac]
- **sshd off:**
  > **Remote Login is off on the MacBook.** Turn it on in System Settings → General → Sharing → Remote Login, or run: `sudo systemsetup -setremotelogin on` → [Open System Settings on that Mac] (sends a control-plane nudge that surfaces the deep link there)

### A.8 Build order

1. `user@host` everywhere + `GSSAPIAuthentication=no` + `IdentitiesOnly=yes` in `ssh_policy.py` (small, kills failures 1-partially and 3 outright).
2. Identity envelope over the join-key channel; `PairedPeer` fields on `EnrolledNode`.
3. Preflight ladder P1–P8 with bidirectional P3/P4 and the failure copy above.
4. The five-screen flow (can initially wrap the ladder in the existing dashboard).
5. Host-key pinning at pairing (tightens `accept-new` to pin-after-first-pair).

---

## B. Better output / diagnostics

### B.1 Goals

1. **No silent failure.** Every abnormal end state of every operation produces a durable, user-visible record.
2. **No misleading failure.** Errors report the *constraint that actually bound* (the 32 GB workstation reserve), not a downstream symptom ("model does not fit").
3. **"Ready" is a proven predicate**, and every not-ready state has a plain-language reason and a remedy.

### B.2 The readiness model

`TransportState` currently tops out at `PEER_LINKED_CONFIG_PENDING` — the docstring admits "This slice can report only through `peer_linked_config_pending`" (`models.py:19`). That ceiling *is* failure #5: the most informative thing the system could say was "config pending," forever. Extend the ladder so every rung between "cable in" and "collective proven" is a named state:

```
Fabric ladder (per link):
  unavailable → disabled → enabled_no_peer → peer_linked
  → addressed          (both ends hold non-link-local, same-subnet IPs on the TB interface)
  → routed             (route to peer's fabric IP resolves to the TB interface — route.uses_rdma_interface, models.py:98)
  → reachable          (bidirectional TCP connect bound to the fabric IPs succeeded)
  → fabric_verified    (RDMA r/w probe or JACCL handshake passed — today's flag, models.py:141, but now earned, not defaulted false at launch.py:1019)
  → collective_ok      (a real 2-rank allreduce of a test tensor completed; bandwidth recorded)
```

Every state carries three strings rendered verbatim in `omlx cluster status` and the dashboard: `state`, `reason` ("The link has a self-assigned 169.254 address — macOS never finished configuring it"), `remedy` ("Run Fabric Doctor → Re-address link"). The regex-rule table in `guidance.explain()` (`guidance.py:370`) is the migration seed: each `_RULES` entry becomes a structured `(state, reason, remedy)` triple keyed by state/code instead of pattern-matched from message text.

**Cluster-level readiness** is the conjunction shown in B.5's panel: `ready = all(peer.ssh_ok, fabric ≥ reachable, model_staged, budget_fits_live, strategy_compatible)`.

### B.3 Error propagation: incidents, not banners

Failure #8's root cause was **client-owned ephemeral state**: a 10 s poll rebuilt Alpine state and wiped the plan and error banner; the 2,615-line `_cluster.html` partial owns far too much truth. Failure #7's root cause was a **silent fallback**. One architecture kills both:

- **Incidents are server-side records.** Any rank, staging job, probe, or worker failure creates `Incident {id, ts, severity, source: rank|peer|coordinator|gui, state_code, message, guidance_id, job_id?, bundle_ref}` in a ring buffer persisted by the coordinator. Workers report incidents upstream over the control channel; a rank that dies unreachable gets an incident synthesized by liveness (`liveness.py`) on its behalf — *the coordinator narrates deaths the deceased can't report*.
- **Polls merge monotonically.** The GUI poll may add/refresh; it may **never remove** an incident or reset a job's state. Incidents leave the screen only via explicit user dismissal or job supersession ("Start attempt #4 replaces #3" — and #3 remains in history).
- **Start Cluster is a job, not a button-state.** Clicking it POSTs; the server returns `job_id`; the button renders the *server's* job record (`preparing → staging(rank1: 62%) → launching → ready | failed(incident_id)`). If the browser reloads, the job is still there. A dead-button is impossible because the button has no local state to kill. `staging_jobs` is the in-repo precedent (surfaced today at `_cluster.html:925–928` but not wired into a durable job model).
- **Silent fallback is banned.** The pattern at `launch.py:1153` — `try: urlopen('http://127.0.0.1:9000/health') … except Exception: pass` then a slow fallback — becomes: probe the *stored* `PairedPeer.admin_port`, and on failure emit `Incident(WARN, "fast ceiling probe unreachable on port 9010; using slow local computation (adds ~Ns)")`. Fallbacks still happen; they just confess.
- **Retry loops narrate.** The JACCL error-60 exponential backoff (failure #6) emits one incident per retry tier: "Collective connect to 172.16.99.2:… timed out (attempt 2/6). If this persists, a firewall or VPN is likely blocking the fabric — run Fabric Doctor." Silence between retries is the bug.
- **Stale-asset handshake.** Every API response carries an `X-Omlx-Asset-Version`; the page compares it to its own build stamp and shows a non-dismissable "Dashboard updated — reload" bar on mismatch. Cached-JS ghosts (failure #8) get named instead of haunting.

### B.4 Dashboard surfaces

Three surfaces, strictly layered:

1. **Status strip** (always visible): one line per node — `mini · coordinator · ready` / `aphoenix-mbp · worker · fabric: reachable ✻ verifying`. Colors follow state, text carries the state name; never color-only.
2. **Incident feed** (badge on the strip, panel on click): reverse-chron incidents with severity, source, age, and per-row **[Details]** → the bundle slice (B.6). The feed is the anti-silent-failure guarantee: if *anything* went wrong since the user last looked, the badge says so.
3. **"Why can't I start?" panel** (B.5): rendered whenever Start is not green, in place of a disabled button. A disabled control without an adjacent explanation is a design defect, full stop.

### B.5 "Why can't I start?" — precondition rows

Each row: name, pass/fail/warn, the *live* evidence, and a fix affordance.

| Row | Evidence shown | Fix affordance |
|-----|---------------|----------------|
| SSH to each peer | "aphoenix@aphoenix-mbp.local ✓ verified 40 s ago (both directions)" | [Re-verify] → A.5 ladder |
| Fabric | Current ladder state + addresses: "reachable · 172.16.99.1 ⇄ .2 · TB5" | [Run Fabric Doctor] |
| Model staged | Per-node: "✓ mini (local) · ⟳ MacBook 8.4/17.2 GB (12 MB/s over Wi-Fi — Thunderbolt idle, see Doctor)" | [Pause] [Details] |
| Memory budget | **The breakdown, always live:** "MacBook ceiling 15.4 GB = 48 GB physical − 32 GB *Workstation reserve* − 0.6 GB in use · model needs 18.6 GB ✗" | Reserve is a link → role editor (`node_role.py:17–38`) |
| Parallelism | "Tensor ✓ · Pipeline — not supported by this model (vision-language models require tensor)" | mode radio disabled *with that inline text* |

The memory row is failure #9's tombstone. The invariant is already written in the code — "Cluster planning must use this same number or a plan can pass in the GUI and be refused by the rank moments later" (`models.py:122–126`) — the row simply renders `admission_ceiling_bytes` live *with its arithmetic*, so a reserve can never impersonate a capacity limit. Copy for the failing case:

> **The model fits in this Mac's memory, but not in its current budget.** The MacBook has 48 GB, but its role is *Workstation*, which reserves 32 GB for your apps — leaving 15.4 GB. The model needs 18.6 GB. → [Change role to Dedicated for this run] · [Pick a smaller quantization]

The parallelism row is failure #10's: `pipeline_assignment_is_honored()` (`pipeline_compat.py:30`) — and a sibling `supports_tensor_parallel` capability — are queried at *plan build time*; a mode the model can't run is never offered, and the 400-at-runtime path becomes unreachable from the GUI.

### B.6 Support-bundle detail, one click away

- Every incident row's **[Details]** opens the structured slice: the raw status JSON, the relevant log window (±30 s around the incident), the SSH/JACCL command line that failed, exit code, stderr.
- A global **"Save support bundle"** in the incident panel writes the existing JSON bundle plus the incident feed and readiness snapshot, then reveals it in Finder. Copy: *"Bundle saved. It contains cluster status, the last 200 incidents, and logs from every node — no prompts or model content."*
- `omlx cluster status --explain` prints the B.5 panel in text form, so CLI and GUI can never tell different stories (failure #9's "fit fine via CLI" divergence).

### B.7 Build order

1. Server-owned incidents + monotonic-merge poll rule (removes the entire silent-failure class; mostly backend + a disciplined rewrite of poll handling in `_cluster.html`).
2. Start-as-job with persisted state.
3. Readiness ladder states past `peer_linked_config_pending` + reason/remedy strings (`models.py`, `transport.py`, `guidance.py` restructure).
4. "Why can't I start?" panel (composes 1–3).
5. Live-budget breakdown row and plan-time parallelism gating.
6. Asset-version handshake; bundle-slice Details links.

---

## C. Fabric Doctor — diagnose-and-fix for the Thunderbolt/RDMA link

### C.1 Goals

1. Detect every link pathology from the incident *before* the user attempts Start, in one guided flow.
2. Every finding: check → plain-language diagnosis → concrete fix (auto-applied with admin consent where safe, exact steps otherwise).
3. Good addressing survives reboots and oMLX's own re-runs.

Entry points: the [Run Fabric Doctor] affordances in A/B, automatic invocation when the fabric ladder regresses, and `omlx fabric doctor` in the CLI. The Doctor runs the ladder checks of B.2 in order and stops at the first red rung — later checks would only report consequences of the first failure.

### C.2 Check 1 — link and address sanity

- **Check:** enumerate TB ports (`ThunderboltPort`, `models.py:74`), active RDMA port, interface IPs; classify against the non-fabric list (loopback/link-local, `transport.py:1031–1035`).
- **Findings & copy:**
  - No peer on any TB port → *"No Thunderbolt connection detected. Check the cable is seated on both Macs (a TB5 cable, not USB-C-only). Port status: mini port 1 'connected', MacBook — none."*
  - 169.254.x on the fabric interface (failure #5's stale address) → *"The link has a self-assigned address, which means it was never configured (or the configuration was lost — see 'Make it stick', C.6). oMLX can fix this."* → [Re-address link] (auto, with admin consent via the Aqua-session privileged runner, `transport.py:663`).
  - Interface renumbering (the en6→en4 case documented at `transport.py:1026`) → *"The Thunderbolt interface changed names since the address was set (en6 → en4). Re-addressing the current interface."*

### C.3 Check 2 — subnet selection with collision detection (the WARP killer)

Today `_setup` reuses whatever subnet exists or defaults to `10.0.1.1/24` (`transport.py:860`). 10.0.1.0/24 sits inside 10.0.0.0/8, which full-tunnel VPNs routinely claim — that default is exactly what WARP swallowed (failure #4). Replace it with **collision-checked selection**:

- **Check:** on *both* nodes, build the set of hostile prefixes: every route currently pointing at any `utun*` interface, every subnet claimed by other interfaces, plus (optional enrichment, C.5) any split-tunnel exclusion list readable from WARP. A candidate subnet is acceptable only if, on both nodes, it collides with nothing — or better, sits *inside* a known VPN exclusion (the 172.16.99.x trick from the incident, systematized).
- **Selection order:** (1) keep an existing manually-configured fabric subnet if it passes the collision check — user/Doctor choices are respected; (2) a subnet inside a detected VPN exclusion; (3) an RFC1918 /29 chosen from ranges no interface or utun route claims; (4) never bare `10.0.1.0/24` again.
- **Copy on collision:** *"The link's addresses (10.0.1.1–2) are inside a range your VPN captures (Cloudflare WARP routes 10.0.0.0/8 through utun4 on the MacBook). Traffic between the Macs is being sent to the VPN instead of the cable. oMLX will move the link to 172.16.99.1–2, which the VPN leaves alone."* → [Move link addresses] (auto on both nodes, admin consent) → re-run ladder.

### C.4 Check 3 — routes, reachability, JACCL, RDMA

- **Route pinning:** `route get <peer_fabric_ip>` must resolve to the TB interface (`RouteCapability.uses_rdma_interface`, `models.py:98`). If a utun wins: same diagnosis/remedy as C.3 (the address is in a captured range) or, when the range is clean but a `catch-all` VPN policy still wins, the VPN-exclusion guidance of C.5.
- **Bidirectional bound connect:** TCP connect *bound to the fabric source IP* in both directions. This is the check that distinguishes "cable works" from "VPN firewall swallows inbound" — in the incident, LAN ping/SSH passed while peer→coord on the fabric was refused. Copy for one-way: *"The MacBook cannot accept connections on the Thunderbolt link (its VPN's firewall applies to all interfaces). Ask IT to exclude 172.16.99.0/29, or in WARP: Settings → … [exact copy-paste per detected client]."*
- **JACCL probe (the error-60 case):** run a **minimal 2-rank handshake outside the real collective** — same ports, same RDMA matrix contract that deployment validates (`deployment.py:323–333`) — with a 10 s budget. Errno-mapped diagnoses: `ETIMEDOUT (60)` → *"connection is being silently dropped — firewall/VPN on the fabric path"* (→ C.3/C.5 remedies); `ECONNREFUSED` → *"the peer's worker isn't listening yet — a launch-order problem, not a network problem; retrying is correct"*; success → record bandwidth, mark `collective_ok`. This converts failure #6 from an opaque retry loop into a named, pre-checked condition.
- **RDMA device vs. address staleness:** devices present (`RDMACapability.devices`) but addresses stale/absent → *"RDMA is enabled but the link addresses were lost (this happens after reboot — see below)."*
- **Dead-port probe (failure #7):** the Doctor calls the peer's health endpoint at `PairedPeer.admin_port`; unreachable → WARN incident with *"Fast memory probe target isn't answering on port 9010 — Start will still work but planning will be slower. Usually the peer's server isn't running."* The hardcoded `127.0.0.1:9000` in `launch.py:1153` is retired by the stored-port change in A.6.

### C.5 Detecting a hungry VPN **before** the user hits it

Two layers, with the empirical probe as the authority (a config read can be MDM-locked or lie):

1. **Heuristic pre-warning at pairing time (Screen 4):** utun interfaces present + a default or /1+/1 route through them + known client signatures (WARP: `warp-cli settings` / Cloudflare plist; also Tailscale, GlobalProtect, AnyConnect) → banner before any fabric setup: *"The MacBook is on a corporate VPN that captures all traffic. oMLX will pick link addresses the VPN ignores and verify the link end-to-end before use."* Enrichment: a readable split-tunnel exclusion list feeds C.3's selection.
2. **Empirical authority:** the C.4 bidirectional bound connect + `route get` check. Only these promote the ladder past `reachable`. No configuration read ever substitutes for them; conversely, a locked-down config can't block a link the probes prove working.

### C.6 Making it stick (the ephemeral-addressing fix)

The incident's 172.16.99.x fix reverted on reboot because it was bare `ifconfig`. Durability design:

- **Persist via macOS network services:** create/adopt a network service on the TB interface and apply `networksetup -setmanual <service> 172.16.99.1 255.255.255.248` (admin consent once). This survives reboots and interface re-enumeration by service identity rather than BSD name — directly addressing the en6→en4 note at `transport.py:1026`.
- **Record intent:** the chosen subnet is written into the cluster config (both nodes) as `fabric_subnet`, with provenance (`chosen_by: doctor|user`, `reason: vpn_exclusion`, date).
- **Link-setup respects intent:** `_setup`/`assess_link` (`transport.py:840–900`) treat a recorded `fabric_subnet` as authoritative — re-running link setup re-applies it, never "helpfully" re-addresses to a default. Re-addressing happens only through the Doctor, which re-runs the collision check first.
- **Drift watchdog:** on wake/network-change events, compare live addressing to intent; on drift, auto-restore silently if the collision check still passes, else raise an incident: *"The link's saved addresses now collide with a new VPN range — Fabric Doctor needs to pick new ones."*

### C.7 Doctor output format

Same row grammar as B.5 (check · state · evidence · fix), ending in either **"Fabric verified — 68 Gb/s measured between mini and MacBook"** or the first red row with its remedy button. Every run appends to the incident feed and is included in support bundles, so "what did the Doctor say last Tuesday" is answerable.

### C.8 Build order

1. Collision-checked subnet selection replacing the `10.0.1.1/24` default (`transport.py:860`) + the bidirectional bound-connect check. (Prevents the worst incident outright.)
2. Durable addressing via `networksetup -setmanual` + recorded `fabric_subnet` intent + setup-respects-intent.
3. Doctor UI running the ladder with the C.2–C.4 findings/copy; errno-mapped JACCL probe.
4. VPN heuristic pre-warning + exclusion-list enrichment.
5. Drift watchdog; dead-port retirement (lands with A.6's `admin_port`).

---

## Appendix: cross-cutting principles (one line each)

1. Identity before SSH; SSH is never asked to guess.
2. `~`-relative is the only legal cross-node path grammar.
3. Every state the user can see carries a reason and a remedy — states without copy don't ship.
4. Errors are server-owned, immortal until dismissed; polls merge, never clear.
5. Fallbacks confess; retries narrate.
6. Live values render with their age and their arithmetic.
7. Capabilities gate the UI at plan time; runtime 4xx from a GUI-offered option is a design bug.
8. Empirical probes outrank configuration reads.
9. Anything the Doctor fixes, it also makes durable and re-checks after drift.
