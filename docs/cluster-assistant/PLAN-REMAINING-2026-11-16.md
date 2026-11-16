# Plan — what is actually left, 2026-11-16

Status: **living checklist.** Everything product-side is built and tested; the
remaining work is one live acceptance run, one PR fold, one push and a cleanup.
This document is the single place that says what to do next, in order, with a
done-when per line. It does not re-open a frozen decision (`AGENTS.md` §1,
`PROJECT-BRIEF.md` Appendix A) and it does not replace the product documents —
it points into them.

Read with: `ROADMAP-ACCEPTANCE-2026-11-16.md` (gate + decisions),
`PR-READY-2026-11-16.md` (fold + DMG), `CLUSTER-TEST-RUNBOOK.md` (the Macs),
`READY-FOR-CLUSTER-TEST.md` §9 (the current artefact).

---

## 0. Measured baseline (2026-11-16, re-verified in this session)

| Fact | Value |
|---|---|
| Branch / HEAD | `rebase/upstream-main` @ `691aaa94`, working tree clean |
| Upstream | `upstream/main` `624cff0e` is an ancestor |
| Backend suite | `tests/cluster_assistant` **658 passed** |
| UI suites (focused) | wizard + `test_cluster_ui_integration_ui` + dashboard **289 passed** |
| Lint | `ruff check omlx/cluster/assistant/` **clean** (8 nitpicks fixed 2026-11-16) |
| Acceptance DMG | `build/oMLX-0.8.0rc1.dmg` `9400d290…`, app binary `760a1695…`, `0.8.0rc1`, notarized + stapled |
| Installed on `.11` | app binary `09f51c2d…` → **one build behind** (no U1–U6 UI) |
| Macs | `.11` `.12` `.13` `.14` reachable, **no oMLX running** anywhere (app must be opened by a person) |
| Cluster state | **not wiped** on `.11`: devices, deployments, measurements, `runtime/prompt-cache-ssd` |
| PR series | `rebase/upstream-main` is **not** an ancestor of `feat/cluster-assistant-run` → fold **not** done (82 commits / 250 files backlog) |

**Consequence:** the acceptance run has not started. Nothing else is missing in
code.

---

## 1. Definition of done for this plan

1. **A1–A7 hold** on a wiped four-Mac cluster with one identical build, recorded
   in `PHASE-2B-RUN.md` §31 with measurement/plan/operation IDs.
2. The branch is **folded** into `feat/cluster-assistant-run` (+ `-run-local`
   rebase), the suites are green on the folded branches, `make web` is a no-op.
3. `PR-BODIES-2026-11-15.md` carries the new SHAs/counts and the series is
   **pushed** in the recorded order (F1, F2, F4, F5 → A1 → A2 → A3, F6 optional).
4. The Macs and this checkout are **clean** (processes, ports, state, Launchpad,
   DerivedData, DMG staging), and the docs no longer contradict the code.

---

## 2. Phase 0 — make the tree presentable (no cluster)

- [x] `ruff check --fix omlx/cluster/assistant/` — 8 fixable findings closed
      (trailing newlines in `reclaim.py`, `redeploy_tuning.py`, `search.py`,
      `tuning.py`, `tuning_store.py`, `tuning_wiring.py`; import order in
      `search_wiring.py`).
- [x] `CHAPTER-10.md` — status line, §6 (helper is a capability, Xcode green,
      79 MB notarized DMG) and §7 (exit criteria) reconciled with the code.
- [x] `docs/cluster-assistant/README.md` — current milestone, read order (adds
      the 2026-11-16 documents), reference pin (`624cff0e`), TP = R2.
- [x] `PROJECT-BRIEF.md` — wired-limit action is `raise_wired_limit` (R5), the
      Qwen3.8 "not pipeline-capable" statement withdrawn (#5033), Appendix A
      pin + "first milestone" updated, pointer to this plan.
- [x] `ROADMAP-TO-CHAPTER-10.md` §11 — dated correction for the Qwen family and
      the remaining gate item.
- [x] `READY-FOR-CLUSTER-TEST.md` §9 — the current artefact with both hashes and
      the measured state check (installed build behind, state not wiped).
- [x] This file, and its entry in the docs index.

**Done when:** suites green, `ruff` clean, `node --check cluster_v2.js` clean,
`make web` a no-op, and no document states something the code contradicts.

---

## 2.1 PyCharm inspection — where it stands (2026-11-16)

The export (67 `<Inspection>.xml`, 09:31) contained **10 593 findings in the
`localDev` scope** across 190 files. The export and its 104 GB `index.html`
report now live outside the tree in `~/cluster-inspection-2026-11-16/`.

| Class | Export | Now | State |
|---|---|---|---|
| `IncorrectFormatting` — block indentation | 8 188 in `cluster_v2.js` | **0** | **Fixed by config, not by reformatting**: `.editorconfig` declared 2-space indentation for `*.js`/`*.html`/`*.swift` while every file here is 4-space. JS/HTML/Swift are 4 again, data/CSS/markdown stay 2, `max_line_length = 120`. |
| `IncorrectFormatting` — continuation/attribute alignment | 249 `cluster_v2.js`, 2 704 template | 231 + 2 692 open | The files align continued expressions/attributes (4-space continuation in JS, attribute alignment in HTML); IntelliJ's default is 8. The `ij_html_*`/`ij_js_*` `.editorconfig` keys are ignored by this IDE build (measured), and one `Reformat Code` would rewrite 3 000 lines — deliberately **not** done. One-time fix if wanted: set *continuation indent 4* (JavaScript) and *align attributes* (HTML) in the IDE code style. |
| `LongLine` (> 120) | 596 | **0 in the files we own** (JS, the three D1/D6 Swift screens and the Swift tests); 385 left in `_cluster_v2.html` | `cluster_v2.js` 28→0, `Uninstall.swift`/`AdvancedScreen.swift`/`PerformanceScreen.swift` 24→0 (Swift string values verified byte-identical), Swift tests 6→0, wizard test 8→0. The template's 385 are single Tailwind class strings or long Alpine expressions: splitting them would need the attribute value broken (unsafe) or read worse. |
| Real defects | ~40 | **0** | `reclaim._inventories`/`routes._calibration_entry` typed `Sequence[Any]`, coordinator replies typed `Any`, five dead initialisers removed, four redundant parenthesis pairs, `diagnostics` credential regex named, `recommendation` layout guard, duplicate `pairingStepOne` removed, 21 pre-increments, 4 regexes (named/non-capturing), **7 inputs got an `aria-label`** reusing their own placeholder key (no new i18n strings), 57 long tags wrapped one attribute per line. |
| `PyUnresolvedReferences` **errors** | 68 | **0** | All were `monkeypatch.setattr(module, "re-exported-name", …)` in `tests/cluster_assistant` (PyCharm cannot follow a name that `routes`/`start` re-export with `from … import`). All 157 module-target string patches now use `monkeypatch.setitem(module.__dict__, "name", value)` — identical restore semantics, resolvable for the IDE, 658 tests green. The remaining `JSHint` error is "Node interpreter not configured" (IDE setting, not code). |
| `JSUnresolvedReference` (71) + `JSUnusedGlobalSymbols` (174) | 245 | 70 + (template) | Property names of JSON payloads (`preset_key`, `placement_signature`, `single_node`, …) and component methods called from Alpine templates. Not fixable without typing the payloads; a real typo fails at runtime and the UI tests cover the flows. |
| `JSIgnoredPromiseFromCall` (15) + `ES6MissingAwait` (4) | 19 | 19 | Deliberate fire-and-forget in the sync Alpine lifecycle (`init()`, event handlers). `void` only trades one weak warning for another (`VoidExpressionJS`); awaiting would change load ordering. Documented, not "fixed". |
| `DuplicatedCode` (116) | 116 | 116 | Structural: JSON-store boilerplate (`state.py`/`operations.py`), two sibling dataclasses (`models.py`), repeated solver blocks (`policy.py`), repeated test fixtures. A refactor is a behaviour-risk trade for a weak warning — refused. |
| Narrowing false positives (`PyUnresolvedReferences` weak, `PyNoneFunctionAssignment`, `PyAttributeOutsideInit`) | 260 | ~240 | `x | None` followed by a guard PyCharm does not follow (mostly tests). AGENTS §8 already says not to "fix" these. |
| `PySimplifyBooleanCheck` (2), `HttpUrlsUsage` (24) | 26 | 26 | `entry.get("distributed") is True` is the strict JS-boolean check — the simplification would change semantics. `http://host:port` is the LAN endpoint; there is no https. |
| `XmlUnboundNsPrefix` (3) | 3 | 3 | Alpine `x-transition:enter` read as an XML namespace — false positive of the XML inspection on an Alpine template. |
| `RegExpAnonymousGroup`/`RegExpRedundantEscape` in the wizard tests (27) | 27 | 27 | Those Python patterns **match literal JS source text** (`\d\{1,3\}` must keep its escapes to match the JS regex in `cluster_v2.js`); "fixing" them breaks the assertion. Test-only. |
| Upstream `omlx/cluster/*.py` (launch, planner, discovery, inference_worker, routes, telemetry, settings) | ~30 non-style | untouched | Dead locals (`launch.failure_reason`, `planner.names`, `routes.model_id`), a dead `raw.encode()` in `discovery`, missing `except ImportError` fallbacks, three `LRUPromptCache`/`MLX group` typing misses. **Not touched on purpose**: these files are shared with the open PR series; changing them costs rebase conflicts for cosmetic wins. Worth one small dedicated patch later. |

**Result for our own surface:** errors **0**; every warning with a real cause
fixed; the rest is enumerated above with the reason it stays. Verification:
`tests/cluster_assistant` 658 passed, `apps/omlx-web/tests` 578 passed
(3 deselected), `apps/omlx-mac/Tests/python` 27 passed, `xcodebuild test`
**441 passed / 5 skipped / 0 failed**, `ruff` clean on the touched files,
`node --check` clean, `make web` a no-op.

---

## 3. Phase 1 — rollout and the acceptance gate (needs people at the Macs)

> **Rollout bewusst zurückgestellt (2026-11-16):** erst die PyCharm-Inspection
> abwarten und die echten Findings im Code bereinigen, dann das DMG ausrollen.
> Die Macs sind inzwischen vollständig gewischt (§6), also ist der Rollout der
> nächste Schritt, sobald der Code sauber ist.

Precondition: a person at each Mac (Local Network prompt, helper dialog, runtime
installer). Do not drive these over SSH.

- [x] **1.1 Wipe** — `scripts/cleanup-cluster-macs.sh --all --with-helper`
      (backs up to `~/.omlx/backup-cluster-<date>/`, keeps models +
      `cluster/identity.json`, kills processes, frees 8099). Done 2026-11-16 on
      all four Macs; the helper still needs its admin dialog (see §6).
- [ ] **1.2 Roll out** — copy `build/oMLX-0.8.0rc1.dmg` to all four Macs,
      install, open the app **interactively**, let Local Network, the runtime
      installer and the helper dialog through.
- [ ] **1.3 Identical-build gate** — `GET /api/cluster/node_id` → same
      `version`; and `shasum -a 256 /Applications/oMLX.app/Contents/MacOS/oMLX`
      → `760a1695…` on **all four**. A mixed build is not an acceptance run.
- [ ] **1.4 A1 Manual** — devices → pair → measure → model → plan → apply →
      start → `ready` → one real completion → stop, through the manual flow.
- [ ] **1.5 A2 Assistant** — the same model through the assistant card; endpoint
      visible only at `ready`; the two runs agree on plan split and coordinator.
- [ ] **1.6 A3 Correction** — trigger a real failure (pin a Mac that cannot hold
      the tail, or a too-low tier), see it **inline at the failing step**, fix it
      in place, retry successfully, no page reload.
- [ ] **1.7 A4 Back/`stale`** — after a green plan change model or context,
      see downstream steps `stale`, finish the flow.
- [ ] **1.8 A5 Capability** — install/reinstall the helper from Welcome **and**
      from Performance, independent of the wired-limit warning; current/target
      MB visible.
- [ ] **1.9 A6 Tear down** — uninstall from the app: app, helper, runtime,
      settings, cluster state removed; models + `cluster/identity.json` kept;
      0 LaunchServices records afterwards.
- [ ] **1.10 A7 No surprises** — no error text exists only in the advanced
      preview; no warning rendered as an error.
- [ ] **1.11 Screenshots** — the eight `data-assistant-screen` states + Manual
      mode (R8 + U6 done-when).
- [ ] **1.12 Record** — new `PHASE-2B-RUN.md` §31: build hashes, measurement /
      plan / operation / deployment IDs, rank order, endpoint, rates, stop,
      failures found and how they were fixed.

**Done when:** A1–A7 hold and §31 exists. Anything found is fixed inside the
existing assistant surface (no second planner, wizard or control plane), the DMG
is rebuilt through `scripts/build-release-dmg.sh`, and 1.2–1.12 are re-run.

---

## 4. Phase 2 — deferred live matrix (same cluster window, in this order)

Each item gets its own `PHASE-2B-RUN.md` section. Evidence from before the
2026-11-14 rebase does not count as proof on this tree.

- [ ] **2.1 MoE routed-expert offload runtime** (#5024): a DeepSeek-V4.1 family
      checkpoint planned with expert offload must stream experts on the pipeline
      ranks, not just be priced.
- [ ] **2.2 Qwen4-Exp/Qwen3.8 PP across Macs** (#5033): plan and start on two
      Macs (single-Mac `run-local` is already proven).
- [ ] **2.3 Tuning on a real cluster**: `/tune`, `/tune-redeploy`, and the preset
      replay on the next Start (`tuning_preset_applied`).
- [ ] **2.4 Copy / free-space / delete**: model only on a peer
      (`layout_read_on_peer`), apply stages from source, `insufficient_disk` →
      `free_disk_space` → delete route; the last copy and the planned model are
      refused.
- [ ] **2.5 Topology**: pull one cable on a 3/4-Mac line → `topology_incomplete`
      names the pair; "Run with 2 Macs" only after explicit consent, and the
      consent salts the plan signature.
- [ ] **2.6 3-Mac line start** (minimal subset chooses three when two do not
      fit). A 4-Mac start stays Phase-3 fabric work.
- [ ] **2.7 `raise_wired_limit` live** — needs a Mac still at the macOS default
      limit; all four rollout Macs are raised, so either lower one limit
      deliberately or record the item as still unproven.
- [ ] **2.8 R16/R17 confirm or drop** — `auto-login` sets no session cookie; the
      card can report `no_model`.
- [ ] **2.9 R24** — the origin process disappeared during a failed start, no
      traceback. Capture the next occurrence with `log show` + `sample`, or
      retire it as not reproducible.

---

## 5. Phase 3 — fold into the PR series and push

Only after Phase 1 (the roadmap order: acceptance first, PRs second).

- [ ] **3.1 Fold once**
      ```sh
      git switch feat/cluster-assistant-run
      git merge --no-ff rebase/upstream-main
      # conflicts: AGENTS.md, cluster_v2.js, template, models.py, routes.py
      # → resolve toward this branch (main is newer)
      git switch feat/cluster-assistant-run-local
      git rebase feat/cluster-assistant-run
      ```
- [ ] **3.2 Re-run gates on the folded branches**: `uv run pytest tests -q`,
      `uv run pytest apps/omlx-web/tests -q`,
      `uv run pytest apps/omlx-mac/Tests/python -q`,
      `uv run pytest tests/cluster_assistant -q`,
      `xcodebuild test -scheme oMLX CODE_SIGNING_ALLOWED=NO -testLanguage en`,
      `make web` (second run a no-op), `ruff` on touched files.
- [ ] **3.3 Update `PR-BODIES-2026-11-15.md`** with the new SHAs, counts and the
      action vocabulary (17 actions incl. `raise_wired_limit`) + the rank-0 pin.
- [ ] **3.4 Push in the recorded order**: F1 (`fix/local-only-load`),
      F2 (`fix/test-infra-conftest`), F4 (`fix/macos-theme-locale-xctest`),
      F5 (`fix/cluster-discovery-inventory`) → A1 (`feat/cluster-assistant-plan`)
      → A2 (`feat/cluster-assistant-run`, draft) → A3
      (`feat/cluster-assistant-run-local`, draft); F6
      (`feat/macos-helper-dmg`) optional, after F1–F3 are in.
- [ ] **3.5** `#5024` (`bbbf18d2`) and `#5033` (`25da70de`) stay untouched until
      upstream moves.

**Done when:** the series is pushed, CI is green, and the bodies match the code.

---

## 6. Phase 4 — cleanup and hygiene

**Stand 2026-11-16 (bereits gelaufen):** `scripts/cleanup-cluster-macs.sh --all
--with-helper` über alle vier Macs. Verifiziert auf `.11`, `.12`, `.13`, `.14`:
0 Prozesse, Port 8099 frei, `~/.omlx` = `cluster/identity.json` + der State-Backup
`backup-cluster-2026-11-16/`, 0 App-Kopien, Runtime/Venv entfernt, 0 DerivedData,
0 Trash-Kopien, 0 LaunchServices-Records. Die Modelle (902 GB) und das notarized
DMG unter `build/` sind unverändert.
**Offen bleibt nur der root-eigene Helper** (auf allen vier noch installiert und
geladen) — er braucht an jedem Mac einen Admin-Dialog, und der sollte **vor** dem
Abnahmelauf weg, damit A5 einen echten Erst-Install zeigt. Ebenso bleibt der
Wired-Limit-Wert im Kernel stehen (122 880 / 60 293 / 29 696 / 29 696 MB).

- [ ] **4.1 Cluster state** wiped again on every Mac after the run; verify: no
      oMLX/`omlx-server`/`pytest`/`xcodebuild`/`dns-sd` processes, port 8099
      free, `~/.omlx/cluster/` contains only `identity.json` (plus backups).
- [x] **4.2 Disk**: `apps/omlx-mac/build/`, the Xcode DerivedData products
      (`DerivedData/oMLX-*`, was 2.0 GB), runtime venv, Trash copies, mounted
      DMG volumes and `/tmp` images removed (2026-11-16).
- [x] **4.3 Launchpad**: `lsregister -gc -f -r -domain user -domain local` +
      `killall Dock`; 0 app copies and 0 LS records on all four Macs.
- [ ] **4.4 Helper**: remove `/Library/PrivilegedHelperTools/app.omlx.helper` +
      `/Library/LaunchDaemons/app.omlx.helper.plist` at each Mac (one admin
      dialog each, no passwordless sudo in the cluster).
- [x] **4.4 Keep**: models (`~/Models`, 902 GB), this repository, the notarized
      DMG under `build/` (`9400d290…`), `cluster/identity.json`, state backup
      `~/.omlx/backup-cluster-2026-11-16/`.
- [ ] **4.5 Docs**: `README.md` / `AGENTS.md` §2 baseline (HEAD, branch, DMG
      hash) point at the final state; this plan's boxes are all ticked or the
      leftovers are moved into a dated follow-up.

---

## 7. Open questions / who is needed

| # | Question | Needed for |
|---|---|---|
| 1 | Is a person available at each of the four Macs for the interactive steps (Local Network, helper admin dialog, runtime installer)? | Phase 1.2–1.9 — without it A5/A6 cannot be proven |
| 2 | Confirm fold **after** acceptance (current roadmap order) — or fold first? | Phase 1 vs Phase 3 order |
| 3 | R16/R17: confirm or drop | Phase 2.8 |
| 4 | Do we deliberately lower one Mac's wired limit to prove `raise_wired_limit`? | Phase 2.7 |
| 5 | Remote credential for the push | Phase 3.4 |

---

## 8. Explicitly not open (do not re-litigate)

- R1–R26 in `ROADMAP-TO-CHAPTER-10.md` §3 are closed or recorded; D1–D6 and
  W1–W5 are implemented (`ROADMAP-ACCEPTANCE-2026-11-16.md` §3).
- The assistant UI is one flow with a render-mode; no backend state, route, poll
  or SSE was added and none is planned.
- Out of scope until after this gate: tensor parallelism (R2), >4 Macs, HA,
  multi-tenancy, RDMA/JACCL, telemetry, privileged network setup, a re-plan of a
  running cluster, a second app/wizard/chat.