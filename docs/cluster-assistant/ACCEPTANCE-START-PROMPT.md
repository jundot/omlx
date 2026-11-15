# Start prompt — acceptance package (paste this into a new session)

> Copy everything between the lines into a fresh session in
> `~/Documents/PycharmProjects/omlx-cluster-assistant`.

---

**Rolle:** Du arbeitest an **oMLX Cluster Assistant** (fokussierter oMLX-Fork,
PP-only MVP, 2–4 Apple-Silicon-Macs). Chat auf Deutsch, Repo/Doku/Code auf
Englisch. Du bist der Umsetzer, nicht der Produktentscheider: bei neuen
Produktfragen **einmal fragen**, nicht raten.

**Lies zuerst, in dieser Reihenfolge** (nicht raten, prüfen):
1. `AGENTS.md` — besonders §1 (frozen decisions), §2 (resolved decisions), §7
   (Kommandos), §11 (Environment-Regeln).
2. `docs/cluster-assistant/PACKAGE-2026-11-15-ACCEPTANCE.md` — die Defekte D1,
   D2, D3, D4, D5, D6 mit Ursache, Lösungsskizze, Abnahmekriterien.
3. `docs/cluster-assistant/ROADMAP-ACCEPTANCE-2026-11-16.md` — Gate A1–A7,
   Arbeitspakete W1–W6 in Reihenfolge, PR-Pfad.
4. `docs/cluster-assistant/PHASE-2B-RUN.md` §29 — der letzte echte Lauf
   (ready in ~50 s, 18,76 tok/s, sauberer Stop) und R23 in
   `UPSTREAM-PR-REVIEW-2026-11-15.md`.
5. `docs/cluster-assistant/READY-FOR-CLUSTER-TEST.md` — Rollout, Suiten, DMG.

**Stand, den du vorfindest** (verifiziere ihn, statt ihm zu glauben):
- Branch `rebase/upstream-main`, HEAD `26d89c60`, Arbeitsbaum clean.
- Enthalten und getestet: Tier-Fix (apply schreibt `memory_guard_tier` auf jeden
  gewählten Mac), Inspection-Fixes, **R5 `raise_wired_limit`**, **Rank-0-Pin**
  (`preferred_coordinator` in Election/Solver/Request/`PlanInputs`/Record/Karte),
  Cleanup- und Build-Skripte.
- 658 Assistant-Tests grün (`uv run pytest tests/cluster_assistant -q`);
  A3-Vollsuite 20 229 passed / 0 failed (Stand vor den neuen Commits).
- DMG `build/oMLX-0.8.0rc1.dmg` = `19a446d9…`, App-Binary `01e7898c…`, auf allen
  vier Macs installiert, alle vier **gewischt** (0 Prozesse, `~/.omlx` =
  `cluster/identity.json`, 0 App-Kopien, 0 LaunchServices-Records).
- Die PR-Serie (`feat/cluster-assistant-plan` / `-run` / `-run-local`, Heads
  `da13ab2e` / `6c37cacc` / `a574ba70`) enthält die neueren Commits **noch nicht**
  — Falten ist ein **einmaliger** Schritt am Ende, kein Nebenprodukt.

**Deine Aufgabe in dieser Session**
1. Stelle die **fünf offenen Entscheidungen** aus `ROADMAP-ACCEPTANCE` §7 —
   kurz, als Liste mit deiner Empfehlung, und warte auf Antwort.
2. Baue danach **W1 (D1 Deinstall-Button)** und **W2 (D6 Helper als Fähigkeit)**
   als ein Paket:
   - W1: Settings → Advanced → „Uninstall…", ein Sheet, Keep-Schalter je Bereich
     (App, Helper, Runtime, Settings, Cluster-State, `identity.json`, Modelle),
     Byte-Zusammenfassung vor dem Löschen, Modelle und `identity.json`
     standardmäßig **behalten**, Helper = der eine Admin-Dialog, am Ende
     LaunchServices säubern (`lsregister -u` je entferntem Pfad, dann `-gc`,
     Dock repainten), App zuletzt beenden. XCTests über eine injizierte
     Pfad-/Locator-Schicht — nichts Echtes löschen.
   - W2: **ein** Seam `PrivilegedHelper.status → {installed, version,
     wiredLimitCurrentMB, wiredLimitRequestedMB}`; Helper-Schritt im Welcome;
     dauerhafte Zeile „Helper: installed (vX) / not installed" mit
     Install/Reinstall in Performance; Wired-Limit-Zeile getrennt mit „Apply"
     sobald Ist ≠ Soll; idempotent und versioniert.
3. Dann `scripts/build-release-dmg.sh --install`, dasselbe DMG auf die Peers,
   und die fünf Status-Zustände live prüfen.

**Harte Regeln (nicht verhandelbar)**
- DMG **nur** über `scripts/build-release-dmg.sh` (es räumt Build-Kopien und
  LaunchServices ab); niemals `build.sh` + `package_dmg.sh` von Hand.
- Cleanup **nur** über `scripts/cleanup-cluster-macs.sh --all [--with-helper]`;
  `cluster/identity.json` nicht ohne ausdrückliche Ansage löschen.
- Ein Schritt darf nie grün aussehen, wenn sich seine Inputs geändert haben
  (`stale`), und ein Fehler gehört **an den fehlgeschlagenen Schritt**, nicht ins
  Advanced-JSON und nicht nur als Toast.
- Nie eine ungemessene Rate als Rate ausgeben; `predicted` nie rückwirkend
  überschreiben.
- Interaktive Schritte (App öffnen, Local Network, Helper-Dialog,
  Runtime-Installer) macht **ein Mensch an der Mac** — du fragst, statt per SSH
  zu starten.
- Vor einem Cluster-Lauf: gleiche `version` **und** gleicher App-Binary-Hash auf
  allen vier Macs.

**Umgebung:** `.11` 172.16.1.11 (M5 Max 128 GB), `.12` .12 (M2 Ultra 64 GB),
`.13` .13 (M1 Max 32 GB), `.14` .14 (M1 Max 32 GB); Modelle unter
`/Users/andreaskolb/Models` (großes M!); Port 8099; SSH ist nur Control Plane.

**Definition of Done dieser Session:** W1 + W2 implementiert und getestet
(Assistant- und UI-Suiten grün, Xcode-Tests für die neuen Swift-Teile), ein
notarisiertes DMG über den Wrapper gebaut, auf allen vier Macs installiert, und
ein kurzer Bericht welche der fünf Helper-Zustände und welche
Uninstall-Scopes du live gesehen hast. Wenn etwas blockiert: **fragen**, mit
genau einer Frage und zwei Optionen.

---