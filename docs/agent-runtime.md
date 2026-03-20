# Agent Runtime Notes

This document tracks the evolving oMLX agent runtime design.

## Current Layers

1. Bootstrap context
Loaded from `agent/*.md` or per-request `profiles/<name>/agent` / `workspaces/<name>/agent`.

2. Structured shared state
Stored per agent as:
- `MISSION.md`
- `FACTS.md`
- `DECISIONS.md`
- `TODO.md`

3. Persistent memory notes
Append-only markdown notes written under the selected memory backend.

4. Session container
Per-agent session logs stored as:
- `session.jsonl`
- `meta.json`
- `index.json`
- `HANDOFF.md`

5. Responses session registry
The OpenAI Responses API store remains the canonical API-facing session chain.

## Current Behavior

- Requests may select `profile` or `workspace`.
- Bootstrap injects:
  - durable MD files
  - MCP tool catalog
  - previous session handoff
  - structured shared state
  - recent memory notes
- Every stored Responses exchange is appended to the per-agent session container.
- Session rollover generates a lightweight index and handoff file.
- New sessions automatically inherit the previous session index pointer.
- Archived sessions can be searched with simple substring matching.

## Why KV Cache Is Separate

KV / prefix cache helps multiple agents reuse prompt-prefill work.
It does not represent durable shared understanding.

Shared understanding lives in:
- structured state
- persistent memory notes
- session handoff / indexes

## Near-Term Roadmap

1. Automatic state writeback
Use conservative heuristics to update mission / facts / decisions / todo after each exchange.

2. Retrieval policy
When context is missing, search session archives and inject matched fragments on demand.

3. Obsidian-first backend polish
Stabilize file layout and naming for long-running vault usage.

4. Admin workbench
Expand the multi-agent session page into a real operator console with state and search visibility.

## File Layout

Typical per-agent memory root:

```text
memory/
  profile-planner/
    _state/
      MISSION.md
      FACTS.md
      DECISIONS.md
      TODO.md
    2026-03-18T10-00-00Z-note.md
    _sessions/
      session_.../
        session.jsonl
        meta.json
        index.json
        HANDOFF.md
```
