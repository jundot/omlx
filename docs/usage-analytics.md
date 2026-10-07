# Local usage history

The web dashboard and macOS app's **Status → Usage History** show Today,
Yesterday, 7/30/90 Days, and This Month, with model filtering, token totals,
per-model generation speed, and a day/hour token heatmap. History starts when
this version first serves requests; existing all-time totals cannot be backfilled.

Usage stays on this server. No telemetry is sent. Only the canonical oMLX model
ID, hourly bucket, request/token counts, and accumulated durations are stored.
There are no prompts, responses, messages, token IDs, API keys, headers, upload
names, or document contents. Client identifiers (API key names or peer IPs) are
stored only if you turn on [per-client history](#per-client-history). Model IDs are the same identifiers used
by the engine pool and existing serving statistics, not display aliases or full
model paths. Renaming an ID starts a new series; replacing weights under the same
ID continues that series. Removed/unloaded models remain in history.

## Accounting

History consumes the existing completed-request serving-statistics hook for
OpenAI completions/chat/responses, Anthropic messages, embeddings, reranking,
and audio. It inherits that hook's coverage: failed/rejected or disconnected
requests that do not reach accounting are not counted. Local benchmark requests
that use these serving endpoints also count. Audio contributes requests with
zero tokens because byte/character counts are not token counts.

- **Prompt tokens**: the engine's full input count, including cached tokens.
- **Output tokens**: the engine's completion count, including generated reasoning
  and tool-call tokens counted by the engine.
- **Total tokens**: prompt + output; cached tokens are already part of prompt.
- **Cached tokens**: the engine-reported prompt tokens actually reused from KV
  prefix cache, after reconstruction, alignment, trimming, and fallback. This
  includes memory/SSD reuse where the engine reports it; it is neither a count of
  cache lookups nor cache writes. Unsupported/no reuse reports zero. Anthropic
  cache-control billing fields do not replace these engine counters.
- **Cache efficiency** (API): cached / prompt, a fraction from 0 to 1.
- **Generation speed**: sum(output tokens) / sum(generation seconds), matching
  existing serving-statistics weighting; not the mean of per-request speeds.
- **Prefill speed** (API): sum(prompt − cached) / sum(prefill seconds).
- **Request seconds** (API): measured serving-handler duration, including waits
  within that measurement, not HTTP middleware/network latency. Loading inclusion
  follows the existing endpoint timer. Unknown durations (currently audio) are
  excluded from `timed_requests` and `average_request_seconds`.

Prefill/generation timing follows existing endpoint accounting (time to first
output and subsequent generation; supported diffusion engines use native timing).
Overlapping requests each contribute their own duration. Summed seconds are not
GPU busy time. Missing speed measurements return `null`, not invented throughput.

## Storage and retention

`<base_path>/usage.sqlite3` lives alongside `stats.json` (normally
`~/.omlx/usage.sqlite3`; follows `OMLX_BASE_PATH` and the app's base-path setting).
Python's built-in SQLite stores one row per active model/hour with schema version
1 (`PRAGMA user_version`). Existing cumulative `stats.json` and its clear controls
remain independent. No new dependency or configuration is required.

A background thread checks for pending aggregates every 5 seconds and flushes
on normal shutdown. With no pending data, it skips database access unless daily
retention maintenance is due or a storage failure needs retrying. The UI polls every 15 seconds. Sudden termination can lose the unflushed
batch. Records are attributed to the local hour when accounting completes, not
split across the hours of a long request. Epoch bucket keys distinguish repeated
DST hours; calendar queries use server-local dates, including 23/25-hour days and
fractional UTC offsets. The heatmap combines repeated hours. Changing the server's
timezone reinterprets historical bucket timestamps; boundaries then have hourly
precision, not exact request-level precision.

Hourly rows older than 400 days are pruned daily; SQLite reuses freed pages and
incrementally reclaims space. Idle models do not generate rows. At most 4,096
pending model/hour aggregates are retained during storage outages; overflow drops
analytics only. The API reports current-process `dropped_requests` and storage
availability. Reads use committed snapshots and never wait for a flush. Storage
failures do not prevent inference; recoverable write failures retry. Corruption
found at startup is moved to one `usage.sqlite3.corrupt` recovery backup (plus any
SQLite sidecars) and a fresh database is created. Future schema versions are left
untouched. Runtime corruption can require a server restart.

To reset history, stop oMLX and remove `usage.sqlite3`, `usage.sqlite3-wal`, and
`usage.sqlite3-shm` from the configured base directory, if present. Remove the
`.corrupt` backup and its sidecars too if desired. Restart oMLX to begin fresh.
Clearing Session or All Time in the dashboard does not erase history.

## Disabling

Usage history is on by default. Turn it off with the **Record usage history**
switch (web dashboard: Settings → Usage History; macOS app: Server → Usage
History), or set `usage.usage_history` to `false` in `settings.json`. The
`OMLX_USAGE_HISTORY` environment variable (`0`/`false`/`off` or `1`/`true`/`on`)
overrides the saved value at startup.

The switch applies immediately, without a restart. Turning it off flushes any
pending aggregates and then stops recording; `usage.sqlite3` stays in place, so
turning it back on resumes the same history. A server started with recording off does not create, open, or repair the database;
initialization waits until recording is enabled. After pending aggregates have
been flushed (including retries after storage failures), nothing is read from or
written to the database while off: the Usage History panel reports that history is off
and points to Settings, and `GET /admin/api/usage` returns `"enabled": false`
with zero totals instead of the unavailable error. Cumulative Session and All
Time statistics are unaffected.

## Per-client history

Turn on **Track usage per client** (web dashboard: Settings → Usage History;
macOS app: Server → Usage History), set `usage.usage_by_client` to `true` in
`settings.json`, or set `OMLX_USAGE_BY_CLIENT=1` at startup. It is off by
default and applies immediately. Usage History then also shows a **Clients**
breakdown, filtered by the selected range and model, with three tabs: **All**
(each key and IP pair), **By key**, and **By IP**.

Each request records two labels:

- **Key**: the name of the API sub key that authenticated the request, **Main
  API key**, or **No key** when no key was checked (loopback without a key,
  skipped verification, or unauthenticated inference). An unnamed sub key is
  shown by the same 8-character fingerprint used in rejected-key log lines.
  Renaming a sub key starts a new series.
- **IP**: the peer address of every request, whichever key it used. IPv4-mapped
  IPv6 peers are shown as IPv4. `X-Forwarded-For` is not trusted, so behind a
  reverse proxy every request shows the proxy's address; give each client its
  own sub key to tell them apart.

Rows live in a separate `client_usage_hourly` table in the same database, one
per model/key/IP/hour, with the same metrics, 400-day retention, and their own
4,096-aggregate pending bound (`dropped_client_requests`). Overflow drops only
the per-client row; model totals are unaffected. Turning tracking off keeps
recorded rows visible. The table is added without changing the schema version,
so older oMLX versions keep reading model history. To erase per-client history
only, stop oMLX and run
`sqlite3 ~/.omlx/usage.sqlite3 'DELETE FROM client_usage_hourly'`.

## Admin API

`GET /admin/api/usage?range=7d&model=<canonical-model-id>` uses the existing admin
session-cookie authentication (and the existing explicit auth-bypass setting).
`range` accepts `today` (default), `yesterday`, `7d`, `30d`, `90d`, or `month`.
Multi-day ranges include today; Yesterday is the preceding calendar day. Omit
`model` for all models. Filtering uses the exact canonical ID, even after unload
or removal. OpenAI-compatible responses and endpoints are unchanged.

The default JSON includes `totals`, `models`, the three client views, and `heatmap`, plus range,
retention, refresh, `enabled`, `by_client`, availability, and overflow metadata.
`clients` has one entry per key and IP pair, `clients_by_key` one per key, and
`clients_by_ip` one per IP. Entries carry `key_kind` (`sub_key`, `main_key`, or
`none`) and `key_id` (the sub key name or fingerprint; empty otherwise), and/or
`client_ip`, plus the aggregate metrics below. Add `include_details=true`
to also compute and return the full `daily` and `hourly` aggregates. The web and
Mac panels use the compact default response. Aggregate metrics are
`requests`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `cached_tokens`,
`prefill_seconds`, `generation_seconds`, `request_seconds`, `timed_requests`,
`cache_efficiency`, `generation_tps`, `prefill_tps`, and `average_request_seconds`.
Unsupported ranges return 422; inaccessible history returns a generic 503.

For example, with an admin session cookie file obtained through the existing
login flow:

```sh
curl -b /path/to/admin-cookies.txt \
  'http://127.0.0.1:8000/admin/api/usage?range=month&model=your-model-id'
```
