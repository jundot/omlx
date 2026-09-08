# Local usage history

The web dashboard and macOS app's **Status → Usage History** show Today,
Yesterday, 7/30/90 Days, and This Month, with model filtering, token totals,
per-model generation speed, and a day/hour token heatmap. History starts when
this version first serves requests; existing all-time totals cannot be backfilled.

Usage stays on this server. No telemetry is sent. Only the canonical oMLX model
ID, hourly bucket, request/token counts, and accumulated durations are stored.
There are no prompts, responses, messages, token IDs, API keys, headers, client
IPs, upload names, or document contents. Model IDs are the same identifiers used
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

A background thread flushes in-memory aggregates every 5 seconds and on normal
shutdown. The UI polls every 15 seconds. Sudden termination can lose the unflushed
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

## Admin API

`GET /admin/api/usage?range=7d&model=<canonical-model-id>` uses the existing admin
session-cookie authentication (and the existing explicit auth-bypass setting).
`range` accepts `today` (default), `yesterday`, `7d`, `30d`, `90d`, or `month`.
Multi-day ranges include today; Yesterday is the preceding calendar day. Omit
`model` for all models. Filtering uses the exact canonical ID, even after unload
or removal. OpenAI-compatible responses and endpoints are unchanged.

The JSON includes `totals`, `models`, `daily`, `hourly`, and `heatmap`, plus range,
retention, refresh, availability, and overflow metadata. Aggregate metrics are
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
