# Admin API

The admin API under `/admin/api/` manages a running oMLX server: settings, models, downloads, quantization, benchmarks, and serving stats. The web UI and the macOS app use the same endpoints. It stays available when the server runs without the web UI (`omlx serve --headless` or `OMLX_HEADLESS=1`), so scripts and other clients can control the server the same way.

## Authentication

Send the main API key as a Bearer token:

```bash
curl -H "Authorization: Bearer $OMLX_API_KEY" http://127.0.0.1:8000/admin/api/stats
```

- Only the main API key is accepted. Sub keys are for the inference API (`/v1/*`) and get 401 here. Rejected keys are logged by fingerprint, never in plain text.
- `POST /admin/api/models/{model_id}/load` and `POST /admin/api/models/{model_id}/import-mtplx` also accept sub keys.
- Browsers use a session cookie instead. `POST /admin/api/login` with `{"api_key": "...", "remember": false}` sets `omlx_admin_session`, valid for 24 hours, or 30 days with `"remember": true`. `POST /admin/api/logout` clears it.
- With `skip_api_key_verification` enabled and a loopback-only bind, admin requests need no credentials.
- Failed authentication returns 401 with a JSON `detail`. A browser request without credentials (`Accept: text/html`) is redirected to the `/admin` login page instead, except in headless mode, where there is no login page.

## Full route list

`GET /openapi.json` and the Swagger page at `GET /docs` list every route with its query parameters and request body schema. Response bodies are plain JSON objects that the schema does not describe yet; the endpoints below show what they hold.

## Endpoints

### Server and settings

| Endpoint | Description |
|---|---|
| `GET /admin/api/server-info` | Host, port, and server aliases |
| `GET /admin/api/global-settings` | All global settings |
| `GET /admin/api/global-settings/defaults` | Default values for the settings form |
| `POST /admin/api/global-settings` | Partial update with flat fields, for example `{"max_concurrent_requests": 16}`. `runtime_applied` in the response lists the changes that took effect without a restart |
| `POST /admin/api/server/restart` | Restart, only when the server runs under a supervisor (the macOS app or `brew services`) |
| `POST /admin/api/sub-keys`, `DELETE /admin/api/sub-keys` | Create or remove sub keys |

### Status and logs

| Endpoint | Description |
|---|---|
| `GET /admin/api/stats?scope=session\|alltime&model=` | Request and token counters, speeds, cache use, and loaded models |
| `GET /admin/api/activity` | Loaded models and their live activity, cheap enough for frequent polling |
| `POST /admin/api/stats/clear`, `POST /admin/api/stats/clear-alltime` | Reset session or all-time counters |
| `GET /admin/api/usage?range=&model=` | Local usage history, see [usage-analytics.md](usage-analytics.md) |
| `GET /admin/api/logs?lines=&file=` | Tail of the server log |
| `GET /admin/api/device-info` | Chip, GPU cores, and memory |
| `POST /admin/api/ssd-cache/clear`, `POST /admin/api/hot-cache/clear` | Clear the SSD or in-memory KV cache |

### Models

| Endpoint | Description |
|---|---|
| `GET /admin/api/models` | Discovered models with load state, size, and settings |
| `POST /admin/api/models/{model_id}/load`, `.../unload` | Load or unload a model |
| `POST /admin/api/reload` | Rescan the model directories |
| `PUT /admin/api/models/{model_id}/settings` | Update per-model settings (sampling, context, aliases, acceleration) |
| `POST /admin/api/models/{model_id}/settings/reset` | Reset per-model settings |
| `GET`, `POST /admin/api/models/{model_id}/settings/optimal` | List or apply benchmarked optimal settings |
| `POST /admin/api/models/{model_id}/settings/recipe` | Apply a shared settings recipe |
| `GET`, `POST /admin/api/models/{model_id}/profiles`, `PUT`, `DELETE .../profiles/{name}`, `POST .../profiles/{name}/apply` | Per-model profiles |
| `GET`, `POST /admin/api/profile-templates`, `PUT`, `DELETE .../{name}` | Global profile templates |

`POST /v1/models/{model_id}/load` and `/unload` do the same as the admin load and unload, and accept the main key or a sub key.

### Downloads and uploads

| Endpoint | Description |
|---|---|
| `GET /admin/api/hf/search?q=`, `GET /admin/api/hf/model-info?repo_id=` | Search Hugging Face and read a model card |
| `POST /admin/api/hf/download` | Start a download, `{"repo_id": "mlx-community/..."}` |
| `GET /admin/api/hf/tasks` | Download progress |
| `POST /admin/api/hf/cancel/{task_id}`, `POST .../retry/{task_id}`, `DELETE .../task/{task_id}` | Cancel, retry, or remove a task |
| `GET /admin/api/hf/models`, `DELETE /admin/api/hf/models/{model_name}` | List or delete downloaded models |
| `/admin/api/ms/...` | The same for ModelScope |
| `/admin/api/upload/...` | Upload oQ models to Hugging Face |

### Quantization

| Endpoint | Description |
|---|---|
| `GET /admin/api/oq/models`, `GET /admin/api/oq/estimate?model_path=&oq_level=` | Candidate models and size estimate |
| `POST /admin/api/oq/start` | Start oQ, `{"model_path": "...", "oq_level": ...}`. See [oQ_Quantization.md](oQ_Quantization.md) |
| `GET /admin/api/oq/tasks`, `POST .../cancel/{task_id}`, `DELETE .../task/{task_id}` | Progress, cancel, remove |

### Benchmarks

| Endpoint | Description |
|---|---|
| `POST /admin/api/bench/start`, `GET .../{bench_id}/results`, `POST .../{bench_id}/cancel` | Throughput benchmark |
| `GET /admin/api/bench/{bench_id}/stream` | Server-sent events for the same run |
| `/admin/api/bench/context/...` | Long-context benchmark, same pattern |
| `/admin/api/bench/accuracy/...` | Accuracy benchmark queue |
| `/admin/api/bench/ane-tune/...` | ANE prefill tuning |

Cluster endpoints are described in [distributed-cluster.md](experimental/distributed-cluster.md).

## Long-running tasks

Downloads, oQ, uploads, and benchmarks run in the background. The start call returns a task or benchmark ID right away; poll the matching `tasks` or `results` endpoint (benchmarks also offer an SSE `stream`), and use `cancel` to stop. The web UI polls downloads every 500 ms and stats every 500 ms while the dashboard is open; scripts rarely need anything that fast.

## Examples

```bash
export OMLX=http://127.0.0.1:8000
export AUTH="Authorization: Bearer $OMLX_API_KEY"

# Loaded models and serving stats
curl -s -H "$AUTH" "$OMLX/admin/api/stats"

# Load a model
curl -s -X POST -H "$AUTH" "$OMLX/admin/api/models/Qwen3-8B-4bit/load"

# Change a setting at runtime
curl -s -X POST -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"max_concurrent_requests": 16}' "$OMLX/admin/api/global-settings"

# Download a model and watch progress
curl -s -X POST -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"repo_id": "mlx-community/Qwen3-8B-4bit"}' "$OMLX/admin/api/hf/download"
curl -s -H "$AUTH" "$OMLX/admin/api/hf/tasks"
```
