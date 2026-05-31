# Cluster router

A request-level reverse proxy that spreads OpenAI-compatible requests across
several independent `omlx serve` backends (e.g. two Macs). It is NOT
GPU/memory-level clustering and NOT a shared KV cache: each backend stays a
standalone server; the router only decides WHICH backend handles each request.

## What it does

- **Model-aware routing.** A request only goes to a backend that hosts the
  requested model. The router discovers each backend's catalog by polling
  `/v1/models/status`, so you never have to remember which machine has which
  model.
- **Weighted load balancing.** Each backend has a `weight` (relative compute
  capacity). Queue depth is normalized by weight, so a faster machine takes
  proportionally more traffic. A 1.0 / 1.3 split sends ~30% more to the 1.3
  backend before it looks as busy as the 1.0 one.
- **Resident-preferred + sticky.** Traffic for a model prefers a backend that
  already has it loaded (avoids cold load + LRU eviction thrash), and stays
  there unless that backend gets meaningfully busier than an idle peer.
- **Soft memory gate.** A backend near its Metal ceiling is deprioritized (a
  ramped load penalty) so new traffic steers away from an imminent
  evict-and-reload. It is soft: a pressured backend is never hard-excluded.
- **Streaming + failover.** SSE is byte-passthrough with no timeout; a client
  disconnect cancels the upstream so a dead client never pins a slot. Connection
  failures fail over to another eligible backend, but only PRE-stream (once SSE
  bytes flow, a transparent failover is impossible).

## Architecture

```
   client
     |  POST http://<router-host>:9000/v1/chat/completions
     v
  router :9000          (stateless; does NOT load models)
     |\
     | \--> backend A  http://127.0.0.1:8000   (omlx serve, loopback)
     |
     \----> backend B  http://10.0.0.1:8000    (omlx serve, over the network)
                              ^
                       models actually run on the :8000 servers
```

The router listens on a DIFFERENT port from the servers (default `:9000`)
because it runs alongside a server on the same host and cannot share `:8000`.
Direct access to any single backend on its own `:8000` keeps working unchanged.

## Backend prerequisites

Each backend's `omlx serve` must be reachable from the router host. By default a
server binds `127.0.0.1` (loopback only); for cross-machine access set
`server.host` to `0.0.0.0` in that machine's `~/.omlx/settings.json` and restart
the server (host is read only at startup). Bind `0.0.0.0` only on a trusted
network (LAN / tailscale) -- the only auth in front of a backend is its
`X-API-Key`.

A router co-located with one backend reaches THAT backend over `127.0.0.1`
without rebinding it; only the OTHER machines need `0.0.0.0`.

### Hostname caveat

Use a real, directly-routable IP for remote backends, not a hostname that a VPN
might hijack. On this fleet `m5max` resolves to a fake-IP address under the VPN
and is unreachable; the Thunderbolt-bridge IP `10.0.0.1` works. Put the IP in
`base_url`, e.g. `http://10.0.0.1:8000`.

## Configuration

Config is read from `$OMLX_CLUSTER_CONFIG`, or `~/.omlx/cluster.json` if unset.
See `omlx/cluster/cluster.example.json` for a template.

```json
{
  "listen": "0.0.0.0:9000",
  "router_api_key": "key-clients-present-to-the-router",
  "poll_interval": 1.5,
  "affinity_hysteresis": 1.0,
  "mem_soft_floor": 0.05,
  "mem_penalty": 10.0,
  "backends": [
    { "name": "m2max", "base_url": "http://127.0.0.1:8000", "api_key": "...", "weight": 1.0 },
    { "name": "m5max", "base_url": "http://10.0.0.1:8000",  "api_key": "...", "weight": 1.3 }
  ]
}
```

| Field | Meaning |
|-------|---------|
| `listen` | `host:port` the router binds. |
| `router_api_key` | Key clients present to the router (`X-API-Key` or `Bearer`). `null` = open. The router replaces it with each backend's real `api_key` upstream. |
| `backends[].base_url` | A directly-reachable URL of that backend's `omlx serve`. |
| `backends[].api_key` | That backend's real `X-API-Key`. |
| `backends[].weight` | Relative compute capacity (> 0). Higher = more traffic. |
| `poll_interval` | Seconds between backend health/catalog polls (default 1.5). |
| `affinity_hysteresis` | Weight-normalized load slack before a model's sticky backend is abandoned (default 1.0). |
| `mem_soft_floor` | Headroom fraction below which the memory penalty starts (default 0.05). |
| `mem_penalty` | Penalty in load units at zero headroom (default 10). |

## Running

```
OMLX_CLUSTER_CONFIG=~/.omlx/cluster.json python -m omlx.cluster.router
```

Point OpenAI clients at `http://<router-host>:9000/v1` with the
`router_api_key`. `GET /health` returns each backend's live snapshot
(healthy / weight / depth / load / score / mem_headroom / resident models).
`GET /v1/models` returns the merged catalog with a `servers` field per model.

### Persisting with launchd (macOS)

Run the router as a LaunchAgent so it starts at login and respawns on crash:

```
~/Library/LaunchAgents/com.flyto.mlx.cluster-router.plist
```

with `RunAtLoad` + `KeepAlive`, `ProgramArguments` invoking
`python -m omlx.cluster.router`, and `OMLX_CLUSTER_CONFIG` in
`EnvironmentVariables`. Load with `launchctl load <plist>`; logs go to
`~/.omlx/cluster-router.log`.

### High availability

The router is stateless, so you can run one on each machine. Clients point at
either `host:9000`; if one router host dies, point clients at the other. Each
router needs every backend reachable from its own host (so every non-loopback
backend must bind `0.0.0.0`).
