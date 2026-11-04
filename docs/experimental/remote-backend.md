# Remote Backend (experimental)

Point an oMLX install at an oMLX server on another Mac. The laptop keeps
every client feature, the server Mac does the model work.

```
MacBook (client)                         Mac Studio (server, often headless)
  oMLX.app  ── screens, menubar, Ask AI ──▶  omlx serve --host 0.0.0.0 --api-key …
  omlx launch claude / codex / …      ──▶    models, settings, logs, cache
```

With a remote backend configured:

- The macOS app does not start a Python server. Every screen (Models,
  Downloads, Settings, Logs, Benchmarks, Security, Usage) talks to the remote
  server's admin API, so edits there change *that* server's settings.json.
- The menubar shows the remote server's reachability and stats. Web Dashboard
  and Chat open the remote server's pages.
- `omlx launch <tool>` configures the tool with the remote server's URL and
  API key.
- Ask AI's "This Mac" choice becomes "oMLX's server", which is the remote one.
  Its own "Another server" setting still works for a third server.
- `omlx start`, `omlx stop` and `omlx restart` report the remote server's
  state and refuse to start or stop it. Manage the server on its own Mac.

## Configure it

**macOS app:** Server → Backend → *Another server*. Enter the address
(`studio.local`, `studio.local:8000`, or `https://omlx.example.com`) and that
server's main API key, click *Test connection*, then *Apply and Relaunch*.
The app stops its local server and reopens using the remote one.

**settings.json** (the app and the CLI read the same section):

```json
"remote": {
  "enabled": true,
  "url": "studio.local:8000",
  "api_key": "the-remote-servers-main-key"
}
```

**Environment** (wins over the file; sets `enabled` when the URL is present):

```bash
OMLX_REMOTE_URL=studio.local:8000 OMLX_REMOTE_API_KEY=… omlx launch claude
```

Address forms: `host`, `host:port`, `http://host[:port]`, `https://host[:port]`,
bracketed IPv6 (`[fe80::1]:8000`). The default port is 8000, or 443 for
`https`. Paths, queries and credentials are rejected.

Precedence for `omlx launch`: explicit `--host` / `--port` target a local
server and bypass the remote backend; otherwise an enabled `remote` wins;
otherwise the local `server.host` / `server.port`.

## On the server Mac

Bind to a reachable address and set a main API key (oMLX refuses a
non-loopback bind without one). `--headless` keeps the admin API and drops
only the web UI, so a headless Mac can still be administered from the laptop.

```bash
OMLX_API_KEY=your-secret-key omlx serve --model-dir ~/models --host 0.0.0.0
```

Use the server's **main** key on the client. `/v1/models/status` accepts sub
keys, but the admin screens need the main key; *Test connection* checks both
and says which one failed.

## Security

The client stores the remote key in its own `settings.json`, next to the
local key, so the CLI can read it. `OMLX_REMOTE_API_KEY` overrides the file
for one process, but like `OMLX_API_KEY` it is written back whenever
settings are saved, so it is not a way to keep the key off disk. Over plain
`http` the key and prompts cross the network unencrypted, and the menubar's
Web Dashboard link carries the key in its auto-login URL; the app warns
about plain http, and `https` avoids it. Rotating the remote key from the
client's Security screen updates the local copy too.

## Not covered yet

- The first-run Welcome window still sets up a local server. A laptop that
  will never run models can skip it by writing the `remote` section into
  `settings.json` before the first launch. Launching with `OMLX_REMOTE_URL`
  also skips it; the setting is only written to `settings.json` if something
  saves settings during that run.
- Addresses are validated the same way on both sides, so an IPv6 link-local
  address with a zone id (`[fe80::1%en0]`) is not accepted yet.
- `omlx launch claude_desktop` writes the remote URL into Claude Desktop's
  gateway profile, but the dashboard's "configured" check on the server still
  looks for a loopback URL.
- The `remote` section is client-side only. It is never sent to a server and
  does not appear in the admin global-settings API.
