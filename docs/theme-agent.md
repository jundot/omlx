# Custom theme for external agents

The oMLX web UI exposes a small, versioned HTTP contract for planning and
applying color themes. It is meant for an **external** agent: the agent reads
the style image, chooses the colors, and calls this API. oMLX does not accept
images, does not extract palettes and does not run the agent.

Base URL: `http://127.0.0.1:8000/admin/api/theme` (use the configured host and
port). Authentication is the same as the admin API: send the main API key as
`Authorization: Bearer <key>`. Sub-keys are for inference only and do not work
here.

## Workflow

1. `GET /schema` once, or whenever `schema_version` changes.
2. Read the style image **in the agent**, then pick hex values for the tokens.
3. `POST /validate` while iterating. It never writes.
4. `GET /` for the current `revision`.
5. `PUT /` with that exact `revision` (or an `If-Match` header).
6. On `409`, re-read and plan again. On `422`, fix the listed tokens.

Do not call `/admin/api/global-settings`; that endpoint updates every setting
and is not the theme contract.

## Endpoints

### `GET /schema`

Returns `schema_version`, modes, groups, 22 token descriptors (role,
description, default per mode, blocking contrast pairs), the contrast rules,
and the non-editable status tints.

### `GET /`

```json
{
  "ok": true,
  "schema_version": 1,
  "revision": 3,
  "source": "custom",
  "overrides": {"light": {"bg-primary": "#fdf6e3"}, "dark": {}},
  "resolved": {"light": {"bg-primary": "#fdf6e3"}, "dark": {}},
  "contrast": [
    {"mode": "light", "foreground": "text-primary", "background": "bg-primary",
     "ratio": 16.4, "floor": 4.5, "blocking": true, "ok": true}
  ],
  "warnings": []
}
```

`resolved` is the effective palette (stock merged with overrides). `contrast`
contains both blocking and advisory pairs; `warnings` repeats the advisory
failures.

### `POST /validate`

Body is `{ "light": {token: hex}, "dark": {token: hex} }`; partial maps are
allowed. The response is the same shape as `GET /` but `ok: false` when a
blocking rule fails, with structured errors:

```json
{
  "ok": false,
  "errors": [{
    "code": "contrast_below_floor",
    "mode": "light",
    "token": "text-primary",
    "surface": "bg-tertiary",
    "ratio": 3.82,
    "floor": 4.5,
    "suggestion": {"value": "#475569", "ratio": 4.71},
    "message": "light: text-primary on bg-tertiary is 3.82:1, below 4.5:1"
  }],
  "resolved": {},
  "contrast": [],
  "warnings": []
}
```

Error codes: `unknown_token`, `invalid_hex`, `invalid_mode`,
`contrast_below_floor`. The schema is the list of allowed tokens; do not
invent token names.

### `PUT /`

```json
{
  "revision": 3,
  "light": {"bg-primary": "#fdf6e3", "text-primary": "#073642"},
  "dark": {"bg-primary": "#002b36", "text-primary": "#eee8d5"}
}
```

Writes both modes atomically and increments `revision`. A missing/old revision
returns `409` with the current state; a blocking contrast error returns `422`
with the same structured errors as `/validate`. An `If-Match: "3"` header is
accepted as an alternative to the body revision.

### `POST /recipe` (optional high-level layer)

If the agent would rather pick a small seed palette than all 22 tokens, send
`background`, `text` and `accent` per mode (all three required). `surface`
and the status accents (`success`, `warning`, `danger`, `info`) are optional;
a missing surface is derived from the background and missing status accents
fall back to the stock ones. Any other key is refused with `unknown_seed`:

```json
{
  "light": {"background": "#fdf6e3", "surface": "#f7efd9",
            "text": "#073642", "accent": "#cb4b16"},
  "dark": {"background": "#002b36", "surface": "#073642",
           "text": "#eee8d5", "accent": "#268bd2"}
}
```

The response contains the derived 22-token `tokens` map plus the resolved
palette and contrast report. The server nudges text, secondary text, links
and the focus ring toward black/white until they pass; if a seed cannot
reach the 4.5:1 floor the response is `422` with the usual
`contrast_below_floor` errors. It never saves. Pass the returned tokens to `/validate` or `PUT` (or use a draft).
This is a convenience layer, not a different contract.

### `POST /draft`

Stores a proposal without changing the live theme:

```json
{
  "light": {"bg-primary": "#fdf6e3", "text-primary": "#073642"},
  "dark": {"bg-primary": "#002b36"},
  "note": "Warm paper from the reference image",
  "source": "agent"
}
```

The response contains `draft.base_revision` (the live revision at proposal
time) and the resolved palette. The Custom Theme page shows the proposal with
Apply / Discard. `GET /draft` returns it, `DELETE /draft` discards it, and
`POST /draft/apply` with `{"revision": <base_revision>}` applies it. A stale
base revision returns `409`; re-read and re-propose. A draft is checked
against the contrast floor both when it is stored and when it is applied.

### `POST /undo`

```json
{"revision": 4}
```

Restores the previous applied theme and increments the revision. `409` if no
history; `GET /history` returns the bounded audit list.

### `DELETE /?revision=4`

Clears all overrides and returns to the stock colors. Guarded by `revision`
the same way.

## Rules

- Only the 22 editable tokens from `/schema` are accepted. Status
  `*-soft`, `*-line`, `*-text` and neutral utility channels are derived.
- Hex is opaque `#rrggbb`, case-insensitive; stored lowercase. Alpha is
  refused: a translucent text color would pass a contrast check it fails on
  screen.
- Blocking rules are exactly the ones marked `blocking: true` in `/schema`
  (primary/secondary text, links, button text). **4.5:1 is a hard floor:**
  `PUT` answers `422` with `contrast_below_floor` and writes nothing, and the
  Custom Theme editor refuses the same pairs. There is no override.
- Advisory rules are reported in `warnings`; the stock theme intentionally
  fails some of them, so they never block a write.
- Overrides are merged over stock. `PUT` replaces both mode maps; send `{}`
  for a mode to keep the stock values there.

## MCP adapter

`scripts/theme_mcp.py` is a dependency-free stdio MCP server that forwards
`theme_schema`, `theme_get`, `theme_validate`, `theme_recipe`, `theme_draft`
and `theme_apply` to this API. It does not read images or store state.

```json
{
  "servers": {
    "omlx-theme": {
      "transport": "stdio",
      "command": "python3",
      "args": ["/path/to/omlx/scripts/theme_mcp.py"],
      "env": {
        "OMLX_THEME_URL": "http://127.0.0.1:8000",
        "OMLX_API_KEY": "<main API key>"
      },
      "enabled": true,
      "timeout": 30
    }
  }
}
```

The adapter is optional; the REST contract above is the source of truth.

## Agent prompt shape

The user supplies a style image and asks for a theme. A good agent:

1. describes the image palette in words;
2. maps it to the schema roles, not to arbitrary CSS;
3. validates contrast before applying;
4. shows the user the proposed palette and **asks before PUT** unless the
   user explicitly asked it to apply. Prefer `POST /draft` when the user
   should review it in the Custom Theme page.

The agent may keep its own preview; oMLX renders the result in the Custom
Theme page after it is applied.
