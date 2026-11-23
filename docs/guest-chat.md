# Anonymous (guest) chat

Guest chat serves the normal chat page to people who have no admin login.
Turn it on and anyone who can reach the server can open `http://<host>:<port>/chat`
and talk to your models. Several people can use it at the same time; each
visitor's conversations are kept in their own browser, exactly as the admin
chat page already does.

It is **off by default** and nothing is exposed until you enable it.

## Enabling it

Web dashboard → **Settings → Auth & Info → Anonymous chat**. Switching it on
mints a guest key and shows the link to share. **Copy** puts the URL on your
clipboard; **New link** issues a fresh key, which signs out anyone using the
old one.

From the command line:

```sh
curl -b admin-cookies.txt -X POST http://127.0.0.1:8000/admin/api/guest-chat \
  -H 'Content-Type: application/json' -d '{"enabled": true}'
```

The reply carries the guest key. `GET /admin/api/guest-chat` reports whether
it is on. Both endpoints need an admin session, like the rest of `/admin`.

## What a guest can and cannot do

A guest page is handed `auth.guest_chat_key` — **never** your main API key.
That key authenticates the inference API and nothing else:

- **Can**: list models, chat, stream replies, use MCP tools and web search if
  you have them on, transcribe audio.
- **Cannot**: reach any `/admin` endpoint. Settings, benchmarks, downloads,
  logs, the API keys themselves and the dashboard are all still behind the
  admin session. The guest page does not even display a link to them.

The guest key is kept out of `auth.sub_keys`, so revoking guest access is one
switch and can never be confused with your own API keys.

## What to keep in mind

Guests share the machine with you. There are no per-guest limits in this
version. The guest key reaches the whole inference API, so a visitor can load
or unload models (`POST /v1/models/{id}/load`, `/unload`) as well as queue
long generations, which compete with your own requests for GPU and memory. On a server bound to
a LAN address this means anyone on that network — so enable it on networks
you trust, and turn it off when you are done.

Guest traffic is ordinary inference traffic. It appears in Usage History like
any other request and is not attributed per visitor.

## Turning it off

Flip the switch off, or:

```sh
curl -b admin-cookies.txt -X POST http://127.0.0.1:8000/admin/api/guest-chat \
  -H 'Content-Type: application/json' -d '{"enabled": false}'
```

`/chat` then redirects to the admin chat page, which asks for the login as
usual, and `/admin/chat/guest` returns 404. The key is kept so that switching
back on resumes the same link; rotate it if you would rather it did not.
