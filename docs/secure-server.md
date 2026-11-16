# Secure Server

**Security → Secure Server** (macOS app) runs the oMLX server as its own macOS
account with no network access of its own: only this Mac can reach it, and it
cannot reach the internet — no model downloads, no telemetry, no update checks.
Everything else on the Mac keeps its network. The web dashboard shows the state
read-only and disables the controls that would need the internet.

## Scope: the server, not the Mac

Every rule in the anchor is qualified with `user _omlx`, and the server is
*spawned* as that account. pf associates a packet with the uid that owns the
socket, so the rules match the server's traffic and nothing else:

> `user ⟨user⟩` — This rule only applies to packets of sockets owned by the
> specified user. For outgoing connections initiated from the firewall, this is
> the user that opened the connection. For incoming connections to the firewall
> itself, this is the user that listens on the destination port. — `pf.conf(5)`

Both halves matter. Outbound, the server cannot dial out. Inbound, the same
qualifier stops *other machines* from reaching a server the sandbox owns, while
the loopback rules keep this Mac able to reach it. Measured on macOS 26 against
a `_omlx` listener: with only the inbound block loaded, a connection from the
console user is dropped (that rule's `Packets:` counter climbs, `curl` times
out); with the loopback pass loaded the same connection answers `200`.

The bind address is pinned to `127.0.0.1` when the mode starts the server
(`--host` beats both the environment and `settings.json`), so reachability does
not depend on pf being up. A `0.0.0.0` in your settings is therefore ignored
while the mode is on: other devices on your network cannot connect.

A machine-wide `block drop out all` is *not* what this feature does, and is what
its guard exists to prevent: it would take the browser, Mail and the updater
down with the server. Per-process filtering by other means (a `NEContentFilter`
system extension) needs a restricted Apple entitlement and a signed extension;
the `user` qualifier gets the same scope from a supported mechanism.

## Mechanism

| Layer | What it does |
|---|---|
| `pf` sub-anchor `com.apple/900.omlx.secure` | Loopback passes, then `block` for everything else, every rule scoped `user _omlx`. Loaded under the existing `anchor "com.apple/*"` point, so `/etc/pf.conf` is never edited. |
| `_omlx` account | Created by the helper with `dscl` if missing: a free uid in 460–499, primary group `staff` (so world- and group-readable models stay readable), shell `/usr/bin/false`, home inside the root-owned config directory. |
| Spawn | The helper re-executes its **own** binary with `--drop-and-exec` (its own trampoline flag, unrelated to Apple's deprecated `sandbox-exec(1)`); that child calls `setgid`/`setuid(_omlx)` then `execve`s the interpreter, so the uid is in place before the listening socket exists. macOS has no `POSIX_SPAWN_SETEUID` and Swift 6 has no `fork()`, so this is done in-process — no `sudo`, no shell. Environment is passed via the spawn, never on a command line. |
| `oMLXSecureServerHelper` | A privileged LaunchDaemon registered with `SMAppService.daemon(plistName:)`, embedded in `oMLX.app/Contents/Library/LaunchDaemons`. Only writer of the config directory, only process that runs `pfctl`, only thing that can change uid here. |
| Authorization | Every flip needs the right `app.omlx.secure-server.manage` (`class = user`, `group = admin`, `timeout = 0`, `shared = false`): the standard administrator dialog, every time, never a cached credential. |
| Caller identity | The daemon cannot prompt, so it never re-evaluates the right (with `timeout = 0` there is nothing to re-evaluate). It verifies the process on the other end of the XPC connection against `cdhash H"<app binary>"` instead, which also pins an ad-hoc signed build. |
| XPC | `com.omlx.secure-server-helper` exposes `getState`, `status`, `setEnabled(_:authorization:)`, `startServer(_:)`, `stopServer()`. `startServer`/`stopServer` need no authorization form — with the file-access gate below they ask for nothing the app could not do to itself — but still require the code-identity check. |
| File access | ACL entries that let `_omlx` traverse and read what the requesting user owns, so a private `~/.local` (uv's interpreters) or a `drwx------` model tree still works. Never wider than that; see below. |
| Python server | Reads the state file, answers egress endpoints with a 403 that explains how to turn the mode off, and skips the benchmark/accuracy uploads. |
| Web dashboard | Read-only status card; download buttons disabled while the mode is on. |

## Why a privileged system service

Four things Secure Server needs are operations macOS only grants to root, and
it grants none of them to a normal app:

1. Load/enable `pf` rules — root-only `/dev/pf` ioctls; `pfctl` is the only
   interface, there is no public framework.
2. Create the `_omlx` account — a Directory Services write, done once.
3. Set the server's uid BEFORE it opens its listening socket — pf tags a socket
   with the effective uid that created it; a process that binds as root and drops
   privileges later keeps root's credentials and matches no rule. macOS has no
   `POSIX_SPAWN_SETEUID` and Swift 6 strict concurrency forbids `fork()`, so the
   only supported place to `setuid` then `exec` is a root process.
4. Grant file ACLs, restore ownership, and own the root-only config directory.

Apple's supported way to ship that root code is a LaunchDaemon registered with
`SMAppService.daemon`, approved once in System Settings and started on every
boot. A per-app network filter would need a restricted `NEContentFilter`
entitlement + a signed system extension; `sandbox-exec` is deprecated and
unsupported. The helper is the SMALLEST privileged component that can do the
job, boxed in by a fresh admin authorization on every change (`timeout = 0`,
`shared = false`, never a cached credential) and a cdhash pin on the caller. It
never filters any process except the one under `_omlx`.

## What the switch sets up (one-shot bootstrap)

Flipping the switch ON runs ONE idempotent sequence; every step is safe to
repeat (retry, crash mid-way, reboot all converge on the same state):

1. Register the privileged helper (one-time System Settings approval) + start it.
2. Create the root-owned config directory and the rules file.
3. Create the `_omlx` account and its 0700 home if missing.
4. Flush the pf anchor, load the rules, read them back from the kernel, and
   REFUSE to continue unless every rule is scoped to `_omlx`; only then enable
   pf and record its reference token.
5. Write state.json and restart the server under `_omlx`, granting the narrow
   file access it needs.

Any failure rolls back and leaves the mode off. OFF reverses it. The app and
helper share ONE bootstrap entry point, so the switch, startup recovery, and a
retry take the same path.

## External processes (and why so few)

In-process now: libproc enumerates processes/FDs/listening sockets; the ACL API
reads ACLs; `setgid`/`setuid`/`execve` drop privileges in the helper's OWN
re-executed binary; a `bind` probe checks a port. REMOVED: `pgrep`, `lsof`,
`netstat`, `ls -led`, `sudo`, `/bin/sh`, and the on-disk `server.env` file.
Environment is passed via the spawn (never on a command line).

Only THREE external programs remain, each with no supported in-process
substitute, none running while the server serves requests:

| Program | When | Why it stays |
|---|---|---|
| `/sbin/pfctl` | every flip | The only interface to the kernel's packet filter. |
| `/usr/bin/dscl` | once, at first enable | Creating a local account is a Directory Services write. |
| `/bin/chmod` | each grant / at teardown | Appending an ACE through the C ACL API means rebuilding its UUID qualifier by hand from Open Directory; `/bin/chmod +a` is the documented text form and preserves the existing ACL. Replacing the whole ACL with `acl_set_file` would drop the user's own entries. |

None of these runs while the server serves requests. The general
(non-sandboxed) app server lifecycle still uses `lsof`/`pgrep` for port
conflicts; that is pre-existing app infrastructure, NOT this service.

## The rules

```pf
pass in  quick on lo0 proto tcp from any to any user _omlx keep state
pass in  quick on lo0 proto udp from any to any user _omlx keep state
pass out quick on lo0 proto tcp from any user _omlx keep state
pass out quick on lo0 proto udp from any user _omlx keep state
block drop in  quick proto tcp from any to any user _omlx
block drop in  quick proto udp from any to any user _omlx
block drop out quick proto tcp from any user _omlx
block drop out quick proto udp from any user _omlx
```

`quick` ends the search at the first match, so the loopback passes must stay
above the blocks. Both protocols are covered in both directions: a UDP listener
under this account would otherwise be reachable from the network while this page
says it cannot be. Only TCP and UDP can be associated with a uid; a rule with no
protocol would match other processes' ICMP as well, so every rule names one.

pf resolves account names as rules are loaded: `pfctl -sr` reports `user = 502`,
never `user _omlx`. The helper's verification therefore matches on the numeric
form — as a whole number, so `user = 5020` never counts as proof that a rule was
scoped to `502` — and matches **every** rule it reads back from the kernel.

## Applying: flush, load, verify, enable — in that order

Enabling pf is the last step, because pf may already hold an anchor with an older
ruleset. An earlier build loaded the rules *after* `pfctl -E`, and a stale
unqualified ruleset became active the moment pf was enabled: the whole Mac lost
its network. Now the anchor is flushed, the file is loaded, the loaded rules are
read back, and unless every one of them is scoped to the sandbox uid the load is
flushed again, the transition fails, and the state is reset to off. The daemon
never enables pf over rules it cannot account for.

## Files

Under a root-owned directory (`root:wheel`), so "changing this needs an
administrator" is literal: no process running as you can edit them.

| Path | Notes |
|---|---|
| `…/secure-server/pf-anchor.conf` | The rules above. |
| `…/secure-server/state.json` | `{"enabled": bool, "pf_token": string|null}` — what was asked for. `pf_token` is the `pfctl -E` reference, released with `pfctl -X` on the way out. |
| `…/secure-server/access.json` | Every ACL entry the helper added, with the owner it must be returned to. `root:wheel 0600`. |
| `…/secure-server/sandbox/` | The sandbox's `HOME`/`TMPDIR` (`_omlx`, `0700`), including its log and cache directories. |
| `…/secure-server/sandbox/.omlx/settings.json` | The copy the server saves into (`OMLX_SETTINGS_FILE`), seeded from your file at every start. Your own `settings.json` therefore never becomes sandbox-owned, which matters because the app reads it at launch as you. |
| `…/secure-server/settings.seed.json` | What the copy looked like when it was seeded: source path, owner, digest. `root:wheel 0600`, and deliberately *not* in the sandbox home — teardown writes the file it names as root, so a record the sandbox could rewrite would be an arbitrary root write. |
| `…/secure-server/server.log` | The sandboxed server's stdout/stderr. Opened by root and handed to the child, so the sandbox needs no write access to any root-owned path. `root:admin 0640` — readable by whoever can manage the mode, not by every local account. |
| `…/secure-server/sandbox/server.pid` | The spawned server's own pid, recorded only once the spawn is confirmed, and validated (alive, owned by the sandbox uid) before teardown signals it. Identity is this pid and never a uid scan: the account also owns launchd's per-user agents (`distnoted`, `lsd`, `cfprefsd`, …). |
| `…/secure-server/sandbox/server.port` | The port the helper confirmed the child actually bound, so the app can adopt the right listener after a relaunch instead of reporting a pid that never owned it. |

`status` is measured, not remembered: enabled flag, `pfctl -s info`, and whether
the anchor holds uid-scoped rules. The UI shows **ON** only when all three hold
and **NOT ENFORCED** when the flag says on but pf is not applying it.

`OMLX_SECURE_MODE_STATE_PATH` redirects that record and exists as a test seam.
The app and the helper honour it **only in Debug builds**, so in release a stray
variable cannot make the spawn decision answer "off" while the record says on.
The Python reader (`omlx/secure_mode.py`) honours it unconditionally, including in
release, because the server has to be testable — so a stray value can still
silence the `403` egress refusal and the upload skip there. That is the second
line of defence only: pf holds the jail regardless, and the app's measured status
does not read it.

## What the sandboxed server can touch

- **Read**: anything world-readable, plus `list,search` ACL entries on the ancestors
  of the paths the app names (interpreter, package, models), and on any
  subdirectory underneath them that is still closed — `~/.omlx` may be open while
  `~/.omlx/models/<repo>` is not. Paths are resolved before anything is decided,
  because `exec` follows a symlink and the chain that actually matters is the one
  behind it (`~/.local` for a uv interpreter). An ancestor the requesting user does
  not own is never opened: that ownership gate is what stops "root, open this
  root-owned file for `_omlx`" from being a local privilege escalation, since the
  app names the paths and the helper runs as root. The gate is judged on the inode
  reached, and the inode is confirmed again after the entry is written, so a path
  swapped in between gets the entry taken back rather than left in place.
- **Write**: its own home (`sandbox/`), plus `settings.json`, `model_settings.json`
  and the `*.db` files in the base directory, with `add_file,delete_child` on that
  one directory — an atomic save (`tmp` + `rename`) was measured to need it. It
  cannot create new directories anywhere. Normally the server does *not* write your
  `settings.json`: it saves into the seeded copy above, which is why the file the app
  reads stays yours. The grant itself is kept because that directory entry already
  allows replacing any top-level entry of `~/.omlx`, and a recorded grant is what
  turns such a replace into something teardown can see and hand back. Logs and the
  SSD cache are redirected to the sandbox home with `OMLX_LOG_DIR` and
  `OMLX_SSD_CACHE_DIR`, so the bulk of what the server writes never goes near your
  files.
- **Back**: turning the mode off removes every recorded ACL entry and hands back
  anything the sandbox ended up owning. A replace leaves the writer as owner, so
  without this `model_settings.json` would belong to `_omlx` afterwards. A save the
  sandbox made to its settings copy is written back to your file — only when your
  copy is still exactly what was seeded, so an edit the app made while the server
  ran is never overwritten by an older one.

## Behaviour

- **On**: create the account, flush → load → verify → `pfctl -E`, record the
  token. Any failure rolls the state file back and releases the reference.
- **Flip restarts the server.** Whether a server may reach the network is decided
  by the uid owning its sockets, which is fixed at spawn: the running server is
  the wrong kind of process after a change, so it is restarted rather than left
  behind. If the helper cannot be reached while the mode is on, the app reports a
  failed start instead of spawning a server as your own user. Two details make the
  replacement reliable: both launch paths wait for the port to be free after stopping
  the previous server (spawning into its closing socket makes the new one exit with
  `Address already in use`, which the app can only read as "the server died"), and the
  app adopts a server the helper already has, so relaunching during the mode does not
  leave one running that nothing can stop.
- **"Port in use" is not reported while the mode is on.** A listener on the port then
  *is* the sandboxed server, so the app attaches to it — decided before the health
  probe, which blocks for seconds and answers "not oMLX" while the server boots. The
  port is probed by attempting the same `bind` the child performs (including
  `SO_REUSEADDR`), because a stricter probe reports a busy port for the whole
  `TIME_WAIT` window after the previous server stopped.
- **The helper does not report a pid until the server has bound.** It waits until
  the process it spawned is the one that owns the listening socket; uvicorn exits
  with `STARTUP_FAILURE` (3) on `Address already in use`, and reporting that dead
  pid made the app burn its auto-restart budget after every flip. The helper clears
  whatever holds the port — found with an in-process scan of listening sockets, at
  any uid, since a child of the user-owned server inherits uvicorn's listening
  socket before it forks — and retries the spawn itself. No bind probe runs alongside
  the child: binding the port to test it competes with the child's own bind.
- **Off**: stop the server and its process tree — captured *before* the parent is
  signalled, because a worker that outlives it gets reparented to launchd and keeps
  a listening socket — then flush the anchor, release the reference, revoke the
  ACLs, restore ownership. Not `pkill -u _omlx`: the account owns per-user launchd
  agents (`distnoted`, `lsd`, `cfprefsd`) that were never ours to signal, and
  launchd keeps restarting them, so an account-wide sweep never reports success.
- **Which process is "the server"**: the pid the daemon records for the spawn,
  validated as alive and owned by the sandbox account. Same reasoning — the process
  table cannot tell the server apart from the account's other processes, and a
  recycled pid must never be signalled.
- **Reboot**: the daemon is bootstrapped on every boot and re-applies everything
  when the state file says enabled, so a restart is not a password-free way out.
- Model downloads, Hugging Face/ModelScope search and model-info lookups, preset
  refresh, web search/fetch, update checks, uploads and oQ quantization starts
  are refused with `403` while the mode is on. Inference, health and task polling
  keep working.
- Benchmark and accuracy results are not uploaded while the mode is on; the run
  records the skip reason `secure_server` instead.

## Limits

- The pf rules cover TCP and UDP. Other protocols are not filtered; the server
  does not use them, and an unqualified rule would have to be dropped to cover
  them, which is the damage this design avoids.
- While the mode is on, the server is unreachable from other devices, including
  a phone on the same Wi-Fi. That is the point of the mode, not an oversight.
- Someone with a terminal can run `sudo pfctl -d`, which stops pf applying
  anything. The status pill then reads **NOT ENFORCED** rather than claiming
  protection, and the server's own 403 list is the remaining barrier.
- A sandboxed server is still a server: it can read anything world-readable and
  anything the ACL entries above cover. It is a network jail, not a defence
  against the server itself being malicious with your files.
- Granting is done with `chmod +a`, which resolves the account name through
  Open Directory and therefore takes a path, not a descriptor. The helper decides
  on the inode it opened (owner, mode) and re-checks the inode afterwards, but a
  user who owns every component of a path could, in principle, swap one between
  the check and the write and leave an entry on a file of root's. The entry would
  be unrecorded and so would not be revoked. What that buys an attacker is small:
  the program executed as the sandbox account is the interpreter the app resolved
  (the `OMLX_PYTHON_OVERRIDE` development escape hatch is only honoured by an
  unsigned build), the sandbox has no network either way, and the only reader is
  the oMLX server itself. Closing it properly means applying ACLs through a file
  descriptor, which macOS's ACL API does not offer for a uid qualifier without
  constructing the account's GUID by hand from Open Directory.

## Verifying

```sh
# Rules loaded, and scoped: names are resolved at load, so expect `user = <uid>`
sudo pfctl -a com.apple/900.omlx.secure -sr
sudo pfctl -a com.apple/900.omlx.secure -vsr      # counters: blocks are counting

# The server runs as the sandbox account
id _omlx
pgrep -u _omlx -l

# Its egress is dropped; yours is not
sudo -u _omlx curl --max-time 5 -o /dev/null -w '%{http_code}\n' https://example.com
curl -sS -o /dev/null -w '%{http_code}\n' https://example.com     # 200, unaffected

# Local traffic works
curl http://127.0.0.1:8000/health

# What has been opened up, and to whom
sudo cat "/Library/Application Support/oMLX/secure-server/access.json"
ls -le ~/.omlx ~/.local

# Only root can change the state
echo '{"enabled":false}' > "/Library/Application Support/oMLX/secure-server/state.json"   # Permission denied
```

## Recovery

To stop enforcement immediately from a terminal:

```sh
sudo pfctl -a com.apple/900.omlx.secure -F all   # drop the rules: connectivity back
sudo pfctl -d                                    # or stop pf entirely
```

Then turn the mode off in the app when the helper is reachable, so the state file
stops claiming the mode is on and the ACL entries are revoked.

Do not re-enable pf by hand with `sudo pfctl -E`: the helper holds the reference
token from the `-E` it performed, records it in `state.json`, and releases exactly
that token on the way out. A hand-taken reference makes the recorded one stale, so
turning the mode off can no longer release it and pf stays enabled with nobody
owning it. Flip the mode off and back on in the app instead — that reloads the
rules under the same flush→load→verify→enable sequence and takes a fresh reference.

If the helper is gone for good:

```sh
# Remove the ACL entries the helper recorded, one by one
sudo cat "/Library/Application Support/oMLX/secure-server/access.json"
sudo chmod -a "_omlx allow list,search" <path>            # directories
sudo chmod -a "_omlx allow read,execute" <path>           # files
sudo chmod -a "_omlx allow list,search,add_file,delete_child" <path>   # writable directories
sudo chmod -a "_omlx allow read,write,execute" <path>     # writable files

sudo dscl . -delete /Users/_omlx
sudo rm -rf "/Library/Application Support/oMLX/secure-server"
```
