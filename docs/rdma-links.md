# RDMA stage links over MCDMA

Status: experimental, off unless MCDMA's link daemon is running

In a mixed Mac and CUDA deployment every pipeline activation normally crosses
MLX's TCP Ring, and the Mac's side of that Ring is usually 10 GbE. A Mac with a
ConnectX card driven by [MCDMA](https://github.com/ashhart/MCDMA) has an RDMA
path to the CUDA workers instead. When such paths are present and proven, oMLX
sends every stage activation over them, including between CUDA workers that
have their own RDMA links, and each decode step's sampled tokens as well. The
Ring stays as the control channel: per-request votes and batch changes use it,
and any edge without a proven link falls back to it.

oMLX never opens an RDMA device. MCDMA's `mcdma-rpcd` daemons own the queue
pairs on both hosts, and oMLX exchanges bytes with them through shared-memory
mailboxes. A Python process that crashes or is killed therefore never leaves a
queue pair behind.

## What you need

- MCDMA installed on the coordinator Mac, with a ConnectX link to the CUDA
  worker that will be rank 1.
- `mcdma-rpcd` from the same MCDMA release on both ends of every edge. Rank `r`
  receives from rank `r + 1`, so the receiving host runs `connect` mode with a
  peer entry for the sender, and the sending host runs `listen` mode. For the
  Mac's edge that is `connect` on the Mac and `listen` on rank 1; for an edge
  between workers it is `connect` on rank `r` and `listen` on rank `r + 1`.
  Mailboxes are owner-only, so the Mac's daemon runs as the user running oMLX,
  and a worker's daemons run as the enrolled SSH user (or as root with
  `--owner` set to that user).
- A distinct link name for every edge of a deployment.
- `libmcdma-rpc` on every host. oMLX looks in `/usr/local/lib` and `/usr/lib`,
  or at the path in `OMLX_MCDMA_RPC_LIBRARY`.
- Every worker enrolled in the Cluster dashboard. A daemon's peer host must
  match exactly one enrolled worker by SSH target, address or hostname.

The daemon must speak mailbox protocol 1: its `STATUS` reply carries `host=`,
`device=`, `req_mib=`, `rep_mib=` and `since=` for every peer. A link whose
peer reports no `host=` cannot be tied to a worker, so oMLX lists it but never
routes traffic through it.

## How oMLX decides a link is live

A link carries activations only after three independent checks agree.

1. **Daemon status.** The receiving host's daemon must report the link up, and
   the link must resolve to the worker that holds the sending rank. oMLX reads
   the Mac's daemon directly and a worker's daemon over the cluster's SSH
   policy.
2. **Byte-checked probe.** oMLX starts a short-lived probe service on the
   sending worker over SSH, with the same SSH target and Python that the launch
   will use for that rank. The probe client runs at the receiving end, on the
   Mac or over SSH on the receiving worker. It sends content-checked round
   trips, a bulk transfer that must come back with a matching CRC-32, and asks
   for a bulk transfer that must match a seeded pattern. One wrong byte fails
   the probe. The **Verify** button runs the full probe on the Mac's links;
   every launch runs a quick one on every edge, all edges at once.
3. **Rank agreement.** After loading, every rank attaches its end of the
   mailbox and votes. An edge uses RDMA only when both ends attached and both
   daemons report the link up; a rank that cannot load the helper or reach its
   daemon votes no and the edge stays on the Ring. On a live edge, the model's
   `send`, `recv_like` and `recv` calls to that neighbour all go through the
   mailbox.

If a check fails for an edge, that edge goes over the Ring and the rest keep
their links. The deployment's cluster status reports every edge, its evidence
and the reason for each decision under `stage_links`. While
serving, a rank waiting on the link rechecks the daemon's link flag, the link
generation and its own service registration at least once a second. A
reconnect loses the request in flight, so the daemons bump the generation and
end the service registration when it happens, and both ranks fail the wait at
once. A link that drops stops the deployment with an error instead of hanging
it, and the next launch re-verifies and falls back to the Ring if the link is
still down.

## Sampled tokens

With rank-zero sampling, rank 0 picks each decode step's tokens and every other
rank needs them before the next step. When every stage edge of the deployment
is live, they travel up the pipeline instead of through the Ring's all-sum:
rank 0 puts them in its request for the next activation, and each worker passes
them on in its own request to the rank after it. A decode step then touches the
Ring not at all, and the tokens cost one RDMA write per rank. If any edge is on
the Ring, or rank-zero sampling is off, the tokens stay on the Ring too. Cluster
status reports the choice under `stage_links.token_relay`.

## Evidence

Each probe result for the Mac's links is kept in `cluster/rdma-links.json`
under the oMLX base directory, readable by the owner only. A result stops
counting as verified after 24 hours, or as soon as any of these change: the
daemon version, the peer host or node, the RDMA device, the mailbox sizes, the
time the link came up, or the loaded MCDMA driver's version and UUID.

## Dashboard and API

The Cluster dashboard shows an **RDMA links** card whenever the Mac's daemon
answers. Each row lists the link, the worker it reaches, whether it is verified,
the measured round-trip latency and throughput in each direction, and the
deployment currently using it. A running deployment's card adds an **RDMA
transport** panel: every stage hop with its link, whether it runs over MCDMA or
the Ring and why, and whether the sampled tokens travel over MCDMA.

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/admin/api/cluster/rdma-links` | Daemon state, helper state and every link with its evidence |
| `POST` | `/admin/api/cluster/rdma-links/verify` | Run the full probe on `{"link": "NAME"}` and record it |

Verification refuses a link that is carrying a deployment, because the probe
would take the worker's end of the mailbox away from the running rank.

## Settings

| Variable | Effect |
| --- | --- |
| `OMLX_RDMA_STAGE_LINKS=0` | Never route stage activations over RDMA |
| `OMLX_MCDMA_RPCD_SOCKET` | Control socket of a host's connect daemon (default `/tmp/mcdma-rpcd.sock`) |
| `OMLX_MCDMA_RPC_LIBRARY` | Path to `libmcdma-rpc` |

## Limits

- Pipeline deployments only. Tensor-parallel and JACCL deployments keep MLX's
  own transport.
- One deployment per link at a time.
- Per-request votes, batch membership and prompts still cross the Ring and the
  control channel. They run once per request or every few dozen steps, not per
  token.
- Received frames of 32 MiB or more go from the mailbox into MLX by GPU copy
  when MCDMA's helper offers Metal buffers; the NIC writes the mailbox and the
  GPU reads it, with no CPU copy. Smaller frames, where a CPU copy is quicker
  than a GPU round trip, are copied once by the CPU. Sending ranks copy their
  activation into the mailbox.

## Operating the daemons

oMLX only reads the daemons' status; it never starts, stops or signals them.
Stop them the way MCDMA documents: the Mac's daemon first, with its `SHUTDOWN`
command rather than a signal, and only then any worker's daemon.

## Mailbox protocol 1

For implementers of other daemons. Each link has one mailbox: a request half
of `R` bytes followed by a reply half of `P` bytes, each starting with a 4 KiB
control page. A word is `seq << 32 | length`, and sequence 0 means empty.

| Offset | End | Meaning |
| --- | --- | --- |
| request +0 | both | Request word: staged by the client, landed at the service |
| request +64 | client | 1 while the daemon's link to the peer is up |
| request +72 | client | Link generation: bumped each time the daemon's link comes up |
| request +256 | both | `R` and `P` as two little-endian u64 values |
| reply +0 | service | Ready word: the daemon took the staged reply to send it |
| reply +64 | client | Reply word: the service's reply has landed |
| reply +128 | service | Staged word: a reply is ready for the daemon to send |

The connect side is the POSIX shared memory object `/mcdma-rpc.NAME`; the listen
side is the file `/dev/shm/mcdma-rpc.NAME`. Payloads are written before their
word, words are stored with release ordering and read with acquire ordering,
which `libmcdma-rpc` provides as `mcdma_rpc_store_word` and
`mcdma_rpc_wait_word`.

A service registers by connecting to the listen daemon's Unix socket (by
default `/tmp/mcdma-rpcd.NAME.sock`) and sending `MODE poll`. The daemon answers
`OK`, or `ERR busy` if another service holds the link. After `OK` it sends
nothing more until the registration ends, then `BYE` or a closed socket, so any
byte on that socket means the service has lost the link. The registration ends
when the link drops, since requests in flight are lost with it. A service that
stops after a reply keeps its registration until the ready word carries that
reply's sequence, because the daemon drops a staged reply once its service is
gone.

The connect daemon's control socket answers `STATUS` with an optional
`VERSION mcdma-rpcd 1 RELEASE` line, one line per peer and a final `END`:

```text
PEER NAME up|down calls N failures N MiB N host=HOST port=PORT device=DEVICE req_mib=R rep_mib=P since=EPOCH
```
