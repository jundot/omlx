# RDMA stage links over MCDMA

Status: experimental, off unless MCDMA's link daemon is running

In a mixed Mac and CUDA deployment every pipeline activation normally crosses
MLX's TCP Ring, and the Mac's side of that Ring is usually 10 GbE. A Mac with a
ConnectX card driven by [MCDMA](https://github.com/ashhart/MCDMA) has an RDMA
path to the CUDA workers instead. When that path is present and proven, oMLX
sends the stage activation from rank 1 to rank 0 over it. Everything else stays
on the Ring: the token all-sum, the final all-gather, and every other stage edge.

oMLX never opens an RDMA device. MCDMA's `mcdma-rpcd` daemons own the queue
pairs on both hosts, and oMLX exchanges bytes with them through shared-memory
mailboxes. A Python process that crashes or is killed therefore never leaves a
queue pair behind.

## What you need

- MCDMA installed on the coordinator Mac, with a ConnectX link to the CUDA
  worker that will be rank 1.
- `mcdma-rpcd` from the same MCDMA release on both hosts: `connect` mode on the
  Mac and `listen` mode on the worker, one peer entry per worker. Mailboxes are
  owner-only, so the Mac's daemon runs as the user running oMLX, and the
  worker's daemon runs as the enrolled SSH user (or as root with `--owner` set
  to that user).
- `libmcdma-rpc` on both hosts. oMLX looks in `/usr/local/lib` and `/usr/lib`,
  or at the path in `OMLX_MCDMA_RPC_LIBRARY`.
- The worker enrolled in the Cluster dashboard. The daemon's peer host must
  match exactly one enrolled worker by SSH target, address or hostname.

The daemon must speak mailbox protocol 1: its `STATUS` reply carries `host=`,
`device=`, `req_mib=`, `rep_mib=` and `since=` for every peer. A link whose
peer reports no `host=` cannot be tied to a worker, so oMLX lists it but never
routes traffic through it.

## How oMLX decides a link is live

A link carries activations only after three independent checks agree.

1. **Daemon status.** The Mac's daemon must report the link up, and the link
   must resolve to the worker that holds rank 1.
2. **Byte-checked probe.** oMLX starts a short-lived probe service on the
   worker over the cluster's SSH policy, with the same SSH target and Python
   that the launch will use for rank 1, then sends content-checked round
   trips, a bulk transfer to the worker that must come back with a matching
   CRC-32, and a bulk transfer from the worker that must match a seeded
   pattern. One wrong byte fails the probe. The **Verify** button runs the full
   probe; every launch runs a quick one first.
3. **Rank agreement.** After loading, every rank attaches its end of the
   mailbox and votes. An edge uses RDMA only when both ends attached and both
   daemons report the link up; a rank that cannot load the helper or reach its
   daemon votes no and the edge stays on the Ring.

If any check fails, the launch goes ahead over the Ring, and the deployment's
cluster status reports the decision and its reason under `stage_links`. While
serving, a rank waiting on the link rechecks the daemon's link flag, the link
generation and its own service registration at least once a second. A reconnect
loses the request in flight, so the daemons bump the generation and end the
service registration when it happens, and both ranks fail the wait at once. A link that drops stops
the deployment with an error instead of hanging it, and the next launch
re-verifies and falls back to the Ring if the link is still down.

## Evidence

Each probe result is kept in `cluster/rdma-links.json` under the oMLX base
directory, readable by the owner only. A result stops counting as verified
after 24 hours, or as soon as any of these change: the daemon version, the peer
host or node, the RDMA device, the mailbox sizes, the time the link came up, or
the loaded MCDMA driver's version and UUID.

## Dashboard and API

The Cluster dashboard shows an **RDMA links** card whenever the Mac's daemon
answers. Each row lists the link, the worker it reaches, whether it is verified,
the measured round-trip latency and throughput in each direction, and the
deployment currently using it.

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
| `OMLX_MCDMA_RPCD_SOCKET` | Control socket of the Mac's daemon (default `/tmp/mcdma-rpcd.sock`) |
| `OMLX_MCDMA_RPC_LIBRARY` | Path to `libmcdma-rpc` |

## Limits

- Only the rank 1 to rank 0 edge is carried. With the Mac as rank 0 this is the
  one stage edge between the Mac and a CUDA worker.
- Pipeline deployments only. Tensor-parallel and JACCL deployments keep MLX's
  own transport.
- One deployment per link at a time.
- Prefill gains the most, because prompt activations are large. A decode step
  moves one small activation and still waits on the Ring's token all-sum, so
  decode speed changes little.

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
when the link drops, since requests in flight are lost with it.

The connect daemon's control socket answers `STATUS` with an optional
`VERSION mcdma-rpcd 1 RELEASE` line, one line per peer and a final `END`:

```text
PEER NAME up|down calls N failures N MiB N host=HOST port=PORT device=DEVICE req_mib=R rep_mib=P since=EPOCH
```
