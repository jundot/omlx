# SPDX-License-Identifier: Apache-2.0
"""Serve link probe requests on the listening host of one link, then exit."""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
import zlib
from collections.abc import Callable
from typing import Any

from . import probe_wire
from .mailbox import MailboxError, ServiceMailbox
from .words import load_word_ops

_BYE_SENT_TIMEOUT_S = 5.0


def _stdin_closed() -> bool:
    """Whether stdin reached end of file, which is how the coordinator's SSH client going away shows up here."""
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(readable) and not os.read(sys.stdin.fileno(), 1)
    except (OSError, ValueError):
        return True


def serve_probe(
    mailbox: ServiceMailbox,
    *,
    idle_timeout_s: float = 30.0,
    deadline_s: float = 300.0,
    clock: Callable[[], float] = time.monotonic,
    abandoned: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Answer probe requests until END, an idle gap, a deadline, a detached daemon or an abandoned probe."""
    started = clock()
    last = started
    served = {"echo": 0, "sink": 0, "source": 0}
    while True:
        # A coordinator that gave up must not leave this service holding the link.
        if abandoned():
            return {"ended": "abandoned", "served": served}
        now = clock()
        if now - started > deadline_s:
            return {"ended": "deadline", "served": served}
        if now - last > idle_timeout_s:
            return {"ended": "idle", "served": served}
        if not mailbox.alive:
            return {"ended": "daemon detached", "served": served}
        got = mailbox.next_request(timeout_s=0.5)
        if got is None:
            continue
        last = clock()
        seq, payload = got
        request = probe_wire.unpack_request(payload)
        body = payload[probe_wire.HEADER_BYTES :]
        if request.kind == probe_wire.ECHO:
            mailbox.reply(seq, (bytes(body)[::-1],))
            served["echo"] += 1
        elif request.kind == probe_wire.SINK:
            mailbox.reply(
                seq, (probe_wire.SINK_REPLY.pack(zlib.crc32(body), len(body)),)
            )
            served["sink"] += 1
        elif request.kind == probe_wire.SOURCE:
            if request.reply_bytes > mailbox.sizes.max_reply:
                raise probe_wire.ProbeError(
                    f"asked for {request.reply_bytes} bytes; "
                    f"the reply half holds {mailbox.sizes.max_reply}"
                )
            mailbox.reply(seq, (probe_wire.pattern(request.seed, request.reply_bytes),))
            served["source"] += 1
        else:
            mailbox.reply(seq, (probe_wire.BYE,))
            # mcdma-rpcd drops a staged reply once its service detaches, so stay until BYE is taken.
            if not mailbox.wait_sent(seq, timeout_s=_BYE_SENT_TIMEOUT_S):
                return {"ended": "bye not sent", "served": served}
            return {"ended": "end", "served": served}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument(
        "--mailbox", default=None, help="mailbox file; defaults to the daemon's"
    )
    parser.add_argument("--idle-timeout", type=float, default=30.0)
    parser.add_argument("--deadline", type=float, default=300.0)
    args = parser.parse_args(argv)
    ops, reason = load_word_ops()
    if ops is None:
        print(json.dumps({"ready": False, "error": reason}), flush=True)
        return 2
    try:
        mailbox = ServiceMailbox.attach(
            args.name, args.socket, ops, mailbox_path=args.mailbox
        )
    except (MailboxError, ValueError) as exc:
        print(json.dumps({"ready": False, "error": str(exc)}), flush=True)
        return 2
    try:
        # The coordinator starts probing only after this line.
        print(
            json.dumps(
                {
                    "ready": True,
                    "max_request": mailbox.sizes.max_request,
                    "max_reply": mailbox.sizes.max_reply,
                }
            ),
            flush=True,
        )
        summary = serve_probe(
            mailbox,
            idle_timeout_s=args.idle_timeout,
            deadline_s=args.deadline,
            abandoned=_stdin_closed,
        )
    except (MailboxError, probe_wire.ProbeError) as exc:
        summary = {"ended": "error", "error": str(exc)}
    finally:
        mailbox.close()
    print(json.dumps(summary), flush=True)
    return 0 if summary.get("ended") == "end" else 1


if __name__ == "__main__":
    sys.exit(main())
