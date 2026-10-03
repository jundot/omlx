# SPDX-License-Identifier: Apache-2.0
"""On a worker at the connect end of a link, print its daemon status or probe the link, as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .daemon import read_status
from .mailbox import ClientMailbox, MailboxError
from .probe_run import ProbeSettings, probe_link
from .probe_wire import ProbeError
from .words import load_word_ops


def run_probe(name: str, settings: ProbeSettings) -> dict[str, Any]:
    """Probe link `name` from this host; the service must already be running on the peer."""
    ops, reason = load_word_ops()
    if ops is None:
        return {"ok": False, "error": reason}
    try:
        mailbox = ClientMailbox.attach(name, ops)
    except (MailboxError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    try:
        if not mailbox.connected:
            return {"ok": False, "error": "mcdma-rpcd reports this link down"}
        return {"ok": True, "measurements": probe_link(mailbox, settings).to_dict()}
    except (ProbeError, MailboxError) as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        mailbox.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="the connect daemon's peers")
    status.add_argument("--socket", default=None)
    probe = commands.add_parser("probe", help="probe one link end to end")
    probe.add_argument("--name", required=True)
    probe.add_argument("--warmup", type=int, default=ProbeSettings.warmup)
    probe.add_argument("--round-trips", type=int, default=ProbeSettings.round_trips)
    probe.add_argument("--bulk-bytes", type=int, default=ProbeSettings.bulk_bytes)
    probe.add_argument("--repeats", type=int, default=ProbeSettings.repeats)
    probe.add_argument(
        "--call-timeout", type=float, default=ProbeSettings.call_timeout_s
    )
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(read_status(args.socket).to_dict()), flush=True)
        return 0
    result = run_probe(
        args.name,
        ProbeSettings(
            warmup=args.warmup,
            round_trips=args.round_trips,
            bulk_bytes=args.bulk_bytes,
            repeats=args.repeats,
            call_timeout_s=args.call_timeout,
        ),
    )
    print(json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
