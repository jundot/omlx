# SPDX-License-Identifier: Apache-2.0
"""``omlx-cuda-helper``: run the helper daemon, or inspect its pairings."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from . import __version__, families, gpus
from .control import ControlServer
from .discovery import Discovery
from .files import ModelStore
from .pairing import Joiner, PairingError
from .state import State
from .supervisor import Supervisor

logger = logging.getLogger("omlx_cuda_helper")


def _indices(value: str | None) -> list[int] | None:
    return [int(part) for part in value.split(",")] if value else None


def _host_port(value: str, default: int) -> tuple[str, int]:
    host, _, port = value.rpartition(":")
    return (host, int(port)) if host else (value, default)


def serve(args: argparse.Namespace) -> int:
    state = State(args.state_dir, name=args.name)
    models = Path(args.models_dir).expanduser() if args.models_dir else state.models_dir
    models.mkdir(parents=True, exist_ok=True)
    only = _indices(args.gpus)
    found = gpus.gpus(only)
    if not found:
        logger.error("no NVIDIA GPU found (nvidia-smi reports none)")
        return 1

    def caps() -> dict[str, Any]:
        current = gpus.gpus(only) or found
        return {
            "chip": gpus.chip_label(current),
            # Whole GiB: the oMLX app reads this as an integer.
            "ram_gb": float(round(sum(gpu["total"] for gpu in current) / 2**30)),
            "backends": ["cuda"],
            "thunderbolt": False,
            "jaccl": False,
            "loopback_only": False,
            "control": "agent",
        }

    supervisor = Supervisor(only, state.root / "run", data_port=args.data_port)
    control = ControlServer(
        state,
        ModelStore(models),
        supervisor,
        families.contracts(),
        gpu_indices=only,
        bind=args.bind,
        port=args.control_port,
    )
    joiner: Joiner

    def on_invite(node_id: str, address: str, port: int | None) -> None:
        if port is None:
            logger.info(
                "PAIRING invited by %s at %s without a port; update that oMLX "
                "or start this helper with --pair HOST:PORT",
                node_id[:8],
                address,
            )
            return
        logger.info("PAIRING invited by %s at %s:%s", node_id[:8], address, port)
        try:
            joiner.join(address, port, node_id)
        except PairingError as error:
            logger.error("PAIRING failed: %s", error)

    discovery = Discovery(
        state,
        caps=caps,
        addresses=gpus.local_ipv4,
        control_port=lambda: control.port,
        on_invite=on_invite,
        cluster_name=args.cluster_name,
        bind=args.bind,
        http_port=args.http_port,
    )
    joiner = Joiner(state, caps, discovery.http_port)
    control.start()
    discovery.start(mdns=not args.no_mdns)
    hosts = state.hosts()
    logger.info(
        "omlx-cuda-helper %s: %s as '%s' (%s), http %s, control %s, %s",
        __version__,
        gpus.chip_label(found),
        state.name,
        state.node_id[:8],
        discovery.http_port,
        control.port,
        (
            f"paired with {', '.join(h['name'] for h in hosts.values())}"
            if hosts
            else "not paired: press Pair on it in the oMLX Cluster tab"
        ),
    )
    if not families.FAMILIES:
        logger.warning("this build serves no model family yet")
    if args.pair:
        try:
            joiner.join(*_host_port(args.pair, 8000))
        except PairingError as error:
            logger.error("PAIRING failed: %s", error)

    stop = threading.Event()
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, lambda *_: stop.set())
    stop.wait()
    joiner.cancel()
    supervisor.stop()
    discovery.stop()
    control.stop()
    return 0


def status(args: argparse.Namespace) -> int:
    state = State(args.state_dir)
    print(f"{state.name} ({state.node_id})")
    for node_id, host in state.hosts().items():
        print(f"  paired with {host.get('name')} ({node_id}) at {host.get('addrs')}")
    return 0


def unpair(args: argparse.Namespace) -> int:
    removed = State(args.state_dir).remove_host(args.node_id)
    print("removed" if removed else "no such host")
    return 0 if removed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omlx-cuda-helper")
    parser.add_argument("--state-dir", default="~/.omlx-cuda-helper")
    parser.add_argument("--log-level", default="info")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("serve", help="run the helper daemon")
    run.add_argument("--models-dir", help="default: <state-dir>/models")
    run.add_argument("--name", help="name shown in the oMLX Cluster tab")
    run.add_argument("--cluster-name", default="omlx")
    run.add_argument("--bind", default="", help="address to listen on (default: all)")
    run.add_argument("--http-port", type=int, default=8850)
    run.add_argument("--control-port", type=int, default=8851)
    run.add_argument("--data-port", type=int, default=8852)
    run.add_argument("--gpus", help="GPU indices to use, e.g. 0,1 (default: all)")
    run.add_argument("--no-mdns", action="store_true")
    run.add_argument("--pair", metavar="HOST[:PORT]", help="join this oMLX at start")
    run.set_defaults(run=serve)

    commands.add_parser("status", help="show the pairings").set_defaults(run=status)
    remove = commands.add_parser("unpair", help="forget a paired host")
    remove.add_argument("node_id")
    remove.set_defaults(run=unpair)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
