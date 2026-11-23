# SPDX-License-Identifier: Apache-2.0
"""Be found by oMLX: the mDNS announcement and the two HTTP endpoints.

oMLX lists a device once ``GET /api/cluster/node_id`` at an announced address
answers with the announced id. Its Pair button then posts an invite here.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import __version__
from .state import State

logger = logging.getLogger("omlx_cuda_helper.discovery")

SERVICE_TYPE = "_omlx._tcp.local."
_MAX_BODY = 4096

Invite = Callable[[str, str, int | None], None]


class Discovery:
    def __init__(
        self,
        state: State,
        *,
        caps: Callable[[], dict[str, Any]],
        addresses: Callable[[], list[str]],
        control_port: Callable[[], int],
        on_invite: Invite,
        cluster_name: str = "omlx",
        bind: str = "",
        http_port: int = 0,
    ):
        self.state, self.caps, self.addresses = state, caps, addresses
        self.control_port, self.on_invite = control_port, on_invite
        self.cluster_name = cluster_name
        self._server = ThreadingHTTPServer((bind, http_port), self._handler())
        self._server.daemon_threads = True
        self.http_port: int = self._server.server_address[1]
        self._zeroconf: Any = None

    def identity(self) -> dict[str, Any]:
        return {
            "node_id": self.state.node_id,
            "version": __version__,
            "cluster_name": self.cluster_name,
            "friendly_name": self.state.name,
            "caps": self.caps(),
            "control_port": self.control_port(),
        }

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        discovery = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                if self.path == "/api/cluster/node_id":
                    self._reply(200, discovery.identity())
                else:
                    self._reply(404, {"detail": "not found"})

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                if self.path != "/api/cluster/pair/invited":
                    self._reply(404, {"detail": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if not 0 < length <= _MAX_BODY:
                        raise ValueError("bad length")
                    body = json.loads(self.rfile.read(length))
                    node_id = str(body["node_id"])
                    port = body.get("http_port")
                    port = int(port) if port is not None else None
                except (KeyError, TypeError, ValueError):
                    self._reply(422, {"detail": "malformed invite"})
                    return
                threading.Thread(
                    target=discovery.on_invite,
                    args=(node_id, self.client_address[0], port),
                    name="omlx-helper-invite",
                    daemon=True,
                ).start()
                self._reply(202, {"ok": True})

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("http %s", format % args)

        return Handler

    def start(self, *, mdns: bool = True) -> None:
        threading.Thread(
            target=self._server.serve_forever, name="omlx-helper-http", daemon=True
        ).start()
        if mdns:
            self._announce()

    def _announce(self) -> None:
        try:
            import zeroconf
        except ImportError:
            logger.warning("zeroconf is not installed; add this helper by address")
            return
        packed = [socket.inet_aton(ip) for ip in self.addresses()][:8]
        if not packed:
            logger.warning("no LAN address to announce; add this helper by address")
            return
        info = zeroconf.ServiceInfo(
            SERVICE_TYPE,
            f"{self.state.name}.{SERVICE_TYPE}"[:255],
            addresses=packed,
            port=self.http_port,
            properties={
                "id": self.state.node_id,
                "name": self.state.name,
                "ver": __version__,
                "cl": self.cluster_name,
                "caps": json.dumps(self.caps(), separators=(",", ":")),
                "port": str(self.http_port),
            },
        )
        try:
            self._zeroconf = zeroconf.Zeroconf()
            self._zeroconf.register_service(info)
        except OSError as error:
            # Docker without host networking cannot send multicast.
            logger.warning(
                "mDNS is not available (%s); add this helper by address", error
            )
            self._zeroconf = None

    def stop(self) -> None:
        if self._zeroconf is not None:
            self._zeroconf.close()
        self._server.shutdown()
        self._server.server_close()
