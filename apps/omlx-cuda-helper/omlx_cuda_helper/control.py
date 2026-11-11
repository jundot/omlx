# SPDX-License-Identifier: Apache-2.0
"""The control channel: a paired host asks, the helper answers.

The connection that starts a prefill session owns it: when that connection
ends, the GPU workers stop and their memory is free.
"""

from __future__ import annotations

import logging
import select
import socket
import threading
from contextlib import suppress
from typing import Any

from . import CONTROL_VERSION, DATA_PROTOCOL, __version__, gpus, wire
from .files import FileError, ModelStore
from .state import State

logger = logging.getLogger("omlx_cuda_helper.control")

# One pushed chunk plus slack for its header.
MAX_CHUNK = 32 << 20


def _closed(sock: socket.socket) -> bool:
    """Whether the host is gone; it sends nothing while it waits for a reply."""
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable) and not sock.recv(1, socket.MSG_PEEK)
    except (OSError, ValueError):
        return True


class ControlServer:
    def __init__(
        self,
        state: State,
        store: ModelStore,
        supervisor: Any,
        contracts: dict[str, int],
        *,
        gpu_indices: list[int] | None = None,
        bind: str = "",
        port: int = 0,
    ):
        self.state, self.store, self.supervisor = state, store, supervisor
        self.contracts, self.gpu_indices = contracts, gpu_indices
        self._listeners = [socket.create_server((bind, port), reuse_port=False)]
        self.port: int = self._listeners[0].getsockname()[1]
        if not bind and socket.has_ipv6:
            # A direct cable without DHCP reaches this machine over IPv6
            # link-local only.
            with suppress(OSError):
                self._listeners.append(
                    socket.create_server(("::", self.port), family=socket.AF_INET6)
                )

    def start(self) -> None:
        for listener in self._listeners:
            threading.Thread(
                target=self._accept,
                args=(listener,),
                name="omlx-helper-control",
                daemon=True,
            ).start()

    def stop(self) -> None:
        for listener in self._listeners:
            listener.close()

    def _accept(self, listener: socket.socket) -> None:
        while True:
            try:
                sock, peer = listener.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve,
                args=(sock, peer),
                name="omlx-helper-host",
                daemon=True,
            ).start()

    def _serve(self, sock: socket.socket, peer: tuple) -> None:
        owner = object()
        store = ModelStore(self.store.root)
        try:
            wire.tune(sock)
            sock.settimeout(10)
            # Nothing is trusted yet: a tiny header and no payload.
            meta, _ = wire.recv(sock, max_header=1024, max_bytes=0)
            node_id = str(meta.get("node_id") or "")
            key = self.state.cluster_key(node_id)
            if (
                meta.get("op") != "control"
                or meta.get("v") != CONTROL_VERSION
                or key is None
            ):
                return
            wire.authenticate(sock, wire.control_key(key), server=True)
            sock.settimeout(None)
            wire.send(
                sock,
                {
                    "op": "hello",
                    "v": CONTROL_VERSION,
                    "version": __version__,
                    "node_id": self.state.node_id,
                },
            )
            while True:
                meta, arrays = wire.recv(sock, max_bytes=MAX_CHUNK)
                try:
                    reply = self._handle(meta, arrays, owner, store, node_id, key, sock)
                # A full disk or a missing file is the helper's answer, not a
                # lost connection: _handle reads and writes no frame.
                except (
                    FileError,
                    OSError,
                    RuntimeError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as error:
                    reply = {"op": "error", "error": str(error)}
                wire.send(sock, reply)
        except (OSError, wire.ProtocolError):
            pass
        finally:
            store.close()
            self.supervisor.stop(owner)
            sock.close()

    def _handle(
        self,
        meta: dict[str, Any],
        arrays: dict[str, wire.HostArray],
        owner: object,
        store: ModelStore,
        node_id: str,
        key: bytes,
        sock: socket.socket,
    ) -> dict[str, Any]:
        op = meta.get("op")
        address = sock.getsockname()
        local = address[0]
        if sock.family == socket.AF_INET6 and address[3]:
            # The data port listens on the link this connection came in on.
            local = f"{local}%{socket.if_indextoname(address[3])}"
        running = self.supervisor.model_dir()
        if op == "ping":
            return {"op": "ok"}
        if op == "info":
            total, available = gpus.memory()
            return {
                "op": "ok",
                "version": __version__,
                "data_protocol": DATA_PROTOCOL,
                "contracts": self.contracts,
                "gpus": gpus.gpus(self.gpu_indices),
                "ram": total,
                "ram_available": available,
                "disk_free": gpus.disk_free(self.store.root),
                "link_mbps": gpus.link_mbps(local),
                "links": gpus.links(),
                "session": self.supervisor.status(),
            }
        if op == "probe":
            files = {str(k): int(v) for k, v in (meta.get("files") or {}).items()}
            return {
                "op": "ok",
                **store.probe(str(meta.get("name") or ""), files, running),
            }
        if op == "put":
            data = arrays["data"][0]
            store.write(
                str(meta["name"]),
                str(meta["file"]),
                int(meta["size"]),
                int(meta["offset"]),
                memoryview(data).cast("B"),
                int(meta["crc"]),
                bool(meta.get("last")),
            )
            return {"op": "ok"}
        if op == "delete":
            names = [str(n) for n in meta.get("names") or []]
            return {"op": "ok", "free": store.delete(names, running)}
        if op == "start":
            model_type = str(meta["model_type"])
            if self.contracts.get(model_type) != meta.get("contract"):
                raise RuntimeError(
                    f"this helper has contract {self.contracts.get(model_type)} for "
                    f"{model_type}, the host asked for {meta.get('contract')}"
                )
            ready = self.supervisor.start(
                owner,
                model_dir=str(store.folder(str(meta["model"]))),
                model_type=model_type,
                layers=int(meta["layers"]),
                warm_tokens=int(meta.get("warm_tokens") or 8192),
                host={"node_id": node_id, "name": str(meta.get("host_name") or "")},
                token=wire.data_token(key, bytes.fromhex(str(meta["session"]))),
                bind=local,
                cancelled=lambda: _closed(sock),
            )
            return {"op": "ready", **ready}
        if op == "stop":
            self.supervisor.stop(owner)
            return {"op": "ok"}
        raise RuntimeError(f"unknown op: {op!r}")
