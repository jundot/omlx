# SPDX-License-Identifier: Apache-2.0
"""Request-level cluster router for multiple omlx servers.

A stateless reverse proxy that fans OpenAI-compatible requests out across N
omlx backends. It is NOT GPU/memory-level clustering and NOT a shared KV cache:
each backend stays an independent ``omlx serve``; this only decides WHICH
backend handles each request.

Design (see docs/ for the full write-up):

- The cluster is really a PARTITIONED MODEL FLEET, not a uniform server pool.
  Backends host different (overlapping) model sets. So routing is model-FIRST:
  a request can only go to a backend that hosts the model on disk; load is just
  the tiebreaker among eligible backends.

- Anti-thrash invariant: a backend loads models LRU within a fixed Metal
  ceiling, so sending a model's traffic to a backend where it is NOT resident
  forces a cold load + eviction. ``pick()`` therefore prefers backends where the
  model is already RESIDENT, which keeps a given model's traffic pinned to one
  backend and stops two machines from churning the same model in and out.

- Load is each backend's ``/api/status`` depth (active+waiting) PLUS a LOCAL
  in-flight counter, then NORMALIZED by a per-backend ``weight`` (relative
  compute capacity) so a faster machine is handed proportionally more traffic.
  The local counter is load-bearing for correctness: without it, every request
  inside one 1.5s poll window reads the same "least busy" snapshot and stampedes
  one backend. A soft MEMORY gate adds a load penalty to a backend whose model-
  memory headroom nears zero, steering new traffic away from an imminent
  evict-and-reload -- without ever hard-excluding it.

- Streaming (SSE) is byte-passthrough with ``timeout=None`` (generations can run
  minutes); a client disconnect cancels the upstream request so a dead client
  does not pin a backend slot. Failover only happens PRE-stream (connection
  failure before any bytes are sent) -- once SSE bytes flow, a transparent
  failover is impossible and the stream just errors.

Run:  ``python -m omlx.cluster.router``  (config from $OMLX_CLUSTER_CONFIG or
~/.omlx/cluster.json). Point OpenAI clients at ``http://<router>:9000/v1``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger("omlx.cluster.router")

# Methods/paths whose JSON body carries a "model" field we route on.
_MODEL_BODY_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/rerank",
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Backend:
    name: str
    base_url: str            # e.g. http://m5max:8000
    api_key: str             # the backend's real X-API-Key
    weight: float = 1.0      # relative compute capacity; raw load is divided by
                             # this, so a faster machine (higher weight) is given
                             # proportionally more traffic before it looks busy


@dataclass
class Config:
    host: str
    port: int
    router_api_key: str | None         # clients must present this (None = open)
    backends: list[Backend]
    poll_interval: float = 1.5
    affinity_hysteresis: float = 1.0   # stay on the sticky backend unless its
                                       # weight-normalized load exceeds the
                                       # least-busy candidate by more than this
    poll_timeout: float = 3.0
    # Soft memory gate: when a backend's model-memory headroom drops below
    # mem_soft_floor (fraction 0..1), add a load penalty that ramps up to
    # mem_penalty as headroom -> 0. mem_penalty is in LOAD units (~request
    # count), so 10 means "a near-OOM backend looks ~10 requests busier" --
    # strongly deprioritized but still reachable when every peer is busier still.
    mem_soft_floor: float = 0.05
    mem_penalty: float = 10.0
    # Optional alias -> real model id (clients may send display aliases while the
    # catalog is keyed by the real id from /v1/models/status).
    aliases: dict[str, str] = field(default_factory=dict)


def _config_path() -> Path:
    from ..settings import resolve_default_base_path

    return Path(
        os.environ.get("OMLX_CLUSTER_CONFIG")
        or (resolve_default_base_path() / "cluster.json")
    )


def load_config() -> Config:
    path = _config_path()
    if not path.exists():
        raise SystemExit(
            f"No cluster config at {path}. Set $OMLX_CLUSTER_CONFIG or create it; "
            f"see omlx/cluster/cluster.example.json for the format."
        )
    raw = json.loads(path.read_text())

    def _weight(b: dict) -> float:
        w = float(b.get("weight", 1.0))
        if w <= 0:
            raise SystemExit(
                f"backend {b.get('name')!r} has non-positive weight {w}; weight "
                f"is relative compute capacity and must be > 0"
            )
        return w

    backends = [
        Backend(name=b["name"], base_url=b["base_url"].rstrip("/"),
                api_key=b.get("api_key", ""), weight=_weight(b))
        for b in raw["backends"]
    ]
    if not backends:
        raise SystemExit("cluster config has no backends")
    listen = raw.get("listen", "0.0.0.0:9000")
    host, _, port = listen.partition(":")
    return Config(
        host=host or "0.0.0.0",
        port=int(port or 9000),
        router_api_key=raw.get("router_api_key") or None,
        backends=backends,
        poll_interval=float(raw.get("poll_interval", 1.5)),
        affinity_hysteresis=float(raw.get("affinity_hysteresis", 1.0)),
        poll_timeout=float(raw.get("poll_timeout", 3.0)),
        mem_soft_floor=float(raw.get("mem_soft_floor", 0.05)),
        mem_penalty=float(raw.get("mem_penalty", 10.0)),
        aliases=dict(raw.get("aliases", {})),
    )


# --------------------------------------------------------------------------- #
# Live snapshot of each backend
# --------------------------------------------------------------------------- #
@dataclass
class Snapshot:
    healthy: bool = False
    # model_id -> resident(bool). Presence of the key == backend hosts the model.
    catalog: dict[str, bool] = field(default_factory=dict)
    active: int = 0
    waiting: int = 0
    mem_used: int = 0
    mem_max: int = 0
    default_model: str | None = None
    last_ok: float = 0.0
    last_err: str = ""


class ClusterRouter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.snap: dict[str, Snapshot] = {b.name: Snapshot() for b in cfg.backends}
        self.inflight: dict[str, int] = {b.name: 0 for b in cfg.backends}
        self.affinity: dict[str, str] = {}      # model_id -> backend name
        self.backends: dict[str, Backend] = {b.name: b for b in cfg.backends}
        # Short timeout for polling; per-request streams override with timeout=None.
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(cfg.poll_timeout))
        self._poll_task: asyncio.Task | None = None

    # ---- lifecycle ----
    async def start(self) -> None:
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
        await self.client.aclose()

    # ---- background poll ----
    async def _poll_loop(self) -> None:
        while True:
            await asyncio.gather(
                *(self._poll_one(b) for b in self.cfg.backends),
                return_exceptions=True,
            )
            await asyncio.sleep(self.cfg.poll_interval)

    async def _poll_one(self, b: Backend) -> None:
        s = self.snap[b.name]
        headers = {"X-API-Key": b.api_key} if b.api_key else {}
        try:
            st = await self.client.get(b.base_url + "/api/status", headers=headers)
            st.raise_for_status()
            status = st.json()
            ms = await self.client.get(b.base_url + "/v1/models/status", headers=headers)
            ms.raise_for_status()
            models = ms.json()
        except Exception as exc:
            s.healthy = False
            s.last_err = f"{type(exc).__name__}: {exc}"
            return
        catalog: dict[str, bool] = {}
        for m in models.get("models", []):
            mid = m.get("id")
            if mid:
                catalog[mid] = bool(m.get("loaded"))
        s.catalog = catalog
        s.active = int(status.get("active_requests", 0) or 0)
        s.waiting = int(status.get("waiting_requests", 0) or 0)
        s.mem_used = int(status.get("model_memory_used", 0) or 0)
        s.mem_max = int(status.get("model_memory_max", 0) or 0)
        s.default_model = status.get("default_model")
        s.healthy = True
        s.last_ok = time.monotonic()
        s.last_err = ""

    # ---- routing ----
    def depth(self, name: str) -> int:
        """Raw queue depth: in-flight + waiting requests on the backend."""
        s = self.snap[name]
        return s.active + s.waiting + self.inflight[name]

    def load(self, name: str) -> float:
        """Weight-normalized queue depth. A backend with weight 2.0 must carry
        twice the raw depth to look as busy as a weight-1.0 backend, so a faster
        machine (higher weight) is handed proportionally more traffic."""
        return self.depth(name) / self.backends[name].weight

    def mem_headroom(self, name: str) -> float:
        s = self.snap[name]
        if s.mem_max <= 0:
            return 0.0
        return max(0.0, (s.mem_max - s.mem_used) / s.mem_max)

    def _mem_penalty(self, name: str) -> float:
        """Soft memory gate: 0 while headroom >= mem_soft_floor, then ramps up to
        mem_penalty as headroom -> 0, added to load when ranking. It is soft: a
        pressured backend is still chosen when every alternative is more loaded
        than the penalty. A backend that reports no memory data (mem_max <= 0) is
        treated as unknown -> not penalized, never as maximally pressured."""
        floor = self.cfg.mem_soft_floor
        s = self.snap[name]
        if floor <= 0 or s.mem_max <= 0:
            return 0.0
        h = self.mem_headroom(name)
        if h >= floor:
            return 0.0
        return self.cfg.mem_penalty * (floor - h) / floor

    def score(self, name: str) -> float:
        """Routing cost (lower wins): weight-normalized load + memory penalty."""
        return self.load(name) + self._mem_penalty(name)

    def resolve_model(self, model: str | None) -> str | None:
        if model is None:
            return None
        return self.cfg.aliases.get(model, model)

    def eligible(self, model_id: str) -> list[str]:
        return [
            name for name in self.backends
            if self.snap[name].healthy and model_id in self.snap[name].catalog
        ]

    def pick(self, model_id: str) -> str | None:
        """Choose a backend for ``model_id`` or None if none can serve it.

        model-first -> resident-preferred -> sticky-affinity -> least-score,
        where score = weight-normalized load + memory penalty.
        """
        elig = self.eligible(model_id)
        if not elig:
            return None
        resident = [n for n in elig if self.snap[n].catalog.get(model_id)]
        # Prefer a backend where the model is already loaded; only fall back to a
        # cold load (any eligible) when none has it resident.
        pool = resident or elig

        least = min(self.score(n) for n in pool)
        a = self.affinity.get(model_id)
        # Sticky: keep the model on its current backend unless it is meaningfully
        # busier than the least-busy candidate (prevents per-request flapping).
        if a in pool and self.score(a) <= least + self.cfg.affinity_hysteresis:
            return a
        # Lowest score wins; break exact ties by more memory headroom.
        chosen = min(pool, key=lambda n: (self.score(n), -self.mem_headroom(n)))
        self.affinity[model_id] = chosen
        return chosen

    def pick_model_less(self) -> str | None:
        """Backend for a request with no resolvable model (e.g. audio multipart).

        MVP: the least-busy healthy backend. Most such endpoints (audio STT/TTS,
        rerank) are single-home, so path-based routing is a later refinement.
        """
        healthy = [n for n in self.backends if self.snap[n].healthy]
        if not healthy:
            return None
        return min(healthy, key=self.score)

    def merged_models(self) -> list[dict[str, Any]]:
        seen: dict[str, dict] = {}
        for name in self.backends:
            for mid in self.snap[name].catalog:
                e = seen.setdefault(mid, {"id": mid, "object": "model",
                                          "owned_by": "omlx-cluster", "servers": []})
                e["servers"].append(name)
        return list(seen.values())


# --------------------------------------------------------------------------- #
# HTTP handlers
# --------------------------------------------------------------------------- #
def _client_authorized(request: Request, router_key: str | None) -> bool:
    if not router_key:
        return True
    presented = request.headers.get("x-api-key")
    if not presented:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            presented = auth[7:].strip()
    return presented == router_key


def _upstream_headers(request: Request, backend: Backend) -> dict[str, str]:
    # Forward content headers; replace auth with the backend's real key; drop
    # hop-by-hop and host headers.
    drop = {"host", "authorization", "x-api-key", "content-length",
            "connection", "accept-encoding"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in drop}
    if backend.api_key:
        headers["X-API-Key"] = backend.api_key
    return headers


def _extract_model(path: str, body: bytes) -> str | None:
    if path not in _MODEL_BODY_PATHS or not body:
        return None
    try:
        return json.loads(body).get("model")
    except Exception:
        return None


def _is_stream(path: str, body: bytes) -> bool:
    if not body:
        return False
    try:
        return bool(json.loads(body).get("stream"))
    except Exception:
        return False


def build_app(router: ClusterRouter) -> Starlette:
    cfg = router.cfg

    async def health(request: Request) -> Response:
        return JSONResponse({
            "status": "ok",
            "backends": {
                name: {
                    "healthy": s.healthy,
                    "weight": router.backends[name].weight,
                    "active": s.active, "waiting": s.waiting,
                    "inflight": router.inflight[name],
                    "depth": router.depth(name),
                    "load": round(router.load(name), 3),
                    "score": round(router.score(name), 3),
                    "mem_headroom": round(router.mem_headroom(name), 3),
                    "models": len(s.catalog),
                    "resident": [m for m, r in s.catalog.items() if r],
                    "last_error": s.last_err,
                }
                for name, s in router.snap.items()
            },
        })

    async def list_models(request: Request) -> Response:
        if not _client_authorized(request, cfg.router_api_key):
            return JSONResponse({"error": "API key required"}, status_code=401)
        return JSONResponse({"object": "list", "data": router.merged_models()})

    async def proxy(request: Request) -> Response:
        if not _client_authorized(request, cfg.router_api_key):
            return JSONResponse(
                {"error": {"message": "API key required", "type": "authentication_error"}},
                status_code=401,
            )
        path = request.url.path
        body = await request.body()
        model = router.resolve_model(_extract_model(path, body))
        stream = _is_stream(path, body)

        # Decide the candidate backend(s).
        if model is not None:
            chosen = router.pick(model)
            if chosen is None:
                served = sorted({m for n in router.backends for m in router.snap[n].catalog})
                return JSONResponse(
                    {"error": {"message": f"Model '{model}' not served by any healthy "
                               f"backend. Available: {served}", "type": "model_not_found"}},
                    status_code=503,
                )
            # Pre-stream failover order: chosen first, then other eligible backends.
            order = [chosen] + [n for n in router.eligible(model) if n != chosen]
        else:
            chosen = router.pick_model_less()
            if chosen is None:
                return JSONResponse(
                    {"error": {"message": "No healthy backend", "type": "unavailable"}},
                    status_code=503,
                )
            order = [chosen]

        qs = ("?" + request.url.query) if request.url.query else ""

        if stream:
            return await _proxy_stream(router, request, path, qs, body, order, model)
        return await _proxy_unary(router, request, path, qs, body, order, model)

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/v1/models", list_models, methods=["GET"]),
        Route("/{path:path}", proxy, methods=["GET", "POST", "PUT", "DELETE"]),
    ]

    @contextlib.asynccontextmanager
    async def lifespan(app):
        await router.start()
        logger.info("cluster router up on %s:%d -> %s", cfg.host, cfg.port,
                    [b.name for b in cfg.backends])
        try:
            yield
        finally:
            await router.stop()

    return Starlette(routes=routes, lifespan=lifespan)


async def _proxy_unary(router, request, path, qs, body, order, model) -> Response:
    last_err = None
    for name in order:
        backend = router.backends[name]
        url = backend.base_url + path + qs
        headers = _upstream_headers(request, backend)
        router.inflight[name] += 1
        try:
            resp = await router.client.request(
                request.method, url, headers=headers, content=body,
                timeout=httpx.Timeout(None),
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError) as exc:
            # Connection-level failure -> try the next eligible backend.
            last_err = exc
            router.inflight[name] -= 1
            logger.warning("backend %s connect failed (%s); failing over", name, exc)
            continue
        except Exception as exc:
            router.inflight[name] -= 1
            return JSONResponse(
                {"error": {"message": f"upstream error: {exc}", "type": "bad_gateway"}},
                status_code=502,
            )
        else:
            router.inflight[name] -= 1
            out_headers = {k: v for k, v in resp.headers.items()
                           if k.lower() not in ("content-length", "transfer-encoding",
                                                "connection", "content-encoding")}
            out_headers["x-omlx-backend"] = name
            return Response(content=resp.content, status_code=resp.status_code,
                            headers=out_headers, media_type=resp.headers.get("content-type"))
    return JSONResponse(
        {"error": {"message": f"all backends failed for model '{model}': {last_err}",
                   "type": "unavailable"}},
        status_code=503,
    )


async def _proxy_stream(router, request, path, qs, body, order, model) -> Response:
    """Open the upstream stream BEFORE returning so pre-stream connection
    failures can fail over. Once the first bytes are committed, no failover."""
    last_err = None
    for name in order:
        backend = router.backends[name]
        url = backend.base_url + path + qs
        headers = _upstream_headers(request, backend)
        ctx = router.client.stream(
            request.method, url, headers=headers, content=body,
            timeout=httpx.Timeout(None),
        )
        # Count from DISPATCH (before the upstream sends its first byte): a
        # request stuck in the backend's cold-load / prefill phase must already
        # weigh on this backend's depth, else concurrent requests in the poll
        # gap all pile onto it.
        router.inflight[name] += 1
        try:
            resp = await ctx.__aenter__()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError) as exc:
            last_err = exc
            router.inflight[name] -= 1
            logger.warning("backend %s stream connect failed (%s); failing over", name, exc)
            await ctx.__aexit__(type(exc), exc, exc.__traceback__)
            continue

        async def body_iter(resp=resp, ctx=ctx, name=name):
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                # Runs on normal end AND on client disconnect (Starlette cancels
                # this generator), which closes ctx -> cancels the upstream
                # request so a dead client never pins a backend slot.
                router.inflight[name] -= 1
                with _suppress():
                    await ctx.__aexit__(None, None, None)

        out_headers = {k: v for k, v in resp.headers.items()
                       if k.lower() not in ("content-length", "transfer-encoding",
                                            "connection", "content-encoding")}
        out_headers["x-omlx-backend"] = name
        return StreamingResponse(
            body_iter(), status_code=resp.status_code,
            headers=out_headers,
            media_type=resp.headers.get("content-type", "text/event-stream"),
        )
    return JSONResponse(
        {"error": {"message": f"all backends failed for model '{model}': {last_err}",
                   "type": "unavailable"}},
        status_code=503,
    )


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=os.environ.get("OMLX_CLUSTER_LOG", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    cfg = load_config()
    app = build_app(ClusterRouter(cfg))
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
