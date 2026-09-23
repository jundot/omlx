# SPDX-License-Identifier: Apache-2.0
"""Jev-compatible structured reads, mounted under ``/jev``.

* ``POST /jev/v1/systemone`` — answer a fixed set of questions in one read.
* ``GET  /jev/v1/models`` — list only the models that can answer reads.

The endpoint compiles the request into canvas read groups and asks the engine for
one read-only denoise of a seeded answer template. Nothing is generated and
nothing is parsed: the per-slot distribution over the allowed labels **is** the
answer, so a response cannot drift off the schema the client sent.

The whole surface sits behind one prefix, and the reason is ``/v1/models``: the
Jev listing is ``{"models": [{name, description, release_date}]}`` while omlx's
OpenAI listing is ``{"object": "list", "data": [{id, ...}]}``, and the SDK builds
both URLs from a single ``base_url`` (``config.base_url + path``). Two shapes
cannot share one path, so the Jev contract gets its own namespace instead of a
mirror inside the canonical OpenAI response. Point the SDK at
``base_url="http://host:8000/jev"`` and both endpoints resolve.

Wiring follows the rerank endpoint's shape (``server.py`` ``create_rerank``):
validate/resolve/load up front, then hold the pool's eviction-proof lease only
around the work itself, honour the pool's abort request before starting, and
record metrics on completion. The engine getter is injected by ``server.py`` so
this module imports without dragging the server in — which is also what lets the
tests drive it with a fake engine.

Resolution happens **before** the pool is asked for an engine, for two reasons:
``get_engine`` keys its entries by physical id and does not resolve aliases, so
``jev-latest`` would fail at the pool; and the block-diffusion gate reads the
discovered entry's config, so a request naming an autoregressive model is refused
before its weights move rather than after a full load.

Errors follow Jev so their SDKs branch correctly:

* ``422`` with FastAPI's validation-list ``detail`` for anything wrong with the
  schema, including a canvas the served model cannot hold;
* ``{"detail": {"error_type", "message"}}`` for ``404`` (unknown model, or a
  model that is not a block-diffusion model and so is not addressable here),
  ``409``/``507`` when the pool asked to abort, ``529`` + ``Retry-After`` when
  the diffusion lane is busy and the caller passed ``queue: false``, and
  ``503`` when the server is not ready to serve reads.

404 is deliberate over 503 for an unreadable model id: the SDK retries every 5xx
(``http_statuses={408, 429, *range(500, 600)}``), so a retryable code on a
request that can never succeed buys three load attempts and a backoff.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..exceptions import (
    EnginePoolError,
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
    ModelUnavailableError,
)
from ..systemone import (
    SchemaError,
    build_prompt_ids,
    build_schema,
    read_groups,
    system_text,
    text_of,
    to_answer,
)
from .systemone_models import SystemOneRequest, SystemOneResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["systemone"])


# Names the TypeSafe SDK sends when the caller names no model: DEFAULT_MODEL is
# "jev-latest" (constants.py) and endpoints.py injects it into the body. They
# are accepted here without configuration, resolving to the default readable
# checkpoint, so an unmodified SDK pointed at base_url=".../jev" just works.
JEV_MODEL_ALIASES: tuple[str, ...] = (
    "jev-latest",
    "jev-preview",
    "openjev-latest",
    "openjev-0.1",
)


def _default_read_target(pool) -> str | None:
    """The checkpoint a Jev alias resolves to.

    The checkpoint configured in Server > Advanced wins: that is the user's
    choice, and it is what makes ``jev-latest`` mean the same thing across
    restarts instead of drifting with whatever happens to be loaded. Otherwise
    whichever readable model is already loaded, else the first discovered.
    """
    ids = pool.systemone_model_ids()
    if not ids:
        return None

    configured = _get_systemone_model() if _get_systemone_model is not None else ""
    if configured:
        target = configured
        if _resolve_model_id is not None:
            target = _resolve_model_id(configured) or configured
        if target in ids:
            return target
        logger.warning(
            "SystemOne: configured model %r resolves to %r, which is not a "
            "readable checkpoint; falling back to automatic choice.",
            configured,
            target,
        )

    loaded = getattr(pool, "get_loaded_model_ids", None)
    loaded_ids = loaded() if callable(loaded) else []
    for model_id in sorted(loaded_ids):
        if model_id in ids:
            return model_id
    return sorted(ids)[0]


def _read_target(pool, name: str) -> tuple[str | None, JSONResponse | None]:
    """Resolve a requested name to a readable physical id.

    Exactly one of the two returns is set. Unknown names and names for models
    with no canvas are both 404, but with different ``error_type``: one is a
    typo, the other is a model that cannot answer a structured read at all.
    """
    resolved = name
    if _resolve_model_id is not None:
        resolved = _resolve_model_id(name) or name

    supports = getattr(pool, "supports_systemone", None)
    if callable(supports) and supports(resolved):
        return resolved, None

    # A Jev alias is not a configured alias: fall back to the default readable
    # checkpoint rather than asking the user to create one per SDK release name.
    entry = pool.get_entry(resolved) if hasattr(pool, "get_entry") else None
    if entry is None and name in JEV_MODEL_ALIASES:
        target = _default_read_target(pool)
        if target is not None:
            return target, None

    if entry is None:
        available = ", ".join(
            sorted(getattr(pool, "systemone_model_ids", lambda: [])())
        )
        message = f"Model {name!r} not found."
        message += (
            f" Readable models: {available}."
            if available
            else " No block-diffusion models are available for structured reads."
        )
        return None, _error(404, "model_not_found", message)
    return None, _error(
        404,
        "model_not_compatible",
        f"Model {name!r} is not a block-diffusion model and cannot answer "
        f"structured reads. Readable models: "
        f"{', '.join(sorted(pool.systemone_model_ids()))}.",
    )


def _release_date(model_path: str) -> str:
    """YYYY-MM-DD for the listing's required ``release_date`` field.

    Taken from the local checkpoint's ``config.json``, so it is the date that
    file landed on this disk — an upper bound on publication, not upstream's
    release date. There is no offline source for the upstream date, and the
    SDK's ``ModelMetadata`` requires the key to be present.
    """
    import os
    from pathlib import Path

    for candidate in (Path(model_path) / "config.json", Path(model_path)):
        try:
            return time.strftime("%Y-%m-%d", time.gmtime(os.path.getmtime(candidate)))
        except OSError:
            continue
    return "1970-01-01"


# Callbacks injected by server.py (see set_systemone_getters). Unset means the
# routes were mounted without a server behind them, which is a 503 rather than a
# 500 — the client's retry logic is the right answer to it.
_get_engine_pool: Callable[[], Any] | None = None
_resolve_model_id: Callable[[str | None], str | None] | None = None
_get_metrics: Callable[[], Any] | None = None
_get_systemone_model: Callable[[], str] | None = None


def set_systemone_getters(
    get_engine_pool: Callable[[], Any],
    resolve_model_id: Callable[[str | None], str | None] | None = None,
    get_metrics: Callable[[], Any] | None = None,
    get_systemone_model: Callable[[], str] | None = None,
) -> None:
    """Inject the pool accessor, model-id resolver, metrics recorder and the
    configured System One checkpoint.

    Mirrors ``set_admin_getters``/``set_cluster_getters``: the router owns the
    contract, the server owns the singletons, and neither imports the other at
    module scope.
    """

    global _get_engine_pool, _resolve_model_id, _get_metrics, _get_systemone_model
    _get_engine_pool = get_engine_pool
    _resolve_model_id = resolve_model_id
    _get_metrics = get_metrics
    _get_systemone_model = get_systemone_model


def _error(status_code: int, error_type: str, message: str, **headers: str):
    return JSONResponse(
        status_code=status_code,
        content={"detail": {"error_type": error_type, "message": message}},
        headers=headers or None,
    )


def _schema_422(exc: SchemaError) -> HTTPException:
    """A schema problem, in FastAPI's validation-list shape.

    ``loc`` travels with :class:`SchemaError` precisely so it does not have to be
    re-derived here — and re-derived wrong, sending the client to fix a field
    that was never the problem.
    """

    return HTTPException(
        status_code=422,
        detail=[
            {
                "type": "value_error",
                "loc": list(exc.loc),
                "msg": f"Value error, {exc}",
                "input": None,
            }
        ],
    )


def _reject_unsupported(request: SystemOneRequest) -> None:
    """Refuse extensions that are not served yet, rather than ignore them.

    Silently dropping ``think: 2048`` would return an answer that looks like the
    client's request was honoured. ``steps`` reaches the engine, which rejects
    anything but 1 with its own ``InvalidRequestError``.
    """

    unsupported: list[tuple[str, str]] = []
    if request.images:
        unsupported.append(("body.images", "images are not served yet."))
    if request.think:
        unsupported.append(("body.think", "thinking is not served yet."))
    if request.sequential:
        unsupported.append(("body.sequential", "sequential reads are not served yet."))
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "type": "value_error",
                    "loc": list(loc),
                    "msg": f"Value error, {msg}",
                    "input": None,
                }
                for loc, msg in unsupported
            ],
        )


@router.get("/v1/models")
async def list_read_models(_: Request) -> JSONResponse:
    """List the models that can answer structured reads, in Jev's shape.

    ``{"models": [{name, description, release_date}]}`` — every key is required
    by the SDK's ``ModelMetadata``, and the wrapper is ``{"models": [...]}``,
    not OpenAI's ``{"object": "list", "data": [...]}``. That is why this lives
    under ``/jev``: the two shapes cannot share a path, and the SDK builds both
    URLs from one ``base_url``.

    Only block-diffusion checkpoints are listed. Offering an autoregressive
    model here would name something this very route answers ``404`` for. The
    Jev aliases ride along so a client can see what ``jev-latest`` currently
    means on this server.
    """

    if _get_engine_pool is None:
        return _error(503, "server_unavailable", "This server cannot serve reads.")

    pool = _get_engine_pool()
    if pool is None:
        return _error(503, "server_unavailable", "Engine pool is not ready.")

    models: list[dict[str, str]] = []
    for model_id in sorted(pool.systemone_model_ids()):
        entry = pool.get_entry(model_id) if hasattr(pool, "get_entry") else None
        path = str(getattr(entry, "model_path", "") or "")
        models.append(
            {
                "name": model_id,
                "description": (
                    "Block-diffusion checkpoint on this omlx server. Answers a "
                    "fixed set of questions in one read-only denoise; no text is "
                    "generated and nothing is parsed."
                ),
                "release_date": _release_date(path),
            }
        )

    target = _default_read_target(pool)
    if target is not None:
        target_date = next(
            (m["release_date"] for m in models if m["name"] == target),
            _release_date(
                str(
                    getattr(
                        pool.get_entry(target) if hasattr(pool, "get_entry") else None,
                        "model_path",
                        "",
                    )
                    or ""
                )
            ),
        )
        for alias in JEV_MODEL_ALIASES:
            models.append(
                {
                    "name": alias,
                    "description": f"Alias for {target}.",
                    "release_date": target_date,
                }
            )

    return JSONResponse(status_code=200, content={"models": models})


@router.post("/v1/systemone")
async def create_systemone(
    request: SystemOneRequest,
    http_request: Request,
) -> JSONResponse:
    """Answer every question in one structured read.

    Example:

    ```json
    {
      "model": "diffusiongemma-26b-a4b-it-4bit",
      "state": "Everything is down and we have a demo at noon.",
      "questions": {
        "urgent": {"type": "noul", "criteria": {"true": "needs a human now"}},
        "team":   {"type": "choice", "criteria": {"outage": "…", "billing": "…"}},
        "tone":   {"type": "score",  "criteria": ["calm", "annoyed", "furious"]}
      },
      "samples": 4
    }
    ```
    """

    _reject_unsupported(request)

    if _get_engine_pool is None:
        return _error(503, "server_unavailable", "This server cannot serve reads.")

    pool = _get_engine_pool()
    if pool is None:
        return _error(503, "server_unavailable", "Engine pool is not ready.")

    start = time.perf_counter()

    # Resolve + load + lease. The pool releases nothing until we say so, so the
    # engine cannot be evicted between the check below and the forward pass.
    #
    # The pool raises its own exception classes, not HTTPException — server.get_engine
    # translates them for /v1/chat, and this route talks to the pool directly, so the
    # same table lives here. Without it an unknown model id answers 503, which tells
    # the client to retry a request that will never succeed.
    # Resolve and gate before the pool is asked for an engine. get_engine loads
    # the model it is given, so without this a request naming an autoregressive
    # model pays a full load (69 GB on this box) before being told 404, and the
    # resulting memory pressure reclaims the diffusion model that was resident.
    resolved, target_error = _read_target(pool, request.model)
    if target_error is not None:
        return target_error

    try:
        engine = await pool.get_engine(resolved, _lease=True)
    except HTTPException:
        raise
    except ModelNotFoundError as exc:
        return _error(404, "model_not_found", str(exc))
    except (ModelTooLargeError, InsufficientMemoryError) as exc:
        return _error(507, "insufficient_storage", str(exc))
    except (ModelUnavailableError, ModelLoadingError, ModelBusyError) as exc:
        return _error(409, "model_unavailable", str(exc))
    except EnginePoolError as exc:
        logger.error("SystemOne: engine pool error for %r: %s", request.model, exc)
        return _error(500, "engine_pool_error", str(exc))
    except Exception as exc:  # anything else: the client cannot fix it
        logger.warning("SystemOne: engine load for %r failed: %s", request.model, exc)
        return _error(503, "backend_unavailable", f"Cannot load model: {exc}")

    try:
        # Aborted before it ever reached the lane: an admin unload or memory
        # pressure that already asked for this model to go.
        abort_reason = None
        get_reason = getattr(pool, "get_abort_requested_reason", None)
        if callable(get_reason):
            abort_reason = get_reason(resolved)
        if abort_reason == "manual admin unload":
            return _error(
                409,
                "request_aborted",
                "Request aborted because this model is being unloaded.",
            )
        if abort_reason is not None:
            return _error(
                507,
                "insufficient_storage",
                "Request aborted before scheduling because memory pressure "
                "requested this model to unload.",
            )

        if not getattr(engine, "is_diffusion_model", False):
            # Not addressable on this endpoint: an autoregressive model has no
            # canvas to read, and there is no read that answers its output.
            return _error(
                404,
                "model_not_compatible",
                f"Model {request.model!r} is not a block-diffusion model and "
                f"cannot answer structured reads.",
            )

        if not request.queue and engine.has_active_requests():
            return _error(
                529,
                "lane_saturated",
                "The diffusion lane is busy with another read.",
                **{"Retry-After": "1"},
            )

        tokenizer = getattr(engine, "tokenizer", None)
        if tokenizer is None:
            return _error(503, "backend_unavailable", "Model has no tokenizer.")

        try:
            schema = build_schema(
                {
                    k: q.model_dump(exclude_none=True)
                    for k, q in request.questions.items()
                },
                tokenizer,
            )
            prompt_ids = build_prompt_ids(
                tokenizer,
                system_text(schema.questions, schema.fmt),
                text_of(request.state),
            )
            groups = read_groups(
                tokenizer,
                schema,
                canvas_length=int(getattr(engine, "diffusion_canvas_length", 0) or 0),
            )
        except SchemaError as exc:
            raise _schema_422(exc) from exc

        try:
            read = await engine.structured_read(
                prompt_ids=prompt_ids,
                groups=groups,
                samples=request.samples,
                steps=request.steps,
                seed=request.seed,
            )
        except HTTPException:
            raise
        except Exception as exc:  # InvalidRequestError and engine faults
            field = getattr(exc, "field", None)
            loc = ["body", field] if field else ["body", "questions"]
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "value_error",
                        "loc": loc,
                        "msg": f"Value error, {exc}",
                        "input": None,
                    }
                ],
            ) from exc

        answers: dict[str, Any] = {}
        for question in schema.questions:
            dist = read.by_key(question["key"])
            if dist is None:
                return _error(
                    503,
                    "backend_unavailable",
                    f"The read returned no distribution for {question['key']!r}.",
                )
            answers[question["key"]] = to_answer(question, dist.probabilities)

        elapsed = time.perf_counter() - start
        input_tokens = int(read.prompt_tokens or len(prompt_ids))

        logger.info(
            "SystemOne: model=%s%s, %d questions (%s), %d group(s), %d sample(s), "
            "canvas=%d, seed=%s, %d forward(s) in %.3fs",
            resolved,
            f" (alias {request.model!r})" if resolved != request.model else "",
            len(schema),
            schema.fmt,
            len(groups),
            read.samples,
            read.canvas_tokens,
            read.seed,
            read.forwards,
            elapsed,
        )
        if _get_metrics is not None:
            _get_metrics().record_request_complete(
                prompt_tokens=input_tokens,
                completion_tokens=0,
                cached_tokens=0,
                prefill_duration=elapsed,
                model_id=resolved,
                request_duration=elapsed,
            )

        body = SystemOneResponse(
            model=request.model,
            answers=answers,
            usage={"input_tokens": input_tokens, "output_tokens": 0},
        )
        headers = {
            "x-systemone-seed": str(read.seed if read.seed is not None else ""),
            "x-systemone-samples": str(read.samples),
            "x-systemone-canvas-tokens": str(read.canvas_tokens),
            "x-systemone-forwards": str(read.forwards),
        }
        request_id = http_request.headers.get("x-request-id")
        if request_id:
            headers["x-request-id"] = request_id
        return JSONResponse(status_code=200, content=body.model_dump(), headers=headers)

    finally:
        await pool.release_engine(resolved)
