# SPDX-License-Identifier: Apache-2.0
"""One-image, fixed-recipe generation; authenticated by the server router mount."""

import asyncio
import base64
import contextlib
import secrets
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..engine.image import _drain

router = APIRouter()


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=32768)
    n: int = Field(default=1, strict=True, ge=1, le=1)
    size: Literal["1024x1024"] = "1024x1024"
    response_format: Literal["b64_json"] = "b64_json"
    seed: int | None = Field(default=None, strict=True, ge=0, lt=2**32)

    @field_validator("prompt")
    @classmethod
    def prompt_is_encodable(cls, value):
        if not value.strip() or len(value.encode("utf-8")) > 32768:
            raise ValueError("prompt must be nonempty and at most 32768 UTF-8 bytes")
        return value


async def _wait_for_disconnect(request):
    # Await the ASGI message instead of polling is_disconnected(), whose
    # immediately cancelled receive can miss disconnects under middleware.
    while True:
        if (await request.receive())["type"] == "http.disconnect":
            return


@router.post("/v1/images/generations")
async def generate_image(request: ImageRequest, http_request: Request):
    from .. import server

    lease = server._LLMEngineLease()
    leased = []
    job = None
    disconnected = None
    try:
        engine = await server.get_engine(
            request.model, server.EngineType.IMAGE, _lease=True, _leased_out=leased
        )
        if leased:
            lease.model_id = leased[0]
        seed = request.seed if request.seed is not None else secrets.randbits(32)
        job = asyncio.create_task(engine.generate(request.prompt, seed))
        disconnected = asyncio.create_task(_wait_for_disconnect(http_request))
        while not job.done():
            await server._raise_if_llm_lease_abort_requested(lease)
            await asyncio.wait(
                {job, disconnected}, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
            )
            if disconnected.done():
                await disconnected
                raise HTTPException(status_code=499, detail="Client disconnected")
        image, _metrics = await job
        return {
            "created": int(time.time()),
            "data": [
                {"b64_json": base64.b64encode(image).decode("ascii"), "seed": seed}
            ],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (MemoryError, RuntimeError, InterruptedError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        if disconnected is not None:
            disconnected.cancel()
            with contextlib.suppress(BaseException):
                await _drain(disconnected)
        if job is not None:
            if not job.done():
                job.cancel()
            with contextlib.suppress(BaseException):
                await _drain(job)
        await _drain(asyncio.create_task(lease.release()))
