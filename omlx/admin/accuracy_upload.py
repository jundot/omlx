# SPDX-License-Identifier: Apache-2.0
"""Intelligence benchmark upload to omlx.ai community benchmarks.

Mirrors the throughput uploader in benchmark.py with a two-step protocol:

1. POST the small summary JSON (score, counts, hardware, flags). The score is
   registered the moment this succeeds.
2. PUT the per-question raw as a client-gzipped blob. The site stores the
   bytes in R2 without inflating them, so a raw failure never loses the score.

The raw records deliberately exclude the question text: datasets ship inside
oMLX and sampling is seeded, so a question id alone reconstructs the exact
prompt locally, while uploading the text would republish benchmark datasets
and quintuple the payload.
"""

import asyncio
import gzip
import json
import logging
import uuid
from typing import Any

import requests

from .._version import __version__
from ..utils.hardware import (
    compute_owner_hash,
    get_chip_name,
    get_gpu_core_count,
    get_io_platform_uuid,
    get_os_version,
    get_total_memory_gb,
    parse_chip_info,
)
from .benchmark import (
    _derive_feature_flags,
    _detect_quantization,
    _filter_uploaded_settings,
    _sanitize_upload_error,
    _upload_model_name,
    _upload_model_repo,
)

logger = logging.getLogger(__name__)

OMLX_AI_INTEL_API_URL = "https://omlx.ai/api/benchmarks/intelligence"

# Progressive per-question raw_response caps. The first step already bounds
# thinking-mode runs whose unclosed <think> block survives tag stripping; the
# later steps only engage when the run is so large the total budget is hit.
_RAW_TRIM_STEPS = (2000, 1000, 500, 200, 0)
# Uncompressed budget for the serialized question_results. Chosen so the gzip
# stays well under the server's 5MB compressed cap for any realistic text.
_TOTAL_RAW_BUDGET = 18_000_000
_MAX_RAW_GZIP_BYTES = 5 * 1024 * 1024
_MAX_ANSWER_LEN = 4000
_TRUNCATION_SUFFIX = " …[truncated]"
# Runs below this are too noisy for the public leaderboard; the server
# rejects them too, this just avoids a pointless round trip.
_MIN_UPLOAD_QUESTIONS = 100
# Server-side cap on category_counts keys; over it the whole summary would
# 400, so oversized maps are dropped instead. 250 clears the largest real
# category set (HellaSwag: 192 activity labels).
_MAX_CATEGORY_KEYS = 250


def build_upload_context(request: Any, engine_pool: Any) -> dict:
    """Snapshot everything upload-relevant at run start (local runs only).

    Settings can change mid-run; scores are tied to whatever was active when
    the run began, same rationale as the throughput bench snapshot.
    """
    chip_string = get_chip_name()
    chip_name, chip_variant = parse_chip_info(chip_string)
    memory_gb = round(get_total_memory_gb())
    gpu_cores = get_gpu_core_count()

    owner_hash_full = None
    io_uuid = get_io_platform_uuid()
    if io_uuid:
        owner_hash_full = compute_owner_hash(io_uuid, chip_name, gpu_cores, memory_gb)

    entry = engine_pool.get_entry(request.model_id)
    model_path = entry.model_path if entry else ""
    # Org-qualified repo id, e.g. "mlx-community/Qwen3-4bit" (#1808). None
    # for flat layouts; the model name then falls back to the model id, and
    # for HF-cache models the repo's own leaf replaces the "org--name" id.
    model_repo = _upload_model_repo(
        request.model_id,
        entry=entry,
        model_dirs=getattr(engine_pool, "_model_dirs", None),
    )

    feature_flags: list[dict] = []
    model_settings: dict | None = None
    sm = getattr(engine_pool, "_settings_manager", None)
    if sm is not None:
        try:
            ms = sm.get_settings(request.model_id)
            feature_flags = _derive_feature_flags(ms)
            model_settings = _filter_uploaded_settings(ms)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Accuracy upload: failed to read settings for "
                f"{request.model_id}: {e}"
            )

    return {
        "chip_name": chip_name,
        "chip_variant": chip_variant,
        "memory_gb": memory_gb,
        "gpu_cores": gpu_cores,
        "omlx_version": __version__,
        "os_version": get_os_version(),
        "model_name": _upload_model_name(
            model_repo if model_repo else request.model_id
        ),
        "model_repo": model_repo,
        "quantization": _detect_quantization(model_path),
        "sampling_profile": request.sampling_profile,
        "batch_size": request.batch_size,
        "feature_flags": feature_flags,
        "model_settings": model_settings,
        "submission_group": str(uuid.uuid4()),
        "owner_hash_full": owner_hash_full,
    }


def trim_question_results(question_results: Any) -> tuple[list[dict], bool]:
    """Project per-question records onto the upload shape and bound their size.

    Keeps id/correct/expected/predicted/raw_response/category/time_s and drops
    everything else — most importantly the 'question' prompt text. Walks the
    trim ladder until the serialized total fits the budget; the pathological
    fallback drops the raw entirely so the summary still uploads.
    """
    trimmed: list[dict] = []
    originals: list[str] = []
    truncated = False
    for q in question_results or []:
        if not isinstance(q, dict):
            continue
        expected = str(q.get("expected", ""))
        predicted = str(q.get("predicted", ""))
        if len(expected) > _MAX_ANSWER_LEN or len(predicted) > _MAX_ANSWER_LEN:
            truncated = True
        rec: dict = {
            "id": str(q.get("id", ""))[:64],
            "correct": bool(q.get("correct")),
            "expected": expected[:_MAX_ANSWER_LEN],
            "predicted": predicted[:_MAX_ANSWER_LEN],
        }
        category = q.get("category")
        if category:
            rec["category"] = str(category)[:80]
        time_s = q.get("time_s")
        if isinstance(time_s, (int, float)):
            rec["time_s"] = round(float(time_s), 3)
        trimmed.append(rec)
        raw = q.get("raw_response")
        originals.append(raw if isinstance(raw, str) else "")

    for cap in _RAW_TRIM_STEPS:
        step_truncated = truncated
        for rec, original in zip(trimmed, originals):
            if not original or cap == 0:
                rec.pop("raw_response", None)
                if original and cap == 0:
                    step_truncated = True
                continue
            if len(original) > cap:
                rec["raw_response"] = original[:cap] + _TRUNCATION_SUFFIX
                step_truncated = True
            else:
                rec["raw_response"] = original
        size = len(
            json.dumps(trimmed, ensure_ascii=False, separators=(",", ":")).encode()
        )
        if size <= _TOTAL_RAW_BUDGET:
            return trimmed, step_truncated

    return [], True


def _category_counts(question_results: Any) -> dict | None:
    """{category: [correct, total]} derived from the untrimmed records."""
    counts: dict[str, list[int]] = {}
    for q in question_results or []:
        if not isinstance(q, dict):
            continue
        category = q.get("category")
        if not category:
            continue
        pair = counts.setdefault(str(category)[:80], [0, 0])
        pair[1] += 1
        if q.get("correct"):
            pair[0] += 1
    if not counts or len(counts) > _MAX_CATEGORY_KEYS:
        return None
    return counts


async def upload_intelligence_result(
    run: Any, ctx: dict, result_data: dict
) -> dict:
    """Upload one suite result. Never raises; returns the outcome dict."""
    if result_data.get("total", 0) < _MIN_UPLOAD_QUESTIONS:
        return {"skipped": "min_questions"}
    try:
        return await _do_upload(ctx, result_data)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Intelligence benchmark upload failed: {e}")
        return {"error": str(e)[:300]}


async def _do_upload(ctx: dict, result_data: dict) -> dict:
    trimmed, truncated = trim_question_results(result_data.get("question_results"))
    raw_bytes = b""
    raw_gzip = b""
    if trimmed:
        raw_bytes = json.dumps(
            trimmed, ensure_ascii=False, separators=(",", ":")
        ).encode()
        raw_gzip = gzip.compress(raw_bytes)
        if len(raw_gzip) > _MAX_RAW_GZIP_BYTES:
            # Text this incompressible is not worth publishing; keep the score.
            logger.warning(
                "Accuracy upload: compressed raw exceeds the server cap, "
                "uploading summary only"
            )
            trimmed, truncated = [], True
            raw_bytes, raw_gzip = b"", b""

    payload = {
        "chip_name": ctx["chip_name"],
        "chip_variant": ctx["chip_variant"],
        "memory_gb": ctx["memory_gb"],
        "gpu_cores": ctx["gpu_cores"],
        "omlx_version": ctx["omlx_version"],
        "os_version": ctx["os_version"],
        "model_name": ctx["model_name"],
        "model_repo": ctx.get("model_repo"),
        "quantization": ctx["quantization"],
        "benchmark": result_data["benchmark"],
        "accuracy": result_data["accuracy"],
        "correct_count": result_data["correct"],
        "total_questions": result_data["total"],
        "dataset_total": result_data.get("dataset_total"),
        "time_s": result_data.get("time_s"),
        "thinking_used": bool(result_data.get("thinking_used")),
        "sampling_profile": ctx["sampling_profile"],
        "batch_size": ctx["batch_size"],
        "category_scores": result_data.get("category_scores"),
        "category_counts": _category_counts(result_data.get("question_results")),
        "feature_flags": ctx["feature_flags"],
        "model_settings": ctx["model_settings"],
        "raw_size": len(raw_bytes),
        "raw_truncated": truncated,
        "submission_group": ctx["submission_group"],
    }
    if ctx.get("owner_hash_full"):
        payload["owner_hash"] = ctx["owner_hash_full"]

    resp = await asyncio.to_thread(
        requests.post, OMLX_AI_INTEL_API_URL, json=payload, timeout=15
    )

    if resp.status_code == 409:
        data = resp.json()
        # Deterministic re-runs reproduce correct_count exactly, so a
        # duplicate is expected and counts as success. Raw upload is skipped —
        # the existing row already has (or intentionally lacks) its raw.
        return {
            "id": data.get("existing_id"),
            "url": data.get("existing_url"),
            "duplicate": True,
        }
    if resp.status_code != 201:
        return {"error": _sanitize_upload_error(resp)}

    data = resp.json()
    outcome: dict = {"id": data.get("id"), "url": data.get("url")}

    if raw_gzip and ctx.get("owner_hash_full"):
        owner_display = ctx["owner_hash_full"][:-1]
        put_url = (
            f"{OMLX_AI_INTEL_API_URL}/{outcome['id']}/raw"
            f"?owner_hash={owner_display}"
        )
        try:
            resp2 = await asyncio.to_thread(
                requests.put,
                put_url,
                data=raw_gzip,
                headers={"Content-Type": "application/gzip"},
                timeout=60,
            )
            outcome["raw_uploaded"] = resp2.status_code == 200
            if resp2.status_code != 200:
                logger.warning(
                    "Accuracy upload: raw upload failed "
                    f"({_sanitize_upload_error(resp2)})"
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Accuracy upload: raw upload failed: {e}")
            outcome["raw_uploaded"] = False
    else:
        outcome["raw_uploaded"] = False

    return outcome
