# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the intelligence benchmark uploader (omlx.ai)."""

import gzip
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import omlx.admin.accuracy_upload as accuracy_upload
from omlx.admin.accuracy_upload import (
    build_upload_context,
    trim_question_results,
    upload_intelligence_result,
)


def _question(i: int = 0, raw: str = "B", **overrides) -> dict:
    q = {
        "id": str(i),
        "correct": True,
        "expected": "B",
        "predicted": "B",
        "question": "FULL PROMPT TEXT THAT MUST NOT UPLOAD",
        "raw_response": raw,
        "category": "anatomy",
        "time_s": 1.2345,
    }
    q.update(overrides)
    return q


class TestTrimQuestionResults:
    def test_strips_question_and_keeps_allowed_fields(self):
        trimmed, truncated = trim_question_results([_question()])
        assert truncated is False
        rec = trimmed[0]
        assert "question" not in rec
        assert set(rec) == {
            "id", "correct", "expected", "predicted",
            "raw_response", "category", "time_s",
        }
        assert rec["raw_response"] == "B"
        assert rec["time_s"] == 1.234

    def test_external_only_fields_dropped(self):
        trimmed, _ = trim_question_results(
            [_question(status="correct", finish_reason="stop", prompt_tokens=10)]
        )
        assert "status" not in trimmed[0]
        assert "finish_reason" not in trimmed[0]
        assert "prompt_tokens" not in trimmed[0]

    def test_per_question_raw_cap(self):
        trimmed, truncated = trim_question_results([_question(raw="x" * 5000)])
        assert truncated is True
        rec = trimmed[0]
        assert rec["raw_response"].startswith("x" * 2000)
        assert rec["raw_response"].endswith("[truncated]")

    def test_budget_ladder_shrinks_raw(self):
        # 100 questions x 2000-char raw ≈ 210KB; a 100KB budget forces the
        # ladder down to the 500-char step.
        questions = [_question(i, raw="y" * 2000) for i in range(100)]
        with patch.object(accuracy_upload, "_TOTAL_RAW_BUDGET", 100_000):
            trimmed, truncated = trim_question_results(questions)
        assert truncated is True
        assert len(trimmed) == 100
        longest = max(len(r["raw_response"]) for r in trimmed)
        assert longest <= 1000 + len(" …[truncated]")

    def test_pathological_budget_drops_raw_entirely(self):
        questions = [_question(i, expected="e" * 3000) for i in range(50)]
        with patch.object(accuracy_upload, "_TOTAL_RAW_BUDGET", 1_000):
            trimmed, truncated = trim_question_results(questions)
        assert trimmed == []
        assert truncated is True

    def test_empty_and_none_input(self):
        assert trim_question_results(None) == ([], False)
        assert trim_question_results([]) == ([], False)


class TestBuildUploadContext:
    def test_context_fields(self):
        request = MagicMock()
        request.model_id = "models/Qwen3-4bit"
        request.sampling_profile = "deterministic"
        request.batch_size = 8

        entry = MagicMock()
        entry.model_path = "/models/Qwen3-4bit"
        pool = MagicMock()
        pool.get_entry.return_value = entry
        pool._settings_manager = None

        with (
            patch.object(accuracy_upload, "get_chip_name", return_value="Apple M4 Max"),
            patch.object(
                accuracy_upload, "parse_chip_info", return_value=("M4", "Max")
            ),
            patch.object(accuracy_upload, "get_total_memory_gb", return_value=128.0),
            patch.object(accuracy_upload, "get_gpu_core_count", return_value=40),
            patch.object(accuracy_upload, "get_os_version", return_value="macOS 15.5"),
            patch.object(
                accuracy_upload, "get_io_platform_uuid", return_value="UUID-1"
            ),
            patch.object(
                accuracy_upload, "compute_owner_hash", return_value="h" * 64 + "a"
            ),
            patch.object(
                accuracy_upload, "_detect_quantization", return_value="4bit"
            ),
        ):
            ctx = build_upload_context(request, pool)

        assert ctx["chip_name"] == "M4"
        assert ctx["chip_variant"] == "Max"
        assert ctx["memory_gb"] == 128
        assert ctx["quantization"] == "4bit"
        # entry is a bare MagicMock (no usable path context), so the name
        # falls back to the trailing component of the model id and no repo
        # is derived.
        assert ctx["model_name"] == "Qwen3-4bit"
        assert ctx["model_repo"] is None
        assert ctx["sampling_profile"] == "deterministic"
        assert ctx["batch_size"] == 8
        assert ctx["owner_hash_full"] == "h" * 64 + "a"
        assert ctx["feature_flags"] == []
        assert len(ctx["submission_group"]) == 36

    def test_org_layout_fills_model_repo(self):
        from pathlib import Path

        request = MagicMock()
        request.model_id = "Qwen3-4bit"
        request.sampling_profile = "deterministic"
        request.batch_size = 8

        entry = MagicMock(spec=["model_path", "source_repo_id"])
        entry.model_path = "/models/mlx-community/Qwen3-4bit"
        entry.source_repo_id = None
        pool = MagicMock()
        pool.get_entry.return_value = entry
        pool._settings_manager = None
        pool._model_dirs = [Path("/models")]

        with (
            patch.object(accuracy_upload, "get_chip_name", return_value="Apple M4"),
            patch.object(accuracy_upload, "parse_chip_info", return_value=("M4", "")),
            patch.object(accuracy_upload, "get_total_memory_gb", return_value=64.0),
            patch.object(accuracy_upload, "get_gpu_core_count", return_value=20),
            patch.object(accuracy_upload, "get_os_version", return_value="macOS 15.5"),
            patch.object(accuracy_upload, "get_io_platform_uuid", return_value=None),
            patch.object(
                accuracy_upload, "_detect_quantization", return_value="4bit"
            ),
        ):
            ctx = build_upload_context(request, pool)

        assert ctx["model_repo"] == "mlx-community/Qwen3-4bit"
        assert ctx["model_name"] == "Qwen3-4bit"


def _ctx(**overrides) -> dict:
    ctx = {
        "chip_name": "M4",
        "chip_variant": "Max",
        "memory_gb": 128,
        "gpu_cores": 40,
        "omlx_version": "0.9.9",
        "os_version": "macOS 15.5",
        "model_name": "Qwen3-4bit",
        "model_repo": "mlx-community/Qwen3-4bit",
        "quantization": "4bit",
        "sampling_profile": "deterministic",
        "batch_size": 8,
        "feature_flags": [{"key": "turboquant_kv_4bit", "label": "TurboQuant KV 4-bit"}],
        "model_settings": {"max_context_window": 32768},
        "submission_group": "group-1",
        "owner_hash_full": "h" * 64 + "a",
    }
    ctx.update(overrides)
    return ctx


def _result_data(**overrides) -> dict:
    data = {
        "model_id": "Qwen3-4bit",
        "benchmark": "mmlu",
        "accuracy": 0.75,
        "correct": 90,
        "total": 120,
        "time_s": 12.3,
        "thinking_used": False,
        "dataset_total": 14042,
        "sampling_profile": "deterministic",
        "category_scores": {"anatomy": 0.75},
        "question_results": [_question(i) for i in range(4)],
    }
    data.update(overrides)
    return data


def _response(status_code: int, body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    return resp


class TestUploadIntelligenceResult:
    @pytest.mark.asyncio
    async def test_success_uploads_summary_then_raw(self):
        post_resp = _response(201, {"id": "abc12345", "url": "https://omlx.ai/benchmarks/intelligence/abc12345"})
        put_resp = _response(200, {"id": "abc12345"})
        mock_to_thread = AsyncMock(side_effect=[post_resp, put_resp])

        with patch("asyncio.to_thread", mock_to_thread):
            outcome = await upload_intelligence_result(
                MagicMock(), _ctx(), _result_data()
            )

        assert outcome == {
            "id": "abc12345",
            "url": "https://omlx.ai/benchmarks/intelligence/abc12345",
            "raw_uploaded": True,
        }
        assert mock_to_thread.await_count == 2

        # Summary POST: correct endpoint, no question_results, full metadata.
        post_call = mock_to_thread.await_args_list[0]
        assert post_call.args[1] == accuracy_upload.OMLX_AI_INTEL_API_URL
        payload = post_call.kwargs["json"]
        assert "question_results" not in payload
        assert payload["benchmark"] == "mmlu"
        assert payload["accuracy"] == 0.75
        assert payload["correct_count"] == 90
        assert payload["total_questions"] == 120
        assert payload["dataset_total"] == 14042
        assert payload["model_repo"] == "mlx-community/Qwen3-4bit"
        assert payload["category_counts"] == {"anatomy": [4, 4]}
        assert payload["owner_hash"] == "h" * 64 + "a"
        assert payload["feature_flags"][0]["key"] == "turboquant_kv_4bit"
        assert payload["raw_truncated"] is False
        assert payload["raw_size"] > 0

        # Raw PUT: gzip body, display hash (verify char stripped), no
        # question text anywhere in the decompressed records.
        put_call = mock_to_thread.await_args_list[1]
        assert put_call.args[1] == (
            f"{accuracy_upload.OMLX_AI_INTEL_API_URL}/abc12345/raw"
            f"?owner_hash={'h' * 64}"
        )
        raw = gzip.decompress(put_call.kwargs["data"])
        records = json.loads(raw)
        assert len(records) == 4
        assert all("question" not in r for r in records)
        assert put_call.kwargs["headers"] == {"Content-Type": "application/gzip"}

    @pytest.mark.asyncio
    async def test_duplicate_409_is_success_and_skips_raw(self):
        post_resp = _response(409, {
            "existing_id": "dup00001",
            "existing_url": "https://omlx.ai/benchmarks/intelligence/dup00001",
        })
        mock_to_thread = AsyncMock(return_value=post_resp)

        with patch("asyncio.to_thread", mock_to_thread):
            outcome = await upload_intelligence_result(
                MagicMock(), _ctx(), _result_data()
            )

        assert outcome["duplicate"] is True
        assert outcome["id"] == "dup00001"
        assert mock_to_thread.await_count == 1  # no raw PUT

    @pytest.mark.asyncio
    async def test_server_error_returns_error_no_raise(self):
        post_resp = _response(400, {"error": "bad payload"})
        post_resp.headers = {}
        post_resp.text = '{"error": "bad payload"}'
        mock_to_thread = AsyncMock(return_value=post_resp)

        with patch("asyncio.to_thread", mock_to_thread):
            outcome = await upload_intelligence_result(
                MagicMock(), _ctx(), _result_data()
            )

        assert outcome == {"error": "bad payload"}
        assert mock_to_thread.await_count == 1

    @pytest.mark.asyncio
    async def test_network_exception_returns_error_no_raise(self):
        mock_to_thread = AsyncMock(side_effect=OSError("connection refused"))

        with patch("asyncio.to_thread", mock_to_thread):
            outcome = await upload_intelligence_result(
                MagicMock(), _ctx(), _result_data()
            )

        assert "error" in outcome
        assert "connection refused" in outcome["error"]

    @pytest.mark.asyncio
    async def test_raw_failure_keeps_summary_success(self):
        post_resp = _response(201, {"id": "abc12345", "url": "u"})
        put_resp = _response(500, {})
        put_resp.headers = {}
        put_resp.text = "oops"
        mock_to_thread = AsyncMock(side_effect=[post_resp, put_resp])

        with patch("asyncio.to_thread", mock_to_thread):
            outcome = await upload_intelligence_result(
                MagicMock(), _ctx(), _result_data()
            )

        assert outcome["id"] == "abc12345"
        assert outcome["raw_uploaded"] is False
        assert "error" not in outcome

    @pytest.mark.asyncio
    async def test_below_min_questions_skips_upload_entirely(self):
        mock_to_thread = AsyncMock()

        with patch("asyncio.to_thread", mock_to_thread):
            outcome = await upload_intelligence_result(
                MagicMock(), _ctx(), _result_data(total=50, correct=38)
            )

        assert outcome == {"skipped": "min_questions"}
        mock_to_thread.assert_not_awaited()

    def test_category_counts_derivation(self):
        from omlx.admin.accuracy_upload import _category_counts

        questions = [
            _question(0, category="anatomy", correct=True),
            _question(1, category="anatomy", correct=False),
            _question(2, category="biology", correct=True),
            _question(3, category=None),
        ]
        assert _category_counts(questions) == {
            "anatomy": [1, 2],
            "biology": [1, 1],
        }
        assert _category_counts([]) is None
        # HellaSwag's 192 activity labels must survive the cap.
        hellaswag_like = [_question(i, category=f"act{i % 192}") for i in range(200)]
        assert len(_category_counts(hellaswag_like)) == 192
        # Over the server-side key cap the whole map is dropped, never sent.
        many = [_question(i, category=f"cat{i}") for i in range(251)]
        assert _category_counts(many) is None

    @pytest.mark.asyncio
    async def test_no_owner_hash_skips_raw_upload(self):
        post_resp = _response(201, {"id": "abc12345", "url": "u"})
        mock_to_thread = AsyncMock(return_value=post_resp)

        with patch("asyncio.to_thread", mock_to_thread):
            outcome = await upload_intelligence_result(
                MagicMock(), _ctx(owner_hash_full=None), _result_data()
            )

        assert outcome["raw_uploaded"] is False
        assert mock_to_thread.await_count == 1
        payload = mock_to_thread.await_args_list[0].kwargs["json"]
        assert "owner_hash" not in payload
