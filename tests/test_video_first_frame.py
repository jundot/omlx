# SPDX-License-Identifier: Apache-2.0
"""Tests for T2I->I2V first-frame generation (omlx.api.first_frame)."""

import asyncio
import types

import pytest

import omlx.api.image_routes as image_routes
from omlx.api.first_frame import FirstFrameError, generate_first_frame


def _entry(tmp_path, model_type="image", pipeline="t2i"):
    return types.SimpleNamespace(
        model_type=model_type,
        image_pipeline=pipeline,
        image_alias="z-image-turbo",
        model_path=tmp_path / "models" / "zimg",
    )


class _Pool:
    def __init__(self, entry):
        self._entry = entry

    def get_entry(self, mid):
        return self._entry


class _Manager:
    """Fake MediaJobManager: submit records, wait_terminal returns a job
    whose artifact file exists on disk."""

    def __init__(self, tmp_path, status="completed", produce=True, timeout=False):
        self._tmp = tmp_path
        self._status = status
        self._produce = produce
        self._timeout = timeout
        self.submitted = []

    async def submit(self, job, **kw):
        self.submitted.append(job)

    async def wait_terminal(self, job_id, timeout):
        if self._timeout:
            return None
        blob = self._tmp / "art" / job_id
        blob.mkdir(parents=True, exist_ok=True)
        files = []
        if self._produce:
            (blob / "frame.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
            files = ["frame.png"]
        return types.SimpleNamespace(
            status=self._status,
            error="boom" if self._status == "failed" else None,
            artifact_path=str(blob / "frame.png") if self._produce else None,
            artifact_files=files,
        )


@pytest.fixture(autouse=True)
def _patch_image_routes(monkeypatch):
    monkeypatch.setattr(image_routes, "_resolve_model", lambda m: m)
    monkeypatch.setattr(
        image_routes,
        "_normalize_params",
        lambda params, entry, isettings, msettings, n: {
            "alias": "z-image-turbo",
            "prompt": params.prompt,
            "width": params.width,
            "height": params.height,
            "response_format": "b64_json",
        },
    )


def _run(entry, manager):
    return asyncio.run(
        generate_first_frame(
            "一个人伸懒腰",
            width=480,
            height=272,
            model_id="z-image-turbo",
            manager=manager,
            engine_pool=_Pool(entry),
            settings_manager=None,
            image_settings=object(),
            timeout_s=5.0,
        )
    )


def test_success_returns_bytes(tmp_path):
    data, suffix = _run(_entry(tmp_path), _Manager(tmp_path))
    assert data.startswith(b"\x89PNG")
    assert suffix == ".png"


def test_unknown_model_raises(tmp_path):
    with pytest.raises(FirstFrameError, match="not found"):
        _run(None, _Manager(tmp_path))


def test_non_image_model_raises(tmp_path):
    with pytest.raises(FirstFrameError, match="not an image model"):
        _run(_entry(tmp_path, model_type="video"), _Manager(tmp_path))


def test_edit_pipeline_rejected(tmp_path):
    with pytest.raises(FirstFrameError, match="text-to-image"):
        _run(_entry(tmp_path, pipeline="edit"), _Manager(tmp_path))


def test_generation_failed_raises(tmp_path):
    with pytest.raises(FirstFrameError, match="failed"):
        _run(_entry(tmp_path), _Manager(tmp_path, status="failed", produce=False))


def test_timeout_raises(tmp_path):
    with pytest.raises(FirstFrameError, match="timed out"):
        _run(_entry(tmp_path), _Manager(tmp_path, timeout=True))


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(FirstFrameError, match="no artifact"):
        _run(_entry(tmp_path), _Manager(tmp_path, produce=False))
