# SPDX-License-Identifier: Apache-2.0
"""Tests for oMLX's mlx-vlm image loading policy."""

from __future__ import annotations

import pytest

from omlx.patches.mlx_vlm_image_compat import apply_mlx_vlm_image_compat_patch


def test_z_image_tokenizer_disables_remote_code(monkeypatch):
    pytest.importorskip("mlx_vlm")
    import transformers

    apply_mlx_vlm_image_compat_patch()
    from mlx_vlm.models.z_image import pipeline

    received = {}

    def fake_from_pretrained(*args, **kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )

    pipeline.AutoTokenizer.from_pretrained(
        "/tmp/tokenizer",
        local_files_only=False,
        trust_remote_code=True,
    )

    assert received["local_files_only"] is True
    assert received["trust_remote_code"] is False
