# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-model coverage for ``unsloth/Qwen3.8-27B-NVFP4``.

The test never downloads weights. Run it explicitly with the published local
checkpoint to exercise strict loading, text generation, and the vision path::

    OMLX_QWEN38_MODELOPT_MODEL_PATH=/absolute/path/to/Qwen3.8-27B-NVFP4 \
        pytest tests/integration/test_qwen38_modelopt_mixed_real_model.py \
        -m slow -s -q
"""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import json
import os
import platform
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason="Qwen3.8 ModelOpt integration requires macOS on Apple Silicon.",
    ),
]

_ENV_VAR = "OMLX_QWEN38_MODELOPT_MODEL_PATH"


@pytest.fixture(scope="module")
def qwen38_model_path() -> Path:
    configured = os.environ.get(_ENV_VAR)
    if not configured:
        pytest.skip(f"Set {_ENV_VAR} to run this real-model test.")
    model_path = Path(configured).expanduser()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        pytest.skip(f"Qwen3.8 config.json not found at {config_path}")

    from omlx.patches.qwen38_modelopt_mixed import is_supported_config

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert is_supported_config(config), (
        f"{_ENV_VAR} must point to the validated mixed ModelOpt "
        "Qwen3.8-27B VLM checkpoint."
    )
    return model_path


def _red_blue_data_uri() -> str:
    from PIL import Image

    image = Image.new("RGB", (128, 64), (255, 0, 0))
    image.paste((0, 0, 255), (64, 0, 128, 64))
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    encoded = base64.b64encode(payload.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_qwen38_modelopt_mixed_text_and_vision(qwen38_model_path: Path):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.patches.qwen38_modelopt_mixed import ScaledQuantizedLinear

    async def validate() -> None:
        engine = VLMBatchedEngine(model_name=str(qwen38_model_path))
        try:
            await engine.start()
            leaves = tree_flatten(
                engine._vlm_model.leaf_modules(),
                is_leaf=lambda module: isinstance(module, ScaledQuantizedLinear),
            )
            assert (
                sum(isinstance(module, ScaledQuantizedLinear) for _, module in leaves)
                == 401
            )

            text = await engine.chat(
                [{"role": "user", "content": "Reply with the word OK."}],
                max_tokens=16,
                temperature=0.0,
            )
            assert text.completion_tokens > 0

            vision = await engine.chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": _red_blue_data_uri()},
                            },
                            {
                                "type": "text",
                                "text": "Name the left and right colors briefly.",
                            },
                        ],
                    }
                ],
                max_tokens=32,
                temperature=0.0,
            )
            assert vision.completion_tokens > 0
            assert "red" in vision.text.lower()
            assert "blue" in vision.text.lower()
        finally:
            await engine.stop()
            gc.collect()
            mx.clear_cache()

    asyncio.run(validate())
