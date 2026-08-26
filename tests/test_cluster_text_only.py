"""Text-only distributed serving of VLM checkpoints (tier a).

Covers the pieces that let a VLM-shaped checkpoint run its language model across
the cluster with vision disabled: planner accounting that ignores the vision
tower and MTP heads, pipeline-stage validation of the wrapper model shape, and
image-request rejection on a text-only deployment.
"""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import pytest

from omlx.cluster.planner import PipelineAssignment, inspect_safetensors_layout


def _write_safetensors(path, tensors):
    offset = 0
    header = {}
    for name, size in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset)


def test_text_only_layout_excludes_vision_and_mtp(tmp_path):
    """Vision-tower and MTP tensors must not be sized into decoder layers.

    ``vision_tower.blocks.N`` collides with decoder layer N and
    ``language_model.mtp.layers.0`` with layer 0; counting them inflated the
    wrong stages. A VLM checkpoint (non-empty ``vision_config``) excludes them.
    """

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "num_hidden_layers": 2,
                "vision_config": {"depth": 4},
            }
        )
    )
    _write_safetensors(
        tmp_path / "model.safetensors",
        [
            ("language_model.model.embed_tokens.weight", 100),
            ("language_model.model.layers.0.mlp.down_proj.weight", 300),
            ("language_model.model.layers.1.mlp.down_proj.weight", 300),
            ("vision_tower.blocks.0.attn.qkv.weight", 999),  # collides w/ layer 0
            ("vision_tower.blocks.1.attn.qkv.weight", 999),  # collides w/ layer 1
            ("language_model.mtp.layers.0.weight", 777),  # collides w/ layer 0
            ("language_model.model.norm.weight", 50),
        ],
    )

    layout = inspect_safetensors_layout(tmp_path)

    # Only the two real decoder layers, equal-sized — no vision/MTP inflation.
    assert layout.layer_weight_bytes == (300, 300)
    assert layout.fixed_weight_bytes == 150  # embed + norm only


def test_pure_text_layout_keeps_all_tensors(tmp_path):
    """A non-VLM checkpoint (no vision_config) is unaffected by the filter."""

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "num_hidden_layers": 2})
    )
    _write_safetensors(
        tmp_path / "model.safetensors",
        [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.mlp.down_proj.weight", 300),
            ("model.layers.1.mlp.down_proj.weight", 300),
            ("model.norm.weight", 50),
        ],
    )

    layout = inspect_safetensors_layout(tmp_path)
    assert layout.layer_weight_bytes == (300, 300)
    assert layout.fixed_weight_bytes == 150


def _assignment(start, end, *, tp=2):
    return PipelineAssignment(
        node_id="n",
        rank=0,
        start_layer=start,
        end_layer=end,
        layer_weight_bytes=0,
        fixed_weight_bytes=0,
        reserve_bytes=0,
        capacity_bytes=0,
        tensor_parallel_size=tp,
    )


def test_validate_loaded_stage_accepts_language_model_wrapper():
    """The qwen3_5 VLM family nests layers under language_model.model."""

    from omlx.cluster.inference_worker import _validate_loaded_stage

    layers = [object(), object(), object()]
    text_model = SimpleNamespace(layers=layers, start_idx=0, end_idx=3)
    model = SimpleNamespace(language_model=SimpleNamespace(model=text_model))

    # Complete TP stage over all three layers — must not raise.
    _validate_loaded_stage(model, _assignment(0, 3, tp=2))


def test_validate_loaded_stage_accepts_classic_shape():
    from omlx.cluster.inference_worker import _validate_loaded_stage

    layers = [object(), object()]
    text_model = SimpleNamespace(layers=layers, start_idx=0, end_idx=2)
    model = SimpleNamespace(model=text_model)

    _validate_loaded_stage(model, _assignment(0, 2, tp=2))


def test_validate_loaded_stage_still_fails_closed_on_mismatch():
    from omlx.cluster.inference_worker import _validate_loaded_stage

    layers = [object(), object(), object()]
    text_model = SimpleNamespace(layers=layers, start_idx=0, end_idx=2)
    model = SimpleNamespace(language_model=SimpleNamespace(model=text_model))

    # Pipeline range [0,2) does not match the approved [0,3) at tp=1.
    with pytest.raises(RuntimeError):
        _validate_loaded_stage(model, _assignment(0, 3, tp=1))


def test_reject_images_only_when_text_only():
    from omlx.engine.distributed import DistributedBatchedEngine

    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ],
        }
    ]
    text_messages = [{"role": "user", "content": "hello"}]

    text_only = SimpleNamespace(deployment=SimpleNamespace(text_only=True))
    with pytest.raises(ValueError, match="text-only"):
        DistributedBatchedEngine._reject_images_if_text_only(text_only, image_messages)
    # Plain text is always fine.
    DistributedBatchedEngine._reject_images_if_text_only(text_only, text_messages)

    # A non-text-only deployment does not apply the guard here.
    not_text_only = SimpleNamespace(deployment=SimpleNamespace(text_only=False))
    DistributedBatchedEngine._reject_images_if_text_only(not_text_only, image_messages)


def test_request_has_image_content_detects_image_parts():
    """The server-level detector (the path that actually fires) sees images.

    The chat route strips images to text for text engines *before* the engine
    is called, so the effective rejection runs at the request layer.
    """

    from omlx.server import _request_has_image_content

    text = [{"role": "user", "content": "hello"}]
    with_image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
            ],
        }
    ]
    assert _request_has_image_content(text) is False
    assert _request_has_image_content(with_image) is True
