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
    ``language_model.mtp.layers.0`` with layer 0; counting them there would
    inflate the wrong stages. Vision exclusion is the caller's own
    ``text_only`` intent (a VLM checkpoint sizes full unless a caller is
    specifically sizing a text-only deployment of it -- see
    ``inspect_safetensors_layout``'s docstring); MTP exclusion from
    ``layer_weight_bytes`` is unconditional, but (#2970) MTP heads are never
    sharded by mlx-lm's native TP -- they're replicated on every rank just
    like embed/norm -- so they land in ``fixed_weight_bytes`` instead of
    being dropped outright (see ``test_cluster_planner.py``'s
    ``fixed_weight_bytes`` tests for the rationale)."""

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

    # Un-flagged (catalogue) sizing: full size, vision included, MTP always
    # routed to fixed_weight_bytes (never sharded, replicated on every rank
    # -- #2970) rather than left in layer_weight_bytes.
    full = inspect_safetensors_layout(tmp_path)
    assert full.layer_weight_bytes == (1299, 1299)
    assert full.fixed_weight_bytes == 150 + 777  # embed + norm + replicated MTP head

    # A caller sizing a concrete text-only deployment of this checkpoint
    # excludes vision too — only the two real decoder layers, equal-sized.
    # MTP is unaffected by text_only -- it's excluded from layer sharding
    # unconditionally, not because of a vision/text-only distinction.
    text_only = inspect_safetensors_layout(tmp_path, text_only=True)
    assert text_only.layer_weight_bytes == (300, 300)
    assert text_only.fixed_weight_bytes == 150 + 777  # embed + norm + MTP head


def test_pure_text_layout_keeps_vision_irrelevant_but_excludes_mtp(tmp_path):
    """A non-VLM checkpoint has no vision prefixes to exclude either way, but
    a pure-text ``-mtp`` checkpoint has the exact same layer-0 collision as a
    VLM's MTP heads, so MTP routing (unconditionally out of layer sharding,
    into fixed_weight_bytes as replicated weight -- #2970) must not be gated
    on VLM-ness."""

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "num_hidden_layers": 2})
    )
    _write_safetensors(
        tmp_path / "model.safetensors",
        [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.mlp.down_proj.weight", 300),
            ("model.layers.1.mlp.down_proj.weight", 300),
            ("mtp.layers.0.weight", 777),  # collides w/ layer 0, no VLM in sight
            ("model.norm.weight", 50),
        ],
    )

    layout = inspect_safetensors_layout(tmp_path)
    assert layout.layer_weight_bytes == (300, 300)
    assert layout.fixed_weight_bytes == 150 + 777  # embed + norm + replicated MTP head


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


def test_reject_image_content_for_text_only_deployment_rejects_only_when_flagged():
    """#2845 review: the fail-closed rejection must be a single choke point
    every route can share, not something each route re-derives -- covers
    the text-only + image case (400), text-only + text (fine), a
    non-text-only deployment (guard does not apply), and a local (no
    ``deployment`` attribute at all) engine (guard does not apply)."""

    from fastapi import HTTPException

    from omlx.server import _reject_image_content_for_text_only_deployment

    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
            ],
        }
    ]
    text_messages = [{"role": "user", "content": "hello"}]

    text_only_engine = SimpleNamespace(deployment=SimpleNamespace(text_only=True))
    with pytest.raises(HTTPException) as exc_info:
        _reject_image_content_for_text_only_deployment(text_only_engine, image_messages)
    assert exc_info.value.status_code == 400

    # Plain text is always fine, even on a text-only deployment.
    _reject_image_content_for_text_only_deployment(text_only_engine, text_messages)

    # A non-text-only distributed deployment does not apply the guard.
    vision_engine = SimpleNamespace(deployment=SimpleNamespace(text_only=False))
    _reject_image_content_for_text_only_deployment(vision_engine, image_messages)

    # A local (non-distributed) engine has no ``deployment`` attribute at
    # all -- must not raise, not even AttributeError.
    local_engine = SimpleNamespace()
    _reject_image_content_for_text_only_deployment(local_engine, image_messages)


def test_reject_image_content_choke_point_wired_into_both_routes():
    """The specific #2845 blocker: /v1/messages must call the same
    pre-extraction choke point /v1/chat/completions does, not silently drop
    images via ``preserve_images`` keying on ``is_vlm`` (False for the
    distributed engine)."""

    import inspect

    from omlx import server

    chat_source = inspect.getsource(server.create_chat_completion)
    anthropic_source = inspect.getsource(server.create_anthropic_message)
    assert "_reject_image_content_for_text_only_deployment" in chat_source
    assert "_reject_image_content_for_text_only_deployment" in anthropic_source
