# SPDX-License-Identifier: Apache-2.0
"""MLX-LM server bridge for the coordinator-owned DeepSeek-V4 vision path."""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from omlx.deepseek_v4_vision import IMAGE_PLACEHOLDER
from omlx.patches.deepseek_v4.vision_inputs import prepare_token_ids

logger = logging.getLogger(__name__)
_IMAGE_TYPES = frozenset({"image", "image_url", "input_image"})


def _request_has_images(request: Any) -> bool:
    for message in getattr(request, "messages", ()) or ():
        for part in message.get("content", ()) if isinstance(message, dict) else ():
            if isinstance(part, dict) and part.get("type") in _IMAGE_TYPES:
                return True
    return False


def _text_messages_and_images(messages):
    rendered = copy.deepcopy(messages)
    images = []
    for message in rendered:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        replacement = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in _IMAGE_TYPES:
                replacement.append(part)
                continue
            images.append(part.get("image_url", part))
            replacement.append({"type": "text", "text": IMAGE_PLACEHOLDER})
        message["content"] = replacement
    return rendered, images


def _image_digest(images) -> str:
    digest = hashlib.sha256()
    for image in images:
        digest.update(memoryview(image.patches))
        digest.update(str((image.n_vit_h, image.n_vit_w, image.types)).encode())
    return digest.hexdigest()


@contextmanager
def install_deepseek_v4_vision_runtime(
    server_module: Any,
    provider: Any,
    *,
    config: dict[str, Any],
    rank: int,
):
    """Teach the pinned text server to perform one rank-zero VLM prefill.

    Only rank zero downloads/decodes images and executes the ViT. Expanded
    sentinel token IDs are shared through the existing small control
    collective; MLX embeddings are broadcast natively inside the model.
    """

    response_type = server_module.ResponseGenerator
    original_tokenize = response_type._tokenize
    original_share_request = getattr(response_type, "_share_request", None)
    original_serve_single = getattr(response_type, "_serve_single", None)
    original_stream_generate = server_module.stream_generate
    base_model_key = provider.model_key
    prepared_images_by_digest = {}
    provider.is_batchable = False

    def clear_request_state(instance=None):
        provider.model_key = base_model_key
        provider.model.set_vision_inputs(None)
        prepared_images_by_digest.clear()
        if instance is not None:
            instance._omlx_prefill_step_size_override = False

    def prepare_shared_request(self, request, args):
        try:
            messages, image_records = _text_messages_and_images(request.messages)
            shared_request = replace(request, messages=messages)
            prompt, _segments, _types, initial_state = original_tokenize(
                self,
                self.model_provider.tokenizer,
                shared_request,
                args,
            )
            image_token_id = self.model_provider.tokenizer.convert_tokens_to_ids(
                IMAGE_PLACEHOLDER
            )
            if (
                image_token_id is None
                or image_token_id == self.model_provider.tokenizer.unk_token_id
            ):
                raise ValueError(
                    "DeepSeek image placeholder is missing from tokenizer: "
                    f"{IMAGE_PLACEHOLDER}"
                )
            prompt, prepared_images = prepare_token_ids(
                prompt,
                image_records,
                image_token_id=int(image_token_id),
                config=config,
            )
            digest = _image_digest(prepared_images)
            prepared_images_by_digest.clear()
            prepared_images_by_digest[digest] = prepared_images
            metadata = ("ok", prompt, initial_state, digest)
        except Exception as exc:
            logger.exception("deepseek_v4_vision stage=multimodal_prompt_failed rank=0")
            prepared_images_by_digest.clear()
            try:
                shared_request = replace(request, messages=[])
            except TypeError:
                shared_request = copy.copy(request)
                shared_request.messages = []
            metadata = ("error", type(exc).__name__, str(exc), None)
        shared_request._omlx_vision_metadata = metadata
        return shared_request

    def share_request(self, request):
        if rank == 0 and request is not None:
            queue, request_payload, request_args = request
            if _request_has_images(request_payload):
                request_payload = prepare_shared_request(
                    self,
                    request_payload,
                    request_args,
                )
                request = (queue, request_payload, request_args)
        try:
            shared_request = original_share_request(self, request)
            if shared_request is None:
                clear_request_state(self)
            return shared_request
        except Exception:
            clear_request_state(self)
            raise

    def tokenize(self, tokenizer, request, args):
        self._omlx_prefill_step_size_override = False
        metadata = getattr(request, "_omlx_vision_metadata", None)
        if metadata is None:
            clear_request_state(self)
            return original_tokenize(self, tokenizer, request, args)
        if not isinstance(metadata, tuple) or len(metadata) != 4:
            raise RuntimeError("DeepSeek-V4 vision request metadata is invalid")
        status, prompt_or_type, initial_state_or_message, digest = metadata
        if status == "error":
            raise RuntimeError(
                f"DeepSeek-V4 vision preprocessing failed on rank zero "
                f"({prompt_or_type}): {initial_state_or_message}"
            )
        if status != "ok" or not isinstance(digest, str):
            raise RuntimeError("DeepSeek-V4 vision request metadata is invalid")
        prompt = prompt_or_type
        initial_state = initial_state_or_message
        prepared_images = prepared_images_by_digest.get(digest) if rank == 0 else None
        self.model_provider.model.set_vision_inputs(prepared_images)
        self.model_provider.model_key = (*base_model_key, "vision", digest)
        self._omlx_prefill_step_size_override = True
        logger.info(
            "deepseek_v4_vision stage=multimodal_prompt_ready rank=%d "
            "images=%d sequence_length=%d",
            rank,
            len(prepared_images or ()),
            len(prompt),
        )
        # Image caches are content-keyed, not token-layout-keyed. A single
        # segment also prevents the server from caching a system prefix that
        # crosses the image replacement boundary.
        return prompt, [prompt], ["assistant"], initial_state

    def serve_single(self, request):
        try:
            return original_serve_single(self, request)
        finally:
            # Keep the image digest through both cache lookup and insertion,
            # then return ModelProvider to the identity expected by load().
            clear_request_state(self)

    def stream_generate(*args, **kwargs):
        prompt = kwargs.get("prompt")
        if prompt is None and len(args) >= 3:
            prompt = args[2]
        if provider.model_key != base_model_key and prompt is not None:
            kwargs["prefill_step_size"] = max(
                int(kwargs.get("prefill_step_size", 0) or 0), len(prompt)
            )
            started = time.perf_counter()
            logger.info(
                "deepseek_v4_vision stage=distributed_prefill_begin rank=%d "
                "sequence_length=%d",
                rank,
                len(prompt),
            )
            first = True
            steady = False
            try:
                for result in original_stream_generate(*args, **kwargs):
                    if first:
                        first = False
                        logger.info(
                            "deepseek_v4_vision stage=distributed_prefill_complete "
                            "rank=%d elapsed_ms=%.1f",
                            rank,
                            (time.perf_counter() - started) * 1000,
                        )
                        logger.info(
                            "deepseek_v4_vision stage=first_token rank=%d", rank
                        )
                    elif not steady:
                        steady = True
                        logger.info(
                            "deepseek_v4_vision stage=steady_decode_begin rank=%d",
                            rank,
                        )
                    yield result
            finally:
                logger.info("deepseek_v4_vision stage=request_complete rank=%d", rank)
            return
        yield from original_stream_generate(*args, **kwargs)

    response_type._tokenize = tokenize
    if original_share_request is not None:
        response_type._share_request = share_request
    if original_serve_single is not None:
        response_type._serve_single = serve_single
    server_module.stream_generate = stream_generate
    try:
        logger.info(
            "deepseek_v4_vision stage=runtime_ready rank=%d vision_owner=%s",
            rank,
            rank == 0,
        )
        yield
    finally:
        clear_request_state()
        response_type._tokenize = original_tokenize
        if original_share_request is not None:
            response_type._share_request = original_share_request
        if original_serve_single is not None:
            response_type._serve_single = original_serve_single
        server_module.stream_generate = original_stream_generate
        logger.info("deepseek_v4_vision stage=teardown rank=%d", rank)
