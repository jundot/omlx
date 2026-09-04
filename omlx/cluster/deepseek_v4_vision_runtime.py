# SPDX-License-Identifier: Apache-2.0
"""MLX-LM server bridge for the coordinator-owned DeepSeek-V4 vision path."""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from contextlib import contextmanager, suppress
from dataclasses import replace
from typing import Any

from omlx.deepseek_v4_vision import (
    IMAGE_END,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
)
from omlx.patches.deepseek_v4.vision_inputs import prepare_token_ids

logger = logging.getLogger(__name__)
_IMAGE_TYPES = frozenset({"image", "image_url", "input_image"})


def vision_prefill_chunks(
    tokens,
    *,
    vocab_size: int,
    max_chunk_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """Partition a prompt without ever splitting an image sentinel block.

    All ranks receive the same expanded token IDs, so this pure function also
    gives them an identical sequence of model calls without another control
    collective. Chunks stay at or below the configured prefill size unless an
    image block itself is larger, in which case exactly that block sets the
    minimum safe size.
    """

    values = [int(token) for token in tokens]
    if not values:
        return ()
    configured = max(1, int(max_chunk_tokens))

    spans = vision_token_spans(values, vocab_size=vocab_size)

    chunks: list[tuple[int, int]] = []
    position = 0
    total = len(values)

    def append_text_until(limit: int) -> None:
        nonlocal position
        while position < limit:
            # Keep ordinary text chunks on the configured global cadence so
            # prompt snapshots remain reusable across requests.
            boundary = min(((position // configured) + 1) * configured, limit)
            if boundary <= position:  # pragma: no cover - defensive invariant
                raise RuntimeError("could not construct a text prefill chunk")
            chunks.append((position, boundary))
            position = boundary

    for begin, end in spans:
        append_text_until(begin)
        # Isolate every image block even when it would fit inside a normal
        # text chunk. Image visibility disables standard causal kernels for
        # the whole model call; mixing surrounding text into that call can
        # produce a much larger attention transient and hang Metal before the
        # next progress keepalive.
        chunks.append((begin, end))
        position = end
    append_text_until(total)
    return tuple(chunks)


def vision_token_spans(
    tokens,
    *,
    vocab_size: int,
) -> tuple[tuple[int, int], ...]:
    """Return image-sentinel spans, accepting a cache suffix inside a block."""

    values = [int(token) for token in tokens]
    spans: list[tuple[int, int]] = []
    # An image block can have up to three alignment pads before IMAGE_START.
    # Treat the complete contiguous sentinel run as the protected span. A
    # prompt-cache hit may also leave this suffix part-way through a block.
    start = None
    saw_start = False
    for index, token in enumerate(values):
        kind = token - int(vocab_size)
        if kind < 0:
            if start is not None:
                raise ValueError("DeepSeek image sentinel block is incomplete")
            continue
        if kind > IMAGE_END:
            raise ValueError(f"invalid DeepSeek image sentinel kind {kind}")
        if start is None:
            if kind == IMAGE_END:
                if index == 0:
                    # The cached prefix can end immediately before IMAGE_END.
                    spans.append((0, 1))
                    continue
                raise ValueError("DeepSeek image end sentinel has no start")
            start = index
            saw_start = kind == IMAGE_START
            continue
        if kind == IMAGE_START:
            if saw_start:
                raise ValueError("nested DeepSeek image sentinel blocks are invalid")
            saw_start = True
        elif kind == IMAGE_END:
            if not saw_start and start != 0:
                raise ValueError("DeepSeek image end sentinel has no start")
            spans.append((start, index + 1))
            start = None
            saw_start = False
    if start is not None:
        raise ValueError("DeepSeek image sentinel block is incomplete")
    return tuple(spans)


def _prefill_prompt_chunk(model, prompt_cache, tokens, kwargs) -> None:
    """Run one non-final prefill chunk exactly like mlx-lm ``generate_step``."""

    import mlx.core as mx
    from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache

    with mx.stream(generation_stream):
        model(mx.array(tokens)[None], cache=prompt_cache)
        maybe_quantize_kv_cache(
            prompt_cache,
            kwargs.get("quantized_kv_start", 0),
            kwargs.get("kv_group_size", 64),
            kwargs.get("kv_bits"),
        )
        mx.eval([cache.state for cache in prompt_cache])
        mx.clear_cache()


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
            instance._omlx_vision_vocab_size = None

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
        self._omlx_vision_vocab_size = None
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
        prepared_images = prepared_images_by_digest.pop(digest, None) if rank == 0 else None
        model_args = getattr(self.model_provider.model, "args", None)
        vocab_size = int(
            config.get("vocab_size")
            or getattr(model_args, "vocab_size", 0)
            or 0
        )
        spans = (
            vision_token_spans(prompt, vocab_size=vocab_size)
            if vocab_size > 0
            else ()
        )
        self.model_provider.model.set_vision_inputs(prepared_images, spans=spans)
        self.model_provider.model_key = (*base_model_key, "vision", digest)
        self._omlx_prefill_step_size_override = True
        self._omlx_vision_vocab_size = vocab_size
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
            configured_step = max(
                1, int(kwargs.get("prefill_step_size", 0) or 2048)
            )
            model = kwargs.get("model") or (args[0] if args else None)
            model_args = getattr(model, "args", None)
            vocab_size = int(
                config.get("vocab_size")
                or getattr(model_args, "vocab_size", 0)
                or 0
            )
            chunks = (
                vision_prefill_chunks(
                    prompt,
                    vocab_size=vocab_size,
                    max_chunk_tokens=configured_step,
                )
                if vocab_size > 0
                else ((0, len(prompt)),)
            )
            started = time.perf_counter()
            logger.info(
                "deepseek_v4_vision stage=distributed_prefill_begin rank=%d "
                "sequence_length=%d chunks=%d max_chunk=%d",
                rank,
                len(prompt),
                len(chunks),
                max((end - start for start, end in chunks), default=0),
            )

            # mlx-lm accepts one fixed prefill size. Process every variable
            # image-safe prefix chunk here, then give its generator the final
            # chunk and the already-advanced cache. This retains its sampling,
            # logits-processor, cancellation, and cache-insertion behavior.
            prompt_cache = kwargs.get("prompt_cache")
            draft_model = kwargs.get("draft_model")
            manual_chunks = bool(
                len(chunks) > 1
                and model is not None
                and prompt_cache is not None
                and draft_model is None
            )
            original_prompt_tokens = len(prompt)
            processed = 0
            progress = kwargs.get("prompt_progress_callback")
            call_args = list(args)
            call_kwargs = dict(kwargs)
            if manual_chunks:
                if progress is not None:
                    progress(0, original_prompt_tokens)
                for begin, end in chunks[:-1]:
                    _prefill_prompt_chunk(
                        model,
                        prompt_cache,
                        prompt[begin:end],
                        kwargs,
                    )
                    processed = end
                    if progress is not None:
                        progress(processed, original_prompt_tokens)
                tail_start, tail_end = chunks[-1]
                tail = prompt[tail_start:tail_end]
                if "prompt" in call_kwargs:
                    call_kwargs["prompt"] = tail
                elif len(call_args) >= 3:
                    call_args[2] = tail
                call_kwargs["prefill_step_size"] = max(
                    configured_step, len(tail)
                )
                if progress is not None:
                    call_kwargs["prompt_progress_callback"] = (
                        lambda done, _total: progress(
                            processed + int(done), original_prompt_tokens
                        )
                    )
            else:
                call_kwargs["prefill_step_size"] = max(
                    configured_step,
                    max((end - start for start, end in chunks), default=0),
                )

            first = True
            steady = False
            prompt_tps = None
            try:
                for result in original_stream_generate(*call_args, **call_kwargs):
                    if first:
                        first = False
                        elapsed = time.perf_counter() - started
                        prompt_tps = original_prompt_tokens / max(elapsed, 1e-9)
                        logger.info(
                            "deepseek_v4_vision stage=distributed_prefill_complete "
                            "rank=%d elapsed_ms=%.1f",
                            rank,
                            elapsed * 1000,
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
                    if manual_chunks:
                        with suppress(TypeError):
                            result = replace(
                                result,
                                prompt_tokens=original_prompt_tokens,
                                prompt_tps=prompt_tps,
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
