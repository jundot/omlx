# SPDX-License-Identifier: Apache-2.0
"""Persistent OpenAI-compatible full-replica prefill/decode rank worker."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import struct
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from typing import Any

from .cache_transfer import (
    prepare_cache_transfer,
    recv_cache_transfer,
    send_cache_transfer,
)

_PROGRESS = struct.Struct("!II")
_CANCEL = struct.Struct("!B")
_OBJECT_LENGTH = struct.Struct("!Q")
_PREFILL_START = struct.Struct("!IIIIQQQQ")
_LOGITS_HEADER = struct.Struct("!III")
_DTYPE_TO_CODE = {"float16": 1, "bfloat16": 2, "float32": 3}
_CODE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CODE.items()}
_CACHE_TIER_TO_CODE = {None: 0, "memory": 1, "ssd": 2}
_CACHE_CODE_TO_TIER = {value: key for key, value in _CACHE_TIER_TO_CODE.items()}


def _phase_event(rank: int, stage: str, **details: Any) -> None:
    print(
        "OMLX_CLUSTER_EVENT:"
        + json.dumps(
            {
                "type": "phase_trace",
                "rank": rank,
                "stage": stage,
                **details,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _broadcast_owned_object(control: Any, value: Any, *, source_rank: int) -> Any:
    """Broadcast one bounded pickle owned by either phase rank."""

    payload = pickle.dumps(value, protocol=5) if control.rank == source_rank else None
    size = len(payload) if payload is not None else 0
    if size > 256 * 1024 * 1024:
        raise RuntimeError("phase control object exceeds 256 MiB")
    length_packet = control.broadcast_owned_bytes(
        _OBJECT_LENGTH.pack(size) if control.rank == source_rank else None,
        source_rank=source_rank,
        expected_size=_OBJECT_LENGTH.size,
    )
    expected = _OBJECT_LENGTH.unpack(length_packet)[0]
    if expected > 256 * 1024 * 1024:
        raise RuntimeError("phase control object has an invalid size")
    encoded = control.broadcast_owned_bytes(
        payload,
        source_rank=source_rank,
        expected_size=expected,
    )
    try:
        return pickle.loads(encoded)
    except Exception as exc:
        raise RuntimeError("phase control object is invalid") from exc


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backend", choices=("ring", "jaccl", "jaccl-ring"), required=True
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--server-host", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--plan-hash", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--state-dir", default="~/.omlx/cluster/runtime")
    parser.add_argument("--control-host", required=True)
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--control-token", required=True)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--prompt-cache-size", type=int, default=1)
    parser.add_argument("--prompt-cache-bytes", type=int, default=None)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--prompt-cache-ssd", action="store_true")
    parser.add_argument(
        "--prompt-cache-ssd-max-bytes",
        type=int,
        default=20 * 1024**3,
    )
    parser.add_argument("--decode-concurrency", type=int, default=1)
    parser.add_argument("--prompt-concurrency", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    args, _unknown = parser.parse_known_args()
    return args


def _dtype_name(dtype: Any) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _cache_states(cache: list[Any]) -> list[Any]:
    return [entry.state for entry in cache]


def _prefill_calls(prompt_tokens: int, step: int) -> int:
    if prompt_tokens <= 0:
        return 0
    if prompt_tokens == 1:
        return 1
    return (max(0, prompt_tokens - 2) // step) + 2


def _prefill_logits(
    mx: Any,
    model: Any,
    cache: list[Any],
    tokens: list[int],
    *,
    step: int,
    progress: Any,
) -> tuple[Any | None, float]:
    started = time.perf_counter()
    values = mx.array(tokens, dtype=mx.int32)
    processed = 0
    total = len(tokens)
    while total - processed > 1:
        width = min(step, (total - processed) - 1)
        _ = model(values[None, processed : processed + width], cache=cache)
        mx.eval(_cache_states(cache))
        processed += width
        if progress(processed, total) is False:
            return None, time.perf_counter() - started
        mx.clear_cache()
    logits = model(values[None, processed:], cache=cache)[:, -1, :]
    mx.eval(logits, _cache_states(cache))
    if progress(total, total) is False:
        return None, time.perf_counter() - started
    return logits, time.perf_counter() - started


def _state_machine(tokenizer: Any, stop_words: list[str]):
    from mlx_lm.generate import SequenceStateMachine

    transitions: dict[str, list[tuple[tuple[int, ...], str | None]]] = {"normal": []}
    sequences: dict[tuple[int, ...], str] = {}
    common = []
    for token in tokenizer.eos_token_ids:
        sequence = (int(token),)
        common.append((sequence, None))
        sequences[sequence] = tokenizer.convert_ids_to_tokens(int(token))
    for word in stop_words:
        sequence = tuple(tokenizer.encode(word, add_special_tokens=False))
        if sequence:
            common.append((sequence, None))
            sequences[sequence] = word
    transitions["normal"].extend(common)
    if tokenizer.has_thinking:
        start = tuple(tokenizer.think_start_tokens)
        end = tuple(tokenizer.think_end_tokens)
        transitions["normal"].append((start, "reasoning"))
        transitions["reasoning"] = [(end, "normal"), *common]
        sequences[start] = tokenizer.think_start
        sequences[end] = tokenizer.think_end
    if tokenizer.has_tool_calling:
        start = tuple(tokenizer.tool_call_start_tokens)
        end = tuple(tokenizer.tool_call_end_tokens or ())
        transitions["normal"].append((start, "tool"))
        transitions["tool"] = ([(end, "normal")] if end else []) + common
        sequences[start] = tokenizer.tool_call_start
        if end:
            sequences[end] = tokenizer.tool_call_end
    return SequenceStateMachine(transitions, initial="normal"), sequences


class _PhasePromptCache:
    """Prefill-rank hot/SSD prefix cache with exact-logit reuse."""

    def __init__(
        self,
        *,
        model: Any,
        model_key: str,
        max_size: int,
        max_bytes: int | None,
        ssd_directory: str | None,
        ssd_max_bytes: int,
        step: int,
    ) -> None:
        from mlx_lm.server import LRUPromptCache

        self.model = model
        self.model_key = model_key
        self.max_size = max(1, int(max_size))
        self.hot = LRUPromptCache(
            max_size=self.max_size,
            max_bytes=int(max_bytes) if max_bytes is not None else 1 << 63,
        )
        self.exact_logits: OrderedDict[tuple[int, ...], Any] = OrderedDict()
        self.active = threading.Event()
        self.ssd = None
        if ssd_directory:
            from .prompt_snapshot_cache import SSDPromptSnapshotStore

            self.ssd = SSDPromptSnapshotStore(
                ssd_directory,
                step=max(1, int(step)),
                max_entries=512,
                max_bytes=max(1, int(ssd_max_bytes)),
                persistent=True,
                write_behind=True,
            )

    def lookup(
        self, tokens: list[int]
    ) -> tuple[list[Any], list[int], Any | None, str | None]:
        from mlx_lm.models.cache import (
            can_trim_prompt_cache,
            make_prompt_cache,
            trim_prompt_cache,
        )

        cache, rest = self.hot.fetch_nearest_cache(self.model_key, tokens)
        if cache is not None:
            exact = self.exact_logits.get(tuple(tokens)) if not rest else None
            if exact is not None:
                return cache, [], exact, "memory"
            if not rest:
                if can_trim_prompt_cache(cache):
                    trim_prompt_cache(cache, 1)
                    rest = tokens[-1:]
                else:
                    cache = None
            if cache is not None:
                return cache, list(rest), None, "memory"

        if self.ssd is not None and len(tokens) > 1:
            boundaries = self.ssd.present_boundaries(self.model_key, tokens)
            boundary = next((value for value in boundaries if value < len(tokens)), 0)
            if boundary:
                restored = self.ssd.load(self.model_key, tokens, boundary)
                if restored is not None:
                    return restored, tokens[boundary:], None, "ssd"

        return make_prompt_cache(self.model), list(tokens), None, None

    def checkpoint(self, tokens: list[int], cache: list[Any], position: int) -> None:
        if self.ssd is not None and position > 0 and position % self.ssd.step == 0:
            self.ssd.put(self.model_key, tokens[:position], cache)

    def insert(self, tokens: list[int], cache: list[Any], logits: Any) -> None:
        key = tuple(tokens)
        self.hot.insert_cache(self.model_key, list(tokens), cache, cache_type="user")
        self.exact_logits[key] = logits
        self.exact_logits.move_to_end(key)
        while len(self.exact_logits) > self.max_size:
            self.exact_logits.popitem(last=False)

    def stats(self) -> dict[str, int]:
        return {
            "memory_entries": len(self.hot),
            "memory_bytes": int(self.hot.nbytes),
            "ssd_entries": len(self.ssd) if self.ssd is not None else 0,
            "ssd_bytes": self.ssd.nbytes if self.ssd is not None else 0,
        }

    def clear(self, *, hot: bool, ssd: bool) -> dict[str, int]:
        if self.active.is_set():
            raise RuntimeError("phase prompt cache is active")
        hot_cleared = len(self.hot) if hot else 0
        if hot:
            self.hot.trim_to(n_sequences=0, n_bytes=0)
            self.exact_logits.clear()
        ssd_deleted = self.ssd.clear(timeout=30.0) if ssd and self.ssd else 0
        return {
            "hot_cleared": hot_cleared,
            "ssd_deleted": int(ssd_deleted),
        }

    def close(self) -> None:
        if self.ssd is not None and not self.ssd.close(timeout=10.0):
            raise RuntimeError("phase prompt-cache SSD writer did not stop")


class _PhaseCacheMaintenance:
    """Poll the existing rank cache-clear protocol on the prefill owner."""

    def __init__(
        self,
        *,
        prompt_cache: _PhasePromptCache,
        state_dir: str,
        deployment_id: str,
        plan_hash: str,
        rank: int,
    ) -> None:
        root = Path(state_dir).expanduser()
        self.prompt_cache = prompt_cache
        self.request_path = root / f"{deployment_id}-cache-clear.json"
        self.ack_path = root / f"{deployment_id}-cache-clear-rank-{rank}.json"
        self.deployment_id = deployment_id
        self.plan_hash = plan_hash
        self.rank = rank
        self.epoch_floor = time.time_ns()
        self.last_epoch = self.epoch_floor
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def poll(self) -> None:
        if not self.request_path.is_file() or self.request_path.stat().st_size > 65536:
            return
        try:
            request = json.loads(self.request_path.read_text(encoding="utf-8"))
            epoch = int(request.get("epoch", 0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return
        if (
            epoch <= self.last_epoch
            or epoch < self.epoch_floor
            or request.get("deployment_id") != self.deployment_id
            or request.get("plan_hash") != self.plan_hash
        ):
            return
        self.last_epoch = epoch
        try:
            report = self.prompt_cache.clear(
                hot=bool(request.get("hot")),
                ssd=bool(request.get("ssd")),
            )
            ack = {"status": "ok", "rank": self.rank, "epoch": epoch, **report}
        except Exception as exc:
            ack = {
                "status": "error",
                "rank": self.rank,
                "epoch": epoch,
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }
        self.ack_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.ack_path.with_name(self.ack_path.name + ".tmp")
        temporary.write_text(
            json.dumps(ack, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.ack_path)

    def start(self) -> None:
        if self.thread is not None:
            return

        def run() -> None:
            while not self.stop_event.wait(0.5):
                with suppress(Exception):
                    self.poll()

        self.thread = threading.Thread(
            target=run,
            name="omlx-phase-cache-maintenance",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        thread, self.thread = self.thread, None
        if thread is not None:
            thread.join(timeout=2.0)


@dataclass
class _ServingRequest:
    request_id: int
    prompt: list[int]
    args: Any
    context: Any
    state_machine: Any
    output: Queue
    progress: Any


class _PhaseResponseGenerator:
    """APIHandler adapter and online two-stage request broker."""

    def __init__(
        self,
        *,
        mx: Any,
        model: Any,
        tokenizer: Any,
        group: Any,
        control: Any,
        model_identity: str,
        prefill_rank: int,
        decode_rank: int,
        prefill_step_size: int,
        cli_args: Any,
        stream: Any,
        deployment_id: str,
        telemetry: Any,
    ) -> None:
        if {decode_rank, prefill_rank} != {0, 1}:
            raise RuntimeError("phase-split serving requires distinct ranks 0 and 1")
        self.mx = mx
        self.model = model
        self.tokenizer = tokenizer
        self.group = group
        self.control = control
        self.model_identity = model_identity
        self.prefill_rank = prefill_rank
        self.decode_rank = decode_rank
        self.prefill_step_size = prefill_step_size
        self.cli_args = cli_args
        self.stream = stream
        self.deployment_id = deployment_id
        self.telemetry = telemetry
        self.requests: Queue = Queue()
        self._stopping = threading.Event()
        self._decode_thread: threading.Thread | None = None
        self._broker = threading.Thread(
            target=self._broker_loop,
            name="omlx-phase-broker",
            daemon=True,
        )
        self._broker.start()

    def _tokenize(self, request: Any, args: Any) -> list[int]:
        from mlx_lm.server import convert_chat, process_message_content

        if request.request_type == "text":
            return list(self.tokenizer.encode(request.prompt))
        messages = copy.deepcopy(request.messages)
        if self.tokenizer.has_chat_template:
            process_message_content(messages)
            template_kwargs = dict(
                tools=request.tools,
                tokenize=True,
                **dict(self.cli_args.chat_template_args or {}),
            )
            if args.chat_template_kwargs:
                template_kwargs.update(args.chat_template_kwargs)
            return list(
                self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            )
        return list(self.tokenizer.encode(convert_chat(messages, request.role_mapping)))

    def generate(self, request: Any, args: Any, progress_callback: Any = None):
        from mlx_lm.server import GenerationContext

        if args.num_draft_tokens:
            raise ValueError("phase-split MTP/speculative decode is not enabled yet")
        prompt = self._tokenize(request, args)
        machine, sequences = _state_machine(self.tokenizer, args.stop_words)
        context = GenerationContext(
            has_tool_calling=self.tokenizer.has_tool_calling,
            has_thinking=self.tokenizer.has_thinking,
            tool_parser=self.tokenizer.tool_parser,
            sequences=sequences,
            prompt=prompt,
            prompt_cache_count=0,
        )
        output: Queue = Queue()
        request_id = self.telemetry.begin_request(
            getattr(request, "_omlx_transport_request_id", None)
        )
        self.telemetry.observe_context(
            request_id,
            prompt_tokens=len(prompt),
            cached_tokens=0,
        )
        self.telemetry.mark_pending_uid(request_id)
        self.telemetry.bind_pending_uid((request_id,))
        self.telemetry.register_context(request_id, context)
        self.requests.put(
            _ServingRequest(
                request_id=request_id,
                prompt=prompt,
                args=args,
                context=context,
                state_machine=machine,
                output=output,
                progress=progress_callback or (lambda *_: None),
            )
        )

        def responses() -> Iterator[Any]:
            while True:
                value = output.get()
                if value is None:
                    return
                if isinstance(value, BaseException):
                    raise value
                yield value

        return context, responses()

    def _recv_progress(self, request: _ServingRequest) -> bytes:
        packet = self.control.broadcast_owned_bytes(
            None,
            source_rank=self.prefill_rank,
            expected_size=_PROGRESS.size,
        )
        processed, total = _PROGRESS.unpack(packet)
        try:
            request.progress(int(processed), int(total))
        except (BrokenPipeError, ConnectionError, OSError):
            request.context.stop()
        self.telemetry.observe_prefill_progress(
            request.request_id,
            processed_tokens=int(processed),
            total_tokens=int(total),
        )
        cancelled = bool(request.context._should_stop)
        self.control.broadcast_owned_bytes(
            _CANCEL.pack(1 if cancelled else 0),
            source_rank=self.decode_rank,
            expected_size=_CANCEL.size,
        )
        return packet

    def _decode(self, request: _ServingRequest, cache: list[Any], logits: Any) -> None:
        from mlx_lm.server import (
            Response,
            _format_top_logprobs,
            _make_logits_processors,
            _make_sampler,
        )

        self.mx.set_default_stream(self.stream)
        args = request.args
        if args.seed is not None:
            self.mx.random.seed(args.seed)
        sampler = _make_sampler(args, self.tokenizer)
        processors = _make_logits_processors(args)
        history = self.mx.array(request.prompt, dtype=self.mx.int32)
        state = request.state_machine.make_state()
        detokenizer = self.tokenizer.detokenizer
        reset = getattr(detokenizer, "reset", None)
        if callable(reset):
            reset()
        failed = False
        try:
            for index in range(args.max_tokens):
                if request.context._should_stop:
                    break
                adjusted = logits
                for processor in processors:
                    adjusted = processor(history, adjusted)
                logprobs = adjusted - self.mx.logsumexp(adjusted, keepdims=True)
                sampled = sampler(logprobs)
                self.mx.eval(sampled, logprobs)
                token = int(sampled.item())
                self.telemetry.observe_token(request.request_id)
                state, matched, current_state = request.state_machine.match(
                    state, token
                )
                detokenizer.add_token(token)
                # State delimiters are control tokens, not assistant content.
                # Qwen's tool-call end token decodes to ``</tool_call>``; if it
                # is forwarded as normal text, agents see a valid tool call plus
                # a leaked protocol marker in ``message.content``.
                visible_text = "" if matched is not None else detokenizer.last_segment
                finish_reason = None
                if matched is not None and current_state is None:
                    finish_reason = "stop"
                elif index + 1 >= args.max_tokens:
                    finish_reason = "length"
                request.output.put(
                    Response(
                        visible_text,
                        token,
                        current_state,
                        matched,
                        float(logprobs[0, token].item()),
                        finish_reason,
                        _format_top_logprobs(
                            logprobs.squeeze(0),
                            args.top_logprobs,
                            self.tokenizer,
                        ),
                    )
                )
                if finish_reason is not None or request.context._should_stop:
                    break
                step_started = time.perf_counter()
                history = self.mx.concatenate(
                    [history, self.mx.array([token], dtype=self.mx.int32)]
                )
                logits = self.model(
                    self.mx.array([[token]], dtype=self.mx.int32),
                    cache=cache,
                )[:, -1, :]
                self.mx.eval(logits)
                self.telemetry.observe_batch_step(
                    prompt_responses=0,
                    generation_responses=1,
                    elapsed_seconds=time.perf_counter() - step_started,
                )
        except BaseException as exc:
            failed = True
            request.output.put(exc)
        finally:
            detokenizer.finalize()
            if request.context._should_stop:
                self.telemetry.cancel_request(request.request_id)
            else:
                self.telemetry.finish_request(request.request_id, failed=failed)
            request.output.put(None)

    def _broker_loop(self) -> None:
        self.mx.set_default_stream(self.stream)
        previous: threading.Thread | None = None
        while not self._stopping.is_set():
            try:
                request = self.requests.get(timeout=0.05)
            except Empty:
                continue
            if request is None:
                break
            try:
                _phase_event(
                    self.decode_rank,
                    "broker_request",
                    prompt_tokens=len(request.prompt),
                )
                _broadcast_owned_object(
                    self.control,
                    {
                        "op": "prefill",
                        "prompt": request.prompt,
                        "prefill_step_size": self.prefill_step_size,
                    },
                    source_rank=self.decode_rank,
                )
                start_packet = self.control.broadcast_owned_bytes(
                    None,
                    source_rank=self.prefill_rank,
                    expected_size=_PREFILL_START.size,
                )
                (
                    cached_tokens,
                    uncached_tokens,
                    progress_calls,
                    cache_tier_code,
                    cache_entries,
                    cache_bytes,
                    ssd_entries,
                    ssd_bytes,
                ) = _PREFILL_START.unpack(start_packet)
                if (
                    cached_tokens + uncached_tokens != len(request.prompt)
                    or progress_calls
                    != _prefill_calls(uncached_tokens, self.prefill_step_size)
                    or cache_tier_code not in _CACHE_CODE_TO_TIER
                ):
                    raise RuntimeError("prefill rank returned invalid cache metadata")
                self.telemetry.observe_context(
                    request.request_id,
                    prompt_tokens=len(request.prompt),
                    cached_tokens=cached_tokens,
                )
                request.context.prompt_cache_count = cached_tokens
                self.telemetry.observe_cache_lookup(
                    prompt_tokens=len(request.prompt),
                    remaining_tokens=uncached_tokens,
                    entries=cache_entries + ssd_entries,
                    nbytes=cache_bytes + ssd_bytes,
                    memory_entries=cache_entries,
                    memory_bytes=cache_bytes,
                    ssd_entries=ssd_entries,
                    ssd_bytes=ssd_bytes,
                    hit_tier=_CACHE_CODE_TO_TIER[cache_tier_code],
                )
                for _ in range(progress_calls):
                    self._recv_progress(request)
                header = self.control.broadcast_owned_bytes(
                    None,
                    source_rank=self.prefill_rank,
                    expected_size=_LOGITS_HEADER.size,
                )
                dtype_code, rows, columns = _LOGITS_HEADER.unpack(header)
                if (dtype_code, rows, columns) == (0, 0, 0):
                    self.telemetry.cancel_request(request.request_id)
                    request.output.put(None)
                    continue
                _phase_event(
                    self.decode_rank,
                    "broker_prefill_ready",
                    logits_columns=columns,
                )
                dtype_name = _CODE_TO_DTYPE.get(dtype_code)
                if dtype_name is None or rows != 1 or columns <= 0:
                    raise RuntimeError("prefill rank returned an invalid logits header")

                # Leave rank 1 waiting on the TCP barrier until the previous
                # decode releases rank 0's Metal stream. No RDMA receive is
                # posted during an arbitrarily long decode.
                if previous is not None:
                    previous.join()
                self.control.barrier()
                _phase_event(self.decode_rank, "broker_cache_recv_start")
                cache, _manifest, stats = recv_cache_transfer(
                    self.mx,
                    src=self.prefill_rank,
                    group=self.group,
                    expected_model_identity=self.model_identity,
                )
                logits = self.mx.distributed.recv(
                    (rows, columns),
                    getattr(self.mx, dtype_name),
                    self.prefill_rank,
                    group=self.group,
                )
                self.mx.eval(logits)
                self.mx.synchronize()
                self.telemetry.observe_phase_handoff(
                    tensor_bytes=stats.tensor_bytes,
                    array_count=stats.array_count,
                    elapsed_seconds=stats.elapsed_seconds,
                    queue_depth=self.requests.qsize(),
                )
                if request.context._should_stop:
                    self.telemetry.cancel_request(request.request_id)
                    request.output.put(None)
                    continue
                _phase_event(self.decode_rank, "broker_cache_recv_done")
                previous = threading.Thread(
                    target=self._decode,
                    args=(request, cache, logits),
                    name="omlx-phase-decode",
                    daemon=True,
                )
                previous.start()
            except BaseException as exc:
                if request.context._should_stop:
                    self.telemetry.cancel_request(request.request_id)
                else:
                    self.telemetry.finish_request(request.request_id, failed=True)
                request.output.put(exc)
                request.output.put(None)
        if previous is not None:
            previous.join()
        with suppress(Exception):
            _broadcast_owned_object(
                self.control,
                {"op": "stop"},
                source_rank=self.decode_rank,
            )

    def stop_and_join(self) -> None:
        self._stopping.set()
        self.requests.put(None)
        self._broker.join(timeout=10.0)

    def join(self) -> None:
        self._broker.join()


def _prefill_rank_loop(
    *,
    mx: Any,
    model: Any,
    group: Any,
    control: Any,
    model_identity: str,
    rank: int,
    decode_rank: int,
    prompt_cache: _PhasePromptCache,
) -> None:
    while True:
        request = _broadcast_owned_object(control, None, source_rank=decode_rank)
        if not isinstance(request, dict):
            raise RuntimeError("phase broker sent an invalid request")
        if request.get("op") == "stop":
            return
        if request.get("op") != "prefill":
            raise RuntimeError("phase broker sent an unknown operation")
        prompt = [int(value) for value in request.get("prompt") or ()]
        step = int(request.get("prefill_step_size") or 2048)
        if len(prompt) < 2 or step < 1:
            raise RuntimeError("phase broker sent an invalid prompt")
        prompt_cache.active.set()
        cache, rest, logits, cache_tier = prompt_cache.lookup(prompt)
        cached_tokens = len(prompt) - len(rest)
        cache_stats = prompt_cache.stats()
        progress_calls = _prefill_calls(len(rest), step)
        control.broadcast_owned_bytes(
            _PREFILL_START.pack(
                cached_tokens,
                len(rest),
                progress_calls,
                _CACHE_TIER_TO_CODE[cache_tier],
                cache_stats["memory_entries"],
                cache_stats["memory_bytes"],
                cache_stats["ssd_entries"],
                cache_stats["ssd_bytes"],
            ),
            source_rank=rank,
            expected_size=_PREFILL_START.size,
        )
        _phase_event(
            rank,
            "prefill_request",
            prompt_tokens=len(prompt),
            cached_tokens=cached_tokens,
            cache_tier=cache_tier or "miss",
        )

        def progress(
            processed: int,
            total: int,
            prompt_tokens: list[int] = prompt,
            prompt_state: list[Any] = cache,
            prefix_tokens: int = cached_tokens,
        ) -> bool:
            prompt_cache.checkpoint(
                prompt_tokens,
                prompt_state,
                prefix_tokens + processed,
            )
            control.broadcast_owned_bytes(
                _PROGRESS.pack(processed, total),
                source_rank=rank,
                expected_size=_PROGRESS.size,
            )
            cancel_packet = control.broadcast_owned_bytes(
                None,
                source_rank=decode_rank,
                expected_size=_CANCEL.size,
            )
            return _CANCEL.unpack(cancel_packet)[0] == 0

        if logits is None:
            logits, _seconds = _prefill_logits(
                mx,
                model,
                cache,
                rest,
                step=step,
                progress=progress,
            )
        if logits is None:
            control.broadcast_owned_bytes(
                _LOGITS_HEADER.pack(0, 0, 0),
                source_rank=rank,
                expected_size=_LOGITS_HEADER.size,
            )
            _phase_event(rank, "prefill_cancelled")
            prompt_cache.active.clear()
            continue
        _phase_event(rank, "prefill_compute_done")
        dtype_code = _DTYPE_TO_CODE.get(_dtype_name(logits.dtype))
        if dtype_code is None or logits.ndim != 2:
            raise RuntimeError("phase prefill produced unsupported logits")
        prompt_cache.insert(prompt, cache, logits)
        control.broadcast_owned_bytes(
            _LOGITS_HEADER.pack(dtype_code, int(logits.shape[0]), int(logits.shape[1])),
            source_rank=rank,
            expected_size=_LOGITS_HEADER.size,
        )
        control.barrier()
        _phase_event(rank, "prefill_cache_send_start")
        prepared = prepare_cache_transfer(
            cache,
            model_identity=model_identity,
            prompt_tokens=len(prompt),
        )
        send_cache_transfer(mx, prepared, dst=decode_rank, group=group)
        mx.eval(
            mx.distributed.send(
                mx.contiguous(logits),
                decode_rank,
                group=group,
            )
        )
        mx.synchronize()
        prompt_cache.active.clear()
        _phase_event(rank, "prefill_cache_send_done")


def run_worker(args: argparse.Namespace) -> int:
    from omlx._torch_stub import install as install_torch_stub

    install_torch_stub()
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.server import APIHandler, _run_http_server

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    from .control_plane import RankControlPlane
    from .deployment import (
        decode_worker_contract,
        decode_worker_execution,
        decode_worker_path_map,
        decode_worker_serving_mode,
    )
    from .inference_worker import (
        RuntimeMarker,
        _apply_rank_wired_limit,
        _emit_event,
        _install_signal_handlers,
        _prompt_cache_ssd_dir,
        _release_metal_memory,
        _runtime_assignment,
        _wait_for_serve_release,
    )
    from .jaccl_lease import acquire_jaccl_communicator_lease
    from .jaccl_side_channel import init_cluster_group
    from .memory_guard import (
        admission_budget,
        assignment_memory_safety,
        guard_rank_load,
    )
    from .staging import model_identity_digest
    from .telemetry import RuntimeTelemetry

    _install_signal_handlers()
    plan_hash, assignments, _profiles, tensor_parallel_size = decode_worker_contract(
        args.plan
    )
    mode, prefill_rank, decode_rank = decode_worker_serving_mode(args.plan)
    execution = decode_worker_execution(args.plan)
    if mode != "disaggregated" or tensor_parallel_size != 1:
        raise RuntimeError("phase server received a non-disaggregated plan")
    if args.plan_hash != plan_hash:
        raise RuntimeError("worker plan hash does not match launch contract")

    init_backend = "jaccl" if args.backend.startswith("jaccl") else "ring"
    lease = (
        acquire_jaccl_communicator_lease(
            deployment_id=args.deployment_id,
            state_dir=args.state_dir,
        )
        if init_backend == "jaccl"
        else None
    )
    marker = None
    phase_prompt_cache = None
    cache_maintenance = None
    preserve_failure_marker = False
    try:
        group = init_cluster_group(mx, backend=init_backend, strict=True)
        rank = int(group.rank())
        if group.size() != 2:
            raise RuntimeError("phase server currently requires two ranks")
        assignment = sorted(assignments, key=lambda item: item.rank)[rank]
        marker = RuntimeMarker(
            state_dir=args.state_dir,
            deployment_id=args.deployment_id,
            rank=rank,
            world_size=2,
            model=args.model,
            backend=args.backend,
            plan_hash=plan_hash,
        )
        marker.update(
            "loading",
            load_stage="initializing_full_replica",
            serving_mode="disaggregated",
            prefill_rank=prefill_rank,
            decode_rank=decode_rank,
        )
        marker.start_heartbeat()
        admission_ceiling = guard_rank_load(
            assignment,
            rank=rank,
            role=assignment.role,
            memory_guard_tier=assignment.memory_guard_tier,
        )
        load_budget = admission_budget(
            admission_ceiling,
            role=assignment.role,
            safety=assignment_memory_safety(assignment),
        )
        _apply_rank_wired_limit(load_budget)
        path_map = decode_worker_path_map(args.plan)
        model_path = Path(path_map.get(assignment.node_id, args.model)).expanduser()
        identity = model_identity_digest(model_path)
        maybe_apply_pre_load_patches(
            model_path,
            model_settings=SimpleNamespace(
                mtp_enabled=False,
                mtp_num_draft_tokens=0,
            ),
        )
        stream = mx.new_thread_unsafe_stream(mx.default_device())
        mx.set_default_stream(stream)
        before = int(mx.get_active_memory())
        model, tokenizer = load(
            str(model_path),
            lazy=False,
            trust_remote_code=args.trust_remote_code,
        )
        model.eval()
        measured = max(0, int(mx.get_active_memory()) - before)
        if rank == prefill_rank:
            phase_prompt_cache = _PhasePromptCache(
                model=model,
                model_key=identity,
                max_size=args.prompt_cache_size,
                max_bytes=args.prompt_cache_bytes,
                ssd_directory=_prompt_cache_ssd_dir(args, rank),
                ssd_max_bytes=args.prompt_cache_ssd_max_bytes,
                step=args.prefill_step_size,
            )
            cache_maintenance = _PhaseCacheMaintenance(
                prompt_cache=phase_prompt_cache,
                state_dir=args.state_dir,
                deployment_id=args.deployment_id,
                plan_hash=plan_hash,
                rank=rank,
            )
        marker.update(
            "loading",
            load_stage="weights_resident",
            measured_weight_bytes=measured,
            assignments=[_runtime_assignment(item) for item in assignments],
        )
        _emit_event(
            {
                "type": "rank_ready",
                "protocol_version": 1,
                "deployment_id": args.deployment_id,
                "plan_hash": plan_hash,
                "world_size": 2,
                "measured_weight_bytes": measured,
                **_runtime_assignment(assignment),
            }
        )
        _wait_for_serve_release(
            args.state_dir,
            args.deployment_id,
            plan_hash,
            2,
        )
        with RankControlPlane(
            rank=rank,
            world_size=2,
            host=args.control_host,
            port=args.control_port,
            token=args.control_token,
            connect_timeout=120.0,
            io_timeout=3600.0,
        ) as control:
            control.barrier()
            marker.update(
                "ready",
                load_stage="ready",
                measured_weight_bytes=measured,
                serving_mode="disaggregated",
                prefill_rank=prefill_rank,
                decode_rank=decode_rank,
            )
            if rank == decode_rank:
                cli_args = SimpleNamespace(
                    allowed_origins=["*"],
                    num_draft_tokens=0,
                    max_tokens=4096,
                    temp=0.0,
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    model=str(model_path),
                    chat_template_args={},
                )
                telemetry = RuntimeTelemetry(
                    marker,
                    execution=execution,
                    assignment=assignment,
                    cancel_path=(
                        marker.path.parent / f"{args.deployment_id}-cancel.json"
                    ),
                    cancel_deployment_id=args.deployment_id,
                    cancel_plan_hash=plan_hash,
                    cancel_epoch_floor=int(time.time() * 1000),
                    prompt_cache_ssd_enabled=execution.prompt_cache_ssd,
                    prompt_cache_ssd_max_bytes=(
                        execution.prompt_cache_ssd_max_bytes
                        if execution.prompt_cache_ssd
                        else 0
                    ),
                )
                generator = _PhaseResponseGenerator(
                    mx=mx,
                    model=model,
                    tokenizer=tokenizer,
                    group=group,
                    control=control,
                    model_identity=identity,
                    prefill_rank=int(prefill_rank),
                    decode_rank=int(decode_rank),
                    prefill_step_size=args.prefill_step_size,
                    cli_args=cli_args,
                    stream=stream,
                    deployment_id=args.deployment_id,
                    telemetry=telemetry,
                )
                _emit_event(
                    {
                        "type": "ready",
                        "protocol_version": 1,
                        "deployment_id": args.deployment_id,
                        "plan_hash": plan_hash,
                        "rank": rank,
                        "world_size": 2,
                        "port": args.port,
                        "serving_mode": "disaggregated",
                        "prefill_rank": prefill_rank,
                        "decode_rank": decode_rank,
                    }
                )

                class PhaseAPIHandler(APIHandler):
                    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                        prefix = "/omlx/internal/cache/"
                        if not self.path.startswith(prefix):
                            return super().do_POST()
                        mode = self.path.removeprefix(prefix).removesuffix("/clear")
                        if self.headers.get(
                            "X-oMLX-Plan-Hash"
                        ) != plan_hash or mode not in {"hot", "ssd", "all"}:
                            status = 403
                            payload = {
                                "status": "error",
                                "error": "invalid cache clear",
                            }
                        elif telemetry.active_request_count():
                            status = 409
                            payload = {
                                "status": "error",
                                "error": "requests are active",
                            }
                        else:
                            status = 200
                            payload = {
                                "status": "ok",
                                "rank": rank,
                                "hot_cleared": 0,
                                "ssd_deleted": 0,
                            }
                        body = json.dumps(payload, separators=(",", ":")).encode()
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)

                    def handle_completion(self, request: Any, stop_words: Any) -> Any:
                        request_id = self.headers.get("X-oMLX-Request-ID")
                        if isinstance(request_id, str) and request_id.strip():
                            request._omlx_transport_request_id = request_id.strip()[
                                :128
                            ]
                        return super().handle_completion(request, stop_words)

                marker.stop_heartbeat()
                telemetry.start_heartbeat()
                try:
                    _run_http_server(
                        args.server_host,
                        args.port,
                        generator,
                        handler_class=PhaseAPIHandler,
                    )
                finally:
                    telemetry.stop_heartbeat()
                    generator.stop_and_join()
            else:
                if phase_prompt_cache is None:
                    raise RuntimeError("prefill rank cache was not initialized")
                if cache_maintenance is not None:
                    cache_maintenance.start()
                _prefill_rank_loop(
                    mx=mx,
                    model=model,
                    group=group,
                    control=control,
                    model_identity=identity,
                    rank=rank,
                    decode_rank=int(decode_rank),
                    prompt_cache=phase_prompt_cache,
                )
        return 0
    except KeyboardInterrupt:
        return 0
    except BaseException as exc:
        preserve_failure_marker = True
        if marker is not None:
            with suppress(Exception):
                marker.update(
                    "failed",
                    error=f"{type(exc).__name__}: {exc}"[:1000],
                )
        raise
    finally:
        if marker is not None:
            marker.stop_heartbeat()
            if not preserve_failure_marker:
                marker.remove()
        if lease is not None:
            lease.close()
        if cache_maintenance is not None:
            cache_maintenance.stop()
        if phase_prompt_cache is not None:
            with suppress(Exception):
                phase_prompt_cache.close()
        with suppress(Exception):
            _release_metal_memory("phase server shutdown")


def main() -> int:
    args = _arguments()
    try:
        return run_worker(args)
    except BaseException as exc:
        _phase_event(
            int(os.environ.get("MLX_RANK", "-1")),
            "rank_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
