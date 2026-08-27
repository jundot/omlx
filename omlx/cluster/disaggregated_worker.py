# SPDX-License-Identifier: Apache-2.0
"""Experimental full-replica prefill/decode disaggregation worker.

One selected rank owns prompt processing and the other owns handed-off decode.
Both load the same full model; only cache tensors and the first sampled token
cross the data plane. This is intentionally a bounded worker/benchmark, not a
serving route. It proves the universal cache wire contract and two-request
pipeline before scheduler and HTTP lifecycle integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .cache_transfer import (
    prepare_cache_transfer,
    recv_cache_transfer,
    send_cache_transfer,
)

EVENT_PREFIX = "OMLX_DISAGG_EVENT:"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("jaccl", "ring"), default="jaccl")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--completion-tokens", type=int, default=32)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--prefill-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--pipeline-requests", type=int, choices=(1, 2, 4, 8), default=1
    )
    parser.add_argument("--control-host", required=True)
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--control-token", required=True)
    parser.add_argument(
        "--state-dir", default="~/.omlx/cluster/runtime-disaggregated"
    )
    parser.add_argument("--deployment-id", default="disaggregated-prefill-decode")
    return parser.parse_args()


def _event(payload: dict[str, Any]) -> None:
    print(EVENT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def _prompt_tokens(
    tokenizer: Any, count: int, *, request_index: int = 0
) -> list[int]:
    if count < 2:
        raise ValueError("disaggregated prompt must contain at least two tokens")
    seed = tokenizer.encode(
        " Apple Silicon prefill decode disaggregation over RDMA. "
        f"Request {request_index}.",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("tokenizer produced an empty benchmark seed")
    prefix = []
    bos = getattr(tokenizer, "bos_token_id", None)
    if isinstance(bos, int) and bos >= 0:
        prefix.append(bos)
    needed = count - len(prefix)
    return prefix + (seed * math.ceil(needed / len(seed)))[:needed]


def _cache_states(cache: list[Any]) -> list[Any]:
    return [entry.state for entry in cache]


def _prefill(
    mx: Any,
    model: Any,
    cache: list[Any],
    tokens: list[int],
    *,
    step: int,
) -> tuple[int, float, int]:
    started = time.perf_counter()
    values = mx.array(tokens, dtype=mx.int32)
    processed = 0
    calls = 0
    while len(tokens) - processed > 1:
        width = min(step, (len(tokens) - processed) - 1)
        _ = model(values[None, processed : processed + width], cache=cache)
        mx.eval(_cache_states(cache))
        processed += width
        calls += 1
        mx.clear_cache()
    logits = model(values[None, processed:], cache=cache)[:, -1, :]
    first = mx.argmax(logits, axis=-1)
    mx.eval(first, _cache_states(cache))
    calls += 1
    return int(first.item()), time.perf_counter() - started, calls


def _fixed_greedy_decode(
    mx: Any,
    model: Any,
    cache: list[Any],
    first_token: int,
    count: int,
) -> tuple[list[int], float]:
    if count < 1:
        raise ValueError("completion token count must be positive")
    result = [int(first_token)]
    current = int(first_token)
    started = time.perf_counter()
    for _ in range(count - 1):
        value = mx.array([[current]], dtype=mx.int32)
        logits = model(value, cache=cache)[:, -1, :]
        next_token = mx.argmax(logits, axis=-1)
        mx.eval(next_token)
        current = int(next_token.item())
        result.append(current)
    mx.synchronize()
    return result, time.perf_counter() - started


def _token_hash(tokens: list[int]) -> str:
    payload = ",".join(map(str, tokens)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _send_first_token(mx: Any, token: int, *, dst: int, group: Any) -> None:
    value = mx.array([token], dtype=mx.int32)
    mx.eval(mx.distributed.send(value, dst, group=group))
    mx.synchronize()


def _recv_first_token(mx: Any, *, src: int, group: Any) -> int:
    value = mx.distributed.recv((1,), mx.int32, src, group=group)
    mx.eval(value)
    mx.synchronize()
    return int(value.item())


def _send_result_array(mx: Any, value: Any, *, dst: int, group: Any) -> None:
    mx.eval(mx.distributed.send(value, dst, group=group))
    mx.synchronize()


def _recv_result_array(
    mx: Any, shape: tuple[int, ...], dtype: Any, *, src: int, group: Any
) -> Any:
    value = mx.distributed.recv(shape, dtype, src, group=group)
    mx.eval(value)
    mx.synchronize()
    return value


def _run_request_pipeline(
    *,
    mx: Any,
    model: Any,
    tokenizer: Any,
    make_prompt_cache: Any,
    control: Any,
    group: Any,
    rank: int,
    prefill_rank: int,
    decode_rank: int,
    model_identity: str,
    model_path: Path,
    prompt_tokens: int,
    completion_tokens: int,
    prefill_step_size: int,
    backend: str,
    requests: int,
) -> int:
    prompts = [
        _prompt_tokens(tokenizer, prompt_tokens, request_index=index)
        for index in range(requests)
    ]
    if rank == prefill_rank:
        pipeline_started = time.perf_counter()
        caches: list[list[Any]] = []
        first_tokens: list[int] = []
        prefill_seconds: list[float] = []
        prefill_calls: list[int] = []
        overlap_seconds: list[float] = []
        transfers = []

        for index, prompt in enumerate(prompts):
            if index > 0:
                control.barrier()
                overlap_started = time.perf_counter()

            cache = make_prompt_cache(model)
            first, seconds, calls = _prefill(
                mx,
                model,
                cache,
                prompt,
                step=prefill_step_size,
            )
            prepared = prepare_cache_transfer(
                cache,
                model_identity=model_identity,
                prompt_tokens=prompt_tokens,
            )
            caches.append(cache)
            first_tokens.append(first)
            prefill_seconds.append(seconds)
            prefill_calls.append(calls)

            if index == 0:
                control.broadcast_owned_bytes(
                    b"\x01",
                    source_rank=prefill_rank,
                    expected_size=1,
                )
                control.barrier()
            else:
                control.barrier()
                overlap_seconds.append(time.perf_counter() - overlap_started)

            transfers.append(
                send_cache_transfer(
                    mx,
                    prepared,
                    dst=decode_rank,
                    group=group,
                )
            )
            _send_first_token(mx, first, dst=decode_rank, group=group)

        # Drain the last decode after all N-1 prefill/decode overlap windows.
        control.barrier()
        control.barrier()
        pipeline_seconds = time.perf_counter() - pipeline_started

        remote_arrays = [
            _recv_result_array(
                mx,
                (completion_tokens,),
                mx.int32,
                src=decode_rank,
                group=group,
            )
            for _ in range(requests)
        ]
        remote_metrics_array = _recv_result_array(
            mx,
            (2 * requests,),
            mx.float32,
            src=decode_rank,
            group=group,
        )
        remote_tokens = [
            [int(value) for value in array.tolist()] for array in remote_arrays
        ]
        raw_remote_metrics = [
            float(value) for value in remote_metrics_array.tolist()
        ]
        decode_seconds = raw_remote_metrics[:requests]
        recv_seconds = raw_remote_metrics[requests:]

        # Parity work happens after the measured pipeline window.
        baseline_tokens = []
        baseline_decode_seconds = []
        for cache, first in zip(caches, first_tokens):
            result, seconds = _fixed_greedy_decode(
                mx,
                model,
                cache,
                first,
                completion_tokens,
            )
            baseline_tokens.append(result)
            baseline_decode_seconds.append(seconds)
        parity_by_request = [
            baseline == remote
            for baseline, remote in zip(baseline_tokens, remote_tokens)
        ]
        parity = all(parity_by_request)
        serial_source_seconds = sum(prefill_seconds) + sum(
            baseline_decode_seconds
        )
        report = {
            "type": "pipeline_result",
            "backend": backend,
            "prefill_rank": prefill_rank,
            "decode_rank": decode_rank,
            "model": str(model_path),
            "model_identity": model_identity,
            "requests": requests,
            "prompt_tokens_per_request": prompt_tokens,
            "completion_tokens_per_request": completion_tokens,
            "prefill_calls": prefill_calls,
            "prefill_seconds": prefill_seconds,
            "prefill_tokens_per_second": [
                prompt_tokens / seconds for seconds in prefill_seconds
            ],
            "cache_tensor_bytes": [item.tensor_bytes for item in transfers],
            "cache_send_seconds": [item.elapsed_seconds for item in transfers],
            "cache_send_bytes_per_second": [
                item.bytes_per_second for item in transfers
            ],
            "cache_recv_seconds": recv_seconds,
            "remote_decode_seconds": decode_seconds,
            "remote_decode_tokens_per_second": [
                max(0, completion_tokens - 1) / seconds
                for seconds in decode_seconds
            ],
            "baseline_decode_seconds": baseline_decode_seconds,
            "overlap_window_seconds": overlap_seconds,
            "pipeline_seconds": pipeline_seconds,
            "pipeline_seconds_per_request": pipeline_seconds / requests,
            "serial_source_seconds": serial_source_seconds,
            "measured_pipeline_speedup": (
                serial_source_seconds / pipeline_seconds
                if pipeline_seconds > 0
                else 0.0
            ),
            "parity": parity,
            "parity_by_request": parity_by_request,
            "baseline_token_sha256": [
                _token_hash(value) for value in baseline_tokens
            ],
            "remote_token_sha256": [
                _token_hash(value) for value in remote_tokens
            ],
        }
        _event(report)
        return 0 if parity else 2

    control.broadcast_owned_bytes(
        None,
        source_rank=prefill_rank,
        expected_size=1,
    )
    control.barrier()
    remote_tokens: list[list[int]] = []
    decode_seconds: list[float] = []
    recv_stats = []
    prompt_lengths: list[int] = []

    for _index in range(requests):
        cache, manifest, received = recv_cache_transfer(
            mx,
            src=prefill_rank,
            group=group,
            expected_model_identity=model_identity,
        )
        first = _recv_first_token(mx, src=prefill_rank, group=group)
        control.barrier()
        result, seconds = _fixed_greedy_decode(
            mx,
            model,
            cache,
            first,
            completion_tokens,
        )
        control.barrier()
        remote_tokens.append(result)
        decode_seconds.append(seconds)
        recv_stats.append(received)
        prompt_lengths.append(int(manifest["prompt_tokens"]))
        del cache
        mx.clear_cache()

    for result in remote_tokens:
        _send_result_array(
            mx,
            mx.array(result, dtype=mx.int32),
            dst=prefill_rank,
            group=group,
        )
    _send_result_array(
        mx,
        mx.array(
            [
                *decode_seconds,
                *(item.elapsed_seconds for item in recv_stats),
            ],
            dtype=mx.float32,
        ),
        dst=prefill_rank,
        group=group,
    )
    _event(
        {
            "type": "pipeline_decode_complete",
            "rank": rank,
            "prompt_tokens": prompt_lengths,
            "cache_tensor_bytes": [
                item.tensor_bytes for item in recv_stats
            ],
            "decode_seconds": decode_seconds,
            "token_sha256": [
                _token_hash(result) for result in remote_tokens
            ],
        }
    )
    return 0

def run(args: argparse.Namespace) -> int:
    if args.prompt_tokens < 2 or args.completion_tokens < 1:
        raise ValueError("prompt/completion sizes are too small")
    if args.prefill_step_size < 1:
        raise ValueError("prefill step size must be positive")

    from omlx._torch_stub import install as install_torch_stub

    install_torch_stub()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    from .control_plane import RankControlPlane
    from .jaccl_lease import acquire_jaccl_communicator_lease
    from .jaccl_side_channel import init_cluster_group
    from .staging import model_identity_digest

    model_path = Path(args.model).expanduser().resolve()
    identity = model_identity_digest(model_path)
    maybe_apply_pre_load_patches(
        model_path,
        model_settings=SimpleNamespace(mtp_enabled=False, mtp_num_draft_tokens=0),
    )
    load_started = time.perf_counter()
    model, tokenizer = load(str(model_path), lazy=False, trust_remote_code=False)
    model.eval()
    load_seconds = time.perf_counter() - load_started

    lease = (
        acquire_jaccl_communicator_lease(
            deployment_id=args.deployment_id,
            state_dir=args.state_dir,
        )
        if args.backend == "jaccl"
        else None
    )
    try:
        group = init_cluster_group(mx, backend=args.backend, strict=True)
        rank = int(group.rank())
        if group.size() != 2:
            raise RuntimeError("disaggregated prototype currently requires two ranks")
        prefill_rank = int(args.prefill_rank)
        decode_rank = 1 - prefill_rank
        tokens = _prompt_tokens(tokenizer, args.prompt_tokens)
        with RankControlPlane(
            rank=rank,
            world_size=2,
            host=args.control_host,
            port=args.control_port,
            token=args.control_token,
            connect_timeout=120.0,
            io_timeout=max(600.0, float(args.prompt_tokens)),
        ) as control:
            control.barrier()
            _event(
                {
                    "type": "rank_loaded",
                    "rank": rank,
                    "role": "prefill" if rank == prefill_rank else "decode",
                    "model_identity": identity,
                    "load_seconds": load_seconds,
                    "peak_memory_bytes": int(mx.get_peak_memory()),
                }
            )

            if args.pipeline_requests > 1:
                return _run_request_pipeline(
                    mx=mx,
                    model=model,
                    tokenizer=tokenizer,
                    make_prompt_cache=make_prompt_cache,
                    control=control,
                    group=group,
                    rank=rank,
                    prefill_rank=prefill_rank,
                    decode_rank=decode_rank,
                    model_identity=identity,
                    model_path=model_path,
                    prompt_tokens=args.prompt_tokens,
                    completion_tokens=args.completion_tokens,
                    prefill_step_size=args.prefill_step_size,
                    backend=args.backend,
                    requests=args.pipeline_requests,
                )

            if rank == prefill_rank:
                cache = make_prompt_cache(model)
                first_token, prefill_seconds, calls = _prefill(
                    mx,
                    model,
                    cache,
                    tokens,
                    step=args.prefill_step_size,
                )
                prepared = prepare_cache_transfer(
                    cache,
                    model_identity=identity,
                    prompt_tokens=len(tokens),
                )
                control.broadcast_owned_bytes(
                    b"\x01", source_rank=prefill_rank, expected_size=1
                )
                control.barrier()
                transfer = send_cache_transfer(
                    mx, prepared, dst=decode_rank, group=group
                )
                first_array = mx.array([first_token], dtype=mx.int32)
                mx.eval(
                    mx.distributed.send(first_array, decode_rank, group=group)
                )
                # Finish the prefill->decode direction before either rank posts a
                # result in the reverse direction. Lazy sibling point-to-point
                # operations may otherwise be topologically reordered.
                mx.synchronize()

                baseline_tokens, baseline_decode_seconds = _fixed_greedy_decode(
                    mx,
                    model,
                    cache,
                    first_token,
                    args.completion_tokens,
                )
                remote_tokens_array = mx.distributed.recv(
                    (args.completion_tokens,),
                    mx.int32,
                    decode_rank,
                    group=group,
                )
                mx.eval(remote_tokens_array)
                mx.synchronize()
                remote_metrics = mx.distributed.recv(
                    (2,), mx.float32, decode_rank, group=group
                )
                mx.eval(remote_metrics)
                mx.synchronize()
                remote_tokens = [
                    int(value) for value in remote_tokens_array.tolist()
                ]
                remote_decode_seconds, remote_recv_seconds = (
                    float(value) for value in remote_metrics.tolist()
                )
                parity = baseline_tokens == remote_tokens
                report = {
                    "type": "result",
                    "backend": args.backend,
                    "prefill_rank": prefill_rank,
                    "decode_rank": decode_rank,
                    "model": str(model_path),
                    "model_identity": identity,
                    "prompt_tokens": len(tokens),
                    "completion_tokens": args.completion_tokens,
                    "prefill_calls": calls,
                    "prefill_seconds": prefill_seconds,
                    "prefill_tokens_per_second": len(tokens) / prefill_seconds,
                    "cache_arrays": transfer.array_count,
                    "cache_tensor_bytes": transfer.tensor_bytes,
                    "cache_manifest_bytes": transfer.manifest_bytes,
                    "cache_send_seconds": transfer.elapsed_seconds,
                    "cache_send_bytes_per_second": transfer.bytes_per_second,
                    "cache_recv_seconds": remote_recv_seconds,
                    "baseline_decode_seconds": baseline_decode_seconds,
                    "baseline_decode_tokens_per_second": (
                        max(0, args.completion_tokens - 1)
                        / baseline_decode_seconds
                        if baseline_decode_seconds > 0
                        else 0.0
                    ),
                    "remote_decode_seconds": remote_decode_seconds,
                    "remote_decode_tokens_per_second": (
                        max(0, args.completion_tokens - 1)
                        / remote_decode_seconds
                        if remote_decode_seconds > 0
                        else 0.0
                    ),
                    "parity": parity,
                    "baseline_token_sha256": _token_hash(baseline_tokens),
                    "remote_token_sha256": _token_hash(remote_tokens),
                    "first_token": first_token,
                    "baseline_text": tokenizer.decode(baseline_tokens),
                    "remote_text": tokenizer.decode(remote_tokens),
                }
                _event(report)
                return 0 if parity else 2

            control.broadcast_owned_bytes(
                None, source_rank=prefill_rank, expected_size=1
            )
            control.barrier()
            cache, manifest, recv_stats = recv_cache_transfer(
                mx,
                src=prefill_rank,
                group=group,
                expected_model_identity=identity,
            )
            first_array = mx.distributed.recv(
                (1,), mx.int32, prefill_rank, group=group
            )
            mx.eval(first_array)
            mx.synchronize()
            first_token = int(first_array.item())
            remote_tokens, decode_seconds = _fixed_greedy_decode(
                mx,
                model,
                cache,
                first_token,
                args.completion_tokens,
            )
            token_array = mx.array(remote_tokens, dtype=mx.int32)
            metrics = mx.array(
                [decode_seconds, recv_stats.elapsed_seconds], dtype=mx.float32
            )
            mx.eval(
                mx.distributed.send(token_array, prefill_rank, group=group)
            )
            mx.synchronize()
            mx.eval(mx.distributed.send(metrics, prefill_rank, group=group))
            mx.synchronize()
            _event(
                {
                    "type": "decode_complete",
                    "rank": rank,
                    "prompt_tokens": manifest["prompt_tokens"],
                    "cache_arrays": recv_stats.array_count,
                    "cache_tensor_bytes": recv_stats.tensor_bytes,
                    "cache_recv_seconds": recv_stats.elapsed_seconds,
                    "decode_seconds": decode_seconds,
                    "token_sha256": _token_hash(remote_tokens),
                }
            )
            return 0
    finally:
        if lease is not None:
            lease.close()


def main() -> int:
    try:
        return run(_arguments())
    except BaseException as exc:
        _event(
            {
                "type": "error",
                "rank": int(os.environ.get("MLX_RANK", "-1")),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
