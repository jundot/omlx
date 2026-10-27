#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manual, matched AR/MTP tuning of Decode Fairness Ratio on Apple Silicon.

Starts only owned loopback servers with isolated settings. No model downloads,
no inference hooks, no retries, and no writes to the user's serving state.
See docs/TESTING.md for the protocol and the historical report.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity(root):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "harness_sha256": sha(Path(__file__).resolve()),
        "diff_sha256": hashlib.sha256(
            git("diff", "HEAD", "--", "omlx").encode()
        ).hexdigest(),
    }


def serve(state, output, source, argv):
    """Isolate startup's cluster helper; do not wrap model/scheduler execution."""
    sys.path.insert(0, source)
    sys.path.insert(0, str(Path(source) / "apps" / "omlx-web"))
    import omlx
    from omlx.cli import main
    from omlx.cluster import worker_shim

    original = worker_shim.ensure_cluster_python_shim
    worker_shim.ensure_cluster_python_shim = lambda: original(home=Path(state))
    save(
        Path(output) / "runtime.json",
        {
            "source": str(Path(omlx.__file__).resolve()),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["omlx", "mlx", "mlx-lm", "mlx-vlm", "transformers"]
            },
            "inference_hooks": False,
        },
    )
    sys.argv = [sys.argv[0], *argv]
    main()


def prompt(tok, family, target):
    ledger = "".join(f"record {i:04d}: alpha beta gamma delta.\n" for i in range(8000))
    if family == "copy":
        text = (
            "Copy the following ledger verbatim, beginning with record 0000. "
            "Continue until every record is copied. Preserve every number and word. "
            "Output only the ledger, without explanations or code fences.\n__DOC__"
            "\nEND OF INPUT LEDGER.\nNow copy the complete ledger from record 0000, "
            "not its final record. Continue with the next numbered record on every line."
        )
        doc = ledger
    else:
        questions = {
            "incident": (
                "Diagnose this production incident using the evidence. Explain the likely "
                "causal chain, competing explanations, the next three measurements, and a safe "
                "rollback plan. Distinguish observation from inference. Use specific evidence "
                "from the logs and explain risks in enough detail for an on-call engineer."
            ),
            "review": (
                "Review the asynchronous Python service code below. Identify concrete "
                "correctness and resource-management bugs, explain triggering interleavings, "
                "and propose a minimal patch plus meaningful regression tests. Prioritize "
                "issues with a reproducible failure over style comments."
            ),
            "retrieval": (
                "Write an engineering decision memo using the supplied internal notes. "
                "Compare the proposed cache, batching and timeout changes, identify conflicting "
                "evidence, recommend an order of experiments, and cite note IDs for every "
                "recommendation. Explain what evidence could overturn the recommendation."
            ),
            "capacity": (
                "Analyze the capacity test records below. Explain why throughput and "
                "tail latency can move in different directions. Recommend admission and "
                "batching experiments, distinguish per-request and aggregate metrics, and "
                "describe how to validate the recommendation under bursty traffic."
            ),
        }
        if family == "incident":
            doc = "".join(
                f"2026-10-24T10:{i%60:02d}:00Z service=checkout event={i:05d} "
                f"latency_ms={80+(i%13)*35} pool_wait_ms={5+(i%11)*41} "
                f"inflight={8+i%18} db_cpu=37% errors={i%7} "
                "trace=payment-retry pool_max=16 timeout_ms=800 retries=3\n"
                for i in range(3000)
            )
        elif family == "review":
            doc = "".join(
                f"File worker_{i:03d}.py\n"
                "async def handle(queue, semaphore, session):\n"
                "    await semaphore.acquire()\n"
                "    item = await queue.get()\n"
                "    response = await session.get(item.url)\n"
                "    if response.status != 200:\n"
                "        return None\n"
                "    queue.task_done()\n"
                "    semaphore.release()\n"
                "    return await response.json()\n"
                "Operational note: cancellation and timeout can happen at each await.\n"
                for i in range(700)
            )
        elif family == "retrieval":
            doc = "".join(
                f"Note N{i:04d}: Test cohort={i%5}, prompt_tokens={1024*(1+i%8)}, "
                f"concurrency={1+i%8}, warm_cache={bool(i%2)}, "
                f"mean_ttft_ms={600+i%11*170}, p95_complete_ms={1500+i%17*380}. "
                "The client reused prefixes in warm runs; cold runs used distinct documents. "
                "Batch limits changed between cohorts. Request counts were small.\n"
                for i in range(2000)
            )
        else:
            doc = "".join(
                f'Run R{i:04d}: arrival_pattern={"burst" if i%2 else "steady"}, '
                f"batch_limit={1+i%8}, offered_rps={1+i%6}, "
                f"completed_rps={1+i%4}, queue_p95_ms={40+i%19*90}, "
                f"first_token_p95_ms={400+i%13*130}, "
                f"completion_p95_ms={1500+i%23*150}, output_tokens={128*(1+i%4)}. "
                "CPU utilization was modest; memory bandwidth and GPU clocks were not isolated.\n"
                for i in range(3000)
            )
        text = (
            "Engineering task:\n"
            + questions[family]
            + "\nEvidence:\n__DOC__\nEnd of evidence.\n"
            + questions[family]
        )
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if family == "copy":
        rendered += "record 0000: alpha beta gamma delta.\nrecord 0001: alpha beta gamma delta.\n"
    head, tail = rendered.split("__DOC__")
    ids = tok.encode(doc, add_special_tokens=False)
    n = target - len(tok.encode(head + tail, add_special_tokens=False))
    for _ in range(40):
        result = head + tok.decode(ids[:n]) + tail
        actual = tok.encode(result, add_special_tokens=False)
        if len(actual) == target:
            break
        n += target - len(actual)
    assert len(actual) == target, (family, target, len(actual))
    return {
        "family": family,
        "input_tokens": target,
        "prompt": result,
        "input_ids": actual,
        "input_sha256": hashlib.sha256(json.dumps(actual).encode()).hexdigest(),
        "expected_visible": (
            "".join(
                f"record {i:04d}: alpha beta gamma delta.\n" for i in range(2, 3000)
            )
            if family == "copy"
            else None
        ),
    }


def design(args):
    preferred = [0.5, 0.25, 0.75, 0.1, 1.0]
    ratios = [r for r in preferred if r in args.ratios]
    ratios.extend(r for r in args.ratios if r not in ratios)
    forward = [
        (mode, ratio)
        for i, ratio in enumerate(ratios)
        for mode in (["ar", "mtp"] if i % 2 == 0 else ["mtp", "ar"])
        if mode in args.modes
    ]
    return [
        {
            "pass": repeat + 1,
            "mode": mode,
            "ratio": ratio,
            "cases": [
                c
                for c in (
                    ["solo", "overlap", "burst", "natural"]
                    if repeat % 2 == 0
                    else ["solo", "natural", "burst", "overlap"]
                )
                if c in args.cases
            ],
        }
        for repeat in range(args.repeats)
        for mode, ratio in (forward if repeat % 2 == 0 else reversed(forward))
    ]


def corpus(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if args.corpus_file:
        data = json.loads(args.corpus_file.read_text())
        if "copy-4096" not in data:
            if len(data) != 1:
                raise ValueError("Select one model's corpus from a multi-model fixture")
            data = next(iter(data.values()))
        for key, entry in data.items():
            actual = tokenizer.encode(entry["prompt"], add_special_tokens=False)
            if actual != entry["input_ids"]:
                raise ValueError(f"Frozen corpus tokenizer differs: {key}")
        return data
    sizes = [256, 512, 2048] if args.quick else [1024, 2048, 4096, 16384]
    result = {f"copy-{n}": prompt(tokenizer, "copy", n) for n in sizes}
    if "natural" in args.cases:
        for family, size in [
            ("incident", 1024),
            ("retrieval", 2048),
            ("review", 4096),
            ("capacity", 8192),
        ]:
            result[f"{family}-{size}"] = prompt(tokenizer, family, size)
    return result


async def request(client, inputs, key, limit, output, label, references, first=None):
    entry = inputs[key]
    body = {
        "model": "benchmark-model",
        "prompt": entry["prompt"],
        "max_tokens": limit,
        "temperature": 0,
        "top_p": 1,
        "repetition_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    request_id = f"dfr-{time.monotonic_ns()}"
    row = {
        "request_id": request_id,
        "submitted_unix": time.time(),
        "label": label,
        "corpus": key,
        "input_sha256": entry["input_sha256"],
        "expected_input_tokens": entry["input_tokens"],
        "output_cap": limit,
        "family": entry["family"],
        "events": [],
        "text": "",
        "usage": {},
    }
    wire = []
    try:
        async with client.stream(
            "POST", "/v1/completions", json=body, headers={"X-Request-ID": request_id}
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                elapsed = time.perf_counter() - start
                wire.append({"elapsed_s": elapsed, "line": line})
                value = line[5:].strip()
                if value == "[DONE]":
                    break
                chunk = json.loads(value)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                if chunk.get("usage"):
                    row["usage"] = chunk["usage"]
                for choice in chunk.get("choices", []):
                    text = choice.get("text") or ""
                    if text:
                        row["events"].append(elapsed)
                        row["text"] += text
                        if first is not None:
                            first.set()
        row["completion_s"] = time.perf_counter() - start
        row["ttft_s"] = row["events"][0] if row["events"] else None
        gaps = [b - a for a, b in zip(row["events"], row["events"][1:])]
        row["maximum_visible_gap_s"] = max(gaps, default=0)
        row["p95_visible_gap_s"] = (
            statistics.quantiles(gaps, n=20, method="inclusive")[18]
            if len(gaps) > 1
            else max(gaps, default=0)
        )
        row["text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
        actual = row["usage"].get("completion_tokens", 0)
        valid = (
            row["ttft_s"] is not None
            and row["usage"].get("prompt_tokens") == entry["input_tokens"]
        )
        valid = valid and (
            actual == limit if entry["family"] == "copy" else 0 < actual <= limit
        )
        if entry["family"] == "copy":
            visible = row["text"].replace("\r\n", "\n").lstrip("\n")
            valid = (
                valid
                and bool(visible)
                and entry["expected_visible"].startswith(visible)
            )
        refkey = f"{key}/{limit}"
        signature = [row["text_sha256"], actual]
        reference = references.setdefault(refkey, signature)
        row["reference_match"] = signature == reference
        if entry["family"] == "copy":
            valid = valid and row["reference_match"]
        row["valid"] = valid
        row["primary_eligible"] = valid and entry["family"] == "copy"
        if not valid:
            raise RuntimeError(
                f"Work/structured continuation/reference check failed: {label}"
            )
    except BaseException as error:
        row["error"] = repr(error)
        raise
    finally:
        save(
            output / "wire" / f"{label}.json",
            {"request": body, "sse": wire, "result": row},
        )
    return row


async def run_case(client, inputs, name, output, references, quick):
    begin = time.perf_counter()
    started_unix = time.time()
    short, long, cap, ongoing = (
        (512, 2048, 16, 128) if quick else (4096, 16384, 128, 768)
    )
    if name == "overlap":
        first = asyncio.Event()
        old = asyncio.create_task(
            request(
                client,
                inputs,
                f"copy-{512 if quick else 2048}",
                ongoing,
                output,
                "existing",
                references,
                first,
            )
        )
        ready = asyncio.create_task(first.wait())
        tasks = [old, ready]
        try:
            done, _ = await asyncio.wait(
                [old, ready], timeout=300, return_when=asyncio.FIRST_COMPLETED
            )
            if ready not in done:
                if old in done:
                    old.result()
                raise RuntimeError("No first visible content before overlap deadline")
            await asyncio.sleep(1)
            new = asyncio.create_task(
                request(
                    client, inputs, f"copy-{long}", cap, output, "newcomer", references
                )
            )
            tasks.append(new)
            rows = await asyncio.gather(old, new)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    else:
        specs = [(f"copy-{short}", cap)] * (4 if name == "burst" else 1)
        if name == "natural":
            specs = [
                (f"{family}-{n}", 128)
                for family, n in [
                    ("incident", 1024),
                    ("retrieval", 2048),
                    ("review", 4096),
                    ("capacity", 8192),
                ]
            ]
        tasks = [
            asyncio.create_task(
                request(client, inputs, key, count, output, f"request-{i}", references)
            )
            for i, (key, count) in enumerate(specs)
        ]
        try:
            rows = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    wall = time.perf_counter() - begin
    result = {
        "case": name,
        "started_unix": started_unix,
        "whole_case_s": wall,
        "mean_ttft_s": statistics.mean(r["ttft_s"] for r in rows),
        "mean_completion_s": statistics.mean(r["completion_s"] for r in rows),
        "output_tokens_per_s": sum(r["usage"]["completion_tokens"] for r in rows)
        / wall,
        "requests": rows,
    }
    save(output / "summary.json", result)
    return result


async def monitor(process, output, initial_swap):
    low = 0
    with output.open("w") as stream:
        while process.poll() is None:
            memory, swap = psutil.virtual_memory(), psutil.swap_memory()
            stream.write(
                json.dumps(
                    {
                        "unix": time.time(),
                        "available_bytes": memory.available,
                        "swap_used_bytes": swap.used,
                    }
                )
                + "\n"
            )
            stream.flush()
            low = low + 1 if memory.available < 1.25 * 2**30 else 0
            if low >= 3 or swap.used - initial_swap > 1.5 * 2**30:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                raise RuntimeError(
                    "Resource guard: low available memory or swap growth"
                )
            await asyncio.sleep(1)


async def arm(args, profile, output, inputs, references):
    # Refuse a busy port before sending any requests or starting a process.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    output.mkdir()
    state = output / "state"
    state.mkdir()
    models = state / "models"
    models.mkdir()
    (models / "benchmark-model").symlink_to(args.model_path, target_is_directory=True)
    api_key = "sk-" + secrets.token_hex(24)
    save(
        state / "settings.json",
        {
            "auth": {"api_key": api_key},
            "scheduler": {
                "decode_fair_share": profile["ratio"],
                "decode_fairness": True,
                "max_concurrent_requests": 4,
                "chunked_prefill": False,
            },
            "cache": {"enabled": False},
        },
    )
    save(
        state / "model_settings.json",
        {
            "version": 1,
            "models": {
                "benchmark-model": {
                    "max_context_window": 32768,
                    "mtp_enabled": profile["mode"] == "mtp",
                    "dflash_enabled": False,
                    "mtp_adaptive_max_depth": None,
                    "mtp_fixed_depth": None,
                }
            },
        },
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OMLX_", "MLX_", "VLLM_"))
    }
    env.pop("PYTHONPATH", None)
    env.update(
        OMLX_BASE_PATH=str(state),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    command = [
        str(args.server_python),
        str(Path(__file__).resolve()),
        "_serve",
        str(state),
        str(output),
        str(args.source_root),
        "serve",
        "--base-path",
        str(state),
        "--model-dir",
        str(models),
        "--headless",
        "--no-cache",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-concurrent-requests",
        "4",
        "--memory-guard-gb",
        str(args.memory_guard_gb),
    ]
    save(output / "launch.json", {**profile, "command": command, "quick": args.quick})
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=args.source_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        guard = asyncio.create_task(
            monitor(process, output / "memory.jsonl", psutil.swap_memory().used)
        )
        try:
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{args.port}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=900,
                trust_env=False,
            ) as client:
                for _ in range(300):
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited: {process.returncode}")
                    try:
                        if (
                            await client.get("/v1/models", timeout=2)
                        ).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(1)
                else:
                    raise RuntimeError("Server readiness timeout")
                login = await client.post("/admin/api/login", json={"api_key": api_key})
                login.raise_for_status()
                runtime = json.loads((output / "runtime.json").read_text())
                if runtime["versions"]["transformers"] != importlib.metadata.version(
                    "transformers"
                ):
                    raise RuntimeError(
                        "Run the harness and server with matching tokenizer dependencies"
                    )
                settings = (await client.get("/admin/api/global-settings")).json()
                effective = settings["scheduler"]
                observed = effective.get("decode_fair_share")
                if observed is None and not args.baseline_default:
                    raise RuntimeError("Server does not expose Decode Fairness Ratio")
                if (
                    effective["decode_fairness"] is not True
                    or (observed if observed is not None else 0.5) != profile["ratio"]
                ):
                    raise RuntimeError(
                        "Effective settings differ from requested settings"
                    )
                settings.pop("auth", None)
                save(output / "effective-settings.json", settings)
                warmkey = f"copy-{256 if args.quick else 1024}"
                warmcap = 32
                warm = [
                    await request(
                        client,
                        inputs,
                        warmkey,
                        warmcap,
                        output / "warmups",
                        "serial",
                        references,
                    )
                ]
                warm.extend(
                    await asyncio.gather(
                        *(
                            request(
                                client,
                                inputs,
                                warmkey,
                                warmcap,
                                output / "warmups",
                                f"parallel-{i}",
                                references,
                            )
                            for i in range(4)
                        )
                    )
                )
                save(output / "warmups.json", warm)
                activated = "MTP path activated" in (output / "server.log").read_text()
                if activated != (profile["mode"] == "mtp"):
                    raise RuntimeError(
                        "Actual MTP activation does not match requested mode"
                    )
                result = {**profile, "cases": []}
                for name in profile["cases"]:
                    if guard.done():
                        guard.result()
                    await asyncio.sleep(2)
                    print(
                        json.dumps({"event": "case", "arm": output.name, "case": name}),
                        flush=True,
                    )
                    case = asyncio.create_task(
                        run_case(
                            client, inputs, name, output / name, references, args.quick
                        )
                    )
                    done, _ = await asyncio.wait(
                        [case, guard], return_when=asyncio.FIRST_COMPLETED
                    )
                    if guard in done:
                        case.cancel()
                        await asyncio.gather(case, return_exceptions=True)
                        guard.result()
                        raise RuntimeError("Resource monitor ended unexpectedly")
                    result["cases"].append(case.result())
                    save(output / "summary.json", result)
                return result
        except BaseException:
            if guard.done() and not guard.cancelled():
                guard.result()
            raise
        finally:
            guard.cancel()
            await asyncio.gather(guard, return_exceptions=True)
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.to_thread(process.wait, 25)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    await asyncio.to_thread(process.wait, 10)
            save(
                output / "shutdown.json",
                {"exit_code": process.returncode, "owned_pid": process.pid},
            )


async def main(args):
    order = design(args)
    if args.dry_run:
        print(json.dumps(order, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = corpus(args)
    save(args.output_dir / "corpora.json", inputs)
    identity = source_identity(args.source_root)
    model_files = [
        {"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(args.model_path.iterdir())
        if p.suffix in (".json", ".jinja", ".safetensors")
    ]
    save(
        args.output_dir / "provenance.json",
        {
            "source": identity,
            "model_files": model_files,
            "platform": platform.platform(),
            "chip": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
            "memory_bytes": psutil.virtual_memory().total,
            "order": order,
            "quick": args.quick,
            "baseline_default": args.baseline_default,
            "corpus_sha256": sha(args.output_dir / "corpora.json"),
        },
    )
    references = {}
    results = []
    try:
        for i, profile in enumerate(order):
            results.append(
                await arm(
                    args,
                    profile,
                    args.output_dir
                    / f"{i + 1:02d}-{profile['mode']}-{profile['ratio']}",
                    inputs,
                    references,
                )
            )
            save(args.output_dir / "results.json", results)
            save(args.output_dir / "reference.json", references)
            if source_identity(args.source_root) != identity:
                raise RuntimeError("Source changed during measurement")
    except BaseException as error:
        save(
            args.output_dir / "failure.json",
            {"error": repr(error), "completed_arms": len(results)},
        )
        raise
    rows = [
        {
            "pass": result["pass"],
            "mode": result["mode"],
            "ratio": result["ratio"],
            **{k: v for k, v in case.items() if k != "requests"},
        }
        for result in results
        for case in result["cases"]
    ]
    with (args.output_dir / "trial-metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"event": "complete", "arms": len(results)}), flush=True)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--server-python", type=Path, default=Path(sys.executable))
    p.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1])
    p.add_argument("--modes", nargs="+", choices=["ar", "mtp"], default=["ar", "mtp"])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument(
        "--cases",
        nargs="+",
        choices=["solo", "burst", "overlap", "natural"],
        default=["solo", "burst", "overlap", "natural"],
    )
    p.add_argument(
        "--corpus-file",
        type=Path,
        help="Replay the report's frozen corpus instead of generating inputs",
    )
    p.add_argument("--port", type=int, default=18109)
    p.add_argument("--memory-guard-gb", type=float, default=40)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Short integration checks; not comparable with full-profile timings",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned order without loading models or starting servers",
    )
    p.add_argument(
        "--baseline-default",
        action="store_true",
        help="Permit a pre-exposure server only at its stock 0.5 default",
    )
    args = p.parse_args()
    if args.repeats < 1 or any(
        not math.isfinite(r) or not 0 <= r <= 1 for r in args.ratios
    ):
        p.error("Use at least one repeat and finite ratios from 0 to 1")
    if (
        not 1 <= args.port <= 65535
        or not math.isfinite(args.memory_guard_gb)
        or args.memory_guard_gb <= 0
    ):
        p.error("Use a valid port and a positive finite memory guard")
    if args.baseline_default and set(args.ratios) != {0.5}:
        p.error("--baseline-default requires --ratios 0.5")
    if args.quick and args.corpus_file:
        p.error("--quick generates its own shorter corpus; omit --corpus-file")
    # Preserve the venv entry point: resolving its symlink selects the base
    # interpreter and silently loses the selected environment's packages.
    args.server_python = args.server_python.expanduser().absolute()
    for name in ("model_path", "output_dir", "source_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_serve":
        serve(*sys.argv[2:5], sys.argv[5:])
    else:
        asyncio.run(main(arguments()))
