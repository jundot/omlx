#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch the two-Mac prefill/decode handoff worker over the configured fabric."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from omlx.cluster.disaggregated_worker import EVENT_PREFIX
from omlx.cluster.launch import (
    _available_control_port,
    _available_launch_ports,
    _install_cluster_ssh_wrapper,
    _rank_python_module_argv,
)
from omlx.cluster.registry import ClusterRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment-id", default="ds4-tp2-equal-safe-v10"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--completion-tokens", type=int, default=32)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--prefill-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--pipeline-requests", type=int, choices=(1, 2, 4, 8), default=1
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    registry = ClusterRegistry(Path.home() / ".omlx")
    deployment = registry.get(args.deployment_id)
    if deployment is None:
        raise SystemExit(f"unknown cluster deployment {args.deployment_id!r}")
    if deployment.world_size != 2:
        raise SystemExit("prototype currently requires a two-node deployment")
    model = Path(args.model).expanduser().resolve()
    if not model.is_dir():
        raise SystemExit(f"model directory does not exist: {model}")

    with tempfile.TemporaryDirectory(prefix="omlx-disaggregated-") as temporary:
        root = Path(temporary)
        _install_cluster_ssh_wrapper(root)
        hostfile = root / "hostfile.json"
        hostfile.write_text(
            json.dumps(
                deployment.hostfile_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        _unused_api_port, collective_port = _available_launch_ports(deployment)
        control_host = deployment.hosts[0].ips[0]
        control_port = _available_control_port(control_host)
        control_token = "disaggregated-prefill-decode-v1"
        launcher = (
            "from mlx._distributed_utils.launch import main; "
            "raise SystemExit(main() or 0)"
        )
        argv = [
            sys.executable,
            "-c",
            launcher,
            "--hostfile",
            str(hostfile),
            "--starting-port",
            str(collective_port),
            "--cwd",
            str(Path(__file__).resolve().parents[1]),
            "--",
            *_rank_python_module_argv(
                [host.python_executable for host in deployment.hosts],
                fallback=sys.executable,
                module="omlx.cluster.disaggregated_worker",
            ),
            "--model",
            str(model),
            "--backend",
            deployment.distributed_init_backend,
            "--prompt-tokens",
            str(args.prompt_tokens),
            "--completion-tokens",
            str(args.completion_tokens),
            "--prefill-step-size",
            str(args.prefill_step_size),
            "--prefill-rank",
            str(args.prefill_rank),
            "--pipeline-requests",
            str(args.pipeline_requests),
            "--control-host",
            control_host,
            "--control-port",
            str(control_port),
            "--control-token",
            control_token,
            "--deployment-id",
            "disaggregated-prefill-decode-prototype",
        ]
        environment = os.environ.copy()
        environment["PATH"] = f"{root}{os.pathsep}{environment.get('PATH', '')}"
        environment["SSH_ASKPASS_REQUIRE"] = "never"
        environment["PYTHONUNBUFFERED"] = "1"
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            env=environment,
        )

    events = []
    for line in completed.stdout.splitlines():
        marker = line.find(EVENT_PREFIX)
        if marker < 0:
            continue
        try:
            events.append(json.loads(line[marker + len(EVENT_PREFIX) :]))
        except json.JSONDecodeError:
            pass
    result_event = next(
        (
            event
            for event in events
            if event.get("type") in {"result", "pipeline_result"}
        ),
        None,
    )
    error_events = [event for event in events if event.get("type") == "error"]
    effective_returncode = (
        completed.returncode
        if completed.returncode != 0
        else (
            0
            if result_event is not None
            and result_event.get("parity") is True
            and not error_events
            else 1
        )
    )
    report = {
        "returncode": effective_returncode,
        "launcher_returncode": completed.returncode,
        "events": events,
        "result": result_event,
        "stdout_tail": completed.stdout.splitlines()[-40:],
        "stderr_tail": completed.stderr.splitlines()[-80:],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return effective_returncode


if __name__ == "__main__":
    raise SystemExit(main())
