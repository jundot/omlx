# SPDX-License-Identifier: Apache-2.0

import io
import json
import platform
import signal
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from omlx.cluster import launch
from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.launch import (
    CudaFabricProbeHost,
    DistributedLaunchError,
    _available_launch_ports,
    _cluster_ssh_argv,
    _install_cluster_ssh_wrapper,
    _local_probe_versions,
    _local_runtime_versions,
    build_mlx_launch_argv,
    discover_remote_python_executable,
    discover_remote_system_python,
    preflight_remote_hosts,
    probe_remote_host,
    probe_remote_system_host,
    run_cuda_fabric_probe,
)
from omlx.cluster.models import CLUSTER_PROTOCOL_VERSION
from omlx.cluster.planner import PipelineAssignment


def _deployment(model: str = "org/model") -> ClusterDeployment:
    assignments = (
        PipelineAssignment("local", 0, 3, 8, 5, 1, 1, 16),
        PipelineAssignment("studio", 1, 0, 3, 3, 1, 1, 8),
    )
    return ClusterDeployment(
        deployment_id="cluster-test",
        model=model,
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("studio", "user@studio.local", ("10.0.0.2",)),
        ),
        assignments=assignments,
        plan_hash="c" * 64,
    )


def test_launcher_argv_keeps_model_as_one_argument(tmp_path):
    model = "org/model with spaces;$(touch nope)"
    deployment = _deployment(model)
    hostfile = (tmp_path / "hosts.json").resolve()

    argv = build_mlx_launch_argv(
        deployment,
        hostfile=hostfile,
        api_port=32100,
        collective_port=32120,
        python_executable="/opt/omlx/bin/python",
        cwd=Path("/opt/omlx"),
    )

    assert argv[0] == "/opt/omlx/bin/python"
    assert argv[argv.index("--model") + 1] == model
    assert argv[argv.index("--backend") + 1] == "ring"
    assert argv[argv.index("--starting-port") + 1] == "32120"
    assert argv[argv.index("--port") + 1] == "32100"
    assert argv[argv.index("--plan-hash") + 1] == deployment.plan_hash
    assert argv[argv.index("--connections-per-ip") + 1] == "2"
    assert argv[argv.index("--execution-profile") + 1] == "balanced"
    assert argv[argv.index("--pipeline-microbatch-size") + 1] == "4"
    assert "--cache-affinity" in argv
    assert "--async-overlap" in argv
    assert argv.count(model) == 1


def test_launcher_rejects_overlapping_api_and_collective_ports(tmp_path):
    with pytest.raises(ValueError, match="must be distinct"):
        build_mlx_launch_argv(
            _deployment(),
            hostfile=(tmp_path / "hosts.json").resolve(),
            api_port=32100,
            collective_port=32100,
            python_executable="/opt/omlx/bin/python",
        )


def test_cuda_fabric_probe_launches_a_real_two_rank_nccl_job():
    captured = {}

    def runner(argv, *, timeout, env):
        captured["argv"] = argv
        captured["timeout"] = timeout
        captured["hostfile"] = json.loads(
            Path(argv[argv.index("--hostfile") + 1]).read_text()
        )
        assert env["SSH_ASKPASS_REQUIRE"] == "never"
        rate = 3 * 1024**3
        stdout = "\n".join(
            json.dumps(
                {
                    "type": "nccl_fabric_result",
                    "rank": rank,
                    "world_size": 2,
                    "payload_bytes_per_second": rate,
                }
            )
            for rank in (0, 1)
        )
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    result = run_cuda_fabric_probe(
        (
            CudaFabricProbeHost(
                "spark-a",
                "spark-a.local",
                ("192.168.100.1", "192.168.101.1"),
                ("enp1s0f0np0", "enp2s0f0np0"),
                ("mlx5_0", "mlx5_1"),
            ),
            CudaFabricProbeHost(
                "spark-b",
                "spark-b.local",
                ("192.168.100.2", "192.168.101.2"),
                ("enp1s0f0np0", "enp2s0f0np0"),
                ("mlx5_0", "mlx5_1"),
            ),
        ),
        python_executable="/opt/omlx/bin/python",
        runner=runner,
    )

    assert result["verified"] is True
    assert result["transport"] == "nccl"
    assert result["payload_bytes_per_second"] == 3 * 1024**3
    assert result["group_id"].startswith("connectx-")
    assert captured["hostfile"]["backend"] == "nccl"
    assert [host["ips"] for host in captured["hostfile"]["hosts"]] == [
        ["192.168.100.1", "192.168.101.1"],
        ["192.168.100.2", "192.168.101.2"],
    ]
    assert captured["argv"][captured["argv"].index("--backend") + 1] == "nccl"
    assert "omlx.cluster.nccl_fabric_worker" in captured["argv"]


def test_cuda_fabric_member_rejects_an_interface_shell_fragment():
    with pytest.raises(ValueError, match="invalid interface"):
        CudaFabricProbeHost(
            "spark-a",
            "spark-a.local",
            ("192.168.100.1",),
            ("eth0;touch-nope",),
            ("mlx5_0",),
        )


def test_launcher_selects_the_probed_python_path_for_each_rank(tmp_path):
    deployment = _deployment()
    deployment = replace(
        deployment,
        hosts=(
            replace(deployment.hosts[0], python_executable="/Applications/oMLX/python"),
            replace(deployment.hosts[1], python_executable="/opt/omlx/bin/python"),
        ),
    )

    argv = build_mlx_launch_argv(
        deployment,
        hostfile=(tmp_path / "hosts.json").resolve(),
        api_port=32100,
        collective_port=32120,
        python_executable="/Applications/oMLX/python",
    )
    worker = argv[argv.index("--") + 1 :]

    assert worker[:2] == ["/bin/sh", "-c"]
    assert '${MLX_RANK:-}' in worker[2]
    assert "/Applications/oMLX/python" in worker[2]
    assert "/opt/omlx/bin/python" in worker[2]
    assert worker[3:6] == [
        "omlx-rank-python",
        "-m",
        "omlx.cluster.inference_worker",
    ]


def test_ring_launch_reserves_full_collective_port_span():
    deployment = _deployment()

    api_port, collective_port = _available_launch_ports(deployment)

    assert api_port not in range(
        collective_port,
        collective_port + sum(len(host.ips) for host in deployment.hosts),
    )


def test_supervisor_preserves_structured_peer_loss_reason():
    supervisor = launch.DistributedJobSupervisor(_deployment(), preflight=False)
    reason = "rank 1 on studio stopped reporting"
    supervisor._stderr.append("generic launcher noise")

    supervisor._drain(
        io.StringIO(
            f'{launch._EVENT_PREFIX}{{"type":"peer_lost","reason":"{reason}"}}\n'
        ),
        supervisor._stdout,
        True,
    )

    assert supervisor.failure_event == {
        "type": "peer_lost",
        "reason": reason,
    }
    assert supervisor.status().failure_reason == reason
    assert reason in supervisor._exit_detail(1)
    assert "generic launcher noise" not in supervisor._exit_detail(1)


def test_supervisor_prefers_rank_marker_over_mlx_cleanup_traceback(monkeypatch):
    supervisor = launch.DistributedJobSupervisor(_deployment(), preflight=False)
    supervisor._stderr.append(
        "subprocess.CalledProcessError: remote PID cleanup returned exit status 1"
    )
    monkeypatch.setattr(
        launch,
        "read_marker",
        lambda _path: {
            "deployment_id": "cluster-test",
            "plan_hash": "c" * 64,
            "rank": 0,
            "phase": "failed",
            "error": "InsufficientMemoryError: planned stage exceeds this Mac",
        },
    )
    monkeypatch.setattr(
        launch,
        "read_remote_marker",
        lambda *_a, **_k: (None, None, None, ""),
    )

    detail = supervisor._exit_detail(0)

    assert "rank 0 (local): InsufficientMemoryError" in detail
    assert "CalledProcessError" not in detail


def test_supervisor_collects_every_rank_ready_event():
    supervisor = launch.DistributedJobSupervisor(_deployment(), preflight=False)
    for rank, node_id in enumerate(("local", "studio")):
        event = {
            "type": "rank_ready",
            "deployment_id": "cluster-test",
            "plan_hash": "c" * 64,
            "rank": rank,
            "node_id": node_id,
            "world_size": 2,
            "measured_weight_bytes": 10 + rank,
        }
        supervisor._drain(
            io.StringIO(launch._EVENT_PREFIX + json.dumps(event) + "\n"),
            supervisor._stdout,
            True,
        )

    status = supervisor.status().to_dict()

    assert [rank["rank"] for rank in status["ranks"]] == [0, 1]
    assert status["ranks"][1]["measured_weight_bytes"] == 11


def test_supervisor_stop_kills_rank_left_after_launcher_exits(monkeypatch):
    class Launcher:
        pid = 43210
        stdout = None
        stderr = None

        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0
            return 0

    supervisor = launch.DistributedJobSupervisor(
        _deployment(),
        preflight=False,
        stop_timeout=0.1,
    )
    launcher = Launcher()
    supervisor.process = launcher
    signals = []
    waits = iter((False, True))

    monkeypatch.setattr(launch, "_process_group_alive", lambda _pgid: True)
    monkeypatch.setattr(
        launch,
        "_wait_for_process_group_exit",
        lambda _pgid, _timeout: next(waits),
    )
    monkeypatch.setattr(
        launch.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )

    supervisor.stop()

    assert signals == [
        (launcher.pid, signal.SIGTERM),
        (launcher.pid, signal.SIGKILL),
    ]
    assert supervisor.process is None
    assert supervisor.status().phase == "stopped"


def test_supervisor_stop_reaps_group_when_launcher_already_exited(monkeypatch):
    class Launcher:
        pid = 43211
        stdout = None
        stderr = None
        returncode = 1

        def poll(self):
            return self.returncode

    supervisor = launch.DistributedJobSupervisor(
        _deployment(),
        preflight=False,
        stop_timeout=0.1,
    )
    supervisor.process = Launcher()
    signals = []

    monkeypatch.setattr(launch, "_process_group_alive", lambda _pgid: True)
    monkeypatch.setattr(
        launch,
        "_wait_for_process_group_exit",
        lambda _pgid, _timeout: True,
    )
    monkeypatch.setattr(
        launch.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )

    supervisor.stop()

    assert signals == [(43211, signal.SIGTERM)]
    assert supervisor.process is None


def test_remote_preflight_uses_prompt_free_noninteractive_ssh():
    calls = []
    versions = _local_runtime_versions()

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                versions
                | {
                    "cluster-protocol": CLUSTER_PROTOCOL_VERSION,
                    "python": platform.python_version(),
                    "model-exists": True,
                    "admission-ceiling-bytes": 1024**4,
                }
            ),
            stderr="",
        )

    result = preflight_remote_hosts(
        _deployment(),
        python_executable="/opt/omlx/bin/python",
        runner=runner,
    )

    assert len(result) == 2
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert "BatchMode=yes" in argv
    assert "StrictHostKeyChecking=accept-new" in argv
    assert "CheckHostIP=no" in argv
    assert "LogLevel=ERROR" in argv
    # The control channel must not die of an idle timeout: both MiniMax runs
    # that reached ready lost their Studio rank to a dropped ssh session.
    assert "ServerAliveInterval=15" in argv
    assert "TCPKeepAlive=yes" in argv
    # Target and command close the argv; asserting position of the target
    # relative to one option broke every time an option was added.
    assert argv[-2] == "user@studio.local"
    # A source wrapper can import oMLX without having wheel/editable metadata.
    # Preflight must accept that supported deployment shape and read the
    # package's canonical source version instead.
    assert "from omlx._version import __version__" in argv[-1]
    assert "package_version(n)" in argv[-1]
    assert "import omlx.adapter.output_parser" in argv[-1]
    assert kwargs["timeout"] == 8.0
    assert kwargs["check"] is False


def test_local_runtime_version_ignores_stale_installed_metadata(monkeypatch):
    real_version = launch.importlib.metadata.version
    metadata_lookups = []

    def stale_metadata(name):
        metadata_lookups.append(name)
        if name == "omlx":
            return "0.0.0-stale"
        return real_version(name)

    monkeypatch.setattr(launch.importlib.metadata, "version", stale_metadata)

    from omlx._version import __version__

    assert launch._local_runtime_versions()["omlx"] == __version__
    assert "omlx" not in metadata_lookups


def test_remote_memory_probe_is_fast_and_uses_prompt_free_ssh():
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"admission_ceiling_bytes": 213 * 1024**3}),
            stderr="",
        )

    ceiling = launch.probe_remote_admission_ceiling(
        "user@studio.local",
        python_executable="/opt/omlx/bin/python",
        runner=runner,
    )

    assert ceiling == 213 * 1024**3
    argv, kwargs = calls[0]
    assert "BatchMode=yes" in argv
    assert "StrictHostKeyChecking=accept-new" in argv
    assert "CheckHostIP=no" in argv
    assert "system_profiler" not in argv[-1]
    # The fast path probes the configured server port first and keeps 9000 as
    # a legacy candidate; both must stay in the port loop.
    assert "for port in" in argv[-1]
    assert "9000" in argv[-1]
    assert "/health" in argv[-1]
    assert kwargs["timeout"] == 8.0


def test_preflight_refuses_a_stage_above_the_live_rank_ceiling():
    deployment = _deployment()

    with pytest.raises(
        DistributedLaunchError,
        match=r"no ranks were started.*rank 0 \(local\).*rank 1 \(studio\)",
    ):
        launch._validate_deployment_admission(
            deployment,
            [
                {"rank": 0, "admission_ceiling_bytes": 5},
                {"rank": 1, "admission_ceiling_bytes": 3},
            ],
        )


def test_launcher_wraps_every_upstream_ssh_call_with_cluster_policy(tmp_path):
    wrapper = _install_cluster_ssh_wrapper(tmp_path)
    content = wrapper.read_text()
    argv = _cluster_ssh_argv("user@studio.local", "true")

    assert wrapper.name == "ssh"
    assert stat.S_IMODE(wrapper.stat().st_mode) == 0o700
    assert "BatchMode=yes" in content
    assert "ConnectTimeout=5" in content
    assert "StrictHostKeyChecking=accept-new" in content
    assert "CheckHostIP=no" in content
    assert "LogLevel=ERROR" in content
    assert "ServerAliveInterval=15" in content
    assert "ServerAliveCountMax=4" in content
    assert "TCPKeepAlive=yes" in content
    assert Path(argv[0]).is_absolute()
    assert "BatchMode=yes" in argv
    assert "StrictHostKeyChecking=accept-new" in argv
    assert "CheckHostIP=no" in argv


def test_remote_preflight_rejects_runtime_drift():
    versions = _local_runtime_versions() | {
        "mlx": "0.0.0-drift",
        "cluster-protocol": CLUSTER_PROTOCOL_VERSION,
        "model-exists": True,
    }

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(versions),
            stderr="",
        )

    with pytest.raises(DistributedLaunchError, match="runtime mismatch.*mlx"):
        preflight_remote_hosts(
            _deployment(),
            python_executable="/opt/omlx/bin/python",
            runner=runner,
        )


def test_remote_preflight_requires_same_model_path():
    versions = _local_runtime_versions() | {
        "cluster-protocol": CLUSTER_PROTOCOL_VERSION,
        "python": platform.python_version(),
        "model-exists": False,
    }

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(versions),
            stderr="",
        )

    with pytest.raises(DistributedLaunchError, match="model directory is missing"):
        preflight_remote_hosts(
            _deployment("/models/nemotron"),
            python_executable="/opt/omlx/bin/python",
            runner=runner,
        )


def test_remote_preflight_requires_matching_model_identity(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "nemotron"
    model.mkdir()
    monkeypatch.setattr(
        launch,
        "validate_staged_model",
        lambda path, start, end: {
            "model_identity": "local",
            "stage_ready": True,
        },
    )
    versions = _local_runtime_versions() | {
        "cluster-protocol": CLUSTER_PROTOCOL_VERSION,
        "python": platform.python_version(),
        "model-exists": True,
        "model_identity": "remote",
        "stage_ready": True,
    }

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(versions),
            stderr="",
        )

    with pytest.raises(DistributedLaunchError, match="model identity differs"):
        preflight_remote_hosts(
            _deployment(str(model)),
            python_executable="/opt/omlx/bin/python",
            runner=runner,
        )


def test_remote_preflight_rejects_an_incomplete_rank_stage(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "nemotron"
    model.mkdir()
    monkeypatch.setattr(
        launch,
        "validate_staged_model",
        lambda path, start, end: {
            "model_identity": "same",
            "stage_ready": True,
        },
    )
    versions = _local_runtime_versions() | {
        "cluster-protocol": CLUSTER_PROTOCOL_VERSION,
        "python": platform.python_version(),
        "model-exists": True,
        "model_identity": "same",
        "stage_ready": False,
        "missing_files": ["model-00002-of-00058.safetensors"],
    }

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(versions),
            stderr="",
        )

    with pytest.raises(DistributedLaunchError, match="model stage is incomplete"):
        preflight_remote_hosts(
            _deployment(str(model)),
            python_executable="/opt/omlx/bin/python",
            runner=runner,
        )


def test_peer_probe_is_prompt_free_and_reports_runtime_compatibility():
    calls = []
    versions = _local_probe_versions()
    status = {
        "protocol_version": "1.0",
        "node": {"hostname": "studio", "recommended_working_set_bytes": 123},
        "runtime": {
            "omlx_version": versions["omlx"],
            "mlx_version": versions["mlx"],
            "mlx_lm_version": versions["mlx-lm"],
            "python_version": platform.python_version(),
        },
        "transport": {
            "rdma": {"enabled": True, "devices": ["rdma_en5"]},
        },
    }

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(status),
            stderr="",
        )

    result = probe_remote_host(
        "user@studio.local",
        route_to="192.168.5.1",
        python_executable="/opt/omlx/bin/python",
        runner=runner,
    )

    assert result["ok"] is True
    assert result["runtime_compatible"] is True
    argv, kwargs = calls[0]
    assert "BatchMode=yes" in argv
    assert "StrictHostKeyChecking=accept-new" in argv
    assert "CheckHostIP=no" in argv
    assert kwargs["check"] is False
    assert "--route-to 192.168.5.1" in argv[-1]


def test_peer_probe_rejects_protocol_drift():
    versions = _local_probe_versions()
    status = {
        "protocol_version": "future",
        "node": {},
        "runtime": {
            "omlx_version": versions["omlx"],
            "mlx_version": versions["mlx"],
            "mlx_lm_version": versions["mlx-lm"],
            "python_version": platform.python_version(),
        },
        "transport": {},
    }

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(status),
            stderr="",
        )

    with pytest.raises(DistributedLaunchError, match="protocol mismatch"):
        probe_remote_host(
            "studio.local",
            python_executable="/opt/omlx/bin/python",
            runner=runner,
        )


def test_peer_probe_discovers_a_different_linux_python_path():
    versions = _local_probe_versions()
    status = {
        "protocol_version": CLUSTER_PROTOCOL_VERSION,
        "node": {"hostname": "spark-a"},
        "runtime": {
            "omlx_version": versions["omlx"],
            "mlx_version": versions["mlx"],
            "mlx_lm_version": versions["mlx-lm"],
            "python_version": platform.python_version(),
            "python_executable": "/opt/omlx/bin/python",
        },
        "transport": {},
    }
    commands = []

    def runner(argv, **_kwargs):
        command = argv[-1]
        commands.append(command)
        if "omlx.cli" in command and command.startswith("/opt/omlx/bin/python"):
            return subprocess.CompletedProcess(argv, 0, json.dumps(status), "")
        # Path-shaped candidates echo themselves (import os,omlx) so a
        # launcher path survives discovery instead of being unwrapped to the
        # bare interpreter it fronts.
        if "import os,omlx" in command and command.startswith("/opt/omlx/bin/python"):
            return subprocess.CompletedProcess(argv, 0, "/opt/omlx/bin/python\n", "")
        return subprocess.CompletedProcess(argv, 127, "", "not found")

    result = probe_remote_host(
        "spark-a.local",
        python_executable="/Applications/oMLX/python",
        runner=runner,
    )

    assert result["runtime_compatible"] is True
    assert commands[-1].startswith("/opt/omlx/bin/python")
    assert result["status"]["runtime"]["python_executable"] == (
        "/opt/omlx/bin/python"
    )


def test_peer_probe_preserves_packaged_cluster_wrapper_path():
    versions = _local_probe_versions()
    status = {
        "protocol_version": CLUSTER_PROTOCOL_VERSION,
        "node": {"hostname": "studio"},
        "runtime": {
            "omlx_version": versions["omlx"],
            "mlx_version": versions["mlx"],
            "mlx_lm_version": versions["mlx-lm"],
            "python_version": platform.python_version(),
            "python_executable": "/Applications/oMLX.app/Contents/Python/python3",
        },
        "transport": {},
    }

    def runner(argv, **_kwargs):
        command = argv[-1]
        if command.startswith(
            "~/.omlx/bin/omlx-cluster-python"
        ) and "import os,omlx" in command:
            return subprocess.CompletedProcess(
                argv,
                0,
                "/Users/test/.omlx/bin/omlx-cluster-python\n",
                "",
            )
        if command.startswith("/Users/test/.omlx/bin/omlx-cluster-python"):
            return subprocess.CompletedProcess(argv, 0, json.dumps(status), "")
        return subprocess.CompletedProcess(argv, 127, "", "not found")

    result = probe_remote_host(
        "studio",
        python_executable="/missing/python",
        runner=runner,
    )

    assert result["runtime_compatible"] is True
    assert result["status"]["runtime"]["python_executable"] == (
        "/Users/test/.omlx/bin/omlx-cluster-python"
    )


def test_remote_python_discovery_prefers_packaged_cluster_launcher():
    commands = []

    def runner(argv, **_kwargs):
        command = argv[-1]
        commands.append(command)
        if command.startswith("~/.omlx/bin/omlx-cluster-python"):
            return subprocess.CompletedProcess(
                argv,
                0,
                "/Users/test/.omlx/bin/omlx-cluster-python\n",
                "",
            )
        return subprocess.CompletedProcess(argv, 127, "", "not found")

    discovered = discover_remote_python_executable(
        "studio",
        preferred="/missing/python",
        runner=runner,
    )

    assert discovered == "/Users/test/.omlx/bin/omlx-cluster-python"
    assert "os.path.expanduser" in commands[1]


def test_remote_python_discovery_falls_back_to_packaged_source_launcher():
    commands = []

    def runner(argv, **_kwargs):
        command = argv[-1]
        commands.append(command)
        if command.startswith("~/.omlx/bin/omlx-source-python"):
            return subprocess.CompletedProcess(
                argv,
                0,
                "/Users/test/.omlx/bin/omlx-source-python\n",
                "",
            )
        return subprocess.CompletedProcess(argv, 127, "", "not found")

    discovered = discover_remote_python_executable(
        "studio",
        preferred="/missing/python",
        runner=runner,
    )

    assert discovered == "/Users/test/.omlx/bin/omlx-source-python"
    assert any(
        command.startswith("~/.omlx/bin/omlx-source-python")
        for command in commands
    )


def test_remote_python_discovery_finds_gui_bootstrapped_cuda_worker():
    commands = []

    def runner(argv, **_kwargs):
        command = argv[-1]
        commands.append(command)
        if command.startswith("/opt/omlx-cluster-worker/venv/bin/python"):
            return subprocess.CompletedProcess(
                argv,
                0,
                "/opt/omlx-cluster-worker/venv/bin/python\n",
                "",
            )
        return subprocess.CompletedProcess(argv, 127, "", "not found")

    discovered = discover_remote_python_executable(
        "cuda-worker-1",
        preferred="/missing/python",
        runner=runner,
    )

    assert discovered == "/opt/omlx-cluster-worker/venv/bin/python"
    assert any(
        command.startswith("/opt/omlx-cluster-worker/venv/bin/python")
        for command in commands
    )


@pytest.mark.parametrize(
    "discover",
    [discover_remote_python_executable, discover_remote_system_python],
)
def test_remote_python_discovery_preserves_ssh_transport_failure(discover):
    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv,
            255,
            "",
            "Permission denied (publickey,password,keyboard-interactive).\n",
        )

    with pytest.raises(DistributedLaunchError, match="Permission denied"):
        discover(
            "user@studio.local",
            preferred="/usr/bin/python3",
            runner=runner,
        )


def test_system_python_discovery_keeps_genuine_interpreter_failure():
    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 127, "", "command not found")

    with pytest.raises(
        DistributedLaunchError,
        match="no Python interpreter for hardware discovery",
    ):
        discover_remote_system_python(
            "user@studio.local",
            preferred="/missing/python3",
            runner=runner,
        )


def test_preinstall_cuda_host_remains_visible_but_not_runnable():
    payload = {
        "protocol_version": None,
        "node": {
            "hostname": "cuda-worker-1",
            "accelerator": "cuda",
            "physical_memory_bytes": 128 * 1024**3,
            "distributed_backends": [],
            "fabric_kind": "connectx-7",
            "worker_runtime_ready": False,
        },
        "runtime": {
            "python_executable": "/usr/bin/python3",
            "omlx_version": "",
            "mlx_version": "",
            "mlx_lm_version": "",
        },
        "transport": {
            "rdma": {
                "devices": ["rocep1s0f1", "roceP2p1s0f1"],
                "addresses": {
                    "rocep1s0f1": "10.0.22.2",
                    "roceP2p1s0f1": "10.0.23.2",
                },
            }
        },
    }

    def runner(argv, **_kwargs):
        command = argv[-1]
        if "import sys; print(sys.executable)" in command:
            return subprocess.CompletedProcess(argv, 0, "/usr/bin/python3\n", "")
        if "worker_runtime_ready" in command:
            return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(argv, 1, "", "No module named 'omlx'")

    result = probe_remote_system_host(
        "cuda-worker-1",
        preferred_python="/Applications/oMLX/python",
        runner=runner,
    )

    assert result["ssh_reachable"] is True
    assert result["bootstrap_required"] is True
    assert result["runtime_compatible"] is False
    assert result["status"]["node"]["accelerator"] == "cuda"
    assert result["status"]["node"]["fabric_kind"] == "connectx-7"


def test_peer_probe_falls_back_to_preinstall_hardware_inventory(monkeypatch):
    expected = {
        "ok": False,
        "ssh": "cuda-worker-1",
        "ssh_reachable": True,
        "status": {"node": {"accelerator": "cuda"}},
        "runtime_compatible": False,
        "runtime_mismatches": ["oMLX worker runtime is not installed"],
        "bootstrap_required": True,
    }
    monkeypatch.setattr(
        launch,
        "discover_remote_python_executable",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            DistributedLaunchError("not installed")
        ),
    )
    monkeypatch.setattr(
        launch,
        "probe_remote_system_host",
        lambda *_args, **_kwargs: expected,
    )

    result = probe_remote_host(
        "cuda-worker-1",
        python_executable="/Applications/oMLX/python",
        runner=lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, 1, "", "No module named 'omlx'"
        ),
    )

    assert result is expected


def test_peer_probe_keeps_legacy_packaged_worker_visible(monkeypatch):
    raw = {
        "ok": False,
        "ssh": "studio",
        "ssh_reachable": True,
        "status": {
            "node": {"accelerator": "metal"},
            "warnings": ["oMLX worker runtime is not installed on this node."],
        },
        "runtime_compatible": False,
        "runtime_mismatches": ["oMLX worker runtime is not installed"],
        "bootstrap_required": True,
    }
    monkeypatch.setattr(
        launch,
        "discover_remote_python_executable",
        lambda *_args, **_kwargs: "/Users/test/.omlx/bin/omlx-source-python",
    )
    monkeypatch.setattr(
        launch,
        "probe_remote_system_host",
        lambda *_args, **_kwargs: raw,
    )

    result = probe_remote_host(
        "studio",
        python_executable="/missing/python",
        runner=lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv,
            2,
            "",
            "omlx: error: invalid choice: 'cluster'",
        ),
    )

    assert result["ssh_reachable"] is True
    assert result["runtime_compatible"] is False
    assert result["runtime_mismatches"] == [
        "installed oMLX worker does not support the cluster protocol"
    ]
    assert result["status"]["warnings"] == [
        "Update the installed oMLX worker to enable cluster execution."
    ]


# --- A packaged-app peer must need no hand-made shim (#2680) ----------------
#
# The coordinator runs inside the .app, so sys.executable is the bundled
# interpreter.  That exact path exists on the peer too and is therefore tried
# first, but without the launcher's PYTHONHOME/PYTHONPATH it cannot import
# omlx.  The peer was then declared "worker runtime is not installed" while the
# app sat in /Applications.  These reproduce the reported host exactly.

_BUNDLED_PYTHON = (
    "/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3.11"
)
_PEER_SHIM = "/Users/test/.omlx/bin/omlx-cluster-python"


def _packaged_app_peer_runner(status: dict) -> tuple[list[str], object]:
    """A peer where only the shipped cluster shim can import oMLX."""

    commands: list[str] = []

    def runner(argv, **_kwargs):
        command = argv[-1]
        commands.append(command)
        if command.startswith(_BUNDLED_PYTHON):
            return subprocess.CompletedProcess(
                argv, 1, "", "ModuleNotFoundError: No module named 'omlx'"
            )
        if command.startswith("~/.omlx/bin/omlx-cluster-python"):
            return subprocess.CompletedProcess(argv, 0, f"{_PEER_SHIM}\n", "")
        if command.startswith(_PEER_SHIM):
            return subprocess.CompletedProcess(argv, 0, json.dumps(status), "")
        return subprocess.CompletedProcess(argv, 127, "", "not found")

    return commands, runner


def test_packaged_app_peer_is_runtime_ready_without_a_hand_made_shim():
    versions = _local_probe_versions()
    status = {
        "protocol_version": CLUSTER_PROTOCOL_VERSION,
        "node": {"hostname": "studio", "distributed_backends": ["ring", "jaccl"]},
        "runtime": {
            "omlx_version": versions["omlx"],
            "mlx_version": versions["mlx"],
            "mlx_lm_version": versions["mlx-lm"],
            "python_version": platform.python_version(),
            "python_executable": _BUNDLED_PYTHON,
        },
        "transport": {},
    }
    commands, runner = _packaged_app_peer_runner(status)

    result = probe_remote_host(
        "studio",
        python_executable=_BUNDLED_PYTHON,
        runner=runner,
    )

    assert result["ok"] is True
    assert result["runtime_compatible"] is True
    assert result["runtime_mismatches"] == []
    assert "bootstrap_required" not in result
    # The shim, not the bundled interpreter, must be carried forward: reusing
    # the raw binary loses the environment and fails the very next probe.
    assert result["status"]["runtime"]["python_executable"] == _PEER_SHIM
    assert commands[-1].startswith(_PEER_SHIM)


def test_admission_ceiling_probe_discovers_the_peer_interpreter_when_unknown():
    """#2680: falling back to sys.executable 503'd node-budgets every poll."""

    commands, runner = _packaged_app_peer_runner({})

    def ceiling_runner(argv, **kwargs):
        command = argv[-1]
        if command.startswith(_PEER_SHIM) and "admission_ceiling_bytes" in command:
            commands.append(command)
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"admission_ceiling_bytes": 213 * 1024**3}), ""
            )
        return runner(argv, **kwargs)

    ceiling = launch.probe_remote_admission_ceiling(
        "studio",
        python_executable=None,
        runner=ceiling_runner,
    )

    assert ceiling == 213 * 1024**3
    assert commands[-1].startswith(_PEER_SHIM)


def test_admission_ceiling_probe_rediscovers_when_the_known_interpreter_broke():
    """A stored bundled-interpreter path must not keep failing forever."""

    commands, runner = _packaged_app_peer_runner({})

    def ceiling_runner(argv, **kwargs):
        command = argv[-1]
        if command.startswith(_PEER_SHIM) and "admission_ceiling_bytes" in command:
            commands.append(command)
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"admission_ceiling_bytes": 7}), ""
            )
        return runner(argv, **kwargs)

    ceiling = launch.probe_remote_admission_ceiling(
        "studio",
        python_executable=_BUNDLED_PYTHON,
        runner=ceiling_runner,
    )

    assert ceiling == 7
    assert commands[0].startswith(_BUNDLED_PYTHON)
    assert commands[-1].startswith(_PEER_SHIM)


def test_admission_ceiling_probe_reports_the_original_failure_when_no_peer_python():
    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv, 1, "", "ModuleNotFoundError: No module named 'omlx'"
        )

    with pytest.raises(DistributedLaunchError, match="memory ceiling probe failed"):
        launch.probe_remote_admission_ceiling(
            "studio",
            python_executable=_BUNDLED_PYTHON,
            runner=runner,
        )


# --- "Not installed" must be measured, not asserted (#2680) ----------------


def _system_probe_runner(payload: dict, *, evidence: list[str]):
    def runner(argv, **_kwargs):
        command = argv[-1]
        if "import sys; print(sys.executable)" in command:
            return subprocess.CompletedProcess(argv, 0, "/usr/bin/python3\n", "")
        if "worker_runtime_ready" in command:
            body = json.loads(json.dumps(payload))
            body["node"]["worker_runtime_evidence"] = evidence
            return subprocess.CompletedProcess(argv, 0, json.dumps(body), "")
        return subprocess.CompletedProcess(argv, 1, "", "No module named 'omlx'")

    return runner


_SYSTEM_PROBE_PAYLOAD = {
    "protocol_version": None,
    "node": {"hostname": "studio", "accelerator": "metal"},
    "runtime": {"python_executable": "/usr/bin/python3"},
    "transport": {},
    "warnings": [],
}


def test_preinstall_inventory_confirms_a_genuinely_absent_runtime():
    result = probe_remote_system_host(
        "studio",
        preferred_python=_BUNDLED_PYTHON,
        runner=_system_probe_runner(_SYSTEM_PROBE_PAYLOAD, evidence=[]),
    )

    assert result["bootstrap_required"] is True
    assert result["runtime_compatible"] is False
    assert result["runtime_mismatches"] == ["oMLX worker runtime is not installed"]
    assert result["worker_runtime_evidence"] == []


def test_preinstall_inventory_will_not_claim_missing_when_omlx_is_installed():
    result = probe_remote_system_host(
        "studio",
        preferred_python=_BUNDLED_PYTHON,
        runner=_system_probe_runner(
            _SYSTEM_PROBE_PAYLOAD, evidence=["/Applications/oMLX.app"]
        ),
    )

    assert result["bootstrap_required"] is True
    assert result["runtime_compatible"] is False
    assert result["runtime_mismatches"] == [
        "oMLX worker runtime could not be verified"
    ]
    assert result["worker_runtime_evidence"] == ["/Applications/oMLX.app"]
    assert result["status"]["warnings"] == [
        "oMLX is installed on this node but its worker runtime could not be run."
    ]


def test_preinstall_probe_measures_installation_instead_of_hardcoding_it():
    """The script itself must look; the verdict is not a constant."""

    assert "worker_runtime_evidence" in launch._REMOTE_SYSTEM_PROBE
    assert "/Applications/oMLX.app" in launch._REMOTE_SYSTEM_PROBE
    assert ".omlx/bin/omlx" in launch._REMOTE_SYSTEM_PROBE
    assert "find_spec" in launch._REMOTE_SYSTEM_PROBE


def test_preinstall_probe_reports_its_own_failures_on_stderr():
    """A swallowed error here reads as 'oMLX is absent'. Say what broke.

    stdout carries the JSON the caller parses, so diagnostics must go to
    stderr or they corrupt the payload.
    """

    script = launch._REMOTE_SYSTEM_PROBE
    body = script.split("def worker_runtime_evidence")[1].split("\ngpu_code")[0]
    # Neither blanket nor silent: every handler names its types and says what
    # it could not do.
    assert "except Exception" not in body
    assert body.count("except ") == body.count("note(")
    assert "except (OSError, ValueError) as exc" in body
    assert "cannot test %s" in body
    assert "cannot look up the omlx package" in body
    assert "sys.stderr.write" in script


def test_preinstall_probe_keeps_stdout_clean_when_every_lookup_fails():
    """Run the real script under an interpreter that cannot import omlx."""

    completed = subprocess.run(
        [sys.executable, "-c", launch._REMOTE_SYSTEM_PROBE],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent-omlx-home"},
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert isinstance(payload["node"]["worker_runtime_evidence"], list)


# --- The node role has to survive the argv the launcher actually runs -------
#
# Every previous test in this file asserted argv by index and never mentioned
# the role, so the flag that was supposed to carry it could go missing without
# turning anything red. The tests below start from a real ClusterDeployment,
# run build_mlx_launch_argv, and take the worker's *own* parser to the argv it
# produced — the same argv mlx.launch ships to each Mac. Nothing is passed by
# hand.

GIB = 1024**3

# Measured on the MacBook this was written for: the GPU ceiling, and the stage
# the planner put on it. Headless admits this; a workstation refuses it by
# about 3 GiB. That 3 GiB is the whole reason the role has to reach the rank.
_MACBOOK_CEILING_BYTES = 115_427_246_080
_MACBOOK_STAGE_BYTES = 60_262_615_040


def _mixed_role_deployment() -> ClusterDeployment:
    """A laptop someone is working on, plus a Studio on a shelf."""

    return ClusterDeployment(
        deployment_id="cluster-roles",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("macbook", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("studio", "user@studio.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment(
                node_id="macbook",
                rank=0,
                start_layer=3,
                end_layer=8,
                layer_weight_bytes=_MACBOOK_STAGE_BYTES - 2 * GIB,
                fixed_weight_bytes=1 * GIB,
                reserve_bytes=32 * GIB,
                capacity_bytes=_MACBOOK_CEILING_BYTES,
                role="workstation",
                kv_cache_bytes=1 * GIB,
            ),
            PipelineAssignment(
                node_id="studio",
                rank=1,
                start_layer=0,
                end_layer=3,
                layer_weight_bytes=30 * GIB,
                fixed_weight_bytes=1 * GIB,
                reserve_bytes=8 * GIB,
                capacity_bytes=256 * GIB,
                role="headless",
            ),
        ),
        plan_hash="c" * 64,
    )


def _worker_argv(deployment: ClusterDeployment, tmp_path) -> list[str]:
    """The tail mlx.launch executes on each host, straight from the launcher."""

    argv = build_mlx_launch_argv(
        deployment,
        hostfile=(tmp_path / "hosts.json").resolve(),
        api_port=32100,
        collective_port=32120,
        python_executable="/opt/omlx/bin/python",
    )
    return argv[argv.index("--") + 1 :]


def _parsed_plan(deployment: ClusterDeployment, tmp_path):
    """Decode the plan exactly as a rank does: parser first, then --plan."""

    from omlx.cluster.deployment import decode_worker_contract
    from omlx.cluster.inference_worker import build_parser

    worker_argv = _worker_argv(deployment, tmp_path)
    # argv[0..2] are the interpreter, -m and the module; argparse sees the rest.
    args = build_parser().parse_args(worker_argv[3:])
    plan_hash, assignments, _profiles, _tp = decode_worker_contract(args.plan)
    return args, plan_hash, assignments


def test_prompt_cache_ssd_reaches_the_rank_and_scopes_its_directory(tmp_path):
    from omlx.cluster.inference_worker import _prompt_cache_ssd_dir

    deployment = _deployment()
    args, _plan_hash, _assignments = _parsed_plan(deployment, tmp_path)

    # On by default: the flag rides the one argv every host runs.
    assert "--prompt-cache-ssd" in _worker_argv(deployment, tmp_path)
    assert args.prompt_cache_ssd is True

    # The directory is scoped by deployment and rank so no two runs or ranks
    # read each other's snapshots.
    rank0 = _prompt_cache_ssd_dir(args, rank=0)
    rank1 = _prompt_cache_ssd_dir(args, rank=1)
    assert rank0 is not None and rank1 is not None
    assert rank0.endswith(f"{args.deployment_id}-rank-0")
    assert rank1.endswith(f"{args.deployment_id}-rank-1")
    assert rank0 != rank1


def test_prompt_cache_ssd_can_be_turned_off(tmp_path):
    from types import SimpleNamespace

    from omlx.cluster.inference_worker import _prompt_cache_ssd_dir

    off = SimpleNamespace(
        prompt_cache_ssd=False, state_dir=str(tmp_path), deployment_id="d"
    )
    assert _prompt_cache_ssd_dir(off, rank=0) is None


def test_the_node_role_reaches_the_rank_through_the_launched_argv(tmp_path):
    deployment = _mixed_role_deployment()

    args, plan_hash, assignments = _parsed_plan(deployment, tmp_path)

    assert plan_hash == deployment.plan_hash
    assert args.plan_hash == plan_hash
    # assignments[rank] is what run_worker indexes.
    assert assignments[0].role == "workstation"
    assert assignments[1].role == "headless"


def test_one_argv_carries_two_different_roles(tmp_path):
    """Why the flag is not the fix.

    mlx.launch runs this argv unchanged on every host, so ``--node-role`` could
    only ever say one thing for the whole cluster. The per-rank plan says two.
    """

    deployment = _mixed_role_deployment()

    args, _plan_hash, assignments = _parsed_plan(deployment, tmp_path)

    assert "--node-role" not in _worker_argv(deployment, tmp_path)
    assert args.node_role == ""  # the flag is an override, and nobody set it
    assert {item.role for item in assignments} == {"workstation", "headless"}


def test_the_launched_argv_resolves_a_different_budget_for_each_rank(tmp_path):
    """One argv in, two admission budgets out.

    The fractions themselves belong to ``node_role``; what has to hold here is
    that the guard resolves a *different*, lower one for the Mac someone is
    working on, from nothing but the argv the launcher produced.
    """

    from omlx.cluster.memory_guard import admission_budget
    from omlx.cluster.node_role import HEADLESS, WORKSTATION, role_for

    _args, _plan_hash, assignments = _parsed_plan(_mixed_role_deployment(), tmp_path)

    assert role_for(assignments[0].role) is WORKSTATION
    assert role_for(assignments[1].role) is HEADLESS
    assert WORKSTATION.admission_fraction < HEADLESS.admission_fraction

    # How many bytes each of those becomes is memory_guard's arithmetic, and
    # it is tested there. What this seam owes is that the two ranks resolve
    # different budgets at all, and that the Mac someone is using gets the
    # smaller one — from one argv, with no role passed by hand.
    workstation_budget = admission_budget(
        _MACBOOK_CEILING_BYTES, role=assignments[0].role
    )
    headless_budget = admission_budget(_MACBOOK_CEILING_BYTES, role=assignments[1].role)
    assert 0 < workstation_budget < headless_budget <= _MACBOOK_CEILING_BYTES


def test_the_stage_that_took_the_macbook_down_is_refused_off_the_launched_argv(
    tmp_path,
):
    """The measured incident, driven from the launcher's own output.

    56.1 GiB of stage peaks near 73.0 GiB while it loads — between what this
    Mac admits headless and what it admits with someone working on it. Nothing
    here supplies a role by hand; it comes out of the argv.
    """

    from omlx.cluster.memory_guard import (
        admission_budget,
        check_rank_fits,
        load_peak_bytes,
    )
    from omlx.exceptions import InsufficientMemoryError

    _args, _plan_hash, assignments = _parsed_plan(_mixed_role_deployment(), tmp_path)
    macbook = assignments[0]
    assert macbook.planned_weight_bytes == _MACBOOK_STAGE_BYTES

    # Stated rather than assumed: if the role fractions ever move so that this
    # stage sits on one side of both budgets, this line says so instead of
    # leaving a "DID NOT RAISE" to be puzzled over.
    peak = load_peak_bytes(macbook.planned_weight_bytes)
    assert (
        admission_budget(_MACBOOK_CEILING_BYTES, role="workstation")
        < peak
        <= admission_budget(_MACBOOK_CEILING_BYTES, role="headless")
    )

    with pytest.raises(InsufficientMemoryError, match="can admit"):
        check_rank_fits(
            macbook.planned_weight_bytes,
            rank=0,
            node_id=macbook.node_id,
            role=macbook.role,
            ceiling_bytes=_MACBOOK_CEILING_BYTES,
            charge_load_peak=True,
        )

    # Same stage, same Mac, headless: admitted. The role is the only difference.
    assert (
        check_rank_fits(
            macbook.planned_weight_bytes,
            rank=0,
            node_id=macbook.node_id,
            role="headless",
            ceiling_bytes=_MACBOOK_CEILING_BYTES,
            charge_load_peak=True,
        )
        == _MACBOOK_CEILING_BYTES
    )


def test_a_plan_with_no_role_still_launches_and_reads_as_headless(tmp_path):
    """Older plans, and callers that never chose: unchanged behaviour."""

    from omlx.cluster.node_role import HEADLESS, role_for

    _args, _plan_hash, assignments = _parsed_plan(_deployment(), tmp_path)

    assert [item.role for item in assignments] == ["", ""]
    assert role_for(assignments[0].role) is HEADLESS


def test_a_planned_workstation_reaches_the_rank_as_a_workstation(tmp_path):
    """The whole chain, with nothing built by hand.

    The planner produces the assignments, the deployment carries them, the
    launcher encodes them, the worker's parser reads the argv and the plan
    decoder rebuilds them. Every previous test in this area started halfway
    along that path, which is why a role that existed at one end and not the
    other stayed invisible.
    """

    from omlx.cluster.node_role import HEADLESS, WORKSTATION, role_for
    from omlx.cluster.planner import (
        ModelLayout,
        NodeBudget,
        plan_unequal_pipeline,
    )

    # A KV rate, so the plan reserves cache the way a real one does and the
    # weight this rank is charged for is not trivially equal to its weights.
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(1536 * 1024 * 1024,) * 40,
        kv_bytes_per_token_per_layer=16 * 1024,
    )
    plan = plan_unequal_pipeline(
        model,
        [
            NodeBudget(
                node_id="macbook",
                capacity_bytes=107 * GIB,
                reserve_bytes=32 * GIB,
                rank=0,
                role="workstation",
            ),
            NodeBudget(
                node_id="studio",
                capacity_bytes=256 * GIB,
                reserve_bytes=25 * GIB,
                rank=1,
                role="headless",
            ),
        ],
    )
    deployment = ClusterDeployment(
        deployment_id="planned-roles",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("macbook", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("studio", "user@studio.local", ("10.0.0.2",)),
        ),
        assignments=plan.assignments,
        plan_hash=plan.plan_hash,
    )

    _args, plan_hash, assignments = _parsed_plan(deployment, tmp_path)

    assert plan_hash == plan.plan_hash
    assert assignments[0].node_id == "macbook"
    assert role_for(assignments[0].role) is WORKSTATION
    assert role_for(assignments[1].role) is HEADLESS
    # The stage the guard will be charged for, still whole after the trip —
    # including the KV cache, which is part of planned_weight_bytes and was
    # arriving as zero.
    assert all(item.kv_cache_bytes > 0 for item in plan.assignments)
    assert [item.planned_weight_bytes for item in assignments] == [
        item.planned_weight_bytes for item in plan.assignments
    ]


def test_supervisor_reaps_remote_ranks_via_ssh_sigterm(monkeypatch, tmp_path):
    deployment = _deployment()
    supervisor = launch.DistributedJobSupervisor(
        deployment,
        preflight=False,
        stop_timeout=0.1,
    )
    calls = []

    def fake_ssh(ssh_target, command, **kwargs):
        calls.append((ssh_target, command))
        return subprocess.CompletedProcess(["ssh"], 0, stdout="", stderr="")

    monkeypatch.setattr(launch, "_run_cluster_ssh", fake_ssh)

    marker_dir = tmp_path / ".omlx" / "cluster" / "runtime"
    marker_dir.mkdir(parents=True)
    marker_file = marker_dir / "cluster-test-rank-1.json"

    victim_code = (
        "import sys, time\n"
        "sys.argv = ['omlx.cluster.inference_worker', '--deployment-id', 'cluster-test']\n"
        "time.sleep(30)\n"
    )
    victim = subprocess.Popen(
        [sys.executable, "-c", victim_code],
    )
    try:
        marker_file.write_text(
            json.dumps(
                {
                    "deployment_id": "cluster-test",
                    "plan_hash": deployment.plan_hash,
                    "rank": 1,
                    "pid": victim.pid,
                }
            )
        )
        supervisor.state_dir = str(marker_dir)

        supervisor._reap_remote_ranks()

        assert len(calls) == 1
        ssh_target, command = calls[0]
        assert "user@studio.local" in ssh_target
        assert "cluster-test-rank-1.json" in command

        # Run the script to verify actual SIGTERM termination
        script = command.split("python3 -c ", 1)[1]
        import shlex as _shlex

        script = _shlex.split(script)[0]
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        victim.wait(timeout=2.0)
        assert victim.poll() is not None
        assert marker_file.exists()  # kept as crash evidence for _runtime_failure_reason
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait()


def test_supervisor_reaps_remote_ranks_escalates_to_sigkill(monkeypatch, tmp_path):
    deployment = _deployment()
    supervisor = launch.DistributedJobSupervisor(
        deployment,
        preflight=False,
        stop_timeout=0.1,
    )
    calls = []

    def fake_ssh(ssh_target, command, **kwargs):
        calls.append((ssh_target, command))
        return subprocess.CompletedProcess(["ssh"], 0, stdout="", stderr="")

    monkeypatch.setattr(launch, "_run_cluster_ssh", fake_ssh)

    marker_dir = tmp_path / ".omlx" / "cluster" / "runtime"
    marker_dir.mkdir(parents=True)
    marker_file = marker_dir / "cluster-test-rank-1.json"

    # Process that ignores SIGTERM
    victim_code = (
        "import sys, signal, time\n"
        "sys.argv = ['omlx.cluster.inference_worker', '--deployment-id', 'cluster-test']\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(30)\n"
    )
    victim = subprocess.Popen(
        [sys.executable, "-c", victim_code],
    )
    try:
        marker_file.write_text(
            json.dumps(
                {
                    "deployment_id": "cluster-test",
                    "plan_hash": deployment.plan_hash,
                    "rank": 1,
                    "pid": victim.pid,
                }
            )
        )
        supervisor.state_dir = str(marker_dir)

        supervisor._reap_remote_ranks()

        assert len(calls) == 1
        ssh_target, command = calls[0]

        script = command.split("python3 -c ", 1)[1]
        import shlex as _shlex

        script = _shlex.split(script)[0]
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        victim.wait(timeout=5.0)
        assert victim.poll() is not None
        assert marker_file.exists()  # kept as crash evidence for _runtime_failure_reason
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait()


def test_supervisor_reap_rejects_pid_reuse_command_mismatch(monkeypatch, tmp_path):
    deployment = _deployment()
    supervisor = launch.DistributedJobSupervisor(
        deployment,
        preflight=False,
        stop_timeout=0.1,
    )
    calls = []

    def fake_ssh(ssh_target, command, **kwargs):
        calls.append((ssh_target, command))
        return subprocess.CompletedProcess(["ssh"], 0, stdout="", stderr="")

    monkeypatch.setattr(launch, "_run_cluster_ssh", fake_ssh)

    marker_dir = tmp_path / ".omlx" / "cluster" / "runtime"
    marker_dir.mkdir(parents=True)
    marker_file = marker_dir / "cluster-test-rank-1.json"

    # Innocent unrelated process
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        marker_file.write_text(
            json.dumps(
                {
                    "deployment_id": "cluster-test",
                    "plan_hash": deployment.plan_hash,
                    "rank": 1,
                    "pid": victim.pid,
                }
            )
        )
        supervisor.state_dir = str(marker_dir)

        supervisor._reap_remote_ranks()

        assert len(calls) == 1
        ssh_target, command = calls[0]

        script = command.split("python3 -c ", 1)[1]
        import shlex as _shlex

        script = _shlex.split(script)[0]
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        # Innocent process must NOT be killed
        assert victim.poll() is None
    finally:
        victim.kill()
        victim.wait()


def test_supervisor_reap_rejects_mismatched_deployment_or_plan(monkeypatch, tmp_path):
    deployment = _deployment()
    supervisor = launch.DistributedJobSupervisor(
        deployment,
        preflight=False,
        stop_timeout=0.1,
    )
    calls = []

    def fake_ssh(ssh_target, command, **kwargs):
        calls.append((ssh_target, command))
        return subprocess.CompletedProcess(["ssh"], 0, stdout="", stderr="")

    monkeypatch.setattr(launch, "_run_cluster_ssh", fake_ssh)

    marker_dir = tmp_path / ".omlx" / "cluster" / "runtime"
    marker_dir.mkdir(parents=True)
    marker_file = marker_dir / "cluster-test-rank-1.json"

    victim_code = (
        "import sys, time\n"
        "sys.argv = ['omlx.cluster.inference_worker', '--deployment-id', 'cluster-test']\n"
        "time.sleep(30)\n"
    )
    victim = subprocess.Popen(
        [sys.executable, "-c", victim_code],
    )
    try:
        # Marker with different deployment_id
        marker_file.write_text(
            json.dumps(
                {
                    "deployment_id": "wrong-deployment",
                    "plan_hash": deployment.plan_hash,
                    "rank": 1,
                    "pid": victim.pid,
                }
            )
        )
        supervisor.state_dir = str(marker_dir)

        supervisor._reap_remote_ranks()

        ssh_target, command = calls[0]
        script = command.split("python3 -c ", 1)[1]
        import shlex as _shlex

        script = _shlex.split(script)[0]
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Victim must NOT be killed due to mismatched deployment_id
        assert victim.poll() is None
    finally:
        victim.kill()
        victim.wait()
