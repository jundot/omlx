# SPDX-License-Identifier: Apache-2.0
"""Tests for managed DS4 subprocess scaffolding."""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ds4_support_fixtures import PINNED_DS4_METAL_FILES
import omlx.ds4_process as ds4_process
from omlx.ds4_process import (
    DS4_HOST,
    DS4LaunchConfig,
    DS4ManagedProcess,
    DS4ProcessError,
    prune_ds4_kv_cache,
    safe_ds4_fs_name,
)
from omlx.ds4_support import DS4_SERVER_BINARY, DS4SupportError
from omlx.settings import DS4Settings


def _write_support_tree(root: Path, script: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    binary = root / DS4_SERVER_BINARY
    binary.write_text(script)
    binary.chmod(0o755)
    (root / "LICENSE").write_text("MIT\n")
    (root / "README.md").write_text("DS4\n")
    metal_dir = root / "metal"
    metal_dir.mkdir()
    for name in PINNED_DS4_METAL_FILES:
        (metal_dir / name).write_text("// metal\n")
    return binary


def _ready_server_script() -> str:
    return f"""#!{sys.executable}
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument('--chdir')
parser.add_argument('--model')
parser.add_argument('--host')
parser.add_argument('--port', type=int)
parser.add_argument('--power')
parser.add_argument('--ctx')
parser.add_argument('--kv-disk-dir')
parser.add_argument('--kv-disk-space-mb')
parser.add_argument('--kv-cache-continued-interval-tokens')
parser.add_argument('--ssd-streaming', action='store_true')
parser.add_argument('--mtp')
parser.add_argument('--mtp-draft')
parser.add_argument('--mtp-margin')
parser.add_argument('--dspark', action='store_true')
parser.add_argument('--trace')
args, _ = parser.parse_known_args()
print('fake ds4 argv model=' + str(args.model), flush=True)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/v1/models':
            body = json.dumps({{'object': 'list', 'data': [{{'id': 'fake'}}]}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *args):
        return

HTTPServer((args.host, args.port), Handler).serve_forever()
"""


def _never_ready_script() -> str:
    return f"""#!{sys.executable}
import sys
import time
if '--help' in sys.argv:
    print('--ssd-streaming --mtp --dspark')
    raise SystemExit(0)
print('fake ds4 never ready', flush=True)
time.sleep(60)
"""


class TestDS4KVPruning:
    """Tests for global DS4 disk KV budget enforcement."""

    def test_prune_ds4_kv_cache_deletes_oldest_kv_files_only(self, tmp_path):
        root = tmp_path / "kv"
        old_dir = root / "old-model"
        new_dir = root / "new-model"
        old_dir.mkdir(parents=True)
        new_dir.mkdir(parents=True)
        old_kv = old_dir / "old.kv"
        new_kv = new_dir / "new.kv"
        non_kv = old_dir / "notes.txt"
        old_kv.write_bytes(b"a" * 10)
        new_kv.write_bytes(b"b" * 10)
        non_kv.write_bytes(b"c" * 100)
        old_mtime = 1_700_000_000
        new_mtime = old_mtime + 100
        os.utime(old_kv, (old_mtime, old_mtime))
        os.utime(new_kv, (new_mtime, new_mtime))

        result = prune_ds4_kv_cache(root, max_bytes=15)

        assert result.bytes_before == 20
        assert result.files_before == 2
        assert result.deleted_files == (old_kv.resolve(),)
        assert result.deleted_bytes == 10
        assert result.bytes_after == 10
        assert result.files_after == 1
        assert not old_kv.exists()
        assert new_kv.exists()
        assert non_kv.exists()

    def test_prepare_directories_no_longer_prunes_global_kv_root(self, tmp_path):
        """Launch-time KV prune is removed — ds4-server handles its own eviction."""
        kv_root = tmp_path / "kv"
        old_dir = kv_root / "old-model"
        new_dir = kv_root / "new-model"
        old_dir.mkdir(parents=True)
        new_dir.mkdir(parents=True)
        old_kv = old_dir / "old.kv"
        new_kv = new_dir / "new.kv"
        old_kv.write_bytes(b"a" * 800_000)
        new_kv.write_bytes(b"b" * 800_000)
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        settings = DS4Settings(
            kv_root=str(kv_root),
            kv_disk_space_mb=1,
            debug_dir=str(tmp_path / "debug"),
        )
        managed = DS4ManagedProcess(
            DS4LaunchConfig(
                model_id="current/model",
                gguf_path=gguf,
                settings=settings,
                base_path=tmp_path,
            )
        )

        managed._prepare_directories()

        assert (kv_root / "current-model").is_dir()
        # DS4 launch-time pruning is removed — ds4-server evicts its own
        # directory at startup using its score-based policy.
        assert old_kv.exists()
        assert new_kv.exists()
        assert managed.last_kv_prune_result is None


class TestDS4LaunchConfig:
    """Tests for DS4 launch command construction."""

    def test_safe_ds4_fs_name(self):
        """Model ids become filesystem-safe per-model artifact names."""
        assert safe_ds4_fs_name("DeepSeek/V4:Flash") == "deepseek-v4-flash"
        assert safe_ds4_fs_name(" ... ") == "ds4-model"

    def test_build_command_includes_ds4_flags(self, tmp_path):
        """Command construction preserves the DS4 performance-path flags."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _ready_server_script())
        gguf = tmp_path / "models" / "DeepSeek.gguf"
        gguf.parent.mkdir()
        gguf.write_bytes(b"gguf")
        settings = DS4Settings(
            support_dir=str(support),
            context_default_tokens=100_000,
            kv_root=str(tmp_path / "kv"),
            kv_disk_space_mb=1234,
            kv_cache_continued_interval_tokens=4096,
            ssd_streaming="auto",
            power=77,
            trace_enabled=True,
            trace_dir=str(tmp_path / "traces"),
        )
        config = DS4LaunchConfig(
            model_id="DeepSeek/V4:Flash",
            gguf_path=gguf,
            settings=settings,
            base_path=tmp_path,
            port=12345,
            auto_enable_ssd_streaming=True,
            trace_timestamp="20260101-010203",
            platform_system="Darwin",
            platform_machine="arm64",
        )

        command = config.build_command(12345)

        assert command[0] == str(support / DS4_SERVER_BINARY)
        assert command[command.index("--chdir") + 1] == str(support.resolve())
        assert command[command.index("--model") + 1] == str(gguf)
        assert command[command.index("--host") + 1] == DS4_HOST
        assert command[command.index("--port") + 1] == "12345"
        assert "--metal" in command
        assert command[command.index("--ctx") + 1] == "100000"
        assert command[command.index("--kv-disk-space-mb") + 1] == "1234"
        assert (
            command[command.index("--kv-cache-continued-interval-tokens") + 1] == "4096"
        )
        assert command[command.index("--power") + 1] == "77"
        assert "--ssd-streaming" in command
        trace_arg = command[command.index("--trace") + 1]
        assert trace_arg.endswith("deepseek-v4-flash-20260101-010203.trace")
        assert "deepseek-v4-flash" in command[command.index("--kv-disk-dir") + 1]

    def test_launch_config_rejects_non_localhost_host(self, tmp_path):
        """Managed DS4 is never allowed to bind to LAN interfaces."""
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")

        with pytest.raises(ValueError, match="127.0.0.1"):
            DS4LaunchConfig(
                model_id="model",
                gguf_path=gguf,
                base_path=tmp_path,
                host="0.0.0.0",
            )

    def test_build_command_omits_metal_on_non_darwin(self, tmp_path):
        """Non-Darwin launchers do not receive the macOS Metal backend flag."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _ready_server_script())
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        config = DS4LaunchConfig(
            model_id="model",
            gguf_path=gguf,
            settings=DS4Settings(support_dir=str(support)),
            base_path=tmp_path,
            platform_system="Linux",
            platform_machine="aarch64",
        )

        assert "--metal" not in config.build_command(12345)

    def test_launch_config_is_frozen_after_localhost_validation(self, tmp_path):
        """The localhost-only invariant cannot be bypassed by mutation."""
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        config = DS4LaunchConfig(
            model_id="model",
            gguf_path=gguf,
            base_path=tmp_path,
        )

        with pytest.raises(FrozenInstanceError):
            config.host = "0.0.0.0"  # type: ignore[misc]

    def test_build_command_omits_kv_trace_and_ssd_when_disabled(self, tmp_path):
        """Disabled optional DS4 features are not passed to ds4-server."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _ready_server_script())
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        settings = DS4Settings(
            support_dir=str(support),
            kv_cache_enabled=False,
            ssd_streaming="off",
            trace_enabled=False,
        )
        config = DS4LaunchConfig(
            model_id="model",
            gguf_path=gguf,
            settings=settings,
            base_path=tmp_path,
            port=12345,
            platform_system="Darwin",
            platform_machine="arm64",
        )

        command = config.build_command(12345)

        assert "--kv-disk-dir" not in command
        assert "--kv-disk-space-mb" not in command
        assert "--ssd-streaming" not in command
        assert "--trace" not in command

    def test_build_command_adds_mtp_and_disables_ssd_streaming(self, tmp_path):
        """MTP sidecars are passed to DS4 and force SSD streaming off."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _ready_server_script())
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        mtp = tmp_path / "mtp.gguf"
        mtp.write_bytes(b"gguf")
        settings = DS4Settings(
            support_dir=str(support),
            ssd_streaming="on",
        )
        config = DS4LaunchConfig(
            model_id="model",
            gguf_path=gguf,
            settings=settings,
            base_path=tmp_path,
            port=12345,
            mtp_path=mtp,
            mtp_draft=2,
            mtp_margin=3.0,
            platform_system="Darwin",
            platform_machine="arm64",
        )

        command = config.build_command(12345)

        assert command[command.index("--mtp") + 1] == str(mtp)
        assert command[command.index("--mtp-draft") + 1] == "2"
        assert command[command.index("--mtp-margin") + 1] == "3.0"
        assert "--ssd-streaming" not in command

    def test_build_command_adds_dspark_and_keeps_ssd_streaming(self, tmp_path):
        """DSpark uses --mtp plus --dspark and permits main-model streaming."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _ready_server_script())
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        dspark = tmp_path / "dspark-support.gguf"
        dspark.write_bytes(b"gguf")
        config = DS4LaunchConfig(
            model_id="model",
            gguf_path=gguf,
            settings=DS4Settings(support_dir=str(support), ssd_streaming="on"),
            base_path=tmp_path,
            mtp_path=dspark,
            mtp_kind="dspark",
            mtp_draft=4,
            mtp_margin=1.5,
            platform_system="Darwin",
            platform_machine="arm64",
        )

        command = config.build_command(12345)

        assert command[command.index("--mtp") + 1] == str(dspark)
        assert "--dspark" in command
        assert "--mtp-draft" not in command
        assert "--mtp-margin" not in command
        assert "--ssd-streaming" in command


class TestDS4ManagedProcess:
    """Tests for managed DS4 subprocess lifecycle."""

    @pytest.mark.asyncio
    async def test_start_waits_for_models_readiness_and_captures_logs(
        self, tmp_path, caplog
    ):
        """A fake ds4-server is started, probed, and terminated cleanly."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _ready_server_script())
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        settings = DS4Settings(
            support_dir=str(support),
            kv_root=str(tmp_path / "kv"),
            debug_dir=str(tmp_path / "debug"),
            ready_timeout_ms=2_000,
        )
        managed = DS4ManagedProcess(
            DS4LaunchConfig(
                model_id="model",
                gguf_path=gguf,
                settings=settings,
                base_path=tmp_path,
                platform_system="Darwin",
                platform_machine="arm64",
            )
        )

        caplog.set_level(logging.INFO, logger="omlx.ds4_process")
        await managed.start()
        try:
            assert managed.is_running is True
            assert managed.port is not None
            assert managed.command is not None
            assert "--host" in managed.command
            assert (tmp_path / "kv" / "model").is_dir()
            assert (tmp_path / "debug" / "model").is_dir()
            assert any("fake ds4 argv" in line.text for line in managed.logs)
        finally:
            await managed.stop()

        assert managed.is_running is False
        assert managed.log_path == tmp_path / "debug" / "model" / "ds4.log"
        log_text = managed.log_path.read_text()
        assert "model_id: model" in log_text
        assert "stdout: fake ds4 argv" in log_text
        assert re.search(r"\[DS4-\d+\] Loading model: model", caplog.text)
        assert re.search(r"\[DS4-\d+\] stdout: fake ds4 argv", caplog.text)

    @pytest.mark.asyncio
    async def test_start_validates_support_off_event_loop(self, tmp_path, monkeypatch):
        """Lazy DS4 provisioning must not block the asyncio server loop."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _ready_server_script())
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        settings = DS4Settings(
            support_dir=str(support),
            ready_timeout_ms=2_000,
        )
        managed = DS4ManagedProcess(
            DS4LaunchConfig(
                model_id="model",
                gguf_path=gguf,
                settings=settings,
                base_path=tmp_path,
                platform_system="Darwin",
                platform_machine="arm64",
            )
        )
        event_loop_thread = threading.current_thread().ident
        support_threads: list[int | None] = []
        original_ensure = ds4_process.ensure_ds4_support

        def fake_ensure(*args, **kwargs):
            support_threads.append(threading.current_thread().ident)
            return original_ensure(*args, **kwargs)

        monkeypatch.setattr(ds4_process, "ensure_ds4_support", fake_ensure)

        await managed.start()
        try:
            assert support_threads
            assert all(thread_id != event_loop_thread for thread_id in support_threads)
        finally:
            await managed.stop()

    @pytest.mark.asyncio
    async def test_start_can_disable_ds4_debug_log_file(self, tmp_path, caplog):
        """DS4 logs can be regular-log/in-memory only without ds4-debug files."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _ready_server_script())
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        settings = DS4Settings(
            support_dir=str(support),
            debug_dir=str(tmp_path / "debug"),
            ready_timeout_ms=2_000,
            logs_to_disk=False,
        )
        managed = DS4ManagedProcess(
            DS4LaunchConfig(
                model_id="model",
                gguf_path=gguf,
                settings=settings,
                base_path=tmp_path,
                platform_system="Darwin",
                platform_machine="arm64",
            )
        )

        caplog.set_level(logging.INFO, logger="omlx.ds4_process")
        await managed.start()
        try:
            assert any("fake ds4 argv" in line.text for line in managed.logs)
        finally:
            await managed.stop()

        assert managed.log_path is None
        assert not (tmp_path / "debug" / "model" / "ds4.log").exists()
        assert re.search(r"\[DS4-\d+\] Loading model: model", caplog.text)
        assert re.search(r"\[DS4-\d+\] stdout: fake ds4 argv", caplog.text)

    @pytest.mark.asyncio
    async def test_start_timeout_stops_process_and_reports_logs(self, tmp_path):
        """Readiness timeout terminates the subprocess and includes logs."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _never_ready_script())
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        settings = DS4Settings(
            support_dir=str(support),
            ready_timeout_ms=1_500,
        )
        managed = DS4ManagedProcess(
            DS4LaunchConfig(
                model_id="model",
                gguf_path=gguf,
                settings=settings,
                base_path=tmp_path,
                platform_system="Darwin",
                platform_machine="arm64",
            )
        )

        with pytest.raises(DS4ProcessError) as exc_info:
            await managed.start()

        assert managed.is_running is False
        assert "timed out" in str(exc_info.value)
        assert "fake ds4 never ready" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_support_files_block_start(self, tmp_path):
        """Start fails before spawning when support validation fails."""
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        managed = DS4ManagedProcess(
            DS4LaunchConfig(
                model_id="model",
                gguf_path=gguf,
                settings=DS4Settings(support_dir=str(tmp_path / "missing")),
                base_path=tmp_path,
                platform_system="Darwin",
                platform_machine="arm64",
            )
        )

        with pytest.raises(DS4SupportError):
            await managed.start()

        assert managed.process is None
