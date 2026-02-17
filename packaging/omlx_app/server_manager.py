"""Server process management for oMLX menubar app."""

import logging
import os
import signal
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Union

import requests

from .config import ServerConfig, get_log_path

logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class PortConflict:
    """Returned by start() when the port is already in use."""
    pid: Optional[int]
    is_omlx: bool


def get_bundled_python() -> str:
    """Get the path to the bundled Python executable."""
    exe = Path(sys.executable)

    # Normal case: running under bundled python directly.
    if exe.name == "python3":
        return str(exe)

    # Menubar launcher case: sys.executable may be .../Contents/MacOS/oMLX.
    candidate = exe.with_name("python3")
    if candidate.exists():
        return str(candidate)

    # Fallback to current executable if layout is unexpected.
    return sys.executable


class ServerManager:
    """Manages the oMLX server process lifecycle."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._status = ServerStatus.STOPPED
        self._error_message: Optional[str] = None
        self._log_file_handle = None
        self._status_callback: Optional[Callable[[ServerStatus], None]] = None
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()
        self._adopted = False

    @property
    def status(self) -> ServerStatus:
        return self._status

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    def set_status_callback(self, callback: Callable[[ServerStatus], None]) -> None:
        self._status_callback = callback

    def _update_status(self, status: ServerStatus, error: Optional[str] = None) -> None:
        self._status = status
        self._error_message = error
        if self._status_callback:
            try:
                self._status_callback(status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    def _get_health_url(self) -> str:
        return f"http://127.0.0.1:{self.config.port}/health"

    def get_api_url(self) -> str:
        return f"http://127.0.0.1:{self.config.port}"

    def check_health(self) -> bool:
        try:
            response = requests.get(self._get_health_url(), timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _health_check_loop(self) -> None:
        while not self._stop_health_check.is_set():
            if self._status == ServerStatus.RUNNING:
                if not self.check_health():
                    if self._adopted:
                        self._adopted = False
                        self._update_status(
                            ServerStatus.ERROR,
                            "External server stopped",
                        )
                    elif self._process and self._process.poll() is not None:
                        exit_code = self._process.returncode
                        self._update_status(
                            ServerStatus.ERROR,
                            f"Server exited with code {exit_code}",
                        )
                        self._process = None
            elif self._status == ServerStatus.STARTING:
                if self.check_health():
                    self._update_status(ServerStatus.RUNNING)
            self._stop_health_check.wait(5)

    def _is_port_in_use(self) -> bool:
        """Check if the configured port is already in use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(("127.0.0.1", self.config.port))
                return True
        except (ConnectionRefusedError, OSError):
            return False

    def _is_omlx_server(self) -> bool:
        """Check if the process on the port is an oMLX server."""
        try:
            resp = requests.get(self._get_health_url(), timeout=2)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _find_port_owner_pid(self) -> Optional[int]:
        """Find the PID of the process listening on the configured port."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{self.config.port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().splitlines()[0])
        except Exception as e:
            logger.debug(f"lsof failed: {e}")
        return None

    def _kill_external_server(self, pid: int) -> bool:
        """Kill an external process by PID (SIGTERM, then SIGKILL)."""
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait up to 5 seconds for process to exit
            import time
            for _ in range(50):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)  # Check if still alive
                except OSError:
                    return True  # Process exited
            # Still alive — force kill
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
            return True
        except OSError as e:
            logger.error(f"Failed to kill PID {pid}: {e}")
            return False

    def adopt(self) -> bool:
        """Adopt an externally-running oMLX server without owning the process."""
        if not self._is_omlx_server():
            return False

        self._adopted = True
        self._process = None
        self._update_status(ServerStatus.RUNNING)

        # Start health check loop to monitor the external server
        self._stop_health_check.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
        )
        self._health_check_thread.start()

        logger.info(f"Adopted external oMLX server on port {self.config.port}")
        return True

    def start(self) -> Union[bool, PortConflict]:
        if self._status in (ServerStatus.RUNNING, ServerStatus.STARTING):
            return False

        # Check for port conflict before launching
        if self._is_port_in_use():
            is_omlx = self._is_omlx_server()
            pid = self._find_port_owner_pid()
            return PortConflict(pid=pid, is_omlx=is_omlx)

        self._adopted = False
        args = self.config.build_serve_args()
        self._update_status(ServerStatus.STARTING)

        try:
            # Ensure base directory exists
            base = Path(self.config.base_path).expanduser()
            base.mkdir(parents=True, exist_ok=True)

            log_path = get_log_path()
            self._log_file_handle = open(log_path, "w")

            python_exe = get_bundled_python()
            cmd = [python_exe, "-m", "omlx.cli"] + args

            self._process = subprocess.Popen(
                cmd,
                stdout=self._log_file_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            self._stop_health_check.clear()
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
            )
            self._health_check_thread.start()

            logger.info(f"Started server (PID: {self._process.pid})")
            return True

        except Exception as e:
            self._update_status(ServerStatus.ERROR, str(e))
            logger.error(f"Failed to start server: {e}")
            return False

    def stop(self, timeout: float = 10.0) -> bool:
        if self._status not in (ServerStatus.RUNNING, ServerStatus.STARTING):
            return False

        # Adopted server: just stop monitoring, don't kill anything
        if self._adopted:
            self._stop_health_check.set()
            if self._health_check_thread:
                self._health_check_thread.join(timeout=2)
            self._adopted = False
            self._update_status(ServerStatus.STOPPED)
            return True

        if not self._process:
            self._update_status(ServerStatus.STOPPED)
            return True

        self._update_status(ServerStatus.STOPPING)

        try:
            self._stop_health_check.set()
            if self._health_check_thread:
                self._health_check_thread.join(timeout=2)

            pid = self._process.pid
            os.killpg(os.getpgid(pid), signal.SIGTERM)

            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                self._process.wait(timeout=5)

            self._process = None
            self._update_status(ServerStatus.STOPPED)

            if self._log_file_handle:
                self._log_file_handle.close()
                self._log_file_handle = None

            return True

        except Exception as e:
            self._update_status(ServerStatus.ERROR, str(e))
            return False

    def restart(self) -> Union[bool, PortConflict]:
        self.stop()
        import time
        time.sleep(1)
        return self.start()

    def is_running(self) -> bool:
        if self._adopted:
            return self._status == ServerStatus.RUNNING
        return self._process is not None and self._process.poll() is None

    def update_config(self, config: ServerConfig) -> None:
        self.config = config
