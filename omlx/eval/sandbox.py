# SPDX-License-Identifier: Apache-2.0
"""Run generated benchmark code in a resource-limited Python subprocess.

The child sets its own limits. A ``preexec_fn`` child forked from the threaded
server can hang before exec while it holds the listening socket.
"""

from __future__ import annotations

_LAUNCHER = """\
import os, resource, sys
mem, cpu, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
for limit, value in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, cpu)):
    try:
        resource.setrlimit(limit, (value, value))
    except (ValueError, OSError):
        pass
sys.argv = [path]
sys.path[0] = os.path.dirname(path)
with open(path) as f:
    code = compile(f.read(), path, "exec")
exec(code, {"__name__": "__main__", "__file__": path, "__builtins__": __builtins__})
"""


def limited_python_command(
    script_path: str, memory_limit_bytes: int, cpu_limit_seconds: int
) -> list[str]:
    """Argv that runs ``script_path`` with RLIMIT_AS and RLIMIT_CPU applied."""
    return [
        "python3",
        "-c",
        _LAUNCHER,
        str(memory_limit_bytes),
        str(cpu_limit_seconds),
        script_path,
    ]
