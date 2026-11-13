# SPDX-License-Identifier: Apache-2.0
"""Run generated benchmark code in a resource-limited Python subprocess.

The limits are applied by the child itself through a small launcher instead of
``subprocess``'s ``preexec_fn``. ``preexec_fn`` runs Python code between fork
and exec, which is unsafe in a multi-threaded process like the server: the
forked child can deadlock on a lock held by another thread at fork time. Such a
child never execs, ignores SIGTERM, leaves ``subprocess.run`` blocked forever
(its timeout never starts), and still holds every inherited file descriptor,
including the server's listening socket, because ``close_fds`` only runs after
``preexec_fn``. Once the server exits, the orphan keeps the port bound and
every restart fails with "Address already in use".
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
