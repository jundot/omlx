#!/usr/bin/env python3
"""Start the existing oMLX server with reversible JANGH2 hooks.

Model scoping covers the loader and chat policy; the scheduler cache adapter
installed here is process-wide (see jangh2_bridge/scheduler_compat.py).
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from importlib.metadata import entry_points, version
from pathlib import Path
import sys

from jangh2_bridge.chat_policy import ChatPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Exact local JANGH2 checkpoint directory')
    parser.add_argument('--checkout', default=str(Path.home()/'src/omlx-pr364'))
    parser.add_argument('--backend', choices=('reference','reference-fast','native','metal'), default='reference')
    parser.add_argument('--chunk-size', type=int, default=None)
    parser.add_argument('--thinking', choices=('auto','on','off'), default=None,
                        help='Default only; explicit request/admin template kwargs take priority')
    parser.add_argument('--reasoning-effort', choices=('low','medium','xhigh'), default=None)
    parser.add_argument('omlx_args', nargs=argparse.REMAINDER, help='oMLX arguments, starting with serve')
    args = parser.parse_args()
    if not args.omlx_args or args.omlx_args[0] != 'serve':
        parser.error('Supply oMLX CLI arguments after the bridge options, starting with serve')
    if sys.prefix == sys.base_prefix:
        parser.error('Use the PR checkout .venv/bin/python, not a system Python')
    import omlx
    checkout = Path(args.checkout).expanduser().resolve()
    if not Path(omlx.__file__).resolve().is_relative_to(checkout/'omlx'):
        parser.error(f'oMLX imported from {omlx.__file__}, not {checkout}/omlx; use the checkout virtualenv')
    environment = ChatPolicy.from_environment()
    policy = ChatPolicy(args.thinking or environment.thinking,
                        args.reasoning_effort or environment.reasoning_effort)
    candidates = [ep for ep in entry_points(group='console_scripts') if ep.name == 'omlx']
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one oMLX console entry point, found {len(candidates)}')

    print('JANGH2 bridge Python:', sys.executable, flush=True)
    print('JANGH2 bridge oMLX:', omlx.__file__, flush=True)
    print('JANGH2 bridge versions:', {p:version(p) for p in ('omlx','mlx','mlx-vlm','transformers')}, flush=True)
    print('JANGH2 bridge backend:', args.backend, flush=True)
    print('JANGH2 chat defaults:', policy.thinking, policy.reasoning_effort or '(template effort)', flush=True)
    from jangh2_bridge.loader import install
    from jangh2_bridge.scheduler_compat import install as install_cache_compat
    from jangh2_bridge.chat_compat import install as install_chat_compat

    old_argv = sys.argv
    try:
        # Every successfully installed hook is undone on normal exit, exceptions,
        # Ctrl-C and failure of later hook installation or entry-point loading.
        with ExitStack() as cleanup:
            from jangh2_bridge.profiles import launch_profile
            perf=cleanup.enter_context(launch_profile(args.model))
            print('JANGH2 performance:',perf,flush=True)
            uninstall_cache_compat = install_cache_compat()
            cleanup.callback(uninstall_cache_compat)
            uninstall = install(args.model, backend=args.backend, chunk_size=args.chunk_size)
            cleanup.callback(uninstall)
            cleanup.callback(install_chat_compat(args.model, policy=policy))
            sys.argv = ['omlx', *args.omlx_args]
            result = candidates[0].load()()
            return result if isinstance(result, int) else 0
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    raise SystemExit(main())
