#!/usr/bin/env python3
"""Test exact device-row inserts, hits and wide IDs on oMLX's actual MLX worker.
No model weights are loaded. This diagnostic does NOT issue a qualification
report or enable an experimental profile. Run prepare.py afterwards.
"""
from __future__ import annotations
import argparse
from importlib import import_module


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    import mlx.core as mx
    from omlx.engine_core import get_mlx_executor
    from jangh2_bridge.scheduler_compat import install
    from jangh2_bridge.device_selftest import run
    if not mx.metal.is_available():
        raise RuntimeError('Actual Apple Metal is required for this check')
    generation = import_module('mlx_lm.generate')
    undo = install()
    try:
        def worker():
            mx.set_default_device(mx.gpu)
            with mx.stream(generation.generation_stream):
                result = run(mx)
                mx.synchronize(generation.generation_stream)
                return result
        results = get_mlx_executor().submit(worker).result()
        print(f'PASS: {len(results)} device-cache cases on the oMLX Metal worker.', flush=True)
        print('PASS: no model weights loaded. Full qualification is still required.', flush=True)
    finally:
        undo()


if __name__ == '__main__':
    main()
