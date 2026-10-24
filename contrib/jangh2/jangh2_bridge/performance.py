"""Explicit performance knobs. No OS limits, package or model edits."""
from __future__ import annotations
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class PerformanceOptions:
    gate_rows: int = 4
    down_rows: int = 4
    sort_threshold: int = 64
    ple_cache_mib: int = 0
    ple_open_files: int = 8
    prefill: str = "legacy"
    prefill_min_tokens: int = 64
    prefill_tile: int = 4
    ple_device_mib: int = 0

    @classmethod
    def from_environment(cls, *, fast_defaults: bool = False):
        def integer(name, default, lo, hi, allowed=None):
            raw=os.environ.get(name,str(default))
            try: value=int(raw)
            except ValueError as exc: raise ValueError(f'{name} must be an integer, got {raw!r}') from exc
            if not lo <= value <= hi or (allowed is not None and value not in allowed):
                raise ValueError(f'Invalid {name}={raw!r}; allowed {allowed or (lo,hi)}')
            return value
        prefill=os.environ.get('OMLX_JANGH_PREFILL','legacy')
        if prefill not in ('legacy','grouped'):
            raise ValueError('OMLX_JANGH_PREFILL must be legacy or grouped')
        return cls(
            integer('OMLX_JANGH_METAL_GATE_ROWS',4,2,8,(2,4,8)),
            integer('OMLX_JANGH_METAL_DOWN_ROWS',4,2,8,(2,4,8)),
            integer('OMLX_JANGH_METAL_SORT_THRESHOLD',64,1,65536),
            integer('OMLX_JANGH_PLE_CACHE_MB',16 if fast_defaults else 0,0,256),
            integer('OMLX_JANGH_PLE_OPEN_FILES',16 if fast_defaults else 8,1,64),
            prefill,
            integer('OMLX_JANGH_PREFILL_MIN_TOKENS',64,2,4096),
            integer('OMLX_JANGH_PREFILL_TILE',4,2,4,(2,4)),
            integer('OMLX_JANGH_PLE_DEVICE_MB',0,0,256))
