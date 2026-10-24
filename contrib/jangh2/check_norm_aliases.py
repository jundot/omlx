#!/usr/bin/env python3
"""Verify global mixer checkpoint key aliases without loading model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jangh2_bridge.metadata import canonical

GLOBAL_HC_NORM = "language_model.model.hyper_connection_mixer.hc_norm.weight"


def matching_raw_keys(keys) -> tuple[str, ...]:
    return tuple(k for k in keys if canonical(k) == GLOBAL_HC_NORM)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    args = ap.parse_args()
    index = json.loads((args.model.expanduser() / 'model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    found = matching_raw_keys(weight_map)
    print('Expected model parameter:', GLOBAL_HC_NORM)
    print('Matching checkpoint tensors:', list(found))
    if len(found) != 1:
        nearby = sorted(k for k in weight_map if 'hyper_connection_mixer' in k and not k.startswith('mtp.'))
        print('Non-MTP checkpoint hyper_connection_mixer keys:', nearby[:30])
        raise SystemExit('Expected exactly one global mixer norm alias; no weights were loaded.')
    print('PASS: the top-level HC norm has an unambiguous canonical name. No model weights loaded.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
