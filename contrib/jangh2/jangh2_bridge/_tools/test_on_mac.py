#!/usr/bin/env python3
"""Validate actual MLX execution before opening the large checkpoint."""
import argparse
import logging
import sys
from jangh2_bridge.selftest import run

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('reference','reference-fast','native','metal'),default='reference')
    parser.add_argument('--small-only',action='store_true',help='Skip the 2560x640 fixture; not recommended for acceptance')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        run(args.backend,full_dimensions=not args.small_only)
    except Exception:
        logging.exception('MLX self-test failed. Do not start the full model with this backend.')
        sys.exit(1)
