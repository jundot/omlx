# SPDX-License-Identifier: Apache-2.0
"""Generate one 1024-square image from the validated native local bundle."""

import argparse
import json
from pathlib import Path

from omlx.image.pipeline import generate_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    image, metrics = generate_png(
        args.bundle,
        args.prompt,
        args.seed,
        progress=lambda phase, step, total: print(
            f"{phase} {step}/{total}", flush=True
        ),
    )
    with args.output.open("xb") as stream:
        stream.write(image)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
