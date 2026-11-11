#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Check out the pinned MLX commit in MLX_SRC and apply the patch series.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX_SRC="${1:?usage: apply_patches.sh MLX_SRC}"
COMMIT="$(tr -d '[:space:]' < "$HERE/MLX_COMMIT")"

git -C "$MLX_SRC" checkout -q --force --detach "$COMMIT"
git -C "$MLX_SRC" clean -q -fd -- mlx python
while read -r patch; do
  [[ -z "$patch" || "$patch" == \#* ]] && continue
  echo "== apply $patch"
  git -C "$MLX_SRC" apply -p1 "$HERE/patches/$patch"
done < "$HERE/series"

# setup.py puts the short hash in the wheel version, and MLX keys its kernel
# cache on that version, so a changed series gets a fresh cache.
git -C "$MLX_SRC" add -A
git -C "$MLX_SRC" -c user.name=omlx-cuda-helper -c user.email=build@localhost \
  commit -q -m "omlx-cuda-helper patch series" --allow-empty
git -C "$MLX_SRC" log -1 --format='patched tree: %h'
