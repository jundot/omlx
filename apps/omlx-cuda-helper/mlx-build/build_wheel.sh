#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the patched MLX CUDA wheel from MLX_SRC into OUT_DIR.
# Needs nvcc, cuDNN, cmake, ninja and a C++ compiler on PATH (see environment.txt).
set -euo pipefail

MLX_SRC="${1:?usage: build_wheel.sh MLX_SRC OUT_DIR}"
OUT_DIR="${2:?usage: build_wheel.sh MLX_SRC OUT_DIR}"
# Set explicitly: the MLX default asks nvcc to query a live GPU.
ARCHS="${MLX_CUDA_ARCHITECTURES:-86}"
PREFIX="${CONDA_PREFIX:-/usr}"

command -v nvcc >/dev/null || { echo "error: nvcc not on PATH" >&2; exit 1; }
mkdir -p "$OUT_DIR"
cd "$MLX_SRC"

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
export CMAKE_ARGS="-DMLX_BUILD_CUDA=ON -DMLX_CUDA_ARCHITECTURES=$ARCHS \
-DMLX_BUILD_GGUF=OFF -DCMAKE_PREFIX_PATH=$PREFIX \
-DCUDNN_INCLUDE_PATH=$PREFIX/include -DCUDNN_LIBRARY_PATH=$PREFIX/lib \
-DLAPACK_INCLUDE_DIRS=$PREFIX/include -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON"
if [[ -n "${CXX:-}" ]]; then
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_HOST_COMPILER=$CXX"
fi

python -m pip wheel . --no-deps -w "$OUT_DIR"
ls -l "$OUT_DIR"/mlx-*.whl
