# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
# omlx-cuda-helper for linux/amd64 (Linux, Docker Desktop on Windows) and
# linux/arm64 (DGX Spark). Build from this folder:
#   docker buildx build --platform linux/amd64 -t omlx-cuda-helper .
ARG CUDA=13.0.2

FROM nvidia/cuda:${CUDA}-cudnn-devel-ubuntu24.04 AS build
ARG TARGETARCH
# Empty: Ampere and newer on amd64 (the kernels need bf16 tensor cores), GB10
# on arm64.
ARG MLX_CUDA_ARCHITECTURES=""
# Parallel compile jobs (default: every core); each nvcc job can take GBs.
ARG CMAKE_BUILD_PARALLEL_LEVEL=""
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git g++ cmake ninja-build python3-dev python3-venv \
        libopenblas-dev liblapacke-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/helper \
    && /opt/helper/bin/pip install --no-cache-dir -U pip setuptools wheel
ENV PATH=/opt/helper/bin:$PATH
COPY mlx-build /src/mlx-build
RUN git clone --quiet --filter=blob:none https://github.com/ml-explore/mlx /src/mlx \
    && /src/mlx-build/apply_patches.sh /src/mlx
RUN archs="$MLX_CUDA_ARCHITECTURES"; \
    if [ -z "$archs" ]; then \
        if [ "$TARGETARCH" = arm64 ]; then archs=121; else archs="80;86;90;120"; fi; \
    fi; \
    MLX_CUDA_ARCHITECTURES="$archs" PREFIX=/usr \
        CUDNN_LIBRARY_PATH="/usr/lib/$(uname -m)-linux-gnu" \
        /src/mlx-build/build_wheel.sh /src/mlx /wheels
COPY pyproject.toml README.md /src/helper/
COPY omlx_cuda_helper /src/helper/omlx_cuda_helper
RUN pip install --no-cache-dir /wheels/mlx-*.whl /src/helper

FROM nvidia/cuda:${CUDA}-cudnn-runtime-ubuntu24.04
# MLX compiles its kernels at run time against the CUDA and CCCL headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 iproute2 libopenblas0 liblapacke \
        cuda-cudart-dev-13-0 cuda-crt-13-0 cuda-cccl-13-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/helper /opt/helper
ENV PATH=/opt/helper/bin:$PATH CUDA_HOME=/usr/local/cuda
# Identity, pairing keys, model files and the kernel cache.
VOLUME /data
EXPOSE 8850 8851 8852
ENTRYPOINT ["omlx-cuda-helper", "--state-dir", "/data", "serve"]
