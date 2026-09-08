# SPDX-License-Identifier: Apache-2.0
#
# Lightweight image for exercising the oMLX admin dashboard (Usage & Cost
# card, cloud-pricing endpoints) on Linux without Apple Silicon/Metal and
# without any model weights. Not used for production inference — the real
# deployment target is macOS with mlx-metal; this image swaps in the
# mlx[cpu] backend purely so `omlx.server:app` can import and boot.
#
# Build:  podman build -t omlx-admin-test .
# Run:    podman run --rm -p 8000:8000 -v omlx-data:/data omlx-admin-test
# Then visit http://localhost:8000/admin to set up the initial admin key
# (no OMLX_API_KEY is baked in — see the README section in PROMPT.md/report
# for the first-run setup flow).

FROM python:3.12-slim AS builder

WORKDIR /build

# git: fetches the git+https pinned deps (mlx-lm, mlx-embeddings, mlx-vlm,
# dflash-mlx). build-essential: satisfies any sdist transitive deps that
# need a compiler; setup.py itself skips build_ext entirely because
# OMLX_WITH_CUSTOM_KERNEL is left unset below (custom Metal kernels are
# macOS-only).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --prefix=/install . && \
    # Base "mlx==0.32.2" only auto-pulls its Metal backend on Darwin (see
    # mlx's own platform-conditional Requires-Dist); Linux needs the "cpu"
    # extra's companion wheel explicitly or `import mlx.core` fails with
    # "libmlx.so: cannot open shared object file".
    pip install --no-cache-dir --prefix=/install "mlx-cpu==0.32.2"


FROM python:3.12-slim

RUN useradd --create-home --uid 1000 omlx

COPY --from=builder /install /usr/local

# Base directory for settings.json, cloud_pricing.json, the usage ledger,
# and (if ever pointed at real weights) models. Owned by the non-root user
# so the anonymous/named volume mounted at container start inherits
# writable permissions instead of root:root.
RUN mkdir -p /data && chown omlx:omlx /data

ENV OMLX_BASE_PATH=/data \
    PYTHONUNBUFFERED=1

VOLUME /data
# Default port from omlx.settings.ServerSettings.port.
EXPOSE 8000

USER omlx

ENTRYPOINT ["omlx", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
