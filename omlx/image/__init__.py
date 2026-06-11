# SPDX-License-Identifier: Apache-2.0
"""Image generation engine (POST /v1/images).

Deliberately does NOT import worker.py: the worker runs under the media
venv (mlx-gen) and must stay importable without omlx on sys.path.
Job management lives in omlx.video.manager (MediaJobManager dispatches
both media kinds against the single enforcer lease).
"""
