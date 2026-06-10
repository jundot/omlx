# SPDX-License-Identifier: Apache-2.0
"""Video generation engine: job manager + subprocess worker.

The video engine runs mlx-gen (Wan2.2 text-to-video) in a subprocess worker
from its own venv, coordinated by VideoJobManager with a memory lease held
against the ProcessMemoryEnforcer ceiling. Design:
docs/video-generation-engine-spec.md.

Note: worker.py is NOT imported here -- it runs under the video venv python
and must stay importable without omlx on sys.path.
"""

from .manager import VideoJob, VideoJobManager

__all__ = ["VideoJob", "VideoJobManager"]
