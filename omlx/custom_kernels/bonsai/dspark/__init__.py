# SPDX-License-Identifier: Apache-2.0
"""Bonsai DSpark speculative-decoding drafter support.

Public API
----------
BonsaiDSparkConfig     Config dataclass (extends mlx_dspark's DSparkConfig with
                       Bonsai-specific fields: log_snr conditioning, bonsai family).
BonsaiDSparkDrafter    MLX nn.Module — the 6-layer cross-attention drafter with
                       optional log-SNR conditioning.
convert_gguf           GGUF → safetensors converter for the Bonsai DSpark checkpoint.
BonsaiTarget           Hidden-state tap wrapper for Qwen3.5 hybrid VLM target.
"""

from .config import BonsaiDSparkConfig
from .convert import convert_gguf
from .drafter import BonsaiDSparkDrafter
from .target import BonsaiTarget

__all__ = [
    "BonsaiDSparkConfig",
    "BonsaiDSparkDrafter",
    "convert_gguf",
    "BonsaiTarget",
]
