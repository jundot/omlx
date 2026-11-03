# SPDX-License-Identifier: Apache-2.0
"""Family adapters by model type."""

from ..adapter import StageAdapter
from .deepseek_v41 import DeepseekV41Adapter

_ADAPTERS: dict[str, StageAdapter] = {
    model_type: adapter
    for adapter in (DeepseekV41Adapter(),)
    for model_type in adapter.model_types
}


def adapter_for(model_type: str | None) -> StageAdapter | None:
    return _ADAPTERS.get(model_type or "")
