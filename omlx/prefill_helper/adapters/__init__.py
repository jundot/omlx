# SPDX-License-Identifier: Apache-2.0
"""Family adapters by model type."""

from ..adapter import StageAdapter
from .deepseek_v41 import DeepseekV41Adapter
from .glm5_next import Glm5NextAdapter
from .inkling2 import Inkling2Adapter
from .minimax_m3 import MiniMaxM3Adapter
from .qwen3_5 import Qwen35Adapter
from .qwen4_exp import Qwen4ExpAdapter

_ADAPTERS: dict[str, StageAdapter] = {
    model_type: adapter
    for adapter in (
        DeepseekV41Adapter(),
        Glm5NextAdapter(),
        Inkling2Adapter(),
        MiniMaxM3Adapter(),
        Qwen35Adapter(),
        Qwen4ExpAdapter(),
    )
    for model_type in adapter.model_types
}


def adapter_for(model_type: str | None) -> StageAdapter | None:
    return _ADAPTERS.get(model_type or "")


def adapter_for_model(
    model_type: str | None, engine_type: str | None, model_path: str | None
) -> StageAdapter | None:
    """The adapter of a model this server knows, or None when the model gets
    no helper. ``model_type`` is the raw model type of its config.json."""
    adapter = adapter_for((model_type or "").replace("-", "_").lower())
    if adapter is None or not adapter.serves(engine_type, model_path):
        return None
    return adapter
