from ..base import install_auto_processor_patch
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .qwen4_exp import Model
from .vision import VisionModel

install_auto_processor_patch("qwen4_exp", Qwen3VLProcessor)

__all__ = [
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
]
