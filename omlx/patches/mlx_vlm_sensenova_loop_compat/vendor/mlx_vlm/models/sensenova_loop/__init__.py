from ..base import install_auto_processor_patch
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .config import ModelConfig, TextConfig, VisionConfig
from .sensenova_loop import LanguageModel, Model, VisionModel

install_auto_processor_patch("sensenova_loop", Qwen3VLProcessor)
