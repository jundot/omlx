from ....models.inkling.config import TextConfig
from .config import InklingMTPConfig as ModelConfig
from .inkling_mtp import InklingMTPDraftModel
from .inkling_mtp import InklingMTPDraftModel as Model

__all__ = [
    "Model",
    "ModelConfig",
    "TextConfig",
    "InklingMTPDraftModel",
]
