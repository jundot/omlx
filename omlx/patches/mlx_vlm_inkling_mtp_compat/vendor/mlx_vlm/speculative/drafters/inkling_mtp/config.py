import inspect
from dataclasses import dataclass
from typing import List, Optional

from ....models.base import BaseModelConfig
from ....models.inkling.config import TextConfig


@dataclass
class InklingMTPConfig(BaseModelConfig):
    model_type: str = "inkling_mtp"
    text_config: Optional[TextConfig] = None
    num_mtp_layers: int = 1
    mtp_local_layer_ids: Optional[List[int]] = None
    block_size: int = 3
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            raw = dict(self.text_config)
            depth = raw.get("num_mtp_layers") or raw.get("num_nextn_predict_layers")
            if depth:
                self.num_mtp_layers = int(depth)
            local = raw.get("mtp_local_layer_ids")
            if local is None:
                local = raw.get("local_layer_ids")
            self.mtp_local_layer_ids = local
            self.text_config = TextConfig.from_dict(raw)
        if self.text_config is not None:
            self.tie_word_embeddings = bool(
                getattr(self.text_config, "tie_word_embeddings", False)
            )

    @classmethod
    def from_dict(cls, params: dict) -> "InklingMTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        depth = (
            text_config.get("num_mtp_layers")
            or text_config.get("num_nextn_predict_layers")
            or 1
        )
        flat.setdefault("block_size", int(depth) + 2)
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
