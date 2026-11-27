import math
from dataclasses import dataclass
from typing import Optional

from ..qwen3_5.config import ModelConfig as Qwen3_5ModelConfig
from ..qwen3_5.config import TextConfig as Qwen3_5TextConfig
from ..qwen3_5.config import VisionConfig
from ..qwen3_vl.config import _config_kwargs, _maybe_deserialize_config


@dataclass
class TextConfig(Qwen3_5TextConfig):
    layer_types: Optional[list[str]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.layer_types is None:
            self.layer_types = [
                (
                    "full_attention"
                    if (i + 1) % self.full_attention_interval == 0
                    else "linear_attention"
                )
                for i in range(self.num_hidden_layers)
            ]


def execution_plan(config):
    """Validate the checkpoint's fixed loop schedule and mHC phase boundaries."""
    loop = config.loop_settings
    if not isinstance(loop, dict):
        raise ValueError("SenseNova Loop requires explicit loop_settings")
    text = config.text_config
    n = text.num_hidden_layers
    start, end, count = (loop.get(k) for k in ("start", "end", "eval_count"))
    if any(type(v) is not int for v in (n, start, end, count)) or not (
        0 <= start < end <= n and count >= 1
    ):
        raise ValueError("Invalid Loop interval or eval_count")
    if loop.get("counts") != [count]:
        raise ValueError("Only fixed-count Loop checkpoints are supported")
    if len(text.layer_types) != n or set(text.layer_types) - {
        "full_attention",
        "linear_attention",
    }:
        raise ValueError("Unexpected physical layer types")
    if "full_attention" not in text.layer_types:
        raise ValueError("At least one full-attention layer is required")

    mhc = config.mhc
    if mhc is not None:
        if not isinstance(mhc, dict):
            raise ValueError("MHC settings must be a dictionary")
        ms, me, streams, iterations = (
            mhc.get(k) for k in ("start", "end", "streams", "iterations")
        )
        if (
            any(type(v) is not int for v in (ms, me, streams, iterations))
            or not 0 <= ms < me <= n
            or streams not in (2, 4)
            or iterations < 1
        ):
            raise ValueError("Invalid MHC range, stream count, or iterations")
        if type(mhc.get("fixed_identity")) is not bool:
            raise ValueError("MHC fixed_identity must be a boolean")
        scale = mhc.get("pre_scale", 1.0)
        if (
            not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("Invalid MHC pre_scale")
        if set(mhc) - {
            "start",
            "end",
            "streams",
            "iterations",
            "fixed_identity",
            "pre_scale",
        }:
            raise ValueError("Unknown MHC configuration fields")
        boundaries = (0, start, end, n)
        if not any(a <= ms < me <= b for a, b in zip(boundaries, boundaries[1:])):
            raise ValueError("MHC cannot cross a Loop phase boundary")
    return tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, n))


@dataclass
class ModelConfig(Qwen3_5ModelConfig):
    loop_settings: Optional[dict] = None
    mhc: Optional[dict] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params["vision_config"] = _maybe_deserialize_config(
            VisionConfig, params.get("vision_config")
        )
        params["text_config"] = _maybe_deserialize_config(
            TextConfig, params.get("text_config"), require_all_fields=True
        )
        return cls(**_config_kwargs(cls, params))
