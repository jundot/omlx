from ..base import install_auto_processor_patch
from ..muse_glimmer.processing_muse_glimmer import (
    MuseGlimmerImageProcessor,
    MuseGlimmerProcessor,
)
from .config import ModelConfig


class MuseSparkProcessor(MuseGlimmerProcessor):
    """Image and text processor for Muse Spark.

    Muse Spark uses the Muse Glimmer image processor and the same
    ``<|image_start|><|patch|>...<|image_end|>`` expansion. Video and audio
    inputs are not supported in oMLX.
    """

    def __call__(self, images=None, text=None, videos=None, audio=None, **kwargs):
        if videos is not None or audio is not None:
            raise ValueError("Muse Spark video and audio inputs are not supported.")
        return super().__call__(images=images, text=text, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import PreTrainedConfig

        config = kwargs.pop("config", None)
        if config is None:
            config_dict, _ = PreTrainedConfig.get_config_dict(
                pretrained_model_name_or_path
            )
            config = ModelConfig.from_dict(config_dict)
        elif isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        return super().from_pretrained(
            pretrained_model_name_or_path, config=config, **kwargs
        )


__all__ = ["MuseGlimmerImageProcessor", "MuseSparkProcessor"]

install_auto_processor_patch("muse_spark", MuseSparkProcessor)
