from ..base import install_auto_processor_patch
from ..muse_glimmer.processing_muse_glimmer import MuseGlimmerProcessor


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


__all__ = ["MuseSparkProcessor"]

install_auto_processor_patch("muse_spark", MuseSparkProcessor)
