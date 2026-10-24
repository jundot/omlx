"""Packed JANGH2 experts and strict checkpoint loading for the VLM engine.

Modules are imported lazily so format discovery does not initialize MLX.
The runtime implementation is shared with the standalone qualification tools.
"""

from collections.abc import Mapping

MTP_UNSUPPORTED_REASON = "JANGH2 packed-expert MTP is not supported"


def is_jangh2_config(config: Mapping) -> bool:
    """Recognize declared packed-expert formats, including malformed metadata.

    Recognition is deliberately separate from strict validation so incomplete
    JANGH2 declarations fail in the format loader rather than falling through
    to an ordinary affine quantizer.
    """
    if "jangtq" in config:
        return True
    declared = config.get("jang_config")
    if isinstance(declared, Mapping) and declared.get("format") == "jangtq2":
        return True
    for key in ("quantization", "quantization_config"):
        quantization = config.get(key)
        if isinstance(quantization, Mapping) and any(
            isinstance(value, Mapping) and value.get("mode") == "jangtq2"
            for value in quantization.values()
        ):
            return True
    return False
