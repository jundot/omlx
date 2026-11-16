"""Load oQh checkpoints through the regular mlx-lm / mlx-vlm loaders.

oQh tensors carry no ``.scales``, so the stock loaders leave their modules as
plain ``Linear`` / ``SwitchLinear``. ``nn.Module.load_weights`` is wrapped once:
when the weights hold ``.oqh_signs`` tensors, the matching modules are swapped
for oQh modules before the weights load, then producer fusion is installed.
Weights without oQh tensors take the original path unchanged.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from ..custom_kernels.oqh import fast
from . import format as fmt
from . import runtime
from .modules import from_tensors

logger = logging.getLogger(__name__)

_SIGNS = ".oqh_signs"
_installed = False


def _swap_modules(model: nn.Module, weights: dict[str, mx.array]) -> int:
    prefixes = [k[: -len(_SIGNS)] for k in weights if k.endswith(_SIGNS)]
    if not prefixes:
        return 0
    leaves = dict(
        tree_flatten(model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module))
    )
    swapped = {}
    for prefix in prefixes:
        if prefix not in leaves:
            raise ValueError(f"oQh tensor {prefix} has no matching module")
        swapped[prefix] = from_tensors(prefix, weights)
    model.update_modules(tree_unflatten(list(swapped.items())))
    return len(swapped)


def install_load_hook() -> None:
    """Make load_weights understand oQh tensors (idempotent, process wide)."""
    global _installed
    if _installed:
        return
    original = nn.Module.load_weights

    def load_weights(self, file_or_weights, strict: bool = True):
        weights = file_or_weights
        if isinstance(weights, str):
            weights = list(mx.load(weights).items())
        as_dict = dict(weights) if not isinstance(weights, dict) else weights
        if not any(k.endswith(_SIGNS) for k in as_dict):
            return original(self, weights, strict=strict)
        count = _swap_modules(self, as_dict)
        result = original(self, list(as_dict.items()), strict=strict)
        layers = runtime.install(self)
        logger.info(
            "oQh: %d modules loaded, producer fusion in %d layers", count, layers
        )
        return result

    nn.Module.load_weights = load_weights
    _installed = True


def require_native(config: dict) -> None:
    """Refuse an oQh checkpoint up front when this build cannot serve it."""
    if not fmt.is_oqh_config(config):
        return
    block = config[fmt.CONFIG_KEY]
    version = block.get("version") if isinstance(block, dict) else None
    # ValueError: engines pass it through as a load refusal.
    if version != fmt.FORMAT_VERSION:
        raise ValueError(
            f"This checkpoint uses oQh format version {version}; this oMLX reads "
            f"version {fmt.FORMAT_VERSION}"
        )
    if not fast.has_native():
        raise ValueError(
            "This oQh checkpoint needs the oQh native kernels, which failed to load: "
            f"{fast.import_error()}"
        )
