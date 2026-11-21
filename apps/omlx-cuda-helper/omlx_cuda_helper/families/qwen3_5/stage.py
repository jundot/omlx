# SPDX-License-Identifier: Apache-2.0
"""One GPU stage of Qwen3.5: layers ``[first, last)``.

The boundary of every cut is the hidden state ``h`` [1, L, hidden], with the
host's rotary frequencies ``rope`` when the job runs an attention layer on the
helper. A linear layer's state is its conv window and recurrent state, sent
whole; an attention layer's state is the K/V rows the chunk appended.
"""

from __future__ import annotations

from pathlib import Path

from .._qwen35_stage import Stage as _Stage
from .layers import Layer


class Stage(_Stage):
    """The shared Qwen3.5 stage over this family's dense layers."""

    def __init__(self, model_dir: str | Path, first: int, last: int):
        super().__init__(model_dir, first, last, Layer)


def load_stage(model_dir: str | Path, first: int, last: int) -> Stage:
    return Stage(model_dir, first, last)
