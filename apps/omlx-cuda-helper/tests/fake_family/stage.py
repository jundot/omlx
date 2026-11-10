# SPDX-License-Identifier: Apache-2.0
"""Layers [first, last) of the fake family.

Each layer keeps a running number across chunks and emits one row per token,
so a split run equals a whole run only when boundary, state, prefix and host
inputs all reach the right stage.
"""

import numpy as np


class FakeStage:
    device = "fake GPU"

    def __init__(self, first, last):
        self.first, self.last = first, last

    def warm(self, tokens):
        return 0.01 * (self.last - self.first)

    def memory_bytes(self):
        return (self.last - self.first) << 30

    def new_job(self, layers):
        return [0.0] * layers

    def load_prefix(self, job, state):
        for name, (data, _) in state.items():
            layer, slot = name.split(".")
            if slot == "acc":
                job[int(layer) - self.first] = float(data[0])

    def run(self, job, *, start, tokens, boundary, inputs, following, capture):
        if boundary is None:
            h = tokens.astype(np.float32)[None, :, None] * np.ones(4, np.float32)
            if inputs:
                h = h + inputs["rows"][0]
        else:
            h = boundary["h"][0]
        out, state = {}, {}
        for j in range(len(job)):
            layer = self.first + j
            job[j] += float(h.sum())
            h = h * 0.5 + (layer + 1) + np.float32(job[j] * 1e-3)
            if inputs:
                # Every layer reads the host's rows, so each stage needs them.
                h = h + inputs["rows"][0] * np.float32(0.125)
            state[f"{layer}.acc"] = (np.array([job[j]], np.float64), "float64")
            state[f"{layer}.rows"] = (h[:, :, :1].copy(), "float32")
            wanted = capture and layer in capture["layers"]
            if wanted and start + h.shape[1] > capture["from"]:
                rows = h[:, max(0, capture["from"] - start) :]
                out[f"cap.{layer}"] = (rows.copy(), "float32")
        out["h"] = (h, "float32")
        return out, state


def load_stage(model_dir, first, last):
    return FakeStage(first, last)
