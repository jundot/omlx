# SPDX-License-Identifier: Apache-2.0
"""Layers [first, last) of the fake family.

Each layer keeps a running number across chunks and emits one row per token,
so a split run equals a whole run only when boundary, state, prefix and host
inputs all reach the right stage. A layer also reads every row the layer
before it emitted, like a layer that attends over another layer's cache.
"""

import numpy as np


class FakeJob:
    def __init__(self, layers):
        self.acc = [0.0] * layers
        # Sum of the rows each layer emitted so far, the prefix included.
        self.rows = {}

    def add_rows(self, state):
        for name, (data, _) in state.items():
            layer, slot = name.split(".")
            if slot == "rows":
                self.rows[int(layer)] = self.rows.get(int(layer), 0.0) + float(
                    data.sum()
                )


class FakeStage:
    device = "fake GPU"

    def __init__(self, first, last):
        self.first, self.last = first, last

    def warm(self, tokens):
        return 0.01 * (self.last - self.first)

    def memory_bytes(self):
        return (self.last - self.first) << 30

    def peak_bytes(self):
        return self.memory_bytes()

    def drop_layer(self):
        self.last -= 1

    def add_layer(self):
        self.last += 1

    def new_job(self, layers):
        return FakeJob(layers)

    def trim(self, job, layers):
        del job.acc[layers:]

    def load_prefix(self, job, state):
        for name, (data, _) in state.items():
            layer, slot = name.split(".")
            if slot == "acc":
                job.acc[int(layer) - self.first] = float(data[0])
        job.add_rows(state)

    def load_upstream(self, job, arrays):
        job.add_rows({name[2:]: value for name, value in arrays.items()})

    def run(self, job, *, start, tokens, boundary, inputs, following, capture):
        if boundary is None:
            h = tokens.astype(np.float32)[None, :, None] * np.ones(4, np.float32)
            if inputs:
                h = h + inputs["rows"][0]
        else:
            h = boundary["h"][0]
        out, state = {}, {}
        for j in range(len(job.acc)):
            layer = self.first + j
            job.acc[j] += float(h.sum())
            h = h * 0.5 + (layer + 1) + np.float32(job.acc[j] * 1e-3)
            h = h + np.float32(job.rows.get(layer - 1, 0.0) * 1e-4)
            if inputs:
                # Every layer reads the host's rows, so each stage needs them.
                h = h + inputs["rows"][0] * np.float32(0.125)
            state[f"{layer}.acc"] = (np.array([job.acc[j]], np.float64), "float64")
            state[f"{layer}.rows"] = (h[:, :, :1].copy(), "float32")
            job.add_rows({f"{layer}.rows": state[f"{layer}.rows"]})
            wanted = capture and layer in capture["layers"]
            if wanted and start + h.shape[1] > capture["from"]:
                rows = h[:, max(0, capture["from"] - start) :]
                out[f"cap.{layer}"] = (rows.copy(), "float32")
        out["h"] = (h, "float32")
        return out, state


def load_stage(model_dir, first, last):
    return FakeStage(first, last)
