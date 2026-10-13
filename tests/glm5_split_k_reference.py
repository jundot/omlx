# SPDX-License-Identifier: Apache-2.0
"""Restrict a shared-path reference to one phase's accepted KV projections."""

from contextlib import contextmanager
from dataclasses import dataclass


class ReferenceRejectedError(AssertionError):
    """The reference's inputs or lifecycle do not match its batch capture."""


def require(condition, reason):
    if not condition:
        raise ReferenceRejectedError(reason)


@dataclass(frozen=True)
class Phase:
    cycle: int
    operation: str
    width: int
    prefixes: tuple


class ProjectionReference:
    def __init__(self, adapter, targets, plan):
        self.adapter = adapter
        self.targets = {id(owner): layer for owner, layer in targets}
        self.layers = tuple(layer for _, layer in targets)
        require(len(self.targets) == len(self.layers), "duplicate projection owner")
        require(self.layers == tuple(sorted(set(self.layers))), "duplicate layer")
        self.plan = tuple(plan)
        require(
            len({(p.cycle, p.operation) for p in self.plan}) == len(self.plan),
            "duplicate phase",
        )
        self.index = 0
        self.active = self.mode = self.layer = self.row = None
        self.slots = {}
        self.captured = self.consumed = []
        self.next_row = 0
        self.batch_tokens = None
        self.captures = self.consumptions = 0

    def begin(self, phase, tokens):
        require(self.active is None and self.index < len(self.plan), "capture reuse")
        require(phase == self.plan[self.index], "cycle or phase mismatch")
        require(
            phase.width > 0
            and phase.prefixes
            and all(type(n) is int and 1 <= n <= phase.width for n in phase.prefixes),
            "invalid accepted prefix",
        )
        require(
            self.adapter.tokens(tokens)
            and tokens.shape == (len(phase.prefixes), phase.width),
            "batch token shape or dtype mismatch",
        )
        self.active, self.mode = phase, "batch"
        self.batch_tokens = self.adapter.snapshot_tokens(tokens)
        self.slots, self.captured, self.consumed = {}, [], []
        self.next_row = 0

    @contextmanager
    def scope(self, layer):
        require(
            self.active is not None and self.mode in ("batch", "single"),
            "inactive projection scope",
        )
        require(self.layer is None and layer in self.layers, "wrong layer scope")
        self.layer = layer
        try:
            yield
        finally:
            self.layer = None

    def project(self, owner, value, original):
        if id(owner) not in self.targets:
            return original(owner, value)
        layer = self.targets[id(owner)]
        require(
            self.active is not None and self.layer == layer, "projection layer mismatch"
        )
        require(value.ndim == 3 and self.adapter.bf16(value), "projection must be BF16")
        if self.mode == "batch":
            require(
                len(self.captured) < len(self.layers)
                and layer == self.layers[len(self.captured)],
                "batch layer order or reuse",
            )
            require(
                value.shape[:2] == (len(self.active.prefixes), self.active.width),
                "batch projection shape mismatch",
            )
            output = original(owner, value)
            require(
                output.ndim == 3
                and output.shape[:2] == value.shape[:2]
                and self.adapter.bf16(output),
                "projection output mismatch",
            )
            for row, prefix in enumerate(self.active.prefixes):
                # Rejected query rows are never stored in the reference.
                self.slots[layer, row] = (
                    self.adapter.snapshot_bf16(value[row : row + 1, :prefix]),
                    self.adapter.snapshot_bf16(output[row : row + 1, :prefix]),
                )
            self.captured.append(layer)
            self.captures += 1
            return output
        require(self.mode == "single" and self.row is not None, "reference not active")
        require(
            len(self.consumed) < len(self.layers)
            and layer == self.layers[len(self.consumed)],
            "reference layer order or reuse",
        )
        require(
            value.shape[:2] == (1, self.active.prefixes[self.row]),
            "rejected tail exposure",
        )
        key = (layer, self.row)
        require(key in self.slots, "reference capture reuse")
        source, output = self.slots[key]
        require(
            source.shape == value.shape and self.adapter.same_bits(source, value),
            "uint16 projection input mismatch",
        )
        del self.slots[key]
        self.consumed.append(layer)
        self.consumptions += 1
        return output

    def seal(self):
        require(
            self.mode == "batch"
            and self.layer is None
            and tuple(self.captured) == self.layers,
            "incomplete batch capture",
        )
        self.mode = "sealed"

    def begin_single(self, phase, row, tokens):
        require(
            self.mode == "sealed" and phase == self.active, "reference phase mismatch"
        )
        require(
            type(row) is int
            and row == self.next_row
            and row < len(self.active.prefixes),
            "row order or reuse",
        )
        prefix = self.active.prefixes[row]
        require(
            self.adapter.tokens(tokens)
            and tokens.shape == (1, prefix)
            and self.adapter.same_tokens(
                tokens, self.batch_tokens[row : row + 1, :prefix]
            ),
            "reference token mismatch or rejected tail",
        )
        self.mode, self.row, self.consumed = "single", row, []

    def finish_single(self):
        require(
            self.mode == "single"
            and self.layer is None
            and tuple(self.consumed) == self.layers,
            "incomplete reference capture",
        )
        self.next_row += 1
        self.mode, self.row = "sealed", None

    def finish(self):
        require(
            self.mode == "sealed"
            and self.next_row == len(self.active.prefixes)
            and not self.slots
            and self.layer is None,
            "unconsumed reference capture",
        )
        self.active = self.mode = self.batch_tokens = None
        self.index += 1

    def complete(self):
        require(
            self.active is None and self.index == len(self.plan),
            "incomplete phase plan",
        )


class MLXBits:
    def __init__(self, mx):
        self.mx = mx

    def bf16(self, value):
        return value.dtype == self.mx.bfloat16

    def tokens(self, value):
        return value.dtype == self.mx.uint32

    def same_bits(self, left, right):
        return bool(
            self.mx.all(left.view(self.mx.uint16) == right.view(self.mx.uint16)).item()
        )

    def same_tokens(self, left, right):
        return bool(self.mx.array_equal(left, right).item())

    def snapshot_bf16(self, value):
        return self.mx.array(
            value.view(self.mx.uint16).tolist(), dtype=self.mx.uint16
        ).view(self.mx.bfloat16)

    def snapshot_tokens(self, value):
        return self.mx.array(value.tolist(), dtype=self.mx.uint32)


class FusedProjectionReference(ProjectionReference):
    """Same-input reference for KDA q/k/v/f_a/g_a/b fused projection outputs."""

    PARTS = ("q", "k", "v", "f_a", "g_a", "b")

    def __init__(self, adapter, targets, plan):
        super().__init__(adapter, targets, plan)
        self.widths = {}
        self.records = []

    def project_fused(self, owner, value, original):
        require(id(owner) in self.targets, "unknown KDA projection owner")
        actual = None

        def emit(target, inputs):
            nonlocal actual
            actual = original(target, inputs)
            require(
                isinstance(actual, (tuple, list)) and len(actual) == len(self.PARTS),
                "KDA fused projection parts mismatch",
            )
            require(
                all(
                    p.ndim == 3
                    and p.shape[:2] == inputs.shape[:2]
                    and self.adapter.bf16(p)
                    for p in actual
                ),
                "KDA fused projection output mismatch",
            )
            self.widths[self.targets[id(target)]] = tuple(p.shape[-1] for p in actual)
            return self.adapter.mx.concatenate(actual, axis=-1)

        projected = self.project(owner, value, emit)
        layer = self.targets[id(owner)]
        if self.mode == "batch":
            self.records.append(
                {
                    "cycle": self.active.cycle,
                    "operation": self.active.operation,
                    "layer": layer,
                    "mode": "batch",
                    "shape": list(value.shape),
                    "prefixes": list(self.active.prefixes),
                    "part_widths": list(self.widths[layer]),
                    "actual_batch_object_preserved": True,
                }
            )
            return actual
        require(layer in self.widths, "KDA fused layout missing")
        boundaries, total = [], 0
        for width in self.widths[layer][:-1]:
            total += width
            boundaries.append(total)
        self.records.append(
            {
                "cycle": self.active.cycle,
                "operation": self.active.operation,
                "layer": layer,
                "row": self.row,
                "mode": "single",
                "shape": list(value.shape),
                "uint16_input_identical": True,
                "accepted_prefix": self.active.prefixes[self.row],
            }
        )
        return self.adapter.mx.split(projected, boundaries, axis=-1)
