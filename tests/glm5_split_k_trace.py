# SPDX-License-Identifier: Apache-2.0
"""Observe real split-K calls without replacing their inputs or returned objects."""

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np


class KernelTrace:
    def __init__(self, mx, reference):
        self.mx, self.reference = mx, reference
        self.calls, self.constructed = [], {}
        self.failures = []
        self.fault_events = []
        self.fault_observation = None
        self.fault_mutations = []
        self.state_checks = []

    def install(self, monkeypatch, split_k):
        original = self.mx.fast.metal_kernel
        # Ensure each test observes constructors rather than a prior test's cached wrappers.
        monkeypatch.setattr(split_k, "_kernels", None)
        monkeypatch.setattr(split_k, "_b_kernels", None)

        def factory(**definition):
            kernel = original(**definition)
            name = definition["name"]
            if not name.startswith("glm53_sparse_mla_decode_"):
                return kernel
            self.constructed[name] = {
                "source_sha256": hashlib.sha256(
                    definition["source"].encode()
                ).hexdigest(),
                "header_sha256": hashlib.sha256(
                    definition.get("header", "").encode()
                ).hexdigest(),
                "constructor_kwargs": {
                    k: self.plain(v)
                    for k, v in definition.items()
                    if k not in ("source", "header")
                },
            }

            def call(**arguments):
                result = kernel(**arguments)
                r, phase = self.reference, self.reference.active
                self.calls.append(
                    (
                        {
                            "kernel": name,
                            "cycle": phase.cycle,
                            "operation": phase.operation,
                            "layer": r.layer,
                            "mode": r.mode,
                            "reference_row": r.row,
                            "prefixes": phase.prefixes,
                            **self.constructed[name],
                        },
                        tuple(definition["input_names"]),
                        {
                            **arguments,
                            "template": tuple(arguments["template"]),
                            "grid": tuple(arguments["grid"]),
                            "threadgroup": tuple(arguments["threadgroup"]),
                            "inputs": [self.mx.array(v) for v in arguments["inputs"]],
                            "non_tensor_kwargs": {
                                k: self.plain(v)
                                for k, v in arguments.items()
                                if k != "inputs"
                            },
                        },
                        [self.mx.array(v) for v in result],
                    )
                )
                return result

            return call

        monkeypatch.setattr(self.mx.fast, "metal_kernel", factory)

    @staticmethod
    def plain(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {k: KernelTrace.plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [KernelTrace.plain(v) for v in value]
        return str(value)

    def describe(self, value):
        mx = self.mx
        if value.dtype in (mx.bfloat16, mx.float16, mx.uint16):
            bits, dtype = mx.uint16, np.uint16
        elif value.dtype in (mx.float32, mx.int32, mx.uint32):
            bits, dtype = mx.uint32, np.uint32
        else:
            assert value.dtype in (mx.bool_, mx.uint8)
            bits, dtype = mx.uint8, np.uint8
        raw = np.array(value.view(bits).reshape(-1).tolist(), dtype=dtype)
        strides = getattr(value, "strides", None)
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "reported_strides": None if strides is None else list(strides),
            "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "first_raw_words": raw[:8].tolist(),
        }

    def materialize(self):
        records = []
        for context, names, arguments, outputs in self.calls:
            inputs = dict(zip(names, arguments["inputs"]))
            dims = inputs["dims"].tolist()
            batched = context["kernel"].endswith("_batched")
            s = dims[0]
            b = len(context["prefixes"]) if batched else 1
            rows = range(b) if batched else [context["reference_row"]]
            logical = []
            for local, row in enumerate(rows):
                prefix = context["prefixes"][row]
                item = {"row": row, "accepted_prefix": prefix}
                if "Q" in inputs:
                    item["q"] = self.describe(
                        inputs["Q"][local : local + 1, :, :prefix]
                    )
                    item["selected_windows"] = (
                        inputs["win"][local, :prefix]
                        if batched
                        else inputs["win"][:prefix]
                    ).tolist()
                    before, left = (
                        inputs["rows"].tolist()[local][1:] if batched else (dims[4], 0)
                    )
                    item["logical_kv"] = self.describe(
                        inputs["KV"][
                            local : local + 1, :, left : left + before + prefix
                        ]
                    )
                    item["before"], item["left_padding"] = before, left
                item["outputs"] = [
                    self.describe(
                        value[local : local + 1, :, :prefix]
                        if "merge" in context["kernel"]
                        else value[local * s : local * s + prefix]
                    )
                    for value in outputs
                ]
                logical.append(item)
            records.append(
                {
                    **context,
                    "variant": "batch" if batched else "scalar",
                    "B": b,
                    "S": s,
                    "H": dict(arguments["template"])["H"],
                    "D": dict(arguments["template"])["D"],
                    "template": [(k, str(v)) for k, v in arguments["template"]],
                    "call_non_tensor_kwargs": arguments["non_tensor_kwargs"],
                    "grid": arguments["grid"],
                    "threadgroup": arguments["threadgroup"],
                    "dims": dims,
                    "physical_KV_batch_stride_from_dims": dims[6] if batched else None,
                    "inputs": {n: self.describe(v) for n, v in inputs.items()},
                    "scale": (
                        inputs["scale_in"].tolist() if "scale_in" in inputs else None
                    ),
                    "mask": (
                        inputs["attn_mask"].tolist() if "attn_mask" in inputs else None
                    ),
                    "return_shapes": [list(v.shape) for v in outputs],
                    "return_dtypes": [str(v.dtype) for v in outputs],
                    "logical_rows": logical,
                }
            )
        return records

    def finish(self, original_error=None):
        folder = os.environ.get("GLM40_TRACE_DIR")
        if folder is None:
            return
        # Read tensors only after the actual test computations; the wrapper returns originals.
        try:
            destination = Path(folder)
            destination.mkdir(parents=True, exist_ok=True)
            node = os.environ.get("PYTEST_CURRENT_TEST", "unknown").split(" (")[0]
            name = (
                hashlib.sha256(node.encode()).hexdigest()[:16]
                + "-"
                + uuid.uuid4().hex
                + ".json"
            )
            result = {
                "node": node,
                "fault_events": self.plain(self.fault_events),
                "fault_observation": self.plain(self.fault_observation),
                "fault_mutations": self.plain(self.fault_mutations),
                "state_checks": self.plain(self.state_checks),
                "test_exception": (
                    None
                    if original_error is None
                    else {
                        "type": type(original_error).__name__,
                        "message": str(original_error),
                    }
                ),
                "constructed": self.constructed,
                "calls": self.materialize(),
                "continuity_failures": self.failures,
                "identical_batch_scalar_shape_assumed": False,
                "identical_batch_scalar_kernel_assumed": False,
                "reported_stride_limit": "Unavailable array strides remain null; kernel dims are observed.",
            }
            kda = getattr(self, "kda_reference", None)
            if kda is not None:
                result["kda_alignment"] = {
                    "layers": list(kda.layers),
                    "captures": kda.captures,
                    "consumptions": kda.consumptions,
                    "records": self.plain(kda.records),
                    "input_guard": "uint16 exact accepted prefix",
                    "actual_batch_outputs_preserved": True,
                }
            with (destination / name).open("x") as stream:
                json.dump(result, stream, allow_nan=False)
        except Exception as exc:
            if original_error is None:
                raise
            original_error.add_note(
                f"Trace collection failed: {type(exc).__name__}: {exc}"
            )
