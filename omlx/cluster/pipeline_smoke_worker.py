# SPDX-License-Identifier: Apache-2.0
"""Tiny multi-rank Nemotron-H graph used by the pipeline diagnostic."""

from __future__ import annotations

import json
import sys

from .pipeline_compat import install_pipeline_compatibility
from .planner import PipelineAssignment


def _smoke_assignments(world_size: int) -> tuple[PipelineAssignment, ...]:
    """Give every rank two layers in the pipeline's reverse execution order."""

    if not 2 <= world_size <= 16:
        raise ValueError("pipeline smoke requires between 2 and 16 ranks")
    layer_count = world_size * 2
    return tuple(
        PipelineAssignment(
            f"rank-{rank}",
            rank,
            layer_count - (rank + 1) * 2,
            layer_count - rank * 2,
            2,
            0,
            0,
            layer_count,
        )
        for rank in range(world_size)
    )


def main() -> int:
    try:
        from omlx._torch_stub import install as install_torch_stub

        install_torch_stub()

        import mlx.core as mx
        from mlx.utils import tree_map
        from mlx_lm.models.nemotron_h import Model, ModelArgs

        group = mx.distributed.init(backend="ring", strict=True)
        assignments = _smoke_assignments(group.size())

        # Each rank gets two layers and one real cache-bearing layer. This
        # catches both activation transport and Nemotron-H's sparse cache index
        # mapping without downloading any model weights.
        layer_count = group.size() * 2
        pattern = (["M", "-", "*", "-"] * ((layer_count + 3) // 4))[
            :layer_count
        ]
        args = ModelArgs(
            model_type="nemotron_h",
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=layer_count,
            max_position_embeddings=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_bias=False,
            mamba_num_heads=2,
            mamba_head_dim=4,
            mamba_proj_bias=False,
            ssm_state_size=4,
            conv_kernel=2,
            n_groups=1,
            mlp_bias=False,
            layer_norm_epsilon=1e-5,
            use_bias=False,
            use_conv_bias=False,
            hybrid_override_pattern=pattern,
        )
        with install_pipeline_compatibility(assignments):
            model = Model(args)
            # MLX's Metal and CUDA random generators are not bit-identical.
            # Giving each rank a random output head makes a healthy mixed
            # pipeline appear divergent even though it transported the same
            # hidden state.  Constant parameters keep this a transport/model
            # execution assertion rather than an RNG conformance test.
            model.update(
                tree_map(
                    lambda value: mx.full(
                        value.shape,
                        0.01,
                        dtype=value.dtype,
                    ),
                    model.parameters(),
                )
            )
            model.model.pipeline(group)
            cache = model.make_cache()
            output = model(mx.array([[1, 2, 3]]), cache=cache)
            mx.eval(output)
            checksum = mx.sum(output.astype(mx.float32))
            checksums = mx.distributed.all_gather(checksum)
            mx.eval(checksums)
            values = [float(value) for value in checksums.tolist()]
            if max(values) - min(values) > 1e-4:
                raise RuntimeError(f"rank outputs diverged: {values}")
            record = {
                "type": "pipeline_result",
                "model_type": "nemotron_h",
                "rank": group.rank(),
                "size": group.size(),
                "start_layer": assignments[group.rank()].start_layer,
                "end_layer": assignments[group.rank()].end_layer,
                "local_layer_count": len(model.layers),
                "local_cache_count": len(cache),
                "output_shape": list(output.shape),
                "checksum": values[group.rank()],
            }
            print(json.dumps(record, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            f"oMLX pipeline smoke worker failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
