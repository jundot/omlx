# SPDX-License-Identifier: Apache-2.0
"""D2: prove tensor parallelism produces the SAME tokens as a single node.

Sharding bugs do not crash. A wrong sharding mode drops an all-reduce, an
unsplit head count reshapes attention, a mis-sharded MoE routes to the wrong
expert — and every one of those still emits fluent text. Shape assertions and
unit tests cannot see it. The only proof is running the same prompt, greedily,
on one node and on N nodes and comparing the token ids.

Run the reference on one machine::

    .venv/bin/python benchmarks/tp_identity_probe.py --model <path> --tokens 40

Then the distributed run, from the coordinator::

    .venv/bin/mlx.launch --hostfile hostfile.json --backend ring \\
        -- .venv/bin/python benchmarks/tp_identity_probe.py \\
        --model <path> --tokens 40 \\
        --tensor-parallel-size 2

Rank 0 prints a JSON line with the token ids. Identical ids means the split is
correct; anything else means it is not, however plausible the text looks.
"""

from __future__ import annotations

import argparse
import json
import time

from omlx.cluster.tensor_strategies import apply_tensor_strategy


def _greedy_token_ids(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    """Greedy decode, returning raw token ids.

    Ids rather than text: detokenisation can hide a divergence that only shows
    up in whitespace or a merged token.
    """

    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    encoded = tokenizer.encode(prompt)
    prompt_array = mx.array(encoded)
    sampler = make_sampler(temp=0.0)  # temp 0 == argmax == reproducible

    ids: list[int] = []
    for token, _logprobs in generate_step(
        prompt_array, model, max_tokens=max_tokens, sampler=sampler
    ):
        ids.append(int(token))
        if len(ids) >= max_tokens:
            break
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument(
        "--prompt",
        default="List the first five prime numbers and explain why one is not prime.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="1 runs single-process and produces the reference ids",
    )
    args = parser.parse_args()

    from omlx._torch_stub import install as install_torch_stub

    install_torch_stub()

    import mlx.core as mx
    from mlx_lm import load

    rank, world = 0, 1
    tp_group = None
    if args.tensor_parallel_size > 1:
        group = mx.distributed.init(strict=True)
        rank, world = group.rank(), group.size()
        if world != args.tensor_parallel_size:
            raise SystemExit(
                f"world size {world} != tensor_parallel_size "
                f"{args.tensor_parallel_size}; this probe uses one pipeline stage"
            )
        # This probe deliberately uses one pipeline stage, so the global group
        # is exactly the tensor-parallel group used by the worker.
        tp_group = group

    model, tokenizer = load(args.model)
    layer_count = len(model.model.layers)

    if tp_group is not None:
        # One pipeline stage: this rank holds every layer, sharded across the
        # tensor-parallel group. Exactly the path the cluster worker takes.
        apply_tensor_strategy(model, tp_group, mx_module=mx)
        mx.eval(model.parameters())

    started = time.perf_counter()
    ids = _greedy_token_ids(model, tokenizer, args.prompt, args.tokens)
    elapsed = time.perf_counter() - started

    if rank == 0:
        print(
            json.dumps(
                {
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "world_size": world,
                    "layers": layer_count,
                    "token_ids": ids,
                    "text": tokenizer.decode(ids),
                    "tokens_per_second": round(len(ids) / elapsed, 2),
                    "seconds": round(elapsed, 3),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
