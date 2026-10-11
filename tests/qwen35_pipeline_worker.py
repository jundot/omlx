# SPDX-License-Identifier: Apache-2.0
"""One rank of tests/test_qwen35_pipeline_forward.py (ring backend over loopback).

Builds a tiny Qwen3.5-style hybrid model (gated-delta-net + full attention) with
identical weights on every rank, records the stock mlx-lm forward BEFORE oMLX's MTP
patch is applied, then pipelines the same model and checks that the pipelined forward
(prefill chunks + single-token decode, with caches) reproduces it on every rank.
"""

import json
import sys

import mlx.core as mx


def main() -> int:
    group = mx.distributed.init(backend="ring", strict=True)
    from mlx_lm.models import qwen3_5

    mx.random.seed(7)
    text = dict(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=96,
        full_attention_interval=4,
        max_position_embeddings=512,
        mtp_num_hidden_layers=0,
    )
    args = qwen3_5.ModelArgs(model_type="qwen3_5", text_config=text)
    model = qwen3_5.Model(args)
    mx.eval(model.parameters())

    prompt = mx.array([[3, 9, 27, 81, 5, 15, 45, 7]])
    steps = [prompt[:, :5], prompt[:, 5:7], prompt[:, 7:8]]

    def run(m):
        cache = m.make_cache()
        outs = []
        for chunk in steps:
            out = m(chunk, cache=cache)
            mx.eval(out)
            outs.append(out)
        return outs

    reference = run(model)  # stock mlx-lm forward, single process, no patch yet

    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

    assert apply_mlx_lm_mtp_patch()
    patched_single = run(model)  # oMLX's replaced forward at world size 1
    for ref, got in zip(reference, patched_single):
        assert mx.allclose(ref, got, atol=1e-4).item(), "patched forward drifted"

    model.model.pipeline(group)
    assert model.model.pipeline_size == group.size()
    cache = model.make_cache()
    assert len(cache) == len(model.model.pipeline_layers)
    for ref, chunk in zip(reference, steps):
        out = model(chunk, cache=cache)
        mx.eval(out)
        assert mx.allclose(ref, out, atol=1e-4).item(), (
            f"rank {group.rank()}: pipelined logits differ, max abs "
            f"{mx.abs(ref - out).max().item()}"
        )
    print(json.dumps({"rank": group.rank(), "ok": True}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
