# SPDX-License-Identifier: Apache-2.0
"""One rank of the RDMA pipeline test, launched by mlx.launch with one stand-in link per edge."""

from __future__ import annotations

import importlib
import json
import os
import sys

import mlx.core as mx
from mlx_lm.models import qwen2
from rdma_loopback import PythonWordOps

from omlx.cluster.performance import ExecutionSettings
from omlx.cluster.rdma.mailbox import ServiceMailbox
from omlx.cluster.rdma.stage_plan import StageLink
from omlx.cluster.rdma.stage_transport import install_stage_links
from omlx.cluster.runtime_optimizations import install_runtime_optimizations

mlx_generate = importlib.import_module("mlx_lm.generate")
PROMPTS = [[3, 17, 42, 9, 128, 5, 77, 31], [11, 200, 31, 4]]
NEW_TOKENS = int(os.environ.get("RDMA_TEST_TOKENS", "12"))


def build() -> qwen2.Model:
    args = qwen2.ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=6,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=256,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    mx.random.seed(1234)
    model = qwen2.Model(args)
    mx.eval(model.parameters())
    return model


def decode(model: qwen2.Model) -> list[list[int]]:
    gen = mlx_generate.BatchGenerator(model, max_tokens=NEW_TOKENS, prefill_step_size=4)
    try:
        uids = gen.insert(PROMPTS, max_tokens=[NEW_TOKENS] * len(PROMPTS))
        out = {uid: [] for uid in uids}
        done: set[int] = set()
        while len(done) < len(uids):
            responses = gen.next_generated()
            if not responses:
                break
            for response in responses:
                out[response.uid].append(int(response.token))
                if response.finish_reason:
                    done.add(response.uid)
        return [out[uid] for uid in uids]
    finally:
        gen.close()


def main() -> int:
    group = mx.distributed.init(backend="ring", strict=True)
    rank = group.rank()
    reference = decode(build())
    # Edge r+1 -> r uses entry r: {"name", "socket", "mailbox"}.
    edges = json.loads(os.environ["RDMA_TEST_LINKS"])
    links = tuple(
        StageLink(index + 1, index, edge["name"], edge["socket"])
        for index, edge in enumerate(edges)
    )
    mailboxes = {edge["name"]: edge["mailbox"] for edge in edges}
    model = build()
    model.model.pipeline(group)
    with (
        install_stage_links(
            mx,
            group,
            links,
            rank=rank,
            ops_loader=lambda: (PythonWordOps(), ""),
            attach_service=lambda name, socket_path, ops: ServiceMailbox.attach(
                name, socket_path, ops, mailbox_path=mailboxes[name]
            ),
            timeout_s=60,
        ) as stage_links,
        install_runtime_optimizations(
            model,
            group,
            ExecutionSettings(prefill_step_size=4),
            batchable=True,
            token_relay=stage_links.relay,
        ) as optimizations,
    ):
        # Counts ring all-sums while decoding; tokens on the relay need none.
        ring_sums = []
        all_sum = mx.distributed.all_sum
        mx.distributed.all_sum = lambda *a, **k: ring_sums.append(1) or all_sum(*a, **k)
        try:
            tokens = decode(model)
        finally:
            mx.distributed.all_sum = all_sum
    print(
        json.dumps(
            {
                "rank": rank,
                "stage_links_active": stage_links.report["active"],
                "token_relay": stage_links.report["token_relay"],
                "ring_sums": len(ring_sums),
                "sampling_rank_only": optimizations["sampling_rank_only"]["active"],
                "prefill_overlap": optimizations["pipeline_prefill_overlap"]["active"],
                "matches": tokens == reference,
            }
        ),
        flush=True,
    )
    return 0 if tokens == reference else 4


if __name__ == "__main__":
    sys.exit(main())
