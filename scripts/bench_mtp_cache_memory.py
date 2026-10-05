# SPDX-License-Identifier: Apache-2.0
"""Short, weight-free MTP allocation probes; not a model/quality benchmark.

Run in separate processes with PYTHONPATH pointing at each source revision.
Activation uses the real MTP initialization and sampler with a synthetic
backbone that appends KV only. No server settings or context limits change.
"""

import argparse
import gc
import json
import time
from types import SimpleNamespace

import mlx.core as mx
from mlx_vlm.models.cache import BatchKVCache, KVCache

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp import prompt_priming
from omlx.patches.vlm_batch_kv_capacity import apply_batch_kv_capacity_patch
from omlx.utils.proc_memory import get_phys_footprint
from omlx.utils.sampling import make_sampler

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("mode", choices=["activation", "replace", "pending-hidden"])
args = p.parse_args()
apply_batch_kv_capacity_patch()
mx.set_memory_limit(8 * 1024**3)
mx.set_cache_limit(4 * 1024**3)
mx.random.seed(917)
rows, layers, tokens = 6, 4, 8192


def sample():
    mx.synchronize()
    return dict(
        active=mx.get_active_memory(),
        pool=mx.get_cache_memory(),
        peak_active=mx.get_peak_memory(),
        phys_footprint=get_phys_footprint(),
    )


class Head:
    _omlx_mtp_chain = True
    _omlx_mtp_depth = 4
    _omlx_mtp_depth_fixed = True
    _omlx_mtp_head_hidden_normed = True
    _language_model = SimpleNamespace(_omlx_mtp_batch_rollback=True)

    def make_mtp_cache(self):
        return [KVCache()]

    def mtp_forward(self, hidden, inputs, cache, return_hidden=False, **kwargs):
        values = inputs[:, None, :, None].astype(mx.float32)
        cache[0].update_and_fetch(values, values)
        logits = mx.broadcast_to(inputs[..., None] * 0.0, (*inputs.shape, 256))
        return (logits, hidden) if return_hidden else logits


forward_shapes = []


def forward(model, inputs, cache, **kwargs):
    forward_shapes.append(list(inputs.shape))
    for layer in cache:
        x = mx.broadcast_to(
            inputs[:, None, :, None], (inputs.shape[0], 4, inputs.shape[1], 256)
        ).astype(mx.bfloat16)
        layer.update_and_fetch(x, x * 2)
        mx.eval(layer.state)
    return mx.zeros((*inputs.shape, 256)), mx.zeros((*inputs.shape, 16)), None, None


if args.mode == "pending-hidden":
    # Exercise the actual capture path with a wide prefill chunk, then drop
    # all callers' references. Only the final hidden row should remain live.
    host = SimpleNamespace()
    ctx = prompt_priming._PrimeCtx(expected_offset=0, deferred_pairs=[])
    setattr(host, prompt_priming._CTX_ATTR, ctx)
    hidden = mx.ones((1, 8192, 4096))
    inputs = mx.ones((1, 8192), mx.uint32)
    anchor = KVCache()
    anchor.offset = 8192
    mx.eval(hidden, inputs)
    mx.clear_cache()
    mx.reset_peak_memory()
    before = sample()
    start = time.monotonic()
    prompt_priming._capture_deferred_history(host, inputs, hidden, [anchor])
    # A completed head fold no longer needs its paired history.
    ctx.deferred_pairs.clear()
    mx.eval(ctx.pending_hidden)
    del hidden, inputs
    gc.collect()
    assert mx.array_equal(ctx.pending_hidden, mx.ones((1, 1, 4096)))
else:
    caches = [BatchKVCache([0] * rows) for _ in range(layers)]
    for c in caches:
        x = mx.ones((rows, 4, tokens, 256), mx.bfloat16)
        c.update_and_fetch(x, x * 2)
        mx.eval(c.state)
    del c, x
    sampler = make_sampler(temp=1, top_k=20, top_p=0.95)
    batch = SimpleNamespace(
        model=Head(),
        uids=list(range(rows)),
        prompt_cache=caches,
        tokens=[[1]] * rows,
        samplers=[sampler] * rows,
        fallback_sampler=sampler,
        logits_processors=[],
        _next_tokens=mx.arange(10, 10 + rows, dtype=mx.uint32),
        _next_logprobs=[mx.zeros((256,)) for _ in range(rows)],
        _token_context=[None] * rows,
        _num_tokens=[0] * rows,
        _matchers=[None] * rows,
    )
    del caches
    batch.extract_cache = lambda i: [c.extract(i) for c in batch.prompt_cache]
    bg._call_backbone_captured = forward
    mx.eval(batch._next_tokens, batch._next_logprobs)
    mx.clear_cache()
    mx.reset_peak_memory()
    before = sample()
    start = time.monotonic()
    if args.mode == "activation":
        state = bg._prepare_mtp_batch_state_for_next(batch)
        assert state is not None and len(state.states) == rows
        for s in state.states.values():
            mx.eval(s.drafts, [c.state for c in s.mtp_cache])
    else:
        replacements = {i: batch.extract_cache(i) for i in (1, 3, 5)}
        for i, cache in replacements.items():
            forward(batch.model, batch._next_tokens[i : i + 1, None], cache)
        del cache
        bg._replace_cache_rows(batch, replacements)
        del replacements
    mx.eval([c.state for c in batch.prompt_cache])
    for c in batch.prompt_cache:
        expected = [
            tokens + int(args.mode == "activation" or i in (1, 3, 5))
            for i in range(rows)
        ]
        assert c.offset.tolist() == expected
        for i in range(rows):
            value = 10 + i if expected[i] > tokens else 1
            assert mx.all(c.keys[i, :, c._idx - 1, :] == value)
    del c

after = sample()
print(
    json.dumps(
        dict(
            mode=args.mode,
            device=mx.device_info()["device_name"],
            source=bg.__file__,
            rows=rows,
            layers=layers,
            tokens=tokens,
            before=before,
            after=after,
            elapsed_seconds=time.monotonic() - start,
            forward_shapes=forward_shapes,
            correct=True,
        ),
        indent=2,
    )
)
