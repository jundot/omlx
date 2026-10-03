# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced MTP activation audit, using an already loaded serving model.

Call replay(model, prompts, continuation, directory) on the inference thread in
separate before/after servers. Compare the resulting directories with this
script. No sampling or adaptive-depth policy is replaced in the serving path.
This is a numerical diagnostic, not a generation-quality or memory benchmark.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_flatten

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.utils.sampling import make_sampler


def snapshot(cache):
    """Record logical state, excluding unused capacity and padding."""
    arrays, metadata = {}, {}
    for i, layer in enumerate(cache):
        for key, value in tree_flatten(layer.state):
            name = f"layer.{i}.{key}"
            if isinstance(value, mx.array):
                arrays[name] = value
            else:
                metadata[name] = value
        offset = getattr(layer, "offset", None)
        metadata[f"layer.{i}.offset"] = (
            offset.tolist() if isinstance(offset, mx.array) else offset
        )
    return arrays, metadata


def replay(model, prompts, continuation, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    size = len(prompts)
    assert size > 1 and len(continuation) > 0
    rows, heads = [], []
    for tokens in prompts:
        cache = bg._rebuild_singleton_cache(model)
        assert cache is not None
        logits, hidden, _ = bg._call_backbone(
            model, mx.array([tokens], mx.uint32), cache
        )
        mx.eval(logits, hidden, [c.state for c in cache])
        bg._clear_rollback(cache)
        head = model.make_mtp_cache()
        model.mtp_forward(
            bg._head_input(model, hidden),
            mx.array([tokens[1:] + [continuation[0]]], mx.uint32),
            head,
        )
        mx.eval([c.state for c in head])
        heads.append(head)
        rows.append([c.extract(0) for c in cache])
    sampler = make_sampler(temp=1, top_k=20, top_p=0.95)
    batch = SimpleNamespace(
        model=model,
        uids=list(range(size)),
        prompt_cache=bg._merge_row_caches(rows),
        tokens=prompts,
        samplers=[sampler] * size,
        fallback_sampler=sampler,
        logits_processors=[],
        _next_tokens=None,
        _next_logprobs=[],
        _token_context=[None] * size,
        _num_tokens=[0] * size,
        _matchers=[None] * size,
    )
    del rows
    batch.extract_cache = lambda i: [c.extract(i) for c in batch.prompt_cache]
    receipt = dict(prompts=prompts, continuation=continuation, steps=[])
    for step, token in enumerate(continuation):
        batch._next_tokens = mx.full((size,), token, mx.uint32)
        result = bg._initial_batch_forward(batch)
        shared = result is not None
        if shared:
            logits, hidden = result[:2]
        else:
            row_caches, row_logits, row_hidden = [], [], []
            for idx in range(size):
                row = bg._make_row_batch(batch, idx)
                bg._set_singleton_mrope_delta(row)
                lp, h, _ = bg._call_backbone(
                    model, row._next_tokens[:, None], row.prompt_cache
                )
                bg._clear_rollback(row.prompt_cache)
                row_caches.append(row.prompt_cache)
                row_logits.append(lp)
                row_hidden.append(h)
            batch.prompt_cache = bg._merge_row_caches(row_caches)
            logits, hidden = mx.concatenate(row_logits), mx.concatenate(row_hidden)
        mx.eval(logits, hidden, [c.state for c in batch.prompt_cache])
        record = dict(step=step, shared=shared, rows=[])
        for idx in range(size):
            arrays, metadata = snapshot(batch.extract_cache(idx))
            arrays.update(logits=logits[idx], hidden=hidden[idx])
            # Identical forced head history in both arms; no RNG draws.
            head_logits = model.mtp_forward(
                bg._head_input(model, hidden[idx : idx + 1]),
                mx.array([[continuation[(step + 1) % len(continuation)]]], mx.uint32),
                heads[idx],
            )
            head_arrays, head_metadata = snapshot(heads[idx])
            arrays.update({f"head.{k}": v for k, v in head_arrays.items()})
            arrays["head.logits"] = head_logits
            metadata.update({f"head.{k}": v for k, v in head_metadata.items()})
            mx.eval(arrays)
            filename = f"step-{step}-row-{idx}.safetensors"
            mx.save_safetensors(str(directory / filename), arrays)
            record["rows"].append(dict(file=filename, metadata=metadata))
        receipt["steps"].append(record)
        (directory / "replay.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def compare(before, after):
    before, after = Path(before), Path(after)
    a = json.loads((before / "replay.json").read_text())
    b = json.loads((after / "replay.json").read_text())
    assert a["prompts"] == b["prompts"] and a["continuation"] == b["continuation"]
    assert len(a["steps"]) == len(b["steps"]) == len(a["continuation"])
    records = []
    for sa, sb in zip(a["steps"], b["steps"], strict=True):
        assert len(sa["rows"]) == len(sb["rows"]) == len(a["prompts"])
        for ra, rb in zip(sa["rows"], sb["rows"], strict=True):
            assert ra["metadata"] == rb["metadata"]
            aa, bb = mx.load(str(before / ra["file"])), mx.load(str(after / rb["file"]))
            assert aa.keys() == bb.keys()
            for name in aa:
                x, y = aa[name], bb[name]
                assert x.shape == y.shape and x.dtype == y.dtype, name
                delta = x.astype(mx.float32) - y.astype(mx.float32)
                record = dict(
                    file=ra["file"],
                    tensor=name,
                    dtype=str(x.dtype),
                    shape=list(x.shape),
                    exact=bool(mx.array_equal(x, y)),
                    finite=bool(mx.all(mx.isfinite(x)) & mx.all(mx.isfinite(y))),
                    max_abs=float(mx.max(mx.abs(delta))),
                    rms=float(mx.sqrt(mx.mean(delta * delta))),
                )
                if name in ("logits", "head.logits"):
                    record["top1_equal"] = bool(
                        mx.array_equal(mx.argmax(x, -1), mx.argmax(y, -1))
                    )
                    record["probability_l1"] = float(
                        mx.sum(
                            mx.abs(
                                mx.softmax(x.astype(mx.float32), -1)
                                - mx.softmax(y.astype(mx.float32), -1)
                            )
                        )
                    )
                records.append(record)
            del aa, bb, x, y, delta
    return dict(
        exact=all(r["exact"] for r in records),
        finite=all(r["finite"] for r in records),
        top1_equal=all(r.get("top1_equal", True) for r in records),
        before_shared=[s["shared"] for s in a["steps"]],
        after_shared=[s["shared"] for s in b["steps"]],
        tensors=records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare(args.before, args.after)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "tensors"}))
    raise SystemExit(0 if report["exact"] and report["finite"] else 1)
