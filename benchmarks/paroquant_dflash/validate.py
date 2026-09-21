# SPDX-License-Identifier: Apache-2.0
"""Validate a local ParoQuant target and DFlash draft on Metal.

Run from the checkout with python -m benchmarks.paroquant_dflash.validate
--target PATH --draft PATH. No model files or serving settings are changed.
"""

import argparse
import dataclasses
import importlib
import json
import time

import mlx.core as mx
from mlx.utils import tree_flatten


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    results = {"target": args.target, "draft": args.draft}
    from importlib.metadata import version

    results["versions"] = {
        name: version(name) for name in ("mlx", "mlx-lm", "dflash-mlx", "paroquant")
    }
    results["device"] = mx.device_info()
    from omlx.patches.dflash_lifecycle import (
        install_dflash_lifecycle_wrap,
        restore_dflash_class_patches,
    )
    from omlx.patches.dflash_paroquant import (
        load_target_bundle,
        validate_paroquant_draft,
    )
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(args.target)
    install_dflash_lifecycle_wrap()
    loader = importlib.import_module("paroquant.inference.backends.mlx.load")
    create = loader._create_model

    def audited_create(*a, **kw):
        model = create(*a, **kw)
        original = model.load_weights

        def audited_load(weights, *a, **kw):
            expected = dict(tree_flatten(model.parameters()))
            actual = dict(weights)
            audit = {
                "missing": sorted(set(expected) - set(actual)),
                "extra": sorted(set(actual) - set(expected)),
                "shape_mismatch": [
                    k
                    for k in expected.keys() & actual.keys()
                    if expected[k].shape != actual[k].shape
                ],
            }
            results["weight_audit"] = audit
            print("WEIGHT_AUDIT", json.dumps(audit), flush=True)
            return original(weights, *a, **kw)

        object.__setattr__(model, "load_weights", audited_load)
        return model

    loader._create_model = audited_create
    begin = time.perf_counter()
    bundle = load_target_bundle(args.target, lazy=False)
    loader._create_model = create
    results["load_seconds"] = time.perf_counter() - begin
    model, tok, ops = bundle.model, bundle.tokenizer, bundle.target_ops
    print("LOADED", results["load_seconds"], flush=True)
    capture = {6, 20, 34, 48, 62}
    ids = tok.encode("The capital of France is")
    x = mx.array([ids])
    # Restore hooks to establish an ordinary-forward reference.
    restore_dflash_class_patches()
    native = model(x, cache=model.make_cache())
    mx.eval(native)
    ops.text_model(model)._dflash_speculative_hooks_installed = False
    ops.install_speculative_hooks(model)
    actual, hidden = ops.forward_with_hidden_capture(
        model,
        input_ids=x,
        cache=ops.make_cache(model, enable_speculative_linear_cache=True),
        capture_layer_ids=capture,
    )
    mx.eval(actual)
    results["forward_max_abs"] = float(mx.max(mx.abs(actual - native)))
    assert bool(mx.all(mx.argmax(actual, axis=-1) == mx.argmax(native, axis=-1)))
    results["rollback"] = []
    for width in (3, 5, 8):
        candidate_ids = tok.encode(
            " Paris is a beautiful city with many famous landmarks."
        )[:width]
        assert len(candidate_ids) == width
        candidates = mx.array([candidate_ids])
        for accepted in range(width):
            cache = ops.make_cache(model, enable_speculative_linear_cache=True)
            out, _ = ops.forward_with_hidden_capture(
                model, input_ids=x, cache=cache, capture_layer_ids=capture
            )
            mx.eval(out)
            ops.arm_rollback(cache, prefix_len=len(ids))
            out, _ = ops.verify_block(
                target_model=model,
                verify_ids=candidates,
                target_cache=cache,
                capture_layer_ids=capture,
            )
            mx.eval(out)
            ops.restore_after_acceptance(
                cache,
                target_len=len(ids) + accepted + 1,
                acceptance_length=accepted,
                drafted_tokens=width - 1,
            )
            next_id = mx.array([[tok.encode(" Next")[0]]])
            after, _ = ops.forward_with_hidden_capture(
                model, input_ids=next_id, cache=cache, capture_layer_ids=capture
            )
            reference = model(
                mx.array([ids + candidate_ids[: accepted + 1] + next_id[0].tolist()]),
                cache=model.make_cache(),
            )[:, -1:, :]
            mx.eval(after, reference)
            row = {
                "width": width,
                "accepted": accepted,
                "max_abs": float(mx.max(mx.abs(after - reference))),
                "same_argmax": bool(
                    mx.all(mx.argmax(after, axis=-1) == mx.argmax(reference, axis=-1))
                ),
            }
            results["rollback"].append(row)
            print("ROLLBACK", json.dumps(row), flush=True)
            del cache
    from dflash_mlx.draft_backend import EagerDraftBackend
    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    from dflash_mlx.engine.target_ops import bind_draft_to_target
    from dflash_mlx.runtime import stream_dflash_generate
    from dflash_mlx.runtime.context import build_offline_runtime_context
    from dflash_mlx.runtime.loading import load_draft_bundle

    draft, meta = load_draft_bundle(args.draft)
    validate_paroquant_draft(bundle.meta, meta)
    bind_draft_to_target(draft, model, target_ops=ops)
    results["generation"] = []
    for prompt in [
        "The capital of France is",
        "def fibonacci(n):\n",
        "Count from one to twenty: one,",
    ]:
        prompt_ids = tok.encode(prompt)
        restore_dflash_class_patches()
        cache = model.make_cache()
        inp = mx.array([prompt_ids])
        baseline = []
        start = time.perf_counter()
        ttft = None
        for _ in range(args.max_tokens):
            logits = model(inp, cache=cache)
            token = int(mx.argmax(logits[0, -1]))
            if ttft is None:
                ttft = time.perf_counter() - start
            baseline.append(token)
            inp = mx.array([[token]])
        baseline_s = time.perf_counter() - start
        del cache
        ops.text_model(model)._dflash_speculative_hooks_installed = False
        ops.install_speculative_hooks(model)
        for width in (3, 5, 8):
            runtime = build_offline_runtime_context(
                verify_mode="off", draft_sink_size=0, draft_window_size=2048
            )
            tokens = []
            summary = None
            for event in stream_dflash_generate(
                target_model=model,
                target_ops=ops,
                tokenizer=tok,
                draft_model=draft,
                draft_backend=EagerDraftBackend(),
                prompt="",
                prompt_tokens_override=prompt_ids,
                max_new_tokens=args.max_tokens,
                stop_token_ids=set(),
                temperature=0.0,
                block_tokens=width,
                runtime_context=runtime,
            ):
                if isinstance(event, TokenEvent):
                    tokens.append(int(event.token_id))
                elif isinstance(event, SummaryEvent):
                    summary = dataclasses.asdict(event)
            row = {
                "prompt": prompt,
                "width": width,
                "baseline_seconds": baseline_s,
                "baseline_ttft": ttft,
                "same_tokens": tokens == baseline,
                "first_difference": next(
                    (i for i, (a, b) in enumerate(zip(tokens, baseline)) if a != b),
                    None,
                ),
                "baseline_ids": baseline,
                "dflash_ids": tokens,
                "text": tok.decode(tokens),
                "summary": summary,
            }
            results["generation"].append(row)
            print(
                "GENERATION",
                json.dumps(
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("baseline_ids", "dflash_ids")
                    }
                ),
                flush=True,
            )
    results["peak_memory_bytes"] = mx.get_peak_memory()
    restore_dflash_class_patches()
    from pathlib import Path

    Path(args.output).write_text(json.dumps(results, indent=2, default=str) + "\n")
    assert not any(
        results["weight_audit"].values()
    ), "Checkpoint text-weight audit failed"
    assert all(
        row["same_argmax"] for row in results["rollback"]
    ), "Rollback changed the next token"
    assert all(
        row["same_tokens"] for row in results["generation"]
    ), "Greedy generation diverged"


if __name__ == "__main__":
    main()
