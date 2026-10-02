"""Phase 2: batched TurboQuant accuracy + memory/occupancy vs single-seq.

Three comparisons on a real model:
  - occupancy: KV-cache bytes/token, TQ vs fp16, single vs batch (+ pad waste),
               measured at a controlled length so over-allocation slack cancels;
               long-context savings projected from per-token bytes.
  - accuracy : concurrent B>1 TQ vs single-seq TQ (token match) + coherence.
  - peak mem : live peak during decode, TQ vs fp16, single vs batch (with the
               caveat that at short context the model weights dominate).

Skips when the model is not cached. Run directly to write the report:
    python tests/test_turboquant_batch_memory.py
"""
import importlib.util
from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_vlm.turboquant import TurboQuantKVCache

MODEL_REPO = "mlx-community/Llama-3.2-1B-Instruct-4bit"
TQ_BITS = 4.0
MAX_TOKENS = 32
OCC_LEN = 512  # multiple of TurboQuant cache_step (256) → no over-alloc slack


def _model_path():
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(MODEL_REPO, local_files_only=True)
    except Exception:
        return None


pytestmark = [
    pytest.mark.turboquant,
    pytest.mark.slow,
    pytest.mark.skipif(_model_path() is None, reason=f"{MODEL_REPO} not cached"),
]


def _helpers():
    spec = importlib.util.spec_from_file_location(
        "itest", str(Path(__file__).parent / "integration" / "test_full_integration.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prompts(tokenizer):
    msgs = [
        "Name three primary colors.",
        "What is the capital of Japan?",
        "Write one sentence about the ocean.",
        "List two kinds of fruit.",
    ]
    return [
        list(tokenizer.apply_chat_template(
            [{"role": "user", "content": m}], add_generation_prompt=True))
        for m in msgs
    ]


def _convert_to_tq(cache, bits, skip_last=True):
    """Mirror Scheduler._apply_turboquant_kv_convert (dense KVCache only)."""
    kv = [i for i, c in enumerate(cache) if isinstance(c, KVCache)]
    last = kv[-1] if (skip_last and len(kv) > 1) else -1
    return [
        (c if (not isinstance(c, KVCache) or i == last)
         else TurboQuantKVCache.from_cache(c, bits=bits))
        for i, c in enumerate(cache)
    ]


def _occupancy_at(model, length, bits=None):
    """KV bytes after feeding `length` tokens (fp16, or TQ-converted)."""
    cache = make_prompt_cache(model)
    model(mx.zeros((1, length), dtype=mx.int32), cache=cache)
    mx.eval([c.state for c in cache])
    if bits is not None:
        cache = _convert_to_tq(cache, bits)
        mx.eval([c.state for c in cache if not isinstance(c, KVCache) or c.offset])
    return sum(c.nbytes for c in cache)


def _peak(fn):
    mx.reset_peak_memory()
    out = fn()
    return out, mx.get_peak_memory()


def _gather():
    from mlx_lm import load

    helpers = _helpers()
    model, tokenizer = load(_model_path())
    prompts = _prompts(tokenizer)
    lens = [len(p) for p in prompts]

    # --- occupancy at a controlled length (over-alloc slack cancels) ---
    occ_fp16 = _occupancy_at(model, OCC_LEN)
    occ_tq = _occupancy_at(model, OCC_LEN, bits=TQ_BITS)
    bpt_fp16 = occ_fp16 / OCC_LEN          # bytes per token, fp16
    bpt_tq = occ_tq / OCC_LEN              # bytes per token, TQ
    # batch (B requests, left-padded to max len): analytical, no over-alloc noise
    max_len = max(lens)
    batch_bytes_tq = len(lens) * max_len * bpt_tq
    batch_bytes_fp16 = len(lens) * max_len * bpt_fp16  # same lengths, fp16
    pad_waste = (len(lens) * max_len - sum(lens)) * bpt_tq

    # --- accuracy + live peak (through the real scheduler) ---
    (single_tq, peak_single_tq) = _peak(
        lambda: [helpers._generate_tokens(model, tokenizer, p, max_tokens=MAX_TOKENS, turboquant_bits=TQ_BITS)[0] for p in prompts])
    (_, peak_single_fp) = _peak(
        lambda: [helpers._generate_tokens(model, tokenizer, p, max_tokens=MAX_TOKENS)[0] for p in prompts])
    (_, peak_batch_fp) = _peak(
        lambda: helpers._generate_batch(model, tokenizer, prompts, mode="concurrent", max_tokens=MAX_TOKENS))
    (batch_tq_res, peak_batch_tq) = _peak(
        lambda: helpers._generate_batch(model, tokenizer, prompts, mode="concurrent", max_tokens=MAX_TOKENS, turboquant_bits=TQ_BITS))

    batch_tq = {rid: toks for rid, toks, _ in batch_tq_res}
    matches = []
    for i in range(len(prompts)):
        s, b = single_tq[i], batch_tq.get(f"batch-{i}", [])
        n = min(len(s), len(b))
        matches.append(100.0 * sum(1 for k in range(n) if s[k] == b[k]) / n if n else 0.0)

    return dict(
        lens=lens, occ_len=OCC_LEN,
        occ_fp16=occ_fp16, occ_tq=occ_tq, bpt_fp16=bpt_fp16, bpt_tq=bpt_tq,
        batch_bytes_tq=batch_bytes_tq, batch_bytes_fp16=batch_bytes_fp16,
        pad_waste=pad_waste, max_len=max_len,
        peak_single_fp=peak_single_fp, peak_single_tq=peak_single_tq,
        peak_batch_fp=peak_batch_fp, peak_batch_tq=peak_batch_tq,
        batch_tq=batch_tq, matches=matches,
    )


_M = None


def _metrics():
    global _M
    if _M is None:
        _M = _gather()
    return _M


def test_batch_tq_coherent_and_tracks_single():
    m = _metrics()
    for i in range(len(m["lens"])):
        assert len(m["batch_tq"].get(f"batch-{i}", [])) >= 5, f"batch req {i} degenerate"
    assert max(m["matches"]) >= 50.0, f"no request tracked single-seq: {m['matches']}"


def test_occupancy_tq_below_fp16():
    m = _metrics()
    ratio = m["occ_tq"] / m["occ_fp16"]
    assert ratio < 0.6, f"TQ occupancy ratio {ratio:.2f} not below fp16"


def test_batch_occupancy_beats_fp16_and_pad_nonnegative():
    m = _metrics()
    # same lengths, so the batch saving equals the per-token ratio (<0.6)
    assert m["batch_bytes_tq"] < 0.6 * m["batch_bytes_fp16"], "batch TQ not saving vs fp16"
    assert m["pad_waste"] >= 0


def test_peaks_recorded():
    m = _metrics()
    for k in ("peak_single_fp", "peak_single_tq", "peak_batch_fp", "peak_batch_tq"):
        assert m[k] > 0


def _write_report(m, path="tq_batch_memory.md"):
    gb, kb = 1024 ** 3, 1024
    nb = len(m["lens"])
    ratio = m["occ_tq"] / m["occ_fp16"]
    # project savings at a long context where KV (not weights) dominates
    proj_ctx = 8192
    proj_fp16 = nb * proj_ctx * m["bpt_fp16"] / gb
    proj_tq = m["batch_bytes_tq"] / m["max_len"] * proj_ctx / gb
    lines = [
        f"# Batched TurboQuant — memory/occupancy ({MODEL_REPO}, {TQ_BITS}-bit)\n",
        f"Batch requests: {m['lens']} tokens; occupancy measured at {m['occ_len']} tokens.\n",
        "## KV occupancy (storage)\n",
        "| metric | value |",
        "|---|---:|",
        f"| fp16 bytes/token | {m['bpt_fp16']:,.0f} B |",
        f"| TQ bytes/token | {m['bpt_tq']:,.0f} B |",
        f"| TQ / fp16 ratio | {ratio:.3f}x |",
        f"| batch(B={nb}) TQ bytes | {m['batch_bytes_tq']/kb:,.0f} KB |",
        f"| batch(B={nb}) fp16 bytes | {m['batch_bytes_fp16']/kb:,.0f} KB |",
        f"| batch TQ / fp16 (same lengths) | {m['batch_bytes_tq']/m['batch_bytes_fp16']:.3f}x |",
        f"| left-padding waste | {m['pad_waste']/kb:,.1f} KB ({100*m['pad_waste']/m['batch_bytes_tq']:.0f}% of batch) |\n",
        f"## Projected KV at {proj_ctx}-token context, B={nb} (where KV dominates)\n",
        "| | total KV |",
        "|---|---:|",
        f"| fp16 | {proj_fp16:.2f} GB |",
        f"| TQ | {proj_tq:.2f} GB |",
        f"| saved | {proj_fp16 - proj_tq:.2f} GB ({100*(1-proj_tq/proj_fp16):.0f}%) |\n",
        "## Peak memory, live decode (short prompts → weights dominate)\n",
        "| scenario | peak |",
        "|---|---:|",
        f"| single-seq fp16 | {m['peak_single_fp']/gb:.3f} GB |",
        f"| single-seq TQ | {m['peak_single_tq']/gb:.3f} GB |",
        f"| batch fp16 | {m['peak_batch_fp']/gb:.3f} GB |",
        f"| batch TQ | {m['peak_batch_tq']/gb:.3f} GB |",
        "",
        "_Note: at short context the 1B model weights (~0.7 GB) dominate peak;_",
        "_TQ's win shows in the projected long-context KV above. B>1 decode now_",
        "_runs the quantized kernels directly (no per-step batch dequantize)._\n",
        "## Accuracy: batch vs single-seq TQ (token match)\n",
        "| request | match % |",
        "|---|---:|",
    ]
    for i, pct in enumerate(m["matches"]):
        lines.append(f"| batch-{i} | {pct:.0f}% |")
    Path(path).write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    p = _write_report(_metrics())
    print(f"wrote {p}\n")
    print(Path(p).read_text())
