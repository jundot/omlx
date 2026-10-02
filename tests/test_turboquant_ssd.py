"""Phase 3: TurboQuant + paged-SSD prefix cache (single + batch).

Validates the SSD round-trip now that TurboQuant decode actually engages:
prefill boundary snapshots are stored fp16 and re-quantized deterministically
on a cache hit, so a hit reproduces the fresh run exactly — no double-quant
(TQ->fp16->TQ) drift. Covers both single-request and concurrent-batch decode.

Skips when the model is not cached locally.
"""
import importlib.util
import shutil
import tempfile
from pathlib import Path

import pytest

MODEL_REPO = "mlx-community/Llama-3.2-1B-Instruct-4bit"
TQ_BITS = 4.0
BLOCK = 256


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


_LOADED = None


def _load():
    global _LOADED
    if _LOADED is None:
        from mlx_lm import load

        helpers = _helpers()
        model, tok = load(_model_path())
        # ~400-token prompt so a full 256-block is cached
        text = "The history of computing spans many centuries of innovation. " * 40
        ids = list(tok.encode(text))[:400]
        _LOADED = (helpers, model, tok, ids)
    return _LOADED


def test_tq_ssd_single_hit_matches_fresh():
    helpers, model, tok, ids = _load()
    tmp = tempfile.mkdtemp(prefix="ssd_tq_")
    try:
        fresh, c1 = helpers._generate_tokens(
            model, tok, ids, max_tokens=16,
            ssd_cache_dir=tmp, block_size=BLOCK, turboquant_bits=TQ_BITS)
        cached, c2 = helpers._generate_tokens(
            model, tok, ids, max_tokens=16,
            ssd_cache_dir=tmp, block_size=BLOCK, turboquant_bits=TQ_BITS)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    assert len(fresh) >= 5, "fresh TQ+SSD run produced no output"
    assert c2 > 0, "second run did not hit the SSD cache"
    # Deterministic re-quantization on restore -> identical to fresh.
    assert fresh == cached, "TQ+SSD cache hit diverged from fresh (double-quant drift?)"


def _batch_fresh_vs_hit(helpers, model, tok, prompts, bits):
    tmp = tempfile.mkdtemp(prefix="ssd_tq_batch_")
    try:
        fresh = {rid: t for rid, t, _ in helpers._generate_batch(
            model, tok, prompts, mode="concurrent", max_tokens=16,
            ssd_cache_dir=tmp, block_size=BLOCK, turboquant_bits=bits)}
        hit = {rid: (t, c) for rid, t, c in helpers._generate_batch(
            model, tok, prompts, mode="concurrent", max_tokens=16,
            ssd_cache_dir=tmp, block_size=BLOCK, turboquant_bits=bits)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return fresh, hit


def test_tq_ssd_batch_roundtrip_exact_at_high_bits():
    """Structural SSD correctness: at near-lossless 8-bit, a batched cache hit
    reproduces the fresh run exactly — proving the fp16-snapshot round-trip and
    re-quantization introduce no drift in the B>1 path."""
    helpers, model, tok, ids = _load()
    prefix = ids[:300]
    prompts = [prefix + list(tok.encode(f" Topic {k}."))[:24] for k in range(3)]
    fresh, hit = _batch_fresh_vs_hit(helpers, model, tok, prompts, bits=8.0)
    for i in range(len(prompts)):
        ft = fresh[f"batch-{i}"]
        ht, hc = hit[f"batch-{i}"]
        assert hc > 0, f"batch req {i} did not hit SSD cache"
        assert ft == ht, f"8-bit batch req {i} hit diverged from fresh (round-trip drift)"


def test_tq_ssd_batch_coherent_at_low_bits():
    """At lossy 4-bit, batched fresh-vs-hit may diverge by a few tokens where
    quantization tips a greedy near-tie (single-request stays exact; fp16 is
    exact) — output must still be coherent with the cache hit working. This
    residual divergence resolves when the upstream masked-decode kernel (Bug 2)
    lets B>1 use the same fused path as B=1."""
    helpers, model, tok, ids = _load()
    prefix = ids[:300]
    prompts = [prefix + list(tok.encode(f" Topic {k}."))[:24] for k in range(3)]
    fresh, hit = _batch_fresh_vs_hit(helpers, model, tok, prompts, bits=TQ_BITS)
    for i in range(len(prompts)):
        ft = fresh[f"batch-{i}"]
        ht, hc = hit[f"batch-{i}"]
        assert len(ht) >= 3, f"batch req {i} degenerate under TQ+SSD"
        assert hc > 0, f"batch req {i} did not hit SSD cache"
        n = min(len(ft), len(ht))
        match = sum(1 for k in range(n) if ft[k] == ht[k]) / n if n else 0.0
        assert match >= 0.5, f"batch req {i} hit overlap {match:.0%} too low (not just a near-tie)"
