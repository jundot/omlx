"""Shim: log the first evaluations of the Qwen4 fused GDN verify gate per (rows, result) with call site."""
import logging, traceback
import omlx.patches.qwen35_gdn_prework as pm
log = logging.getLogger("omlx.patches.qwen35_gdn_prework")
_orig = pm._qwen4_verify_dynamic_eligible
_seen = {}


def probe(module, inputs, mask, cache, gdn_sink):
    r = _orig(module, inputs, mask, cache, gdn_sink)
    S = inputs.shape[1] if getattr(inputs, "ndim", 0) == 3 else None
    key = (S, r)
    if _seen.get(key, 0) < 2:
        _seen[key] = _seen.get(key, 0) + 1
        c0 = cache[0] if cache is not None else None
        c1 = cache[1] if cache is not None else None
        stack = [f"{f.name}:{f.lineno}" for f in traceback.extract_stack()[-9:-1]]
        log.info(
            "GATEPROBE S=%s result=%s sink=%s mask=%s lengths=%s leftpad=%s conv=%s/%s rec=%s/%s static=%s in=%s/%s stack=%s",
            S, r, gdn_sink is not None, type(mask).__name__ if mask is not None else None,
            getattr(cache, "lengths", None) is not None, getattr(cache, "left_padding", None) is not None,
            getattr(c0, "shape", None), getattr(c0, "dtype", None), getattr(c1, "shape", None), getattr(c1, "dtype", None),
            pm._qwen4_decode_static_eligible(module), getattr(inputs, "dtype", None), getattr(inputs, "shape", None), stack,
        )
    return r


pm._qwen4_verify_dynamic_eligible = probe
print("[gdn_gate_probe] installed", flush=True)
