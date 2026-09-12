"""Shim: for 2-row gdn_sink calls, run the stock GDN forward on a shadow cache next to the patched
one and log any numeric difference (out, conv, recurrent, q/k/v, intermediate states)."""
import logging, os
import mlx.core as mx
import omlx.patches.qwen35_gdn_prework as pm
from mlx_vlm.models.qwen3_5 import language as q35

log = logging.getLogger("omlx.patches.qwen35_gdn_prework")
ROWS = int(os.environ.get("OMLX_DUAL_ROWS", "2"))
LIMIT = int(os.environ.get("OMLX_DUAL_LIMIT", "3600"))
_apply = pm.apply_qwen35_gdn_prework_patch
_orig_call = q35.Qwen3_5GatedDeltaNet.__call__
state = {"n": 0, "layer": 0, "diff_calls": 0}


class Shadow:
    def __init__(self, cache):
        self._s = {0: cache[0], 1: cache[1]}
        self.lengths = getattr(cache, "lengths", None)
        self.left_padding = getattr(cache, "left_padding", None)

    def __getitem__(self, i):
        return self._s[i]

    def __setitem__(self, i, v):
        self._s[i] = v

    def advance(self, n):
        pass


def apply_and_wrap():
    ok = _apply()
    cls = q35.Qwen3_5GatedDeltaNet
    patched = cls.__call__

    def dual(self, inputs, mask=None, cache=None, gdn_sink=None, target_verify=False):
        if gdn_sink is None or cache is None or inputs.shape[1] != ROWS or state["n"] >= LIMIT:
            return patched(self, inputs, mask=mask, cache=cache, gdn_sink=gdn_sink, target_verify=target_verify)
        state["n"] += 1
        shadow = Shadow(cache)
        sink2 = []
        ref = _orig_call(self, inputs, mask=mask, cache=shadow, gdn_sink=sink2, target_verify=True)
        out = patched(self, inputs, mask=mask, cache=cache, gdn_sink=gdn_sink, target_verify=target_verify)
        pairs = [("out", out, ref), ("conv", cache[0], shadow[0]), ("rec", cache[1], shadow[1]),
                 ("q", gdn_sink[-1][0], sink2[0][0]), ("k", gdn_sink[-1][1], sink2[0][1]),
                 ("v", gdn_sink[-1][2], sink2[0][2]), ("a", gdn_sink[-1][3], sink2[0][3]),
                 ("b", gdn_sink[-1][4], sink2[0][4]), ("conv_in", gdn_sink[-1][9], sink2[0][9]),
                 ("inter", gdn_sink[-1][11], sink2[0][11])]
        diffs = {}
        for name, x, y in pairs:
            if x is None or y is None:
                if x is not y:
                    diffs[name] = "None-mismatch"
                continue
            if x.shape != y.shape:
                diffs[name] = f"shape {x.shape} vs {y.shape}"
                continue
            d = mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max().item()
            if d > 0:
                diffs[name] = d
        if diffs:
            state["diff_calls"] += 1
            if state["diff_calls"] <= 40:
                log.info("DUALPROBE call=%d layer=%d S=%d bits=%s diffs=%s", state["n"], (state["n"] - 1) % 36, inputs.shape[1],
                         (self.in_proj_qkv.bits, self.in_proj_z.bits), diffs)
        elif state["n"] % 360 == 0:
            log.info("DUALPROBE call=%d no diffs so far (diff_calls=%d)", state["n"], state["diff_calls"])
        return out

    cls.__call__ = dual
    log.info("DUALPROBE installed (rows=%d)", ROWS)
    return ok


pm.apply_qwen35_gdn_prework_patch = apply_and_wrap
print("[gdn_dual_probe] installed", flush=True)
