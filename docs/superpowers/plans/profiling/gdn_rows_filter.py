"""Shim: restrict the fused Qwen4 GDN verify arm to the row counts in OMLX_GVF_ROWS (comma list)."""
import os
import omlx.patches.qwen35_gdn_prework as pm
ROWS = {int(x) for x in os.environ.get("OMLX_GVF_ROWS", "1").split(",") if x}
_orig = pm._qwen4_verify_dynamic_eligible


def gate(module, inputs, mask, cache, gdn_sink):
    if getattr(inputs, "ndim", 0) == 3 and inputs.shape[1] not in ROWS:
        return False
    return _orig(module, inputs, mask, cache, gdn_sink)


pm._qwen4_verify_dynamic_eligible = gate
print(f"[gdn_rows_filter] fused verify arm limited to rows {sorted(ROWS)}", flush=True)
