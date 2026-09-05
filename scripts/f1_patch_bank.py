from pathlib import Path

p = Path('omlx/patches/expert_streaming/shard_bank.py')
src = p.read_text()

old = '_PROFILE_READS = os.environ.get("OMLX_EXPERT_STREAMING_PROFILE", "") == "1"'
new = '''_PROFILE_READS = os.environ.get("OMLX_EXPERT_STREAMING_PROFILE", "") == "1"

# Runtime arm switch for demand-read telemetry. _PROFILE_READS freezes at
# import; long-lived processes (server engines) need to flip the default
# AFTER backings exist, without an env-var restart. arm_read_telemetry()
# flips both this global and every live backing instance (weak-refs, so a
# dead backing never blocks the loop).
_ARM_REGISTRY: "weakref.WeakSet[ExpertBackingStore]" = None  # lazily built


def arm_read_telemetry(enabled: bool = True) -> bool:
    """"""Arm (or disarm) demand-read telemetry for live + future backings.

    Returns the previous default state. Existing ExpertBackingStore
    instances are flipped in place; backings created later inherit the
    new default because __init__ reads this module global.
    """""""
    global _PROFILE_READS
    prev = _PROFILE_READS
    _PROFILE_READS = bool(enabled)
    if _ARM_REGISTRY is not None:
        for store in list(_ARM_REGISTRY):
            tel = getattr(store, "read_telemetry", None)
            if tel is not None and tel.enabled != _PROFILE_READS:
                tel.enabled = _PROFILE_READS
    return prev'''

assert src.count(old) == 1, f"anchor count {src.count(old)}"
src = src.replace(old, new)

# Registry population: subscribe in __init__ after read_telemetry is built.
old2 = '        self.read_telemetry = ReadTelemetry(enabled=_PROFILE_READS)'
new2 = '''        self.read_telemetry = ReadTelemetry(enabled=_PROFILE_READS)
        # Runtime-arm registry (lazily created; weak so stores die freely).
        global _ARM_REGISTRY
        if _ARM_REGISTRY is None:
            import weakref
            _ARM_REGISTRY = weakref.WeakSet()
        _ARM_REGISTRY.add(self)'''
assert src.count(old2) == 1
src = src.replace(old2, new2)

p.write_text(src)
print('patched ok')
