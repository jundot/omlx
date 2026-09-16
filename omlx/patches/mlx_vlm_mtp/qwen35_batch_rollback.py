"""Select accepted GDN states without concatenating layers."""

from functools import wraps


def apply(language):
    cls = language.LanguageModel
    original = cls.rollback_speculative_cache
    if getattr(original, "_omlx_layer_rollback", False):
        return

    @wraps(original)
    def rollback(self, caches, gdn_states, accepted, block_size):
        if not (
            isinstance(accepted, (list, tuple))
            and len(accepted) > 1
            and gdn_states
            and all(len(s) > 11 and s[11] is not None for s in gdn_states)
        ):
            return original(self, caches, gdn_states, accepted, block_size)

        ssm = [
            c
            for c in caches
            if c is not None
            and not c.is_trimmable()
            and not hasattr(c, "zero_row_tail")
        ]
        if len(ssm) != len(gdn_states) or any(
            c[0] is None or c[1] is None for c in ssm
        ):
            return original(self, caches, gdn_states, accepted, block_size)
        if len({int(s[10]) for s in gdn_states}) != 1:
            return original(self, caches, gdn_states, accepted, block_size)

        # Delegate all KV trimming/padding to the existing model contract.
        kv = [
            c
            for c in caches
            if c is None or c.is_trimmable() or hasattr(c, "zero_row_tail")
        ]
        result = original(self, kv, [], accepted, block_size)
        accepted_mx = language.mx.array(accepted, dtype=language.mx.int32)
        for cache, entry in zip(ssm, gdn_states):
            conv_input, kernel_size, intermediate = entry[9:12]
            cache[1], cache[0] = language.gated_delta_accept_states(
                intermediate,
                conv_input,
                cache[1],
                cache[0],
                accepted_mx,
                kernel_size,
                use_kernel=True,
            )
        return result

    rollback._omlx_layer_rollback = True
    cls.rollback_speculative_cache = rollback
