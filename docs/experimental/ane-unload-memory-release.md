# ANE procedure-bank release on model unload

## Problem

Qwen ANE prefill attaches native procedure-bank objects to model modules. The
objects retain mapped weight blobs and IOSurfaces until their native references
are dropped. The existing pressure-shedding path releases those references,
but the ordinary engine unload path does not call it. Repeated load/unload
cycles can therefore leave the process footprint elevated until restart.

## Design

Reuse `release_qwen35_ane_prefill()` as the single release operation. It already
handles ordinary and fused ANE state, latches the per-module fallback flags
before dropping state, resets ANE status counters, and is idempotent. Call it
after the engine has stopped and closed, but before the wrapper drops its model
reference. This ordering prevents a live engine from observing a deliberately
disabled ANE state while it is still serving teardown work.

Apply the same ordering to both `BatchedEngine` and `VLMBatchedEngine`, because
both load paths can enable Qwen ANE prefill. Failures remain contained like
the existing optional ANE setup: teardown must continue to clear wrapper
references even if the release helper is unavailable or raises.

## Verification

Add unit coverage that uses fake ANE state objects and mocked engine teardown to
prove that:

1. ordinary unload calls the release helper after engine shutdown and before
   the model reference is cleared;
2. fused/GDN state is released through the shared helper; and
3. a release-helper failure does not prevent the normal engine cleanup.

The test suite is mock-only and does not start oMLX, load a model, alter model
locations, or change live oMLX configuration. Hardware validation remains a
separate follow-up using repeated ANE-enabled load/unload cycles and process
memory evidence.
