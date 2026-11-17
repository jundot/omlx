# Image engine lifecycle (unadvertised)

PR 5 adds an `ImageEngine` using the existing non-streaming activity and shared
MLX executor conventions. It is deliberately not registered in discovery/pool
or exposed by an endpoint until PR 6.

The standalone implementation now also exposes an iterator. Each encoder layer,
denoiser block and VAE stage evaluates before yielding. The engine submits one
iterator step per executor call and uses `ForwardFairnessGate` to account for
GPU time between steps. The synchronous runner consumes the same iterator.

There is at most one request per engine; a second request fails busy instead of
building an unbounded queue. Admission reserves a conservative headroom estimate
of 1.10 times the complete artifact weight bytes against the configured soft
limit (or Apple's recommended working set). This exceeds the measured standalone
peak for this exact recipe; it is not an estimate for arbitrary models.

Weights are loaded per request and released between pipeline phases. `start()`
inspects structure; it does not pin all weights. On cancellation, an event asks the
iterator to stop at the next boundary. The engine waits for the current submitted
worker operation even through repeated task cancellation, then closes the
iterator on the executor. Activity remains live until cleanup drains. `stop()`
requests cancellation and waits for that ownership lock before marking stopped.
Engine execution does not reset the process-global peak allocator counter.

## Evidence

Deterministic barrier tests cover normal completion, repeated cancellation while
a worker is active, busy rejection, stop waiting for work, failed admission and
failed start. Native M3 Max/64 GB acceptance on current upstream pins:

- Cancelled during fresh prompt encoding; worker drained and activity cleared.
- Reused the same engine successfully after cancellation.
- Generated the same seed-19 sailboat PNG byte-for-byte as the standalone PR 4
  run (`699f8a5c6e73c3b96ad0c5baad9117b05ec36dd3fbfbc68dc9d5fc8c92960dab`).
- Stopped cleanly with no active operation.

That run took 74.54 seconds; this is not a performance comparison with PR 4.
Real chat coexistence, pool eviction/TTL, disconnect and manual unload through
HTTP remain activation gates for PR 6. A long submitted kernel cannot be
interrupted, and bounded steps do not guarantee a fixed chat-latency ceiling.
