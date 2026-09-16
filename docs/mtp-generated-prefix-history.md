# Generated MTP prefix history

Qwen4's backbone prefix cache can include generated tokens, but prompt priming
only publishes MTP sidecars while processing the prompt. A later turn that
restores a block from generated history can therefore miss the matching MTP
head history.

The scheduler now registers each eligible text request by model and generation
UID. After a committed MTP fold reaches a full block boundary, the emission
path detaches the head cache at boundary minus one and its pending normalized
hidden row. It publishes the snapshot only after successful token emission,
including a terminal token that clears the generation batch.

The existing prefix cache owns token-chain hashing, its four-entry LRU and
backbone eviction. A sidecar cannot be restored until its matching backbone
block is live. Snapshots are materialized at capture so they cannot retain the
live model graph, and request plans hold the cache through a weak reference.
Cancellation and scheduler resets discard unused plans.

This path accepts MTP hosts reporting `qwen4_exp` or `qwen4_exp_text`; the
Qwen4 language model reports the latter even when its outer configuration
uses `qwen4_exp`. It requires enabled chained MTP priming, a successful full
prompt-history handoff and commit alignment equal to the prefix block size.
It skips requests with media cache metadata, stale UIDs, incomplete head
history and unsupported cache types. A miss keeps the existing behavior.
There is no SSD format change, partial-block storage or new attention kernel.

The tests check admission with a tiny Qwen4 language model constructed from
its native configuration, both directly and through language-model wrappers.
They exercise real MTP folds with a tiny random Qwen3.5 model that shares the
fold/cache contract, and check native cache publication, restore, eviction,
terminal cleanup and request isolation. Both supported host names are covered
for publication, restoration and media exclusion. This is not a full Qwen4
model quality, long-context memory or throughput benchmark. Any end-to-end
speed claim needs a repeated-prompt comparison on the same model, requests
and sampling settings.
