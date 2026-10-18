# Mapped weights (experimental)

Mapped weights load aligned safetensors tensors as read-only file views rather
than copying them into new allocations. This can reduce loading work and let
macOS reclaim an idle model's clean file pages without writing them to swap.
It does **not** reduce the RAM required for active inference: Metal wires every
buffer a kernel references, including an MoE layer's complete expert tensor.
A low process `phys_footprint` excludes file-backed pages and is not a measure
of the model's total RAM consumption.

The feature is off by default. Enable **Mapped weights** in the model's Memory
Saving settings, or set `mapped_weights_enabled` through the normal model
settings API. Reload the model for a change to take effect:

```json
{"mapped_weights_enabled": true}
```

## Scope and compatibility

- Requires oMLX's pinned MLX 0.32.4 or newer. Its foreign-buffer ownership fix
  prevents sanitizers such as Qwen3.5's `norm + 1` from donating a read-only
  imported buffer. CPU and GPU transformations must produce regular new arrays.
- Only synchronous loads of safetensors in the selected model directory use the
  loader. Other threads, other directories, other formats, and explicit streams
  use the existing loader. Temporary proxies in the two loader modules are restored when loading
  returns or raises; `mlx.core.load` itself is never replaced. The existing mlx-lm/mlx-vlm paths still sanitize and construct models.
- Each tensor must start at an absolute file offset divisible by eight. This
  also covers byte tensors later reinterpreted as larger scalar types. MLX's
  dtype-view fix alone does not cover every external misaligned base pointer.
- Misaligned tensors are copied. Unsupported dtypes use the stock shard loader.
  The log reports mapped and copied bytes per shard, so an enabled toggle does
  not necessarily mean the entire model is mapped.
- Checkpoint files are never rewritten. There are no realignment jobs, extra
  model registrations, or proxy-cache files. A virtual mapping preserves the
  file offset within an OS page; changing the mmap base cannot realign data.
- NumPy/DLPack retains the mapping until the last derived MLX array is released.
  Weak exporter references only support physical accounting; they do not own mappings.
  There is no per-engine mapping registry or process-wide wired-limit override.
- MoE expert offload uses its own resident slots and cannot be combined with
  mapped weights. Fusion of MoE gate/up projections is skipped because it
  concatenates and copies the expert weights. This can reduce decode speed.

## Memory guard and idle behavior

Admission estimates, tier reserves, static ceiling and Metal cap are unchanged.
The dynamic automatic ceiling includes a conservative correction for read-only
mapped Metal buffers, which macOS excludes from process physical footprint.
It uses fresh `VM_REGION_TOP_INFO` object counters for whole-file mappings,
without a page walk, background sampler or cached result.

Only live NumPy exporters successfully imported without copies are tracked,
using weak references. Aliases of one file are deduplicated by device/inode.
All non-live tensor bytes, header bytes and page padding are subtracted from
object residency as possible overlap. Partial regions, COW objects, unsupported
queries and failed observations receive zero credit. All reclaimable file-cache
bytes are then subtracted as another possible overlap with the resulting bound.
Nonresident weights receive no credit. Residency changes do not enter upstream's
delayed-release credit ledger.

This correction aligns physical accounting with MLX's charged buffers; it never
discounts model admission or predicts idle eviction as extra headroom. Automatic
tiers can still refuse a model under desktop pressure. Custom remains an ordinary
optional fixed ceiling, not a requirement for mapped weights. The guard is never
disabled, and no limit or process-wide wired request is raised by this feature.

The region query uses the public Mach API; unsupported/split mappings can make
the bound conservative enough to cause refusals. It is a best-effort observation,
like the existing system-memory snapshot, not an atomic reservation of RAM.

Reclamation depends on macOS, GPU residency and activity. oMLX's GPU keep-warm
window and `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS` can delay it. A model being
idle does not guarantee its pages are already reclaimable. Compare both modes
with the same settings and record system wired memory, compression, swap,
page-ins, loading time, time to first token and sustained generation speed.

## Limits

- A model larger than the available RAM or GPU working-set cap can still fail
  or thrash. Active generation does not selectively evict cold GPU experts.
- Competing file I/O can evict model pages and cause slow requests. First use
  after load or pressure can pay SSD page-in costs; faster reload alone does
  not guarantee a faster first answer.
- Do not edit or truncate a checkpoint while it is loaded. Read-only mapping
  prevents this loader from writing, not another process from changing the
  backing file. Header parsing and mapping use the same opened file descriptor
  to avoid pathname replacement races during load.
- Sanitization, dtype conversion, fusion and quantization can allocate copies.
  Report actual memory behavior rather than assuming all weights remain mapped.

The intended benefit is lower load-time copying and cheaper recovery from idle
under memory pressure. It needs to be weighed against the cost of foregoing
fusion and the remaining temporary loader interception. It is not expert paging.
