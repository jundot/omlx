# Mapped Weights: serve weights straight from the model files

By default a model's weights are read from its safetensors files into fresh
Metal buffers and wired (pinned) in memory. With **Mapped Weights** the shards
are mapped read-only instead, and every tensor is handed to MLX as a view of
the file (`mx.from_dlpack(copy=False)`): no copy is made.

What that changes:

- The weights live in the macOS page cache, not in the oMLX process
  footprint.
- They are not wired, so macOS can drop pages the model is not using (cold
  MoE experts, for example) and read them back from the SSD when a token
  routes to them.
- Reloading the same model does not copy it again.

What it does **not** change: Metal counts a mapped buffer in full against its
GPU working-set cap (`iogpu.wired_limit_mb`) once a kernel uses it. A model
that does not fit under that cap still does not run; use
[MoE Expert Offload](MoE_Expert_Offload.md) for that.

Measured on an M2 Max 32 GB, Ornith-1.5-9B oQ4e (dense), 256-token greedy
generation, warm:

| | not mapped | mapped |
|---|---|---|
| oMLX phys_footprint | 8.2 GB | **1.5 GB** |
| decode | 46-48 tok/s | 45-48 tok/s |
| load | 2.4 s | 2.1 s |

Ornith-1.5-35B-A3B oQ4e (MoE, 22 GB), same machine with other apps open,
memory guard `aggressive`, 256 generated tokens, two rounds per mode:

| | not mapped | mapped |
|---|---|---|
| load | 12.6-13.2 s | **2.2-2.3 s** |
| oMLX phys_footprint | 23 GB | **2.0-2.6 GB** |
| decode, 1.2k-token prompt (warm) | 69-80 tok/s | 65-66 tok/s |
| 19.6k-token prompt | 1 of 4 completed (56 tok/s), 3 aborted by the guard | 4 of 4 completed, 40-54 tok/s |
| prefill, first request after load | 540-1000 tok/s | 143 tok/s (weights paged in from the SSD) |

With the guard on `balanced` the stock model loaded but every prompt was
rejected by the prefill guard, while the mapped one served both prompt sizes.

## Enabling it

Per model, in the admin dashboard: **Model Settings → Memory Saving → Mapped
Weights**, or through the settings API:

```json
{"mapped_weights_enabled": true}
```

The setting applies on the next load. It cannot be combined with MoE Expert
Offload, which reads experts into its own resident slots.

## Alignment

A GPU kernel reads a typed buffer at the alignment of its element type
(4 bytes for `uint32`, 2 for `float16`/`bfloat16`). A tensor whose offset in
the file is not a multiple of its element size cannot be mapped: Metal would
read it shifted. The loader checks every tensor and **copies the misaligned
ones** into memory exactly as a normal load does, so the output is always
correct; only the memory saving is lost for those tensors.

Most MLX checkpoints are misaligned this way. `mx.save_safetensors` does not
pad the JSON header, so the data section, and every tensor after it, starts
1-7 bytes off. (Checkpoints written by the Hugging Face `safetensors` library
pad the header and are usually aligned already.)

When you turn the setting on for a misaligned model, the dashboard tells you
how much of it cannot be mapped and lets you choose:

- **Cancel**: leave the files and the setting as they are.
- **Realign in place**: rewrite this model's misaligned shards one at a time
  (each into a temporary file next to it, then an atomic rename). Needs free
  disk for the largest shard and the model unloaded. Not offered for files
  that are symlinks into the Hugging Face cache, whose blobs are shared.
- **Realign as a copy**: write an aligned copy of the whole model next to your
  other models (`<name>-aligned`), with Mapped Weights enabled on the copy.
  The original is not touched. Needs free disk for the full model.

A realigned file is a normal safetensors file with the same tensors and
metadata: the header is padded with spaces and tensors are ordered by element
size, so no gaps are needed. Any other loader reads it exactly as before.

API:

- `GET /admin/api/models/{model_id}/mapped-weights`: alignment report
  (`aligned`, `misaligned_bytes`, `misaligned_files`, `can_realign_in_place`,
  `can_realign_copy`, `symlinked`, `loaded`) and the current realign job.
- `POST /admin/api/models/{model_id}/mapped-weights/realign` with
  `{"mode": "in_place"}` or `{"mode": "copy"}`: starts a realign job; poll the
  `GET` endpoint for `job.state` (`running`, `done`, `error`),
  `job.done_bytes`/`job.total_bytes` and, for a copy, `job.new_model_id`.

## Memory guard

The guard's usage side reads `mx.get_active_memory()`, which counts the whole
mapped file, while none of the file is in the process `phys_footprint`. With a
mapped model the dynamic ceiling therefore counts the mapped bytes as oMLX's
own, at load admission and afterwards. The part of the file that is resident
(measured with `mincore`, sampled off the request path) is also counted as
reclaimable: those pages are clean, and while Metal uses them macOS lists them
as active or wired rather than file cache. When the KV cache or other
allocations grow into them, macOS drops weight pages and the tokens that need
them read them back from the SSD, so decode slows down instead of the request
being aborted.

The Metal cap counts the full file and, with the static ceiling, is the limit
that binds a mapped model; raising `iogpu.wired_limit_mb` is what gives it more
room.

While any mapped model is loaded, oMLX lifts the MLX wired limit so mapped
pages are not pinned; it is restored when the last mapped model unloads. The
limit is process-wide in MLX, so other models loaded at the same time are not
wired either.

## Trade-offs

- **Gate/up fusion is skipped** for mapped models. Fusion concatenates each
  MoE layer's gate and up expert weights into a new buffer, which is a full
  copy of the routed experts. On the 35B above this costs about 10-15% of
  short-context decode speed.
- **The first request after a load is slower** while the weights are paged in
  from the SSD; later requests run at full speed.
- **Under memory pressure decode slows down instead of failing.** When other
  apps or a long context push mapped pages out, the next tokens that need them
  read from the SSD. With plenty of free memory, decode speed matches a normal
  load.
- Tensors that the model transforms at load time (dtype casts, for instance)
  become regular arrays and are not served from the mapping.
