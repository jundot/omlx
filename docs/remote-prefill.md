# Remote prefill on a vLLM server over MCDMA

Status: experimental, off unless `OMLX_REMOTE_PREFILL_URL` is set

Prefill is compute-bound and decode is bandwidth-bound, so a CUDA machine can
process a long prompt many times faster than a Mac, while the Mac decodes well.
With remote prefill, oMLX sends long prompts for one model to a vLLM server, its
KV cache comes back over [MCDMA](https://github.com/ashhart/MCDMA) RDMA links,
and oMLX decodes from it. oMLX computes the generation prompt itself, so the
first token comes from its own model.

## What you need

- A vLLM server running the same model, with MCDMA's KV connector enabled (see
  MCDMA's `docs/kv-handoff.md`). vLLM may use a different quantization; see
  [Numerics](#numerics).
- `mcdma-rpcd` from the same MCDMA release on both sides: on each vLLM host,
  one `listen` daemon per tensor-parallel rank, which the connector registers
  with; on the Mac, the `connect` daemon with one peer entry per rank.
- `libmcdma-rpc` on the Mac, in `/usr/local/lib`, `/usr/lib` or
  `OMLX_MCDMA_RPC_LIBRARY`.
- A model whose layers use standard attention caches (mlx-lm `KVCache` or
  `QuantizedKVCache`). DeepSeek-style MLA works when vLLM keeps its latent cache
  in bf16 or fp16.

## How a request is served

1. When a request reaches the head of the queue, oMLX checks its local prefix
   cache. If the prompt still has at least `OMLX_REMOTE_PREFILL_MIN_TOKENS`
   uncached tokens before its generation prompt, oMLX first checks that the
   connector answers on every link: before vLLM has the prompt, it answers the
   handoff's OPEN with WAIT within milliseconds, and silence for 5 seconds fails
   the attempt. oMLX then sends the prompt's token IDs to vLLM with the handoff
   ID, asking it to export only what the Mac lacks.
2. The request waits at the head of the queue while vLLM prefills. Requests
   already running keep decoding.
3. oMLX reads every rank's manifest before any page moves. The manifest must
   carry the digest of this prompt and the local model's KV geometry: its layer
   count, and per layer its key/value heads and head size, or for MLA its
   latent and rope sizes, as the model's config gives them. oMLX then pulls the
   pages frame by frame, checks each frame's CRC-32 and place in the manifest,
   and frees the pages on the vLLM side.
4. The pages become cache entries: key and value heads from every rank are put
   back in order, appended to what the local prefix cache restored, and the
   request is admitted with only its generation prompt left to prefill here.
   When the request finishes, the full cache is stored in the local prefix
   cache as usual, so follow-up turns hit locally.

Any failure falls back to local prefill for that request, with the reason in
the log: vLLM refusing, a link going down or reconnecting, a bad checksum or
manifest, pages for a different prompt or model, or an error applying them.
Three failures in a row pause remote prefill for a minute. A connector that
does not answer, a vLLM that does not answer within the timeout, or an export
that never appears after vLLM answers pauses it at once. Each further pause
before a remote prefill succeeds is twice as long, up to 15 minutes.

A request gets one remote prefill at most. If the scheduler puts it back in the
queue after its handoff, it keeps the handed-over cache. When a request is
aborted while its handoff runs, the pull stops between frames and frees the
pages on the vLLM side, and only then may the next handoff use the links.

## Dashboard and API

The Cluster dashboard shows a **Remote prefill** card for each loaded model
that uses it: the vLLM server and links, whether remote prefill is ready,
prefilling or paused, and the latest handoff's size, prefill time, transfer time
and rate. `GET /admin/api/remote-prefill` returns the same, including the last
failure.

## Settings

| Variable | Effect |
| --- | --- |
| `OMLX_REMOTE_PREFILL_URL` | vLLM's base URL, for example `http://prefill-host:8000` |
| `OMLX_REMOTE_PREFILL_MODEL` | The model name vLLM serves |
| `OMLX_REMOTE_PREFILL_FOR` | The oMLX model that uses remote prefill |
| `OMLX_REMOTE_PREFILL_LINKS` | The Mac's link names, one per vLLM rank, in rank order |
| `OMLX_REMOTE_PREFILL_MIN_TOKENS` | Uncached prompt tokens that make remote prefill worth it (default 4096) |
| `OMLX_REMOTE_PREFILL_TIMEOUT` | Seconds for one remote prefill, transfer included (default 120) |
| `OMLX_REMOTE_PREFILL_CHECKSUM` | `0` skips the per-frame CRC-32 once a setup is proven (default on) |
| `OMLX_REMOTE_PREFILL_API_KEY` | Bearer token for vLLM, if it requires one |

## Numerics

The KV cache vLLM computes matches what the Mac would compute only as closely
as the two runtimes agree: different quantizations and kernels give slightly
different keys and values. Check a model before relying on it, for example by
comparing next-token distributions for a few long prompts with remote prefill
on and off.

## Limits

- One remote prefill at a time: the request at the head of the queue. Requests
  behind it wait, as they do behind a cache freshness wait, for as long as the
  timeout at most.
- The KV geometry comes from the model's config (`num_key_value_heads`,
  `head_dim`, or `kv_lora_rank` and `qk_rope_head_dim` for MLA), the same for
  every layer. Models whose config lacks it always prefill locally.
- Multimodal and SpecPrefill prompts, and models with sliding-window, recurrent
  or hybrid caches, always prefill locally.
- vLLM's fp8 KV caches are not accepted.
- The CRC-32 costs roughly one pass over the data on each side.
