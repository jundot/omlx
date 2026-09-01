# DeepSeek-V4-Flash-Vision two-Mac validation

Status: experimental; single-Mac/loopback validated, physical two-Mac run pending.

This path is intentionally specific to
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. Rank 0 preprocesses images, owns
the `vision.*`, `aligner.*`, and image-sentinel parameters, and broadcasts the
merged MLX prompt embeddings. The existing DeepSeek-V4 pipeline owns the
language prefill and decode. Other distributed VLM families and tensor
parallelism fail closed.

The public checkpoint is about 167.8 GB (156.3 GiB) before runtime buffers.
That number is not a per-Mac fit guarantee. The signed planner output is the
authority for per-rank weights, KV reserve, and headroom.

## Prepare both Macs

Use the same commit, absolute model path, Python, MLX, MLX-LM, and oMLX
environment on Mac A and Mac B. Mac A is rank 0/coordinator.

```bash
git fetch origin
git switch codex/deepseek-v4-vision-distributed
uv sync --dev
export DSV4_VISION_MODEL=/Users/SHARED/models/DeepSeek-V4-Flash-Vision-Exp
test -f "$DSV4_VISION_MODEL/config.json"
uv run omlx --version
uv run python -c 'import mlx, mlx_lm; print(mlx.__version__, mlx_lm.__version__)'
uv run omlx start
```

Replace `/Users/SHARED/...` with one identical absolute path on both Macs.
In **Settings > Advanced**, enable **Distributed Inference**, save, and restart.
Pair Mac B in Mac A's **Cluster** tab. Prefer the direct Thunderbolt addresses;
use Ring first, then JACCL only after the baseline succeeds.

## Ordered test procedure

1. Verify connectivity on both Macs.

   ```bash
   uv run omlx cluster status --json | tee /tmp/omlx-cluster-status.json
   uv run omlx cluster worker-smoke
   uv run omlx cluster collective-smoke
   uv run omlx cluster pipeline-smoke
   ```

   Success: every command exits zero and collective/pipeline smoke reports both
   ranks. Capture route, interface, backend, and measured bandwidth on failure.

2. Build the planner-only dry run on Mac A.

   ```bash
   uv run omlx cluster plan \
     --model "$DSV4_VISION_MODEL" \
     --node mac-a=128GiB \
     --node mac-b=128GiB \
     --reserve 16GiB \
     --json | tee /tmp/dsv4-vision-plan.json
   ```

   Success: `supports_pipeline` is true, `tensor_parallel_size` is 1, rank 0
   has non-zero `coordinator_weight_bytes`, rank 1 has zero, layer ranges are
   contiguous with no gap/overlap, and both ranks have positive headroom. Rank
   0 should normally receive fewer late language layers because it also owns
   the ViT. Do not continue if either planned resident total exceeds its
   capacity-minus-reserve.

3. In Mac A's Cluster tab select the downloaded model, the two 128 GiB nodes,
   a 16 GiB reserve, `tensor_parallel_size=1`, and Ring. Stage, review the exact
   shard map, and activate. Request the model once to trigger lazy load.

   Success: both ranks reach `ready`; rank 0 logs `vision_owner=true`; loaded
   layer ranges equal `/tmp/dsv4-vision-plan.json`; measured parameter bytes
   remain within the planner tolerance. If not, collect both rank markers and
   logs from `config/model recognized`, `manifest scanned`, `loading_weights`,
   and `validating`.

4. Send a text-only request.

   ```bash
   export OMLX_URL=http://127.0.0.1:8000
   export OMLX_MODEL=DeepSeek-V4-Flash-Vision-Exp
   curl -fsS "$OMLX_URL/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"model":"'"$OMLX_MODEL"'","messages":[{"role":"user","content":"Reply with exactly: text path ready"}],"max_tokens":32}'
   ```

   Success: normal text response and no `vision_encode_begin` event.

5. Send one short image request. Put a small local test image behind a URL
   reachable by Mac A (or substitute a base64 data URL).

   ```bash
   export TEST_IMAGE_URL=http://127.0.0.1:8080/test.png
   curl -fsS "$OMLX_URL/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"model":"'"$OMLX_MODEL"'","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"'"$TEST_IMAGE_URL"'"}},{"type":"text","text":"Describe this image in one sentence."}]}],"max_tokens":64}'
   ```

   Success: one each of `vision_encode_begin`, `vision_encode_complete`,
   `multimodal_embeddings`, `distributed_prefill_begin`,
   `distributed_prefill_complete`, and `first_token`; only rank 0 reports an
   image count. Both ranks must report the same expanded sequence length.

6. Repeat step 5 with `"stream":true` and `curl -N`. Success is incremental
   `data:` chunks ending in `[DONE]`, with one vision encode and continuous
   decode after the first token.

7. Send a different second image with the same dimensions. Success is a new
   `vision_encode_begin` and a different content-keyed prompt-cache identity;
   the first image's KV state must not be reused.

8. Send a longer text prompt plus one image. Success is a single unchunked
   image-bearing prefill, then ordinary decode. The vision tower must not run
   again after prefill.

9. During steps 5–8 capture the Cluster dashboard and:

   ```bash
   uv run omlx cluster status --json | tee /tmp/omlx-cluster-after-vlm.json
   ```

   Success: peak memory remains below each rank's approved ceiling and no rank
   enters memory-pressure teardown. If it does, reduce context/prefill load or
   increase reserve; do not treat aggregate 256 GiB as one address space.

10. Gracefully unload the model in the Models/Cluster UI, verify both rank
    processes exit, then load it again and repeat the text and one-image short
    requests. Success is clean `teardown`, followed by two new `ready` ranks
    and identical answers without stale cache state.

## Capture checklist

- [ ] Exact git commit, `uv run omlx --version`, Python, MLX, MLX-LM, macOS.
- [ ] `/tmp/dsv4-vision-plan.json`; rank 0 late range plus coordinator bytes;
      rank 1 early range and zero coordinator bytes.
- [ ] Per-rank planned, measured, peak, reserve, and headroom bytes.
- [ ] Complete stage logs from recognition through teardown; no prompt/image
      contents or tensor dumps.
- [ ] Ring/JACCL interface, route, collective bandwidth, and transport errors.
- [ ] Model load time and image-encode time.
- [ ] TTFT, prefill tok/s, decode tok/s, and peak memory per Mac.
- [ ] Text, one-image, streaming, second-image, long-prompt, unload/reload results.

Passing loopback tests does not prove Thunderbolt RDMA/JACCL correctness. Only
record the backend as hardware-validated after the physical run completes.
