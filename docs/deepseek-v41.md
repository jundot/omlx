# DeepSeek V4.1 Flash

Place the original checkpoint or an oQ export in a configured model directory. oMLX detects V4.1 and loads it through the VLM engine. Original FP4/FP8 tensors are repacked automatically without modifying the checkpoint; no separate conversion is required.

## Engram storage

Enable **SSD N-gram Offload (DeepSeek V4.1)** in model settings to gather Engram rows from SSD. Changing the setting reloads the model. If estimated residency exceeds the configured model-memory ceiling, oMLX forces offload without changing the saved preference.

RAM mode keeps packed Engram tables in Metal-managed buffers shared with CPU gathers. It does not expand the tables to BF16. SSD mode prefetches selected pages and waits for pending reads before closing mappings. Engram offload and SSD KV prefix caching are independent settings.

## Quantization

Use the normal Web UI quantization flow, or:

```python
from omlx.oq import quantize_oq_streaming

quantize_oq_streaming(
    "/path/to/DeepSeek-V4.1-Flash",
    "/path/to/models/DeepSeek-V4.1-Flash-oQ3e-mtp",
    oq_level=3,
    enhanced=True,
    preserve_mtp=True,
)
```

Supported levels are oQ3/oQ3e and oQ4/oQ4e. Engram has a separate budget and uses affine 3-bit or 4-bit groups of 32 respectively; scales and biases add one effective bit per weight. Vision, aligner and image-marker tensors retain original precision. All tensors share the normal safetensors index with a 5 GB shard target; an individual larger tensor occupies its own shard.

The source linear weights already fit the oQ4 budget, so oQ4e preserves them without applying an imatrix. oQ3e collects calibration statistics and uses mixed affine 2/3/4/6/8-bit allocation. Unobserved expert rows use uniform importance at their allocated bit width. Modules outside calibration coverage retain source precision and count against the remaining-weight budget. `quantization_report.json` records the allocation and preserved draft tensors.

The specialized exporter requires the original checkpoint and does not support re-quantizing an oQ export, text-only export, or a different runtime dtype. Interrupted exports retain `conversion.inprogress.json` and are not treated as completed models.

## Lightning MTP and cache

Export with `preserve_mtp=True` to retain DSpark stages and heads, then enable Lightning MTP in model settings. MTP is single-stream; multiple active rows use standard decode. Quantized verification can produce different greedy sequences between MTP ON and OFF.

Window KV, compressed KV and index keys remain packed. SSD prefix caching stores compressed-row deltas and the state needed to resume at 2,048-token boundaries. Memory guards can reduce execution chunks without changing snapshot boundaries. A 9,216-token prompt stores its completed 8,192-token prefix and recomputes the remaining 1,024 tokens on reuse.

## Images, tools and limits

OpenAI image inputs use the vision encoder and aligner before batching. A real-image oQ4e smoke test correctly identified two cats, a sofa and two remote controls. Broader OCR and multi-image validation remain incomplete.

The processor uses the official conversation encoder and V4.1 spaced DSML grammar. Namespaced tools use `namespace::name` in the OpenAI adapter. Partial assistant continuation is unsupported.

Current multi-row execution evaluates rows separately. There is no 1M-context or broad quality evaluation. SSD reads, compilation and concurrent prefill can affect throughput.

## Verification

```bash
python -m pytest tests/test_deepseek_v41*.py -q
node --test tests/deepseek_v41_offload_ui.test.cjs
```

Tests use synthetic weights and small recorded official outputs; no checkpoint download is required. PyTorch is optional for independent FP8/FP4 arithmetic checks. Metal cases require the custom extension built against the active MLX and Python versions. Expected-output provenance is documented in `tests/fixtures/deepseek_v41_expected.md`.
