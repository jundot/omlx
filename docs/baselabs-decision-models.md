# Mammoth-1 and Elephant-1

oMLX serves Base Labs' [Mammoth-1](https://huggingface.co/baselabs/mammoth-1)
and [Elephant-1](https://huggingface.co/baselabs/elephant-1) through
`POST /v1/systemone`. The Qwen3.5 backbone reads the state once and scores
each question from a copy of its cache. It returns probabilities directly,
without generating answer tokens.

## MLX checkpoints

Existing text-only MLX conversions retain the required decision metadata:

| Model | MLX repository | Revision |
|---|---|---|
| Mammoth-1, 8-bit | [midium-ai/mammoth-1-8bit](https://huggingface.co/midium-ai/mammoth-1-8bit) | `4729fa023eabe29d7bb956131dc560a42bde5943` |
| Elephant-1, 8-bit | [midium-ai/elephant-1-8bit](https://huggingface.co/midium-ai/elephant-1-8bit) | `68917ae4280a2218d3e0dbcd360e38eddeb3dd95` |

Download a checkpoint into the configured model directory, including
`config.json`, `readout.json`, tokenizer files and all weight shards. oMLX
detects the decision format from its metadata even if its directory is
renamed. These two conversions omit the vision tower: image requests return
400. A checkpoint with the complete vision tower and processor can use the
same adapter with `images` data URIs and `<image:1>`, `<image:2>`, ...
placeholders appearing once each, in order, inside `state`. Video is unsupported.

## Request

The `model` is the directory/model ID shown by `GET /v1/models`.

```json
{
  "model": "elephant-1-8bit",
  "state": {"ticket": "I was charged twice. Please refund the extra charge."},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}
    },
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    "urgency": {"type": "score", "criteria": ["Can wait", "Today", "Blocking now"]}
  }
}
```

Choice supports 1–255 options; Noul returns P(yes); Score supports 1–10 levels
and returns the expected zero-based level and per-level probabilities. Criteria
preserve their input order, and question IDs do not enter the prompt. Requests
exceeding 16,384 tokens for a branch or 65,536 tokens across the shared prefix and suffixes
return 413. Evidence is never silently truncated, including with `truncate: true`.

## Inference and validation

The adapter follows [OneJev at 81ce62f](https://github.com/OmniJev/OneJev/tree/81ce62f1597c91e46767d4d02d6ac2e18534fe94)
with the `serve.patch` and `slot_lm` settings published with the Base Labs
weights. Full-attention layers read bidirectionally within each block;
linear-attention layers retain their causal recurrence. The state never sees
the questions, and each question never sees another question. The readout
uses the 256 released slot-label rows of the output projection in FP32.
Optional `calibration.json` temperatures are applied by question type/count.
Noncausal blocks yield between groups of layers to let other engines use the GPU.

Unit tests compare cached block execution against an independent full-prompt
block mask on tiny text and vision backbones. Opt-in integration tests use
local MLX weights and can compare against `decision_mlx.py` from
[0x36's independently released MLX port](https://huggingface.co/0x36/elephant-1-9B-MLX-4bit/blob/16de0e500eea4e44fc7db79928c03c53e324bcb0/decision_mlx.py).
They never download weights or execute remotely fetched code automatically.

```bash
OMLX_BASELABS_MODEL_PATH=/absolute/path/to/local/elephant-1-8bit \
OMLX_BASELABS_REFERENCE_PATH=/absolute/path/to/decision_mlx.py \
python -m pytest tests/integration/test_baselabs_real_model.py -m slow -s -q
```

That parity check uses the same quantized weights on both implementations.
It does not measure quantization loss against the original bf16 checkpoint
or reproduce the authors' decision benchmarks. Real-image validation requires
a checkpoint containing the vision tower; the listed text-only conversions
cannot provide it.

Validation on an M3 Max (64 GiB), 23 October 2026: both pinned 8-bit
checkpoints passed the two opt-in tests. Eight questions per checkpoint
matched the independent implementation with a maximum probability difference
of 0.000050, including a 255-option choice, ten score levels, structured
criteria and multilingual state. The native HTTP smoke check also covered
both decision endpoints, rejection of chat/image requests for the text-only
conversion, oversized-input errors, the advertised 16K limit and unloading.
