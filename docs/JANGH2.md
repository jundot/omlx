# JANGH2 models

Packed JANGH2 Qwen4-Exp checkpoints, including Allosaurus-v0.1-125B-A6B-JANGH2,
load through the normal oMLX VLM engine. Discovery recognizes the checkpoint's
JANGH2 metadata and selects the packed Metal expert implementation automatically.

Place the checkpoint beneath your model directory and start oMLX as usual:

```sh
omlx serve --model-dir /path/to/models
```

Use the ordinary model picker or the checkpoint's model ID in chat requests.
Continuous batching, streaming, reasoning parsing, cache management and model
unload use the existing engine. The backend needs no bridge launcher, runtime
monkeypatches, environment variables or machine-specific tuning profile.

## Loading and memory

The loader validates tensor names, shapes, quantization metadata and the declared
norm convention before loading. It retains the packed experts and applies affine
quantization only to the remaining eligible modules. Checkpoint files are read
without rewriting them.

The large PLE embedding table stays in read-only SSD mappings. Only requested
rows are copied and dequantized; the table is excluded from GPU parameters and
model memory admission uses the resident-weight estimate. Keep the checkpoint
on a fast local SSD. Mappings are owned by the model and released on unload or
load failure.

Packed-expert MTP and MoE expert offload are currently unsupported. oMLX reports
MTP as unavailable for this format and does not automatically enable it. Old
saved MTP settings are disabled in the effective runtime settings without
rewriting the saved configuration. Ordinary Qwen models retain their existing
MTP behavior.

The native backend uses conservative packed-kernel defaults. Grouped prefill and
device PLE caching remain diagnostic options pending separate qualification.

## Optional thinking

Allosaurus uses the existing model **Thinking Mode** setting and per-request
`chat_template_kwargs.enable_thinking`; it needs no JANGH2-specific thinking
setting. **Auto** leaves the choice to the checkpoint's chat template. The
Allosaurus template enables thinking by default.

To request a direct answer, include this in an OpenAI-compatible chat request:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

To enable thinking with a shorter reasoning effort:

```json
"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "low"}
```

The checkpoint accepts `low`, `medium`, and `xhigh` reasoning effort. Request
kwargs override the model's defaults unless the administrator has forced that
key. An explicit `enable_thinking: false` also takes precedence over automatic
activation from a thinking budget. Both streaming and non-streaming responses
use the normal reasoning/content channels. Historical assistant reasoning stays
in `reasoning_content` so the model's template renders one thinking block.

## Validation and diagnostics

On an Apple M5 Max, three fixed-prefix and cached-continuation comparisons
matched the FP32 expert reference exactly at all 36 sampled logit positions.
The ordinary HTTP server also passed direct answers, concurrent requests,
thinking-enabled streaming, model unload/reload and shutdown. These checks cover text generation;
they do not establish image-input or HTTP tool-calling acceptance for this
checkpoint. See the [verification record](../contrib/jangh2/VERIFICATION.md).

The optional template contract tests load only the local tokenizer:

```sh
OMLX_TEST_QWEN4_TEMPLATE_MODEL=/path/to/checkpoint \
  python -m pytest tests/test_qwen4_thinking_template.py -q
```

The [diagnostic package](../contrib/jangh2/README.md) provides standalone kernel
qualification, numerical comparisons, performance profiling and a legacy PR
#364 serving adapter. Its compatibility modules share the production numerical
implementation. Changing that implementation invalidates old diagnostic profile
fingerprints; regenerate reports instead of editing their fingerprints. These
tools are optional for ordinary native serving.
