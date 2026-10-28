# LFM3 models

Native MLX support for `LiquidAI/LFM3-1.7B` and
`LiquidAI/LFM3-VL-2B`. These checkpoints require Hugging Face access.
Loading and inference use the native implementation and do not require
`trust_remote_code=True` or PyTorch. Checkpoint licenses continue to apply to
the weights. The implementation adapts the Apache-2.0 reference code supplied
with the vision checkpoint; see [LICENSE](LICENSE).

## Usage

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("LiquidAI/LFM3-1.7B")
prompt = apply_chat_template(
    processor, model.config, "What is 17 + 25?",
    enable_thinking=False,
)
print(generate(model, processor, prompt, max_tokens=128, temperature=0).text)

model, processor = load("LiquidAI/LFM3-VL-2B")
prompt = apply_chat_template(
    processor, model.config, "Describe this image.",
    num_images=1, enable_thinking=False,
)
print(generate(
    model, processor, prompt, image=["image.png"],
    max_tokens=128, temperature=0,
).text)
```

`apply_chat_template` and the command line default to `enable_thinking=False`;
the checkpoints' templates and model cards default to thinking on. Pass
`enable_thinking=True` (`--enable-thinking`) to match them; the LFM3-1.7B
card warns that disabling thinking "can be inconsistent, especially for
reasoning-heavy tasks". Sampling options default to the checkpoint's
`generation_config.json` unless explicitly overridden. Its configuration of LFM3-1.7B and LFM3-VL-2B sets `temperature=0.1`,
`top_k=50` and `repetition_penalty=1.1`; the LFM3-1.7B model card recommends
`temperature=1.0`, `top_p=0.95`, `top_k=20` for math. On the command line,
`top_p` and `top_k` are set with `--gen-kwargs`.

LFM3-1.7B writes tool calls as `<tool_call><function=...>` XML, which the
`qwen3_coder` parser reads (picked automatically from the template). The
template expects tool-call arguments as objects; JSON-string arguments are
parsed before rendering.

`LiquidAI/d1-2B` shares the LFM3-VL `model_type` but is a decision
model: it loads as `d1_lfm3_*` and answers through `model.system_one(...)`
or `python -m mlx_vlm.decide`, not `generate`; see
[d1 on MLX](../d1_lfm3_vl/README.md).

The vision model accepts multiple images and pre-sampled video frames. The
standard generation API decodes video paths and supplies the sampled FPS;
direct processor calls should supply `fps` or `video_metadata` to preserve
timestamps. Images use the checkpoint's resolution budget, with optional
`min_pixels` and `max_pixels` processor overrides. NumPy/PIL preprocessing
avoids a PyTorch runtime dependency. Resized pixels differ from the reference
torchvision bicubic resize by up to 2 8-bit levels on most test images (16 on
a noisy synthetic image); token ids and grids match the reference in every
image and video case tested.

Standard conversion quantizes the language backbone and retains the vision
tower's original precision. For LFM3-1.7B on four texts, 8 bits (group size
64) changes float32 perplexity by -0.25% to +0.38%, about as much as bf16
(-0.15% to +0.50%), and 32 greedy tokens match float32 on 7 of 9 prompts.
4 bits (affine, group size 64) raises perplexity by 6% to 63%, matches on 3
of 9 prompts, and answered one tool-call prompt in text instead of calling
the tool:

```sh
python -m mlx_vlm convert \
  --hf-path LiquidAI/LFM3-VL-2B \
  --mlx-path ./LFM3-VL-2B-4bit \
  -q --q-bits 4 --q-group-size 64
```

## Architecture and validation

The shared text backbone combines gated short convolutions, full attention,
and sliding attention. Full attention uses partial RoPE across matching pairs
in the two head halves; sliding attention rotates the entire head. Attention
has a sigmoid gate per query head. Sandwich normalization is checkpoint
configurable. Sliding caches retain their absolute rotary positions after
wrapping, including merged batches with different prompt lengths.

The vision tower uses a linear temporal patchifier, gated spatial depthwise
convolutions, and frame-local bidirectional attention with axial RoPE. The
projector normalizes each patch before merging spatial blocks. Weight layout
conversion is idempotent for both original and converted checkpoints.

Validated on an Apple M3 Ultra with MLX 0.32.4 and Transformers 5.20.0:

| Checkpoint | Float32 last-logit relative error |
| --- | --- |
| `LiquidAI/LFM3-1.7B` | `3.0e-6` to `9.6e-6` (max abs `1.32e-4`) |
| `LiquidAI/LFM3-VL-2B` | `2.9e-6` to `2.95e-5` (max abs `2.46e-4`) |

These errors compare final prompt logits against the checkpoints' PyTorch
code using identical input tensors, with the same top-20 tokens in order.
32 greedy tokens match the reference on 9 of 9 text prompts, including a
4826-token prompt past the 256-token sliding window at prefill chunk sizes
64, 300 and 2048. Tiny models also match all reference logits within `4e-7`.
BF16 inference answered "The capital of Australia is Canberra." and read
`CODE 5739` next to a red square and a blue circle. These are smoke checks,
not a comprehensive quality benchmark.

Regression coverage lives in `test_models.py`, `test_processors.py`, and
`test_model_ops.py` under `mlx_vlm/tests`, with tiny configs in `model_cases.json`.
It includes mixed caches, sliding-window wraparound, batch offsets, partial
rotary coordinates, frame separation, placeholder counts, and quantized
checkpoint round trips.

LFM3-Omni is documented in [its README](../lfm3_omni/README.md).
