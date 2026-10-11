# oQ: oMLX Quantization

oQ turns a full-precision Hugging Face model into a smaller MLX model. The result is a standard MLX safetensors model, so it runs in oMLX and also loads in mlx-lm, mlx-vlm and other apps that read MLX models. No custom loader is needed.

oQ is mixed precision. Before writing anything it measures how much each layer suffers from quantization, then spends extra bits where they help most while keeping the total size close to the level you picked.

## Quick start

1. Open the admin dashboard and go to **Models > oQ Quantization**.
2. Pick a **Source Model** and an **oQ Level**, and turn on **Enhanced quantization (oQe)**.
3. Press **Start**. The dashboard shows the estimated size and memory before you start, and the queue shows progress.

The new model is saved in your model directory with the level in its name, for example `Qwen3.5-9B-oQ4e`, and appears in the model list when it finishes. Quantization runs one model at a time.

For most models, **oQ4e** is the place to start.

## Choosing a level

| Level | Base bits | Size (bits per weight) | When to use |
|---|---|---|---|
| oQ2 | 2 | about 2.8 to 3.0 | Very large MoE models that do not fit otherwise |
| oQ2.5 | 2 | about 3.1 to 3.3 | Large MoE models, with some expert layers lifted to 3-bit |
| oQ2.7 | 2 | about 3.25 to 3.35 | Same as oQ2.5 with more 3-bit expert layers |
| oQ3 | 3 | about 3.5 to 3.7 | When memory is tight |
| oQ3.5 | 3 | about 3.8 to 4.0 | oQ3 with 4-bit expert down projections |
| oQ4 | 4 | about 4.6 to 4.7 | Recommended default |
| oQ5 | 5 | about 5.5 to 5.7 | Higher quality |
| oQ6 | 6 | about 6.5 to 6.7 | Very close to the original |
| oQ8 | 8 | about 8.5 | Every quantized tensor at 8-bit |

Bits per weight includes the small scale and offset stored with every group of 64 weights. To estimate the file size, multiply the parameter count by bits per weight and divide by 8. A 9B model at oQ4 comes out at about 5.6 GB.

The 2-bit levels are meant for large MoE models, where most weights belong to experts that only a few tokens use at a time. Dense models lose much more quality at 2 bits, so start dense models at oQ3e or higher.

## oQ and oQe

Both use the same bit plan and produce models of the same size and speed. The difference is how carefully each tensor is rounded.

**oQ** rounds each group of weights with the standard MLX method, which spreads the available levels evenly between the group's smallest and largest value.

**oQe** first runs sample prompts through the model and records which input channels carry the most signal. This is called an importance matrix, or imatrix. When it rounds the weights, it keeps the important channels more accurate:

- For each group it searches for the scale and offset that give the smallest importance-weighted error.
- It then refits the scale and offset by weighted least squares, repeating a few times, and keeps the refit only when it lowers the error.

oQe takes longer to make, but the extra work happens only once. On an M3 Ultra, Qwen3.5-9B takes about 2.5 minutes for the first oQe run and about 1 minute after that, and Qwen3.6-35B-A3B about 5 and 2.5 minutes.

The first oQe run of a model collects the imatrix and saves it in `.oqe_imatrix` inside your model directory. Making another level of the same model, for example oQ3e after oQ4e, reuses it.

## What oQ does

1. **Reads the model from disk** one tensor at a time instead of loading the whole model.
2. **Measures layer sensitivity.** It runs built-in calibration prompts through the model and compares each layer's quantized output with the original.
3. **Plans the bits.** It gives extra bits to the most sensitive tensors until the level's size budget is used.
4. **Collects the imatrix** (oQe only).
5. **Quantizes and saves** each tensor, then writes the config, tokenizer and chat template next to the weights.

### What gets protected

- **Output head and embeddings:** 8-bit when the size budget allows.
- **MoE routers:** kept at full precision. Shared-expert gates are 8-bit.
- **Vision and audio encoders:** not quantized.
- **Norms and recurrent state parameters** (SSM and linear attention): not quantized.
- **Attention and other sensitive projections:** extra bits where the measurement says so.
- **Routed experts:** stay at the base bits, because they are most of the model and the most expensive to raise. oQ3.5 gives their down projections one extra bit, and oQ2.5 and oQ2.7 raise whole expert layers to 3-bit.
- **MTP heads** (with **Preserve MTP weights**): kept, with the small connecting layers at full precision and the rest at 4-bit or more.

Every model ends up with its own bit layout, because the plan follows that model's measurements.

### Calibration data

Both data sets ship with oMLX, so nothing is downloaded.

- **Sensitivity:** 600 samples of code, English, Korean, Chinese and Japanese text, tool calling and reasoning. oQ uses 128 sequences of 256 tokens.
- **oQe imatrix:** about 2,700 samples covering tool calling, chat, reasoning, code and English, Korean, Chinese and Japanese text. oQe uses 128 sequences of 512 tokens. For MoE models it keeps sampling, up to 1,024 sequences, until every expert has seen enough tokens.

Sample selection uses a fixed seed, so repeated runs on the same model see the same calibration text.

## Options

| Option | What it does |
|---|---|
| oQ Level | Size and quality level, see the table above |
| Enhanced quantization (oQe) | Uses the imatrix when rounding. Recommended |
| Text Only | Leaves out the vision encoder of a vision-language model and writes a text-only model |
| Preserve MTP weights | Keeps the MTP heads so Lightning MTP works after quantization. Adds `-mtp` to the name |
| Combine other model's MTP head | Grafts an MTP head from another checkpoint of the same architecture, or a Gemma 4 assistant model, into the output |
| Non-quant weight dtype | bfloat16 (default) or float16 for unquantized weights and quantization scales. float16 adds `-fp16` to the name |
| Sensitivity Model | Measures sensitivity on an already quantized copy of the model to use about 4x less memory |
| Reuse imatrix cache, Imatrix cache path | Control where the imatrix is stored and whether a compatible one is reused |
| Calibration samples, Sequence length | How much text oQe runs for the imatrix |
| Strict imatrix coverage | Stops with an error when a tensor has no imatrix entry, instead of quantizing that tensor with standard oQ |

## Memory and disk

Writing the quantized model needs little memory because tensors are read and written one at a time. Calibration needs the model in memory:

- If the model does not fit in about 75% of the available memory, oQ makes a temporary 4-bit copy on disk, calibrates on that copy and deletes it afterwards. This needs extra free disk space while it runs.
- MiniMax M3 and Qwen4-Exp models calibrate one layer at a time instead.

## Supported models

oQ works with models that mlx-lm or mlx-vlm can load, including MoE and vision-language models. The source must be the original checkpoint:

- BF16 or FP16 weights.
- FP8 or MXFP8 weights. oQ reads them with their original scales, and keeps a tensor at its source precision when the plan would not lower it.
- Gemma 4 QAT checkpoints.

Models that are already MLX-quantized cannot be used as a source.

Some models have their own rules:

- **DeepSeek V4.1:** oQ3, oQ3e, oQ4 and oQ4e with bfloat16 output, from the original checkpoint. oQ4 keeps the original FP4/FP8 projection precision and quantizes the Engram tables to 4-bit.
- **DeepSeek V4:** float16 output is not supported.

## Benchmarks

All results use greedy decoding with thinking off. Scores are correct answers over all questions, so each question counts once.

### oQ and oQe compared with uniform 4-bit

Measured when oQe was introduced ([#2057](https://github.com/jundot/omlx/pull/2057)), before the least-squares refit. MMLU 1000, Winogrande 300 and MBPP 300, 1,600 questions in total.

| Model | Original | mlx-lm 4-bit | oQ4 | oQ4e |
|---|---|---|---|---|
| gemma-4-26B-A4B-it | 82.75 (51.6 GB) | 79.88 (15.4 GB) | 80.06 (15.8 GB) | 81.62 (15.8 GB) |
| gemma-4-31B-it | 86.44 (62.6 GB) | 85.38 (18.4 GB) | 85.56 (19.0 GB) | 85.50 (19.0 GB) |
| Qwen3.5-9B | 74.25 (19.3 GB) | 70.12 (6.0 GB) | 70.31 (6.1 GB) | 72.38 (6.1 GB) |
| Qwen3.6-35B-A3B | 80.44 (71.9 GB) | 81.00 (19.5 GB) | 80.75 (21.1 GB) | 81.00 (21.1 GB) |
| Qwen3.6-27B | 85.50 (55.6 GB) | 84.25 (16.1 GB) | 84.50 (16.7 GB) | 84.81 (16.7 GB) |

### oQe with the least-squares refit

Measured with the refit added in [#4385](https://github.com/jundot/omlx/pull/4385), at batch size 32. MMLU 1000, Winogrande 300, MBPP 300 and GSM8K 300, 1,900 questions in total.

| Model | Original | oQ3e before | oQ3e now | oQ4e before | oQ4e now |
|---|---|---|---|---|---|
| Qwen3.5-9B | 76.53 (18 GB) | 72.53 | 73.79 (4.6 GB) | 75.11 | 77.05 (5.6 GB) |
| Qwen3.6-35B-A3B | 82.68 (67 GB) | 81.58 | 82.68 (16 GB) | 82.79 | 83.05 (20 GB) |

The refit also brings the output distribution closer to the original model's: the KL divergence from the original drops by 12% to 15% on general text for both models at both levels.

## Acknowledgments

The importance matrix follows the imatrix approach used by llama.cpp. The least-squares refit was prompted by the 3-bit quantizer in @tacos8me's m5-ultra project.
