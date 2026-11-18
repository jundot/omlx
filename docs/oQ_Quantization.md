# oQ: oMLX Quantization

oQ turns a full-precision Hugging Face model into a smaller MLX model. The result is a standard MLX safetensors model, so it runs in oMLX and also loads in mlx-lm, mlx-vlm and other apps that read MLX models. No custom loader is needed. The one exception is [oQh](#oqh), which only oMLX can run.

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
| oQ3 | 3 | about 3.6 to 3.8 | When memory is tight |
| oQ3.5 | 3 | about 3.8 to 4.0 | oQ3 with 4-bit expert down projections |
| oQ4 | 4 | about 4.7 to 4.8 | Recommended default |
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
- After the model is written, it runs the sample prompts through the original model again, one layer at a time, and re-picks the rounded values of every group. The imatrix looks at each input channel on its own, but channels that move together can cancel each other's rounding errors. oQe moves a few percent of the values to the neighbouring level wherever that lowers the error of the layer's output. Scales and offsets stay as they are, so the model loads anywhere a plain oQe model loads.

oQe takes longer to make, but the extra work happens only once. On an M3 Ultra, Qwen3.5-9B takes about 7 minutes for the first oQe run and about 5.5 minutes after that, and Qwen3.6-35B-A3B about 10 and 7 minutes. The output-error step reads the original model again for every build, so it is not cached. If the model does not fit in memory, only the model types that oQe can stream one layer at a time get this step. The others, and DeepSeek V4.1, Muse Spark and Inkling 2, skip it and keep the plain oQe values. Block-FP8 checkpoints of those model types are always read one layer at a time, because loading the whole model converts their FP8 weights to another format. `oq_imatrix_report.json` in the output records whether it ran.

The first oQe run of a model collects the imatrix and saves it in `.oqe_imatrix` inside your model directory. Making another level of the same model, for example oQ3e after oQ4e, reuses it.

## oQh

oQh is a variant of oQe for the Qwen3.5, Qwen3.6 and Qwen3.8 dense and MoE models, Qwen3.8-Flash-Next, GLM-5.3 and DeepSeek V4.1. It gives the same file size as oQe at the same level with lower error. In Qwen3.8-Flash-Next, GLM-5.3 and DeepSeek V4.1 only the routed experts change; the other layers keep their current format and their fast kernels. DeepSeek V4.1 takes oQh below oQ4 only, because oQ4 keeps the original FP4 experts. Turn on **Rotated codes (oQh)** in the oQe section; the output name ends in `h`, for example `Qwen3.5-9B-oQ2.5h`.

oQh is an oMLX format. It needs the oQh kernels, which `make install`, `make dev` and the app build include. mlx-lm, mlx-vlm and other MLX apps cannot load it, and Splash, A8 prefill, ANE prefill and MoE expert offload do not run it. The Prefill Accelerator runs oQh models on Mac helpers and on CUDA helpers that list the model's family as oQh capable (Qwen3.5 / 3.6 / 3.8 dense, Qwen3.8 Flash-Next, GLM-5.3 and DeepSeek V4.1). In a cluster, GLM-5.3 and DeepSeek V4.1 oQh models run tensor parallel on up to two nodes; other oQh models do not run tensor parallel.

How it differs from oQe:

- Affine quantization stores a scale and an offset for every group of 64 weights, about 0.5 bits per weight. oQh rotates the input of each decoder linear layer with a signed Walsh-Hadamard transform and stores the weights in that rotated basis as integer codes with one scale per output row.
- At the same bit width, one scale per row rounds less accurately than group scales. The bytes oQh saves buy one more bit, first for the tensors with the fewest bits, until the file is the size of the oQe plan.
- From oQ4 up, the Qwen models and GLM-5.3 keep a scale and an offset for every 128 rotated weights instead of the row scale, and run on the regular affine kernels. At 4 bits that measured about 9% lower KL than the row scale at the same size; at 2 and 3 bits the row scale is better. DeepSeek V4.1 takes oQh below oQ4 only, so it always keeps the row scale.
- The rounded values are then re-picked against each layer's output error, as in oQe.
- The norm and activation kernels write the rotated input next to their normal output, so the rotation adds almost no work at run time.

| Model | Level | Size | oQe KL | oQh KL |
|---|---|---|---|---|
| Qwen3.5-4B | oQ3.5 | 2.1 GB | 0.091 | 0.079 (-13%) |
| Qwen3.5-9B | oQ2.5 | 3.8 GB | 0.251 | 0.115 (-54%) |
| Qwen3.5-9B | oQ3.5 | 4.5 GB | 0.082 | 0.057 (-30%) |
| Qwen3.6-35B-A3B | oQ3.5 | 17.5 GB | 0.034 | 0.023 (-32%) |

KL divergence from the original model on general text (FLORES), lower is better. oQe and oQh files of the same level have the same size. The 9B and 35B rows compare the decoder layers only, with the embeddings and the LM head kept in BF16.

Qwen3.8-Flash-Next oQ3.5 (95 GB, routed experts only) against its FP8 original on English prose: KL 0.167 for oQe and 0.127 for oQh (-24%), with similar gains on code and multilingual text.

## What oQ does

1. **Reads the model from disk** one tensor at a time instead of loading the whole model.
2. **Measures layer sensitivity.** It runs built-in calibration prompts through the model and compares each layer's quantized output with the original.
3. **Plans the bits.** It gives extra bits to the most sensitive tensors until the level's size budget is used.
4. **Collects the imatrix** (oQe only).
5. **Quantizes and saves** each tensor.
6. **Re-picks the rounded values against each layer's output error** (oQe only), then writes the config, tokenizer and chat template next to the weights.

### What gets protected

- **Output head and embeddings:** 8-bit when the size budget allows.
- **MoE routers:** kept at full precision. Shared-expert gates are 8-bit.
- **Vision and audio encoders:** not quantized, and not counted in the size budget.
- **Norms and recurrent state parameters** (SSM and linear attention): not quantized.
- **Attention and other sensitive projections:** extra bits where the measurement says so. In dense models that mix full attention with sliding or linear attention, the full-attention projections get one bit above the base from oQ3 up.
- **Routed experts:** stay at the base bits, because they are most of the model and the most expensive to raise. oQ3.5 gives their down projections one extra bit, and oQ2.5 and oQ2.7 raise whole expert layers to 3-bit.
- **MTP heads** (with **Preserve MTP weights**): kept, with the small connecting layers at full precision and the rest at the base bits and at least 4-bit. The head is planned after the rest of the model, so the other layers get the same bits as in a build without the head. Budget left under the size cap raises the head further.

Every model ends up with its own bit layout, because the plan follows that model's measurements.

### Calibration data

Both data sets ship with oMLX, so nothing is downloaded.

- **Sensitivity:** 600 samples of code, English, Korean, Chinese and Japanese text, tool calling and reasoning. oQ uses 128 sequences of 256 tokens.
- **oQe imatrix:** about 2,700 samples covering tool calling, chat, reasoning, code and English, Korean, Chinese and Japanese text. oQe uses 128 sequences of 512 tokens. For MoE models it keeps sampling, up to 1,024 sequences, until every expert has seen enough tokens.
- **oQe rounding refinement:** 176 sequences of 512 tokens from the imatrix data, plus 16 sequences each of German, French and Spanish Wikipedia articles and 16 more code sequences, 240 in total. Without the European text the refinement learned channel patterns that did not hold for those languages.

Sample selection uses a fixed seed, so repeated runs on the same model see the same calibration text.

## Options

| Option | What it does |
|---|---|
| oQ Level | Size and quality level, see the table above |
| Enhanced quantization (oQe) | Uses the imatrix when rounding. Recommended |
| Text Only | Leaves out the vision encoder of a vision-language model and writes a text-only model |
| Preserve MTP weights | Keeps the MTP heads so Lightning MTP works after quantization. Adds `-mtp` to the name |
| Keep original Engram tables | DeepSeek V4.1 only. Copies the Engram tables in their original FP8 storage instead of 4-bit. The output is larger on disk, and memory use with SSD N-gram Offload stays the same. Adds `-ngram` to the name |
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

- **DeepSeek V4.1:** oQ3, oQ3e, oQ4 and oQ4e with bfloat16 output, from the original checkpoint. oQ4 keeps the original FP4/FP8 projection precision and quantizes the Engram tables to 4-bit unless Keep original Engram tables is on.
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

Muse Spark Open output uses the checkpoint layout of mlx-lm's own Muse Spark model (mlx-lm 0.32.2 and later). That layout stores q, k, v and the attention output gate as one projection, so oQ gives the four one bit width per layer. oQ also keeps the LM head in BF16, because mlx-lm's model gives wrong logits with a quantized LM head.

Muse Spark Open NVFP4 checkpoints keep their NVFP4 routed experts unchanged at oQ4 and above, together with the per-expert tensor scales. oQ quantizes only the other BF16 tensors (attention, the first and last MoE layers, and embeddings). At oQ3 and below, oQ requantizes the routed experts and folds each expert's tensor scale into its weights, so the output is a plain affine checkpoint that mlx-lm also loads.

## Acknowledgments

The importance matrix follows the imatrix approach used by llama.cpp. The least-squares refit was prompted by the 3-bit quantizer in @tacos8me's m5-ultra project. The rounding refinement follows the Flip-Refinement step of [MACaQ](https://arxiv.org/abs/2611.10649) (Huawei Computing Systems Lab).
