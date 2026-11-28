---
license: apache-2.0
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash
tags:
- speculative-decoding
- dfly2
- angelspec
---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Tencent/AngelSpec/blob/main/docs/_static/logo_banner_white.png?raw=true">
    <img alt="AngelSpec" src="https://github.com/Tencent/AngelSpec/blob/main/docs/_static/logo_banner.png?raw=true" width="420">
  </picture>
</p>

<h3 align="center">
A unified training framework for MTP and block-parallel speculative decoding.
</h3>

<p align="center">
          ✒️ <a href="https://arxiv.org/abs/2607.25852">Technical Report</a>&nbsp&nbsp | &nbsp&nbsp📄 <a href="https://arxiv.org/abs/2611.34776">DFly2 Paper</a>&nbsp&nbsp | &nbsp&nbsp📖 <a href="https://angelspec.readthedocs.io">Documentation</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/AngelSlim/angelspec">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp💻 <a href="https://github.com/Tencent/AngelSpec">GitHub</a>
<br>
</p>

# DeepSeek-V4.1-Flash DFly2 Drafter (Block Size 6)

## Model Summary

This repository contains a **DFly2** speculative-decoding drafter for **DeepSeek-V4.1-Flash**, trained with **AngelSpec**. DFly2 is a block-parallel drafter for natively sparse targets: each of its three draft layers is paired with one of the target's DeepSeek Sparse Attention (DSA) layers and reads the KV that layer has already cached (**KV sharing**), at the positions that layer's indexer has already selected (**index sharing**). The drafter keeps no KV cache of its own, runs no indexer, and reads a fixed number of positions from anywhere in the context.

It should be deployed together with the DeepSeek-V4.1-Flash target model.

- **Target model:** [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- **Drafter type:** DFly2
- **Block size:** 6 (one anchor token + 5 draft tokens)
- **Training framework:** AngelSpec
- **Serving backend:** vLLM speculative decoding (support in progress)

<p align="center">
  <img src="https://github.com/Tencent/AngelSpec/blob/main/docs/_static/dfly2.png?raw=true" alt="DFly2 overview: each draft layer reads the KV cache and the indexer top-k positions of a paired DSA layer of the target" width="75%">
</p>

## Highlights

- **No KV cache of its own.** A three-layer drafter that kept a full cache of its own would add 3.4 KiB per token on top of the target's 4.4 KiB, raising the per-token KV to 1.76×. DFly2 adds none, so the same KV pool holds **1.73×** as many requests at 32K tokens. It needs no place on the prefill instances under prefill-decode disaggregation, and every prefix-cache hit of the target is also a drafter hit.
- **Fixed-size, query-aware reads.** Each draft step reads the positions the target's own sparse indexer selected for the token before the block. They can lie anywhere in the context, their number does not grow with it, and the drafter runs no indexer of its own. Made one step early, this selection still covers 0.873–0.880 of the top-k mass that each later token in the block would select, against 0.147–0.175 for a window of the same size.
- **Constant drafting cost.** A fixed read of 128 to 2K positions costs 4%–6% of a decode step at every context length, while reading a full cache of its own climbs to 14%–23% at 128K.
- **Higher acceptance on every benchmark.** Against DFlash and DSpark trained on the same recipe and data, DFly2 accepts more tokens per step on every benchmark below, at both temperatures.
- **Generalizes past its 32K training length.** On the agentic sets, whose prompts mostly run longer, its average accepted length stays close to its general-purpose average (4.06 / 3.75 vs. 4.12 / 3.88 at T = 0 / 1), while DSpark's drops from 3.81 / 3.51 to 3.35 / 3.08.
- **Faster at high concurrency.** At matched per-user decode speed, DFly2 delivers 28% more throughput than DSpark on short text, and 62% and 81% more on CodeSpan 64K and SWE 64K.

## Model Details

DeepSeek-V4.1-Flash stores a compressed latent in four of its 40 layers and shares it across the others. DFly2 pairs with three layers that each run an indexer of their own, so every draft layer reads a different view of the context.

| Item | Value |
|---|---|
| Draft layers | 3 MoE layers, 128 routed experts each |
| Paired target layers (KV + index sharing) | 28, 32, 36 |
| Context read per draft step | Target's top-512 selection at the row before the anchor, plus its 128-token near window (646 keys per step with the 6-token block, at any context length) |
| Initialization | Attention copied from the paired target layer; 128 of that layer's routed experts drawn at random, with their router rows and biases |
| Last-layer hidden injection | Target's last-layer hidden state, fused with the anchor embedding and gated into the anchor slot |
| Intra-block correction | Hidden correction plus a context attention and a gated prefix attention, slot by slot |
| Embedding / LM head | Shared with the target (not stored in this repository) |
| Precision | FP4 routed experts, BF16 elsewhere |

## Training Data

The drafter is trained on **DFly-SFT**, a general-purpose instruction-tuning corpus of **989,861 conversations and 7.42B tokens**, all from public datasets. The conversations fall into ten capability domains, labeled by their source where it gives a label and by an LLM annotator otherwise. Samples are capped at 32,768 tokens and each domain is filled with its longest samples first, which makes the mixture long-form and reasoning-heavy: 61.6% of samples carry an explicit reasoning trace and 34.9% run past 8K tokens. The corpus is deduplicated globally and decontaminated against the twelve DeepSpec evaluation sets with an n-gram overlap test.

| Data type | Mix | Samples | Tokens | Median len. | Think |
|---|---:|---:|---:|---:|---:|
| Math (advanced) | 19.2% | 190,000 | 2.90B | 13,982 | 98.6% |
| Code (competitive) | 11.1% | 110,000 | 1.96B | 18,175 | 94.7% |
| Code (engineering) | 9.6% | 95,479 | 0.56B | 4,064 | 46.9% |
| Knowledge & STEM | 15.2% | 150,000 | 0.48B | 2,075 | 51.3% |
| Instruction following | 7.1% | 70,000 | 0.47B | 5,751 | 85.7% |
| Code (function-level) | 11.1% | 110,000 | 0.41B | 686 | 37.9% |
| Writing | 10.1% | 100,000 | 0.24B | 1,425 | 40.1% |
| Text transformation | 6.1% | 60,000 | 0.19B | 1,972 | 37.4% |
| Logical reasoning | 2.5% | 24,382 | 0.11B | 1,461 | 65.0% |
| Math (word problems) | 8.1% | 80,000 | 0.09B | 477 | 20.2% |
| **Total** | **100.0%** | **989,861** | **7.42B** | **3,621** | **61.6%** |

*Mix* is the share of samples. Token counts and lengths use the GLM-5.3-Flash tokenizer, and *Think* is the fraction of samples whose assistant turns carry an explicit reasoning trace.

<details>
<summary>Per-source composition</summary>

Each domain's share is given on its first row, and *Mix* is each source's share of the full 989,861-sample corpus.

| Data type | Dataset | Mix |
|---|---|---:|
| Math (advanced), 19.2% | OpenMathReasoning | 9.68% |
|  | Dolci-Think-SFT-32B | 3.29% |
|  | Step-3.5-Flash-SFT | 2.02% |
|  | Nemotron-Math-v4 | 2.02% |
|  | Nemotron-Math-v2 | 1.01% |
|  | SYNTHETIC-2 (math) | 0.81% |
|  | WildChat-4.8M | 0.20% |
|  | Nemotron-Chat-v3 | 0.10% |
|  | Arena-Human-Pref-140k | 0.06% |
| Code (competitive), 11.1% | Nemotron-CompProg (Python) | 4.80% |
|  | Dolci-Think-SFT-32B | 1.91% |
|  | CodeForces-CoTs | 1.62% |
|  | Step-3.5-Flash-SFT | 1.41% |
|  | SYNTHETIC-2 (code) | 0.71% |
|  | Nemotron-CompProg (C++) | 0.51% |
|  | WildChat-4.8M | 0.16% |
| Code (engineering), 9.6% | WildChat-4.8M | 4.74% |
|  | Step-3.5-Flash-SFT | 3.96% |
|  | Nemotron-Chat-v3 | 0.59% |
|  | Arena-Human-Pref-140k | 0.36% |
| Knowledge & STEM, 15.2% | Step-3.5-Flash-SFT | 5.05% |
|  | WildChat-4.8M | 4.83% |
|  | Arena-Human-Pref-140k | 2.53% |
|  | Nemotron-Chat-v3 | 2.02% |
|  | Dolci-Think-SFT-32B | 0.72% |
| Instruction following, 7.1% | Dolci-Think-SFT-32B | 3.54% |
|  | Nemotron-IF-v3 | 1.21% |
|  | Step-3.5-Flash-SFT | 0.81% |
|  | SYNTHETIC-2 (format) | 0.51% |
|  | WildChat-4.8M | 0.51% |
|  | AutoIF-61k | 0.51% |
| Code (function-level), 11.1% | OpenCodeInstruct | 6.30% |
|  | Dolci-Think-SFT-32B | 3.20% |
|  | Step-3.5-Flash-SFT | 0.90% |
|  | WildChat-4.8M | 0.56% |
|  | Nemotron-Chat-v3 | 0.13% |
|  | Arena-Human-Pref-140k | 0.03% |
| Writing, 10.1% | WildChat-4.8M | 5.59% |
|  | Step-3.5-Flash-SFT | 3.00% |
|  | Nemotron-Chat-v3 | 1.08% |
|  | Arena-Human-Pref-140k | 0.25% |
|  | Charcard-Roleplay | 0.18% |
| Text transformation, 6.1% | WildChat-4.8M | 3.59% |
|  | Step-3.5-Flash-SFT | 1.82% |
|  | Nemotron-Chat-v3 | 0.45% |
|  | Arena-Human-Pref-140k | 0.20% |
| Logical reasoning, 2.5% | UltraInteract-SFT | 0.62% |
|  | SYNTHETIC-2 (output-pred.) | 0.54% |
|  | R1-Brain-Teasers | 0.38% |
|  | Step-3.5-Flash-SFT | 0.29% |
|  | SYNTHETIC-2 (reasoning-gym) | 0.24% |
|  | Nemotron-Chat-v3 | 0.22% |
|  | WildChat-4.8M | 0.13% |
|  | Arena-Human-Pref-140k | 0.04% |
| Math (word problems), 8.1% | Orca-Math-200k | 5.78% |
|  | Step-3.5-Flash-SFT | 1.42% |
|  | WildChat-4.8M | 0.61% |
|  | Nemotron-Chat-v3 | 0.23% |
|  | Arena-Human-Pref-140k | 0.04% |

</details>

## Training Recipe

The same recipe trains DFly2 and every baseline below, on all three targets of the technical report.

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Global batch size | 256 |
| Sequence length | 32K |
| Optimizer | Muon |
| Objective | Per-slot LK distillation loss (KL + TV) against the target, weighted by D-PACE |
| Draft tokens per block (γ) | 5 |

Only the drafter is trained. At serving time the selection comes from the target's verify pass. Training has no verify pass, so it recomputes the selection with a frozen copy of each paired layer's indexer, and a draft layer sees the same latents and positions in training as at serving.

## Evaluation: Accepted Length

Mean accepted length τ, counting the bonus token, in the 2K chat setting (thinking disabled, at most 2048 generated tokens). Each cell gives T = 0 / T = 1, and bold marks the best per column and temperature. Accepted length is measured on a single TP8 engine on NVIDIA H20 GPUs, as the mean of the last three runs.

All baselines are trained by us with the same recipe, data and number of steps as DFly2, on the same three-layer MoE stack (within 2% of its size). DFlash keeps its own KV cache over the whole context, DSpark bounds its own cache with a 128-token sliding window, and DFlash† is DFlash held to that same window. All methods draft a block of 5 tokens on vLLM.

### General-purpose benchmarks

GSM8K, MATH-500 (MATH), AIME25, MBPP, HumanEval, LiveCodeBench (LCB) and MT-Bench, seven of the twelve evaluation sets that ship with DeepSpec.

| Drafter | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DFlash | 3.95 / 3.64 | 3.93 / 3.42 | 3.54 / 2.97 | 3.82 / 3.63 | 4.25 / 3.99 | 3.34 / 2.80 | 2.70 / 2.51 | 3.65 / 3.28 |
| DFlash† | 3.94 / 3.59 | 3.91 / 3.40 | 3.51 / 2.94 | 3.81 / 3.58 | 4.08 / 3.82 | 3.26 / 2.77 | 2.68 / 2.53 | 3.60 / 3.23 |
| DSpark | 4.24 / 4.00 | 4.18 / 3.74 | 3.69 / 3.24 | 4.00 / 3.83 | 4.27 / 4.07 | 3.44 / 3.04 | 2.82 / 2.67 | 3.81 / 3.51 |
| DFly2 | **4.62** / **4.45** | **4.43** / **4.10** | **4.01** / **3.60** | **4.28** / **4.17** | **4.70** / **4.56** | **3.76** / **3.38** | **3.03** / **2.91** | **4.12** / **3.88** |

### Long-context and agentic benchmarks

Long context: LongSpec 32K, CodeSpan 64K and LongSWE-Bench 128K (LSWE 128K). Agentic: AgentLongBench (ALB), MemoryAgentBench (MemAgent), the long-context multi-turn category of BFCL-v3, and LongSWE-Bench issues at 64K (SWE 64K). *Long Avg.* and *Agen. Avg.* average the three long-context and four agentic sets. Every drafter is trained at 32K.

| Drafter | LongSpec 32K | CodeSpan 64K | LSWE 128K | ALB | MemAgent | BFCL-v3 | SWE 64K | Long Avg. | Agen. Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DFlash | 2.98 / 2.55 | 3.81 / 3.13 | 3.42 / 2.96 | 4.38 / 4.13 | 2.84 / 2.60 | 3.40 / 3.11 | 3.58 / 3.14 | 3.40 / 2.88 | 3.55 / 3.25 |
| DFlash† | 2.74 / 2.40 | 3.43 / 2.86 | 3.15 / 2.77 | 3.78 / 3.62 | 2.80 / 2.52 | 3.21 / 2.98 | 3.19 / 2.80 | 3.11 / 2.68 | 3.24 / 2.98 |
| DSpark | 2.82 / 2.48 | 3.58 / 2.92 | 3.29 / 2.98 | 3.87 / 3.59 | 2.85 / 2.65 | 3.33 / 3.09 | 3.35 / 2.99 | 3.23 / 2.79 | 3.35 / 3.08 |
| DFly2 | **3.12** / **2.81** | **4.14** / **3.59** | **3.97** / **3.55** | **5.10** / **4.72** | **3.36** / **3.13** | **3.77** / **3.55** | **4.02** / **3.60** | **3.74** / **3.32** | **4.06** / **3.75** |

## Evaluation: Speedup

Speedup over the target alone on NVIDIA H20 GPUs, greedy, at a concurrency of one, with prefill and decode on separate TP8 instances. Each cell gives decoding / end-to-end speedup, the latter including prefill, and bold marks the best per column and speedup. DFlash† reads a window at decode but still builds a full cache at prefill. On long prompts, both DFlash variants pay for building their own cache over the prompt with up to twice the time to the first token, and end the agentic sets slower than not speculating. DFly2 builds no cache of its own at prefill and leaves prefill speed unaffected.

### General-purpose benchmarks

| Drafter | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DFlash | 3.23× / 2.87× | 3.26× / 3.14× | 2.86× / 2.73× | 3.11× / 3.04× | 3.41× / 3.29× | 2.67× / 2.54× | 2.21× / **2.16×** | 2.96× / 2.82× |
| DFlash† | 3.21× / 2.83× | 3.19× / 3.06× | 2.75× / 2.71× | 3.09× / 3.02× | 3.28× / 3.17× | 2.62× / 2.49× | 2.19× / 2.14× | 2.91× / 2.77× |
| DSpark | 3.30× / 2.91× | 3.24× / 3.11× | 2.80× / 2.76× | 3.09× / 3.02× | 3.31× / 3.18× | 2.63× / 2.52× | 2.15× / 2.11× | 2.93× / 2.80× |
| DFly2 | **3.56×** / **3.09×** | **3.42×** / **3.27×** | **3.03×** / **2.98×** | **3.28×** / **3.20×** | **3.57×** / **3.36×** | **2.87×** / **2.74×** | **2.27×** / 2.15× | **3.14×** / **2.97×** |

### Long-context and agentic benchmarks

| Drafter | LongSpec 32K | CodeSpan 64K | LSWE 128K | ALB | MemAgent | BFCL-v3 | SWE 64K | Long Avg. | Agen. Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DFlash | **2.46×** / 1.75× | 3.05× / 1.67× | 2.59× / 1.16× | 3.46× / 0.69× | 2.22× / 0.63× | 2.67× / 0.78× | 2.79× / 1.49× | 2.70× / 1.52× | 2.78× / 0.90× |
| DFlash† | 2.32× / 1.70× | 2.70× / 1.55× | 2.52× / 1.13× | 3.09× / 0.83× | 2.25× / 0.69× | 2.63× / 0.84× | 2.55× / 1.47× | 2.51× / 1.46× | 2.63× / 0.96× |
| DSpark | 2.28× / 1.99× | 2.63× / 2.06× | 2.49× / 1.68× | 2.95× / 1.21× | 2.37× / **1.27×** | 2.58× / **1.37×** | 2.54× / 1.89× | 2.47× / 1.91× | 2.61× / 1.44× |
| DFly2 | 2.44× / **2.10×** | **3.23×** / **2.37×** | **2.94×** / **1.86×** | **3.82×** / **1.25×** | **2.60×** / 1.24× | **2.97×** / 1.36× | **3.00×** / **2.13×** | **2.87×** / **2.11×** | **3.10×** / **1.50×** |

## Evaluation: High-Concurrency Throughput

Throughput against per-user decode speed, greedy, with every instance at DP8 × TP1 and EP8. Short text pools the twelve general-purpose sets on one prefill and one decode instance (1P1D) at 1 to 128 concurrent requests. CodeSpan 64K and SWE 64K run at 1 to 32 concurrent requests with eight prefill instances per decode instance (8P1D). Gains are measured against DSpark.

| Workload | Deployment | Throughput at equal per-user speed | Per-user speed at equal throughput |
|---|---|---:|---:|
| Short text | 1P1D | +28% | +15% |
| CodeSpan 64K | 8P1D | +62% | +23% |
| SWE 64K | 8P1D | +81% | +24% |

## Deployment

**vLLM: support in progress.** Serving instructions will be added here once DFly2 support is available in vLLM. The drafter is deployed together with the DeepSeek-V4.1-Flash target model and drafts 5 tokens per step (block size 6).

## License

This drafter is released under the [Apache-2.0 License](LICENSE). The DeepSeek-V4.1-Flash target model is subject to its own license.

## Technical Report

DFly2 and the results above are described in the DFly2 technical report:

> **DFly2: Turbocharging Long-Context Speculative Decoding by Sharing the Target's Sparsity**  
> arXiv: https://arxiv.org/abs/2611.34776

The drafter is trained with AngelSpec. For the training framework, see:

> **AngelSpec: Towards Real-World High Performance Inference with Speculative Decoding**  
> arXiv: https://arxiv.org/abs/2607.25852
