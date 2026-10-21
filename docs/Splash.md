# Splash engine

[Splash](https://github.com/incoai/splash) runs Qwen3.8-27B, Qwen3.6-35B-A3B and Qwen3.8-Flash-Next on its own Metal engine with DFlash2 speculative decoding. oMLX can serve a model through Splash as an external engine instead of its in-process MLX engines, including GGUF files, which only an external engine runs. The model stays in oMLX's model list, API, dashboard and usage statistics like any other; Splash does the inference in its own process.

## Setup

1. Install Splash 1.3.0 or later (1.4.0 or later for Flash-Next):

   | Build | Launcher | Macs | Install |
   | --- | --- | --- | --- |
   | Splash (official) | `splash` | M3 or later | `brew install incoai/tap/splash` |
   | Splash M1 | `splash-m1` | M1 and M2 as well | `install-m1.sh` from [paperniuk/splash releases](https://github.com/paperniuk/splash/releases) |

2. Open **Settings → Integrations → External engines**. oMLX looks for the launcher on `PATH`, then in `/opt/homebrew/bin`, `/usr/local/bin` and `~/.local/bin`, preferring Splash M1 on M1/M2 Macs and the official build elsewhere. When it finds one, the card shows its path and version. Otherwise, enter the launcher's path there (a source checkout's `./splash`, for example).
3. Download the DFlash2 draft Splash pairs with the model's family into a model directory: `incoai/Qwen3.8-27B-DFlash2` (`z-lab/Qwen3.8-27B-DFlash2` holds the same weights), `incoai/Qwen3.6-35B-A3B-DFlash2` or `incoai/Qwen3.8-Flash-Next-DFlash2`. The model's **Model Settings → Advanced** offers a button that downloads it.
4. In the model's **Model Settings → Advanced**, turn on **Use external engine** and pick Splash. A GGUF model has it on already.

The toggle stays off, with the reason shown, until all of this is in place. The macOS app has the same settings under Integrations and the model's Advanced tab. In the model list, a model Splash can serve carries a Splash badge.

## What runs on Splash

Quantized checkpoints of the families Splash supports:

- MLX checkpoints with affine quantization. Splash 1.3.0 (and Splash M1 1.3.0) serves affine 4-bit/group-64 checkpoints, for example `mlx-community/Qwen3.8-27B-4bit`. Splash 1.3.1 also serves mixed-bit checkpoints, and checks the bit widths itself when it loads the model.
- GGUF files, for example a variant of `unsloth/Qwen3.8-27B-GGUF` (see [GGUF models](#gguf-models)). Splash checks the file's tensor types itself. Split GGUFs (`-00001-of-0000N`) need Splash 1.4.0 or later and all parts in the same folder.
- Splash 1.3.2 also serves NVFP4 and FP8 checkpoints from NVIDIA Model Optimizer and compressed-tensors, for example `nvidia/Qwen3.8-27B-NVFP4` and `unsloth/Qwen3.8-27B-NVFP4`.
- Qwen3.8-Flash-Next GGUFs need Splash 1.4.0 or later, for example the Q2_0 variant of `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` or an Unsloth GGUF. Flash-Next MLX checkpoints do not run on Splash. Its DFlash2 draft is `incoai/Qwen3.8-Flash-Next-DFlash2`.

Prebuilt Splash packages (`incoai/*-Splash`) are not supported; Splash 1.3.1 replaced them with the MLX checkpoints above. The model downloader refuses them and names the MLX model that holds the same weights.

## GGUF models

In the model downloader, the **GGUF only** view lists GGUF repositories. Downloading one opens its file list: check the GGUF to download, and its vision projector (`mmproj-*.gguf`) is checked with it. Only the checked files are fetched, into `<model dir>/<owner>/<repo>/`.

oMLX lists each GGUF in a model directory, an organization folder (`<owner>/<repo>/`, as the downloader and LM Studio lay them out) or the Hugging Face cache as its own model, named by its file, for example `Qwen3.8-27B-UD-Q4_K_M`. Split GGUFs can also be in a folder named for their variant, such as `Q2_0/Model-Q2_0-00001-of-00002.gguf`. All parts must be present; a vision projector at the repository root is paired with them. Vision projectors, imatrix files and draft subfolders are not models. For each GGUF oMLX generates a folder under `<base path>/gguf` with a `config.json` derived from the file's header and links to the file and its projectors; for the Qwen families the folder also holds the tokenizer the header defines. A GGUF runs only on an external engine: it never loads in oMLX's MLX engines, and `/v1/models` lists it only while an engine can serve it. Deleting a GGUF in the model manager removes its parts, and the folder goes with its last GGUF.

## How it works

Splash finds models by Hugging Face repository ID in a Hugging Face cache. oMLX presents the model folder to it as an entry of a private cache under `<base path>/splash/hub`: a snapshot made of symlinks to the model's files. For a GGUF that is the GGUF and its projectors, and Splash reads the tokenizer and configuration from the GGUF itself. Splash starts with `--offline` against that cache, so it never downloads the model, and the files it reads are the ones in your model directory. The draft is passed with `--draft-model`.

On the first start Splash hashes the model files once (about one full read of the weights); later starts only check their sizes and times. A model folder that is itself a Hugging Face cache snapshot skips the hashing. Splash keeps its own selection links under `~/Library/Application Support/Splash/models/omlx/`.

Loading the model starts `splash serve` from the selected launcher on a free loopback port with a random API key, which travels in the environment rather than on the command line. Requests are forwarded to Splash's OpenAI-compatible API. Reasoning comes back inside `<think>` tags for oMLX's reasoning parser, and tool calls come back structured, so chat completions, the Responses API and the Anthropic Messages API all work. The dashboard's active-models card shows Splash requests live (prefill progress from Splash's progress events, then generation speed), and the cache block lists Splash models next to MLX ones with Splash's own counters.

Unloading the model stops the Splash process. Splash runs in its own process group under a small watchdog shell: if oMLX exits without unloading, even from `kill -9`, the watchdog stops Splash within two seconds. Splash's log is in oMLX's log directory under `splash/`.

## Settings

| oMLX | Splash |
| --- | --- |
| Model's max context window, when set | `--max-context` (at most 256K); Splash refuses to start when its memory budget cannot hold it |
| Integrations → Memory limit (empty: enough for the model's context window) | `--max-memory`, never more than the oMLX memory limit leaves free |
| Integrations → KV cache precision | `--kv-format` |
| Integrations → SSD cache (empty: the oMLX SSD cache size, 0: off) | `--max-cache-disk` with `--persistent-cache` under `<SSD cache dir>/splash/` |
| Integrations → Neural Engine prefill | off adds `--disable-ane` |
| `reasoning_effort`, `enable_thinking: false` | `reasoning_effort` (`enable_thinking: false` becomes `none`) |
| `response_format` (JSON object / JSON schema) | enforced by Splash itself, no prompt injection |

oMLX unloads idle models itself, so Splash starts with `--idle-release off`.

## Memory

Splash spends whatever memory budget it gets beyond the weights on KV pages, and keeps the pages it no longer needs as a prefix cache. Its context window is the whole KV pool, and concurrent requests that outgrow the pool wait for pages. With the memory limit empty, oMLX sizes the budget for the model's context window: the weights, the draft, scratch and one window of KV pages (about 29 GB for Qwen3.8-27B at 256K with int8 KV). A larger limit gives the prefix cache more room.

When the memory left under the oMLX limit cannot hold the full window (a 32 GB Mac, say), Splash fits a shorter context to it, and oMLX takes that as the model's context limit: longer prompts get the usual context overflow error, and `/v1/models` reports the shorter `max_model_len`. Below the memory Splash needs for the weights and one request (about 18 GiB for Qwen3.8-27B), Splash cannot start.

After Splash starts, oMLX counts the budget Splash planned as the model's size, and the memory guard counts the Splash process's memory as oMLX's own. Loading other models therefore sees Splash's share, and evicting the Splash model frees it.

With Splash 1.4.0, Flash-Next can stream experts and its n-gram table from SSD when they do not fit in memory. It can also spill active KV cache to SSD when a writable SSD cache has room. Splash plans these stores within the memory limit oMLX passes; no extra oMLX setting is needed.

## Benchmarks

The admin benchmark runs on Splash models through Splash's chat API, with thinking off, a unique prefix on every prompt so Splash's prefix cache cannot skew prefill, and the Splash process group's peak memory. Splash results are not uploaded to omlx.ai.

## Limits

- Splash serves chat requests only. `/v1/completions` returns HTTP 400.
- Image input works for checkpoints with a vision tower (`vision_config` in `config.json`): oMLX passes image parts through to Splash, in chat completions, the Responses API and the Anthropic Messages API. Checkpoints without one start with `--language-only`. Splash does not read video or audio input.
- The other acceleration settings (MTP, DFlash, TurboQuant, ANE prefill, and so on) configure oMLX's MLX engines and do not apply while Splash serves the model.
