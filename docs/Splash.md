# Splash engine

[Splash](https://github.com/incoai/splash) runs Qwen3.8-27B and Qwen3.6-35B-A3B on its own Metal engine with DFlash2 speculative decoding. oMLX can serve a model through Splash as an external engine instead of its in-process MLX engines. The model stays in oMLX's model list, API, dashboard and usage statistics like any other; Splash does the inference in its own process.

## Setup

1. Install Splash 1.3.0 or later:

   | Build | Launcher | Macs | Install |
   | --- | --- | --- | --- |
   | Splash (official) | `splash` | M3 or later | `brew install incoai/tap/splash` |
   | Splash M1 | `splash-m1` | M1 and M2 as well | `install-m1.sh` from [paperniuk/splash releases](https://github.com/paperniuk/splash/releases) |

2. Open **Settings → Integrations → External engines**. oMLX looks for the launcher on `PATH`, then in `/opt/homebrew/bin`, `/usr/local/bin` and `~/.local/bin`, preferring Splash M1 on M1/M2 Macs and the official build elsewhere. When it finds one, the card shows its path and version. Otherwise, enter the launcher's path there (a source checkout's `./splash`, for example).
3. Download the DFlash2 draft for the model's family into a model directory, for example `z-lab/Qwen3.8-27B-DFlash2` (the same weights as `incoai/Qwen3.8-27B-DFlash2`) or `incoai/Qwen3.6-35B-A3B-DFlash2`.
4. In the model's **Model Settings → Advanced**, turn on **Use external engine** and pick Splash.

The toggle stays off, with the reason shown, until all of this is in place. The macOS app has the same settings under Integrations and the model's Advanced tab.

## What runs on Splash

MLX checkpoints of the two families Splash supports, quantized with affine quantization:

- Splash 1.3.0 (and Splash M1 1.3.0) serves affine 4-bit/group-64 checkpoints, for example `mlx-community/Qwen3.8-27B-4bit`.
- Splash 1.3.1 also serves mixed-bit checkpoints, and checks the bit widths itself when it loads the model.

Prebuilt Splash packages (`incoai/*-Splash`) are not supported; Splash 1.3.1 replaced them with the MLX checkpoints above.

## How it works

Splash finds models by Hugging Face repository ID in a Hugging Face cache. oMLX presents the model folder to it as an entry of a private cache under `<base path>/splash/hub`: a snapshot made of symlinks to the model's files. Splash starts with `--offline` against that cache, so it never downloads the model, and the files it reads are the ones in your model directory. The draft is passed with `--draft-model`.

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

## Benchmarks

The admin benchmark runs on Splash models through Splash's chat API, with thinking off, a unique prefix on every prompt so Splash's prefix cache cannot skew prefill, and the Splash process group's peak memory. Splash results are not uploaded to omlx.ai.

## Limits

- Splash serves chat requests only. `/v1/completions` returns HTTP 400.
- Image input works for checkpoints with a vision tower (`vision_config` in `config.json`): oMLX passes image parts through to Splash, in chat completions, the Responses API and the Anthropic Messages API. Checkpoints without one start with `--language-only`. Splash does not read video or audio input.
- The other acceleration settings (MTP, DFlash, TurboQuant, ANE prefill, and so on) configure oMLX's MLX engines and do not apply while Splash serves the model.
