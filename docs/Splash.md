# Splash engine

[Splash](https://github.com/incoai/splash) runs Qwen3.8-27B and Qwen3.6-35B-A3B
on its own Metal engine with DFlash2 speculative decoding. oMLX can serve a
model through Splash instead of its in-process MLX engines. The model stays in
oMLX's model list, API, dashboard and usage statistics like any other; Splash
does the inference in a separate process.

## What runs on Splash

- **MLX checkpoints**, when you enable **Model Settings → Splash**. The toggle
  is offered for MLX affine 4-bit/group-64 checkpoints of the two families
  Splash supports, for example `mlx-community/Qwen3.8-27B-4bit`.
- **Splash packages** (`incoai/Qwen3.8-27B-Splash`,
  `incoai/Qwen3.6-35B-A3B-Splash`). These carry weights packed for Splash and
  a `manifest.json` instead of a `config.json`, so only Splash can run them.
  oMLX discovers them in its model directories and always serves them through
  Splash.

Splash resolves a model by its Hugging Face repository ID from the Hugging
Face cache, so the model must already be there. A model folder laid out as
`<owner>/<repo>` (how oMLX and LM Studio download) names its repository, and
Hub cache entries carry it. oMLX never lets Splash download a target model:
the toggle stays unavailable, with the reason shown, until the snapshot is
cached. On first use Splash downloads the small DFlash2 draft trained for the
family.

## Splash builds

oMLX works with two Splash builds, installed side by side:

| Build | Launcher | Macs | Install |
| --- | --- | --- | --- |
| Splash (official) | `splash` | M3 or later | `brew install incoai/tap/splash` |
| Splash M1 | `splash-m1` | M1 and M2 as well | `install-m1.sh` from [paperniuk/splash releases](https://github.com/paperniuk/splash/releases) |

The official build requires an Apple GPU family 9 (M3 or later). Splash M1 is
a community port that adds kernels for M1/M2 GPUs.

oMLX looks for each launcher on `PATH`, then in `/opt/homebrew/bin`,
`/usr/local/bin` and `~/.local/bin`, and reads `serve --help` once to learn
what the build accepts. Builds before Splash 1.1 serve Splash packages only;
oMLX offers MLX checkpoints only to a build that loads them.

**Model Settings → Splash build** chooses the build per model. Automatic, the
default, picks Splash M1 on M1/M2 Macs and the official build otherwise,
falling back to whichever is installed. `OMLX_SPLASH_PATH` names another
launcher, such as a source checkout's `./splash`; automatic selection then
prefers it.

## How it works

Loading the model starts `splash serve` from the selected build on a free
loopback port with a random API key, which travels in the environment rather
than on the command line. Loading returns when Splash serves the model; the first start takes
longer while Splash prepares it. Requests are forwarded to Splash's
OpenAI-compatible API. Reasoning comes back inside `<think>` tags for oMLX's
reasoning parser, and tool calls come back structured, so chat completions,
the Responses API and the Anthropic Messages API all work. The dashboard's
active-models card shows Splash requests live (prefill progress from Splash's
own progress events, then generation speed).

Unloading the model stops the Splash process. Splash runs in its own process
group under a small watchdog shell: if oMLX exits without unloading, even
from `kill -9`, the watchdog stops Splash within two seconds. Splash's log is
in oMLX's log directory under `splash/`. The in-memory prefix cache is Splash's
own and, when oMLX's SSD cache is on, Splash spills KV pages and states to
SSD within the same size limit; hot-cache-only or a disabled cache leaves
Splash's disk tier off. The dashboard's cache block lists Splash models next
to MLX ones, with Splash's own counters (prefix hits, cached KV pages, disk tier
usage), polled from Splash every few seconds.

Settings mapped onto Splash:

| oMLX | Splash |
| --- | --- |
| Max context window | `--max-context` (at most 256K) |
| `reasoning_effort` | `low`, `medium`, `xhigh`; `high`/`max` become `xhigh`, and `enable_thinking: false` becomes `none` |
| `top_k` | clamped to 32 |
| `min_p`, presence/frequency/repetition penalties, thinking budget | ignored, with a one-time warning |
| SSD cache (Settings → Cache) | `--max-cache-disk` with the same size limit; the cache file goes under `<SSD cache dir>/splash` |
| `response_format` (JSON object / JSON schema) | enforced by Splash itself, no prompt injection |

## Benchmarks

The admin benchmark runs on Splash models through Splash's chat API,
with thinking off, a unique prefix on every prompt so Splash's prefix cache
cannot skew prefill, and the Splash process group's peak memory. Splash
results are not uploaded to omlx.ai.

## Limits

- Splash serves chat requests only. `/v1/completions` returns HTTP 400.
- Requests are text only: Splash starts with `--language-only` where the
  build supports it.
- The other acceleration settings (MTP, DFlash, TurboQuant, ANE prefill, and
  so on) configure oMLX's MLX engines and do not apply while Splash serves the
  model.
- The dashboard's memory bar and the model's observed size include the Splash
  process group's memory. oMLX's process memory limit does not evict or throttle
  Splash: Splash budgets its own memory, and unloading the model stops it.
