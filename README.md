<p align="center">
  <img alt="Flyto MLX" src="docs/images/icon-rounded-light.svg" width="120">
</p>

<h1 align="center">Flyto MLX</h1>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
</p>

---

**中文** · [English](#english)

Flyto MLX 是基于 [@jundot/oMLX](https://github.com/jundot/omlx) 派生的 Mac 本地大模型推理服务，聚焦中文与国产模型生态。它保留了 oMLX 全部能力（OpenAI 兼容 API、多模型按需调度、KV 分页缓存、菜单栏 GUI），并在此之上加入了上游目前还没合并或不支持的功能。

最显著的一项是音频对话。`/v1/chat/completions` 接受 OpenAI 标准的 `input_audio` 内容类型，可以让 `gemma4-e2b` / `gemma4-e4b` 听一段音频再回答问题——不是简单替代专用语音转写，而是让语速、停顿、犹豫这些声音信号一起参与推理。实测一段 158 秒的中文销售电话录音，模型给出贴近原文的转写加上对客户态度的判断。上游 oMLX 在六个不同位置（内容解析器、Pydantic schema、chat 模板、Gemma 4 adapter、引擎 prepare_inputs、最外层 gate）把音频路径切断了，这次都修通了。

DFlash 双引擎让通义千问和 Gemma 4 共用一套草稿模型加目标模型的 Metal 内存布局，跑 30B 以上模型时吞吐量有明显提升。Google 官方的 Gemma 4 MTP assistant drafter 也已接通：12B 目标模型挂 0.24B drafter 后单流解码实测从 42 提到 57 tok/s（提升 1.38 倍），模型设置里配 `vlm_mtp_enabled` 加 drafter 路径即可启用。

macOS 26（Tahoe）把菜单栏遮挡检测的标志位从 `0x2` 改成了 `0x2000`，不改这一处菜单栏状态会判错，已修。

回填了上游已合但还没发版的五处修复：tokenizer 词表大小取 `lm_head` 权重、缓存命中时 TokenBuffer 种子重建、健康检查复用 HTTP Session 防端口耗尽，以及另外两处。

通义千问 3.5（Dense 与 MoE）、DeepSeek V4、Gemma 4 全家的中文别名开箱即用。MoE 别名按上游模型卡的命名习惯显式带活跃参数量，例如 `qwen-moe-35b-a3b`、`qwen-moe-122b-a10b`、`gemma4-moe-26b-a4b`。

## 安装

推荐用 Homebrew：

```
brew tap panwudi/flyto-mlx https://github.com/panwudi/flyto-mlx
brew install flyto-mlx
brew services start flyto-mlx
```

命令行入口 `fmlx serve --port 8000`。出于对上游脚本的兼容，`omlx serve` 也保留为同一程序的别名。

如果在 Linux 上，或者已经有 Python 环境想做开发，可以直接从 git 装：

```
pip install flyto-mlx
```

从 0.6.0 起 PyPI 通道已开通：mlx-vlm 0.6.x 把此前只存在于 git 提交里的依赖正式发版了，整条依赖链都能从 PyPI 解析。想跟最新主干也可以从 git 装：

```
pip install git+https://github.com/panwudi/flyto-mlx@v0.6.0
```

## 一个示例

```python
import base64, requests

with open("recording.wav", "rb") as f:
    audio = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer 你的密钥"},
    json={
        "model": "gemma4-e2b",
        "max_tokens": 400,
        "temperature": 0.3,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "总结这段电话的关键信息"},
                {"type": "input_audio",
                 "input_audio": {"data": audio, "format": "wav"}}
            ]
        }]
    },
)
print(resp.json()["choices"][0]["message"]["content"])
```

## 视频与图像生成

除了语言模型, fmlx 还能在同一个 server 里跑视频生成 (`/v1/videos`, Wan2.2) 和图像生成 (`/v1/images`). 两者共用一套独立 venv 的子进程 worker (mlx-gen 运行时), 生成任务持有内存租约, 与正在服务的 LLM 共驻而不互相挤爆——这是单机统一内存才有的玩法.

图像侧支持三类模型, 放进模型目录 (`~/.fmlx/models/AbstractFramework/<repo>`) 重启即被自动识别:

| 模型 | 用途 | 实测 (M5 Max, 1024x1024) |
|---|---|---|
| z-image-turbo-4bit | 快速文生图, 9 步出图 | 22 秒, 峰值 8.6GB |
| qwen-image-2512-4bit | 中文排版/海报字天花板, 40 步 | 5 分钟; 挂 Lightning LoRA 8 步 73 秒 |
| qwen-image-edit-2511-4bit | 指令改图/图内改字, 上传图片+一句话 | 15 分钟 (40 步) |

开启方式: admin 设置页打开图像生成开关 (settings.image.enabled), worker venv 与视频引擎共用一个 (`uv venv -p 3.12 ~/.fmlx/venvs/video && uv pip sync --python ~/.fmlx/venvs/video/bin/python omlx/video/requirements.lock`). 聊天页直接选图像模型就能用: 纯文字生成图片, 带图自动路由到改图模型. API 走 OpenAI images 形态:

```
curl http://localhost:8000/v1/images \
  -H "X-API-Key: 你的密钥" -H "Content-Type: application/json" \
  -d '{"model": "z-image-turbo-4bit", "prompt": "咖啡店开业海报, 标题\"开业大吉\"",
       "size": "1024x1024", "response_format": "url"}'
```

默认同步返回; 加 `"sync": false` 改为返回 job 对象轮询进度 (`GET /v1/images/{id}`), 适合分钟级的 qwen 任务. 扩展参数有 negative_prompt / steps / seed / guidance / n / image_strength / lora_paths, 官方 openai SDK 的 `client.images.generate` 也能直接打通 (`POST /v1/images/generations`). 步数和默认尺寸可以全局设, 也可以在模型设置里按模型覆盖. 设计细节见 [docs/image-generation-engine-spec.md](docs/image-generation-engine-spec.md) 与 [docs/video-generation-engine-spec.md](docs/video-generation-engine-spec.md).

## 多机集群路由

如果有多台 Mac，集群路由器可以把请求按模型和负载自动分发到多台 `omlx serve`。它不是 GPU/显存级集群，也不是共享 KV 缓存——每台仍是独立 server，路由器只决定每个请求由哪台处理。请求只会发给装有目标模型的机器；更快的机器配更高 `weight`，自动多分流量；显存吃紧的机器会被降权避开冷加载。客户端把地址从某台的 `:8000` 换成路由器的 `:9000` 即可，单机直连不受影响。配置见 `omlx/cluster/cluster.example.json`，完整说明见 [docs/cluster.md](docs/cluster.md)。

```
OMLX_CLUSTER_CONFIG=~/.omlx/cluster.json python -m omlx.cluster.router
# 客户端指向 http://<router-host>:9000/v1
```

## 跟上游 oMLX 的关系

Flyto MLX 是 oMLX 的下游派生，遵循 Apache 2.0。我们定期从上游回挑 bug 修复和新模型支持，但不再把自己的功能反向 PR 给上游。如果只想要纯净的上游体验，请直接用 [@jundot/oMLX](https://github.com/jundot/omlx)。完整版权与署名见 [NOTICE](NOTICE) 与 [LICENSE](LICENSE)。

---

## English

Flyto MLX is a downstream fork of [@jundot/oMLX](https://github.com/jundot/omlx) for Mac users working primarily with Chinese and sovereign-AI models (Qwen, DeepSeek, Gemma 4). It preserves all of oMLX's capabilities (OpenAI-compatible API, multi-model LRU scheduling, KV paged cache, menubar GUI) and adds a few things upstream has not merged yet.

The most visible addition is audio chat. `/v1/chat/completions` now accepts OpenAI's `input_audio` content type, letting `gemma4-e2b` or `gemma4-e4b` actually listen to audio rather than just transcribe it. Prosody, hesitation, and accent information feed into the answer, which an ASR-then-LLM pipeline cannot do. We verified this against a 158-second Chinese sales call: faithful transcription plus a meaningful analysis of the customer's attitude. Upstream oMLX silently broke the audio path in six places (content parser, Pydantic schema, chat template, Gemma 4 adapter, engine `prepare_inputs`, outer gate); all six are fixed here.

DFlash Path A runs Qwen and Gemma 4 backends with drafter and target model co-loaded into the same Metal heap, giving measurable throughput gains for 30B+ models on Mac mini and Studio. Google's official Gemma 4 MTP assistant drafters are wired up too: a 0.24B drafter lifts the 12B target from 42 to 57 tok/s single-stream (1.38x, measured); enable per model with `vlm_mtp_enabled` plus the drafter path.

macOS 26 (Tahoe) shifted NSStatusItem's occlusion bit from `0x2` to `0x2000`. Without the fix the menubar status check is wrong. Fixed.

Five upstream-merged but not-yet-released fixes are backported: `lm_head` tokenizer vocab size, TokenBuffer cache hit seeding, health-check session reuse, and two more.

Chinese model aliases come preconfigured for Qwen 3.5 (Dense and MoE), DeepSeek V4, and Gemma 4. MoE aliases follow upstream model-card naming with explicit active-params suffix: `qwen-moe-35b-a3b`, `qwen-moe-122b-a10b`, `gemma4-moe-26b-a4b`.

### Install

```
brew tap panwudi/flyto-mlx https://github.com/panwudi/flyto-mlx
brew install flyto-mlx
brew services start flyto-mlx
```

CLI: `fmlx serve --port 8000` (primary) or `omlx serve --port 8000` (kept as an alias for compatibility with upstream scripts).

For Linux or development use:

```
pip install flyto-mlx
```

The PyPI channel is live as of 0.6.0: mlx-vlm 0.6.x released the commits we previously had to pin from git, so the whole dependency chain now resolves from PyPI. To track the latest main instead:

```
pip install git+https://github.com/panwudi/flyto-mlx@v0.6.0
```

### Video and image generation

Beyond language models, the same server runs video generation (`/v1/videos`, Wan2.2) and image generation (`/v1/images`). Both execute in a subprocess worker from a separate venv (the mlx-gen runtime) and hold a memory lease against the server's ceiling, so a render co-resides with your serving LLMs instead of fighting them -- a unified-memory trick a discrete-GPU stack cannot pull off.

Three image model families are supported; drop them under `~/.fmlx/models/AbstractFramework/<repo>` and restart to auto-discover:

| Model | Use | Measured (M5 Max, 1024x1024) |
|---|---|---|
| z-image-turbo-4bit | fast text-to-image, 9 steps | 22 s, 8.6 GB peak |
| qwen-image-2512-4bit | best-in-class CJK typography, 40 steps | 5 min; 73 s with the Lightning LoRA at 8 steps |
| qwen-image-edit-2511-4bit | instruction editing, in-image text replacement | 15 min (40 steps) |

Enable via the admin settings page (settings.image.enabled); the worker venv is shared with the video engine (`uv venv -p 3.12 ~/.fmlx/venvs/video && uv pip sync --python ~/.fmlx/venvs/video/bin/python omlx/video/requirements.lock`). In the chat page just pick an image model: plain text generates, an attached image auto-routes to the edit model. The API speaks OpenAI images shape (`POST /v1/images`, sync by default with `b64_json`/`url`; `"sync": false` returns a pollable job for the minute-scale Qwen renders), and the official openai SDK works through `POST /v1/images/generations`. Extensions: negative_prompt / steps / seed / guidance / n / image_strength / lora_paths. Defaults are settable globally and per model. Design notes: [docs/image-generation-engine-spec.md](docs/image-generation-engine-spec.md), [docs/video-generation-engine-spec.md](docs/video-generation-engine-spec.md).

### Multi-machine cluster routing

With more than one Mac, the cluster router spreads requests across several
`omlx serve` backends by model and load. It is not GPU/memory-level clustering
and not a shared KV cache -- each backend stays standalone; the router only
picks which one handles each request. A request only goes to a machine that
hosts the model; a faster machine gets a higher `weight` and proportionally
more traffic; a memory-pressured machine is deprioritized to avoid cold loads.
Clients just swap a backend's `:8000` for the router's `:9000`; direct
single-backend access still works. See `omlx/cluster/cluster.example.json` and
[docs/cluster.md](docs/cluster.md).

```
OMLX_CLUSTER_CONFIG=~/.omlx/cluster.json python -m omlx.cluster.router
# point clients at http://<router-host>:9000/v1
```

### Relationship to upstream

Flyto MLX is a downstream fork of oMLX under Apache 2.0. We cherry-pick upstream fixes and new model support; we do not upstream our own features. For pure upstream behaviour, use [@jundot/oMLX](https://github.com/jundot/omlx) directly. See [NOTICE](NOTICE) and [LICENSE](LICENSE) for attribution and copyright.
