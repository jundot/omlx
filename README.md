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

DFlash 双引擎让通义千问和 Gemma 4 共用一套草稿模型加目标模型的 Metal 内存布局，跑 30B 以上模型时吞吐量有明显提升。

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
pip install git+https://github.com/panwudi/flyto-mlx@v0.4.1
```

`pip install flyto-mlx` 走 PyPI 这条路目前不可用。oMLX 全家依赖 mlx-vlm 几个还没发布到 PyPI 的提交，而 PyPI 不接受包的依赖里出现 git URL（PEP 508 §6 的硬约束）。等 mlx-vlm 0.6.x 把那些提交正式发版后再开通。

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

DFlash Path A runs Qwen and Gemma 4 backends with drafter and target model co-loaded into the same Metal heap, giving measurable throughput gains for 30B+ models on Mac mini and Studio.

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
pip install git+https://github.com/panwudi/flyto-mlx@v0.4.1
```

Plain `pip install flyto-mlx` is not currently available. Flyto MLX, like oMLX itself, depends on unreleased mlx-vlm commits that PEP 508 §6 prevents from being declared in PyPI packages. Once `mlx-vlm 0.6.x` ships with those commits we will enable the PyPI channel.

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
