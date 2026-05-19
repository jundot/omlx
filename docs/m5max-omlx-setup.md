# M5 Max oMLX 本地推理 Setup

> 日期：2026-05-08（migration 落地）
> 状态：dev-doc 测试 infra 已切到 oMLX 当主 backend（`tests/tool_calling/`），mlx_lm.server 留作并列 backend（`BASE_URL_OVERRIDE=http://m5max:8081/v1` 切回）
> 范围：M5 Max 128GB 上跑 oMLX (jundot/omlx) 作为多模型 LRU 推理 server，OpenAI 兼容，端口 8000
> 关系：本文档是 dev-doc 测试 infra 的 canonical backend 文档；`m5max-mlx-setup.md` 标 deprecated，仅作历史参考

---

## 一、Quick-start

m5max 实测 2026-05-08 通过：

```bash
brew install uv && uv python install 3.12
uv venv --python 3.12 ~/.venvs/omlx && source ~/.venvs/omlx/bin/activate
git clone https://github.com/jundot/omlx ~/omlx-src
cd ~/omlx-src && git checkout 7395ca5     # 本文实测 commit，main 漂移得快
uv pip install -e .                        # 核心 ~50 个包，2-3 分钟
mkdir -p ~/.omlx/models
ln -sfn ~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/*/ \
        ~/.omlx/models/gemma4-e2b           # 模型 = ~/.omlx/models/<name>/ 的 subdir
omlx serve --host 100.67.236.97 --port 8000 --max-model-memory 24GB --no-cache
```

实测 omlx 版本：`omlx==0.3.9.dev1` (head `7395ca5`)。

```bash
# 验证（任何同 Tailscale 设备）
curl http://m5max:8000/v1/models                          # 200, list 出 gemma4-e2b
curl -X POST http://m5max:8000/v1/chat/completions \
  -d '{"model":"gemma4-e2b","messages":[{"role":"user","content":"hi"}]}' \
  -H "Content-Type: application/json"
```

后续段落详解为什么用 oMLX、装机踩坑、多模型 EnginePool 配置、tool calling 实测、和 mlx_lm.server 行为差异、migration plan。

## 二、为什么 oMLX（vs mlx_lm.server）

`mlx_lm.server` 是 mlx-lm 自带的最小 OpenAI-compat HTTP server，单 model serve、单进程、无并发调度、无 KV 复用。oMLX (jundot/omlx) 是社区在 mlx-lm 上做的 production-grade 推理 server，覆盖以下短板：

| 维度 | mlx_lm.server | oMLX |
|---|---|---|
| 多模型同时服务 | 单 model（重启换模型） | EnginePool LRU，多 model 同时常驻 |
| Continuous batching | 无 | 有（`--max-concurrent-requests`） |
| KV cache 复用 | 进程内每次推理重算 | Paged SSD KV cache + hot cache (in-mem)，prefix 复用 |
| Tool call parser | model 名硬匹配（gemma3、qwen3 等） | 自动检测 + 显式输出 parser（log 见 `Output parser detected: gemma4`） |
| OpenAI 兼容 | 是（`/v1/chat/completions` etc.） | 是（同上 + 多了 `/v1/models` 自动 list） |
| Anthropic 兼容 | 无 | 有（`/v1/messages` style，未在本次 smoke 里验） |
| VLM (gemma4-e2b/e4b 等) | 需要 `mlx_vlm.server` 单独起 | 自动按 model 分发（VLM engine vs batched LLM engine） |
| Structured output (xgrammar) | 无 | 有，需装 extra `omlx[grammar]` (~2GB torch 依赖) |
| MCP tool 集成 | 无 | 有，需 `omlx[mcp]` extra |
| 启停 / 监督 | 手动 `pkill` | brew services 单元支持 (formula 自带 `keep_alive`) |
| 模型寻址 | `--model <hf_repo>` | **`--model-dir <dir>`，扫描 subdir，按 dir name 起别名** |

**用 oMLX 的场景**：多 model 同时挂着（4B 跑工具、27B 跑推理）、长 context 反复请求需要 prefix cache、希望生产化这个 box（brew services 守护、统一日志、内存 watchdog）。

**用 mlx_lm.server 的场景**：单 model 单 client、临时跑个 benchmark、不想引入 oMLX 那一堆 git-pinned 依赖。当前主线的 tool_calling 矩阵（`tests/tool_calling/run_matrix_local.sh`）走 `mlx_lm.server` 是合理的——它就是依次对每个候选独占 serve 一遍，没有"多模型并存"的需求。

**取舍现状**：oMLX 不是无损替换。它的依赖链长、装一次比 mlx_lm.server 重得多（pin 死了 mlx-lm/mlx-vlm/mlx-embeddings/dflash-mlx 各自的 git commit），破坏了 `~/.venvs/mlx`（mlx-lm 0.31.3 + mlx-vlm 0.5.0）的依赖图。所以单独起一个 venv `~/.venvs/omlx`，**两个 venv 并存，按需切换**。

## 三、装机步骤（实际验证过的）

### 3.1 前置

`uv` 0.11.11 + `python 3.12.13` (mlx 要求 ≥3.10) + Tailscale 已上线（绑 `100.67.236.97` 不暴露 LAN）+ HF cache 里至少有一个 mlx-community 4bit 模型已下载完。这些前置和 `m5max-mlx-setup.md` 的第五节完全一致，不重复。

### 3.2 venv + 源码 install

oMLX 的 `pyproject.toml` pin 死了 mlx-lm/mlx-vlm/mlx-embeddings/dflash-mlx 各自的 git commit，不能装在 `~/.venvs/mlx` 里覆盖现有 mlx-lm 0.31.3。**必须开新 venv**：

```bash
uv venv --python 3.12 ~/.venvs/omlx
source ~/.venvs/omlx/bin/activate
git clone https://github.com/jundot/omlx ~/omlx-src
cd ~/omlx-src
git checkout 7395ca5         # 本文实测 head；不 pin 的话 main 每天漂移
uv pip install -e .          # 核心，~50 个包
# 可选 extras：
# uv pip install -e ".[grammar]"   # +xgrammar +torch ~2GB（结构化输出才需要）
# uv pip install -e ".[mcp]"       # +mcp 客户端（接 MCP server）
```

⚠️ **2026-05-08 实测踩过的坑**：第一次 `uv pip install -e .` 跑到一半网络中断（前一轮 sub-agent G 死在这步），site-packages 只剩 `_virtualenv.pth` / `_virtualenv.py` 两个 bootstrap 文件，看起来像装了一半其实没装上。**直接重跑 `uv pip install -e .`** 即可（uv 的依赖 resolver 是幂等的，没装的会补上），不需要删 venv 重来。

### 3.3 brew formula（备选）

仓库里有 `Formula/omlx.rb`，对应 v0.3.8 release。本来想走 `brew tap jundot/omlx https://github.com/jundot/omlx && brew install omlx` 最简，但 formula 用 `python@3.11`（不是我们已经在用的 3.12）+ 还会从 source 编 rust（pydantic-core / rpds-py），不省时间，反而新引入 python 3.11 环境。**当前选择源码 venv**。如果未来要做 brew services 守护（auto-restart），可以再切 brew install。

### 3.4 验证 CLI

```bash
omlx --help
# 应输出 {serve, launch, diagnose}
omlx serve --help | head -20
```

oMLX 没有 `--version` flag。版本看 `pip show omlx`（活着的 venv 里）或 `head -3 ~/omlx-src/pyproject.toml` + `git log -1`。

## 四、起 server / 多模型 EnginePool

### 4.1 关键概念：model-dir，不是 model-repo

**与 mlx_lm.server 最大的差别**：oMLX 不接 HF repo path 作 `--model`，它扫 `--model-dir`（默认 `~/.omlx/models`）下的所有 subdirectory，每个 subdir 是一个 model，用 **subdir 名**作为 model_id。每个 subdir 必须含 `config.json` + `*.safetensors`。

最便利的做法是 symlink HF cache 里的 snapshot dir，**dir name 自取**：

```bash
mkdir -p ~/.omlx/models

# 例：把 SmolLM3-3B 注册成 model_id="smollm3-3b"
ln -sfn ~/.cache/huggingface/hub/models--mlx-community--SmolLM3-3B-4bit/snapshots/<hash>/ \
        ~/.omlx/models/smollm3-3b

# 例：gemma4-e2b 注册成 model_id="gemma4-e2b"
ln -sfn ~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/<hash>/ \
        ~/.omlx/models/gemma4-e2b

# 例：Qwen3.5-4B-4bit
ln -sfn ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-4bit/snapshots/<hash>/ \
        ~/.omlx/models/qwen-dense-4b

ls -la ~/.omlx/models/    # 三个软链
```

⚠️ **migration 时这个差异很 load-bearing**：现有 tests/tool_calling/config.py 里 `MODELS` 数组是 HF repo path（`mlx-community/Qwen3.5-4B-4bit`），不能直接打到 oMLX。要么把 symlink dir name 取成跟 repo path 一模一样（含斜杠不行——文件系统 dir 不能含 `/`），要么改 `config.py` 加映射层。第八节 migration plan 详细讨论。

### 4.2 启动命令（canonical，2026-05-09 修正）

```bash
source ~/.venvs/omlx/bin/activate
mkdir -p ~/.omlx/cache
caffeinate -i nohup omlx serve \
  --model-dir ~/.omlx/models \
  --host 100.67.236.97 \
  --port 8000 \
  --max-model-memory 80GB \
  --max-concurrent-requests 8 \
  --paged-ssd-cache-dir ~/.omlx/cache \
  > /tmp/omlx-server.log 2>&1 &
disown
```

⚠️ **不要传 `--no-cache`**。早期版本 docs 里写的 `--no-cache --max-concurrent-requests 4`
是首次 smoke 时的"最稳保守值"，跑出来 oMLX 比 mlx_lm.server 慢 33% 是 **配置错误**导致，
不是 oMLX 本身的限制。详见 §8.4 修正后实测。

- `--max-model-memory 80GB`：M5 Max 128GB 给 oMLX 80GB，剩 48GB 给系统 + 客户端。
- `--max-concurrent-requests 8`：跟 `tests/tool_calling/run_extended.py` 的
  `RUNNER_CONCURRENCY=8` 默认值匹配。设小了 throughput 上不去，设大了显存不够时会拒。
- `--paged-ssd-cache-dir ~/.omlx/cache`：开启 oMLX 主推的 paged SSD KV cache。
  注意 §8.4 实测：在 SYSTEM_PROMPT ≤ 2048 token 的场景里**不会命中**（block_size=2048
  落地，不足一块的 prefix 不存）。要落地命中需要 prompt > 2k tokens。

启动 log 关键行：

```
omlx.model_discovery - INFO - Discovered model: gemma4-e2b (type: vlm, engine: vlm, size: 3.50GB)
omlx.model_discovery - INFO - Discovered model: smollm3-3b (type: llm, engine: batched, size: 1.69GB)
omlx.engine_pool - INFO - Discovered 2 models, max memory: 24.00GB
omlx.server - INFO - Default model: gemma4-e2b
omlx.process_memory_enforcer - INFO - Process memory enforcer started (limit: 120.0GB, interval: 1.0s)
```

注意 **engine 自动按 model 类型分发**：gemma4-e2b 是 VLM → `vlm` engine + `parser=gemma4`，smollm3-3b 是 LLM → `batched` engine。多 model 同时常驻：smollm3 加载后 1.69GB，再触发 gemma4 加载后 5.19GB total，**没有 evict**（在 24GB 预算下）。

### 4.3 关键参数说明（实测 + `--help` 摘录）

| flag | 实测推荐 | 说明 |
|---|---|---|
| `--model-dir` | `~/.omlx/models` (default) | 扫描 subdirs |
| `--host` | `100.67.236.97` | Tailscale IP，不写默认 `127.0.0.1` 只能本机访问 |
| `--port` | `8000` (default) | 不要跟 mlx_lm.server 的 8080 冲突 |
| `--max-model-memory` | `24GB` ~ `80GB` | 加载模型总和上限（驱逐阈值），默认系统 RAM 80% |
| `--max-process-memory` | `auto` (default = RAM-8GB) | 进程 RSS 软限，超了 OOM kill 自己 |
| `--max-concurrent-requests` | **`8`**（推荐） | continuous batching 并发上限，跟 RUNNER_CONCURRENCY 对齐 |
| `--paged-ssd-cache-dir` | **`~/.omlx/cache`**（推荐启用） | KV cache 落 SSD，开启 prefix cache 路径 |
| `--paged-ssd-cache-max-size` | `auto` (盘可用 ~75%) | 默认无限制；`auto` 会保留 25% 给系统 |
| `--hot-cache-max-size` | `0` (default 关) | in-mem hot KV cache，开了 prefix 命中时省 SSD I/O，目前阶段没必要 |
| `--no-cache` | **不要用** | 早期 doc 推荐过，已确认是性能反优化（详见 §8.4） |
| `--initial-cache-blocks` | 256 (default) | 预分配 KV block 数；hybrid 模型自动放大到 2048 块 |
| `--api-key` | (空) | 不写则不校验 key，admin 处于 setup 状态 |
| `--hf-endpoint` | `https://hf-mirror.com` (国内) | 模型自动下载时走镜像 |
| `--mcp-config` | (空) | MCP server 配置文件路径，要装 `omlx[mcp]` |

### 4.4 brew services 守护（备选，未本次实测）

如果走 brew install 路线（3.3 节），formula 自带 service unit：

```bash
brew services start omlx     # 后台守护，crash 自动重启
brew services stop omlx
```

源码 venv 路线没这个，要自己包 launchd plist，或继续用 `caffeinate -i nohup ... &` + 手动 `pkill`。

## 五、OpenAI client 调用（实测）

完全兼容 OpenAI SDK：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://m5max:8000/v1",
    api_key="dummy",   # 启服务没设 --api-key 就不校验，但 SDK 必传非空
)

resp = client.chat.completions.create(
    model="gemma4-e2b",        # ← 注意是 subdir 名，不是 hf repo
    messages=[{"role": "user", "content": "用三句话解释 unified memory"}],
    max_tokens=200,
    temperature=0.7,
)
print(resp.choices[0].message.content)
print(resp.usage)
```

`/v1/models` 自动 list 出注册的所有 model：

```bash
curl http://m5max:8000/v1/models
# {"object":"list","data":[{"id":"gemma4-e2b",...},{"id":"smollm3-3b",...}]}
```

`/health` 端点（mlx_lm.server 没有这个）：

```bash
curl http://m5max:8000/health
# {"status":"healthy","default_model":"gemma4-e2b","engine_pool":{...}}
```

## 六、Tool calling 兼容性（实测）

按 dev-doc 现有 `tests/tool_calling/tools.py` 的 6 工具定义打 A1 case ("查询设备 CNC-001 的主轴温度") 和 set_device_params 多参数 case。**结论：OpenAI tool format 完全兼容，结果与预期一致**。

### 6.1 单工具调用（gemma4-e2b，PASS）

请求：

```python
client.chat.completions.create(
    model="gemma4-e2b",
    messages=[{"role": "user", "content": "查询设备 CNC-001 的主轴温度"}],
    tools=TOOL_DEFINITIONS,
    temperature=0,
    max_tokens=128,
)
```

响应（精简）：

```
finish_reason: tool_calls
content: '<eos>'
tool_calls: [
  Function(name='query_device_params',
           arguments='{"device_id": "CNC-001", "param_name": "spindle_temperature"}')
]
```

正确选 tool（`query_device_params`）+ 正确参数。`content='<eos>'` 是 Gemma 4 的特性（emit `<end_of_turn>` 后又 emit `<eos>`），mlx_lm.server 上观察到的也是同一行为，不是 oMLX 的 parser bug。

### 6.2 多参数工具调用（gemma4-e2b，PASS）

请求："把设备 CNC-002 的进给速率(feed_rate)设置为 1500 mm/min"

响应：

```
tool_calls: [Function(name='set_device_params',
  arguments='{"device_id":"CNC-002","param_name":"feed_rate","unit":"mm/min","value":1500}')]
```

4 个参数（含 optional 的 unit）全部正确。

### 6.3 streaming + tool calling（gemma4-e2b，PASS）

```python
stream = client.chat.completions.create(..., stream=True)
for ch in stream:
    if ch.choices[0].delta.tool_calls: ...
```

实测：TTFT 0.42s，total 1.30s，5 chunks，tool_calls 字段在 delta 里正常 emit。

### 6.4 SmolLM3-3B：tool calling 不出（FAIL，符合预期）

`smollm3-3b` 在同一 prompt 下没有 emit tool_call，而是吐了一段"我无法直接访问数据，建议您查看显示屏..."的 hallucinated 文字。`tool_calls=None`, `finish_reason=length`。

**这是 model 行为，不是 oMLX 的 parser bug**。前期对 SmolLM3-3B 的 tool_calling 矩阵实测就是 ~25% 通过率，本来就不是工具可靠的 model（详见 `tests/tool_calling/REPORT.md`）。换 gemma4-e2b 立刻 100%。oMLX 的 server log 里有 `Output parser detected: gemma4` —— 它能识别多 model 的 tool parser，不是单一 hardcoded。

### 6.5 性能数据（gemma4-e2b，单 client 串行）

| 阶段 | tokens/s | 备注 |
|---|---|---|
| Cold（首请求带 model load + 编译） | 5.7 | 一次性，4-5s 内完成 |
| Warm 第 2 次同 model | 13.0 | KV / arch 还在 warmup |
| Warm 稳定（≥3 次） | **47-49** | server log 内部统计，与 SDK 端测一致 |
| TTFT (streaming) | **0.42s** | 4bit + 3.5GB 权重 |

数量级和 mlx_lm.server 上对 gemma4-e2b 的实测（~50 tok/s）基本一致——预期之中，因为 oMLX 在 batch_size=1 的单 client 路径上没有继承 mlx-lm 之外的加速。优势要在多 client / 长 prefix / 多 model 场景才显现，本次 smoke 没覆盖。

## 七、跟 mlx_lm.server 的行为差异（坑、注意点）

| 项 | mlx_lm.server | oMLX | 备注 |
|---|---|---|---|
| `model` 字段值 | hf repo path (`mlx-community/X`) | subdir 名 | **migration 主要改动点** |
| 默认端口 | 8080 | **8000** | 共存时一定不要冲突 |
| `/v1/models` | list 启动 model 一项 | list 所有发现的 model | |
| `/health` | 没有 | 有，含 engine_pool 状态 | |
| 启动并发选项 | 无 | `--max-concurrent-requests` | |
| `reasoning` 字段 | mlx_lm 1.0+ 有 reasoning attribute | omlx response 有 `reasoning_content` 字段（顶层 message 上） | OpenAI SDK 都 dump 得出来，但 key name 不同 |
| 关停 | `pkill -f mlx_lm.server` | `pkill -f "omlx serve"` | |
| 日志详细度 | 一般 | 高（每请求 token/s、scheduler timing 都吐） | 调试体验更好 |
| Gemma 4 `<eos>` content | 有 | 有 | 同源（mlx-lm 那层） |
| 装机依赖 | `mlx-lm` 一个包 | 50+ 包，git-pinned 重 | 必须独立 venv |

**踩过的坑**：

1. **G 在 source build 装到一半中断**：直接重跑 `uv pip install -e .`，幂等。如果还失败，先看是不是 transformers / pydantic 这种重 dep 在编 native 部分超时——oMLX 拉的是 transformers >= 5.0.0，build 不快。
2. **port 8000 vs 8080**：oMLX 默认 8000，mlx_lm.server 默认 8080，先确认 `lsof -iTCP:8000 -sTCP:LISTEN` 没占。
3. **Model 不发现**：检查 subdir 里有 `config.json`。symlink 的 snapshot dir 是好的（实测 OK），但要 `ls -L` 看穿透解引用后是否完整。
4. **Process memory enforcer**：oMLX 默认按 RAM-8GB 设进程 RSS 软限（128GB 机器 → 120GB），超了直接 SIGKILL 自己。不想被 kill 把 `--max-process-memory disabled`，但默认值是合理的。
5. **VLM model 内存比看起来大**：gemma4-e2b 标 3.5GB（权重），加载后 EnginePool 累计 5.19GB（含 vision tower + KV 预分配）。

## 八、Migration 实测记录（2026-05-08，已落地）

dev-doc 测试 infra 已切到 oMLX 当主 backend，mlx_lm.server 留作并列 backend
（设 `BASE_URL_OVERRIDE=http://m5max:8081/v1` 即可切回）。本节是**已发生**的
migration 记录，不是计划。

### 8.1 实际改动

| 文件 | 改动 | 实测 cost |
|---|---|---|
| `scripts/setup-omlx-models.sh` | 新增：HF cache snapshot → `~/.omlx/models/<alias>/` 幂等 ln | 15 min |
| `tests/tool_calling/config.py` | `BASE_URL` → `:8000`，`ALL_MODELS` → alias 名（不再是 HF repo path），保留 `BASE_URL_OVERRIDE` env 兼容 | 5 min |
| `tests/tool_calling/run_test.py` | reasoning fallback：`reasoning` (mlx_lm.server) ‖ `reasoning_content` (oMLX)；错误提示串改 oMLX | 5 min |
| `tests/tool_calling/run_multiturn.py` | 同上 reasoning fallback | 1 min |
| `tests/tool_calling/run_matrix_local.sh` | **结构性重写**：去掉 ssh kill+start 循环，server 启动一次，wrapper 只负责 warmup + 跑 python；preflight 检查 daemon 活着、所有 alias 注册 | 25 min |
| `tests/tool_calling/_run_extended_matrix.sh` | CANDIDATES 改 alias，去掉 mlx_lm.server 进程检测，commit msg 标 `oMLX` | 5 min |

合计 ~1h 实改动。advisor pre-call 里第一条警告（subdir mapping 不完整）救了
30 分钟——首次列表漏了 F2 (Huihui)、35B-A3B、9b-nvfp4 等。

### 8.2 alias 表 (canonical mapping)

| alias | HF repo | 备注 |
|---|---|---|
| `qwen-dense-27b` | `mlx-community/Qwen3.5-27B-4bit` | E1 主推 27B Dense |
| `qwen-dense-27b-claude4.6` | `Jackrong/MLX-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-4bit` | B 蒸馏 27B |
| `qwen-moe-35b-a3b` | `mlx-community/Qwen3.5-35B-A3B-4bit` | E2 MoE |
| `qwen-moe-35b-a3b-claude4.6-uncensored` | `mlx-community/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-4bit` | F2 蒸馏+abliterated |
| `qwen-moe-122b-a10b` | `mlx-community/Qwen3.5-122B-A10B-4bit` | 122B MoE |
| `qwen3.6-dense-27b-nvfp4` | `mlx-community/Qwen3.6-27B-nvfp4` | D Qwen 3.6 + nvfp4 |
| `qwen-dense-27b-nvfp4` | `dumtjul/Qwen3.5-27B-mlx-nvfp4` | 量化对照 |
| `qwen-dense-9b` | `mlx-community/Qwen3.5-9B-4bit` | 中间档 Dense |
| `qwen-dense-4b` | `mlx-community/Qwen3.5-4B-4bit` | 工具层 |
| `gemma4-e4b` | `mlx-community/gemma-4-e4b-it-4bit` | Gemma 4 工具层（VLM engine） |
| `gemma4-e2b` | `mlx-community/gemma-4-e2b-it-4bit` | Gemma 4 小（VLM engine） |
| `smollm3-3b` | `mlx-community/SmolLM3-3B-4bit` | 工具调用 fail，留作对照 |

未列入：`mlx-community/Qwen3.5-9B-nvfp4`（HF 下载未完成，缺 config.json，oMLX
拒绝注册）。补完下载后重跑 `bash scripts/setup-omlx-models.sh` 即可拉进来。

### 8.3 实测验证

**功能等价**：9B 单 model 20 case × R=3 跑 single-turn，oMLX 53.0/60 = 88.3%，
mlx_lm.server 历史结果（`results_20260507_163238.json`）53.0/60 = 88.3%，**完全一致**。
低分 case 同样是 C1/C2/D5。说明 oMLX 的 tool parser + sampling 路径与 mlx_lm.server
等价，迁移无评分回归。

**NVFP4 兼容性**：spot-test `qwen-dense-27b-nvfp4` + `qwen3.6-dense-27b-nvfp4` 都能加载 + 出
chat completion，且 oMLX log 里 `Output parser detected: qwen` 正常。advisor 担心
的 mlx-lm pin commit 不支持 NVFP4 的风险**不存在**——pinned mlx-lm 在 7395ca5 上已
经覆盖 NVFP4 路径。

**性能对比**（9B medium prompt，详见 §8.4）：oMLX 单请求 TTFT 与 mlx_lm.server 接近
（1.7-1.8s），但 N=8 并发 wallclock **慢一截**（85s vs 57s）——继续读下一节。

### 8.4 性能对比实测（2026-05-09 修正版）

> 历史踩坑：2026-05-08 第一版报"oMLX 慢 33%"的数据来自 **错误配置 + broken
> baseline**：oMLX 启动时传了 `--no-cache --max-concurrent-requests 4`（关掉了 oMLX
> 的 prefix cache 路径），同时 mlx_lm.server 那边 `run_perf.py` 只读
> `delta.content`，但 Qwen3.5 系 `thinking_default=true`，思考链全在
> `delta.reasoning` 字段里，于是 mlx_lm 的 `gen_chars=0`、`agg_tok_per_s=0`，wallclock
> 测出来"57.2s 飞快"是因为它根本没真正跑生成。两侧都修正后重测见下。

修正项：
- oMLX 启 `--paged-ssd-cache-dir ~/.omlx/cache --max-concurrent-requests 8`，删掉 `--no-cache`
- mlx_lm.server 启 `--decode-concurrency 8 --prompt-concurrency 8 --prompt-cache-size 8`
- `run_perf.py` 加 `chat_template_kwargs={"enable_thinking": false}`（apples-to-apples 关闭思考链）

#### 8.4.1 perf bench (`run_perf.py`，thinking OFF)

| 场景 | mlx_lm.server (8081) | oMLX (8000) | 备注 |
|---|---|---|---|
| Cold start TTFT | 1.29s | 0.95s | 都已 warm，含 cold prefix |
| Warm sustained TTFT mean (n=5) | 0.77s | 0.62s | oMLX 略快 |
| Warm sustained gen tok/s mean | 69.5 | 72.7 | 同档 |
| N=1 wall | 2.05s | 2.03s | 持平 |
| N=2 wall | 2.28s | 2.44s | mlx_lm 略快 |
| N=4 wall | 3.20s | 3.07s | oMLX 略快 |
| **N=8 wall** | **5.74s** | **5.27s** | **oMLX 快 ~9%** |
| Prefix cache ratio (long ×2) | 0.69 | 0.91 | mlx_lm 内存 cache 命中率反而更好 |
| Long-prompt prompt-eval | 1193 tok/s | 925 tok/s | mlx_lm prefill 更快 |

**结论**：在短/中 prompt + 短生成 workload 下，oMLX 跟 mlx_lm.server **基本同档**，
N=8 上 oMLX 略胜（continuous batching 的微弱优势）。**不是 33% 反向**。

#### 8.4.2 EXTENDED 50 × R=3 × concurrency 8（thinking ON，真实 tool_calling）

`tests/tool_calling/run_extended.py` 跑 SYSTEM_PROMPT (~1100 tok) + tool defs，
让模型决定调哪个 tool。Qwen3.5-9B `thinking_default=true`，会先生成思考再决策。

| 场景 | mlx_lm.server | oMLX | 备注 |
|---|---|---|---|
| Wallclock (N=8, 150 trials) | **169s** | **220s** | mlx_lm **快 23%** |
| Per-request mean elapsed | 8.87s | 11.58s | oMLX 慢 30% |
| Score (out of 150) | 111.0 (74.0%) | 111.7 (74.4%) | 评分等价 |
| completion_tokens mean | 214 | 210 | token 量持平，差距非生成体积 |
| prompt_tokens mean | 1112 | 1112 | 持平 |
| oMLX cache_efficiency 末端 | n/a | **0.0** | paged SSD cache 完全没命中 |

**反直觉点**：oMLX 主推的 paged SSD prefix cache 在 SYSTEM_PROMPT 重复 150 次的
理想场景下 **cache_efficiency = 0**，total_cached_tokens = 0。

**根因**（`/tmp/omlx-server.log` 一行说穿）：

```
omlx.scheduler - INFO - Enlarging paged cache block_size=256 to 2048
                       for ArraysCache hybrid model
omlx.scheduler - INFO - paged SSD-only mode: max_blocks=100000, block_size=2048
```

oMLX 对 hybrid model（Qwen3.5）自动把 cache block 放大到 **2048 token**。SYSTEM_PROMPT
1106 token 不足一块，paged SSD cache 按 block 存取，**子块前缀不计入命中**。要让
prefix cache 真生效，prompt 必须 > 2048 token（agent loop 里跨 turn 累积上下文才会
踩到这条路径）。dev-doc 当前 single-turn tool_calling workload 撞不上。

外加：默认 `--hot-cache-max-size 0` 关闭 in-mem hot cache，即便命中也走 SSD I/O，
对 ≤ medium prompt 可能净亏。

**结论**：在 dev-doc 当前 single-turn tool_calling workload 下，oMLX **比 mlx_lm.server
慢 23%**，paged SSD cache 触发不到。**真实价值仍在功能侧**（多 model 常驻 / 自动
分发 / `/v1/models` 列表 / `/health` / admin GUI / 一键集成槽位），不在性能侧。

#### 8.4.3 何时 oMLX 性能会真正胜出（推断，未本次实测）

- 长 SYSTEM_PROMPT > 2048 token + 多轮 reuse（agent loop）
- 多 model 同时挂着、按请求 model_id 分发（mlx_lm.server 单 model serve 必须重启）
- 跨用户共享前缀（多个 client 同 system message）

**目前阶段**：dev-doc 这种粗筛 workload 用 oMLX 是为了**多 model 不重启换模**便利，
不是为了吞吐。CCM 项目要拼吞吐另说。

参见 `tests/tool_calling/results/perf_omlx_qwen3.5-9b_20260508_201452.json`、
`perf_mlx_lm_*_20260508_201540.json`、`extended_results_20260508_201655.json`(oMLX)、
`extended_results_20260508_202126.json`(mlx_lm)。

### 8.5 还没验证的 oMLX 卖点

按 advisor 建议保留作后续坑，本次未做：

- `--hot-cache-max-size` > 0 开启后 prefix cache 在长 prompt + agent loop 场景的命中率
  （需要 prompt > 2048 token 才能踩到 paged SSD cache 路径）
- 多 model 并发：4B + 27B 同时挂着，agent 循环里两个 model 同时调，看
  EnginePool LRU swap 成本
- `omlx[grammar]` extra 装 xgrammar 跑结构化输出（macOS arm64 PyO3 build 风险）
- brew services / launchd 守护（当前手动 nohup）

这些验证后如果 oMLX 真有性能优势再考虑深用，目前阶段就当**多功能 OpenAI server**
用。

## 九、Admin GUI（dashboard / chat / benchmark / downloader）

oMLX **自带一套完整的 web admin**，挂在 `:8000/admin`。route inventory（来自
`/openapi.json`）：

### 9.1 顶层 HTML 页面（浏览器开）

| 路由 | 说明 |
|---|---|
| `GET /admin` / `/admin/` | 登录页，没设 `--api-key` 时自动跳到 setup 页 |
| `GET /admin/dashboard` | 主仪表盘（model 列表 + 实时 metrics） |
| `GET /admin/chat` | 内置 chat UI，可选 model + 调 sampling + thinking toggle |
| `GET /admin/static/{path}` | 自托管 Tailwind / Alpine.js / Lucide / Inter 字体 |

未设 `--api-key` 时 admin 处于 **Setup 状态**，所有 `/admin/api/*` 返回
`{"detail": "Admin authentication required"}`。第一次访问浏览器会提示设密码（写入
`~/.omlx/settings.json` 的 `auth.api_key`）。本次实测没设 key（dev-doc 仓内不放生产
密钥），所以下面 API endpoint 没有逐个实跑，只列。

### 9.2 Dashboard / Stats / 模型管理（`/admin/api/*`）

实时监控、配置改写、模型按需 load/unload —— 把 `mlx-cli daemon` 的需求全包了：

| 路由 | 用途 |
|---|---|
| `GET /admin/api/server-info` | 进程信息、版本、uptime |
| `GET /admin/api/device-info` | M5 Max 硬件、unified memory、Metal 设备 |
| `GET /admin/api/global-settings` / `POST` | 改 `~/.omlx/settings.json`（host/port/cache/sampling/integrations） |
| `GET /admin/api/models` | 模型列表（含 loaded/loading/size） |
| `POST /admin/api/models/{id}/load` / `unload` | 主动 load / 驱逐 |
| `POST /admin/api/reload` | rescan model_dir |
| `GET/POST/PUT/DELETE /admin/api/models/{id}/profiles[/{name}]` | per-model sampling profile（temperature/top_p/system_prompt 等）|
| `POST /admin/api/models/{id}/profiles/{name}/apply` | 切默认 profile |
| `GET/POST/PUT/DELETE /admin/api/profile-templates[/{name}]` | 跨 model 的 profile 模板 |
| `GET /admin/api/grammar/parsers` | xgrammar 路径（需装 `omlx[grammar]`） |
| `GET /admin/api/logs` | server log 末尾 N 行 |
| `GET /admin/api/stats` / `POST /clear` / `clear-alltime` | request 计数 / token 统计 / cache_efficiency |
| `POST /admin/api/ssd-cache/clear` | 清空 paged SSD cache |
| `POST /admin/api/cache/probe` | 给定 prompt，返回会命中哪些 cache block |
| `GET /admin/api/sub-keys` / `POST` / `DELETE` | 子 key 管理（粒度授权） |

### 9.3 内置 benchmark（`/admin/api/bench/*`）

oMLX **自带一个 perf 跑分系统**，**比我们的 `tests/tool_calling/run_perf.py` 全功能**：

| 路由 | 用途 |
|---|---|
| `POST /admin/api/bench/start` | 启 perf bench (prefill / generation tok/s) |
| `GET /admin/api/bench/{id}/stream` | SSE 流式拿进度 |
| `POST /admin/api/bench/{id}/cancel` / `GET /results` | cancel / 取结果 |
| `POST /admin/api/bench/accuracy/queue/add` | accuracy bench（含 ground truth）排队 |
| `GET /admin/api/bench/accuracy/results` | 查 accuracy bench 结果 |

**启示**：将来 dev-doc 重新跑 perf bench 不必自己写 `run_perf.py`，可以直接 hit
`/admin/api/bench/start`，省一层维护。但本次任务为了 apples-to-apples 对 mlx_lm.server，
仍然走自己的 `run_perf.py`。

### 9.4 模型 in-browser 下载（`/admin/api/hf/*`、`/admin/api/ms/*`）

直接在 admin GUI 里搜 + 拉 HF / ModelScope 模型：

| 路由 | 用途 |
|---|---|
| `GET /admin/api/hf/recommended` | oMLX 推荐 model list |
| `GET /admin/api/hf/search` | HF model 搜索 |
| `GET /admin/api/hf/model-info` | 拉 model card / config |
| `POST /admin/api/hf/download` | 启动下载任务 |
| `GET /admin/api/hf/tasks` | 看正在下的任务 |
| `POST /admin/api/hf/cancel/{task_id}` / `retry/{task_id}` / `DELETE /task/{task_id}` | 任务控制 |
| `GET /admin/api/hf/models` / `DELETE /admin/api/hf/models/{name}` | 已下模型列表 / 删 |
| 同上 `/admin/api/ms/*` | ModelScope（国内镜像）镜像版 |

→ 等于把 `scripts/setup-omlx-models.sh` 那套手动 ln HF cache 的活做成 GUI 流程。
未来新人上手 oMLX 直接浏览器拉，不需要懂 HF cache 结构。

### 9.5 内置 quantizer（`/admin/api/oq/*`）

oQ = oMLX Quantizer。不光是 server，还能在 GUI 里把已下模型量化：

| 路由 | 用途 |
|---|---|
| `GET /admin/api/oq/models` | 可量化的 model 列表 |
| `GET /admin/api/oq/estimate` | 估算量化后大小 + 时间 |
| `POST /admin/api/oq/start` | 启动 quant 任务 |
| `GET /admin/api/oq/tasks` / `cancel/{id}` / `DELETE /task/{id}` | 任务控制 |
| `/admin/api/upload/*` | 把量化产物上传回 HF |

→ 把 `mlx_lm.convert` 这种 CLI 操作图形化。

### 9.6 一键集成 OpenClaw / OpenCode / Codex / Pi（`settings.json: integrations`）

`~/.omlx/settings.json` 里有这两个段：

```json
"claude_code": {
    "context_scaling_enabled": false,
    "target_context_size": 200000,
    "mode": "cloud",
    "opus_model": null,
    "sonnet_model": null,
    "haiku_model": null
},
"integrations": {
    "codex_model": null,
    "opencode_model": null,
    "openclaw_model": null,
    "pi_model": null,
    "openclaw_tools_profile": "coding"
}
```

意思是 oMLX 给 **Anthropic 兼容 `/v1/messages` 接口**当本地 backend，把
Claude Code / OpenClaw / OpenCode / Codex / Pi 这些 agent 客户端 alias 到本地 model：
admin GUI 里点几下 → 客户端切 base_url 就用本地 27B/9B 跑。**这是 oMLX 相对
mlx_lm.server 最有杀伤力的差异化能力**——mlx_lm.server 没有 Anthropic 兼容路径。

dev-doc 这边 work key (FlySafe) + 国内 3PL print agent 长期都需要"本地 LLM 接 Claude
Code 形态客户端"，这个槽位日后会用上。本次没真启 integration，标记保留。

### 9.7 没有截图的限制

agent 跑在 dev box 上，没浏览器，本次没法截图描述 UI 视觉。真实评估
dashboard 视觉要 yuanwei 在 m5max 本机或 Tailscale 网内浏览器打开
`http://m5max:8000/admin` 自己看一遍。Setup 流程提示：第一次访问会让设
admin password（写入 `auth.api_key`），设完后所有 `/admin/api/*` 才能命中。

## 十、Cleanup / 关停

```bash
# 关 oMLX server
ssh yuanwei@m5max 'pkill -f "omlx serve"'

# 验证
ssh yuanwei@m5max 'lsof -iTCP:8000 -sTCP:LISTEN'

# 想完全 reset：
ssh yuanwei@m5max 'rm -rf ~/.venvs/omlx ~/omlx-src ~/.omlx'
```

⚠️ 注意 `~/.omlx/` 下除了 `models/` 还有 `settings.json` + 自动生成的 auth secret，重装时清掉就行。

## 十一、参考与回链

- 上游：`https://github.com/jundot/omlx` (本文 pin commit `7395ca5`，formula 对应 v0.3.8)
- 并列文档：`docs/m5max-mlx-setup.md`（`mlx_lm.server` 主线，依然推荐做单 model benchmark 用）
- 自研 CLI 设计：`docs/m5max-mlx-cli-design.md`（**已 deprecated**，所有功能 oMLX 已现成提供，详见 §9 admin GUI）
- 工具调用矩阵测试：`tests/tool_calling/`（dev-doc 主线已切 oMLX，详见 §8）
- 选型背景：`docs/deployment-architecture-design.md`（按步骤分模型 + 4B/27B 双常驻方案，是 oMLX EnginePool 真正用上的场景）
