# M5 Max MLX-LM 本地推理 Setup

> 日期：2026-05-07
> 状态：**deprecated** —— 2026-05-08 起 dev-doc 测试 infra 已切 oMLX 当主 backend，本文档保留作历史参考 + mlx_lm.server fallback playbook
> 主文档：`docs/m5max-omlx-setup.md`（canonical，含 migration 实测记录）
> 范围：M5 Max 128GB 上跑 MLX-LM 作为本地开发箱，不上生产
>
> ⚠️ **何时还会用到本文档**：
> 1. perf 对比 oMLX vs mlx_lm.server 时（设 `BASE_URL_OVERRIDE=http://m5max:8081/v1`）
> 2. oMLX 装机失败的临时 fallback
> 3. 单 model 极简 benchmark（不想引入 oMLX 依赖图）

---

## 一、Quick-start

Mac 在手边、粘贴即跑（**国内 m5max 实测 2026-05-07 通过**）：

```bash
brew install uv && uv python install 3.12
uv venv --python 3.12 ~/.venvs/mlx && source ~/.venvs/mlx/bin/activate
uv pip install mlx-lm hf_transfer
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1   # 国内必须
mlx_lm.generate --model mlx-community/Qwen3.5-4B-4bit --prompt "你好" --max-tokens 50
```

几条命令跑完，本机 MLX-LM 推理链路通了。每条都是独立稳定的，不要省略 Python 3.12 (mlx 要求 ≥3.10)、`hf_transfer`（国内不稳网络下稳健下载）、HF 镜像三件 —— 任何一件缺会陷入冗长 debug，详见第五节和第十节踩坑记录。后续段落是详解、内存预算、服务化、对照基线。

## 二、背景与适用场景

硬件：M5 Max 128GB unified memory，Apple Silicon。

定位：**本地开发箱**——蒸馏流水线打样、prompt 调试、工具调用对比、LoRA 试跑。**不上生产**——生产仍走 deployment-architecture-design.md 里的云 GPU 路线。

为什么先在 M5 Max 上做：呼应 deployment-architecture-design.md 的坑 2"先跑通再优化"——本机零成本、零等待、零审批，是验证流水线最快的环境。等链路打磨好再迁云端，比在云上反复 debug 便宜得多。

## 三、为什么 MLX 不 Ollama

主线推理框架 **MLX-LM**（Apple 原生），对照基线保留 **llama.cpp**（直接用，不通过 Ollama）。

不用 Ollama：在 Mac 上是 llama.cpp 封皮，多套了一层抽象，对 token 级行为可控度差（采样参数、stop tokens、grammar 约束、tool call parser 都不暴露），微调出来的领域模型上去之后行为定位困难。LM Studio 仅作演示工具，日常开发用不上。

Python 管理用 **uv**（Astral 标准、装 mlx-lm 比 pip 快一个数量级、Brewfile 友好）。本仓库此前没有 Python 约定，本文档建立这一项。

## 四、内存预算（128GB 上能跑什么）

模型本身只是一部分，KV cache + 激活 + 上下文 buffer 还要留余量，IDE / 浏览器 / 系统自身还要 30-40GB。下表给参考值：

| 模型 | 量化 | 权重大小 | 备注 |
|------|------|---------|------|
| Qwen3.5-4B | 4bit | ~2 GB | 冒烟测试主力 |
| Qwen3.5-4B | 8bit | ~4 GB | 工具调用对比常用档位 |
| Qwen3.5-4B | bf16 | ~8 GB | 微调起点 |
| Qwen3.5-9B | 4bit | ~5 GB | 4B 不够时备选 |
| Qwen3.6-27B（Dense 28B）/ Gemma 4 31B | 4bit | ~17 GB | 主力推理候选 Dense |
| Qwen3.6-35B-A3B（MoE 36B 激活 3B） | 4bit | ~22 GB | 主力推理候选 MoE |
| 70B class | 4bit | ~39 GB | 偶尔验证用，速度感人 |

**最甜区：4B + 27/31B 双常驻（约 19 GB 权重 + 20-30 GB KV/激活）+ 60 GB 余量给 IDE / 浏览器 / 系统**。两个模型同时挂着，agent 循环里推理步骤走主力（Qwen3.6-27B / Gemma 4 31B），工具调用步骤走 4B（参考 deployment-architecture-design.md 的分工方案），无需切换、无需重新加载。

## 五、安装步骤

### 5.1 前置

Xcode CLI + Homebrew，按 mac-developer-tooling.md 第三节"新 Mac 一键引导"流程。如果是已配置好的开发机，跳过。

### 5.2 uv

```bash
brew install uv
```

如果维护 Brewfile，加一行 `brew "uv"`。

### 5.3 Python venv（必须 3.10+）

⚠️ **关键约束**：mlx 核心库 (`mlx==0.31+`) 要求 **Python >= 3.10**。macOS 系统自带的 `/usr/bin/python3` 是 3.9.6，**不能直接用**——uv 在 3.9 环境会 resolve 到老旧的 mlx-lm 0.29.x，缺新模型架构（如 `qwen3_5`），跑 Qwen 3.5/3.6 系列会报 `ModuleNotFoundError: No module named 'mlx_lm.models.qwen3_5'`。

```bash
# uv 自带 Python 下载，不污染系统
uv python install 3.12

# 用 3.12 创建 venv
uv venv --python 3.12 ~/.venvs/mlx
source ~/.venvs/mlx/bin/activate
python --version  # 应输出 Python 3.12.x
```

约定：`~/.venvs/mlx` 专给 MLX 链路用，跟其它 Python 项目隔离。每次新开 shell 重新 `source` 一下。

### 5.4 mlx-lm + hf_transfer

```bash
uv pip install mlx-lm hf_transfer
```

verify：

```bash
python -c "import mlx_lm, mlx; print('mlx-lm:', mlx_lm.__version__); print('mlx:', mlx.__version__)"
```

预期 `mlx-lm 0.31.3`（PyPI 最新，2026-05-07 实测）+ `mlx 0.31.2`（自动作为 mlx-lm 依赖装下来）+ `mlx-metal`（Metal backend，Apple Silicon 必需）。

`hf_transfer` 是 HuggingFace 的 Rust 下载器，配合环境变量 `HF_HUB_ENABLE_HF_TRANSFER=1` 启用，**国内走 hf-mirror 必需**——能解决纯 Python 下载器在不稳定网络下 `IncompleteRead` / `SSLEOFError` 的卡死问题。

`huggingface-hub` 1.x 与推理调用 (`snapshot_download` Python API) 完全兼容，不需要降级；变化只在打包：`[cli]` 和 `[hf_transfer]` 两个 extra 在 1.x 已废弃。所以 `hf_transfer` 必须 **单独装**（`uv pip install hf_transfer`），不能再用 `huggingface_hub[hf_transfer]` 写法（在 1.x 会安静地不装）。`huggingface-cli` 命令仍在（默认包含），新统一命令 `hf` 也可用。

### 5.5 拉模型

### ⚠️ 国内网络：必须走 HF 镜像

m5max 在国内直连 `huggingface.co` ping 100% 丢包、SSL EOF。所有 HF 拉取必须走镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

实测：m5max → hf-mirror.com 200 OK ~3.4s，可用。建议把这一行加到 `~/.zshrc` 或 venv 的 `activate` 末尾，每次新 shell 自动生效。Modelscope（`https://www.modelscope.cn`）更快但需要换 API 客户端，不直接走 hf_hub_download，本文档不展开。

### 拉模型

两种方式：

```bash
# 方式 A：mlx_lm.generate 自动拉（推荐，懒加载，HF_ENDPOINT 已设）
mlx_lm.generate --model <repo-id> --prompt "ping" --max-tokens 5

# 方式 B：CLI 预拉（带宽不稳时建议）—— huggingface_hub 1.x 起 CLI 默认包含
huggingface-cli download <repo-id>
# 或新统一命令（同等功能）：
# hf download <repo-id>
```

冒烟用的小模型（~2GB）和主力候选 Qwen 3.6 / Gemma 4 系列具体 repo-id 见第九节"后续工作"。本步骤的冒烟模型选择见 5.6。

### 5.6 冒烟测试

```bash
mlx_lm.generate \
  --model mlx-community/Qwen3.5-4B-4bit \
  --prompt "用一句中文解释什么是 unified memory。" \
  --max-tokens 100
```

**实测（2026-05-07，M5 Max 128GB / Apple M5 Max 40 GPU cores / mlx-lm 0.31.3 / Python 3.12.13 / mlx-community/Qwen3.5-4B-4bit）：**

| 阶段 | 指标 |
|------|------|
| Prompt eval | 18 tokens @ **19.78 tok/s** |
| Generation | 100 tokens @ **167.57 tok/s** |
| Peak memory | 2.525 GB |

Qwen 3.5 4B 是 reasoning 模型（输出含 `Thinking Process` + 最终答复两段，参考 DeepSeek-R1 / o1 风格）。167 tok/s 已远超人类阅读速度，4B 这个尺寸在 M5 Max 上完全实时。要纯 instruct 输出（不要 thinking 段）的具体处理见 5.8 节末尾。

### 5.7 服务化

OpenAI-compatible HTTP server。

#### 首次（前台跑，看 log）

```bash
# 推荐：绑定 m5max 自己的 Tailscale IP，不暴露 LAN/公网
mlx_lm.server \
  --model mlx-community/Qwen3.5-4B-4bit \
  --host 100.67.236.97 \
  --port 8080
```

`100.67.236.97` 是 m5max 的 Tailscale IP（`tailscale ip -4` 拿到）。绑这个 IP 的好处：Tailscale 网络内任何被 ACL 授权的设备都能直连 `http://m5max:8080`（Tailscale magic DNS），但 LAN/WiFi 上的设备看不到端口（隔离公司/咖啡店 WiFi 风险）。**比 `0.0.0.0`（暴露所有接口）安全，比 `127.0.0.1`（仅本机）实用**。

常用 flags：

| Flag | 作用 |
|------|------|
| `--model` | HF repo id 或本地路径，必填 |
| `--host` / `--port` | 监听地址。三档选择：`127.0.0.1`（仅本机自用）/ `100.67.236.97`（Tailscale IP，推荐）/ `0.0.0.0`（所有接口含公网，**慎用**：mlx_lm.server 自带 warning「仅基础安全检查」） |
| `--adapter-path` | LoRA adapter 路径（推理时挂载微调权重） |
| `--draft-model` | speculative decoding 的草稿模型 |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` |
| `--temp` / `--top-p` | 采样默认值（请求里也可覆盖） |

#### 复用启动（重启/新 shell/Mac 重开后）

完整 5 步，**任何一步缺都会报错或行为异常**：

```bash
# 1. brew PATH（如果 .zprofile 没固化，每次新 shell 都要）
eval "$(/opt/homebrew/bin/brew shellenv)"

# 2. 激活 venv（venv 不会自动激活）
source ~/.venvs/mlx/bin/activate

# 3. 国内 HF 镜像 + Rust 下载器（下次拉新模型才用得上，启已有模型也不影响设着）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# 4. 后台起 server（caffeinate 防睡眠 + nohup 脱离终端 + 绑 Tailscale IP）
caffeinate -i nohup mlx_lm.server \
  --model mlx-community/Qwen3.5-4B-4bit \
  --host 100.67.236.97 --port 8080 \
  > /tmp/mlx-server.log 2>&1 &
disown

# 5. 等 ready（首次约 3 秒，模型已缓存）
for i in $(seq 1 30); do
  curl -s -m 1 http://m5max:8080/v1/models > /dev/null && echo "READY after ${i}s" && break
  sleep 1
done
```

把 1-4 步固化到 `~/.zprofile` 或一个启动脚本（如 `~/bin/mlx-up.sh`）就一行起服务。要做成开机自启走 launchd plist，不在本文档展开。

#### 状态查询 / 关停

```bash
# 看进程
pgrep -fl mlx_lm.server

# 看 log
tail -f /tmp/mlx-server.log

# 关停（连同 caffeinate 父进程）
pkill -f mlx_lm.server
pkill -f "caffeinate -i nohup mlx_lm"
```

### 5.8 OpenAI-compatible 端点验证

```bash
curl http://m5max:8080/health

curl http://m5max:8080/v1/models

curl http://m5max:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-4B-4bit",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'
```

端点：`/v1/chat/completions`、`/v1/models`、`/health`。完全 OpenAI-compatible，OpenAI SDK / LangChain / LlamaIndex 直接换 base_url 就能用。

**实测（2026-05-07，端到端）：**

```
=== /v1/models ===
{ "object": "list", "data": [{ "id": "mlx-community/Qwen3.5-4B-4bit", ... }] }

=== /v1/chat/completions ===
{
  "id": "chatcmpl-...",
  "system_fingerprint": "0.31.3-0.31.2-macOS-26.4.1-arm64-arm-64bit-applegpu_g17s",
  "model": "mlx-community/Qwen3.5-4B-4bit",
  "choices": [{
    "finish_reason": "length",
    "message": { "role": "assistant", "reasoning": "Thinking Process:\n..." }
  }],
  "usage": { "prompt_tokens": 15, "completion_tokens": 80, "total_tokens": 95 }
}
```

⚠️ **Reasoning model 字段差异**：Qwen 3.5 是 thinking model，response 里 `message.reasoning` 字段（**不是** `message.content`）—— OpenAI o1/o3 风格。客户端代码若直接读 `choices[0].message.content` 会拿到空 / undefined。适配方式：

| 客户端 | 适配 |
|--------|------|
| OpenAI 官方 SDK | 兼容（SDK 会保留未知字段） |
| 手写 JSON 解析 | 同时读 `reasoning` 和 `content`，二选一 |
| LangChain `ChatOpenAI` | 0.2+ 已支持 reasoning models |
| 业务代码 | 加 `msg.get("content") or msg.get("reasoning")` 兜底 |

如果不需要 thinking 链路占 token：**实测 `/no_think` 指令在 Qwen 3.5 4B 上无效**（thinking 仍跑、content 仍空，2026-05-07 m5max 验证）。要纯 content 输出走非 reasoning 系列模型，verified 选项是 `mlx-community/Qwen3-4B-Instruct-2507-4bit`（Qwen 3 系列 2025-07 instruct 刷新版）。Qwen 3.6 没出 4B class，要 4B 纯 instruct 必须回 Qwen 3 / 3.5 之外的 instruct fork（待主力选型阶段一并 verify）。

### 5.9 客户端调用示例

四种典型场景，**全部 2026-05-07 在 m5max 上实测通过**。

#### A. m5max 自己（WezTerm）—— 最简

```bash
# 单轮对话
curl http://m5max:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-4B-4bit",
    "messages": [{"role": "user", "content": "用三句话解释什么是 unified memory"}],
    "max_tokens": 300,
    "temperature": 0.7
  }' | python3 -m json.tool

# 多轮对话（含 system prompt）
curl http://m5max:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-4B-4bit",
    "messages": [
      {"role": "system", "content": "你是 3PL 仓储领域的专家"},
      {"role": "user", "content": "WMS 选型关键考量？"},
      {"role": "assistant", "content": "看库存精度、订单吞吐、对接 ERP/电子面单的能力。"},
      {"role": "user", "content": "对接电子面单具体看什么？"}
    ],
    "max_tokens": 500
  }' | python3 -m json.tool
```

#### B. 从其它机器（推荐：Tailscale 直连）

server 绑 m5max Tailscale IP `100.67.236.97`（5.7 节默认），Tailscale 网内任何被 ACL 授权的设备直接通过 magic DNS 访问：

```bash
# 调用方任何机器（已加入同 Tailscale 账号）
curl http://m5max:8080/v1/models
# 或显式 IP
curl http://100.67.236.97:8080/v1/models
```

不需要 SSH tunnel —— Tailscale 自带加密 + 设备级认证，等价"私有加密 LAN"。

**降级备选：SSH tunnel**（如果 server 临时绑 `127.0.0.1` 或不在 Tailscale 网中）：

```bash
ssh -fN -L 8080:127.0.0.1:8080 yuanwei@m5max
curl http://127.0.0.1:8080/v1/models  # 走 tunnel
pkill -f "ssh -fN -L 8080"
```

#### C. Python OpenAI SDK

```python
# uv pip install openai
from openai import OpenAI

client = OpenAI(
    base_url="http://m5max:8080/v1",   # Tailscale magic DNS
    api_key="dummy",   # MLX server 不校验 key，但 SDK 必传
)

resp = client.chat.completions.create(
    model="mlx-community/Qwen3.5-4B-4bit",
    messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
    max_tokens=200,
    temperature=0.7,
)

# Reasoning model 关键：先读 reasoning 再读 content（实测 SDK 会自动暴露 reasoning 属性）
msg = resp.choices[0].message
output = getattr(msg, "reasoning", None) or msg.content or ""
print(output)
print(f"\nUsage: {resp.usage}")
```

实测 `model_dump()` 返回的 keys：`['content', 'refusal', 'role', 'annotations', 'audio', 'function_call', 'tool_calls', 'reasoning']` —— `reasoning` 是 SDK 已识别的字段，不是 unknown extra。

#### D. 工具调用（端到端）

```bash
curl http://m5max:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-4B-4bit",
    "messages": [{"role": "user", "content": "查询设备 CNC-001 的温度"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "query_device_params",
        "description": "查询指定设备的运行参数",
        "parameters": {
          "type": "object",
          "properties": {
            "device_id": {"type": "string"},
            "param": {"type": "string", "enum": ["temperature", "vibration", "rpm"]}
          },
          "required": ["device_id", "param"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 300
  }' | python3 -m json.tool
```

**实测响应（精简）**：

```json
"choices": [{
  "finish_reason": "tool_calls",
  "message": {
    "content": null,
    "reasoning": "用户想要查询设备 CNC-001 的温度参数...",
    "tool_calls": [{
      "function": {
        "name": "query_device_params",
        "arguments": "{\"device_id\": \"CNC-001\", \"param\": \"temperature\"}"
      },
      "type": "function",
      "id": "7f8f67e9-..."
    }]
  }
}]
```

注意点：

- reasoning model 在 tool call 前还会先输出 thinking（"我需要使用 query_device_params..."），但 `tool_calls` 字段独立填充，**客户端读 `tool_calls` 不受 reasoning 干扰**
- `arguments` 是 **JSON 字符串**（不是嵌套对象），要 `json.loads()` 才能拿到 dict
- `finish_reason` 此时是 `"tool_calls"` 而非 `"stop"` / `"length"`，agent loop 用这个分支

### 5.10 多模态：mlx-vlm（Gemma 4 / Qwen2.5-VL 等）

mlx-lm 只支持**纯文本**架构（Qwen 系列、Llama text 等）。**Gemma 4 整个系列是 multimodal 架构**（text + vision + audio 三模态拼成 `Gemma4ForConditionalGeneration` 类），mlx-lm load 时会报 `ValueError: Received N parameters not in model`（缺 vision_tower / audio encoder 的 weight 定义）—— **不是 mlx-lm bug**，是工具栈选错。

要跑 Gemma 4，用 sister 库 **mlx-vlm**：

```bash
uv pip install mlx-vlm
```

启动方式跟 mlx_lm.server 完全对齐（同样 OpenAI-compatible `/v1/chat/completions`）：

```bash
mlx_vlm.server --model mlx-community/gemma-4-e4b-it-4bit \
  --host 100.67.236.97 --port 8080
```

**实测（2026-05-08，Gemma 4 E4B）**：

| 场景 | 结果 |
|---|---|
| 纯文本 chat | ✅ 中文输出流畅，`content` 字段填充（instruct 不是 reasoning） |
| 工具调用 | ✅ `tool_calls` 数组正确填充 `query_device_params(device_id="CNC-001", param="temperature")`，`finish_reason: "tool_calls"`。**mlx-lm issue #1096 在 mlx-vlm 上不存在** |
| 多模态 image 识别 | ✅ 颜色 + 形状正确识别（OpenAI vision API 标准 `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` 格式） |

**mlx-lm 与 mlx-vlm 选择规则**：
- 看 HF model card 的 `library_name` 字段：`mlx` → mlx-lm，`mlx-vlm` → mlx-vlm
- 看 `config.json` 里有没有 `vision_config` / `audio_config`
- 看 `architectures` 是否带 `*ForConditionalGeneration`（多模态特征）

`mlx-community` namespace 同时包含两类模型，**仅看 repo 名字 `gemma-4-e4b-it-4bit` 不带 vlm 后缀容易误判**——必须 verify config.json 的 model_type / architectures。

**矩阵 wrapper（`tests/tool_calling/run_matrix_local.sh`）已支持 dual-server 自动分发**：Gemma 4 系列走 mlx_vlm.server，其它走 mlx_lm.server，端口共用 8080，client 接口（OpenAI-compatible）完全相同。

## 六、工具调用验证

本仓库已有的 `tests/tool_calling/` 接入完成（2026-05-08）：
- `config.py` 已切到 `http://m5max:8080/v1` + dummy API key + 7 候选 MODELS 列表
- `run_test.py` 加 reasoning / elapsed_s / usage 字段记录 + 中文比例观察
- `run_matrix_local.sh` 自动串行起 mlx_lm.server 加载每个候选跑测试
- 一键跑：`bash tests/tool_calling/run_matrix_local.sh`

注意 dev-doc 这套测试只做 **Mac 上的开发期粗筛**，真正的智能体 / 生产场景评测在 CCM 项目里跑（参考记忆 `project_ccm_agent_engine.md`）。

**已知风险：**

| 模型 family | native tool_calls | 备注 |
|-------------|-------------------|------|
| Qwen3 / Qwen3.5 / Llama / DeepSeek | ✅ 端到端工作 | family parser 内建。Qwen 3.5 4B 实测 `tool_calls` 数组 + `finish_reason: "tool_calls"` 正确（2026-05-07，详见 5.9 D） |
| **Gemma 4** | ✅ **mlx-vlm 上工作（不是 mlx-lm）** | Gemma 4 是多模态架构 mlx-lm 加载不了，要走 mlx-vlm（5.10 节）；issue #1096 警告原本针对 mlx-lm，**在 mlx-vlm 上不存在**（实测 2026-05-08，详见 5.10）|

另外 MLX-LM 没有 server-level grammar / structured-output 约束（不像 llama.cpp 的 GBNF），结构化输出全靠模型自身指令遵循 + family parser。需要强结构约束的场景目前要切 llama.cpp 验证。

## 七、llama.cpp 对照基线

```bash
brew install llama.cpp
```

何时切：MLX 行为诡异时（采样不对、tool_calls 漏拼、结构化输出走样）用 llama.cpp 跑同一个 GGUF 量化版本，定位是模型问题还是 MLX 后端问题。日常开发不用。

## 八、LM Studio

仅作演示工具（给非工程同事看一眼模型能力），日常开发用不上。

## 九、后续工作

- [x] **主力候选 mlx 矩阵实测**（2026-05-08 完成，8 候选 × 60 inference）
  - 详细 case-by-case 数据：`tests/tool_calling/results/results_*.json`
  - 跨候选对比表 + 选型建议：`deployment-architecture-design.md § 六.1`
  - 单 model 重跑：`MATRIX_ONLY=<repo-id> bash tests/tool_calling/run_matrix_local.sh`
- [ ] LoRA 微调流程（`mlx_lm.lora`）→ 后续单独文档
- [ ] **真正智能体场景评测在 CCM 项目**（dev-doc 这套是 Mac 粗筛，参考记忆 `project_ccm_agent_engine.md`）
- [ ] Gemma 4 31B / 26B-A4B 等更大 multi-modal 候选 mlx-vlm 上跑（需自行 `mlx_lm.convert` 量化或拉社区版）

## 十、踩坑记录

2026-05-07 setup 期间踩过的坑，**新机引导照抄 Quick-start 即可避开全部**：

### 坑 1：macOS 系统 Python 3.9.6 太老，uv 静默 resolve 到老版本 mlx-lm

**现象**：`uv pip install mlx-lm` 装到 0.29.1（不是 PyPI 最新 0.31.3），跑 `mlx_lm.generate --model mlx-community/Qwen3.5-4B-4bit` 报 `ModuleNotFoundError: No module named 'mlx_lm.models.qwen3_5'`。

**原因**：mlx (核心库) 0.31.x 明确 `requires_python >= 3.10`。在 Python 3.9 venv 里 uv 自动降级到能兼容的 0.29.x，但 0.29.x 没有 `qwen3_5` 模型架构定义，跑 Qwen 3.5/3.6 系列直接报错。

**解**：`uv python install 3.12 && uv venv --python 3.12 ~/.venvs/mlx`。uv 自带 Python 下载，不污染系统。

### 坑 2：国内直连 huggingface.co 100% 不通

**现象**：`mlx_lm.generate` 拉模型时 `SSLError(SSLEOFError)` 或 `MaxRetryError`。

**实测**：m5max → huggingface.co ping 100% 丢包；hf-mirror.com / modelscope.cn 都通。

**解**：`export HF_ENDPOINT=https://hf-mirror.com`。

### 坑 3：纯 Python 下载器在不稳网络下 IncompleteRead

**现象**：HF_ENDPOINT 切到 hf-mirror 后下载到一半 `IncompleteRead(4090 bytes read, 12294 more expected)`，14 分钟才 fail。

**原因**：`huggingface_hub` 默认用 Python 多线程分片下载，单分片连接断了不重传整体失败。

**解**：装 Rust 实现的 `hf_transfer`：`uv pip install hf_transfer` + `export HF_HUB_ENABLE_HF_TRANSFER=1`。注意 hf_hub 1.x 后 `[hf_transfer]` 不再是 extra，**必须单独装**（`pip install "huggingface_hub[hf_transfer]"` 在 1.x 会安静地不装）。

### 坑 4：Qwen 3.5 系列是 reasoning model，response 字段不是 `content`

**现象**：`/v1/chat/completions` 返回 200 OK 但 `choices[0].message.content` 是空，把客户端代码搞蒙。

**原因**：Qwen 3.5 是 thinking model（DeepSeek-R1 / OpenAI o1 风格），输出走 `message.reasoning` 字段。

**解**：客户端读 `msg.get("content") or msg.get("reasoning")`，或换非 reasoning 模型（如 `Qwen3-4B-Instruct-2507`）。注意 `/no_think` 指令在 Qwen 3.5 4B 上**实测无效**，不要写进客户端代码。详见 5.8 节适配表。

### 坑 5：`uv venv` 不装 `pip` 命令

**现象**：venv 里 `pip list` 报 `command not found`。

**原因**：uv 自家管理依赖，`uv venv` 创建的 venv **不包含** `pip` binary。

**解**：用 `uv pip list` 替代 `pip list`；要 pip 命令的脚本另装 `uv pip install pip`。日常维护用 `uv pip` 子命令体系即可。
