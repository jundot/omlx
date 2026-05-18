# Reasoning / Thinking API Reference

Authoritative reference for the reasoning (chain-of-thought) features of
the chat endpoints in flyto-mlx / oMLX: how the thinking block is
returned separately from the answer, and how to control or disable it.

English version first. Chinese translation follows below.

---

## English

### 1. Background

Reasoning-capable models (Qwen3 / Qwen3.5, DeepSeek V4, GPT-OSS,
MiniMax, etc.) emit their chain-of-thought wrapped in `<think>...</think>`
tags before the final answer. oMLX parses these tags on the server side,
so callers never have to strip `<think>` themselves — the thinking text
and the answer text arrive in separate fields.

Two things are covered here:

1. **Reading** the reasoning — where the thinking text shows up in the
   response.
2. **Controlling** the reasoning — how to cap it, or turn it off.

Both the OpenAI-compatible endpoint (`/v1/chat/completions`) and the
Anthropic-compatible endpoint (`/v1/messages`) are supported. The field
names differ; the behaviour is the same.

### 2. Reading the reasoning

#### 2.1 OpenAI — `/v1/chat/completions`

The thinking block is returned as a separate string field
`reasoning_content`, alongside the normal `content`:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning_content": "The user asks for 17 * 23. 17 * 23 = 391.",
      "content": "17 × 23 = 391."
    }
  }]
}
```

`content` holds only the final answer. `reasoning_content` is `null`
when the model did not think (or thinking was disabled).

In streaming mode the deltas arrive in the same two fields — all
`reasoning_content` deltas first, then the `content` deltas:

```
data: {"choices":[{"delta":{"reasoning_content":"The user asks"}}]}
data: {"choices":[{"delta":{"reasoning_content":" for 17 * 23."}}]}
data: {"choices":[{"delta":{"content":"17 × 23 = 391."}}]}
```

#### 2.2 Anthropic — `/v1/messages`

The response `content` is a block array. The thinking block (`type:
"thinking"`) comes before the answer block (`type: "text"`):

```json
{
  "content": [
    {"type": "thinking", "thinking": "The user asks for 17 * 23. ..."},
    {"type": "text", "text": "17 × 23 = 391."}
  ]
}
```

A response with no thinking simply omits the `thinking` block.

### 3. Controlling the reasoning

There are three request-level controls. Use the first one that fits —
they are listed cheapest-to-learn first.

#### 3.1 `reasoning_effort` — OpenAI, recommended

`reasoning_effort` is the standard OpenAI field. It is the simplest way
to control thinking and needs no oMLX-specific knowledge:

| Value | Effect | Default thinking budget |
|---|---|---|
| `"off"` | thinking disabled — model answers directly | — |
| `"low"` | short thinking | 512 tokens |
| `"medium"` | moderate thinking | 2048 tokens |
| `"high"` | long thinking | 8192 tokens |

```json
POST /v1/chat/completions
{
  "model": "qwen-dense-27b",
  "messages": [{"role": "user", "content": "..."}],
  "reasoning_effort": "low"
}
```

The budget is a cap, not a target: a model that finishes its reasoning
early emits `</think>` on its own and the cap is never reached. When the
cap *is* hit, the server forces the `</think>` token so generation moves
on to the answer (see §5).

The default budgets above are server defaults and can be overridden
per model — see §4.2.

#### 3.2 `thinking_budget` — OpenAI, explicit numeric cap

When you want an exact token cap rather than a named level, send an
integer `thinking_budget`:

```json
{
  "model": "qwen-dense-27b",
  "messages": [...],
  "thinking_budget": 1200
}
```

`thinking_budget` always wins over `reasoning_effort` if both are sent.

#### 3.3 `chat_template_kwargs.enable_thinking` — low-level toggle

The underlying on/off switch is the chat-template kwarg
`enable_thinking`. `reasoning_effort: "off"` is sugar for setting this
to `false`. You can also set it directly:

```json
{
  "model": "qwen-dense-27b",
  "messages": [...],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Prefer `reasoning_effort: "off"` — `enable_thinking` is documented here
only so existing callers recognise it.

#### 3.4 Anthropic — `thinking`

The Anthropic endpoint uses the standard `thinking` object instead of
`reasoning_effort`:

```json
POST /v1/messages
{
  "model": "qwen-dense-27b",
  "messages": [...],
  "thinking": {"type": "enabled", "budget_tokens": 1200}
}
```

`type: "disabled"` turns thinking off. `budget_tokens` is the numeric
cap and behaves exactly like `thinking_budget` on the OpenAI side.

### 4. Server-side defaults

#### 4.1 Precedence

For a single request the effective thinking budget resolves in this
order, first match wins:

1. explicit numeric budget — OpenAI `thinking_budget` or Anthropic
   `thinking.budget_tokens`
2. OpenAI `reasoning_effort` (`"off"` resolves to "no budget, thinking
   disabled")
3. per-model setting `thinking_budget_tokens` (when
   `thinking_budget_enabled` is true)
4. otherwise: no budget — the model thinks freely until it stops

A request-level value always overrides the per-model default.

#### 4.2 Per-model settings

Three `ModelSettings` fields shape the defaults for a model. They live
in `~/.omlx/model_settings.json` and can be hand-edited there or set
through the admin API:

| Field | Meaning |
|---|---|
| `enable_thinking` | force thinking on / off for the model (`null` = follow the model's own default) |
| `thinking_budget_enabled` + `thinking_budget_tokens` | a default numeric budget applied when the request sends none |
| `reasoning_effort_budgets` | override the `reasoning_effort` → token map, e.g. `{"low": 256, "medium": 1024, "high": 4096}`; unset keys fall back to the server defaults in §3.1. Edit `model_settings.json` directly — no admin-API field yet |

Example `model_settings.json` fragment:

```json
{
  "qwen-dense-27b": {
    "reasoning_effort_budgets": {"high": 4096}
  }
}
```

With this, `reasoning_effort: "high"` on `qwen-dense-27b` caps thinking
at 4096 tokens; `"low"` and `"medium"` keep the server defaults.

### 5. How the budget is enforced

The budget is applied as a logits processor (`ThinkingBudgetProcessor`).
It counts tokens generated inside the `<think>` block; when the count
reaches the budget it forces the `</think>` close sequence one token at
a time, then becomes a no-op. The model never sees a truncated tag — it
resumes cleanly into the answer. Thinking that finishes before the
budget is untouched.

### 6. Examples

#### 6.1 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://m5max:8000/v1", api_key="<api_key>")

resp = client.chat.completions.create(
    model="qwen-dense-27b",
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
    extra_body={"reasoning_effort": "low"},
)

msg = resp.choices[0].message
print("thinking:", msg.reasoning_content)
print("answer:  ", msg.content)
```

`reasoning_effort` goes through `extra_body` only if your SDK version
predates the field; current SDKs accept it as a named argument.

#### 6.2 curl — disable thinking

```bash
curl http://m5max:8000/v1/chat/completions \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-dense-27b",
    "messages": [{"role": "user", "content": "Capital of France?"}],
    "reasoning_effort": "off"
  }'
```

### 7. Validation and errors

`reasoning_effort` accepts only `off`, `low`, `medium`, `high`
(case-insensitive, surrounding whitespace trimmed). Any other value is
rejected with HTTP 422 before the model is touched.

---

## 中文

### 1. 背景

支持推理的模型（Qwen3 / Qwen3.5、DeepSeek V4、GPT-OSS、MiniMax 等）
会把思维链用 `<think>...</think>` 标签包起来，放在最终答案前面。oMLX
在服务端解析这些标签，调用方不用自己去剥 `<think>`——思考文本和答案
文本分别落在不同字段里。

这里讲两件事：

1. **读取**推理——思考文本在响应里的位置。
2. **控制**推理——怎么限制它的长度，或者关掉它。

OpenAI 兼容端点（`/v1/chat/completions`）和 Anthropic 兼容端点
（`/v1/messages`）都支持。字段名不同，行为一致。

### 2. 读取推理

#### 2.1 OpenAI——`/v1/chat/completions`

思考块作为独立字符串字段 `reasoning_content` 返回，跟正常的 `content`
并列：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning_content": "用户问 17 * 23。17 * 23 = 391。",
      "content": "17 × 23 = 391。"
    }
  }]
}
```

`content` 只放最终答案。模型没思考（或思考被关掉）时
`reasoning_content` 为 `null`。

流式模式下增量也走这两个字段——先吐完所有 `reasoning_content` 增量，
再吐 `content` 增量：

```
data: {"choices":[{"delta":{"reasoning_content":"用户问"}}]}
data: {"choices":[{"delta":{"reasoning_content":" 17 * 23。"}}]}
data: {"choices":[{"delta":{"content":"17 × 23 = 391。"}}]}
```

#### 2.2 Anthropic——`/v1/messages`

响应 `content` 是一个块数组。思考块（`type: "thinking"`）排在答案块
（`type: "text"`）前面：

```json
{
  "content": [
    {"type": "thinking", "thinking": "用户问 17 * 23。……"},
    {"type": "text", "text": "17 × 23 = 391。"}
  ]
}
```

没有思考的响应直接省掉 `thinking` 块。

### 3. 控制推理

有三个请求级控制项。挑第一个够用的——下面按"上手成本从低到高"
排列。

#### 3.1 `reasoning_effort`——OpenAI，推荐

`reasoning_effort` 是 OpenAI 标准字段。这是控制思考最简单的方式，
不需要任何 oMLX 专有知识：

| 取值 | 效果 | 默认思考预算 |
|---|---|---|
| `"off"` | 关闭思考——模型直接作答 | —— |
| `"low"` | 短思考 | 512 token |
| `"medium"` | 中等思考 | 2048 token |
| `"high"` | 长思考 | 8192 token |

```json
POST /v1/chat/completions
{
  "model": "qwen-dense-27b",
  "messages": [{"role": "user", "content": "……"}],
  "reasoning_effort": "low"
}
```

预算是上限，不是目标：推理提前结束的模型会自己吐 `</think>`，根本
碰不到上限。一旦真的撞上限，服务端强制吐 `</think>` token，让生成
转入答案（见 §5）。

上表的默认预算是服务端默认值，可以按模型覆盖——见 §4.2。

#### 3.2 `thinking_budget`——OpenAI，显式数字上限

需要精确的 token 上限而不是命名档位时，传整数 `thinking_budget`：

```json
{
  "model": "qwen-dense-27b",
  "messages": [...],
  "thinking_budget": 1200
}
```

两个都传时，`thinking_budget` 永远压过 `reasoning_effort`。

#### 3.3 `chat_template_kwargs.enable_thinking`——底层开关

最底层的开关是 chat-template kwarg `enable_thinking`。
`reasoning_effort: "off"` 就是把它设成 `false` 的语法糖。也可以直接
设：

```json
{
  "model": "qwen-dense-27b",
  "messages": [...],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

优先用 `reasoning_effort: "off"`——这里写 `enable_thinking` 只是为了
让现有调用方认得它。

#### 3.4 Anthropic——`thinking`

Anthropic 端点用标准的 `thinking` 对象，而不是 `reasoning_effort`：

```json
POST /v1/messages
{
  "model": "qwen-dense-27b",
  "messages": [...],
  "thinking": {"type": "enabled", "budget_tokens": 1200}
}
```

`type: "disabled"` 关闭思考。`budget_tokens` 是数字上限，行为跟
OpenAI 那边的 `thinking_budget` 完全一致。

### 4. 服务端默认值

#### 4.1 优先级

单次请求的有效思考预算按以下顺序解析，命中第一个即停：

1. 显式数字预算——OpenAI `thinking_budget` 或 Anthropic
   `thinking.budget_tokens`
2. OpenAI `reasoning_effort`（`"off"` 解析为"无预算，思考关闭"）
3. 模型级设置 `thinking_budget_tokens`（当 `thinking_budget_enabled`
   为 true）
4. 否则：无预算——模型自由思考直到自己停下

请求级取值永远覆盖模型级默认值。

#### 4.2 模型级设置

三个 `ModelSettings` 字段决定一个模型的默认行为。它们存在
`~/.omlx/model_settings.json` 里，可以手工改，也可以走 admin API：

| 字段 | 含义 |
|---|---|
| `enable_thinking` | 强制该模型思考开 / 关（`null` = 跟随模型自身默认） |
| `thinking_budget_enabled` + `thinking_budget_tokens` | 请求没带预算时套用的默认数字预算 |
| `reasoning_effort_budgets` | 覆盖 `reasoning_effort` → token 映射，例如 `{"low": 256, "medium": 1024, "high": 4096}`；没写的键回退到 §3.1 的服务端默认值。直接改 `model_settings.json`——暂无 admin API 字段 |

`model_settings.json` 片段示例：

```json
{
  "qwen-dense-27b": {
    "reasoning_effort_budgets": {"high": 4096}
  }
}
```

这样在 `qwen-dense-27b` 上 `reasoning_effort: "high"` 把思考压到 4096
token；`"low"` 和 `"medium"` 保持服务端默认。

### 5. 预算怎么强制执行

预算以 logits processor（`ThinkingBudgetProcessor`）的形式生效。它统计
`<think>` 块内生成的 token 数；计数到达预算时，逐 token 强制吐出
`</think>` 收尾序列，之后变成空操作。模型不会看到被截断的标签——它
干净地转入答案。预算之前就结束的思考不受影响。

### 6. 示例

#### 6.1 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://m5max:8000/v1", api_key="<api_key>")

resp = client.chat.completions.create(
    model="qwen-dense-27b",
    messages=[{"role": "user", "content": "17 乘 23 等于几?"}],
    extra_body={"reasoning_effort": "low"},
)

msg = resp.choices[0].message
print("思考:", msg.reasoning_content)
print("答案:", msg.content)
```

只有当你的 SDK 版本早于这个字段时才需要走 `extra_body`；当前版本的
SDK 直接当具名参数收。

#### 6.2 curl——关闭思考

```bash
curl http://m5max:8000/v1/chat/completions \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-dense-27b",
    "messages": [{"role": "user", "content": "法国首都是哪?"}],
    "reasoning_effort": "off"
  }'
```

### 7. 校验与报错

`reasoning_effort` 只接受 `off`、`low`、`medium`、`high`（大小写不敏感，
首尾空白会去掉）。其他取值在碰到模型之前就以 HTTP 422 拒掉。
