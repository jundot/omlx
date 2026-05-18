# flyto-mlx Engineering Roadmap

Engineering backlog for the speech / chat / reasoning surface of
flyto-mlx (oMLX). Priorities and effort estimates as of 2026-05-18.
Updated when sessions ship.

English version first. Chinese translation follows below.

---

## English

### P0 — agent-loop blockers, ship before anything else

1. **`reasoning_content` / `content` separation in chat-completions response**
   *(was item 3 on the original list, promoted)*

   **Status: shipped 2026-05-18.** Both shapes are live; see
   `docs/reasoning-api.md`.

   Reasoning-capable models (Qwen3, DeepSeek V4, GPT-OSS, etc.) emit
   `<think>...</think>` blocks that today get glued into a single
   `message.content` string. Downstream agent loops have to parse the
   tags by hand, and a missed parse breaks the loop. Fix is to split
   into:

   - Anthropic-style: `message.content = [{type:"thinking", thinking:"..."}, {type:"text", text:"..."}]`
   - OpenAI-style: `message.content` (final answer only) +
     `message.reasoning_content` (the thinking block)

   Expose both shapes via a response-shape selector or a default
   choice + ergonomic field aliases. Estimate ~80–100 LOC + tests.

2. **`reasoning_effort` / thinking budget control**
   *(was item 2 on the original list)*

   **Status: shipped 2026-05-18.** OpenAI `reasoning_effort`
   (`off`/`low`/`medium`/`high`), numeric `thinking_budget`, and
   Anthropic `thinking.budget_tokens` are all live, with per-model
   budget overrides via `ModelSettings.reasoning_effort_budgets`. See
   `docs/reasoning-api.md`.

   Two complementary APIs in one feature:

   - OpenAI: `reasoning_effort: "off" | "low" | "medium" | "high"`
     (syntactic sugar)
   - Anthropic: `thinking: {type: "enabled", budget_tokens: N}`
     (direct numeric budget)

   Map effort levels to per-model token budgets (e.g. low=512,
   medium=2048, high=8192; tune per model card). Apply by reading the
   live token count inside the `<think>` block and forcing
   `</think>` emission when the budget is hit. Estimate ~100–150 LOC.

   Couple tightly with item 1 — same session, same PR. Both touch the
   same chat-completions response shape.

3. **STT LLM post-processing layer**
   *(missing from original list, added after design discussion)*

   The 3-pass `energy_tripass` diarization backend ships clean speaker
   attribution but leaks two known issues: per-channel ASR
   hallucinations in low-signal regions, and forced-aligner-stripped
   punctuation in the merged segment text. A server-side optional LLM
   pass cleans both at once and also gives semantic role re-mapping
   (so `SPEAKER_00` → `speakers[0]` ordering becomes 销售/客户 by
   content, not by who-spoke-first).

   API surface:

   - new form field `postprocess_model: str = Form(None)` on
     `/v1/audio/transcriptions`
   - per-model `ModelSettings.default_postprocess_model: Optional[str]`
     (mirrors the existing `aligner_model` field)
   - effective model: per-request > per-model setting > skip
   - on failure: fall through to raw transcript + log warning (no 500)

   The LLM call goes through the same `engine_pool` so a co-loaded
   model (Gemma 4 26B-A4B is the current candidate per
   `[[gemma4_26b_a4b_sweet_spot]]`) has zero load cost.
   Estimate ~150–200 LOC + prompt template + tests.

### P1 — quality-of-life, ship in order

4. **`response_format={"type":"json_object"}` for chat-completions**

   OpenAI client-library compatibility. Two shapes to support:

   - `{"type": "json_object"}` — soft enforcement via grammar that
     starts/ends with `{`/`}`. Cheap.
   - `{"type": "json_schema", "schema": {...}}` — strict schema-driven
     grammar. The interesting case (tool args validation). 1.5–2× the
     work of soft enforcement.

   Infra `omlx/api/grammar.py` already handles arbitrary grammars.
   Main work: chat-completions route plumbing + JSON-schema-to-grammar
   compiler. Estimate ~150–250 LOC depending on `json_schema` scope.

5. **Usage timing fields**
   *(was item 5)*

   Add to chat-completions `response.usage`:

   - `prefill_time_ms`
   - `decode_time_ms`
   - `tokens_per_sec`

   `mlx-lm`'s `generate_step` already returns these stats; the work is
   threading them through the response builder. Estimate ~50–80 LOC.

6. **pyannote MPS / Metal acceleration**
   *(missing from original list, added)*

   pyannote.audio 4.x supports `device=` for the pipeline. The current
   wrapper runs on CPU (8–12 s for 158 s 8 kHz audio per the
   benchmark in `docs/stt-diarize-api.md` §8). Switching to MPS should
   give a 3–4× speedup. Estimate ~30 LOC + device-aware retry on
   loading failure.

### P2 — research / longer-tail

7. **STT per-request observability**
   *(missing from original list, added)*

   Log each transcribe request to a structured store (SQLite append or
   JSONL on disk): timestamp, model, backend, channels, duration_s,
   wall_time_ms, response_size, caller IP, response_format. Enables
   "which configurations work in production" analysis later.
   Estimate ~80 LOC + log-rotation cron.

8. **Prompt caching on MLX**
   *(was item 6)*

   Anthropic-style prompt caching: TTL'd cache slices keyed by prompt
   prefix, with `cache_control: {type: "ephemeral"}` breakpoint markers
   in the request. oMLX already has disk KV cache; extending to
   prompt-cache semantics needs:

   - multi-breakpoint slice tracking (a single conversation can have
     multiple cache control marks)
   - TTL management (5-min and 1-hour tiers like Anthropic)
   - memory-budget-aware eviction
   - per-request cache-hit metrics

   Realistic total ≥ 500–800 LOC across several PRs. Start with a
   design doc / spike. Estimate 0.5 day for the spike, then size each
   sub-PR after.

9. **Async / webhook transcribe**
   *(missing from original list, added)*

   Long-form audio (≥ 30 min calls, ≥ 2 h interview) hits HTTP
   gateway timeouts on synchronous `/v1/audio/transcriptions`. Add
   `POST /v1/audio/transcriptions/async` returning `{job_id}`, plus
   either polling (`GET /v1/audio/transcriptions/jobs/{id}`) or
   webhook callback (`POST callback_url` on completion). Estimate
   ~200 LOC + a job-store table.

   Conditional priority: ship only if there's a real caller with
   ≥ 30 min audio in production. Otherwise the sync endpoint suffices.

### Explicitly not on the roadmap

- **`tripass_llm` built-in backend.** The LLM cleanup belongs in the
  generic post-processing layer (item 3), not as a fourth diarize
  backend. Avoids combinatorial backend explosion.
- **Model alias / default.** Caller passes the exact model id today;
  alias indirection is a misfeature that hides which model actually ran.
- **Built-in VAD.** Upstream ASR models already gate on energy /
  silence internally. A separate VAD layer would duplicate effort and
  silently drop audio the user wanted transcribed.
- **Chat streaming (SSE).** Current use cases are batch (sales call
  upload, transcript generation, agent reasoning runs). Adding SSE
  doubles the response-shape surface and breaks the chat-completions
  unit tests for marginal value. Revisit when a UI consumer needs it.

### Verified non-issues (closed before backlog)

- **STT `top_p` / `frequency_penalty` passthrough.** Verified on
  m5max 2026-05-18: source path is clean (audio_routes →
  transcribe_kwargs → engine.transcribe → either default path
  `model.generate(**gen_kwargs)` for top_p/top_k/min_p/repetition_penalty
  or extended `_generate_single_chunk` for frequency_penalty/
  presence_penalty). A/B with temperature=1.0 ± top_p=0.01 and
  ± frequency_penalty=2.0 on med.wav produced three distinct outputs,
  confirming all sampling fields reach the model. No fix needed.

---

## 中文

### P0 — agent 循环阻塞项, 一切之前先做

1. **chat-completions response 里 `reasoning_content` / `content` 分离**
   *(原清单 #3, 提级)*

   **状态: 2026-05-18 已 ship.** 两种 shape 都上线了, 见
   `docs/reasoning-api.md`.

   推理模型 (Qwen3, DeepSeek V4, GPT-OSS 等) 产出的
   `<think>...</think>` 块现在被粘进单个 `message.content` 字符串.
   下游 agent loop 要手工解析标签, 一次漏解析整条循环就废. 修法
   是拆成:

   - Anthropic 风格: `message.content = [{type:"thinking",
     thinking:"..."}, {type:"text", text:"..."}]`
   - OpenAI 风格: `message.content` (只放 final answer) +
     `message.reasoning_content` (放思考块)

   两种 shape 都暴露, 用 response-shape selector 或一个默认选择 +
   语义化字段别名. 估 ~80-100 LOC + 测试.

2. **`reasoning_effort` / 思考预算控制**
   *(原清单 #2)*

   **状态: 2026-05-18 已 ship.** OpenAI `reasoning_effort`
   (`off`/`low`/`medium`/`high`)、数字 `thinking_budget`、Anthropic
   `thinking.budget_tokens` 都已上线, 并支持按模型用
   `ModelSettings.reasoning_effort_budgets` 覆盖预算. 见
   `docs/reasoning-api.md`.

   一个功能两套互补 API:

   - OpenAI: `reasoning_effort: "off" | "low" | "medium" | "high"`
     (语法糖)
   - Anthropic: `thinking: {type: "enabled", budget_tokens: N}`
     (直接数字预算)

   把 effort 等级映射到每个模型的 token 预算 (如 low=512,
   medium=2048, high=8192; 按 model card 调). 实现方式: 读 `<think>`
   块内实时 token 计数, 到预算时强制吐 `</think>`. 估 ~100-150 LOC.

   跟 #1 强耦合, 一起做一次 PR. 都改 chat-completions response shape.

3. **STT LLM 后处理层**
   *(原清单漏的, 设计讨论后加上)*

   `energy_tripass` 3-pass diarization 落地了干净的说话人归属, 但留
   两个已知问题: 安静声道单独 ASR 的幻觉, 以及 forced-aligner 撕掉
   的 segment text 标点. 一次服务端可选的 LLM pass 一并清掉, 同时
   还能做语义角色重映射 (让 `SPEAKER_00` → `speakers[0]` 顺序变成
   按内容判断的 销售/客户).

   API:

   - `/v1/audio/transcriptions` 加 `postprocess_model: str = Form(None)`
   - per-model `ModelSettings.default_postprocess_model: Optional[str]`
     (镜像现有 `aligner_model` 字段)
   - effective model 解析: 单次请求 > per-model 设置 > 跳过
   - 失败时: 回退到原始 transcript + 日志 warning (不 500)

   LLM 调用走同一个 `engine_pool`, 共驻模型 (当前候选 Gemma 4 26B-A4B,
   按 `[[gemma4_26b_a4b_sweet_spot]]`) 零加载成本.
   估 ~150-200 LOC + prompt 模板 + 测试.

### P1 — quality-of-life, 按顺序 ship

4. **chat-completions 的 `response_format={"type":"json_object"}`**

   OpenAI 客户端库兼容. 两种 shape 要支持:

   - `{"type": "json_object"}` — 软强制, 用 grammar 限制以 `{`
     开头 `}` 结尾. 便宜.
   - `{"type": "json_schema", "schema": {...}}` — 严格 schema-driven
     grammar. 有意思的那种 (tool 参数验证用). 1.5-2x 软强制的工作量.

   基础设施 `omlx/api/grammar.py` 已经能跑任意 grammar. 主要工作:
   chat-completions 路由插管 + JSON-schema 到 grammar 的编译器.
   估 ~150-250 LOC, 取决于 `json_schema` 覆盖范围.

5. **Usage 加 timing 字段**
   *(原清单 #5)*

   chat-completions `response.usage` 加:

   - `prefill_time_ms`
   - `decode_time_ms`
   - `tokens_per_sec`

   `mlx-lm` 的 `generate_step` 已经返这些 stats, 工作量在 response
   builder 串通. 估 ~50-80 LOC.

6. **pyannote MPS / Metal 加速**
   *(原清单漏的, 加上)*

   pyannote.audio 4.x 支持 `device=` 参数. 当前 wrapper 跑在 CPU
   (`docs/stt-diarize-api.md` §8 实测 158s 8 kHz 音频要 8-12s).
   切到 MPS 应该 3-4x 加速. 估 ~30 LOC + 加载失败时回退 CPU.

### P2 — 调研 / 长尾

7. **STT 请求级 observability**
   *(原清单漏的, 加上)*

   每个 transcribe 请求 log 到结构化存储 (SQLite append 或磁盘
   JSONL): 时间戳, model, backend, 通道数, 时长_s, 墙钟_ms,
   response 大小, 调用方 IP, response_format. 用于事后分析
   "哪些配置在生产上真的好用". 估 ~80 LOC + 日志轮转 cron.

8. **MLX 上的 prompt caching**
   *(原清单 #6)*

   Anthropic 风格 prompt caching: 按 prompt 前缀作 key 的 TTL'd
   缓存切片, 请求里用 `cache_control: {type: "ephemeral"}` 标
   breakpoint. oMLX 已有磁盘 KV cache; 扩成 prompt-cache 语义需要:

   - 多 breakpoint 切片追踪 (单次对话可以有多个 cache 标记点)
   - TTL 管理 (跟 Anthropic 一样 5 分钟 / 1 小时两档)
   - 内存预算感知的淘汰策略
   - 每请求 cache-hit 指标

   现实估总量 ≥ 500-800 LOC, 拆几个 PR. 先开 design doc / spike.
   spike 估 0.5 天, 之后再拆每个子 PR 的 size.

9. **Async / webhook transcribe**
   *(原清单漏的, 加上)*

   长音频 (≥30 分钟通话, ≥2 小时访谈) 同步走
   `/v1/audio/transcriptions` 会撞 HTTP 网关超时. 加
   `POST /v1/audio/transcriptions/async` 返 `{job_id}`, 再加 poll
   (`GET /v1/audio/transcriptions/jobs/{id}`) 或 webhook 回调
   (`POST callback_url` 完成时). 估 ~200 LOC + 一张 job-store 表.

   条件性优先级: 仅当生产里真有 ≥30 分钟音频的调用方时才做.
   否则同步端点够用.

### 明确不放路线图上

- **`tripass_llm` 内置 backend.** LLM 清洗属于通用后处理层 (#3),
  不该作为第四个 diarize backend 存在. 避免 backend 组合爆炸.
- **Model alias / default.** 当前调用方传精确 model id;
  alias 间接化是反 feature, 会掩盖实际跑了哪个 model.
- **内置 VAD.** 上游 ASR 模型内部已按 energy / silence gate.
  另起一层 VAD 重复劳动, 还会把用户想要转录的音频静音段静默丢掉.
- **Chat streaming (SSE).** 当前用例都是 batch (上传销售通话,
  生成 transcript, 跑 agent reasoning). 加 SSE 会让 response shape
  surface 翻倍, 又破坏 chat-completions 的单元测试, 收益边际.
  等有 UI 消费方真需要时再开。

### 已 verify 的非问题 (进入 backlog 前关闭)

- **STT `top_p` / `frequency_penalty` 透传.** 2026-05-18 在 m5max
  上 verify: 源码路径干净 (audio_routes → transcribe_kwargs →
  engine.transcribe → 默认路径 `model.generate(**gen_kwargs)` 走
  top_p/top_k/min_p/repetition_penalty, 或扩展路径
  `_generate_single_chunk` 走 frequency_penalty/presence_penalty).
  med.wav 上 temperature=1.0 ± top_p=0.01 ± frequency_penalty=2.0
  的 A/B 实测三组输出都不同, 确认所有采样字段都到达模型. 不修.
