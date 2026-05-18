# 上游同步台账 (upstream sync log)

记录 flyto-mlx 从上游 `jundot/omlx` 引入了哪些 commit/PR、确认已有哪些、
评估后跳过哪些。**每次做上游同步,都更新本文件。**

上游:<https://github.com/jundot/omlx>(git remote 名 `upstream`)

## 方法(重要)

判断"某个上游 commit 是否已在 flyto" —— **唯一可靠的方式是实际
`git cherry-pick`**:cherry-pick 后若提示 "nothing to commit" 即已存在
(三方合并会把已有内容识别掉)。

不可靠、会虚报的方式:

- `git cherry`(按 patch-id 比对)—— flyto 经常把上游 patch 换形状引入,
  patch-id 对不上,会把已有的报成"缺失"。
- 临时 `grep` —— 容易写错。`grep -E` 模式下 `\|` 是**字面竖线**不是"或"
  (2026-05-18 踩过,误判 chunked prefill "整个缺失",实际早已有)。

未合并的 open PR:用 `git fetch upstream pull/<N>/head` 取 PR HEAD,再从
GitHub API `/pulls/<N>/commits` 拿该 PR **自己的** commit SHA 逐个
`cherry-pick -x`(PR branch 基于旧 upstream/main,直接 merge 会拖进无关
commit)。

## 最近同步

- **2026-05-18 (一)** — 对齐 `upstream/main` @ `51907f0`
- **2026-05-18 (二)** — review 上游 71 个 open PR + 74 个 open issue。
  挑出 7 个 PR cherry-pick 到分支 `sync/upstream-prs-2026-05-18`
  (基于 `main` @ `0d28e26`)。已 push 到 origin,在 m2max `~/Code/omlx`
  的 `.venv` 跑过 `pytest`:
  - 7 个 PR 直接相关的 4 个测试文件:**248 / 248 pass**。
  - 初次完整套件 **4403 pass / 12 fail**;12 个 fail 全部在 `main` 上也 fail
    —— cherry-pick **零回归**。pre-existing fail 集对应上游 issue #1259。
  - 顺带修了一个 `main` 既有 bug:`list_models` 的显式 settings dict
    漏了 4 个 `ModelSettings` 字段(`e3f0912`)。
  - **12 个 pre-existing fail 已全部修掉**(见下「上游 issue 处理记录」
    #1259 + flyto-divergence stale test):cherry-pick 上游 #1268/#1286/
    #1287 修 6 个,flyto 自己改 4 个(model_profiles 漏分类字段、
    server_manager auto-restart cap、`_prepare_vision_inputs` audios kwarg、
    engine_pool MagicMock 误判 DFlash),2 个 full-suite 污染 flake 也修了
    (`test_includes_python_heap` 加大分配防 allocator 复用)。
    **最终完整套件:4415 pass / 0 fail**(2026-05-18 m2max)。
  - **仍未并回 main**,等人工 review 后 `git merge --ff-only`。
  - #1241(structured output strict enforcement)同批修复 —— 见下。

## 已引入(cherry-picked)

| 上游 commit | flyto commit | 内容 | 引入日期 |
|---|---|---|---|
| `d736bfd` | `2e4d7c1` | chunked prefill: RuntimeError 作为 request error 上报 | 2026-05-18 |
| `c003b2e` | `ee2342e` | chunked prefill: 显存检查 + 进度回调 + dead-abort 检查 | 2026-05-18 |
| `386e16f` (#1244) | `cdaec79` | 测试: xgrammar import guard + 修上游既有测试失败 | 2026-05-18 |
| `51907f0` | `81f9815` | oQ: 给 VLM sensitivity 恢复 MTP head attach | 2026-05-18 |

cherry-pick 一律带 `-x`,commit message 里保留 "cherry picked from commit …"
溯源行,可用 `git log --grep="cherry picked from"` 反查。

### 2026-05-18 第二批(open PR,在分支 `sync/upstream-prs-2026-05-18`)

| 上游 PR | 上游 commit | flyto commit | 内容 | 冲突处理 |
|---|---|---|---|---|
| #1273 | `6359b54a` | `acd5a58` | cache: 注意力引导的分层 KV cache 驱逐 | 干净 |
| #1274 | `c091ad50` | `51cbb25` | cache: 非对称 KV 量化 (K=INT4, V=INT2) | `ablations/__init__.py` add/add —— 把 #1274 的通用名 `install/remove/get_stats` 改成带命名空间的 `*_asymmetric_kv`,与同包另两个 ablation 模块风格对齐 |
| #1275 | `820e8013` | `884708e` | cache: 基于 layer-1 hidden state 哈希的语义前缀匹配 | `ablations/__init__.py` add/add —— 三个 ablation PR 各自 bootstrap 该包,合并三方 export |
| #1153 | `467ad67d` `4c4464c0` | `673a428` `860ffaa` | tool_calling: 解析 Llama-3 风格 `{"name","parameters"}` JSON | 干净 |
| #1269 | `8b0cb178` | `314a36d` | server: 非流式 usage 响应补 `total_time` | 干净 |
| #1183 | `0de60746` `b49963b7` | `edf0c7d` `137d91d` | cache: per-model cache 命中率可观测性 | 干净(注:这两个 commit 也是 #1149 的子集) |
| #1245 | `7d038950` `d8d99a8d` | `331b0f4` `6ea2fcb` | responses: Responses API 原生 reasoning 支持 | `admin/routes.py` —— #1245 顺手把 settings dict 重构成 `dataclass_fields` 推导式,flyto 是**刻意维护的显式白名单**(见 #1268 / `0d28e26`),保留 flyto 版本,并回退随之多余的 `dataclass_fields` import;PR 第一个 commit `dbde075d8`(test 修复)cherry-pick 报 empty,确认 flyto 已有 |

## 确认已在 flyto(评估时已存在,勿重复引入)

- `11e6ea7` (#1224) chunked prefill 基座 —— flyto 早已有(换形状引入,
  本次补的是它的两个 follow-up 修复 `d736bfd`/`c003b2e`)
- `ccfba1d` (#1247) oQ-quant VLM 加载修复
- `37c73a0` phys_footprint enforcer + prefill 峰值 admission control
- `196d667` SchedulerQueueFullError → HTTP 503 + Retry-After
- `521cccf` (#1211) health-check Session 复用(防端口耗尽)
- `19bb34e` (#1214) `/v1/audio/transcriptions` 的 word_timestamps ——
  flyto 有更强的自有实现(aligner auto-chain + on_aligner_overflow)
- **#1141** `catch TypeError when accessing think token properties` ——
  flyto 已有更好的封装 `Scheduler._get_think_token_id()`(单一 helper
  统一 catch `(ValueError, TypeError)`,而 #1141 是三处内联 try/except)。
  2026-05-18 cherry-pick 时确认冲突即此,**跳过**。

## 评估后跳过(价值低或不适用,勿重复评估)

- `c54de70` / `be3b024` (#1251) 日志查看器 level filter —— admin UI QoL
- `5994dc5` (#1223) / `290587f` (#1255) codex/claude CLI 参数透传 —— 按需
- `4fe004d` (#1250) Hermes Agent quick launch
- `fc5171b` (#1088) 周期 health timer 重新检查更新
- `04a0ce6` / `25c312f` / `68b5c25` / `71beab7` / `7fab13b` 杂项小修
- **open PR 中明确跳过**(与 flyto 定位无关):#975 VS Code 扩展、
  #988 MseeP 徽章、#987 俄语 README、#1026 Nix flake、#855 图像生成 API、
  #1025 docs/CLAUDE.md、#952 Crush / #1282 Pi 集成、
  UI QoL 类(#1278 / #1213 / #1187 / #830 / #1052 / #350)

## 待引入(评估为有价值,下一批,尚未 cherry-pick)

下次同步优先处理。优先级按对 flyto 核心(工具调用 / DFlash / KV cache /
Qwen-Gemma / oQ)的相关度。

| 上游 PR | 内容 | 优先级 | 备注 |
|---|---|---|---|
| #844 | tool_calling: 回收 Qwen3-Coder 裸 `<function=...>` 调用 | 高 | 工具调用核心 + Qwen |
| #822 | SpecPrefill retry RoPE AttributeError + Qwen3 MoE VLM 误判 | 高 | Qwen3 MoE bug |
| #1056 / #818 | dflash: 把带图请求路由到 VLM fallback engine | 高 | dflash 正确性,flyto 有 Gemma4 VLM(#818 范围更大,含 tts) |
| #805 | DDTree 树状推测解码(搭在 DFlash 上) | 中高 | 大特性,需评估与 Path A 的耦合 |
| #933 | oq: mlx_vlm sanitize 前预融合 per-expert MoE 权重 | 中 | oQ 相关 |
| #1225 | specprefill: 无 draft model 的自打分(单请求 TTFT) | 中 | specprefill |
| #1268 | profiles: 分类 9 个新 ModelSettings 字段 | 中 | 与 flyto `0d28e26` 同类,修 #1259 测试失败之一 |
| #1149 | cache: 多槽 LRU MRU partial block cache | 暂缓 | 19-commit **draft**,且包含 #1183 两个 commit;等它在上游定稿再说 |

## 上游 issue 处理记录

记录 flyto 修掉的、与上游 issue 对应的问题(flyto 自身 bug 见
`docs/roadmap.md`)。

- **#1259** "some failing tests" —— **已全部解决**(2026-05-18,分支
  `sync/upstream-prs-2026-05-18`)。flyto 完整套件初始 12 个 fail:
  - cherry-pick 上游 #1244(`cdaec79`,早先)+ #1268 / #1286 / #1287
    修掉 6 个(profiles 字段分类、Scheduler/Memory `to_dict`、
    `test_mlx_lm_mtp_patch`、`test_vlm_torch_free_image_processor`)。
  - 另 4 个是 flyto 自身 divergence 的 stale test,上游 PR 不覆盖,
    flyto 自己改:`model_profiles` 补分类 `dflash_max_concurrent` /
    `dflash_kv_pressure_threshold`;`test_omlx_app` 跟进 server_manager
    auto-restart cap 3→10000(`3bed072`);`test_vlm_engine` 跟进
    `_prepare_vision_inputs` 的 `audios` kwarg;`test_engine_pool`
    的 MagicMock 被 Path A 的 `hasattr(engine,"_dflash_bundle")`
    duck-type 误判成 DFlash engine,给 mock `del _dflash_bundle`。
  - 2 个 full-suite ordering/内存污染 flake 也修:`test_includes_python_heap`
    加大分配额防 allocator page 复用。
  - 结果:**4415 pass / 0 fail**。
- **#1241** `response_format json_schema strict` 不强制 —— flyto 在
  `sync/upstream-prs-2026-05-18` 分支上自己修(上游 issue 开着没修):
  - 根因有两层:① 服务器 venv 没装 xgrammar(可选依赖)→ `grammar_compiler`
    为 `None` → 100% 走 prompt 注入;② 即便 xgrammar 在,`response_format`
    路径在编译失败时**静默降级**(`structured_outputs` 路径会抛 400),
    且 `strict` 字段全程没代码读。
  - 2026-05-18 已做:m5max venv 装 `xgrammar 0.2.1`(0.2.x 不再拽 torch,
    ~24MB)+ 重启 server,enum 排除测试实证 grammar 在 logit 层硬强制了;
    `pyproject.toml` 把 xgrammar 提为**核心依赖**(不再可选,免再踩坑)。
  - 代码修复 layer ①(`09ed68b`):`strict:true` 且 grammar 强制不了时
    抛 HTTP 400,不再静默 200。layer ②③(降级响应头 + `/api/status`
    能力位 + 启动日志)待做。
  - 注:数值 `minimum/maximum` 即便 grammar 成功也不强制,仍需客户端
    `jsonschema` 兜底。

## 上游未解决 issue 观察(2026-05-18 review,74 个 open issue)

只挑与 flyto 技术栈相关的。**这些是上游 bug,不是 flyto 待办** —— 列出
是为了:① flyto 撞到同款问题时知道上游也没修;② 评估要不要主动修。

### 工具调用 / 结构化输出(flyto 核心,重点盯)

- **#1290** OpenAI API:`has_tool_calling=False` 时 tool result 被转成
  `role=user`,破坏多轮工具调用 —— flyto 刚引入工具调用 PR,需验证不受影响
- **#1148** "Tool calling seems to be broken"(信息少,待复现)
- **#1258** Anthropic `/v1/messages` 忽略 forced strict tool use
- **#1241** `/v1/chat/completions` 接受 `response_format.type=json_schema`
  但不强制执行

### DFlash(flyto Path A 核心,必须盯)

- **#1292** DFlash 在 Qwen3.6 35B-A3B 上不工作
- **#1109** DFlash 启动失败 Qwen3.6-35B-A3B-4bit:`mtp.layers.0.mlp.experts` key 缺失
- **#1291 / #1162** Qwen3.6 27B DFlash 性能差(跑分模式 OK,实际任务慢)
- **#1264 / #1276** DFlash window 配置(context 限制 / 暴露 draft_window_size 等)
- **#1233** specprefill 的 pp-tps 增益在 dflash 同时开启时静默丢失
- **#1102** 请求加 Gemma4 DFlash 支持

### SpecPrefill

- **#1262** SpecPrefill 在 Qwen3.6-35B-A3B 上拖慢 token 生成 ~50%
- **#1263** SpecPrefill threshold 设置在 Qwen3.6-35B-A3B 上被忽略
- **#1145** v0.3.8 SpecPrefill 模型设置不加载,总是 fallback 默认值

### oQ / MTP 量化(flyto 做 oQ)

- **#1133** DeepSeek-V4-Flash MTP patch 静默失败(`has no mtp_forward`)
- **#1124** 请求 Gemma4-31B 3/4bit oQ + MTP
- **#1097** oQ-MTP 在 M1 上性能不及预期
- **#1195** 请求 Nemotron-H 的 MTP 支持
- **#1253** 加载 TurboQuant(tq3)模型报 `KeyError: 'turboquant'`

### Gemma 4 / Qwen3.6 VLM(flyto 重点)

- **#1093** `_strip_thinking` 在 `extract_gemma4_messages` 里冗余,可能引入
  unicode 问题 —— 上游已有同名 PR,建议连带评估
- **#1099** Qwen3.6-27B-oQ8-mtp 的 vision 能力不工作
- **#1261** Qwen3.6 35B-A3B 的 vlm 被 auto-disable

### 崩溃 / 稳定性(环境相关,flyto 未必撞到)

- **#1281** Qwen3.6-35B-A3B-mxfp8 在 M1 prefill 阶段崩(QuantizedMatmul)
- **#1265** macOS 26.4.1 Kernel Panic
- **#1200** MLX Core SIGABRT(MetalAllocator::malloc)
- **#1128** DeepSeek V4 频繁 cache 损坏

### 流式 / server(flyto 刚动了 #1269 usage)

- **#1267 / #1293** 流式响应 chunked encoding 收尾不当,
  破坏部分 HTTP 客户端 / Copilot CLI

> 下次 review 上游 open PR 时,把结论(引入 / 跳过)回填到对应小节。
