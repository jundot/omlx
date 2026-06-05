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

- **2026-05-26** — 上游 8 天发了 4 个 release(`v0.3.9` → `v0.3.12`,
  共 38 个 commit,排除 bump/图片上传)。在分支 `sync/upstream-2026-05-26`
  (基于 `main` @ `6cbc7b7`,即 "禁用 Mac app 内的上游 update check")cherry-pick:
  - **A 组(DFlash / MTP / tool_calling / oQ)12 个**:`941fcbe` `b413356`
    `42fc129` `6f927ec` `ea2eaa1` `a53bf11`(#1356) `90d7e40`(#1392/#1393)
    `64f7d93` `915190d`(#1388) + 顺带 `878c892`(#1336,#1388 的前置依赖,
    新测试用了它的 `_is_greedy.temp` 语义) `b33cb6a` `56ae7f0`(#1404)
    `ecb610e`(#1412)。`d0f60ec`(#1344 dflash 多模态 VLM fallback)
    跳过 —— 上游用 lazy `_fallback_engine` swap,flyto Path A 是永久
    `_embedded_vlm` 双引擎,routing 基于 prompt token len 而不是 content;
    要做等同效果需要重写 message 路由层,**留独立 spike**。
  - **B 组(scheduler / memory 正确性)6 个**:`ef49351`(#1383)
    `f0f3138`(#1389) `3b15958`(#1405) `7d30401` `ea7efd4`(=0169f15)
    `3af848b`(#684)。`0169f15` 解冲突时:test `test_hard_limit_honors_user_explicit_max`
    跳过(`user_explicit_max` 字段属于 C 组 `acd0533` 引入,flyto 没有),
    `test_hard_limit_auto_mode_uses_size_aware_reserve` 去 `user_explicit_max` kwarg
    后保留。
  - **D 组(中低优,trial cherry-pick)5/12 干净进入**:`f6fdaf2`(=f1d1fc3 #1339)
    `5749613`(=5d8145b) `31d31be`(=db07311) `ef1e842`(=7d640c1 #1417)
    `5e394cf`(=1010fd3)。冲突跳过 7 个:`cf4023c` `c4ebb7f` `f8174a9`
    `2f2f508` `8c70903` `1b666af` `6a77fd5` —— 都是低优,与 flyto 自身
    改动相撞,沿用上游版本价值不大。
  - **C 组 `c645c9f` 内存配置重写**(memory_guard_tier 替 max_*_memory):
    涉及 29 个文件,flyto 113 处引用 `max_process_memory`/`max_model_memory`,
    需独立 spike 评估配置迁移路径(settings.json 字段名变更 + admin UI 改造)。
    `3ef7b94` `4cfbc8b` `acd0533` `64bd2a2`(#1431) `b129a19`(#1425)
    均依赖它,一并延后。
  - 顺带修了一个 main 残留:`tests/test_admin_auth.py::TestCheckUpdate`
    与 `test_admin_update_check.py` 同名重复,任务 1 漏改 —— sync 分支上
    一并删除(`efc40cd`)。
  - **完整套件 m5max:4493 pass / 3 fail / 37 skip**。3 个 fail 在 `main` 上
    同样 fail,均为 `test_settings.py` 里 mock 文件 `auth.api_key` 被
    `OMLX_SERVER_API_KEY` 环境变量覆盖的测试设计缺陷,**与本批 cherry-pick
    零回归**。
  - 仍未并回 main,等人工 review 后 `git merge --ff-only`。

- **2026-05-27 dflash 多模态路由** — 分支 `sync/dflash-multimodal-routing`
  (基于 `main` @ `e8e0967`)。不 cherry-pick 上游 `d0f60ec`(#1344),而是
  按 flyto Path A(永久 `_embedded_vlm` 双引擎)做等价设计:
  - `DFlashEngine.supports_multimodal_fallback` property:`_embedded_vlm` 是
    `VLMBatchedEngine` 时返 True,否则 False(text-only fallback 不算).
  - `DFlashEngine._has_multimodal_content(messages)` helper:检测 OpenAI
    `image_url` / Anthropic `image` / `input_audio` 等结构化 content part.
  - `DFlashEngine.chat / stream_chat`:多模态请求直接 forward 到
    `self._embedded_vlm.chat / stream_chat`,绕过 `_apply_chat_template`
    的 text-only 路径(否则图像在 template flatten 时被丢).
  - `server.py` 两处(OpenAI chat completions + Anthropic messages):
    `is_vlm` 判断扩展为 `isinstance(engine, VLMBatchedEngine) or
    supports_multimodal_fallback`,使得 dflash+VLM 走
    `extract_multimodal_content` / `preserve_images=True`,把图像保留到
    `engine.chat()` 入口.
  - 新增 `tests/test_dflash_multimodal_routing.py`(15 test,全绿).
  - 完整套件:4509 pass / 4 fail / 37 skip. 4 fail 中 3 个是已知 pre-existing
    settings env var 问题,1 个 `test_boundary_snapshot_store` 是 full-suite
    ordering flake(isolated 通过, 与本批改动无关).

- **2026-05-27 memory_guard_tier 收尾** — 三段 PR 把 C 组 `c645c9f` 重写
  落地 + 5 个 follow-up 一起带回, 加 10 个独立 upstream 修复:
  - **PR #5(backend)**:settings/process_memory_enforcer/engine_pool/server/cli
    全切到 `memory_guard_tier`. 老字段 deprecated alias 一个 release.
    `ModelTooLargeError.max_memory` -> `.ceiling`. 4511 pass.
  - **PR #6(admin UI)**:两个 slider -> tier dropdown, 修了 PR-5 之后
    admin POST 隐性 500. 4510 pass.
  - **PR #7(5 个 follow-up)**:acd0533/4cfbc8b/3ef7b94/b129a19/64bd2a2.
    引入 Metal wired-limit clamp + watermark-tier shrink +
    tier-aware active-memory reclaim + `custom` tier. 4536 pass.
  - **PR #8(10 个独立修复)**:boundary-store race 修(消掉
    `test_cleanup_all_drains_queue` flake), per-engine MLX 线程/流,
    VLM lazy state, profiles 重构. 4567 pass.
  - **最终 baseline**:4567 pass / 3 known env-override fails / 36 skip
    (boundary_snapshot flake 由 #1423 修掉, 不再算).
  - 全部 sync/* PR 已 self-merge 进 main.

- **2026-06-05 Gemma4 Unified VLM 图像修复** — 分支 `sync/gemma4-vlm-image-dev2`
  (基于 `main` @ `3605e36`), PR #46. cherry-pick 上游 v0.4.2.dev2 两个 commit:
  - `ff041ed` "accept gemma4 unified assistant drafter" — 干净 cherry-pick.
  - `77fb32a` "preserve VLM prompt kwargs for Gemma4" — 主修复, 保住
    `mm_token_type_ids` / `token_type_ids` 走 external prefill 路径, 处理 Gemma4
    Unified compacted vision features, vision feature cache 加 token-count 校验.
  - 冲突解决: `omlx/engine/vlm.py` 保留 flyto 的 `has_vision` guard (audio
    fallback) + 上游 `image_token_count` 初始化 (203 行修复仅 1 行冲突);
    `tests/test_scheduler.py` 上游 diff 夹带无关漂移 (#1459 async-store-cache
    泄漏测试 + mock 签名重构, 依赖 flyto 缺失的功能), 5 个冲突全取 flyto 侧, 只
    补回真正相关的 `TestVLMExtraSlicing` (验证 `_slice_vlm_extra` /
    `_advance_vlm_extra` 对 token_type_ids 的处理, 两函数 flyto 已有) +
    `scheduler_module` 别名; `tests/test_vision_feature_cache.py` 干净自动合.
  - 测试: 针对性 318 pass (scheduler 全量 + 全部 VLM 套件); 完整套件 m2max
    4525 pass / 3 fail / 19 skip, 3 个 fail 是已知 `OMLX_SERVER_API_KEY`
    env-override (test_settings.py), 零回归.
  - 生产部署验证: m5max + m2max 都 pull + 重启 serve. 带图实测
    (gemma4-dense-12b-bf16 + 测试图): 修复前 44s, content 是 "thought thought"
    垃圾 + 图像幻觉成无关内容; 修复后 5s, content 准确描述图像. 根因 = 图像
    prefill 的 token-type IDs 丢失导致模型输出整体崩坏 (正文被 thought 垃圾顶掉
    + 幻觉), 纯文本不受影响. self-merge `f940162`.

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

### 2026-05-26 第三批(v0.3.9..v0.3.12,在分支 `sync/upstream-2026-05-26`)

| 上游 commit | flyto commit | 内容 | 冲突处理 |
|---|---|---|---|
| `941fcbe` | `c036019` | mtp: reset state across batch reshapes | 干净 |
| `b413356`(#1320) | `1d70b9e` | load: wire MTP sanitize-preservation patch into VLM 加载 | 干净 |
| `42fc129` | `21f3342` | test: cover VLM MTP sanitize patch wiring | 干净 |
| `6f927ec` | `7b61c1f` | mtp: reconcile cache to standard state on batch reshape | 干净 |
| `ea2eaa1`(#1386) | `4eb5c7c` | oq: 拷 `processor_config.json` 保留 VLM 能力 | 干净 |
| `a53bf11`(#1356) | `76b4b7c` | Anthropic `tool_use` stream block indices | 干净 |
| `90d7e40`(#1392/#1393) | `afb6c88` | tool_calling: thinking tool call 用 name-matching 替 Guard 1 启发 | 干净 |
| `64f7d93` | `4eb0e8c` | tool_calling: 区分 `tools=[]` 与 `tools=None` | 干净 |
| `915190d`(#1388) | `c544c25` | mtp: 自愈 patches + dflash hook lifecycle wrap | `omlx/engine/dflash.py` —— 上游引入 `_evict_dflash_and_start_fallback` 跟 flyto 的 `_embedded_vlm` 双引擎不兼容,保留 flyto 路径;`install_dflash_lifecycle_wrap()` 移到 `_load_drafter_bundle`,`restore_dflash_class_patches()` 加到 `stop()`。`tests/test_mlx_lm_mtp_patch.py` `SimpleNamespace` import 漏合,手动补 |
| `878c892`(#1336) | `b33cb6a` | mtp: `_is_greedy` 检查真实 sampler.temp(#1388 测试的前置依赖) | 干净 |
| `56ae7f0`(#1404) | `56ae7f0` | mtp: `mtp_enabled=False` 时也 attach VLM MTPModule | `omlx/engine/vlm.py` —— 3 处冲突:① specprefill `_load_draft` 接受 upstream 版本(`set_mtp_active(False)` + finally restore);② / ③ chat template + token counting 保留 flyto 的 audio divergence |
| `ecb610e`(#1412) | `ecb610e` | load: mlx-vlm MoE sanitize 给 Qwen3.6 无 MTP head 的 VLM | 干净 |
| `ef49351`(#1383) | `25cdb67` | scheduler: 内存压力下 cap async store-cache pipeline | 干净 |
| `f0f3138`(#1389) | `f0f3138` | engine: guard late aborts after engine close | 干净 |
| `3b15958`(#1405) | `3b15958` | scheduler: hard-limit RuntimeError 后清 prefill 状态 | 干净 |
| `7d30401` | `7d30401` | vlm_mtp: 每轮清 mlx cache 限内存峰 | 干净 |
| `ea7efd4`(=0169f15) | `ea7efd4` | memory: aborted prefill 清 MLX cache + size-aware hard cap reserve | `omlx/process_memory_enforcer.py` docstring 单冲突取上游;`test_process_memory_enforcer.py` `user_explicit_max` 测试跳过(字段来自 C 组未引入的 `acd0533`)|
| `3af848b`(#684) | `3af848b` | engine: 每请求清 MLX cache(不仅 idle 时) | 干净 |
| `f6fdaf2`(=f1d1fc3 #1339) | `f6fdaf2` | hf: 跨域永久重定向 follow | 干净 |
| `5749613`(=5d8145b) | `5749613` | hardware: 用绝对路径调 macOS 系统工具 | 干净 |
| `31d31be`(=db07311) | `31d31be` | admin: 用绝对路径调 sysctl | 干净 |
| `ef1e842`(=7d640c1 #1417) | `ef1e842` | vlm: per-image lookup + whole-request fallback | 干净 |
| `5e394cf`(=1010fd3) | `5e394cf` | admin: 运行时 propagate `model_dirs` 到 OQManager + HFUploader | 干净 |

### 2026-05-27 memory_guard_tier 三段(PR-1 backend / PR-3 admin UI / 5 个 follow-up)

C 组 `c645c9f` 重写终于动手, 用 3 个 PR 分阶段落地(spike doc § 3.4):

| 阶段 | flyto PR | flyto commits | 内容 |
|---|---|---|---|
| spike doc | #4 | `0d2ec29` | 设计稿: `docs/memory-guard-tier-migration-spike.md` |
| PR-1 backend | #5 | `53ed139` `b9fa4a0` `07e46a6` | settings.py / process_memory_enforcer.py / engine_pool.py / server.py / cli.py: 把 `max_*_memory` 换成 `memory_guard_tier`. 老字段 / 老环境变量 / 老 CLI flag 保留 deprecated alias 一个 release. `ModelTooLargeError.max_memory` -> `.ceiling` |
| PR-3 admin UI | #6 | `f1c3d43` `80d7066` | 两个 slider 换成 tier dropdown; admin POST handler 修复(PR-1 backend 落地后, `routes.py` 写不存在的 `global_settings.model.max_model_memory` 会 500). i18n en/zh/zh-TW 补翻译; 其余 5 个语言先用英文占位 |

### 2026-05-27 C 组 5 个 follow-up(分支 `sync/memory-guard-tier-followups`,PR #7)

| 上游 commit | flyto commit | 内容 | 冲突处理 |
|---|---|---|---|
| `acd0533` | `bd9a159` | scheduler: adaptive prefill throttle + (legacy) user-explicit hard cap | settings.py / server.py / process_memory_enforcer.py 取 HEAD —— `user_explicit_max` / `max_process_memory_is_explicit` 已被 c645c9f 删, scheduler.py + 新 helper(`prefill_transient_tracker.py`)是本 commit 的实用部分; test 里 2 个 `user_explicit_max` 测试删 |
| `4cfbc8b` | `5b3fe20` | scheduler: 切到 watermark-tier shrink | 干净 |
| `3ef7b94` | `109ac76` | memory: clamp 到 effective Metal cap + sysctl 警告 | enforcer.py 主体 auto-merge; admin UI(routes/i18n/css/js)取 HEAD —— flyto 的 admin UI 是 PR-3 自己的形状, 上游 UI 改不直接适用; 测试取上游(`TestMetalWiredLimit`) |
| `b129a19`(#1425) | `7033c3b` | test: catch up renames from c645c9f + 沉默 enforcer 警告 | 干净 |
| `64bd2a2`(#1431) | `bdda9d2` | memory: tier-aware active-memory reclaim + Custom ceiling | settings.py `memory_guard_custom_ceiling_gb` 字段加; `MemoryGuardTier` Literal 加 `"custom"`; validate() 检查 custom > 0 ceiling; admin UI 取 HEAD(Custom 选项暴露留作后续) |

### 2026-05-28 独立修复批(分支 `sync/upstream-2026-05-28`,PR #8)

10 个跟 memory tier 无关的上游修复, 主要是 boundary-store race + 每引擎 stream + VLM 修复.

| 上游 commit | flyto commit | 内容 | 冲突处理 |
|---|---|---|---|
| `4f3a9b9`(#1423) | `7b2e849` | boundary-store: serialize cleanup_all + cleanup_request with writer thread | 干净 —— 消掉一直拖着的 `test_cleanup_all_drains_queue` flake |
| `bc1c427` | `0a65ddc` | boundary-store: drop unreachable shutdown(cleanup=) path | 干净 |
| `2916ab4`(#1422) | `89f3b99` | cache: 删 dead TieredCacheManager | 干净 |
| `56860b3`(#1304) | `fc26ab3` | engine: 每引擎线程 + mx.Stream, 消除 cross-engine 流污染 | scheduler.py / batch_generator.py 多处冲突 —— 取上游(`self._stream` 替模块级 `generation_stream`, 三段 phase timer 重构) |
| `a62f953` | `b7cb489` | engine: 删 redundant `_ensure_wired_limit` guard | 干净(依赖 56860b3) |
| `e6d8a3f`(#1445) | `c50d64e` | test(mtp): drop monkeypatch of removed `_get_generation_stream` | 干净 |
| `2e698ff`(#1437) | `f554f19` | scheduler: wait on generation_stream in store-cache worker | scheduler.py 主体 conflict —— 56860b3 已用更通用的 `_safe_sync_stream(self._stream)` 替, 取 HEAD; paged_ssd_cache 的非冲突部分留 |
| `9d5bed8` | `ebf2c21` | engine: VLM model lazy state 在 loader 线程实例化 | 干净 |
| `ff7522b` | `414b843` | load: checkpoint 无 mtp.* 权重时跳过 VLM MTPModule attach | 干净 |
| `0c881f5`(#1399) | `59d9e7e` | profiles: three-scope template contract + drop is_builtin emission | 干净 |

收尾 baseline: 4567 pass / 3 fail / 36 skip(3 fail 都是已知 OMLX_API_KEY env override flake; boundary_snapshot flake 由 #1423 修掉了).

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
- **2026-05-26 D 组冲突 commits**(低优,与 flyto 自身改动相撞,沿用上游
  版本价值不大):`cf4023c`(admin chat preserve_thinking history #1329)、
  `c4ebb7f`(hf_downloader cancel callback)、`f8174a9`(hf_downloader disable
  xet)、`2f2f508`(integrations env scrub #1350)、`8c70903`(responses
  tag-free as content #1348 —— flyto Responses 与上游 diverge 大)、
  `1b666af`(cache ssd_write_drops counter #1406)、`6a77fd5`(scheduler
  memcheck log gate —— 与已引入的 #1383 路径冲突)。
- **`d0f60ec`(#1344)dflash 多模态 VLM fallback** —— 上游用 lazy
  `_fallback_engine` swap + `supports_multimodal_fallback` content detect,
  flyto Path A 是永久 `_embedded_vlm` 双引擎,routing 基于 prompt token len
  而非 content。`message_extractor` hook 也只覆盖 gemma4/harmony,不通用。
  **2026-05-27 用 flyto 自己的设计落地了**,等价效果,不再 cherry-pick
  上游 commit(见下「2026-05-27 dflash 多模态路由」段)。

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
| `c645c9f` + `3ef7b94` + `4cfbc8b` + `acd0533` + `64bd2a2`(#1431) + `b129a19`(#1425) | memory: drop `max_*_memory`,add `memory_guard_tier` with dynamic ceiling(+ size-aware reserve、user-explicit hard cap、tier-aware active-memory reclaim) | 高(需 spike) | **breaking config**:删 `max_process_memory` / `max_model_memory`,换 `memory_guard_tier {safe,balanced,aggressive}`,涉及 29 个上游文件,flyto 113 处引用旧字段。spike 要覆盖:① 是否值得换 API;② settings.json 迁移路径;③ admin UI 改造;④ CLI args 改动;⑤ 用户已有配置兼容性。本次因 0169f15 已经引入,`user_explicit_max` test 临时 skip,等迁移落地一并打开。|
| ~~#1344~~ | ~~dflash 多模态请求 → VLM fallback~~ | ~~高(需 spike)~~ | **2026-05-27 用 flyto 自己的设计落地**(见下「2026-05-27 dflash 多模态路由」段),不走 cherry-pick 路径。|
| #1056 / #818 | dflash: 把带图请求路由到 VLM fallback engine | 待评估 | flyto 已用自己的 `supports_multimodal_fallback` + dflash.chat 路由覆盖 image,可能仍可参考 #818 的 audio / tts 部分。|

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
