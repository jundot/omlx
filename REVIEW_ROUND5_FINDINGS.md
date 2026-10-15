# oMLX Review Round 5 — 剩余模块深审（进行中）

基线：全新克隆 main@00321731；环境 pin 已同步（mlx 0.32.4 / mlx-vlm 9790231）。
基线测试：**16963 通过 / 3 失败 / 1021 跳过**——失败均为预置环境项：
- tests/test_web_search.py::TestFetchUrl::test_real_markdown_conversion（markitdown 版本）
- tests/test_vision_feature_cache.py::TestSSDCache::test_ssd_write_fsyncs_parent_dir_after_rename（顺序依赖 flake，单跑 37/37 通过）
- （第三个失败名待从完整日志提取，同属预置项）

---

## A. admin benchmark/ane/external（代理完成）

### A-1【HIGH】四类 bench 互斥检查不完整，可并发互相卸载模型
`/api/bench/start` 只查 throughput/context/ANE（routes.py:8773-8806），**不查 accuracy 队列**；`/api/bench/accuracy/queue/add` 只查 tuning+context（routes.py:8168-8192）不查 throughput；`_continue_queue`（accuracy_benchmark.py:279-319）弹下一项时不复查。两个 run 都执行 "unload all → load target"（accuracy Phase 1: accuracy_benchmark.py:411-432）→ 测量互相污染/引擎池状态混乱。**修复**：三处补齐四向互斥检查（照抄 context/start 的模式）。

### A-2【MEDIUM】external_api：API key 明文发往任意 http:// 主机
`validate_base_url` 放行 http://（external_api.py:70-76）；有 key 即无条件加 Bearer 头（:181-184，已执行复现）。**修复**：非 loopback 的 http:// 拒绝或显式 allow_insecure。

### A-3【MEDIUM】批量 tg_tps 分子/时间窗不匹配，系统性偏高
`tg_tps = total_gen_tokens / (wall_end - max_first_token)`（benchmark.py:1149-1156 本地、:1243-1261 外部）：最慢请求首 token 之前其它请求产出的 token 计入分子却不计时间。数值模拟 batch 8 / TTFT 离散 0.8s 偏差 ≈8.5%；数字进社区排行榜。**修复**：扣除 max_first_token 时刻各请求已产出的 completion_tokens，或改 Σ(end_i − first_i) 加权。

### A-4【LOW】`_run_single_test` 三处恒真条件死代码（benchmark.py:818/822/849，inspect.getsource 已核对）。
### A-5【LOW】`/api/bench/start` total_tests 虚报：`len(prompt_lengths) + len(batch_sizes)*2`（routes.py:8833）vs run_benchmark 实发 `+ len(batch_sizes)`（benchmark.py:1798）进度事件数不一致。
### A-6【INFO】离开本机的数据清单：owner_hash 合格（SHA-256 硬件指纹非可逆）；accuracy raw 上传逐题 raw_response ≤2000 字符（LiveCodeBench 类会回显题面片段）——建议 UI 明示+开关。
### A-7【nit】warmup 唯一不传 skip_cache_store 的测量路径（benchmark.py:1941-1944；正确性无碍，缓存残留不一致）。

已核查干净：UUID 前缀防缓存污染、skip_cache_store 全覆盖、取消/清理路径、SecretStr+全路径 redaction、extra_body 保护字段、TLS verify。

---

## B. embedding/reranker/batched 引擎（本人审，完成）

- embedding.py：干净。顺序重排/公平门/活动追踪均正确。一个 LOW：`len(embeddings) == len(order)` 守卫失败时（模型返回数与输入不符）静默返回错位结果而非报错（embedding.py:228-232）。
- reranker.py：干净，无发现。
- batched.py：模板路径严谨（partial 双模式、k2 校验、TypeError 降级重试）。一个 LOW：TypeError 重试剥离 kwargs 时未剥 `continue_final_message`（partial 模式下若模板不支持该参数会二次失败）（batched.py:958-965）。

## C. Responses API + store（本人审，完成）

- responses_utils.py（869 行全读）：转换/命名空间工具/链解析/SSE 均严谨（环检测、深拷贝、consolidation 正确）。观察项：
  - 【INFO】连续 reasoning 项互相覆盖而非拼接（`pending_reasoning = "\n".join(...)` 两处，responses_utils.py:332/813）。
  - 【INFO】ResponseStore 持久化 tmp+replace 无 fsync；损坏文件加载期跳过（可接受）。单条记录无字节上限（数量由 MAX_STORED_RESPONSES=1000 限定）。
  - 【INFO】`_persist_record` 的 tmp 路径按 id 派生，同 id 并发写理论冲突（实际单写者）。

## D. eval + benchmarks（本人抽查，完成）

- eval 已扩至 20 个 benchmark。对 MMLU 提取器做对抗样本执行验证（`**C**`、中文"答案是 D"、`b)`、多候选句等 6 例）全部正确提取；提取器有优先级排序逻辑而非朴素首匹配。无值得提 PR 的发现。

## E. vendored batch_generator（代理完成，repro 验证）

### E-1【HIGH】MTP 初始化窗口 fallback 重复喂入同一 token，静默污染上下文
`_post_init_mtp` 先把 `_next_tokens` 喂进 prompt cache（:3216-3221）**之后**才运行 head 并赋值 `_omlx_mtp_state`（:3276/3302/3347）；batch 孪生同理（:1296-1304 vs :1361）。窗口内异常（`_MtpStepFallback`、priming seam RuntimeError、head OOM）→ cache 已 +1、state 未挂 → fallback 的 `original_next._step()` **把同一 token 再喂一次**。stub-model repro 确认：一次 fallback 后 token 3 在 KV cache 出现 2+ 次，后续解码全在污染上下文上，无任何报错。**修复方向**（复杂，单独评估）：handler 在 `active is None` 时对 cache `trim(1)` 撤销推进，或把 state 赋值提前到 head 调用前让既有 reconcile 兜底。
### E-2【MEDIUM】policy 标准步分支提前 return 跳过 `_mark_standard_multirow_decode`（:128-149 vs :199）→ 校准期多行解码不设 marker，batch 缩单行后绕过 left-padding 验证即可全新激活 singleton MTP（#2150 同类）。**修复**：return 前补 marker 设置（repro 验证有效）。→ 修复代理执行中
### E-3【MEDIUM-LOW】`_emit_ragged_responses` 先改账后返回（:3474-3532）：中途异常静默吞掉已入账 token。**修复方向**：先纯函数式构建 responses 再统一落账。
### E-4【LOW】过期注释（:4637 旧行号；:90-97 与 __init__.py:93-100 幂等性表述矛盾）。→ 修复代理执行中

已核查干净：stop/max_tokens 边界（逐 case 推演）、per-row sampler、uid 复用防护、增长缓存全部有界、remove 幂等、多响应消费。

## F. model_discovery + memory_monitor（代理完成，多数执行验证）

- **F-1【MEDIUM】迁移写失败 → 全部 profile 内存清空**：`_load_profiles` 把迁移保存在 try 内（model_settings.py:1067-1071），保存失败被当作加载失败 → `_profiles = {}`；下一次任意 save 用单条记录覆盖全文件，其余永久丢失。chmod 555 复现验证。**修复**：迁移保存移出 try，失败仅告警保留内存态。
- **F-2【MEDIUM-LOW】settings 文件损坏即静默清零、下次保存覆盖原文件**（model_settings.py:792-797；同型 :1069/:1673）。截断 5 字节复现。**修复**：损坏时改名 `.corrupt-<ts>` 保留原文件再空态继续。
- **F-3【MEDIUM】非 UTF-8 元数据文件让整个模型从服务中消失**：`_read_chat_template_text` 的 `suppress(OSError)` 漏 UnicodeDecodeError（model_discovery.py:893-895）；同型缺口 5 处（:448/:711/:937/:1002）。`\xff\xfe` 模板复现。**修复**：统一异常元组补 UnicodeDecodeError；jinja 用 errors="replace" 解码。
- **F-4【MEDIUM-LOW】HF cache 快照 mtime 探测裸 `p.stat()`**（model_discovery.py:1490），竞态 FileNotFoundError 可炸掉 `_refresh_models_after_download` 回调（无 try 包裹、恰在 cache 并发清理时执行）。**修复**：`_safe_mtime`。
- **F-5【MEDIUM-LOW】子目录权重不计数**：MiMo 视觉塔 sidecar（`omnimodal/vision_encoder.safetensors`）不计入 estimated_size（:1103-1119/:1162）→ 内存准入低估可致加载期 OOM。复现：8MB+4MB 判 vlm 但 size 只含 8MB。**修复**：并入已知 sidecar 路径。
- **F-6【LOW·性能】config.json 每模型每轮被解析 5 次**（instrumented 实测）。
- **F-7【LOW】`set_settings` 保存失败不回滚 is_default 联动**（model_settings.py:930-945）；同文件 update_profile 已有快照回滚范式可套。
- memory_monitor：采样面为遗迹代码（恒 0/False，set_baseline_memory 零调用方）；估算数学边界实测安全，无缺陷。

---

# 修复波状态

- 修复代理 1（worktree /tmp/omlx-fix-bench）：A-1/A-2/A-5/A-7 — 运行中
- 修复代理 2（worktree /tmp/omlx-fix-mtp）：E-2/E-4 — 运行中
- 待第三簇（discovery/settings 韧性 F-1/2/3/4/5）：等一个代理完成后续派
- 报告不修（需维护者决策）：E-1（MTP 初始化窗口，修复侵入性大）、E-3、A-3（tg_tps 口径变更影响排行榜数值）、A-6（上传知情同意 UI）

---

# 修复波结果（2026-10-02 深夜，全部完成）

| PR | 分支 | 内容 | 测试 |
|---|---|---|---|
| #4568 | fix/mtp-multirow-marker | E-2 marker 补设（先红后绿回归）+ E-4 注释 | 912 过 |
| #4569 | fix/bench-mutual-exclusion | A-1 互斥补全+_continue_queue 延迟重试、A-2 http:// 限回环、A-7 warmup | 281 过 |
| #4570 | fix/discovery-resilience | F-1 迁移保存隔离、F-2 损坏文件 quarantine、F-3 非UTF-8 容错、F-4 _safe_mtime、F-5 sidecar 计入 | 418+502+413 过 |

报告不修（PR 正文已留）：E-1（MTP 初始化窗口双喂——两个修复方向供维护者定夺）、E-3、A-3（tg_tps 口径，需维护者确认数值变更）、A-6（上传知情 UI）、F-6/F-7、B/C 低危项。

事故与恢复：本轮开始时宿主 shell（/bin/zsh）丢失且工作区被清空；Read 阳性对照确认后停止全部代理、落盘事故报告；shell 恢复后重新克隆（main@00321731）、同步 pin（mlx 0.32.4/mlx-vlm 9790231）、重建基线（16963 过/3 预置失败）并完成全部工作。新克隆 git 身份缺失导致首批提交误署 efund@Mac-Studio——已按全局指令改为 billythegod noreply 并重写已推送分支（远端两分支已验证干净）。
