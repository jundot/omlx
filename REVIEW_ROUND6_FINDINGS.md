# oMLX Review Round 6 — 新特性与残余面深审（进行中）

基线：main@1c55c03e（较第五轮 +445 文件/+30822 行：splash 引擎、GGUF 服务、决策引擎、集群 VLM 分布、缓存重构）。
缓存不变量回归检查：我 #4163 的 staging 标记在三次上游重构后完好，缓存套件 387 全绿。
omlx-web routes.py（182 行）：本人审，干净（路径穿越防护、locale 竞态良性）。

---

## A. splash / decision 新引擎（代理完成，执行验证）

### A-1【MEDIUM】`GET /api/models` 在事件循环上同步跑 `splash --version`，可冻结整个服务器最长 30s
调用链：admin/routes.py:958-963（同步 `_external_engines_for_model`）→ list_models 每模型循环（:2386-2393，无 to_thread）→ splash.py:192-205 阻塞 `subprocess.run(timeout=30)`。首启或 launcher 慢时全部流式响应/keepalive 冻结，每 60s 重试一次。inline repro 确认事件循环停顿。对照正确写法已存在于 routes.py:3142、engine_pool.py:2112-2127。**修复中**（worktree /tmp/omlx-fix-splash）。
### A-2【LOW】SSE 行解析无容错（splash.py:1274 坏 chunk → 500）。**修复中**
### A-3【LOW】看门狗父进程死亡分支缺 SIGKILL 升级（:861-874）。**修复中**
### A-4【LOW】`/v1/systemone` questions 无上限（`/v1/decisions` 有 MAX_QUESTIONS=200，此路径 model_construct 绕过）→ 租约内 ≥5000 次 readout 且引擎不可逐出。**报告**
其余备忘：_free_port TOCTOU（可自愈）、<1.3.2 静默丢 thinking_budget、日志无限追加。
已验证干净：路径穿越（repo_id 折叠+sha1 消歧实测）、上下文预算交接（#4677 链路）、SSRF（仅 data:image）、子进程生命周期与隔离（127.0.0.1+随机 env key+进程组收编）。

## B. speculative / specprefill（代理完成，执行验证）

### B-1【HIGH】SpecPrefill 准入可与已在解码的普通请求混批 → `_OffsetAdjustedRoPE` 平移整批所有行的 RoPE 位置
scheduler.py:12784 守卫只查本轮 `scheduled`，不查 `self.running/self.prefilling`；adjustment（16k prompt/20% keep ≈ 12.7k）作用于批内**每一行**（per-row offset 数组 + 单一 adjustment，inline repro：`[-3,150,152] + 79999`）。同批普通行后续 Q/K 旋转全部错位 + KV 永久劣化，静默无报错。**已修复**（worktree /tmp/omlx-fix-sp @ fbec2ac4，本人修，525 测试过）。
### B-2【MEDIUM】滑窗模型尾合并把 target.py 刚移除的 kickoff token 加回 → 最后一个 prompt token 双重处理（重复 KV、双倍注意力计权、计数偏 1）——正是注释声称避免的 "Bug #2"。**已修复**（sparse_prefill 增加 kickoff_index 参数，尾合并排除）。
### B-3【LOW→提问】`plan_specprefill_scoring` 的 `or` 把显式 0 覆盖当未设置——但测试套件**显式锚定** 0→默认（参数化用例），疑为有意设计。未改，PR 中提问。
已核查干净：位置链三路径自洽、稀疏 KV 全部隔离于前缀缓存、VLM 占位符永不丢弃、processing_sampler/vlm_mtp fail-closed、dflash 状态机。

## D. 推送受阻说明（重要）

三个修复簇（splash e60ab17d / api 31f00c6c / specprefill fbec2ac4）已完成并本地全绿，保存在 worktree。但：
- 旧基线 cherry-pick 推送法已失效：本轮触及文件（routes/openai_models/cli/scheduler）在 5dcfe243→1c55c03e 间有大量上游漂移（445 文件 delta），移植会产生反向漂移的脏 PR diff（实测确认污染后放弃，误推的 splash 坏分支已删除）。
- 直接推送当前 main 基分支被 OAuth workflow-scope 拒绝（复核确认）。
- **解锁方式：用户执行一次 `gh auth refresh -s workflow -h github.com`（浏览器确认），随后三个 PR + 积压的 #4157 rebase + workflow-followup.patch 立即可推。**

## C. cli.py + api 残余面（代理完成，执行验证）

### C-1【MEDIUM】`omlx launch` 对 IPv6 host 构造非法 URL 且错误被吞（cli.py:525-529 + integrations/base.py:39-40；`::1` 可正常 serve 但 launch 报"server not running"退出）。**修复中**（/tmp/omlx-fix-api）
### C-2【MEDIUM】环境变量覆盖被首启别名保存"冻结"进 settings.json（server.py:477-495 lifespan save 序列化了 env 合并后的内存态；save_cli_overrides 与 _migrate_saved_network_auth 都刻意避开 env 冻结，此路径违背设计意图）→ `OMLX_LOG_LEVEL=trace omlx serve` 一次即永久 trace。**报告**（修复涉及持久化语义，需维护者定夺）
### C-3【MEDIUM-LOW】CLI 与运行中服务器的 settings.json 丢更新竞态（全链路 read-modify-write 无跨进程锁；失败的 serve 启动也会先落盘）。**报告**
### C-4【LOW】rerank `top_n` 无下界（0→空结果 200，-1→少一条）。**修复中**
### C-5【LOW】embeddings `dimensions` 无下界（-1→静默 3 维向量）。**修复中**
### C-6【LOW】audio/chat 请求 max_tokens 与采样参数无界直通引擎。**修复中**
已验证干净：ThinkingParser 流式边界 9 组对抗用例、reasoning_effort 回退梯子、prefill_boundaries 无除零、system_sampler ctypes 细节。
