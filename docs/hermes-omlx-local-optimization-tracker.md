# Hermes + oMLX 本地优化跟踪

日期：2026-06-27

## 范围

本文档用于长期跟踪本机 Hermes + oMLX 组合使用时的性能、缓存、内存和稳定性优化。

当前从日志中观察到的部署形态：

- Hermes 默认 provider：`omlx`。
- oMLX base URL：`http://localhost:11335/v1`。
- 主模型：`Qwen3.6-35B-A3B-nvfp4`。
- Hermes 使用的 oMLX 辅助模型：
  - `Qwen3.5-4B-4bit`：用于压缩、标题生成、网页抽取和 skills hub 等辅助任务。
  - `Qwen3-Embedding-0.6B-8bit`：用于 embedding。
  - `bge-reranker-v2-m3`：用于 rerank。
- oMLX 源码启动命令：
  `scripts/start-omlx-app-config.sh --host 127.0.0.1 --port 11335`
- oMLX 当前关键设置：
  - `memory_guard_custom_ceiling_gb=76`
  - `max_concurrent_requests=2`
  - `ssd_cache_dir=/Users/zhouwei/.omlx/cache`
  - `ssd_cache_max_size=55GB`
  - `hot_cache_max_size=10GB`
  - `max_context_window=128000`
  - `max_tokens=16384`

## 当前日志快照

从 Hermes 本地日志解析出的 API 调用统计：

- 解析到的 Hermes API 调用总数：`6241`。
- 主本地模型调用：
  - `Qwen3.6-35B-A3B-nvfp4`：`5651` 次。
- 平均缓存比例：`92.6%`。
- 缓存比例范围：`2%` 到 `100%`。
- 缓存比例分桶：
  - `0-50%`：`191` 次。
  - `50-80%`：`305` 次。
  - `80-90%`：`380` 次。
  - `90-100%`：`5365` 次。
- 延迟：
  - 平均 `10.78s`；
  - p50 `5.5s`；
  - p95 `33.2s`；
  - 最大 `489.7s`。
- 输入 token：
  - 平均 `62533`；
  - p95 `149047`；
  - 最大 `237258`。
- 输出 token：
  - 平均 `235`；
  - p95 `824`；
  - 最大 `13699`。

oMLX `/api/status` 快照：

- `models_loaded=4`
- 已加载模型：
  - `Qwen3-Embedding-0.6B-8bit`
  - `Qwen3.5-4B-4bit`
  - `Qwen3.6-35B-A3B-nvfp4`
  - `bge-reranker-v2-m3`
- `model_memory_used=25.76GB`
- `model_memory_max=76.00GB`
- `total_prompt_tokens=7889207`
- `total_completion_tokens=41149`
- `total_cached_tokens=7098368`
- `cache_efficiency=90.0`
- `avg_prefill_tps=924.0`
- `avg_generation_tps=60.7`
- `active_requests=0`
- `waiting_requests=0`

oMLX server 日志样本：

- 解析到的 chat completion：`325` 次。
- 解析到的 embedding 请求：`300` 次。
- 解析到的 rerank 请求：`35` 次。
- 模型加载：`118` 次。
- 模型卸载：`110` 次。
- TTL 触发卸载：`16` 次。
- warning/error：`80` 条。
- Chat prompt 平均/最大：
  - 平均 `47458` tokens；
  - 最大 `140251` tokens。
- Chat 延迟平均/最大：
  - 平均 `7.57s`；
  - 最大 `184.30s`。
- 平均生成吞吐：`74.4 tok/s`。
- Embedding 延迟平均/最大：
  - 平均 `0.142s`；
  - 最大 `1.831s`。
- Rerank 延迟平均/最大：
  - 平均 `0.279s`；
  - 最大 `0.679s`。

## 主要发现

### 1. 缓存机制有效，但超长上下文仍是主要延迟来源

Hermes 的缓存比例整体较高。最近本地回合多次达到 `91%-100%` 缓存命中，oMLX 也报告 `cache_efficiency=90.0`。

慢路径不是普通的缓存失效，而是输入上下文过大：

- 最近 oMLX 最大 prompt：`140251` tokens；
- 历史 Hermes 最大输入：`237258` tokens；
- 历史最慢 API 调用：`489.7s`，输入 `179373` tokens，缓存只有 `11%`；
- 也存在缓存比例很高但总上下文过大导致仍然很慢的调用。

判断：

- 高缓存率能显著降低 prefill 成本，但不能让 `100k+` 上下文变得便宜。
- 当上下文很大时，即使只有较小的未缓存尾部，也可能带来明显 prefill 成本。
- 即使 oMLX prefix cache 有效，Hermes 侧的上下文压缩和工具输出治理仍然关键。

### 2. 主模型和辅助模型共用同一个 SSD cache 目录

当前 cache 目录：

```text
/Users/zhouwei/.omlx/cache
```

当前目录大小：

```text
55G /Users/zhouwei/.omlx/cache
```

该目录已经基本达到配置的 `55GB` 上限。

日志显示，不同模型或不同 cache signature 共用目录时，会产生大量 incompatible cache：

```text
SSD cache scan complete: scanned=551, indexed=71, errors=0,
total_size=7.84 GB, skipped_incompatible=479 blocks (47.44 GB)
```

在这次扫描中，对当前加载模型真正兼容可用的只有 `7.84GB`，但另有 `47.44GB` 被识别为 incompatible。也就是说，目录看起来有 `55GB`，但对当前模型的有效容量远小于这个数。

判断：

- 单个共享 SSD cache 目录使用方便，但不适合 Hermes 这种混合模型工作负载。
- `Qwen3.6-35B-A3B-nvfp4` 和 `Qwen3.5-4B-4bit` 的 cache signature 不兼容。
- 辅助模型 cache 会挤占或驱逐主模型 cache，降低主模型长期复用收益。

### 3. 辅助模型反复加载和卸载

日志显示模型生命周期抖动较明显：

- 模型加载：`118` 次。
- 模型卸载：`110` 次。
- TTL 触发卸载：`16` 次。

近期示例：

```text
TTL expired for model 'Qwen3.5-4B-4bit' (idle 301s > ttl 300s)
Unloaded model: Qwen3.5-4B-4bit
```

在一次大 prefill 中，oMLX 为主模型腾出 headroom，主动驱逐了空闲辅助模型：

```text
Request ... needs prefill headroom before throttling
Evicting idle model 'bge-reranker-v2-m3' for prefill headroom on
'Qwen3.6-35B-A3B-nvfp4'
Evicting idle model 'Qwen3-Embedding-0.6B-8bit' for prefill headroom on
'Qwen3.6-35B-A3B-nvfp4'
```

随后 Hermes 又需要 embedding/rerank，于是这些模型重新加载。

判断：

- 当前单个 oMLX 进程同时混用了：
  - 一个长上下文大主模型；
  - 一个 4B 辅助模型；
  - embedding 模型；
  - reranker 模型。
- 这个方案能工作，但 prefill headroom 和 TTL 策略会导致模型 churn。
- churn 会增加延迟，也可能加剧 allocator/cache 行为的不稳定。

### 4. Embedding compile fallback 反复出现

重复 warning：

```text
compiled embedding path failed for Qwen3-Embedding-0.6B-8bit:
[eval] Attempting to eval an array during function transformations like compile
or vmap is not allowed.; disabling compile and falling back to eager generate()
```

判断：

- oMLX 能正确恢复：禁用 compile 并回退 eager mode。
- 但每次重新加载后仍会重复出现 warning。
- 这大概率不是主要延迟瓶颈，但会污染日志；如果该模型路径已知无法稳定 compile，最好在加载前就跳过 compile。

### 5. Hermes 存在与生成性能无关但影响可维护性的配置/平台噪音

Lark/Feishu 重复失败：

```text
Could not find a suitable TLS CA certificate bundle, invalid path:
/Users/zhouwei/code/hermes-agent/venv/lib/python3.11/site-packages/certifi/cacert.pem
```

Hermes 配置 warning：

```text
providers.llama: unknown config keys ignored: model_type
providers.lms: unknown config keys ignored: model_type
providers.omlx: unknown config keys ignored: model_type
config.yaml has empty section(s): `context_file_max_chars`,
`max_concurrent_sessions`
display.personality is set but agent.personalities is empty/null
```

Browser CDP warning：

```text
Failed to resolve CDP endpoint http://localhost:9222
```

判断：

- 这些不是 oMLX 推理瓶颈。
- 但它们持续污染日志，也可能影响 gateway、cron、平台消息收发可靠性。
- `certifi` 问题需要优先修复，因为它会破坏 Feishu/Lark 重连和 cron job。

## 推荐优化计划

### P0：在单实例内实现逻辑 cache 域和策略治理

最新结论：不优先拆成两个 oMLX 实例。

拆分实例可以绕开主模型和辅助模型互相挤占的问题，但会增加 Hermes 路由、启动脚本、配置同步、监控和故障定位复杂度。更重要的是，如果只能靠多实例隔离 cache，说明 oMLX 单实例的多模型 cache 策略还不够成熟。更优雅的方向是在一个 oMLX 进程内，把 SSD cache 从“一个全局 LRU 池”升级为“按模型、签名、角色和热度治理的共享 cache”。

当前代码已经具备可利用的基础：

- hot cache 淘汰时会写回 SSD。
- hot cache 已经有 LRU 和共享预算机制。
- SSD cache metadata 已包含 `model_name`、`cache_signature`、`last_access`、`token_count`。
- SSD 层已有 compatible/incompatible index，并按 `last_access` 做 LRU。
- 已有 stale layer signature 的保守清理逻辑。

缺口不是缺少 cache 层，而是策略粒度偏粗：

- hot cache 被驱逐后，当前策略更接近“写回 SSD”，缺少准入判断。
- SSD 淘汰主要看全局 `last_access`，没有充分区分主模型、辅助模型、当前活跃 signature、命中价值和重算成本。
- incompatible block 会被统计和参与淘汰，但缺少比例上限和后台主动清扫策略。
- 模型卸载和 cache 降权之间缺少明确关系。

建议的单实例机制：

1. 逻辑 cache 域

   在同一个 `ssd_cache_dir` 内按以下维度划分逻辑域，不拆进程、不拆服务：

   - `model_name`
   - `cache_signature`
   - `model_role`，例如 `primary_chat`、`assistant_chat`、`embedding`、`rerank`
   - compatible / incompatible

2. SSD cache 准入策略

   不要所有 hot cache eviction 都无条件落盘。优先持久化：

   - hot cache 命中率高的 block；
   - 被 hot cache 驱逐、但近期有复用证据的 block；
   - token 数较大、重算成本高的 block；
   - 属于主 chat 模型稳定 prefix 的 block；
   - 当前活跃 `cache_signature` 下的 block。

   降低或跳过持久化：

   - 一次性长尾 prompt；
   - 低命中辅助模型 block；
   - rerank/embedding 的短生命周期 block；
   - 已经属于 inactive signature 的 block。

3. 按域预算

   保持单实例，但给不同逻辑域设置软预算：

   - 主 chat 模型：保底大部分 SSD cache，例如 `70%-80%`。
   - 辅助 chat 模型：小比例预算，例如 `10%-20%`。
   - embedding/rerank：默认很小，必要时只进入 hot cache，不长期 SSD 持久化。
   - incompatible/stale：设置硬比例上限，例如不超过总 SSD cache 的 `5%-10%`。

   这些预算应是软约束：空闲空间可以共享，但发生压力时按域回收。

4. 淘汰评分

   从单纯全局 LRU 升级为评分制。候选字段：

   - `last_access`
   - `hit_count`
   - `hot_cache_eviction_count`
   - `model_role`
   - `token_count`
   - `cache_signature` 是否当前活跃
   - 是否 compatible
   - 是否超过域预算或 TTL

   淘汰优先级建议：

   ```text
   stale/incompatible
   > 超过预算的低命中辅助模型 block
   > 冷的辅助模型 block
   > 冷的主模型 block
   > 高命中主模型 block
   ```

5. 后台 cache janitor

   增加轻量后台清扫器，避免在请求路径里一次性清理大量文件：

   - 定期释放长期未访问 block；
   - 控制 incompatible/stale block 占比；
   - 空闲时清理 inactive signature；
   - 分批 unlink，避免推理线程被文件删除阻塞；
   - 输出每个逻辑域的 cache 使用量和清理原因。

   互斥要求：

   - 当所有模型都已经卸载，并进入 cache 自愈/内存清理治理作业后，后续模型加载和业务请求必须等待治理完成。
   - 治理作业应有明确的 maintenance gate，不能与 `get_engine()`、模型加载、TTL 卸载、手动卸载或 runtime settings reload 并发修改 engine/cache 状态。
   - 治理期间可以拒绝或排队新请求；更适合本地 Hermes 场景的策略是排队等待，而不是直接返回错误。
   - 治理必须是 bounded：限制单轮 unlink 文件数和最长耗时，避免最后一个模型卸载后长时间阻塞下一次加载。
   - 互斥治理不应调用 `clear_hot_cache()` 或 `SharedHotCacheBudget.clear_all_owners()`，因为目标是保留 hot cache，只治理 SSD cache 和释放模型运行态内存。

6. 模型卸载时只降权，不立即清 cache

   模型卸载应该释放内存，但不等于删除有价值的 SSD cache：

   - 主模型卸载：保留高命中、高 token 成本 block；
   - 辅助 chat 模型卸载：降低优先级，只保留近期命中过的少量 block；
   - embedding/rerank 卸载：优先短 TTL 或不长期持久化。

预期收益：

- 保持单 oMLX 实例，降低配置和维护复杂度。
- 避免辅助模型 cache 长期挤占主模型 cache。
- 提高 `55GB` SSD cache 的有效可用比例。
- 降低 incompatible block 占用。
- 为后续多模型长期运行提供稳定机制，而不是依赖部署拓扑规避。

执行顺序：

- [x] 增加 per-role/per-signature cache 统计输出：`get_stats_dict()` 新增 `cache_domain_usage`。
- [x] 为 SSD metadata 增加 `hit_count`、`last_hit_at`、`model_role`、`hot_cache_evictions` 策略字段，并保持旧 metadata 兼容读取。
- [x] 在 hot cache eviction 写回 SSD 前增加准入判断：默认保持既有写回契约，只对明确的低复用 embedding/rerank block 降低持久化优先级。
- [x] 在 SSD 淘汰逻辑中加入域预算和评分制：优先清理 incompatible/inactive signature，再清理超过预算的 embedding/rerank 和辅助 chat 域，保护高命中主 chat block。
- [x] 增加显式 cache domain janitor：`run_cache_domain_janitor()`，当前为手动/后续调度调用，不额外启动后台线程。
- [x] 增加测试覆盖：主模型高命中 block 不应被低命中辅助模型优先挤出；incompatible/stale 占比超过阈值时应优先清理；低复用 embedding hot eviction 可跳过 SSD 持久化。
- [x] 增加 maintenance gate：全模型卸载后的治理作业执行期间，`get_engine()` 和后续模型加载请求必须等待治理完成。
- [x] 将治理接入最后一个模型卸载后的空闲窗口：当没有 loaded/loading/in-use/pending unload 模型时，执行 SSD cache janitor 和额外 MLX 内存清理。
- [x] 治理使用临时 SSD cache manager，`hot_cache_max_bytes=0`、`hot_cache_budget=None`，不调用共享 hot cache 清理。

实现记录：

- 分支：`feature/single-instance-cache-domains`。
- 核心代码：`omlx/cache/paged_ssd_cache.py`、`omlx/engine_pool.py`。
- 测试代码：`tests/test_paged_ssd_cache.py`、`tests/test_engine_pool.py`。
- 策略边界：
  - 不拆分 oMLX 实例；
  - 不拆分 `ssd_cache_dir`；
  - 不改变默认 hot cache write-back 契约；
  - 只对明确低价值的 embedding/rerank 短生命周期 block 做 SSD 准入降级；
  - SSD 淘汰使用小窗口候选评分，不在请求路径做全量扫描。
  - 全模型卸载后的治理会阻塞后续模型加载，直到治理完成。
  - 治理只扫描/清理 SSD cache 和执行 MLX allocator 清理，不释放共享 hot cache。
- 已验证：
  - `git diff --check` 通过；
  - `.venv/bin/python -m compileall -q omlx/cache/paged_ssd_cache.py tests/test_paged_ssd_cache.py` 通过；
  - `.venv/bin/python -m pytest tests/test_hot_cache.py tests/test_paged_ssd_cache.py` 通过，`185 passed`。
  - `.venv/bin/python -m compileall -q omlx/cache/paged_ssd_cache.py omlx/engine_pool.py tests/test_paged_ssd_cache.py tests/test_engine_pool.py` 通过；
  - `.venv/bin/python -m pytest tests/test_hot_cache.py tests/test_paged_ssd_cache.py tests/test_engine_pool.py` 通过，`284 passed`。
- 未完成/后续：
  - `uv run ruff ...` 当前仍受依赖解析问题阻塞：解析 Python 3.14 split 时找不到 `num2words>=0.5.14`；本地 `.venv` 中没有 `ruff` 可执行文件。
  - 当前 janitor 已接入“全模型卸载后”的空闲窗口，但尚未暴露管理 API 或状态 API。

### P1：清理或扩容当前 SSD cache

当前状态：

- `ssd_cache_max_size=55GB`。
- cache 目录已经是 `55G`。
- 最近扫描中，对某个模型只有 `7.84GB` compatible，另有 `47.44GB` incompatible。

方案：

1. 短期：停服后清理当前 SSD cache。
2. 中期：如果磁盘允许，把 SSD cache 预算提高到 `100GB-150GB`。
3. 更优方案：扩容与“单实例内逻辑 cache 域、准入策略、域预算和后台清扫”一起做。

跟踪项：

- [ ] 记录基线：`du -sh ~/.omlx/cache`。
- [ ] 记录每个模型启动时的 SSD scan 统计。
- [ ] 切换 cache 策略时清理 stale/incompatible cache，并记录清理原因。
- [ ] 预热后重新检查 `indexed` 与 `skipped_incompatible`。

### P1：减少不必要的模型 churn

当前问题：

- 部分模型 TTL 实际约为 `300s`。
- 辅助模型会在 Hermes 活跃工作流中卸载又重载。
- 长 prefill 会为了 headroom 驱逐 embedding/reranker。

可能策略：

- 提高常用辅助模型 TTL。
- 在单实例内降低 embedding/rerank 的 SSD 持久化优先级，优先让它们使用 hot cache 或短 TTL cache。
- Hermes 工作期间避免在同一个 oMLX 进程里加载探索性大模型。
- 如果内存紧张，优先调整模型 TTL、cache 准入和域预算，而不是拆分进程。

跟踪项：

- [ ] 每天统计 load/unload 次数。
- [ ] 跟踪 embedding 和 rerank 的 p95 延迟变化。
- [ ] 跟踪 prefill headroom 是否仍会驱逐辅助模型。
- [ ] 跟踪辅助模型被卸载后，其 SSD cache 是否按策略降权或清扫。

### P1：降低 Hermes 上下文压力

当前模式：

- 平均输入：`62533` tokens。
- p95 输入：`149047` tokens。
- 历史最大输入：`237258` tokens。

Hermes 侧可做的工作：

- 对本地模型会话使用更激进的压缩策略。
- 降低保留的工具输出体积。
- 避免把长日志或大文件反复注入主对话上下文。
- 优先传递文件引用和聚焦摘录，而不是全文塞进主上下文。
- 评估 `compression.protect_last_n=30` 是否过高；如果质量允许，可以降低。

跟踪项：

- [ ] 跟踪每轮 Hermes 输入 token 的 p50/p95。
- [ ] 按缓存率分桶跟踪延迟。
- [ ] 调整压缩策略后跟踪质量退化。

### P2：禁用已知不稳定的 embedding compile 路径

当前问题：

- `Qwen3-Embedding-0.6B-8bit` 反复尝试 compile，失败后回退 eager。

可能修复：

- 如果 oMLX 已支持相关配置，为该模型显式跳过 embedding compile。
- 如果没有配置，可以考虑一个窄补丁：记住该模型/config 的 compile 失败，重载时直接 eager。

跟踪项：

- [ ] 检查是否已有 embedding compile 控制项。
- [ ] 若无控制项，记录为小型后续 issue 或补丁。
- [ ] 修改后验证 embedding 延迟和 warning 数量。

### P2：清理 Hermes 配置和平台噪音

事项：

- 修复 Hermes venv 中的 `certifi` 路径。
- 如果当前 Hermes 不再接受 `model_type`，从 provider 配置中移除该字段。
- 将空 YAML section 改成 `{}` 或删除：
  - `context_file_max_chars`
  - `max_concurrent_sessions`
- 如果不使用 browser tools，禁用 CDP；如果使用，则启动 `localhost:9222` Chrome debug endpoint。

跟踪项：

- [ ] 修复 certifi 路径，并验证 Lark/Feishu 可重连。
- [ ] 确认不再重复出现 `unknown config keys ignored`。
- [ ] 确认不使用 browser tools 时不再重复出现 CDP warning。

## 长期跟踪指标

每轮调优至少采集一次：

```bash
curl -s -H "Authorization: Bearer <OMLX_API_KEY>" \
  http://127.0.0.1:11335/api/status

du -sh ~/.omlx/cache ~/.omlx/cache/vision_features 2>/dev/null

rg "SSD cache scan complete|PagedSSDCacheManager initialized" \
  ~/.omlx/logs/server.log

rg "API call #[0-9]+:.*cache=" ~/.hermes/logs/agent.log

rg "Loading model:|Unloaded model:|TTL expired|prefill headroom" \
  ~/.omlx/logs/server.log
```

建议长期跟踪字段：

- Hermes：
  - 每轮 API 调用次数；
  - 输入 token p50/p95/max；
  - 输出 token p50/p95/max；
  - 缓存比例 p50/p95/min；
  - 延迟 p50/p95/max。
- oMLX：
  - 已加载模型；
  - 模型内存占用；
  - total cached tokens；
  - cache efficiency；
  - SSD compatible vs incompatible block；
  - hot cache size 和 hot cache hit；
  - load/unload 次数；
  - prefill headroom eviction 次数。

## 当前判断

当前组合已经明显受益于 oMLX prefix cache。最大收益不太可能来自拆分 oMLX 实例，而更可能来自在单实例内完善 cache 准入、逻辑域预算、淘汰评分和后台清扫机制，同时控制 Hermes 上下文增长、降低辅助模型 churn。

优先级：

1. 在单 oMLX 实例内实现逻辑 cache 域、准入策略、域预算和后台 janitor。
2. 策略上线后清理或扩大 SSD cache。
3. 降低 Hermes 上下文大小和重复大工具输出。
4. 减少辅助模型加载/卸载 churn。
5. 清理 Hermes 配置和日志噪音。
