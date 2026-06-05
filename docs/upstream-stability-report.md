# 上游 omlx 稳定性问题报告 + flyto 对照

调研日期 2026-06-05. 数据源: jundot/omlx 的 open + closed issues (15.9k star, ~480 open).
方法: 子代理深读 7 大类 issue 的 maintainer 定性 + 关单 commit, 主代理对候选做 git
cherry-pick 实测 + 对照 flyto 台账 (docs/upstream-sync.md).

## 给决策者的一句话

大多数"干大活崩 / 死机"的最底层根因是 Apple GPU 驱动 / MLX 框架的 bug (下游改不掉,
上游想修的 per-stream 锁还被 MLX 团队拒了). 但上游反复证明: 真正让它"频繁触发"的, 是
omlx 自己代码里几个内存释放时机 / cache 重建 / batch 状态泄漏的 bug, 这些上游在下游修
掉了. 所以**"把崩溃频率大幅降下来"能靠 cherry-pick 做到, "彻底不崩"做不到**. flyto
目前疑似漏了好几个这类降频修复 (下面清单, 待逐个核实落地).

实测重要提醒: 16 个候选 cherry-pick 实测**全部** MISSING, 但 flyto 常换形状引入 (如内存
管理那套已通过 memory_guard_tier 重写引入, 实测仍报 MISSING). 所以 conflict/apply ≠
确定缺, 每个都要对台账 + 代码核实再落地.

## 7 大类问题 (上游事实)

### 1. Kernel panic / Metal GPU 崩溃 / 死机重启 -- 头号痛点

- 症状: 跑着跑着整台 Mac 黑屏重启 (kernel panic), 或 server 进程被 Metal 杀掉.
- 触发: Qwen3.5/3.6-35B-A3B 大 MoE, 持续负载几分钟 (agentic / 并发), 开 SSD cache 更勤;
  长上下文 / 内存吃紧风险升高.
- 根因 (分层, 最重要):
  - 最底层 = **平台层 (改不掉)**: `IOGPUMemory.cpp:550 completeMemory() prepare count
    underflow` (Apple GPU 驱动显存引用计数下溢) + MLX Metal backend 线程竞争. jundot:
    "this is not an oMLX regression but an MLX framework bug. The upstream fix
    (ml-explore/mlx#3247) was declined by the MLX team".
  - 触发器 = **omlx 可修, 已修**: omlx 在请求结束 / 周期性 `mx.clear_cache()` 一次性整池
    释放显存, 制造一批 IOGPU 引用计数翻转, 撞上驱动 bug. 改释放时机就能大幅降频.
- 上游修复: `1831649` (gate periodic mx.clear_cache on accumulated bytes) + `f3859bd`
  (drop worker-thread mx.eval in async store_cache) + `37c73a0` (phys_footprint enforcer
  + 准入控制) + `196d667` (队列满 -> 503). **警告**: `37c73a0` 这套 Memory Guard 正是
  Cluster 6 性能回退的起因, 要连 `1efb140`+`fd10281` 调参一起拿.
- 运维 workaround (平台层降不掉时): `MLX_MAX_OPS_PER_BUFFER` 从默认降到 10-20,
  `MLX_MAX_MB_PER_BUFFER` 同理; 关 SSD cache; 留内存余量; 限上下文.
- 代表 issue: #300 (43 评论, 旗舰) #978 #435 #173 #248 #1372 #511.

### 2. DFlash 投机解码无限循环 / 卡死

- 症状: 开 dflash (或 Qwen3.6 自己) 陷入无限思考 / GPU 99% 卡死不返回, 跑满 max_tokens.
- 触发: Qwen3.6-35B-A3B / 27B (4bit/8bit/oQ 都中); #131 是 VLM 并发 prefill.
- 根因 = **omlx / 依赖集成层**: 采样确定性回归 + Qwen 前向与 mlx-lm 对不齐 + mlx-lm cache
  结构变更. #131 是 omlx bug: Qwen3.5 缓存了 stale `_position_ids` (mRoPE), 第二个 prompt
  复用旧 position 导致 broadcast 失败.
- 上游修复: `9d742d193` (#131 clear stale mRoPE) + `a3c249b7d` (cache 坏自动清重跑防死循环)
  + `cb33a761f` `318678019` (#934 align forward + per-row logits_processors) + `69becb31`
  (#911 dflash-mlx 0.1.5.1 overhaul).
- 注: flyto 自己修过 gemma 上的 dflash 无限循环 (PR #29), 但 qwen #131/#934 可能没覆盖.
- 代表 issue: #934 (57 评论, 最热) #911 #131.

### 3. MTP / 批处理下崩坏 / 慢 / 输出乱

- 症状: 开 native MTP 的模型, batch>=2 / 并发下输出乱码 / 串中文 / 准确率掉到 35% / 更慢;
  单流正常.
- 触发: `*-mtp` 模型 (Qwen3.6-35B-A3B-oQ6-mtp / 27B) 且 batch>=2 或并发.
- 根因 = **omlx, 部分按设计不支持**: MTP UI 标注 "Single-stream only", 多请求本就不打算
  支持; bug 是 batching guard 没挡住, MTP 状态从 reshape/late-join 漏进不安全路径导致 cache
  corruption. **运维规则: 开 MTP 就别上并发/批处理.**
- 上游修复: `9aed907` (#1550 safe row-wise decoding for aligned batches) + `60c26b6`
  `878c8925` (#1097 测量假象修正, 878c8925 台账已引入).
- 代表 issue: #1550 #1504 #1551 #1401.

### 4. KV cache 落盘 / cache corruption

- 症状: `Cache corruption not recoverable: [broadcast_shapes]` 或 `'BatchKVCache' has no
  attribute '_quantized'`; 早期 SSD cache 启动扫描触发 GPU hang / OOM 崩溃循环.
- 触发: 并发请求打同一模型; SSD prefix cache 部分命中; 大量 SSD 文件启动扫描.
- 根因 = **全是 omlx 自己的 Python bug**: #409 SSD prefix cache 存错 KVCache offset;
  #422 cache-merge prefill 漏 turboquant 转换; #25 SSD cache deadlock -> hang -> 重启 ->
  驱动显存泄漏累积 -> OOM 循环.
- 上游修复: `abaa478` (#409) + `014b17f` (#422) + `170cec9` (#25 异步后台写) + `e693921`
  (#1413 SSD stale block 校验).
- 代表 issue: #409 #422 #463 #25 #15 (全 closed).

### 5. 长上下文下崩溃 / 静默挂掉

- 症状: 上下文涨到 81.5K / 120-130K / >157K 时 server 进程突然死或不响应, 日志空白.
- 触发: Qwen3.6-27B-8bit ~80K; Qwen3.5-122B-A10B-oQ4 ~157K.
- 根因: #511 = **平台层** (Metal 驱动在 `encodeSignalEvent` 内部 assert SIGABRT, 大上下文
  GPU 资源压力下驱动状态变坏); #1014 = 同 Cluster 1 的 store_cache 增长模式.
- 上游修复: #511 没修复 commit (平台层, 只加了 faulthandler 出 crash.log 诊断); #1014 用
  Cluster 1 的 `1831649`/`f3859bd`. **运维上长上下文只能限长度 / 留内存余量.**
- 代表 issue: #511 #1014 (都 open, 长上下文是仍未根治的薄弱区).

### 6. 性能回退 (新版变慢)

- 症状: 升级后 TTFT +15-20%, prefill/generation 吞吐都掉.
- 触发: v0.3.10->v0.3.12 / v0.4.0; 低内存机 (32G/64G) 更明显, 长 prompt 更明显, 即使
  MTP/dflash/SSD 全关也中.
- 根因 = **omlx, Memory Guard 调参问题**: 节流从 ceiling 68% 就开始, 32G 机上模型本身吃
  19G 就触发, 每个 prefill chunk 被无谓砍小.
- 上游修复: `1efb140` (节流起点 68%->80%, 软驱逐 85%->90%) + `fd10281` (custom tier 2G
  reserve). #1630 (v0.4.0) jundot 说是最高优先级但仍在查, 未根治.
- 关键依赖: 这套回退起因正是 Cluster 1 的 Memory Guard (37c73a0). 要么连调参一起拿, 要么
  承担回退.
- 代表 issue: #1630 #1519 #114 (未解决) #1097 (按设计).

### 7. thinking 不停 / 只输出 thinking -- 置信度最低

- 症状: preserve_thinking 不生效 / 只返回 reasoning 没 content.
- 根因: **maintainer 基本证伪**. #900 jundot 复现不出, 指测试方法是 re-derivation 非
  recall; #903 在 7 个 Qwen3.6 变体复现不出, 逐字节 diff 模板一致. 偏未确认.
- 上游修复: 无关单修复 (没确认是 bug).
- 代表 issue: #900 #903 (都 open, 未被确认为 bug).
- 注: flyto 自己这次的 gemma4 12b "thinking 没正文" 是另一码事 (图像 prefill, 已由 PR #46
  修, 见 docs/upstream-sync.md).

## flyto 行动清单 (待下个会话逐个核实 + 落地)

每个候选都要: (1) 在 upstream/main 确认真实 SHA; (2) 对台账核实是否换形状已引入;
(3) 读代码确认 flyto 是否已有等价; (4) 确认缺的才 cherry-pick (优先级见下).

实测信号 (cherry-pick 实测, conflict/apply 不代表确定缺):

| 候选 | 解决 | 实测 | 台账/初判 |
|---|---|---|---|
| `1831649` | C1/C5 kernel panic 降频 | conflict | 台账无记录, 高价值, 重点核实 |
| `f3859bd` | C1/C5 同上 (同伴) | conflict | 同上 |
| `9d742d193` | C2 dflash 无限循环 (#131) | conflict | 台账无, flyto dflash Path A 相关, 核实 |
| `a3c249b7d` | C2/C4 死循环兜底 | conflict | 台账无, 核实 |
| `cb33a761f` | C2 #934 align forward | conflict | 依赖 mlx-vlm bump, 核实 |
| `318678019` | C2 #934 logits_processors | conflict | 核实 |
| `69becb31` | C2 #911 dflash-mlx 0.1.5.1 | conflict | 含子依赖升级, 评估 |
| `9aed907` | C3 #1550 MTP row-wise | clean apply | 台账无明确, 可能真缺, 高优 |
| `60c26b6` | C3 #1097 mtp 测量 | conflict | 核实 (878c8925 已引入) |
| `abaa478` | C4 #409 cache offset | conflict | 台账无, 核实 |
| `014b17f` | C4 #422 turboquant | conflict | TurboQuant 默认关时不触发, 低优 |
| `170cec9` | C4 #25 SSD 异步写 | conflict | 核实 (flyto 已有 paged SSD) |
| `e693921` | C4 #1413 SSD stale block | clean apply | 可能真缺, 核实 |
| `37c73a0` | C1 memory enforcer | conflict | 台账确认 memory_guard_tier 已引入等价 |
| `1efb140` | C6 性能回退调参 | clean apply | 台账有 acd0533 不含 relax, 可能真缺, 高优 |
| `fd10281` | C6 custom tier | clean apply | 核实 (台账 64bd2a2 含 custom tier) |

不要 pick (是 bug 起因 / 已废弃): `7af715320` (#435 panic 起因), `af97a0fb9` (跨线程
mx.eval 起因), `483f4430a` per-stream-lock libmlx (性能回退已删).

## 优先级建议 (给决策者)

1. **最高**: kernel panic 降频 (`1831649`+`f3859bd`) -- 头号生产痛点, flyto 疑似缺.
2. **高**: 性能回退调参 (`1efb140`+`fd10281`) -- 直接影响日常体感, 且和已有 memory_guard
   配套.
3. **高**: MTP #1550 (`9aed907`) -- flyto 生产跑 MTP, batch corruption 命中.
4. **中**: dflash 无限循环 (#131/#934 那组) -- flyto dflash 生产栈, 但要看 PR #29 是否已覆盖.
5. **中**: cache corruption (#409 等) -- 并发命中.
6. **运维 (非 cherry-pick)**: kernel panic 平台层部分, 配 MLX_MAX_OPS_PER_BUFFER 等环境变量
   + 限上下文 + 留内存余量.
