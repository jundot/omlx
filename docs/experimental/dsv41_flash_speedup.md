# DeepSeek V4.1 Flash prefill 提速：S1 宽步落地 + S2 稠密前缀裁决

> 承接 `dsv41-flash-prefill-speed-handoff.md`（2026-09-26）。执行日 2026-09-26，
> Studio M3 Ultra 512GB，DeepSeek-V4.1-Flash-oQ4e-mtp（vlm engine，非 NAX）。
> 代码：`main-retention` @ `838600e8`。

## S1 · 宽步 prefill —— ✅ 落地，实测远超预期

### 接线（照抄 GLM #3944 模式）
- `scheduler.py`：`_DSV41_WIDE_PREFILL_STEP = 8192`；
  `_detect_dsv41_wide_prefill_step()`（默认门控 = native `deepseek_v41_packed_attention`
  symbol + NAX + ≥64GB；`OMLX_DSV41_WIDE_STEP_FORCE=N` env 早期返回，供非 NAX 摸底）；
  `_base_prefill_step_size` 消费点（首 chunk 即宽，无 SSD n-gram gather 需错峰）；
  ArraysCache block 目标纳入 `_dsv41_wide_prefill_step`（否则 2048 block 把宽步钳回窄）。
- `admin/benchmark.py`：`[benchmark-scheduler-config]` 增报
  `qwen4_wide_step/glm53_wide_step/dsv41_wide_step` 三字段（A/B 臂锚定）。
- 内存 profile **无需新写**：handoff §3 "make_prefill_memory_profile 返回 None" 已过时，
  PR #3808（09-23）已加 `_DeepSeekV41PrefillMemoryProfile` 且 flat-overhead accounting 已含。

### Studio live A/B（20260926-102112，OFF→ON 各 2 轮中位数）
harness `~/ab_dsv41_wide.sh`（改自 ab_glm53_port.sh）；OFF=stock 2048，ON=FORCE=8192。

| pp | OFF tps | ON tps | Δ | 轮次区间 |
|---|---|---|---|---|
| 1024 | 498.0 | 519.8 | +4.4% | OFF[470,526] ON[516,523]（噪声重叠，不计）|
| 4096 | 658.0 | 746.7 | **+13.5%** | OFF≈[633,683] ON[738,756] 不重叠 ✅ |
| 8192 | 699.9 | 765.2 | **+9.3%** | OFF[686,714] ON[761,770] 不重叠 ✅ |

Engage 证据：ON 臂 `dsv41_wide_step=8192`、`effective_block_size=8192`、
`requested_step=8192` ×6；OFF 臂全 0。ON 臂全程零 throttle/eviction/MemoryExceeded。

> 预期 +2~5%（GLM 参照），实测 +9~13%——v41 收益 > GLM 的合理解释：40 层
> ratio-1/2 池 + hc_mult=4 使每 chunk 的 CPU enqueue 与核 launch 次数更多，
> 宽步摊薄的固定开销更大；且 chunk 数减半直接砍半 CED/压缩器边界 rem 处理。

### 边界摸底（16384）——❌ 弃，8192 定为最终宽度
`~/ab_dsv41_wide16.sh`（pp8192/16384，OFFx=stock 2048 vs ON=FORCE=16384，各2轮）：

| pp | OFFx tps | ON(16k) tps | Δ | 备注 |
|---|---|---|---|---|
| 8192 | 636.1 | 536.6 | **-15.6%** | 同样单 chunk 8191，慢 35% |
| 16384 | 714.8 | 778.2 | +8.9% | 单 chunk 16383；warm 轮 +15% |

**根因**：benchmark 自带 `speed_priority=True`（benchmark.py:674，admission 按
full step 计费）。FORCE=16384 对 8192 prompt 超额预留 ~2× transient →
驱逐驻留 → 同几何 chunk 反而慢。16384 只在 prompt≥16k 时不吃亏，收益（+9~15%）
不足以抵短 prompt 回归。**8192 为最终宽步**。

### 默认值决定 —— ✅ 默认开（native + ≥64GB，去掉 NAX 门控）
非 NAX M3 Ultra 实测 +9~13% 且零 guard 事件 ⇒ NAX 要求对 v41 无必要
（`7d9e1ab2`）。门控 = native `deepseek_v41_packed_attention` symbol + ≥64GB。
- `OMLX_DSV41_WIDE_STEP_FORCE`：未设=默认 8192；>0=强制宽度（摸底用）；**0=kill switch** 回 stock 2048。
- 默认门控 Studio 终验 A/B（OFF=FORCE:0 vs ON=默认）：见文末回填。

### 16384 摸底结果（追加区）
见上表；默认门控终验数据跑完回填。

## S2 · 稠密前缀旁路 —— ❌ 纸面推导判弃（不写代码）

实际几何（config.json）：40+3 层；ratios L0-1=0、L2-19=2、L20-39=1；
kv_src=[2,8,14,20]；idx_src=[2,8,14,20,24,28,32,36]；candidate_src=20
（top-2048 块 × block_size 8）；index_topk=512；window=128。

Indexer 成本分层（Attention.forward：非 idx_src 层复用 `shared["idx"]`，零成本）：
- L{2,8,14}（ratio 2，`packed_index_topk` 全扫）：行 pos 扫 ⌊pos/2⌋+1 键；
- L20（ratio 1，全扫）：行扫 pos+1 键；
- L{24,28,32,36}（候选打分 `packed_index_scores`）：每行恒定 16384 候选槽。

稠密前缀条件（可见 pool ≤ index_topk ⇒ 全选 ≡ arange）：
- ratio-2 层：pos ≤ 1022（前 1023 行）；ratio-1 层：pos ≤ 511。

pp8192 收益算账：全扫层省 1.6%/0.4%；候选层省 6.25%（512 行 ×16384 槽）；
indexer FLOPs 合计省 ≈34.4M/645M ≈ **5.3%**；indexer ≈30% attn FLOPs、
attn ≈25-40% chunk 时间 ⇒ 端到端 **≈0.4-0.7%**。
GLM 的 25-50% 来自 kpool 窗口几何把稠密边界推得很深；v41 的 ratio-1/2 池 +
top-512 让稠密前缀恰好落在最便宜的行上。**触"先量再写"闸门 → 弃。**

### S2 副产品线索（未动工，勿与 S1 混改）
pp8192 时 ratio-1 池 8192 条 = 1024 块 < candidate_topk_blocks=2048
⇒ L{24,28,32,36} 候选预筛是 **no-op**，白付 16384 槽/行（一半是 -1 padding）。
候选宽度 clamp 到实际池块数×8 可省这 4 层打分 ~50%（仅 <16k 上下文；
16k 时 2048 块恰好用满）。估算端到端 ~1-3%。落点：`deepseek_v41/language.py`
Indexer 调用点 + `kernels.py::packed_index_scores` grid。

## S3 · knobs 盘点
- `deepseek_v41_engram_ssd_offload=false`、`deepseek_v41_ced_prefill_enabled=false`
  （Studio 现值，per-model settings，admin API 可切）。
- affine8：v41 走 `routing.py`（非 v4 `switch_layers.py`），Studio 模型为 oQ4e
  （affine bits=4，门控 `bits in (2,3)` 不吃）——**确认吃不到，勿开**。

### CED prefill A/B —— 🎯 免费午餐，但属语义级开关，留给用户验收
`PUT /admin/api/models/{id}/settings {deepseek_v41_ced_prefill_enabled:true}` +
重载（宽步默认同时在位，`dsv41_wide_step=8192` 锚定确认），pp=[4096,8192]×2 轮：

| pp | 宽步-only 中位 | CED+宽步 轮次 | CED 中位 | CED vs 宽步 | CED vs stock |
|---|---|---|---|---|---|
| 4096 | 755.0 | 800.8 / 823.9 | 812.4 | **+7.6%** | **+23.6%** |
| 8192 | 773.1 | 894.6 / 908.4 | 901.5 | **+16.6%** | **+28.8%** |

分离干净（CED 每轮均高于宽步臂）。输出连贯性冒烟通过（中文问答正常）。
**CED 改变 prefill 语义**（decoder 半仅 SWA 尾注意力，trained-in 布局但
质量验收是产品决策）——测毕已还原 `false` 并重载出厂状态。
用户点头即可常驻：Studio pp8192 合计 +28.8%，远超 +15% 目标线。

## 测试与基线
- 新增 4 测（wide 首块×paged、native 门控、env force 非 NAX）+ 既有 2048 边界测试钉窄。
- 全量套件 20 failed / 15177 passed——20 条在干净树逐条复现（既有基线，含未跟踪
  mimo 试验文件），零新增。

## 复现命令
```bash
# Studio
ssh ailab@192.168.114.162 'bash ~/ab_dsv41_wide.sh'      # 8192 A/B
ssh ailab@192.168.114.162 'bash ~/ab_dsv41_wide16.sh'     # 16384 摸底
# 结果行 ~/.omlx/logs/server.log [benchmark-pp-result]；锚定 [benchmark-scheduler-config]
```

## 16384 摸底结果（追加区）
见"边界摸底"节表格。默认门控终验 A/B（20260926-112546，OFF=FORCE:0 vs ON=默认，各2轮）：

| pp | OFF tps | ON(默认) tps | Δ | 轮次区间 |
|---|---|---|---|---|
| 1024 | 498.8 | 520.5 | +4.4% | 区间重叠，噪声 |
| 4096 | 657.1 | 755.0 | **+14.9%** | OFF≤680.1 < ON≥753.7 不重叠 ✅ |
| 8192 | 699.9 | 773.1 | **+10.5%** | OFF≤714.8 < ON≥772.2 不重叠 ✅ |

Engage：ON 臂 `dsv41_wide_step=8192` + `requested_step=8192`×6；OFF 臂全 0。
零 throttle/MemoryExceeded。**S1 完结：Studio 默认宽步生效，pp8192 +10.5%。**
目标 +15%：宽步单杆 +10.5% 未达；叠加 S3 CED 免费午餐后 +28.8%（待用户质量验收）。

### S2 副产品线索 —— 撤回（已核实无浪费）
复核 `kernels.py::packed_index_topk`：`block_count = min(block_count,
ceil(width/block_size))`（L758）已把候选块数 clamp 到实际池块数，
pp8192 ratio-1 时候选宽度=1024×8=8192 而非 16384。**不存在半 padding 浪费，勿再动工。**
