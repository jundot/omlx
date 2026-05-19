# DFlash Path A 实施 spec

承接 `m2max:/tmp/dflash-integration-spike-report.md`（已持久化到 `~/.claude/projects/-home-admin-dev-llm/artifacts/dflash-integration-spike-report.md`）的"Path A — 2-4 day version"路线。

## 1. 目标

`DFlashEngine` 重构为"智能容器"：内部抱一份长寿 `VLMBatchedEngine`，两条 decode 路径共享同一份 `target_model` 权重，按 (KV 压力, 并发数) 在 DFlash decode 和 BG decode 之间路由。

**主要收益：**

- 干掉现有 `dflash_max_ctx` fallback 触发时的 2x 权重内存
- 高并发自然降级到 `BatchedEngine` 吞吐路径（不再被 `DFlashDraftModel` 并发非验证的安全约束卡死）
- 不动 `omlx/scheduler.py`、`omlx/engine/vlm.py`——所有改动收敛在 `omlx/engine/dflash.py` + 1 个新 wrapper

**明确不做：**

- scheduler-level per-request routing（Path B 的范畴，需要 fork dflash_mlx 或维护 ~400 LOC 散在核心文件的 patch）
- 任何对 `dflash_mlx` 上游的修改

## 2. 必做 scope（A.0 + A.1）

### A.0 — 引擎重构 + 权重共享

**`omlx/engine/dflash.py` 重构** (~250 LOC delta)：

```python
class DFlashEngine:
    def start(self):
        # 1. 急切构造内嵌 BG 引擎（不再 lazy）
        self._embedded_vlm = VLMBatchedEngine(self._model_name, ...)
        self._embedded_vlm.start()

        # 2. wrapper 把 mlx_vlm 形态包成 mlx_lm 形态
        wrapped_target = DFlashVLMTargetWrapper(self._embedded_vlm._vlm_model)

        # 3. 跑 dflash_mlx 加载流程，但传入已 load 的 target
        self._bundle = attach_dflash_to_loaded_target(
            target_model=wrapped_target,
            draft_path=self._dflash_draft_path,
            draft_quant=self._dflash_draft_quant,
            runtime_context=self._runtime_context,
        )

    def generate(self, request):
        route = self._route(request)
        if route == "dflash":
            return self._dflash_generate(request)
        else:
            return self._embedded_vlm.generate(request)
```

**`omlx/speculative/dflash_vlm_target_wrap.py`** (~120 LOC，新文件)：

非破坏性 `LangModelView` proxy（不修改 mlx_vlm model 本身），暴露 `Gemma4TargetOps.supports_model` 期望的 mlx_lm 表面：

- `wrapped.language_model.args` → 代理到 `vlm_model.language_model.config`
- `inner._get_per_layer_inputs` → `inner.get_per_layer_inputs`
- `inner._project_per_layer_inputs` → `inner.project_per_layer_inputs`
- `final_logit_softcapping`、`tie_word_embeddings` 等属性桥接
- 通过 `__getattr__` 兜底其余 passthrough

**`omlx/speculative/dflash_factory.py`** (~80 LOC，新文件)：

```python
def attach_dflash_to_loaded_target(target_model, draft_path, draft_quant, runtime_context):
    """复用已 load 的 target，只跑 dflash_mlx 的 draft + bind 流程，不重复 load target。"""
    target_ops = resolve_target_ops(target_model)
    target_ops.install_speculative_hooks(target_model)
    draft_model, draft_meta = load_draft_bundle(draft_path, lazy=True, draft_quant=draft_quant)
    draft_backend = make_draft_backend()
    bind_draft_to_target(draft_model, target_model, target_ops=target_ops)
    return AttachedDFlashBundle(...)
```

**`omlx/engine_pool.py`** (~30 LOC delta，**风格 B：让 admin 可 tune**)：

```python
# engine_pool.py:637 附近的 DFlashEngine 实例化点
engine = DFlashEngine(
    model_name,
    dflash_draft_path,
    # ... 原有参数 ...
    # 新增（都带默认值，老 caller 兼容）：
    dflash_max_concurrent=getattr(model_settings, "dflash_max_concurrent", 1),
    dflash_kv_pressure_threshold=getattr(model_settings, "dflash_kv_pressure_threshold", 0.7),
    # dflash_max_ctx 已存在，保留语义不变作为 optional 硬上限
)
```

兼容性分析（已 grep 验证）：

- 整个 omlx 仓**只有 engine_pool.py:637 一处生产实例化 DFlashEngine**
- `engine_pool.py:702` 用 `type(engine).__name__ == "DFlashEngine"` 字符串判断，不依赖 ctor 签名
- 测试在 `tests/test_dflash_engine.py` 11 处实例化，由于新参数都带默认值不会 break

**结论：选风格 B（admin 可 tune 配置）成本 5-10 行，价值远大于风格 A（DFlashEngine 内部写死默认）。**

### A.1 — 自适应路由 + metric 埋点

**路由判据**（替代固定 `dflash_max_ctx`）：

```python
def _route(self, request) -> Literal["dflash", "bg"]:
    scheduler = self._embedded_vlm._engine.engine.scheduler
    active = len(scheduler._active_requests)
    kv_usage = scheduler._paged_kv_cache.usage_ratio   # 0.0 - 1.0

    # 决策（阈值在配置里，初始默认值见 §4）
    if active >= self._max_dflash_concurrent:
        return "bg"
    if kv_usage > self._kv_pressure_threshold:
        return "bg"
    return "dflash"
```

**新配置项（model_settings 上）：**

- `dflash_max_concurrent: int = 1`（DFlashDraftModel 并发非验证，默认 1；report §8）
- `dflash_kv_pressure_threshold: float = 0.7`（KV cache 用了 70% 就让位给 BG；初值，bench 后调）
- `dflash_max_ctx: int | None = None`（**保留兼容**，非 None 时作为硬上限叠加在 kv_pressure 之外）

**Metric 埋点**（每次 `_route()` 调用记录一行）：

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| ts | float | epoch seconds |
| request_id | str | 关联 scheduler 内的 request |
| ctx_len | int | prompt token 数 |
| active_count | int | 路由决策瞬间的 active 请求数 |
| kv_usage_ratio | float | PagedKVCache 占用比 |
| projected_kv_after | float | 假设接纳本请求后预计占用比（A.2 用得到，先采上） |
| routed_to | str | "dflash" or "bg" |
| reason | str | "kv_pressure" / "concurrency" / "default" |

落盘：`~/.omlx/metrics/dflash_routing.jsonl`（append-only，每行一个 JSON）。

**Size guard（day-1 内置，防忘）：** 写入前 stat 文件大小，超过 `DFLASH_METRIC_MAX_SIZE`（默认 500MB）就停写并打 warning：

```python
MAX_METRIC_SIZE = int(os.environ.get("DFLASH_METRIC_MAX_SIZE", 500 * 1024 * 1024))
if metric_path.exists() and metric_path.stat().st_size > MAX_METRIC_SIZE:
    if not _metric_size_warned:
        logger.warning(
            "dflash_routing.jsonl exceeded %d bytes — STOPPED writing metrics. "
            "Rotate manually (rm/archive the file) or implement rotate before prod.",
            MAX_METRIC_SIZE,
        )
        _metric_size_warned = True
    return  # silent skip
```

不做精细 rotate（开发期不值得），但 500MB 是硬上限——真到了就 server 日志里大字 warning 不能忽略。生产化 rotate 列在 §11 production hardening checklist。

读取：bench 脚本读这个 jsonl 出散点图（ctx_len × active_count，颜色按 routed_to）。

## 3. 文件清单

| 文件 | 状态 | LOC 估值 |
|---|---|---|
| `omlx/engine/dflash.py` | 重构 | ~250 delta |
| `omlx/speculative/dflash_vlm_target_wrap.py` | **D1 已完工** | 实际 **140 行**（~60 LOC executable，剩 docstring） |
| `omlx/speculative/dflash_factory.py` | 新 | ~80 |
| `omlx/engine_pool.py` | 改 | ~30 delta |
| `omlx/metrics/dflash_routing.py`（新 helper，可选） | 新 | ~40 |
| **A.1 路由判据 + metric 埋点（包含在上面 dflash.py 里）** | — | +50 |

**合计：A.0 ~ 410 LOC，A.1 +50 LOC，总 ~460 LOC**

（spike 验证发现 wrapper 比原估简单很多——DecoderLayer/make_cache/install_verify_linears 都不需要桥接，wrapper 只需绕过 `supports_model` predicate + 1-2 个属性别名。仍在 2-4 天 scope 内。）

## 4. 初始阈值默认值

第一次跑用保守值，bench 后调：

| 配置 | 初值 | 调整方向 |
|---|---|---|
| `dflash_max_concurrent` | 1 | 上游若验证并发安全，可放宽到 2-4 |
| `dflash_kv_pressure_threshold` | 0.7 | 看 bench 散点图：DFlash 在多大 kv_usage 下还能稳定 accept rate |
| `dflash_max_ctx` | None（不用） | 如果发现 kv_pressure 信号不够，再 fallback 到固定 ctx 上限 |

## 5. A.2 触发条件（延后做）

**触发标志：** A.0 + A.1 跑完 bench，散点图显示**KV pressure 不能可靠预测 DFlash speculative cache 即将爆**——具体表现：

- `kv_usage_ratio < 0.7` 但 DFlash decode 出现 OOM 或 cache overflow（speculative_linear_cache 比普通 cache 多占一截，pressure ratio 没反映出来）
- 或：在 `kv_usage_ratio 0.5-0.7` 这个中段区间，accept rate 出现非单调下降（暗示 cache 内部结构压力，不只是占用率）

**A.2 实施 checklist**（~50-80 LOC delta + scheduler.py 一个新 property ~10 LOC）：

- [ ] `scheduler.py` 加 `@property speculative_cache_pressure: float`（暴露 DFlash 视角的 cache 余量预测，纯读 PagedKVCache 内部状态计算）
- [ ] `dflash.py:_route()` 路由判据加入 `projected_kv_after` 维度：用 active 请求各自的 `remaining_budget` 总和估算决策后的占用比
- [ ] metric 埋点已经在 A.1 阶段采了 `projected_kv_after`，A.2 只是把这个字段从"记录"升级到"参与决策"
- [ ] 阈值默认 0.8（projected 比 instantaneous 紧一档）

**不触发就别做** —— A.0 + A.1 已经覆盖 90% 场景。

## 6. Spike 前置验证结果（已完成 2026-05-11）

3 个 spike 都在 m2max `~/Code/omlx/tmp_spike/` 跑过（gemma4-e2b q4-affine 模型），**全部 PASS**。spike 脚本起点：`~/.claude/projects/-home-admin-dev-llm/artifacts/dflash-integration-spike.py`，扩展版在 m2max 上。

### 验证点 1：mlx_vlm DecoderLayer 调用签名是否跟 mlx_lm 兼容 ✅ PASS

**结果：** `mlx_vlm.models.gemma4.language.DecoderLayer.__call__` 跟 `mlx_lm.models.gemma4_text.DecoderLayer.__call__` 参数列表**完全一致**：

```
(self, x, mask=None, cache=None, per_layer_input=None, shared_kv=None, offset=None)
```

`Gemma4TargetOps.forward_with_hidden_capture` 传的 3 个 kwarg 直接 verbatim 接受。**wrapper 不需要桥接 DecoderLayer。**

### 验证点 2：install_verify_linears 不误伤 vision projector ✅ PASS

**结果：**

- gemma4-e2b q4-affine 模型有 247 个 `nn.QuantizedLinear`（245 在 language_model、1 在 vision projector `embed_vision`、1 在 audio_tower）+ 1 个 plain `nn.Linear`（`audio_tower.output_proj`）
- `install_verify_linears(model, enable_qmm=True)` 替换了 317 modules，**只动 `QuantizedLinear` 不动 plain `nn.Linear`**
- 图像 forward 对照：last-token argmax 2094 → 2094（一致），max logit diff = **0.0**

**结论：** wrapper **不需要 vision projector 白名单 exclude**。`install_verify_linears` 默认行为是安全的。

**Caveat：** 上述结论是 q4-affine build 路径。如果未来用 bf16 build 且 vision projector 是 plain `nn.Linear`，install 也会跳过（因为只动 Quantized 那条）——同样安全。

### 验证点 3：mlx_vlm Gemma4 SWA cache 跟 Gemma4TargetOps.make_cache 兼容 ✅ PASS

**结果：** 两条路径返回**完全一致**的 15-layer cache 列表：

- 12 × `mlx_lm.models.cache.RotatingKVCache`
- 3 × `mlx_lm.models.cache.KVCache` 在 layers {4, 9, 14}（Gemma 4 的 SWA + full-attn split）

**关键发现：** `Gemma4TargetOps.make_cache` 对 Gemma4 **忽略 `enable_speculative_linear_cache` 参数**（target_gemma4.py:169-186 里 delegate 到 `wrapper.make_cache()`）。所以两条路径其实是同一个 call。

**结论：** wrapper 可以直接用 `LanguageModel.make_cache()` 或 `Gemma4TargetOps.make_cache(...)`，等价。**不需要 cache 桥接。**

### Side finding：supports_model() 在 wrapper 装上后自然返 True，无需任何绕过

最初 spike（裸 mlx_vlm Model）`supports_model` 返 False。但 **D1 wrapper 实施后实测**：

> `Gemma4TargetOps.supports_model(wrapped)` 返 **True**

原因：predicate 实际 walk 是 `text_wrapper → args.layer_types`，而 wrapper 的 `_LangModelView` 正好暴露 `args` 属性。**所以**：

- 不需要绕过 dispatch
- 不需要 monkey-patch predicate
- 直接 `resolve_target_ops(wrapped)` 或 `Gemma4TargetOps()` 都能工作

### D1 wrapper 实施时发现的唯一真实 API 分歧

`get_per_layer_inputs` 不是简单 rename，是**签名分歧**：

| 调用方 | 签名 |
|---|---|
| mlx_vlm `inner.get_per_layer_inputs` | `(input_ids)` — 1 参数 |
| dflash 调 `inner._get_per_layer_inputs` | `(input_ids, input_embeddings)` — 2 参数 |

wrapper 处理：写真适配方法。`input_ids is not None` 时 drop 第二个参数；`input_ids is None` 走 embeddings-only fallback 时 `raise NotImplementedError`（mlx_lm 那条 fallback 用 nearest-vocab 重构，mlx_vlm 没等价路径）。

**Path A 影响：** 0。Path A 是 text-only target，不会触发 embeddings-only fallback。但 multimodal 场景如果将来要复用这个 wrapper，需要补这条 fallback 路径。已在 wrapper docstring 标注。

## 7. 验收

A.0 + A.1 完工标准：

- [x] m5max 端到端跑通 dflash 路径（spike7 PASS）：
  - 短 ctx → routing = `dflash` ✅
  - max_ctx 触发 → routing = `bg` ✅
  - 并发 ≥ max_concurrent → routing = `bg`（reason=`concurrency`）✅
- [x] 权重共享 ID 级验证：`engine._target_model._vlm is engine._embedded_vlm._vlm_model = True` ✅
- [x] `~/.omlx/metrics/dflash_routing.jsonl` 字段完整（spike6 4 条 + spike7 几条）✅
- ~~[ ] BFCL tool-calling 测试矩阵~~ ⛔ 用户取消 2026-05-12
- [x] decode throughput vs pure VLM batched ≥ +10% — **实测 +10.9%**（2026-05-12 17:00 m5max bench）
  - DFlash ON: 93.97 tok/s（5 次平均，stddev 1.7）
  - DFlash OFF: 84.76 tok/s（5 次平均，stddev 4.0）
  - 同 prompt × temperature=0.0 × max_tokens=150 deterministic 对照
  - Note: 这是单条 sequential 请求场景；并发场景下 dflash_max_concurrent=4 行为待 bench

## 8. 实施顺序

1. ~~**D1 上午**：3 个 spike 验证点~~ ✅ 已完成 2026-05-11
2. ~~**D1 下午**：写 `dflash_vlm_target_wrap.py`~~ ✅ D1 已完工
3. ~~**D2**：核心架构重构~~ ✅ 已完成 2026-05-11
   - dflash.py 重构完成（+211 -182）
   - dflash_factory.py 新建（203 LOC）
   - engine_pool.py:637 加 kwargs（+10）
   - smoke 5/5 PASS（架构骨架 + 权重共享 ID 级验证）
   - **未验证**：factory 函数体（smoke 桩掉了，需 D3 用真 drafter 端到端跑过）
   - **未验证**：install_verify_linears 后 BG forward 输出不变（理论 transparent）
   - **小 bug**：engine_pool 用 `... or 1` 把 0 静默改 1，留 D3 修
4. ~~**D3**：测试 + engine_pool 修 + metric + KV pressure 信号~~ ✅ 完成 2026-05-12
   - 测试：rewrote 2 个 `_should_fallback` test 调 `_route()`，pytest **34/34 PASS**
   - engine_pool 两处修完：`or 1` collapse-on-0 fix + :702 鸭子判断
   - `omlx/metrics/dflash_routing.py` 90 LOC（含 size guard + env disable）
   - `omlx/engine/dflash.py`：`_kv_pressure()` 多 attr 兜底 + 三路 `_route()`（concurrency / KV pressure / max_ctx）
   - **spike6 端到端 6/6 PASS** on m5max:
     - factory body 真跑：`Gemma4TargetOps` + `DFlashDraftModel` 真实例化
     - 权重共享 ID 级验证：`engine._target_model._vlm is engine._embedded_vlm._vlm_model = True`
     - 三种 routing reason 都触发（default / max_ctx / concurrency）
     - `_kv_pressure()` 返 None（attr 兜底路径）——routing 仍正常工作
     - metric jsonl 4 行写入字段完整
   - git status 两台都只在 7 文件清单内，无 scope 溢出
5. ~~**D3 已知问题 / D4 followup**~~ → D4 阶段已解决：
   - ✅ **上游 tokenizer 兼容 bug 解决**：在 `omlx/speculative/__init__.py` 加 import 时 monkey-patch `dflash_mlx.runtime.get_stop_token_ids`，coerce `eos_token_ids: int` → `[int]`。idempotent，不动 .venv 内文件
   - ✅ **fallback_engine_type 默认 "batched" → "auto"**：新增 `_resolve_fallback_engine_type` 静态方法 + `omlx/speculative/__init__.py:detect_fallback_engine_type`，通过 `config.json` 里的 `vision_config` / `audio_config` 判别（canonical 标记，比 processor_config.json 可靠——qwen-dense-9b 这种多模态模型也正确识别为 vlm）
   - ✅ **`_kv_pressure()` 返 None 不是 bug**：scheduler 的 `block_aware_cache` / `get_cache_stats()` 是**懒初始化**，必须有请求触发才有 cache。real bench 时会自然有数据。当前的 None fallback 是正确的安全行为
6. **D4 spike7 全 PASS**（spike6 之后追加的 full-generate 验证）：
   - 默认 ctor（不传 fallback_engine_type）→ auto-detect = vlm ✅
   - 短 prompt → real dflash decode 出 token：`output text="HelloHelloHello..."`（11 tokens generated）✅
   - 长 prompt → bg fallback ✅
   - stop 干净 ✅
   - dflash prefix cache 起来了（log 确认 entries=0/4, end-of-request snapshot saved）
7. ~~**D4 部署 + smoke**~~ ✅ 完成 2026-05-12 16:48
   - m5max:8000 server 切到 D4 代码（PID 91595 → 93457）
   - rollback artifacts：`~/omlx-d4-rollback.txt` + `~/omlx-d4-snapshot-2026-05-12.diff`（793 行 git diff）
   - 27 model discover 干净，无 error/warn（除无关 mel filter）
   - 三 smoke 全过：gemma4-e2b（vlm）/ gemma4-moe-26b-a4b（reasoning vlm）/ smollm3-3b（batched）
   - metric jsonl 没 server 触发写入（生产配置都是 `dflash_enabled: False`，正确）
8. ~~**真生产打开 dflash 验证**~~ ✅ 完成 2026-05-12 17:00
   - 改 `~/.omlx/model_settings.json`：`gemma4-moe-26b-a4b.dflash_enabled: True`，`vlm_mtp_enabled: False`（互斥），`dflash_kv_pressure_threshold: 0.999`
   - 重启 server，DFlashEngine 真实例化（log 确认 "DFlashEngine loaded (Path A double-engine)..."）
   - dflash_factory 实跑（"installed verify_linear on 236 QuantizedLinear modules of Model"）
   - 真请求路由到 dflash（metric: `routed_to=dflash, reason=default`），HTTP 200 / 4.1s 响应
   - 备份在 `m5max:~/.omlx/model_settings.json.bak-2026-05-12`
9. ~~**D5 followup**~~ 已收 2026-05-13
   - ✅ **D3 漏修的 bug 已补**：`ModelSettings` 加 `dflash_kv_pressure_threshold` 字段
   - ✅ **kv_usage_ratio=0.9974 谜值**：根因是 omlx `PagedCacheManager.usage` property 公式语义错——分子是 `free_block_queue.num_free_blocks`（**bounded free queue size**，~256 上限），不是真实未分配块数。修 `_kv_pressure()` 用 `allocated_blocks_len / max_blocks` 替代，实测 0.997 → 0.00256（cache 几乎空，正确反映）
   - ✅ **真生产部署 + smoke**：m5max:8000 切到 D4 代码，三引擎类型（vlm/batched/reasoning）smoke 全过
10. ~~**D5 性能验收**~~ 完成 2026-05-13
    - **单条 sequential**：DFlash ON 93.97 tok/s / OFF 84.76 tok/s → **+10.9%** ✅
    - **4 并发 (eager, mc=1)**：DFlash ON 115 / OFF 180 → -36% ❌
    - **诊断 isolate**：drafter co-loaded 但不参与（mc=0 eager）= 129 tok/s（-28% vs OFF）。**Drafter 占着 Metal 内存 = 大锅，install_verify_linears 不是锅（hypothesis A 否定，bench 113 vs 115 同档）**
11. **D5 lazy_drafter 实施 + 验收** ✅ 完成 2026-05-13
    - **新增配置** `dflash_lazy_drafter: bool = False`（model_settings + DFlashEngine ctor + engine_pool 传递）
    - **`start()` 重构**：抽出 `_load_drafter_bundle()` helper，lazy 模式跳过；eager 模式行为不变
    - **新增 `_ensure_drafter_loaded()`**：用 asyncio.Lock + double-check 保证 race-free；generate/stream_generate dflash 分支前调用
    - **bench 验证**：lazy + mc=0 + 4 并发 = **165.42 tok/s**（vs eager 129，**回收 70% 损失**，drafter 永不加载时 Metal contention 完全消除）
    - 剩 8% 跟 OFF baseline 差距：来自 DFlashEngine wrapper 自身（VLMBatchedEngine 嵌套等），量级小不阻 ship
12. **Path A 最终特性矩阵**（按 workload 选配）：

| Workload | 推荐配置 | 单条 latency | 并发 throughput |
|---|---|---|---|
| 单用户 / dev workflow（你目前主要场景）| `enabled=True, mc=1, lazy=False`（spec default） | +10.9% | -36% |
| 多人 server / 高并发 | `enabled=True, mc=0, lazy=True` | (不走 dflash) | **-8%**（仅 wrapper 开销）|
| 不要 dflash | `enabled=False` | baseline | baseline |

m5max:8000 当前配置：spec default（dev workflow 场景）。生产化时按上表换配置。
6. **D4**：bench 跑全 alias 矩阵，对比 baseline；写 result 报告

**注：** D2/D3 拆分是为了让 D2 完工后有清晰的"核心架构能跑"checkpoint，验证完才推 D3 的覆盖性工作。D2 失败不影响 D3 计划——可以回退。

## 9. 风险 + 回退

**风险点：**

- ~~spike 验证点 1-3 任一失败~~ ✅ 已 PASS
- `_active_requests` / `_paged_kv_cache.usage_ratio` 这些 scheduler 私有属性 upstream 可能换名 → wrapper 用 try/except 兜底 + 单元测试
- DFlash speculative cache 跟 BG PagedKVCache 在内存预算分配上可能打架（两条路径都吃同一个 mem pool）→ 监控 `mlx.core.metal.get_active_memory()` 在切换瞬间的变化
- **`_in_fallback_mode` 单向闸门移除的语义变更**：现状代码逻辑"翻转一次就永远 fallback"被 Path A 改成"每请求重新决策"。需要确保现有靠这个语义的代码路径（observability/log/stats）正确处理"engine 又回到 dflash 模式"这种以前不会发生的情况。Reader 已确认无外部弱引用、无 atexit hook 依赖，但 `get_stats()` 的字段语义要重新定义

**回退：** Path A 重构后旧 `dflash_max_ctx` 行为可以由 `dflash_kv_pressure_threshold=1.0`（永不切 BG）+ `dflash_max_ctx=<original_value>` 模拟，相当于完全退化到 A.0 之前的硬阈值。这条 fallback 路径保留至少一个 release 周期。

## 10. 跟 CCM 的衔接

按 CLAUDE.md "本仓 latest 选型结论会 commit 到 CCM 决策档"：

- A.0 + A.1 落地并跑通 bench 后，把"Gemma 4 dflash 路径在 oMLX 上可用"的结论 commit 到 CCM 决策档
- bench 散点图 + 阈值 tuning 结果一起进 CCM

## 11. Production hardening checklist（上线前必做）

Path A 的开发期实现里以下几项**故意延后**，理由：开发期不值得做、做了反而限制 bench 收集数据。**部署到正式环境之前必须逐项消化。** 这一节就是为了"将来忘了"准备的——开 PR 时把这一节链接到 PR description，谁要 deploy 必须先 review 这条 checklist。

### 11.1 Metric jsonl 自动 rotate

- 现状：500MB size guard 触发后**直接停写**，靠人手清。
- 上线前要做：换成基于 `logging.handlers.RotatingFileHandler` 或自实现的滚动写（按大小 100MB × 5 份，或按天）。
- 决策点：开发期 bench 数据可能要保留很久供分析，所以**默认 rotate 关掉**，admin 设 `OMLX_DFLASH_METRIC_ROTATE=size:100M:5` 才打开。
- 实施位置：`omlx/metrics/dflash_routing.py`（如果建了这个 helper）或 `dflash.py` 内 metric 写入函数。

### 11.2 Server launchd 自启 ✅ 已通过 GUI bundle hack 解决（2026-05-13）

- **现状已变**：m2max:8000 和 m5max:8000 都通过 `/Applications/oMLX.app` 的 GUI menubar 启动，GUI 自身有 launchd plist (`~/Library/LaunchAgents/com.omlx.app.plist`)，**重启后 GUI 自启 → spawn server subprocess → 走我们 fork venv**
- 实施细节见 `docs/omlx-fork-gui-bundle-hack.md`
- 旧的手动 nohup m2:8002 + m5:8000 都已 kill
- **不再需要**单独写 omlx server 的 launchd plist；本节保留为历史决策记录

### 11.3 Path A 路由阈值 tune 后回写默认值

- 现状：`dflash_max_concurrent=1`、`dflash_kv_pressure_threshold=0.7` 是保守初值。
- 上线前要做：bench 完拿到散点图，把"在我们 workload 上明显最优"的阈值回写到 `DFlashEngine.__init__` 默认值，并 commit。
- 决策点：阈值跟模型 + 硬件强相关。Mac (M2/M5) 的最优值跟将来云 GPU 的最优值可能完全不同——所以默认值应该匹配**最常见的部署目标**（先 Mac，因为它就是 dev 主战场），其他场景由 model_settings 显式覆盖。

### 11.4 DFlash speculative cache vs PagedKVCache 内存预算冲突

- 现状：两条路径都吃同一个 mlx Metal mem pool，没有显式预算分隔。
- 上线前要做：在 `DFlashEngine.start()` 里加一次 `mlx.core.metal.get_active_memory()` 基线采样 + 每次 routing 决策时打 metric（`metal_active_mb` 字段加进 jsonl）。bench 完看是否需要为 DFlash speculative cache 预留固定内存预算。
- 决策点：等 bench 数据，可能不需要任何特殊处理。

### 11.5 测试覆盖率

- 现状：`tests/test_dflash_engine.py` 11 处实例化测试，覆盖原有 DFlashEngine 行为。
- 上线前要做：
  - [ ] 全部 11 处 test 在 Path A 重构后仍 pass（回归测试）
  - [ ] 新增至少 4 个 case：embedded VLM 加载成功 / KV pressure 触发 BG / 并发触发 BG / metric 写入正确
  - [ ] 新增 1 个端到端 case：用真实 gemma4-e2b 模型跑短/长 ctx，validate 内存峰值不超过 baseline 1.1x（验证权重共享生效）
