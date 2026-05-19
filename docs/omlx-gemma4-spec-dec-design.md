# oMLX 集成 Gemma 4 speculative decoding 设计文档

**Status**: Draft (2026-05-10), 待 user 回来 review
**Branch**: `feat/gemma4-spec-dec` (待创建)
**Repo**: panwudi/omlx fork (m2max:~/Code/omlx)

## 1. 背景与动机

### 1.1 Gemma 4 26B-A4B 是 Mac 本地部署 mid-tier 真甜蜜点
- 已实测 Q4 单流 **94 tok/s**（3x 快于 Qwen 27B Dense 的 33 tok/s）
- BF16 单流 **~50 tok/s**（vs Qwen 27B Dense BF16 ~5 tok/s）
- 质量 4/4 paradox 通过率（BF16）
- 多模态原生（VLM engine）

### 1.2 mlx-vlm 上游已有 Gemma 4 spec dec 完整支持
- model class: `mlx_vlm/models/gemma4/language.py` 已有 `capture_layer_ids` + `rollback_speculative_cache` hook
- drafter: `mlx_vlm/speculative/drafters/gemma4_assistant/` 独立 drafter class（separate checkpoint，非 native MTP weights）
- CLI runner: `mlx_vlm/generate.py:219 --draft-model` flag 已暴露
- drafter kind 注册: `DRAFTER_KIND_BY_MODEL_TYPE = {"gemma4_assistant": "mtp"}` ("mtp" 在此为 round-loop 算法分类，非 weights pattern)

### 1.3 oMLX 现状缺这条路径
oMLX 现有 spec dec 路径仅 LM-only：

| 路径 | 触发 | engine 类 | 适用模型 |
|---|---|---|---|
| Native MTP | `mtp_enabled=True` + mtp.* weights in checkpoint | BatchedEngine (forced) | Qwen 3.5 PR990, DS V4 |
| DFlash | `dflash_enabled=True` + draft checkpoint | DFlashEngine | Qwen3.5 DFlash |
| SpecPrefill | `specprefill_enabled=True` (prefill only) | VLMBatchedEngine/BatchedEngine | 任意 |

`omlx/engine_pool.py:611-624` 显式规定：mtp_enabled + VLM 时**强制 LM-only**（vision 丢弃），因为 mlx_lm_mtp.sanitize patch 只在 LM 路径存在。

### 1.4 期望加速（已对照 bare 实测，结论翻转）

**Paper claim**: 1.5x（mlx-vlm 文档）

**实测 (2026-05-10 m5max, mlx-vlm 0.5.0 server with gemma4_assistant drafter on Q4 target, 3 trials avg)**:

| Setup | tok/s | vs bare |
|---|---|---|
| oMLX Q4 bare (no spec dec) | **107** | baseline |
| mlx-vlm Q4 **bare** (no drafter) | **110** | 等价（修正之前 28% 误判）|
| mlx-vlm Q4 + spec dec, thinking OFF | **83** | **-25%** ⚠️ |
| mlx-vlm Q4 + spec dec, thinking ON | **92** | -16% |

**spec dec 当前 net negative**：accept rate 1.04-1.69 时 drafter overhead 比 token gain 大。

**advisor 抓的 catch**：accept=1.04 实际是 2.04 tok/round（accept+bonus），理论该提速 2x。但实测反而慢——drafter overhead + rollback 时间超过 token 增益。

### 1.5 ROI 最终结论（已实测，不是推测）

**oMLX 集成 spec dec 当前 default config 下不值得做**：
- 集成后跑同 drafter 同算法，大概率同样 net negative
- 11h dev + 维护成本 投入产出比惨

**生产替代方案**：**Q6 + thinking OFF**（user 实测 4/4 paradox 不靠 thinking）。

wall-clock per-task 对比：

| 路径 | tok/s | thinking | 每任务输出 tok | wall-clock 估 |
|---|---|---|---|---|
| Q4 + thinking ON | 107 | ✅ | ~2000 | ~19 s |
| **Q6 + thinking OFF** | **89** | ❌ | ~800 | **~9 s** ⭐ |
| BF16 + thinking ON | ~30 | ✅ | ~2000 | ~67 s |

**Q6 是真王者**（per-token 慢 17% 但 per-task 快 2x）。spec dec 不需要。

## 2. 架构方案

### 2.1 决策：新 engine 路径 (路线 B)，不复用 mtp_enabled

**否决路线 A（复用 mtp_enabled）的理由**：

1. oMLX `mtp_enabled` 当前语义 = "main checkpoint 含 mtp.* weights，走 sanitize patch + mlx_lm 路径"。Gemma 4 不符合（独立 drafter checkpoint）。
2. `engine_pool.py:614-624` 已硬编码：`mtp_enabled + vlm` 强制 LM-only。改这个会破坏 Qwen 3.5 native MTP 现有行为。
3. mlx-vlm 内部用 "mtp" 是 round-loop 算法名，跟 oMLX `mtp_enabled` 概念混淆。

**采用路线 B（新 vlm_drafter 配置）**：

新增配置字段（`model_settings.py`）：
```python
vlm_drafter_enabled: bool = False
vlm_drafter_model: Optional[str] = None      # HF repo 或本地路径
vlm_drafter_quant_bits: Optional[int] = None # 4 / 8 / None=bf16
vlm_drafter_kind: Optional[str] = None       # 默认 auto-detect from drafter HF model_type
```

新增 engine class（或扩展 VLMBatchedEngine 中加 drafter 路径）：
```python
omlx/engine/vlm_drafter.py
  class VLMDrafterEngine(VLMBatchedEngine):
      def __init__(self, ..., drafter_model_path, drafter_kind, ...):
          # load via mlx_vlm.speculative.drafters.load_drafter
      def _step(self, ...):
          # 用 drafter 提议 N tokens → target 一次 forward → accept until mismatch
          # 调 mlx_vlm 的 _speculative_walk + rollback_speculative_cache
```

`engine_pool.py` dispatch 加分支：
```python
# ~line 605-700 区间
if model_settings is not None and getattr(model_settings, "vlm_drafter_enabled", False):
    drafter_path = getattr(model_settings, "vlm_drafter_model", None)
    if drafter_path and effective_type == "vlm":
        from .engine.vlm_drafter import VLMDrafterEngine
        engine = VLMDrafterEngine(
            model_name=entry.model_path,
            drafter_model_path=drafter_path,
            drafter_kind=getattr(model_settings, "vlm_drafter_kind", None),
            drafter_quant_bits=getattr(model_settings, "vlm_drafter_quant_bits", None),
            ...
        )
```

### 2.2 互斥矩阵

`vlm_drafter_enabled` 跟其他 spec dec **互斥**（参考 `mtp_enabled` 跟 dflash/turboquant 的互斥规则在 `ModelSettings.__post_init__`）：

| | mtp | dflash | vlm_drafter | specprefill |
|---|---|---|---|---|
| mtp | - | ❌ | ❌ | ✅ |
| dflash | ❌ | - | ❌ | ✅ |
| vlm_drafter | ❌ | ❌ | - | ✅ |
| specprefill | ✅ | ✅ | ✅ | - |

(SpecPrefill 是 prefill-only 优化，正交于 decode 加速)

### 2.3 加载 drafter

走 mlx-vlm 上游 API：
```python
from mlx_vlm.speculative.drafters import load_drafter

drafter_model, resolved_kind = load_drafter(
    path_or_repo=drafter_model_path,
    kind=drafter_kind,  # None → auto-detect via HF model_type
)
# resolved_kind 是 "mtp" / "dflash"
```

oMLX 不需要重写 drafter 加载逻辑，复用上游 `load_drafter`。

### 2.4 Round-loop dispatch

VLMDrafterEngine 单步：
```python
# 1. drafter 提议 N tokens
draft_tokens, draft_cache_state = drafter.draft_block(
    prefix_cache=draft_cache,
    block_size=N,
)

# 2. target model forward 一次 (verify)
target_logits = self.model(
    input_ids=draft_tokens.reshape(1, -1),
    cache=target_caches,
    capture_layer_ids=[...],  # 给下一轮 drafter 用
)
target_tokens = sample(target_logits)

# 3. _speculative_walk (mlx-vlm 上游函数) — exact-greedy 验证
n_accepted = mlx_vlm.generate._speculative_walk(
    draft_tokens=draft_tokens,
    target_tokens=target_tokens,
)

# 4. rollback target cache (mlx-vlm 上游)
self.model.language_model.rollback_speculative_cache(
    caches=target_caches,
    gdn_states=None,  # gemma 4 不用
    accepted=n_accepted,
    block_size=N,
)

# 5. return n_accepted 个 tokens (continuous batching scheduler 接住)
```

## 3. 实现工作量

| Task | LOC | 时长 |
|---|---|---|
| `model_settings.py` 加 4 个字段 + 互斥检查 | ~30 | 30 min |
| `model_profiles.py` whitelist 字段 | ~10 | 10 min |
| `omlx/engine/vlm_drafter.py` 新 engine | ~250 | 4 h |
| `engine_pool.py` dispatch 分支 | ~20 | 30 min |
| `admin/static` + `admin/routes.py` UI 开关 | ~50 | 1 h |
| 测试 `tests/test_vlm_drafter.py` | ~150 | 2 h |
| 集成测试 (gemma 4 26B + e2b drafter) | - | 2 h |
| 文档 + README | ~80 | 1 h |
| **总计** | **~590 LOC** | **~11 h** |

之前 memory 写"~200 LOC wire 工作"**低估**了 — wire 本身 LOC 不多，但配置 schema、UI、互斥逻辑、测试加起来 600 LOC 左右。

## 4. 验证计划

### 4.1 单元测试
- drafter 加载: 给 `gemma-4-e2b-it` 当 drafter，target=`gemma-4-26b-a4b-it`，确认 `load_drafter` 返回 (model, "mtp")
- round-loop: 输入 fixture prompt → 比对 spec dec 输出 vs 直接 decode 输出（语义等价，不一定 token 级一致因为采样）
- cache rollback: 验证 target cache 长度跟 accepted token 数对齐

### 4.2 集成测试
- m2max 本地起 oMLX：`vlm_drafter_enabled=True` for `gemma4-moe-26b-a4b`
- 跑 paradox 4 格（洗车+棍子 × ZH/EN）验证质量 == 不 spec dec
- bench 单流速度，对比 baseline 94 tok/s

### 4.3 quality regression check
- 比对 spec dec ON/OFF 输出长度、reasoning trace 长度、最终 commit token 序列
- 不能因为 spec dec 让模型 commit decision drift（理论上 exact-greedy 不会，但实测验证）

## 5. 风险与未决问题

### 5.1 drafter checkpoint 可用性 ✅ 已验证
HuggingFace 上已发布完整 drafter 集（Google 官方 + mlx-community 量化版）：

| Target | Official | mlx-community |
|---|---|---|
| gemma-4-26b-a4b-it | `google/gemma-4-26B-A4B-it-assistant` | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` |
| gemma-4-31b-it | `google/gemma-4-31B-it-assistant` | `mlx-community/gemma-4-31B-it-assistant-bf16` |
| gemma-4-e4b-it | `google/gemma-4-E4B-it-assistant` | `mlx-community/gemma-4-E4B-it-assistant-bf16` |
| gemma-4-e2b-it | `google/gemma-4-E2B-it-assistant` | `mlx-community/gemma-4-E2B-it-assistant-bf16` |

注意 mlx-community 目前只有 `bf16` 版本，没有 Q4/Q8 量化版。drafter bf16 大小估 ~3-5 GB（不大）。

**Action**: 在集成测试阶段下载 `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` 配合 `gemma4-moe-26b-a4b` (Q4) target 测。

### 5.2 mlx-vlm pin 必须先 bump ⚠️ 关键前置
oMLX `pyproject.toml` 现 pin `mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@e41cd255` —— **51 commits 落后 main**。Gemma 4 spec dec 关键 PR 都在 ahead 区间：

| PR | commit | 关键性 |
|---|---|---|
| **#1112** | `244f4bb` | Add Gemma 4 MTP speculative-decoding drafter ⭐ 核心 |
| **#1115** | `0c2bbf5` | Server: add Gemma 4 MTP drafter support |
| #1110 | `d8f3a6c` | Fix TurboQuant batch cache offset merging |
| #1119 | `d6e2df8` | Fix Qwen3.5 quantization config keys |
| #1127 | `40aad1f` | Fix mixed-length Gemma 4 batching |
| #1129 | `da5dd8e` | Fix streamed detokenization for byte fallback |
| **#1140** | `aec6998` | Speculative decoding fixes: auto-detect drafter kind |

**Action 1: bump pin to `aec6998`** (#1140 or later) 在 pyproject.toml。
**Action 2: 跑全量 oMLX 测试** 看 51 commits 是否引入回归（Qwen 3.5 quantization config keys 改了，oMLX 自定义 quant 工具链可能要适配）。

bump 完后 mlx-vlm API 才有：
- `mlx_vlm.speculative.drafters.{KNOWN_DRAFTER_KINDS, load_drafter, DRAFTER_KIND_BY_MODEL_TYPE}`
- `LanguageModel.rollback_speculative_cache`

### 5.3 mtp_enabled VLM 强制 LM-only 的现状
- `engine_pool.py:611-624` 现在 Qwen 3.5 VLM + mtp_enabled 会强制 LM-only
- vlm_drafter_enabled 是另一套路径，不冲突
- **要不要清理 mtp_enabled+VLM 现状**: 暂不动（避免破坏 Qwen 3.5 现有行为）

### 5.4 batch 模式 spec dec accept rate 稀释
- 之前 session 实测：Qwen 3.5 MTP 单流 1.41x，batch 模式 ~1x（accept rate 被多请求冲走）
- VLM drafter spec dec 也会有相同问题
- **Action**: design doc 说明 vlm_drafter 主要 benefit 在单流 / 低并发场景

## 6. 落地步骤

1. **HF 上找 gemma4_assistant drafter checkpoint** (优先 mlx-community 量化版)
2. 创建 branch `feat/gemma4-spec-dec` from main
3. 实现 `model_settings.py` 字段 + 互斥
4. 实现 `omlx/engine/vlm_drafter.py`
5. 接 `engine_pool.py` dispatch
6. 单元测试
7. 集成测试 (m2max 真跑)
8. admin UI 开关
9. 文档
10. PR to panwudi/omlx → 拿到稳定 commit 后再 upstream PR to jundot/omlx

## 7. Open questions for user

- ❓ **大决策**：mlx-vlm 自己有 server (#1115)，能直接当 OpenAI 兼容 endpoint。要不要先评估 **直接用 mlx-vlm server** 替代部分 oMLX，省掉本仓所有 wire 工作？trade-off：放弃 oMLX 的 multi-model engine pool / admin UI / tiered cache / X-API-Key auth。**如果只要 Gemma 4 spec dec 一个 model，mlx-vlm server 可能就够**
- ❓ 倾向路线 A（复用 mtp_enabled）还是路线 B（新 vlm_drafter）？我倾向 B（理由见 2.1）
- ❓ 第一里程碑要不要先做 minimal viable (skip admin UI，pure config-file driven)？这样可以 4 小时跑通 prototype
- ❓ 拿到工作 prototype 后，PR upstream jundot/omlx 还是先 fork 长期 maintain？

## 8. Decision sequence (action order)

1. **Bump mlx-vlm pin** to `aec6998` (#1140) in `pyproject.toml`
2. **重装 omlx editable**：`uv pip install -e .` 在 m2max
3. **跑全量 omlx tests** 看回归：`pytest tests/ -x` —— 至少 mlx_lm_mtp 测试、Qwen 3.5 quant 测试需要绿
4. **如果 1-3 顺**：write VLMDrafterEngine 实现
5. **如果 1-3 出回归**：评估单独 fix vs 用 mlx-vlm server 替代路径
6. 集成 / bench / PR upstream

## 9. Status snapshot (2026-05-10)

- ✅ Fork cloned to m2max:~/Code/omlx
- ✅ Python 3.12 venv via uv 装好
- ✅ omlx editable install 完成 (mlx-vlm 0.4.5 装上但是 51 commits 落后)
- ✅ design doc 完成
- ✅ drafter HF 可用性 confirmed (mlx-community 4 sizes 全有 bf16 版)
- ✅ **mlx-vlm 0.5.0 server 实测** spec dec on m5max (port 8002, 已 kill)
- ✅ **alias 命名** `gemma-4-26b-a4b-it-6bit/8bit` → `gemma4-moe-26b-a4b-q6/q8` 已在 m5max rename
- ⚠️ **实测重估** accept rate 1.04-1.69（不是 paper 1.5x），推测增益 ~18% 而非 50%
- ⏸ 待 user 回来 review **是否还做 oMLX 集成**（ROI 边缘）+ 路线 A/B 选择

## 10. user 回来阅读建议（TL;DR 最终版）

**实测核心数据**：

| Setup | tok/s |
|---|---|
| oMLX Q4 bare | **107** |
| oMLX Q6 bare | **89** |
| mlx-vlm Q4 bare | **110**（与 oMLX 等价）|
| mlx-vlm Q4 + spec dec | **83-92**（spec dec **net negative**）|

**关键 finding**：**当前 default config 下 spec dec 反而比 bare 慢**（drafter overhead > token gain at accept rate 2.04-2.69 tok/round）。

**强烈推荐**：**不做 oMLX 集成**。
- 集成跑同 drafter 同 algorithm 大概率同样负收益
- 11h dev 浪费

**生产用 Q6 + thinking OFF**：
- user 实测 4/4 paradox 不靠 thinking
- per-token 比 Q4 慢 17%，但**总任务 wall-clock 快 2x**（thinking 链节省）
- 内存 25 GB（m5max 128 GB / m2max 充裕）

**未来再 revisit spec dec 触发条件**：
- mlx-vlm 上把 accept rate 拉到 4-5 tok/round（如 paper），才有 1.5x 落地空间
- 或换更轻 drafter（如 e2b for 26b-a4b target，待验证 vocab 一致性）
- 或测 31B Dense target（更大 target → drafter overhead 相对更小）

## 11. 已建产物总览

### 代码
- m2max:~/Code/omlx clone + .venv + editable install (142 tests pass，可继续做 dev)
- m5max:/tmp/mlxvlm-test venv（mlx-vlm 0.5.0，可复现实验）

### 文档
- `docs/omlx-gemma4-spec-dec-design.md` (本文)
- `docs/eval-stage1-status.md` (上一 session 的 BFCL 结果)

### Memory updates
- `gemma4_26b_a4b_sweet_spot.md` ← Q6 thinking OFF 4/4 finding
- `ds_thinking_default.md` ← Q6 + thinking OFF 表格
- `gemma4_spec_dec_not_mtp.md` ← 标题修正后版本
- `mlx_vlm_spec_dec_low_accept_rate.md` ← 实测数据（新）

### 配置变更
- m5max:~/.omlx/models/{gemma-4-26b-a4b-it-6bit, gemma-4-26b-a4b-it-8bit} → `gemma4-moe-26b-a4b-{q6, q8}`
- (alias rename 后 model_settings.json 的 key 可能需要刷 — 当前 m5max 上 oMLX no active engines 时改安全)
