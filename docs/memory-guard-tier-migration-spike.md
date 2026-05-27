# memory_guard_tier 迁移 spike

**状态**: design doc, 待 review. 实现见后续 PR.

**触发**: 上游 `c645c9f` + 5 个 follow-up (`3ef7b94` `4cfbc8b` `acd0533` `64bd2a2` `b129a19`) 删了 `max_process_memory` / `max_model_memory`, 换 `memory_guard_tier` + 实时 dynamic ceiling. flyto 113 处生产代码 + 129 处测试引用了旧字段, breaking change.

**这是 spike doc, 不是实现 PR.** 实现走 `feat(memory): adopt memory_guard_tier` 单独 PR, 等本文件 review 通过.

## 1. 现状 inventory

### 1.1 旧 API 形态 (flyto main @ `7b40ff2`)

两个独立 dataclass 字段, 各自支持 `"auto"` / `"<size>"` / `"disabled"` 三种字符串:

```python
@dataclass
class ModelSettings:
    max_model_memory: str = "auto"  # "auto" = 80% RAM
    # ... model_dirs, model_dir, model_fallback

@dataclass
class MemorySettings:
    max_process_memory: str = "auto"  # "auto" = RAM - 8GB, "disabled", or "XX%"
    soft_threshold: float = 0.85
    hard_threshold: float = 0.95
    prefill_memory_guard: bool = True
```

### 1.2 引用面 (生产代码 113, 测试 129)

| 文件 | 生产 | 测试对应 | 性质 |
|---|---:|---:|---|
| `omlx/settings.py` | 32 | `test_settings.py` 62 | 字段定义 + env var + CLI override + validation |
| `omlx/admin/routes.py` | 31 | — | admin API endpoints (GET/POST settings) |
| `omlx/admin/static/js/dashboard.js` | 19 | — | 两个 slider + 当前值显示 + 提交 |
| `omlx/engine_pool.py` | 15 | `test_engine_pool.py` 41 | model loading budget, LRU 驱逐 (functional, 不止 config) |
| `omlx/server.py` | 9 | `test_status_endpoint.py` 7, `test_audio_memory.py` 4 等 | init_server 把 `max_model_memory` 传给 engine_pool, status endpoint 报数据 |
| `omlx/cli.py` | 5 | `test_cli.py` 8 | `--max-model-memory` / `--max-process-memory` flag |
| `omlx/process_memory_enforcer.py` | 1 | (覆盖过) | enforcer 用 `max_bytes` 构造 (上游 #1383 + #1405 已经引入到 flyto) |
| `omlx/admin/templates/dashboard/_settings.html` | 1 | — | slider 模板 |

env vars: `OMLX_MAX_MODEL_MEMORY`, `OMLX_MAX_PROCESS_MEMORY`.
CLI args: `--max-model-memory`, `--max-process-memory`.

## 2. 上游新设计 (c645c9f + follow-ups)

### 2.1 settings shape

`ModelSettings.max_model_memory` **删**. `MemorySettings.max_process_memory` **删**, 换成:

```python
@dataclass
class MemorySettings:
    prefill_memory_guard: bool = True
    memory_guard_tier: Literal["safe", "balanced", "aggressive"] = "balanced"
    soft_threshold: float = 0.85
    hard_threshold: float = 0.95
    prefill_safe_zone_ratio: float = 0.80   # acd0533 加
    prefill_min_chunk_tokens: int = 32       # acd0533 加
```

### 2.2 tier 到 reserve 的映射 (process_memory_enforcer.py)

```python
_STATIC_RESERVE_LARGE = {
    "safe":       12 * 1024**3,  # >= 16GB 系统从总 RAM 减 12GB
    "balanced":    8 * 1024**3,
    "aggressive":  6 * 1024**3,
}
_OTHER_APP_RESERVE = {
    "safe":        2 * 1024**3,
    "balanced":    1 * 1024**3,
    "aggressive":  512 * 1024**2,
}
_SMALL_SYSTEM_RESERVE = 4 * 1024**3   # < 16GB 系统一律 4GB, 不分 tier
```

### 2.3 ceiling 算法

```
ceiling = min(static_ceiling, dynamic_ceiling)
  static_ceiling  = total_ram - tier.static_reserve
  dynamic_ceiling = omlx_phys_footprint + system_available - tier.other_app_reserve
```

`static_ceiling` 是绝对天花板, `dynamic_ceiling` 跟着 `psutil.virtual_memory().available` 实时变. 别的 app 抢内存时 dynamic 立刻下降, 触发 LRU 驱逐 / scheduler abort.

### 2.4 engine_pool 变化

`max_model_memory` **删**. budget 不再是 user 配的固定值, 改为通过 callback `_get_final_ceiling()` 实时向 enforcer 拿 ceiling:

```python
# 上游 engine_pool.py
class EnginePool:
    def __init__(self, scheduler_config: SchedulerConfig | None = None):
        # 不再接 max_model_memory 参数
        self._get_final_ceiling: object | None = None  # 由 server.init_server() 装

    def _current_ceiling(self) -> int:
        return self._get_final_ceiling() if self._get_final_ceiling else ∞
```

加载新模型时检查 `current_model_memory + estimated_size <= current_ceiling`. ceiling 实时反映系统压力.

### 2.5 admin UI 变化

两个 slider (Memory Limit Total + Memory Limit Models Only) 删, 改一个 dropdown:

```
Memory guard tier: [ safe / balanced / aggressive ]   [Custom...]
```

`64bd2a2` (#1431) 后加了 Custom ceiling 选项 — 用户可以填一个 GB 数字覆盖 tier 的 static_reserve.

### 2.6 上游 6 个 commit 的职责拆分

| commit | 内容 |
|---|---|
| `c645c9f` | 主体重写: 删 max_*_memory, 加 memory_guard_tier, 改 enforcer + engine_pool + admin UI |
| `3ef7b94` | clamp ceiling 到有效 Metal cap (`iogpu.wired_limit_mb`) + sysctl 警告 |
| `4cfbc8b` | adaptive throttle 切到 watermark-tier shrink |
| `acd0533` | adaptive prefill throttle + user-explicit hard cap (引入 `prefill_safe_zone_ratio` / `prefill_min_chunk_tokens` / `user_explicit_max`) |
| `64bd2a2` (#1431) | tier-aware active-memory reclaim + Custom ceiling 选项 |
| `b129a19` (#1425) | test 跟 c645c9f 同批: 测试用 `memory_guard_tier` 替 `max_*_memory` |

## 3. 决策点

### 3.1 采不采?

**采**. 三条理由:

1. **dynamic ceiling 价值真**: 别的 app (Chrome / VS Code / Slack) 抢内存时立刻反映, 而不是 oMLX 仍按 user 配的 hardcode 数字硬跑. m2max / m5max 这种共用机器收益明显.
2. **UI 简化合理**: 用户对"split a budget between process and models"困惑, dropdown 比两个 slider 直观. flyto 用户群 (技术性比上游强, 但仍多是个人 dev) 也会喜欢.
3. **不采的代价高**: 上游已经全速向这个方向跑, 后续 fixes (adaptive throttle, tier-aware reclaim) 都依赖新 API. 不采每次 sync 都要手动跳过, 越拖越脏.

唯一担心: **breaking config**. 老用户 settings.json 里有 `max_model_memory: "64GB"`, 升级后直接被忽略, 用 balanced 默认. 解决方案见 3.2.

### 3.2 配置迁移

**flyto 选项 (与上游不同)**: 旧字段保留为 **deprecated alias** 三个 release, 然后删. 期间:

- 旧 `max_process_memory: "92GB"` → 启动时映射到最接近的 tier (按 `total_ram - value` 算出 reserve, 落到 safe/balanced/aggressive 的最近一档), 同时 log warning "max_process_memory deprecated, 推断 tier=balanced, 请改 memory_guard_tier".
- 旧 `max_model_memory` → **直接忽略**, log warning. engine_pool 用新的 ceiling-based budget 已经覆盖这个需求.
- env var `OMLX_MAX_PROCESS_MEMORY` → 同字段, deprecated alias.
- env var `OMLX_MAX_MODEL_MEMORY` → 忽略, log warning.
- CLI `--max-process-memory` / `--max-model-memory` → 同上 (alias + ignore).

新字段: `memory_guard_tier`, env `OMLX_MEMORY_GUARD_TIER`, CLI `--memory-guard-tier`.

### 3.3 admin UI

dropdown + Custom 选项一起做. 不要分两次:

- dropdown: safe / balanced / aggressive
- Custom ceiling input: 用户填 GB 数字, 后端转成 `custom_ceiling_bytes` (额外字段), enforcer 看到 custom 时用 `min(custom, dynamic_ceiling)` 而不是 tier 的 static.

UI 拷贝上游的 `_settings.html` 修改, dashboard.js 同步.

### 3.4 滚动: 单 PR vs 分阶段

**分阶段, 3 个 PR**:

1. **PR-1 backend**: settings.py + process_memory_enforcer + engine_pool + server.py 的字段切换. 保留 deprecated alias. admin UI 暂不动 (旧 slider 仍工作, 后端转换). 测试: test_settings, test_engine_pool, test_process_memory_enforcer 重写.
2. **PR-2 cli / env**: CLI args + env var 切换 + deprecation warning. test_cli 重写.
3. **PR-3 admin UI**: dropdown + Custom, 删旧 slider. admin routes 同步.

每个 PR 都跑零回归 baseline. 这样 review 可控 (1 个 PR 不超过 ~400 lines 改动), 万一某段引入 bug 单独 revert.

### 3.5 测试影响

`test_settings.py` 62 个 + `test_engine_pool.py` 41 个 + 其他 26 个 = 129 处需要改. 大部分是 mock 旧字段的 dataclass 构造, 改成新字段就 OK. 不需要重设计 test, 只需要 search-and-replace.

`test_process_memory_enforcer.py` 已经有 `test_hard_limit_honors_user_explicit_max` skip (见 2026-05-26 同步), 这条以及相关的 acd0533 测试在 PR-1 一并 unskip.

## 4. 风险

- **配置迁移 silent**: 老用户 settings.json 里的 `max_*_memory` 字段被忽略, 如果没读启动 log 就不知道. 缓解: 同时在 admin dashboard 顶部弹个 banner ("settings.json 含 deprecated 字段, 点这里看迁移指南") 三个 release 内.
- **dynamic ceiling 误判**: psutil 在某些 macOS 版本上 `virtual_memory().available` 不准 (尤其是 macOS 26 Tahoe 新内核), 可能导致 ceiling 抖动. 缓解: 加 EMA smoothing (上游 `acd0533` 引入的 `prefill_safe_zone_ratio` 已经部分覆盖).
- **engine_pool 加载新模型测算**: 旧设计是 `estimated_size <= max_model_memory` 单点判断; 新设计要每次拿 callback ceiling. 性能影响可忽略 (callback 是同步 dict 读 + 算), 但要确保 callback 在 server 启动完之前已经接好 (否则 fallback 到 unconditionally admit, 这是上游设计行为).
- **Custom ceiling 边界**: 如果用户填 `1GB` 这种小到不能加载任何模型的值, 要早 fail 而不是 silently 让 engine_pool 拒绝所有加载. 在 admin POST settings 时校验.

## 5. 不在本 spike 范围

- mlx_lm 自己的 `iogpu.wired_limit_mb` sysctl 调整 (`3ef7b94` 的 clamp 是发现 cap 而不是改 cap).
- TurboQuant KV cache 的内存占用估算 — 与 model_memory 概念正交.
- DFlash drafter 共享 target weights 的"双引擎其实只占一份内存"那条 — engine_pool 当前已经在算 estimated_size 时正确处理.

## 6. 启动条件

review 通过后:

- 开 `sync/memory-guard-tier-pr1-backend` 分支做 PR-1.
- 不一次性引入 6 个上游 commit. `c645c9f` 主体做完后单独 review, `3ef7b94` `4cfbc8b` `acd0533` `64bd2a2` `b129a19` 各自再 cherry-pick (或挑相关部分手实现) 走 follow-up PR.
- 每个 PR 跑完整套件, 4509 pass / 4 fail / 37 skip 作 baseline. acd0533 那个 skip 在 PR-1 完成时 unskip.

## 7. 时间估计

- PR-1 backend: ~3-4 小时 (settings + enforcer + engine_pool + server + 测试重写)
- PR-2 cli / env: ~1 小时
- PR-3 admin UI: ~2-3 小时

总计 ~6-8 小时, 单人. 可一次会话完成 (PR-2 / PR-3 在 PR-1 merge 后顺接).
