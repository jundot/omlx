# Flyto MLX 脱离上游路线图

品牌已于 2026-05-16 独立 (panwudi/omlx -> panwudi/flyto-mlx)。下一步是技术独立。
本文把 flyto 对上游的依赖结构、脱离的现实约束、分阶段路径讲清楚, 供业务排期决策。

技术判断在本文里给完, 需要拍板的地方都翻译成业务取舍 (投入 / 优先级 / 风险 / 排期)。

## 一、依赖现状: 两条主线, 已双向分叉

数据采于 2026-06-04 (`git log` / 隔离 venv 实测)。

### 线 1 - jundot/omlx 的代码

- flyto 的引擎主体是从 jundot vendored 来的, 自有代码占比小。
- 分叉量化:
  - flyto main 独有 214 commit (其中 cherry-pick 自上游 68, 真 flyto 自有 146)
  - flyto 落后上游 287 commit
  - 即双向分叉。cherry-pick 成本随分叉扩大递增。
- 146 个真自有 commit, 看内容主要是:
  - rebrand 表皮: `~/.fmlx`、menubar 瘦壳 (apps/omlx-mac)、PyPI 发布、命名
  - 国内适配: ModelScope 下载 (HF 的 Xet 在国内死路)
  - bug fix: tool-calling、vlm_mtp stream、model size、grammar-thinking
  - 发布 / 打包工程: 0.5.0、venvstacks
  - 引擎级真特性主要是 cluster router (request-level resource-aware 路由)
- 结论: flyto 的差异化在表皮 + 国内适配 + 一个路由特性。核心引擎
  (LLM/VLM/cache/speculative/oq) 基本是上游的。脱离 jundot = 接管上游引擎主体的维护。

### 线 2 - mlx-vlm 的私有 API (真正的硬依赖)

mlx-vlm 不是 Apple 官方库。Apple 官方 (ml-explore org) 只做 mlx 核心 + mlx-lm (纯文本 LLM)
和 swift 绑定, 没有 VLM 库。mlx-vlm / mlx-audio / mlx-embeddings 都是 Prince Canuma
(GitHub Blaizzy, 波兰独立开发者, 前 Arcee AI) 一个人的项目, 是社区事实标准。所以 flyto 的
多模态命脉押在一个人的独立项目上, 这放大了下面所有的脆弱性。

- oMLX 钩 mlx-vlm 内部 49 个符号, 其中 18 个是 turboquant 私有符号
  (`_build_codec` / `_slice_state` / `_write_state` ...)、speculative 私有函数
  (`_mtp_rounds`)、qwen3_5 内部类。
- 这是上游 oMLX 引入的深耦合, 不是 flyto 加的 (flyto 只在
  `patches/qwen3_5_attention.py` 自加 6 处)。
- 它比 jundot 更脆 (私有 API 随时重构)、更不可控 (Blaizzy 节奏, 一年才发一次 PyPI,
  全靠 git pin)。实测从 f96138e 升 041f889, 49 钩 import 不破, 但运行时语义需要配套
  适配 -- 上游升这个 pin 时同 commit 删 / 改了 16 个文件 (含整删
  `qwen3_5_attention.py` 327 行) 就是证据。
- 结论: 真正卡脖子的是这条线, 不是 jundot。脱离 jundot 但不管这条, 等于从
  "被 jundot 牵" 变成 "被 Blaizzy 牵"。

### 线 3 - 其它 git-pin 依赖

| 包 | 来源 | 备注 |
|---|---|---|
| mlx-lm | ml-explore (官方) | 相对稳定, 官方维护 |
| mlx-vlm | Blaizzy | 线 2 的来源, 高速变动 |
| mlx-embeddings | Blaizzy | |
| mlx-audio | Blaizzy | STT/TTS |
| dflash-mlx | bstnxbt | 投机解码 |

5 个 git-pin 里 4 个是 Blaizzy 生态。"脱离 jundot" 不等于脱离 Blaizzy。

## 二、给决策用的三点认知

1. flyto 的价值在表皮 + 国内适配 + cluster router, 不在引擎。脱离 jundot 的代价是
   接管上游引擎主体的长期维护 (bug fix + 新模型支持, 这些现在上游免费做)。
2. 真正的硬依赖是 mlx-vlm 私有 API (Prince Canuma/Blaizzy 个人项目, 非 Apple 官方),
   比 jundot 更难脱。只脱 jundot 是治标。唯一 Apple 官方、相对稳的依赖是 mlx-lm (纯文本)。
3. 已事实分叉 (独有 214 / 落后 287), 继续跟随的 cherry-pick 成本只会涨。脱离有一部分
   是追认现实, 不是凭空多出来的工作。

## 三、三种脱离深度 (业务选择)

逐步推进, 程度递增。可以先走浅的, 跑顺了再决定要不要更深。

### A. 维护独立 (最浅, 治标)

- 做什么: 停止被动跟随 jundot。锁定当前依赖快照, 由 flyto 自己主导节奏 --
  自己测试、自己决定要不要某个上游 commit、自己出升级 commit。代码仍可参考上游 (开源),
  但主导权在 flyto。
- 成本: 低。主要是建立 flyto 自己的测试 / CI baseline。
- 失去: 不多 -- 上游仍可选择性参考。
- 追新能力: 仍依赖上游做新模型支持, 只是不再自动吸收。
- 适合: 想先稳住阵脚、把 "不等上游也能升级" 的流程跑通。

### B. 收敛依赖 + 选择性吸收 (中等, 治本一半)

- 做什么: 在 A 基础上, 把对 mlx-vlm 的 49 个私有钩收敛成一个 flyto 自己控制的适配层
  (adapter/shim), 让 mlx-vlm 升级只冲击这一层而不是散落 37 个文件。jundot 降级为
  "新模型支持的选择性参考源"。
- 成本: 中。要设计适配层 + 一次性迁移 49 钩。
- 失去: 上游引擎演进要主动判断吸收, 不再无脑 cherry-pick。
- 追新能力: 自主性显著提升, mlx-vlm/jundot 变动的冲击被适配层挡住。
- 适合: 认定要长期独立, 愿意一次性投入把最脆的耦合 (mlx-vlm 私有 API) 收口。

### C. 深度自主 / 固化分叉 (最深, 治本)

- 做什么: 选定 mlx-vlm 版本固化甚至 vendor 进 flyto, 把关键推理路径自己拥有,
  从此停止跟 jundot / mlx-vlm 做 diff, 完全自管。
- 成本: 高。等于接管整个引擎栈的维护。
- 失去: 上游全部免费维护 + 新模型支持, 追新全靠自己。
- 追新能力: 完全自主, 但要自己有持续投入的工程力量。
- 适合: 有长期专属团队 / 战略要彻底掌控, 且能接受追新变慢。

## 四、分阶段路径 (逐步)

不论最终选 A/B/C, 起步阶段是共用的:

- 阶段 0 - 摸清并锁住家底 (立即, 低成本)
  - 建立 flyto 自己的测试 baseline (现在测试套件有已知 pre-existing 失败, 先固化基线)。
  - 锁定 5 个 git-pin 快照, 记录每个的耦合面 (本文线 2/3)。
  - 产出: 一份 "自己维护需要什么" 的清单。

- 阶段 1 - 第一次自主消化一次 mlx-vlm 升级 (gemma4_unified 作练习)
  - 不等上游 sync, flyto 自己把 mlx-vlm 升到含 gemma4_unified 的版本, 自己处理那 16
    个文件的适配、patch 删除、6 处 qwen3_5 定制冲突, 隔离环境跑通 + 加载验证, 出一个
    flyto 自己的升级 commit。
  - 价值: 这是 A 的核心能力练习 -- 跑通后就有了 "不靠上游适配也能升级" 的流程和成本
    基线, 顺带产出适配笔记 (喂给阶段 2)。

- 阶段 2 - 收敛 mlx-vlm 私有耦合 (走 B 才需要)
  - 把 49 钩收敛成适配层。

- 阶段 3 - jundot 关系降级为选择性参考 (走 B/C)
  - 停止全量 cherry-pick, 改为按需挑新模型支持。

## 五、建议

脱离深度由维护能力决定, 不是由意愿决定。真正的约束是: 能投入多少持续维护。一旦停止跟随
上游, 就失去它免费的 bug fix + 新模型支持, 这些要自己接。所以三档不是平等选项:

- C (完全自管引擎栈): 如果 flyto 长期是小规模投入, 这是陷阱不是选项 -- 独自追一个高速
  变动的 mlx 生态会拖垮维护。
- B (收敛 mlx-vlm 耦合成适配层): stretch 目标。只有愿意一次性投入做适配层、且之后持续
  维护它时才划算。
- A (维护独立 + 选择性参考上游): 小规模运营的现实天花板。拿到自主节奏, 又保留上游作为
  免费参考源。

技术推荐: 走 A, 先做 阶段 0 + 阶段 1 起步。阶段 1 (自主消化 gemma4_unified 升级) 一举
多得 -- 既让那个 12B 跑起来, 又把 A 的核心能力 (不等上游也能升级) 实战跑通, 还摸清独立
维护的真实成本 (为以后要不要走 B 提供依据)。全程隔离验证, 不碰生产, 出 commit 后由你
greenlight 上 m2max/m5max。

要你拍的业务 go: 以 flyto 的维护投入水平, A 是我的推荐, 阶段 0+1 起步 -- 做不做?
(B/C 不急, 等阶段 1 摸到真实成本再定。)
