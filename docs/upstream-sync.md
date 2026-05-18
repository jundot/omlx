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

## 最近同步

- **2026-05-18** — 对齐 `upstream/main` @ `51907f0`

## 已引入(cherry-picked)

| 上游 commit | flyto commit | 内容 | 引入日期 |
|---|---|---|---|
| `d736bfd` | `2e4d7c1` | chunked prefill: RuntimeError 作为 request error 上报 | 2026-05-18 |
| `c003b2e` | `ee2342e` | chunked prefill: 显存检查 + 进度回调 + dead-abort 检查 | 2026-05-18 |
| `386e16f` (#1244) | `cdaec79` | 测试: xgrammar import guard + 修上游既有测试失败 | 2026-05-18 |
| `51907f0` | `81f9815` | oQ: 给 VLM sensitivity 恢复 MTP head attach | 2026-05-18 |

cherry-pick 一律带 `-x`,commit message 里保留 "cherry picked from commit …"
溯源行,可用 `git log --grep="cherry picked from"` 反查。

## 确认已在 flyto(评估时已存在,勿重复引入)

- `11e6ea7` (#1224) chunked prefill 基座 —— flyto 早已有(换形状引入,
  本次补的是它的两个 follow-up 修复 `d736bfd`/`c003b2e`)
- `ccfba1d` (#1247) oQ-quant VLM 加载修复
- `37c73a0` phys_footprint enforcer + prefill 峰值 admission control
- `196d667` SchedulerQueueFullError → HTTP 503 + Retry-After
- `521cccf` (#1211) health-check Session 复用(防端口耗尽)
- `19bb34e` (#1214) `/v1/audio/transcriptions` 的 word_timestamps ——
  flyto 有更强的自有实现(aligner auto-chain + on_aligner_overflow)

## 评估后跳过(价值低或不适用,勿重复评估)

- `c54de70` / `be3b024` (#1251) 日志查看器 level filter —— admin UI QoL
- `5994dc5` (#1223) / `290587f` (#1255) codex/claude CLI 参数透传 —— 按需
- `4fe004d` (#1250) Hermes Agent quick launch
- `fc5171b` (#1088) 周期 health timer 重新检查更新
- `04a0ce6` / `25c312f` / `68b5c25` / `71beab7` / `7fab13b` 杂项小修

## 上游 issue 处理记录

记录 flyto 修掉的、与上游 issue 对应的问题(flyto 自身 bug 见
`docs/roadmap.md`)。

- 暂无。

## 待评估(open PR / issue,尚未决定)

见 `docs/roadmap.md` 之外的待办讨论;下次 review 上游 open PR 时,
把结论(引入 / 跳过)回填到本文件对应小节。
