# fmlx 视频生成引擎 spec (Wan2.2 T2V, mlx-gen 运行时)

状态: 设计稿 v2 (2026-06-10), 未实现, 待拍板.
v2 = v1 经 6 视角对抗评审修订: 22 条 blocker/major 发现全部确认并吸收
(拒绝臂位置, OpenAI SDK multipart 兼容, Metal wired 双进程治理, 租约双重
计数, venv 污染, A/B 协议算术错误等). 评审记录见会话工单.
定位: Flyto MLX (fmlx) 自有功能, 不回流上游 (soft-fork 自有分化, 参见 §10).
本文档所有代码事实均经子代理逐行核实 (file:line 可验证), mlx-gen 事实核实
到其源码与 pyproject (2026-06-10, v0.18.14).

## 0. 背景与定位

fmlx 当前是 LLM/VLM/audio 推理引擎. 战略方向调整: Apple Silicon 单机统一内存
(128GB 级) 对本地多媒体生成 (文生视频/文生图) 是结构性优势 -- 大权重 + 大激活
全在 UMA 里, 不需要多卡切分. fmlx 要把 "单机多媒体" 做成与上游 oMLX 的核心差异.

第一个落点: Wan2.2 T2V A14B (MLX 量化 8bit, diffusers 布局, 42.4GB) 已完整下载
并逐文件校验通过, 位于 m5max `~/.fmlx/models/AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit`.
该权重就是为纯 MLX 运行时 mlx-gen 制作的 (safetensors dtype = U32+scales+biases,
即 mx.quantize 格式), mlx-gen 文档的示例命令逐字引用这个 repo.

## 1. 目标与非目标

目标 (MVP, P1):

1. fmlx 能发现 diffusers 布局的视频模型 (model_index.json), 类型化为 `video`,
   在 /v1/models 与 admin 列表中正确展示, 可删除, 不污染 chat 模型列表,
   不会成为隐式默认模型.
2. 新增 OpenAI 形态的异步 job API: POST /v1/videos 提交, GET 轮询, list 枚举,
   content 下载. 官方 openai SDK 的 client.videos.* 可直接打通 (含其
   multipart 提交形态).
3. 生成跑在独立 venv 的 subprocess worker 里, 与 LLM 服务进程隔离; worker
   自身被 Metal wired limit 钉死在租约内 (预防性, 非反应性).
4. 视频任务持有内存租约 (lease), 经现有 ProcessMemoryEnforcer 单一咽喉点
   传播. 与中小 LLM (权重 + 工作集能与 lease 共存于 ceiling 内) 真共驻;
   与超大 LLM (如 glm4.5 85GB) 是设计上的互斥 -- job 排队等内存, 不硬挤,
   不重蹈 m5max kernel panic.
5. 全链路在 m5max 真机 A/B 验证后才可合并 (本项目铁律: 单测过 != 真机过).

非目标 (MVP 不做, 部分进 P2):

- SSE 进度流 (轮询够用), 图生视频 (I2V) 输入上传, 文生图 (FLUX 系), TI2V-5B,
  admin 专属视频 UI 页, 多并发生成 (Semaphore(1) 一次一个), 分布式队列,
  ModelScope 下载视频模型 (有 flat symlink 陷阱, 见 §4.1), 训练/LoRA,
  为视频任务主动驱逐已加载 LLM (MVP 只被动排队, 驱逐策略 P2).

## 2. 关键事实 (设计依据)

以下两小节分别是外部运行时与本仓代码的核实结论, 全部影响 §3 的架构取舍.

### 2.1 mlx-gen 运行时

| 维度 | 事实 |
|---|---|
| 真身 | filipstrand/mflux 的 fork; Python 包名是 `mflux`, `import mlxgen` 只是 sys.modules 别名 |
| 视频类 | `mflux.models.wan.variants.Wan2_2_TI2V` 一个类管全部 Wan 变体; `ModelConfig.wan2_2_t2v_a14b()` + `model_path=<本地目录>` 即可加载我们已下载的目录 (路径解析规则 1 "exists_locally" 短路一切) |
| 生成 API | `generate_video(seed, prompt, steps, height, width, num_frames, fps, ..., progress_callback)` 阻塞同步, batch=1; ProgressEvent 带 phase/step/total_steps |
| 取消 | 无一等取消 API; callback 里抛异常可中断但实例报废 -- 健壮取消 = 杀 subprocess |
| 依赖 | 不是纯 MLX: torch 是硬依赖 (UMT5 text encoder 走 torch/CPU), 另有 transformers>=5, huggingface-hub>=1.1.6,<2, opencv, matplotlib, av (PyAV 自带 ffmpeg wheel); twine 混在 runtime deps 里 (供应链卫生信号, 计入 §9 风险评级) |
| 输出 | GeneratedVideo (PIL frames + 元数据), .save() 写 MP4 + 健康校验 + metadata sidecar |
| license / 版本 | MIT; v0.18.14 (2026-06-08); 两周内 15 个 release, bus factor 1 (lpalbou) |
| 实测内存 (官方, M5 Max 128GB) | T2V A14B q8: 物理峰值 20.7 GiB, MLX 峰值 15.5 GiB, 154.8s @ 384x224, 33 帧, 12 步; 生产分辨率 (480x240, 101 帧, 25 步) 约 30 分钟. 注意: 这是唯一公开测点, 是小 profile |
| 多线程 | 文档明言 model 实例有状态, 必须串行访问 |

依赖结论: mlx-gen 的 transformers>=5 / hf-hub<2 / torch 与 fmlx 主 venv 共装冲突
风险高且无必要 -- 这是 subprocess + 独立 venv 方案的第一推力.

### 2.2 fmlx 代码侧事实 (全部 file:line 已核实)

- 发现机制只认根 config.json (`_is_model_dir`, model_discovery.py:697-699).
  Wan2.2 目录在 owner/repo 两级布局下整体隐身; 但在 FLAT 布局下 (恰好是
  ModelScope 下载器产出的 symlink 形态, ms_downloader.py:665) 会被当成 org
  文件夹下钻, transformer/ transformer_2/ vae/ text_encoder/ 各自带 config.json,
  会注册成 4 个幽灵 "llm" 模型, 甚至可能成为默认模型 (server.py:1279-1290).
  这是现存隐患, 发现机制改造必须先行.
- pool.get_engine 在调用 _load_engine 之前就跑内存准入循环 (engine_pool.py:
  359-396): projected = current + entry.estimated_size, 不够就 LRU 驱逐已加载
  模型. 任何 "在 _load_engine 里拒绝" 的方案都晚了 -- 拒绝必须在准入循环
  之前 (§3, §4.1).
- 隐式默认模型 = available_models[0] (server.py:1279-1292), 不分类型; model
  fallback (server.py:698-711) 会重试默认模型. video 条目必须从两处排除.
- 全局 MLX executor 是 max_workers=1 的单线程池 (engine_core.py:106-120),
  全部非 batched 引擎的 GPU 操作串行其上 (Metal command-buffer race, #85).
  in-process 跑分钟级 diffusion 会头部阻塞 audio/embedding/unload. 第二推力.
- mlx-gen 无取消 API + audio 模板全链路无超时 (audio_routes.py 零 asyncio.wait_for).
  第三推力: subprocess kill 是唯一可靠的取消/超时手段.
- 内存防护活体只有 phys-based 链路: ProcessMemoryEnforcer (1s tick) ->
  `_get_hard_limit_bytes` (process_memory_enforcer.py:495-517) 是单一咽喉点,
  pool 准入/软硬水位/prefill gate cap 全部从它派生; estimate/monitor 那套在
  生产 inert (memory_monitor 永远 None), 不得依赖.
- 动态 ceiling 在 safe/balanced/aggressive 档 (生产默认 balanced) 是
  系统级感知的 (own_phys + free + inactive + active*ratio, process_memory_
  enforcer.py:483-493): worker 真实占用会经由 free 下降自动压低父进程
  ceiling -- 与显式租约叠加就是双重计数, 必须修正 (§4.4).
- `get_phys_footprint(pid)` 接受任意 pid (utils/proc_memory.py:94-118), 父进程
  可测 worker 足迹; 失败返回 0, 必须按错误处理. 同文件 :63 已声明
  `ri_lifetime_max_phys_footprint` (进程生命期峰值 ledger) 但全仓未读 --
  P0 测量的正确仪器 (§7).
- Metal wired limit 是 per-process 可设的 (mx.set_wired_limit; enforcer 对
  自己进程已这么做, process_memory_enforcer.py:407-446). worker 子进程默认
  继承机器级 cap (~107.5GB), 不主动设限就没有任何 Metal 级约束 -- 这是
  双进程 wired-sum panic 的根源, 也是 §4.4 预防性方案的依据.
- 后台 job 的成熟范式在 admin 侧: HFDownloader (task dict + status enum +
  asyncio.create_task + 协作取消) 与 OQManager (Semaphore(1) + is_quantizing
  被推理端点 503 联动). /v1/responses 的 `background` 字段是死的 (零消费者),
  ResponseStore 只存终态无生命周期, 都不能直接复用.
- 非 LLM 引擎接入范式: audio 三件套 (BaseNonStreamingEngine + api/audio_routes.py
  直连 pool). 注意 audio 路由的条件挂载 (server.py:439-448) 发生在模块 import
  时, 彼时 settings 尚未初始化 (init_server 才注入) -- 它能工作只因为它的门
  是 "mlx_audio 可 import". settings 驱动的门不能放在那里 (§4.3).

### 2.3 共驻内存风险 (m5max 教训)

128GB 机 Metal wired cap 约 107.5GB, 越线 = 整机 kernel panic (已发生过).
要害不是稳态而是瞬时尖峰; 1s 轮询与 chunk 边界读数都看不见 sub-poll 瞬时.
M5 Max 内存带宽 ~0.5TB/s 量级, 一次 mx.eval 可以在远小于 2s 的窗口里物化
几十 GB -- 任何 "轮询 + 杀进程" 的反应式手段都不构成 panic 兜底, 只能做
次级清理. 预防性手段只有两类: Metal wired limit (per-process, 越限退化为
非常驻页或分配失败, 不越机器 cap) 与余量常数 (prefill 侧 12GB margin 的
方法论, settings.py:404-412). 视频侧两者都要用 (§4.4).

## 3. 总体架构

决策: subprocess worker + 独立 venv + job manager, 视频模型注册进 pool 名册
但被 typed 拒绝在加载链路之外 (拒绝点在准入循环之前).

```
fmlx server 进程 (主 venv, 无 mlx-gen 依赖)
  |- model_discovery: 认出 video 模型 (model_index.json), 列表/删除/设置
  |- server.get_engine: alias 解析后 entry.model_type=="video" -> 400 + 指引
  |- pool.get_engine: 入口处 (准入循环之前) video -> ModelTypeNotLoadableError
  |- /v1/videos 路由 (api/video_routes.py, 无条件挂载, handler 内按设置门控)
  |- VideoJobManager (omlx/video/manager.py, lifespan 内构造, 注入 enforcer)
  |    |- queue + Semaphore(1) + job 持久化 (JSON per job)
  |    |- 内存租约: enforcer.acquire_video_lease(bytes) / set_video_worker_pid
  |    |    / release_video_lease
  |    |- spawn: <video_venv>/bin/python -I <omlx>/video/worker.py --spec job.json
  |    |- 监控: stdout JSONL (进度 + 相位心跳) + 足迹 watchdog + 停滞超时
  |    |- 取消/超时: SIGTERM -> 5s -> SIGKILL
  |- ProcessMemoryEnforcer: ceiling -= lease; 动态 ceiling 加回 min(worker, lease)
  |
  +-- worker 子进程 (video venv: 锁定依赖集, 不 import omlx)
       |- 进场即 mx.set_wired_limit(lease 内值) -- Metal 级自缚 (预防性)
       |- mflux Wan2_2_TI2V(model_config=..., model_path=registry 提供的本地目录)
       |- generate_video(progress_callback -> stdout JSONL)
       |- video.save(<artifacts>/<job_id>/output.mp4) -> exit 0
```

为什么不 in-process (按否决强度排序):

1. 依赖冲突: torch/transformers>=5/hf-hub<2 装进主 venv 风险不可控.
2. 取消与超时: mlx-gen 无取消 API, in-process 卡死的 denoise 永久占住全局
   MLX executor 且无 kill 手段; subprocess 杀进程即回收一切 (含 Metal 内存).
3. executor 头部阻塞: 分钟级任务串行在 max_workers=1 的全局执行器上,
   audio/embedding 全堵.
4. 崩溃隔离: 视频管线 NaN/Metal 错误不殃及 LLM 服务.
5. 内存回收确定性: 进程退出即归零, 无碎片/泄漏累积.

代价与对策:

- worker 内存对父进程 phys_footprint 不可见, 但对动态 ceiling 可见 (经 free
  下降) -> 租约 + 加回修正, 计一次不计两次 (§4.4).
- 每个 job 冷加载权重 (42GB 读盘) -> MVP 接受; P2 再考虑常驻 worker + idle TTL.
- 双进程 wired-sum -> worker 自缚 wired limit (预防) + watchdog (清理), §4.4.

视频模型与 engine pool 的关系: 发现机制注册 entry (model_type=engine_type=
"video"), 使列表/设置/删除/类型护栏全部生效. 但 video 条目永不可加载:

- pool.get_engine 在 entry 查到后, already-loaded 快路径与准入循环之前,
  对 video 抛新 typed 异常 `ModelTypeNotLoadableError` (子类 EnginePoolError,
  消息携带 "use POST /v1/videos"). 这保证零驱逐/零 settle barrier/零 507
  副作用 -- 若拒绝放在 _load_engine 里, 一次误指 video 模型的 chat 请求
  就会先按 42GB 跑准入, 驱逐在驻的生产 LLM 再被拒 (评审 blocker, 已核实).
- server.get_engine 在 alias 解析后, 进 pool 之前, 查 entry.model_type ==
  "video" -> HTTPException 400 + /v1/videos 指引 (chat/embeddings/rerank 全部
  流经此函数, 一处护全). 异常映射链在 EnginePoolError->500 之前加
  ModelTypeNotLoadableError->400 臂. 原 v1 计划的 _suggest_endpoint_for_engine
  加提示是死代码 (该函数只对成功返回的 engine 实例 isinstance, video 永远
  没有实例), 撤销.
- /v1/models/{id}/load 与 admin load 端点各自加 pre-pool 类型检查 -> 400
  (公共 load 端点的 blanket except Exception->500 会吞 typed 异常, 必须在
  进 pool 前查).
- _load_engine 的 dispatch 链里保留防御性 raise (同 typed 异常), 护住其他
  pool.get_engine 调用方.
- 默认模型卫生: 隐式默认选择 (server.py:1279-1292) 过滤到 model_type in
  {"llm","vlm"}, 无候选则 default=None (落到现有干净 400); model_fallback
  (server.py:698-711) 重试前校验默认模型类型; admin 默认模型设置器
  (admin/routes.py:2171-2173) 拒绝 video 条目.

权重生命周期完全归 worker 子进程; pool 的 42GB 准入与卸载 settle barrier
对 video 条目因前置拒绝而永不触发.

## 4. 模块设计

按改动面从发现层到 API 层再到内存与配置依次展开.

### 4.1 模型发现与类型系统

改动点 (全部小而集中):

- `_is_model_dir` (model_discovery.py:697-699): `config.json 存在` 或
  `model_index.json 存在` 均算模型根. 后者必须先于 org-folder 下钻判定,
  这同时修掉 §2.2 的幽灵组件隐患 (flat 布局不再下钻 transformer/ 等子目录).
- `detect_model_type` (model_discovery.py:385-549): 在 "config.json 缺失 ->
  llm" 早退 (404-406) 之前加 model_index.json 分支: 读 `_class_name`,
  在允许清单内 (MVP = {"WanPipeline"}) -> "video"; 不在清单 -> 跳过哨兵,
  `_register_model` 据此跳过并 log warning (不注册不可跑的管线, 也不产
  幽灵). 契约说明: 所有 config.json 路径保持现有 str 返回契约不变, 哨兵
  只出现在 "model_index.json 存在且 _class_name 不在清单" 的新分支 --
  现有测试零破坏.
- Literal 与映射五处同改 + 一致性测试: model_discovery.py:26-27,
  engine_pool.py:56-57, `_MODEL_TYPE_TO_ENGINE` (engine_pool.py:203-211),
  `_register_model` if/elif (model_discovery.py:737-751), admin valid_types +
  type_to_engine (admin/routes.py:1860, 1870-1878). 三份重复映射已是现存
  债务, 加断言测试防 silent "batched" 降级.
- 加载链路拒绝 (位置是要害, 见 §3): pool.get_engine 入口 typed 拒绝 +
  server.get_engine pre-pool 400 + 两个 load 端点 pre-pool 400 +
  _load_engine 防御臂. 新异常类入 engine_pool 异常族.
- 默认模型与 fallback 卫生 (见 §3 末尾): 隐式默认过滤 / fallback 校验 /
  admin setter 拒绝, 配 "video 模型按字典序排第一" 的发现 fixture 单测.
  这顺带修掉 embedding/audio 模型当默认的同款现存隐患.
- `estimate_model_size`: 递归 **/*.safetensors 分支 (679-681) 已覆盖 diffusers
  布局 (42GB), 不改; 该值对 video 只作展示 -- 准确表述: video 条目因前置
  拒绝永不进入会消费 estimated_size 的准入循环.
- /v1/models 卫生: ModelInfo (api/openai_models.py:409-415) 增加 `model_type`
  字段并在 server.py:1717-1722 填充 (对 OpenAI 客户端是 additive);
  这同时激活 cli.py:349 现成但 inert 的 llm/vlm 过滤. admin chat picker
  (dashboard.js:2081) 已天然排除未知类型.
- admin DELETE / 本地列表的 config.json 门 (admin/routes.py:4538/4547/4495/
  4511) 放宽为 config.json|model_index.json, 否则 42GB 模型在 UI 不可见
  不可删. 共享一个 is_model_root() helper, 不再三处发散.

### 4.2 VideoJobManager 与 worker 协议

构造与接线: VideoJobManager 在 lifespan 启动序里构造, 紧跟 enforcer 块之后
(server.py:367 后), 构造器注入 `enforcer: ProcessMemoryEnforcer | None`
(镜像 server.py:365-366 给 pool 注入 enforcer 的先例); 实例存
`_server_state.video_job_manager`, 路由经 `_get_video_job_manager()` 懒访问器
取用 (audio_routes.py:68-80 范式, 单测可 patch). 不得在 init_server 里构造
(init_server 先于 lifespan, 彼时 enforcer 不存在); "仿 OQManager" 只指
job/队列/持久化形态, 不指构造位置.

job 模型:

- id (uuid4, 前缀 "video_"), 对外 status 严格四值 `queued|in_progress|
  completed|failed` (与 openai SDK Video.status Literal 完全一致; 取消不是
  wire 状态, 内部记日志/metrics 即可, to_dict() 永不输出 cancelled).
- progress 0-100, phase 字符串, created_at / started_at / completed_at,
  `expires_at` (nullable; 产物被保留策略清除时置为清除时刻, 记录本身保留
  且 status 不变), 请求参数回显, 产物路径, error.
- error: null 或结构化 `{code, message}` (对齐 openai SDK Video.error 形态).
  稳定 code 集: `worker_crashed` (非零退出), `worker_stalled` (停滞超时),
  `job_timeout` (单次运行超时), `memory_lease_exceeded` (watchdog 足迹超租约),
  `monitor_failed` (连续 3 次足迹读 0), `server_restarted` (启动回放),
  `output_invalid` (exit 0 但 mp4 健康校验失败). worker 的 failure manifest
  用同一 {code, message, detail?} schema, manager 透传.

队列与时钟:

- FIFO + asyncio.Semaphore(1); 队列深度上限 (settings, 默认 4), 超限提交
  直接 503. 一次只有一个 worker 子进程.
- 内存准入只在 dispatch (spawn 前) 评估 (判据见 §4.4): 不满足 -> job 留在
  队头, 乘 enforcer 1s tick 节奏每 ~5s 重查, 永不对已接收的 job 503.
  饱和的 LLM 服务可以让视频 job 长等 -- 这是接受的取舍 (§9), 用户可 DELETE
  取消排队中的 job.
- job_timeout_seconds (默认 7200) 的时钟从 worker spawn 起算 (per-run),
  排队等待不计时. 停滞超时见下.

持久化与产物:

- 每 job 一个 JSON, 原子写 (tmp+replace, 仿 responses_utils.py:447-454),
  目录 {base_path}/video-jobs/; 产物 {base_path}/video-artifacts/{job_id}/.
  启动时回放: in_progress/queued 的标记为 failed (code=server_restarted).
- 保留策略: 数量 + 总字节双上限, LRU 清产物; 清除只删 blob 并置 expires_at,
  job 记录保留 (list 与 GET 仍可见历史).

worker (omlx/video/worker.py, 只 import mflux + mlx + 标准库, 不 import omlx):

- spawn 形态: `<video_venv>/bin/python -I <omlx>/video/worker.py --spec
  <job_spec.json>`. `-I` (isolated) 隔离 sys.path/PYTHONPATH/用户 site,
  防 worker 误 import 主仓 omlx; env 由 manager 白名单构造 (PATH, HOME,
  TMPDIR + 刻意选择的 HF 变量), 不整体继承.
- spec 内 model dir 只能取自 registry entry.model_path (discovery 扫描产物,
  server 自有根目录下); request.model 字符串在任何分支都不得参与路径构造,
  resolve 失败一律 404.
- 进场顺序: 先 `mx.set_wired_limit(lease_bytes - wired_margin)` (lease 经
  spec 传入; mlx 本来就是 mflux 依赖, 不破 "不 import omlx" 规则), 再加载
  模型. 这是 wired-sum 治理的承重墙 (§4.4).
- 进度协议: stdout 每行一个 JSON. 两类行: 相位转换心跳
  `{"phase": "loading"|"text_encoding"|"denoise"|"vae_decode"|"saving"}`
  (静默长相位 -- 42GB 权重加载/torch 文本编码/VAE decode -- 也有活性信号)
  与步进行 `{"phase": "denoise", "step": n, "total_steps": m}` (接
  ProgressCallback).
- 结束: video.save(输出路径, validate_health=True) + metadata sidecar;
  异常时写 failure manifest JSON 后 exit 非零.

监控与终止 (manager 侧, 统一在 2s watchdog tick 里):

- 足迹: get_phys_footprint(worker_pid); 连续 3 次读 0 -> 杀,
  code=monitor_failed; 足迹 > lease -> SIGKILL, code=memory_lease_exceeded.
  注意 watchdog 定位是次级清理/泄漏检测, 不是 panic 兜底 (§2.3, §4.4).
- 停滞: 追踪 last_jsonl_line_at, in_progress 且静默超过
  progress_stall_timeout_seconds (settings, 默认 600) -> SIGTERM -> 5s ->
  SIGKILL, code=worker_stalled. 相位心跳的存在使该阈值在生产分辨率
  (单步可 ~70s+) 下既不误杀也不失效.
- 单次运行超时 job_timeout_seconds 同终止路径, code=job_timeout.
- DELETE 取消: SIGTERM -> 5s -> SIGKILL, 释放租约, 删记录与产物.

mlx-gen 演进风险的真实缓解次序 (v1 的 "CLI 兜底" 评审降级): 第一道 = 锁定
依赖集的精确 pin (§4.5 lockfile) -- 依赖冻结后 API/CLI 都不会在运行期破裂,
破裂只能经显式升级 PR 进来, 是可 review 的代码变更; 第二道 = vendor wan
子树 (MIT 允许; 真实规模约 130 文件含 models/wan + models/common + utils +
callbacks, 且 torch/transformers 依赖不因 vendor 消失 -- 诚实代价见 §9);
CLI 形态切换只是第三道且会破坏 JSONL 进度协议, 不作为设计依赖.

### 4.3 /v1/videos API (OpenAI 形态)

路由文件 api/video_routes.py. 挂载: 无条件 include_router (mcp_router
先例, server.py:435-437) -- 不能用 audio 的条件挂载范式, 因为那发生在
import 时而 settings 彼时未初始化 (§2.2). 门控全部在 handler 内:
settings.video.enabled 为 false 或 manager 未初始化 -> 503 + 指引;
venv 探测失败 -> 503 + 安装指引 (指引文本用 §4.5 修正后的命令).
router 级 Depends(verify_api_key).

| 端点 | 行为 |
|---|---|
| POST /v1/videos | 见下方提交语义 |
| GET /v1/videos | MVP 必做 (LRU 清产物后这是唯一枚举手段). 参数 limit (默认 20, 上限 100) / after (游标 = job id) / order (asc|desc, 默认 desc, 按 created_at). 响应 {"object": "list", "data": [...], "has_more": bool, "last_id": ...} -- openai SDK 游标分页所需字段 |
| GET /v1/videos/{id} | job 对象 (status, progress, phase, error, expires_at, ...) |
| GET /v1/videos/{id}/content | FileResponse mp4 (media_type=video/mp4, 支持 Range); 未完成 -> 409; completed 但产物已被保留策略清除 -> 404 + code=artifact_expired (响应体指向 expires_at); handler 必须先查文件存在 (FileResponse 对缺失路径会 500) |
| DELETE /v1/videos/{id} | queued/in_progress: 杀 worker (SIGTERM->5s->SIGKILL) + 释放租约 + 删记录与产物; completed/failed: 删记录与产物. 返回 {"id", "object": "video.deleted", "deleted": true} (openai SDK VideoDeleteResponse 形态); 之后 GET -> 404 |

提交语义 (POST /v1/videos):

- 兼容要害 (评审 blocker): openai SDK 的 client.videos.create 发送的是
  multipart/form-data (为 input_reference 文件域), 纯 JSON pydantic body
  会对官方 SDK 一律 422. handler 收原始 Request, 按 Content-Type 分支:
  multipart -> await request.form(); JSON/缺失 -> await request.json();
  两路归一进同一个内部 pydantic 模型 (video_models.py 保留). FastAPI 不能
  按 content type 在同路径派发两个 handler, 必须单 handler 手工解析 --
  与 audio_routes 的 pydantic-body 范式刻意不同, 此处注明原因.
- 字段: model, prompt, 可选 size "WxH", seconds (SDK 发的是字符串字面量
  "4"|"8"|"12"; multipart 下所有字段都是字符串, 数值字段必须走 pydantic
  lax 强转), 以及 fmlx 扩展 negative_prompt/steps/fps/seed/guidance/
  guidance_2 (扩展字段碰撞政策: 若未来 OpenAI 占用同名字段, fmlx 语义让位,
  扩展迁移到 fmlx_ 前缀; MVP 不预先加前缀).
- seconds 按 fps 折算帧数, 强制 4n+1; size 向上取整到 16 的倍数.
- 准入即拒 (400/413): 参数越静态上限 (max_frames/max_steps/max_pixels,
  settings); 或按 §4.4 的逐请求峰值预测器 predicted_peak(W,H,frames) +
  margin > lease -- 响应体带 预测值 vs lease 数字. 静态上限是 UX 边界,
  预测器才是内存边界.
- 接受 -> 立即返回 job 对象 (status=queued).
- 503 仅三种: 队列满 / venv 缺失 / 内存 guard 关闭 (均为提交时点的持久性
  条件, 带可操作原因). 内存紧张不 503, 进队等 (§4.2).

错误映射: 模型不存在 404 (带 available 提示); 模型非 video 类型 400.
每请求计入 metrics (record_request_complete, 0 token, 仿 audio_routes.py:
426-436).

### 4.4 内存共驻: 三层治理 (wired 自缚 / 租约 / watchdog)

第一层, 预防 (承重墙): Metal wired limit 把两个进程各自钉死.

- worker 进场即 `mx.set_wired_limit(lease_bytes - wired_margin)` (§4.2).
  越限退化为非常驻页 (变慢) 或分配失败 (job 失败, manager 上报 failed) --
  永不向机器 cap 方向增长. 与 enforcer 对父进程的现有做法同机制
  (process_memory_enforcer.py:407-446).
- acquire_video_lease 时父进程把自身 wired limit 重设为 (static_ceiling -
  lease), release 时恢复. 若父进程在驻 wired 已超新限 (如 85GB 模型在载),
  MLX 退化为非常驻页 (decode 变慢) 而非 panic -- 可接受, 且 §4.4 准入判据
  使该情形罕见.

第二层, 预算 (租约, 改 process_memory_enforcer.py 约 50 行):

- `_video_lease_bytes` 在 `_get_hard_limit_bytes` (495-517) 末尾扣减:
  `ceiling = max(0, ceiling - lease)`. 单一咽喉点, pool 准入/软硬水位/
  admission_paused/prefill gate cap 下一个 1s tick 全部自动收紧, 零
  scheduler 改动.
- 双重计数修正 (评审 major): 动态 ceiling (483-493, 非 custom 档) 会因
  worker 占用经 free 下降而再降一次. 修正: 非 custom 分支加回
  `min(get_phys_footprint(worker_pid), lease)` -- worker 被精确计一次.
  clamp 到 lease 保证失控 worker (watchdog 杀掉前的窗口) 不会反向抬高
  父进程 ceiling; 足迹读 0 时退化为今天的双重计数, 即 fail-conservative.
- API: `acquire_video_lease(bytes)` (spawn 前, 此时无 pid, 加回项为 0,
  正确 -- 尚未分配), `set_video_worker_pid(pid)` (spawn 后立即),
  `release_video_lease()` (清两者). 改值即 `_propagate_memory_limit()`
  (现有 runtime setter 范式, 372-400).
- guard 关闭 (get_final_ceiling()==0) -> 拒绝视频任务 (提交时 503, §4.3),
  不在无防护机器上引入 panic 源.

dispatch 准入判据 (评审修正: 不得在 "落租约即触发硬压力" 的窗口里放行):

- enforcer 存在且 guard 启用, 且 `recent_peak_bytes() <= min(
  soft_ratio * (ceiling - lease), (ceiling - lease) - prefill_transient_
  margin)` -- 用滚动峰值而非瞬时值, 且要求落租约后系统直接处于 "ok 压力 +
  在驻负载不触 prefill gate" 的状态. 不满足 -> 留队重查 (§4.2).
- 在途长 prefill 的残余情形 (判据通过后, 租约落地前才进来的增长型负载):
  租约落地使 gate cap 收紧, 该 prefill 的下一个 chunk 被 gate 干净拒绝
  (503 类错误, 无 panic) -- 这是设计内行为, 记入 §9 取舍表. MVP 不做
  drain (等 prefill 排空再落租约), P2 视实测再议.
- 与超大模型的互斥算术 (评审 blocker 的修正): 107.5 ceiling - 28 lease =
  79.5, glm4.5 (85GB 权重) 根本放不进 -- 即设计上 video 与 >=80GB LLM
  互斥, job 排队直到大模型被 TTL/手动卸载. 真共驻的适用域是 "LLM 权重 +
  工作集 <= ceiling - lease - 余量", 128GB 机上约 <=50GB 级模型. §1 目标 4
  与 §7 A/B 协议均按此表述.

第三层, 清理 (watchdog, §4.2): 2s 足迹轮询超租约杀 + 停滞杀 + 超时杀.
定位是泄漏检测与次级清理 -- sub-2s 的 wired 冲刺由第一层挡, 不靠它.

lease 大小: settings.video.memory_lease_gb, 默认初值 28, 由 P0 实测校准
(§7: 用 lifetime-max ledger 测真峰 + 最差单步瞬时); 校准值与依赖 lock
digest 绑定 (§4.5, §9.1).

### 4.5 settings (VideoSettings 新 section)

按四件套范式接 (settings.py:789-817 / 879-912 / 1136-1154 / 1376-1397) +
admin GET/POST + GlobalSettingsRequest 平铺字段 + _settings.html 表单.

| 字段 | 默认 | 说明 |
|---|---|---|
| enabled | false | 总开关; false 时 handler 一律 503 (路由仍挂载, §4.3) |
| worker_python | "" | video venv 的 python 路径; 空 = {base_path}/venvs/video/bin/python |
| memory_lease_gb | 28 | P0 实测后校准, 与 lock digest 绑定 |
| max_queued_jobs | 4 | 超限提交 503 |
| job_timeout_seconds | 7200 | 单次运行超时 (spawn 起算), 排队不计 |
| progress_stall_timeout_seconds | 600 | JSONL 静默杀 (§4.2) |
| default_steps / default_fps | 20 / 16 | 未显式给参时的生成默认 (P0 校准) |
| max_frames / max_steps / max_pixels_per_frame | 121 / 50 / 1280x720 | 请求 UX 上限; 内存边界由峰值预测器把守 (§4.3/§4.4) |
| artifacts_max_count / artifacts_max_gb | 50 / 50 | 产物保留 (LRU 清 blob, 记录保留) |

venv 管理 (评审 blocker 修正: v1 的裸命令会从仓库 cwd 装进生产主 venv):

- 锁定: 仓库提交 `omlx/video/requirements.in` (一行 `mlx-gen==0.18.14`) 与
  `omlx/video/requirements.lock` (`uv pip compile --generate-hashes`, 必须在
  macOS arm64 + 与 worker venv 相同 Python minor 上生成 -- mlx 只有 darwin
  轮子, 理想在 m5max 上生成).
- 创建 (文档化命令, 也是 503 指引文本):

```
uv venv -p 3.12 {base_path}/venvs/video
uv pip sync --python {base_path}/venvs/video/bin/python omlx/video/requirements.lock
```

- 警告: 裸 `uv pip install` 按 VIRTUAL_ENV / 最近 .venv 解析目标, 从仓库根
  执行就是生产 fmlx venv -- 该形态永远不得用于此用途.
- 启动探测: 跑 `<worker_python> -c "import mflux"`, 且断言 worker_python
  与主进程 sys.executable 不是同一解释器 (防误配); 失败 -> 提交一律 503
  带安装指引. admin 一键安装是 P2.

### 4.6 admin 面 (MVP 最小)

- 模型列表自动获得 video 条目 (get_status 透传 model_type, 零改动);
  类型下拉 (_modal_model_settings.html:272-280) 加 video 选项; 删除可用
  (§4.1 的门放宽).
- job 可见性 MVP 靠 GET /v1/videos (已升必做) 与日志; admin 视频页 P2.

### 4.7 与下载链路的关系 (顺带修复, 建议拆独立小 PR)

- HF 下载器对 diffusers repo 零改动可用 (snapshot_download 全树落
  <model_dir>/<owner>/<repo>, on_complete 触发再发现).
- 中国网络 Xet 墙: `HF_HUB_DISABLE_XET` 在 huggingface_hub import 时冻结,
  只能进程级注入 -- 加到 cli.py:115-140 的 serve 启动 env 块, 由
  settings.huggingface.disable_xet 驱动 (默认 false, 文档建议国内开).
  本次 42GB 下载即是被 Xet 卡死 6.5 小时, 换 LFS 链路后 8.8MB/s 拉完.
- ModelScope 下载视频模型 MVP 明确不支持 (flat symlink 触发幽灵组件,
  §4.1 的发现修复使其不再产幽灵, 但 MS 路径的正式支持等 P2).

## 5. 文件清单

| 路径 | 新/改 | 预估 LOC | 内容 |
|---|---|---|---|
| omlx/video/__init__.py | 新 | 10 | 导出 |
| omlx/video/manager.py | 新 | ~650 | job 模型/队列/持久化/spawn/watchdog/停滞/租约/保留策略 |
| omlx/video/worker.py | 新 | ~180 | 子进程脚本 (wired 自缚 + JSONL + manifest), 只依赖 mflux/mlx |
| omlx/video/requirements.in + .lock | 新 | -- | 依赖锁 (§4.5) |
| omlx/api/video_routes.py | 新 | ~300 | 5 端点 + 双 content-type 解析 + 门控 |
| omlx/api/video_models.py | 新 | ~110 | pydantic 内部模型/响应/error code 枚举 |
| omlx/model_discovery.py | 改 | ~60 | _is_model_dir / detect_model_type / 注册臂 / Literal |
| omlx/engine_pool.py | 改 | ~30 | Literal / 映射 / get_engine 入口拒绝 + 新异常 / _load_engine 防御臂 |
| omlx/server.py | 改 | ~40 | 路由挂载 / pre-pool 400 / 异常映射臂 / 默认模型与 fallback 卫生 / ModelInfo.model_type / manager 构造接线 |
| omlx/process_memory_enforcer.py | 改 | ~50 | 租约三 API + 扣减 + 动态 ceiling 加回 + 父进程 wired 重设 |
| omlx/settings.py | 改 | ~120 | VideoSettings 四件套 |
| omlx/admin/routes.py | 改 | ~45 | valid_types / 映射 / DELETE 与列表门 / global-settings / 默认 setter 拒绝 video |
| omlx/cli.py | 改 | ~6 | disable_xet env 注入 |
| templates/static | 改 | ~20 | 类型下拉 + settings 表单 |
| tests/ (多文件) | 新 | ~1500 | 见 §7 |

合计新增约 1.25k (业务) + 1.5k (测试), 修改约 370, 分布在 8 个上游同源
文件的小补丁 (§10).

## 6. 初始默认值

生成参数默认 (服务端兜底, 客户端可覆盖, UX 上限受 settings 钳制, 内存
边界由预测器把守): size 480x272 (16 倍数), seconds 3 (按 fps=16 折 49 帧,
4n+1), steps 20, guidance 4.0 / guidance_2 3.0 (mlx-gen A14B 模型默认),
seed 随机. 预期单 job 时长: 按官方 480x240x101 帧 25 步约 30 分钟外推,
默认参数应落在个位数分钟 -- P0 实测后把默认调到 "5 分钟内出片" 的档位.

## 7. 测试计划

单测 (CI 无 GPU, 全部不碰真权重):

- discovery: diffusers 布局 fixture (空权重文件) -> 认出 video / 不产幽灵
  组件 / 未知 pipeline 跳过 + log / flat 与 owner-repo 两种布局 / video
  模型按字典序第一时不成为默认模型.
- 类型映射一致性断言 (三份映射 + valid_types 同步).
- 加载拒绝: pool.get_engine 对 video 条目零驱逐零加载直接 typed 异常;
  server 侧 chat/embeddings/load 端点 400 + 指引.
- manager 状态机: 提交/排队/取消/超时/停滞/worker 非零退出/manifest 透传/
  启动回放 (code=server_restarted), worker 用假脚本 (输出 JSONL + touch
  mp4) 替身; enforcer 经构造注入假实现 (§4.2 接线即测试缝).
- 并发与竞态: asyncio.gather 多提交 -> 恰一 running + 队列上限 503, 永不
  双 worker; watchdog 足迹读 0 路径; worker 退出与 watchdog tick 竞态.
- 租约: acquire/release 对 ceiling 的影响, 动态 ceiling 加回 clamp,
  guard 关闭拒绝, 准入判据 (滚动峰值口径).
- API: 双 content-type 解析 (multipart 字符串字段强转 + JSON), schema/
  错误码/Range 下载/越限 400/预测器 413/list 分页游标/expires_at 与
  artifact_expired/DELETE 全状态语义.
- 保留策略: 数量与字节双上限 LRU, blob 删除后记录可见且 expires_at 置位.
- 回归保护: /v1/models payload 增量字段不破坏现有断言 (排查既有精确匹配
  测试), settings 章节枚举类测试同步.

P0 真机测量 (m5max, 无 fmlx 代码, 用户 go-ahead 后执行):

- 仪器: 外部足迹轮询曲线 (相位归因, 接 JSONL 流) + 内核 lifetime-max
  ledger (`ri_lifetime_max_phys_footprint`, proc_memory.py:63 已声明未读,
  加一个 get_phys_footprint_lifetime_max 变体在 video.save 返回后, 进程
  退出前自读) -- worker 每 job 一进程, lifetime max == 含加载/VAE/全部
  sub-poll 尖峰的真峰值. 轮询曲线只定相位形状, 真峰值用 ledger.
- 测量矩阵: 默认档 (480x272, 49f) / 中档 (832x480, 81f) / 上限角
  (1280x720, 121f) / 一个 steps 变体 (验证 steps 不动峰值). 拟合
  peak ~= W + a*latent_tokens (若非融合 SDPA 注意叠加二次项), 同时记录
  每档最差单步瞬时 (sub-poll delta) -- 它而非稳态峰决定 lease 内 margin
  (settings.py:404 方法论).
- 产出: lease 默认值 + 峰值预测器系数 (回填 §4.3/§4.4/§4.5) + 默认参数
  档位 + lock digest 绑定记录.

P1 真机 A/B (评审修正: v1 的 "glm4.5 共驻" 算术不可能, 85+28 > 107.5):

- 场景 A, 互斥语义 (glm4.5 85GB): 大模型在载且保持活跃 (pin 或持续流量,
  防 TTL 中途卸载), POST /v1/videos. 断言: job 停留 queued 且 GET 可见
  内存原因; 无 worker 进程 (ps 验证); 零 OVER_HARD; 整机存活. 然后卸载
  glm4.5, 断言 job 在 ~2 个 enforcer tick 内转 in_progress. 测试用短超时,
  不用 2h 默认.
- 场景 B, 真共驻 (<=50GB 级模型, 如 gemma4-26b 量化档): 视频 job 运行中
  发 LLM 长 prefill. 断言: prefill gate 在 (ceiling - lease) 收紧后的 cap
  下干净拒绝或正常完成 (按预算算术预期), admission pause 行为符合水位,
  零 OVER_HARD, 零 panic; 视频 job 正常完成且 mp4 健康.
- 场景 C, 释放与恢复: job 结束 (完成与 DELETE 两路) 后断言租约释放,
  父进程 wired limit 恢复, LLM 满额服务恢复, 产物可 Range 下载.
- 回归: 完整 pytest 套件零回归 (基线见 docs/upstream-sync.md).

## 8. 阶段划分

- P0 真机测量 (先行, 零集成代码): §7 P0. 产出校准数据回填本 spec.
- P1 MVP: §4 全部 + §7 单测 + §7 A/B 三场景. 单分支 feat/video-engine,
  人审人合.
- P2 (按需排期): admin 视频页 + SSE 进度, I2V (图上传), TI2V-5B 与 bf16
  变体, 文生图 (FLUX 系同运行时, /v1/images), 常驻 worker + idle TTL,
  per-model 生成默认, ModelScope 正式支持, admin 一键装 venv, 视频任务
  主动驱逐 LLM 的策略, drain 式租约落地.

## 9. 风险与对策

| 风险 | 等级 | 对策 |
|---|---|---|
| mlx-gen 高速 0.x 演进 + bus factor 1 (twine 混 runtime deps 的卫生信号) | 高 | 第一道: hash 锁全依赖集 (§4.5), 破裂只能经升级 PR 进来; 第二道: vendor wan 子树 (诚实规模 ~130 文件, torch/transformers 不因 vendor 消失); CLI 切换仅第三道. 升级程序见 §9.1 |
| 生产分辨率内存未实测 (官方数是小 profile, 上限角约 39x 测点 latent 量) | 高 | P0 测量矩阵 + lifetime-max ledger 定真峰; 逐请求峰值预测器把内存边界从静态 caps 解耦 (§4.3); worker wired 自缚保底 |
| 双进程 Metal wired-sum 越机器 cap | 中 | 预防层: worker 进场 wired 自缚 + 父进程 acquire 时 wired 重设 (§4.4 第一层); watchdog 仅作次级清理; A/B 场景 B 专项验证 |
| 租约落地瞬间触发硬压力误伤在途 LLM 请求 | 中 | 准入判据用滚动峰值且要求落地后即处 ok 压力 (§4.4); 残余: 落地后才增长的在途 prefill 被 gate 干净拒绝 (设计内, 无 panic) |
| 与 >=80GB LLM 互斥导致视频 job 长等 | 低 | 设计内取舍, 排队原因对 GET 可见, 可 DELETE; 主动驱逐策略 P2 |
| worker 卡死 (不出步进也不退出) | 中 | 相位心跳 + progress_stall_timeout_seconds 停滞杀 + 单次运行超时, 双层杀 (§4.2, 已入 settings 与状态机) |
| 队列任务跨重启丢失 | 低 | 持久化 + 启动标记 failed (code=server_restarted), 不静默消失; MVP 不做断点续跑 |
| 产物盘占用 | 低 | 双上限 LRU 清 blob, 记录保留 + expires_at |
| settings 旧版本降级丢字段 | 低 | 已知 from_dict 行为, 文档注明 |

### 9.1 升级与依赖漂移程序

1. 锁整个 venv 而非顶层包: requirements.lock (hash) 进仓, venv 创建/重建
   一律 `uv pip sync`. 一切依赖变更 (顶层 bump 或传递漂移) 必须经 PR.
2. 每次 lock 变更的合并门: 在 m5max 重跑 P0 测量矩阵 (至少默认档 + 上限角),
   PR body 携带数字 (真峰/最差瞬时); 新峰值 + margin 逼近在配 lease 时,
   同 PR 重校准 memory_lease_gb 与预测器系数.
3. lease 默认与预测器系数的有效性与 lock digest 绑定 (spec 与 settings
   注释双处记 digest); digest 不匹配时启动 log warning.
4. 输出质量回归: 升级 PR 附固定 seed 的 golden 短片对比 (人工目检即可,
   MVP 不做自动指标).

## 10. 与上游 soft-fork 的关系

本功能是 fmlx 自有分化, 永不回流. 冲突面控制策略: 业务全部在新文件
(omlx/video/, api/video_*), 对上游同源文件只做小而可 grep 的补丁
(§5 清单中 8 个 "改" 文件, 约 370 行, 其中 cli.py/templates 各 <=20 行).
上游 cherry-pick 撞到这些文件时, 冲突块小且语义独立, 解决成本可控.
docs/upstream-sync.md 记一条分化标记.

## 11. 待拍板的未决问题

1. lease 默认 28GB / 预测器系数 / 默认生成参数档位 -- P0 实测后回填,
   无需现在拍.
2. settings.video.enabled 默认 false (需手动开启) vs 默认 true (venv 缺失
   时 503 指引) -- 倾向 false, 灰度心智.
3. Xet 修复 (§4.7) 是否拆独立小 PR 先行 (与视频无耦合, 运维价值即时) --
   倾向拆.
4. 真共驻适用域的产品表述: 文档要不要给出 "128GB 机建议 <=50GB LLM 与
   视频并用" 的明确指引 -- 倾向给, 写进 README 视频章节.
