# fmlx 图像生成引擎 spec (/v1/images, mlx-gen 运行时)

状态: 实装中 (feat/image-engine, 2026-06-11). 定位: Flyto MLX 自有功能,
不回流上游 (soft-fork 自有分化). 本引擎是视频引擎
(docs/video-generation-engine-spec.md) 的 P2 落点之一, 与其同构同运行时:
凡本文未覆盖的机制 (租约算术, watchdog 语义, venv 纪律, 拒绝臂位置,
持久化与保留策略) 一律以视频 spec 为准.

## 0. 模型与运行时

三个模型, 全 Apache 2.0, 全部 mlx-gen 0.18.14 (mflux) 原生支持,
AbstractFramework 4bit 量化, 落在 `~/.fmlx/models/AbstractFramework/`:

| 目录 | mflux 别名 | pipeline | 权重 | 用途 |
|---|---|---|---|---|
| z-image-turbo-4bit | z-image-turbo | t2i | 5.9GB | 快速文生图默认 (~9 步) + 强度重绘 |
| qwen-image-2512-4bit | qwen-image | t2i | 17.4GB | 中文排版/细节天花板 (40 步) |
| qwen-image-edit-2511-4bit | qwen-image-edit-2511 | edit | 18.3GB | 指令编辑/多图参考/图内改字 |

mlx-gen 关键事实 (源码核实, 2026-06-11):

- 图像类没有 `progress_callback` kwarg (那是 Wan 视频专属); 进度走
  `model.callbacks.subscribe_progress(cb)`, ProgressEvent 带
  phase/step/total_steps, 订阅本身会触发逐步 mx.eval.
- `ModelConfig.from_name(alias)` 是唯一安全的解析路径 --
  qwen-image-edit-2511 没有静态工厂方法.
- 保存布局目录 (transformer/ + text_encoder/ + vae/, 无 model_index.json)
  没有任何类标记; 初始化器也不做交叉校验, 错配会晚爆在权重应用 --
  所以发现层用目录名启发式定 kind, worker 进场先跑组件级 preflight.
- 多图参考编辑只有 edit-plus 配置 (2509/2511) 支持, Python API 不守卫
  (CLI 才守卫) -- 路由层补 400.
- z-image-turbo 强制 guidance=0 且忽略 negative_prompt
  (supports_guidance=False).
- LoRA: HF collection 语法 `org/repo:file.safetensors`; 裸 repo id 在
  多 safetensors 仓 (如 lightx2v/Qwen-Image-Lightning) 直接 ValueError.
  worker 跑在 HF_HUB_OFFLINE=1 下, LoRA 必须预下载进 MFLUX_CACHE_DIR.
- 不用 mflux MemorySaver (--low-ram): 它在首次 prompt encode 后删
  text encoder, 模型实例单用 -- n>1 会炸. worker 自己做
  mx.set_cache_limit(1GB) + TilingConfig (VAE tiling, SeedVR2 的教训).

## 1. 发现层

结构检测 `is_image_model_dir` (model_discovery.py): transformer/ +
text_encoder/ + vae/ 三个 model.safetensors.index.json 同时存在, 且无根
config.json/model_index.json, 且无 scheduler/ 或 transformer_2/ 目录
(屏蔽 model_index.json 尚未落地的半下载 Wan 目录). 与
is_video_upscaler_dir 互斥天然成立 (upscaler 臂排除 text_encoder).

kind 与别名由 `read_image_model_kind` 从目录名启发式得出 (先例:
_is_causal_lm_reranker). 名字映射不到已知 mflux 别名的目录整体跳过
(不注册, log warning, 不产幽灵). model_type 统一 "image", entry 携带
image_pipeline ("t2i"|"edit") 与 image_alias, config_model_type = 别名.

拒绝臂与视频同构: pool.get_engine 准入循环之前 typed 拒绝 +
_load_engine 防御臂 + server.get_engine pre-pool 400 + load 端点 400 +
admin valid_types/type_to_engine 同步. ModelTypeNotLoadableError 提示
"Use POST /v1/images.".

## 2. job 管理: 共用 MediaJobManager

视频引擎的 VideoJobManager 泛化为 MediaJobManager (保留兼容别名),
一条 FIFO + 一个 dispatcher 调度两种 kind. 这不是省事 -- enforcer
同时只允许一个租约 (acquire_video_lease 被持有时 raise), 双 manager
会在 admission 通过后竞态硬失败. 单队列让图像与视频任务对唯一租约
天然串行, 也符合单机一次一个 diffusion 的内存现实.

per-kind 差异全部参数化: jobs/artifacts 目录 ({base}/image-jobs,
{base}/image-artifacts), worker 脚本 (omlx/image/worker.py), 超时与
队列上限 (ImageSettings), 租约 (per-job lease_bytes, 路由按别名定),
进度带 (n 张均分 5-95), 产物校验 (manifest.outputs 逐文件非空).
job id 前缀 img_, wire object "image.job". 新增 wait_terminal(job_id,
timeout) 支撑同步请求 (per-job asyncio.Event, _finish/delete 触发).

## 3. /v1/images API

| 端点 | 行为 |
|---|---|
| POST /v1/images | sync 默认 true: 阻塞至完成, 返回 OpenAI images 形态 {created, id, data: [{b64_json}|{url}]}; sync=false: 立即返回 job 对象供轮询 (聊天 UI 用) |
| POST /v1/images/generations | 官方 SDK 兼容别名 (client.images.generate), 恒同步 |
| GET /v1/images | 游标分页 list (只列 image kind) |
| GET /v1/images/{id} | job 对象 |
| GET /v1/images/{id}/content?index=N | 单张 PNG (FileResponse); 未完成 409; 保留策略清除后 404 artifact_expired |
| DELETE /v1/images/{id} | 杀 worker + 删记录与产物 |

请求: model 可省 (自动选型: 带图 -> 第一个 edit 模型, 纯文 ->
z-image-turbo 别名优先), prompt 必填, n (<=max_n, worker 内 seed+i
顺序循环, 权重只载一次), size "WxH"|"auto", response_format
b64_json|url. fmlx 扩展: negative_prompt, steps, seed, guidance,
width/height, image_strength (t2i 重绘), lora_paths/lora_scales.
输入图: multipart 重复 "image" 文件域, 或 JSON "image": str|[str]
(data URL / base64), PNG/JPEG/WebP, 单张 16MB 上限.

校验: edit 模型无图 400; t2i 多图 400; 非 edit-plus 别名多图 400;
steps/max_pixels/n 越界 400; guard 不可用/venv 缺失/队列满 503;
同步超时 504 (job 继续跑, 提示轮询). 尺寸默认: t2i 落
default_size (1024x1024), edit 不传尺寸 (mflux 按首参考图长宽比
~1MP 自适应 -- 正方形强制输出会劣化 qwen-edit, 上游已知).

步数默认 (显式 > per-model image_default_steps > 全局 default_steps >
按别名): z-image-turbo 9, z-image 28, qwen* 40.

## 4. 内存

租约按别名预设, 已经 m5max 实测校准 (M5 Max 128GB, mlx-gen 0.18.14,
2026-06-11, worker lifetime-max phys manifest, 1024x1024):

| 别名 | 实测真峰值 | 租约 | 用时 |
|---|---|---|---|
| z-image-turbo (9 步) | 8.6GB | 12GB | 22.4s |
| qwen-image (40 步) | 21.0GB | 26GB | 312s |
| qwen-image + Lightning LoRA (8 步) | 21.9GB | 26GB | 73s |
| qwen-image-edit-2511 (40 步, 1MP) | 22.0GB | 26GB | 885s |

峰值对步数不敏感 (权重 + 工作集主导), ~4GB pad 覆盖瞬时;
settings.image.memory_lease_gb > 0 时全局覆盖. mlx-gen lock 升级必须
重测 (视频 spec 9.1 纪律). worker 进场 mx.set_wired_limit(lease - 2GB),
watchdog/停滞/超时语义与视频完全一致. 图像无峰值预测器 (MVP); 静态
caps (max_pixels 默认 2048x2048) + 租约 + wired 自缚 + watchdog 四层
够住. 共驻算术: 107.5 ceiling - 26 lease = 81.5GB 留给 LLM.

8bit 量化明确不用: edit-2511-8bit 权重 30.3GB, 租约下激活放不下.
Lightning LoRA 实测可用 (V1.1-bf16, 8 步 guidance 2.5, 4.3x 提速,
质量目检可接受), 是 qwen 延迟的主要解药; edit-2511 全 40 步要
~15 分钟, 同步调用方注意 sync_timeout (默认 900s 勉强够).

## 5. settings

ImageSettings (settings.image): enabled=false, worker_python (空 =
{base}/venvs/video/bin/python, 与视频同 venv -- mlx-gen 同包, 无需第二
份锁), memory_lease_gb=0 (auto), max_queued_jobs=8,
job_timeout_seconds=1800, progress_stall_timeout_seconds=300,
default_steps=0 (auto), default_size="1024x1024", max_steps=60,
max_pixels=2048x2048, max_n=4, sync_timeout_seconds=900,
artifacts_max_count=200, artifacts_max_gb=10.

per-model: image_default_steps / image_default_size (三件套:
ModelSettings + modal + MODEL_SPECIFIC_PROFILE_FIELDS).

## 6. 聊天 UI

picker 放行 image 类型, 徽标按 image_pipeline (文生图/图生图). 发送
路由: 选中图像模型时, 带图自动路由 edit 模型, 纯文走 t2i (镜像
resolveVideoModel). 提交 sync=false + 轮询进度, 完成后 content 端点
取 blob 渲染图片气泡; msg._image 状态机与 msg._video 同构 (blob URL
不落 localStorage, loadChat 按 job_id 重取, artifact_expired 显示
占位). i18n 全部扁平键.

## 7. 真机验证纪律 (合并门)

- 三模型各出一张: 中文海报字测 z-turbo 与 qwen-2512, 改字测 edit-2511.
- 记录 lifetime-max 真峰值 (manifest 自带) 回填 §4 租约表.
- Lightning LoRA 冒烟 (lightx2v/Qwen-Image-Lightning collection 语法,
  8 步 + guidance 2.5, 预下载后挂 lora_paths).
- qwen-image-2512 权重 x mflux "qwen-image" 配置兼容性属未验证假设,
  首测优先.
- 全量 pytest 零回归 (基线 docs/upstream-sync.md).

## 8. 已知取舍

- 图像与视频共队列: 图像任务可能排在长视频任务后面 (分钟-小时级).
  设计内 -- 单租约约束下无并行可言; 排队原因经 job.phase 可见.
- mlx-gen 比 ComfyUI/MPS 慢 2-3x (上游 mflux#338, 速度非质量).
- z-image 量化版中文字渲染无公开评测; 文字向不行换 8bit
  (z-image-turbo-8bit 11GB 也轻).
- canvas_policy 不显式传 (MVP): 带输入图时 mflux 默认 source-aspect,
  恰是想要的行为; 显式 exact-resize 等真机验证后再暴露.
