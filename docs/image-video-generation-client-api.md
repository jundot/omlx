# Image & Video Generation Client API

Integration reference for callers driving flyto-mlx (fmlx) as a local image
and video generation backend. Covers the three pipelines that matter for
cross-shot / cross-frame consistency work: image edit (qwen-image-edit),
image text-to-image (qwen-image / z-image), and image-to-video (wan2.2-i2v).

Every field name, limit and behaviour below is taken from the server code
(`omlx/api/image_routes.py`, `omlx/api/video_routes.py`,
`omlx/image/worker.py`). The OpenAPI schema for these endpoints is empty
because the request body is parsed by hand (it accepts both JSON and
multipart); this document is the authoritative request contract.

English version first. Chinese translation follows below.

---

## English

### 1. Conventions (read this first)

**Authentication.** Both header forms work:

- `Authorization: Bearer <key>` (OpenAI SDK style)
- `X-API-Key: <key>` (Anthropic SDK style)

**Model ids.** Pass a registered model id (or alias) in the `model` field.
Always confirm the exact ids on your target server with `GET /v1/models`.
The set registered on the reference server at the time of writing:

| Pipeline | Registered id |
|----------|---------------|
| image edit | `qwen-image-edit-2511-4bit` |
| image t2i | `qwen-image-2512-4bit`, `z-image-turbo-4bit` |
| video i2v | `wan2.2-i2v-a14b-diffusers-8bit` |
| video t2v | `wan2.2-t2v-a14b-diffusers-8bit` |

**Sync vs async.** `POST /v1/images` defaults to `sync=true`: the request
blocks until the image is ready and returns the OpenAI image-API shape
(`data[].b64_json`) directly, no polling needed. Pass `sync:false` to get a
job object back immediately and poll it. `POST /v1/videos` is **always
async**: it returns a job immediately and you must poll, then download.

**Single memory lease (serialization).** Image and video jobs share one
job manager and one memory lease. At most one generation runs at a time;
everything else queues. Effective client concurrency is 1 -- submit
serially or tolerate queueing; do not design for parallel throughput.

**Input images are never remote URLs.** Reference / source images travel
as multipart file fields, or as a base64 / data-URL string in a JSON body.
There is no server-side URL fetch -- read the image bytes yourself and send
base64 (or a multipart file).

### 2. Image edit -- qwen-image-edit-2511

This is the pipeline for product lock (keep a subject, change the
background) and character lock (use reference images to keep a
character / object consistent across shots).

#### 2.1 Passing reference images

- Field name is **`image`** (singular, not `images`).
- Multipart: repeat `-F image=@...` for multiple images. JSON: `image` is a
  single string or an array of strings.
- Each value is a multipart file, or a data URL / raw base64 string.
- At most **4** images, each <= 16MB, format PNG / JPEG / WebP.
- Multi-reference (more than one image) requires `qwen-image-edit-2509` or
  `qwen-image-edit-2511`. Single-reference edit models accept exactly one.

#### 2.2 Expressing the two intents -- important limitation

There is **no mask parameter and no denoise / strength parameter** for the
edit pipeline. (`image_strength` exists but is silently dropped for edit;
it only applies to img2img on a t2i model.) There is no pixel-level lock
knob. fmlx does not assign roles to the images either -- it forwards the
ordered list straight to mflux `QwenImageEdit`. The semantics come entirely
from two things:

1. **Image order.** The first image sets the default output aspect ratio
   (about 1MP). Subsequent images are additional references.
2. **The prompt.** Address the inputs with the qwen-image-edit
   `Picture 1 / Picture 2 / ...` convention.

So the two intents are expressed as prompts, not as structured flags:

- **Product lock (change only the background).** Put the product image as
  Picture 1; prompt e.g. `Keep the product in Picture 1 exactly unchanged
  (same shape, label, colour); replace only the background with ...`.
  Pixel-exactness is best-effort (no hard mask) -- pin the `seed` and
  generate several candidates (see 2.3) to pick the most faithful one.
- **Character lock (recompose with reference identity).** Pass the
  character reference(s); prompt e.g. `The character is the person shown in
  Picture 1; place them in <new scene> ...`.

The `Picture N` wording is qwen-image-edit-2511 model behaviour; tune the
exact phrasing against that model card. The API mechanics (ordered `image`
list plus a single `prompt`) are fixed.

#### 2.3 Seed and candidates

- Field **`seed`**, integer, range `[0, 2^31)`.
- For multiple candidates: either call repeatedly with different seeds, or
  send `n` (default cap 4) in one request -- the worker derives per-image
  seeds as `seed, seed+1, seed+2, ...`, giving a batch of consecutive-seed
  candidates from one call.

#### 2.4 Output size / aspect ratio

- Use `size:"WxH"` or explicit `width` / `height` (the explicit pair beats
  `size`). Values round up to a multiple of 16, max `2048x2048`.
- Edit-pipeline specific: when size is omitted, the edit does **not**
  default to 1024x1024 -- it follows the **first reference image's aspect**
  at about 1MP (forcing a square output degrades qwen-edit quality). To get
  a 9:16 vertical output, either send `width=768 height=1344`, or make the
  first reference image itself 9:16.

#### 2.5 curl -- edit with multiple references (multipart, the common case)

```bash
curl -s http://localhost:8000/v1/images \
  -H "Authorization: Bearer $FMLX_KEY" \
  -F model=qwen-image-edit-2511-4bit \
  -F image=@product.png \
  -F image=@scene_ref.png \
  -F 'prompt=Keep the product in Picture 1 exactly unchanged (same shape, label, colour). Replace the background with the bright kitchen counter from Picture 2. 9:16 vertical.' \
  -F width=768 -F height=1344 \
  -F seed=12345 \
  -F n=4 \
  -F sync=true
# returns: {"created":..., "id":"img_...", "data":[{"b64_json":"..."}, ... x4]}
```

#### 2.6 curl -- edit via JSON base64 (async, then poll)

```bash
curl -s http://localhost:8000/v1/images \
  -H "Authorization: Bearer $FMLX_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-image-edit-2511-4bit",
    "image": ["data:image/png;base64,<PRODUCT_B64>", "data:image/png;base64,<REF_B64>"],
    "prompt": "Keep the product in Picture 1 unchanged; restyle background per Picture 2.",
    "width": 768, "height": 1344,
    "seed": 12345,
    "sync": false
  }'
# returns a job -> poll GET /v1/images/{id} for status
#               -> GET /v1/images/{id}/content?index=0 for the PNG bytes
```

### 3. Image text-to-image -- qwen-image / z-image

This is the standard OpenAI `/v1/images/generations` surface plus fmlx
extension fields.

- Standard: `model`, `prompt`, `n`, `size`, `response_format`
  (`b64_json` default, or `url`).
- fmlx extensions: `negative_prompt`, `seed`, `steps`, `guidance`,
  `width` / `height`, `image_strength` (img2img), `lora_paths` /
  `lora_scales`.
- `/v1/images/generations` is the OpenAI SDK compatibility alias and is
  always sync. `POST /v1/images` is the same implementation with the
  sync / async switch.

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer $FMLX_KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen-image-2512-4bit","prompt":"a product hero shot, studio light","n":1,"size":"768x1344","seed":12345}'
# returns OpenAI shape: {"created":...,"data":[{"b64_json":"..."}]}
```

### 4. Image-to-video -- wan2.2-i2v

- First-frame field is **`input_reference`**: a multipart file field, or a
  data URL / base64 string in a JSON body. No remote URL fetch.
- `model` is **required**; `prompt` is required.
- Duration: `seconds` (float) **or** `frames` (`frames` wins). `fps`
  controls frame rate. Frame count is rounded to `4n+1` (Wan requirement).
- Resolution: `size:"WxH"` or `width` / `height` (multiples of 16; default
  `480x272` if omitted).
- Video is always async: `POST` returns a job (`object:"video"`); poll
  `GET /v1/videos/{id}` until `status` is `completed`, then
  `GET /v1/videos/{id}/content` returns the mp4 bytes (Range supported).

```bash
curl -s http://localhost:8000/v1/videos \
  -H "Authorization: Bearer $FMLX_KEY" \
  -F model=wan2.2-i2v-a14b-diffusers-8bit \
  -F input_reference=@first_frame.png \
  -F 'prompt=The character slowly turns toward the camera and smiles, gentle camera push-in.' \
  -F size=480x832 \
  -F seconds=4 \
  -F seed=12345
# returns {"id":"video_...","object":"video","status":"queued",...}
# poll:     GET /v1/videos/{id}          (status: queued -> in_progress -> completed)
# download: GET /v1/videos/{id}/content  -> mp4 bytes
```

### 5. Polling and content retrieval

The job object returned by a submit (or by `GET /v1/images/{id}` /
`GET /v1/videos/{id}`) carries:

- `status` -- one of `queued | in_progress | completed | failed`.
- `progress` -- integer 0..100.
- `phase` -- fmlx extension, a short stage string.
- `error` -- populated on failure (`{code, message}`).
- plus echoed params: `model`, `size`, `prompt`, `seed`, `steps`, and for
  video `frames` / `fps`, for image `n` / `outputs`.

Content endpoints return **raw bytes**, not JSON, not a URL wrapper:

- `GET /v1/images/{id}/content` -> `image/png`. Use `?index=N` to fetch the
  Nth image when `n > 1`.
- `GET /v1/videos/{id}/content` -> `video/mp4` (HTTP Range supported).

If the artifact was purged by retention, the content endpoint returns a
JSON error body with `code: "artifact_expired"` and an `expires_at`.

When `POST /v1/images` runs with the default `sync=true`, you skip all of
this: the response is the OpenAI image shape with `data[].b64_json`. With
`response_format:"url"`, the sync response carries `data[].url` pointing at
the content endpoint (still behind the same API key).

### 6. Cross-shot / cross-frame consistency notes

Directly relevant to keeping a character or product consistent across
shots:

- **`extend_video_id`** (video). Pass a completed video job id; the new i2v
  segment continues from that video's last frame (frame geometry is pinned
  to the source automatically). This is the seam-free way to chain shots.
- **Seed-pinned candidate selection.** With no hard mask available, the
  practical way to lock a subject is a fixed `seed` plus `n` consecutive
  candidates, then score consistency client-side and keep the best.
- **`first_frame_from_text`** (video). For i2v / ti2v models, set this to
  generate the conditioning first frame from the prompt (via the configured
  first-frame model) instead of uploading one. Requires no `input_reference`
  and no `extend_video_id`.

### 7. Field reference

**POST /v1/images** (JSON or multipart):

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | optional; auto-picked by pipeline if omitted |
| `prompt` | string | required |
| `image` | string / string[] / file(s) | input image(s); base64 or data URL in JSON, repeatable file field in multipart; max 4 |
| `n` | int | default 1; cap `settings.image.max_n` (default 4) |
| `size` | string | `"WxH"` or `"auto"` |
| `width`, `height` | int | override `size`; rounded up to 16; given together |
| `response_format` | string | `b64_json` (default) or `url` |
| `sync` | bool | default true; false returns a job to poll |
| `seed` | int | `[0, 2^31)`; per-image seed is `seed+i` |
| `steps` | int | per-model / per-alias default if omitted |
| `negative_prompt` | string | ignored by z-image-turbo |
| `guidance` | float | z-image-turbo forces 0 |
| `image_strength` | float | `(0, 1]`, img2img on t2i only; ignored by edit |
| `lora_paths`, `lora_scales` | string[], float[] | HF refs only (`org/repo:file.safetensors`), no filesystem paths |

**POST /v1/videos** (JSON or multipart):

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | required |
| `prompt` | string | required |
| `input_reference` | file / string | i2v first frame; base64 or data URL in JSON |
| `size` | string | `"WxH"`, default `480x272` |
| `width`, `height` | int | override `size`; rounded up to 16 |
| `seconds` | float | duration; or use `frames` |
| `frames` | int | overrides `seconds`; coerced to `4n+1` |
| `fps` | int | frame rate |
| `steps` | int | denoise steps |
| `seed` | int | random if omitted |
| `guidance`, `guidance_2` | float | classifier-free guidance |
| `negative_prompt` | string | |
| `extend_video_id` | string | continue a completed video from its last frame |
| `upscale_resolution` | int | SeedVR2 per-frame upscale, short side, `480..max` |
| `prompt_extend` | bool | LLM prompt expansion override |
| `first_frame_from_text` | bool | generate the i2v first frame from the prompt |

---

## 中文

### 1. 通用约定 (先看这条)

鉴权. 两种 header 都支持:

- `Authorization: Bearer <key>` (OpenAI SDK 风格)
- `X-API-Key: <key>` (Anthropic SDK 风格)

模型 id. `model` 字段传注册的模型 id 或别名. 用目标服务器的 `GET /v1/models`
复核确切 id. 撰写时参考服务器上注册的集合:

| 管线 | 注册 id |
|------|---------|
| 图像 edit | `qwen-image-edit-2511-4bit` |
| 图像 t2i | `qwen-image-2512-4bit`, `z-image-turbo-4bit` |
| 视频 i2v | `wan2.2-i2v-a14b-diffusers-8bit` |
| 视频 t2v | `wan2.2-t2v-a14b-diffusers-8bit` |

同步 vs 异步. `POST /v1/images` 默认 `sync=true`: 请求阻塞直到出图, 直接返回
OpenAI 图像 API 形状 (`data[].b64_json`), 不需要轮询. 传 `sync:false` 才立即拿
job 去轮询. `POST /v1/videos` 永远异步: 立即返回 job, 必须轮询后再下载.

单一内存租约 (串行). 图像和视频任务共用一个 job manager 和一把内存租约. 同一
时刻最多跑一个生成任务, 其余排队. 客户端有效并发等于 1 -- 串行提交或容忍排队,
不要按并行吞吐设计.

输入图永远不是远程 URL. 参考图 / 源图通过 multipart 文件字段传, 或在 JSON body
里以 base64 / data URL 字符串传. 服务端不做 URL 拉取 -- 自己把图片读成 base64
(或用 multipart 文件).

### 2. 图像 edit -- qwen-image-edit-2511

这是产品锁 (保留主体, 只换背景) 和角色锁 (用参考图保持角色 / 物体跨分镜一致)
的管线.

#### 2.1 多参考图怎么传

- 字段名是 `image` (单数, 不是 `images`).
- multipart: 多个 `-F image=@...` 表示多张. JSON: `image` 是单个字符串或字符串
  数组.
- 每个值是 multipart 文件, 或 data URL / 纯 base64 字符串.
- 最多 4 张, 单张 <= 16MB, 格式 PNG / JPEG / WebP.
- 多参考图 (超过 1 张) 必须用 `qwen-image-edit-2509` 或 `qwen-image-edit-2511`.
  单参考图的 edit 模型只收 1 张.

#### 2.2 两种意图怎么表达 -- 重要限制

edit 管线没有 mask 参数, 也没有 denoise / strength 参数. (`image_strength`
字段存在但对 edit 被静默丢弃, 只对 t2i 的 img2img 生效.) 没有像素级锁定开关.
fmlx 也不给图片分配角色 -- 它把有序的图片列表原样转给 mflux 的 `QwenImageEdit`.
语义完全来自两点:

1. 图片顺序. 第一张图决定默认输出长宽比 (约 1MP), 后续图作为附加参考.
2. prompt 文字. 用 qwen-image-edit 的 `Picture 1 / Picture 2 / ...` 约定指代各张图.

所以两种意图是用 prompt 表达, 不是结构化开关:

- 产品锁 (只换背景). 把产品图作为 Picture 1; prompt 例如 `Keep the product in
  Picture 1 exactly unchanged (same shape, label, colour); replace only the
  background with ...`. 像素不变是尽力而为 (没有硬 mask) -- 固定 `seed` 并出多
  张候选 (见 2.3) 再选最忠实的那张.
- 角色锁 (用参考身份重构图). 传角色参考图; prompt 例如 `The character is the
  person shown in Picture 1; place them in <新场景> ...`.

`Picture N` 是 qwen-image-edit-2511 的模型行为; 具体措辞对着该模型卡微调. API
机制 (有序 `image` 列表加单个 `prompt`) 是固定的.

#### 2.3 seed 与候选

- 字段 `seed`, 整数, 范围 `[0, 2^31)`.
- 出多候选: 多次请求传不同 seed, 或一次请求传 `n` (默认上限 4) -- worker 按
  `seed, seed+1, seed+2, ...` 派生, 一发拿一组相邻 seed 的候选.

#### 2.4 输出尺寸 / 宽高比

- 用 `size:"WxH"` 或显式 `width` / `height` (显式优先于 `size`). 取值向上取整到
  16 的倍数, 上限 `2048x2048`.
- edit 管线特有: 不传尺寸时不会默认 1024x1024 -- 而是跟随第一张参考图的长宽比
  (约 1MP, 强制方形会降质). 要 9:16 竖屏, 要么传 `width=768 height=1344`, 要么
  让第一张参考图本身是 9:16.

#### 2.5 curl -- edit 带多参考图 (multipart, 最常用)

```bash
curl -s http://localhost:8000/v1/images \
  -H "Authorization: Bearer $FMLX_KEY" \
  -F model=qwen-image-edit-2511-4bit \
  -F image=@product.png \
  -F image=@scene_ref.png \
  -F 'prompt=Keep the product in Picture 1 exactly unchanged (same shape, label, colour). Replace the background with the bright kitchen counter from Picture 2. 9:16 vertical.' \
  -F width=768 -F height=1344 \
  -F seed=12345 \
  -F n=4 \
  -F sync=true
# returns: {"created":..., "id":"img_...", "data":[{"b64_json":"..."}, ... x4]}
```

#### 2.6 curl -- edit 走 JSON base64 (异步, 再轮询)

```bash
curl -s http://localhost:8000/v1/images \
  -H "Authorization: Bearer $FMLX_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-image-edit-2511-4bit",
    "image": ["data:image/png;base64,<PRODUCT_B64>", "data:image/png;base64,<REF_B64>"],
    "prompt": "Keep the product in Picture 1 unchanged; restyle background per Picture 2.",
    "width": 768, "height": 1344,
    "seed": 12345,
    "sync": false
  }'
# returns a job -> poll GET /v1/images/{id} for status
#               -> GET /v1/images/{id}/content?index=0 for the PNG bytes
```

### 3. 图像 t2i -- qwen-image / z-image

这是标准 OpenAI `/v1/images/generations` 接口外加 fmlx 扩展字段.

- 标准: `model`, `prompt`, `n`, `size`, `response_format` (`b64_json` 默认, 或
  `url`).
- fmlx 扩展: `negative_prompt`, `seed`, `steps`, `guidance`, `width` /
  `height`, `image_strength` (img2img), `lora_paths` / `lora_scales`.
- `/v1/images/generations` 是 OpenAI SDK 兼容别名, 永远 sync. `POST /v1/images`
  是同一套实现, 带 sync / async 开关.

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer $FMLX_KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen-image-2512-4bit","prompt":"a product hero shot, studio light","n":1,"size":"768x1344","seed":12345}'
# returns OpenAI shape: {"created":...,"data":[{"b64_json":"..."}]}
```

### 4. 图像转视频 -- wan2.2-i2v

- 首帧字段是 `input_reference`: multipart 文件字段, 或 JSON body 里的 data URL /
  base64 字符串. 不做远程 URL 拉取.
- `model` 必填; `prompt` 必填.
- 时长: `seconds` (浮点) 或 `frames` (`frames` 优先). `fps` 控帧率. 帧数会规整到
  `4n+1` (Wan 要求).
- 分辨率: `size:"WxH"` 或 `width` / `height` (16 的倍数; 不传默认 `480x272`).
- 视频永远异步: `POST` 返回 job (`object:"video"`); 轮询 `GET /v1/videos/{id}`
  直到 `status` 为 `completed`, 然后 `GET /v1/videos/{id}/content` 返回 mp4
  二进制 (支持 Range).

```bash
curl -s http://localhost:8000/v1/videos \
  -H "Authorization: Bearer $FMLX_KEY" \
  -F model=wan2.2-i2v-a14b-diffusers-8bit \
  -F input_reference=@first_frame.png \
  -F 'prompt=The character slowly turns toward the camera and smiles, gentle camera push-in.' \
  -F size=480x832 \
  -F seconds=4 \
  -F seed=12345
# returns {"id":"video_...","object":"video","status":"queued",...}
# poll:     GET /v1/videos/{id}          (status: queued -> in_progress -> completed)
# download: GET /v1/videos/{id}/content  -> mp4 bytes
```

### 5. 轮询与内容下载

提交返回的 job 对象 (或 `GET /v1/images/{id}` / `GET /v1/videos/{id}`) 带有:

- `status` -- `queued | in_progress | completed | failed` 之一.
- `progress` -- 整数 0..100.
- `phase` -- fmlx 扩展, 简短的阶段字符串.
- `error` -- 失败时填充 (`{code, message}`).
- 以及回显的参数: `model`, `size`, `prompt`, `seed`, `steps`, 视频还有
  `frames` / `fps`, 图像还有 `n` / `outputs`.

内容端点返回原始二进制, 不是 JSON, 也不是 URL 包装:

- `GET /v1/images/{id}/content` -> `image/png`. `n > 1` 时用 `?index=N` 取第 N
  张.
- `GET /v1/videos/{id}/content` -> `video/mp4` (支持 HTTP Range).

若产物被保留策略清理, 内容端点返回带 `code: "artifact_expired"` 和 `expires_at`
的 JSON 错误体.

`POST /v1/images` 走默认 `sync=true` 时可以跳过以上全部: 响应就是 OpenAI 图像
形状, 含 `data[].b64_json`. 若 `response_format:"url"`, 同步响应里是
`data[].url` 指向内容端点 (仍在同一把 API key 之后).

### 6. 跨分镜 / 跨帧一致性说明

直接关系到角色或产品跨分镜保持一致:

- `extend_video_id` (视频). 传一个已完成视频的 job id; 新 i2v 片段从该视频的
  最后一帧续生 (帧几何自动锁定到源视频). 这是无缝接镜的方式.
- seed 固定 + 候选筛选. 没有硬 mask 时, 锁主体的实用做法是固定 `seed` 加 `n` 张
  相邻候选, 然后客户端打一致性分留最优.
- `first_frame_from_text` (视频). 对 i2v / ti2v 模型, 设为 true 可由 prompt
  (经配置的首帧模型) 生成条件首帧, 无需上传. 要求不带 `input_reference` 且不带
  `extend_video_id`.

### 7. 字段速查

POST /v1/images (JSON 或 multipart):

| 字段 | 类型 | 说明 |
|------|------|------|
| `model` | string | 可选; 不传则按管线自动选 |
| `prompt` | string | 必填 |
| `image` | string / string[] / 文件 | 输入图; JSON 里 base64 或 data URL, multipart 里可重复文件字段; 最多 4 |
| `n` | int | 默认 1; 上限 `settings.image.max_n` (默认 4) |
| `size` | string | `"WxH"` 或 `"auto"` |
| `width`, `height` | int | 覆盖 `size`; 向上取整到 16; 必须成对给 |
| `response_format` | string | `b64_json` (默认) 或 `url` |
| `sync` | bool | 默认 true; false 返回 job 去轮询 |
| `seed` | int | `[0, 2^31)`; 每张图 seed 为 `seed+i` |
| `steps` | int | 不传则用 per-model / per-alias 默认 |
| `negative_prompt` | string | z-image-turbo 忽略 |
| `guidance` | float | z-image-turbo 强制 0 |
| `image_strength` | float | `(0, 1]`, 仅 t2i 的 img2img; edit 忽略 |
| `lora_paths`, `lora_scales` | string[], float[] | 仅 HF 引用 (`org/repo:file.safetensors`), 不接文件系统路径 |

POST /v1/videos (JSON 或 multipart):

| 字段 | 类型 | 说明 |
|------|------|------|
| `model` | string | 必填 |
| `prompt` | string | 必填 |
| `input_reference` | 文件 / string | i2v 首帧; JSON 里 base64 或 data URL |
| `size` | string | `"WxH"`, 默认 `480x272` |
| `width`, `height` | int | 覆盖 `size`; 向上取整到 16 |
| `seconds` | float | 时长; 或用 `frames` |
| `frames` | int | 覆盖 `seconds`; 规整到 `4n+1` |
| `fps` | int | 帧率 |
| `steps` | int | 去噪步数 |
| `seed` | int | 不传则随机 |
| `guidance`, `guidance_2` | float | classifier-free guidance |
| `negative_prompt` | string | |
| `extend_video_id` | string | 从已完成视频的最后一帧续生 |
| `upscale_resolution` | int | SeedVR2 逐帧超分, 短边, `480..max` |
| `prompt_extend` | bool | LLM prompt 扩写覆盖 |
| `first_frame_from_text` | bool | 由 prompt 生成 i2v 首帧 |
