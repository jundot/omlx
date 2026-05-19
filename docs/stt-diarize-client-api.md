# STT Diarization API — Client Integration Guide

This document describes the speaker-attribution (diarization) features of
the oMLX `/v1/audio/transcriptions` endpoint as deployed on the Flyto
production server. It is the client-facing companion to the authoritative
spec in the [`flyto-mlx`](https://github.com/panwudi/flyto-mlx) repository
(`docs/stt-diarize-api.md`); both documents are kept in sync.

English version first. Chinese translation follows below.

---

## English

### 1. Endpoint and authentication

```
POST  <base_url>/v1/audio/transcriptions
```

| Environment | `<base_url>` |
|---|---|
| Production (m5max) | `http://m5max:8000/v1` |
| Dev / staging (m2max) | `http://m2max:8000/v1` |

Authentication is by API key, sent via the `Authorization: Bearer <key>`
header (the endpoint accepts the OpenAI-style header for tooling
compatibility; internally the server stores this as `X-API-Key`).

```
Authorization: Bearer <api_key>
```

The request body is `multipart/form-data` — all parameters below are
sent as form fields.

### 2. Backend selection at a glance

The `diarize_backend` form field selects how speakers are attributed to
words. Five values are accepted:

| Backend | Audio shape | ASR cost | Speaker assignment | When to pick |
|---|---|---|---|---|
| `none` | any | 1× | not assigned | plain transcription, no speaker info needed |
| `energy` | stereo, one speaker per channel | 1× | per-word L/R RMS ratio | cheap default for stereo recordings with known channel mapping |
| `energy_tripass` | stereo, one speaker per channel | 3× | per-channel ASR + mix-pass merge | maximum word coverage on stereo, even during simultaneous speech |
| `pyannote` | mono / multi-speaker / single-mic | 1× ASR + 1× pyannote | speaker-diarization-3.1 model | mono recordings, conference audio, podcasts with a shared mic |
| `auto` (default) | dispatcher | depends | depends | let the server choose based on input shape + signals of intent |

The `auto` dispatcher resolves as follows:

- if `left_speaker` AND `right_speaker` are both given → `energy` (or
  `energy_tripass` when the server-side per-model preference
  `default_diarize_quality` is set to `"high"` — see §5)
- else if any of `speakers`, `num_speakers`, non-default
  `min_speakers`/`max_speakers` are given → `pyannote`
- else → no diarization (plain transcription)

### 3. Form-field reference

#### 3.1 Required for every call

| Field | Type | Notes |
|---|---|---|
| `model` | string | Model alias, e.g. `Qwen3-ASR-1.7B-audio8-text4-mlx` |
| `file` | file | Audio file (wav, mp3, m4a, mp4, mov, mkv — video containers are repacked through ffmpeg) |

#### 3.2 Common transcription options

| Field | Type | Default | Notes |
|---|---|---|---|
| `language` | string | (auto) | `zh`, `en`, … — recommended to set explicitly |
| `word_timestamps` | bool | `false` | Auto-enabled when diarize backend ≠ `none` or `auto`-resolved-to-`none` |
| `response_format` | string | `json` | `json`, `verbose_json`, `text`, `srt`, `vtt` |
| `prompt` | string | | ASR-specific bias prompt, passed straight to the model |
| `temperature` | float | `0` | Greedy by default; >0 only when ASR loops on quasi-silent input |
| `max_tokens` | int | (model default) | Raise for long-form audio (e.g. VibeVoice ASR truncates at 8192) |

The full sampling-tail (`top_p`, `top_k`, `min_p`, `repetition_penalty`,
`repetition_context_size`, `frequency_penalty`,
`frequency_context_size`, `presence_penalty`, `presence_context_size`)
is also forwarded — see the upstream Qwen3-ASR docs.

#### 3.3 Diarization-specific fields

| Field | Type | Notes |
|---|---|---|
| `diarize_backend` | string | One of `auto`, `energy`, `energy_tripass`, `pyannote`, `none` |
| `left_speaker` | string | Required for `energy` / `energy_tripass`; the label string for words on the L channel |
| `right_speaker` | string | Required for `energy` / `energy_tripass`; the label string for words on the R channel |
| `diarize_threshold` | float | `energy` only. `max(L,R)/min(L,R)` below this is tagged `[overlap]`. Default `1.3` |
| `speakers` | string | `pyannote` only. Comma-separated canonical names. First emitted SPEAKER_00 maps to `speakers[0]`, etc. |
| `num_speakers` | int | `pyannote`. Constrains the pipeline to exactly N speakers |
| `min_speakers` | int | `pyannote`. Soft lower bound (default 2; non-default value counts as intent for `auto`-routing) |
| `max_speakers` | int | `pyannote`. Soft upper bound (default 8) |

### 4. Per-backend behavior

#### 4.1 `none` and `auto`-resolved-to-none

Standard transcription. Response has `text`, `segments[].text`, and
`segments[].words[]` (the latter only when `word_timestamps=true` is
either explicit or implied by another mode). No `speaker` field is set
on any word or segment.

#### 4.2 `energy` (1-pass stereo)

Requires **stereo input** (two channels, one speaker per channel — the
canonical case is FreeSWITCH two-leg call recording or a separate-mic
podcast). The server:

1. Down-mixes L+R to mono with 0.5 scaling and runs a single ASR pass on
   the mix.
2. For each word in the ASR output, walks the original stereo buffer
   over `[word.start - 0.05s, word.end + 0.05s]`, computes per-channel
   RMS, and assigns the speaker by whichever channel is louder.
3. If `max(L_rms, R_rms) / min(L_rms, R_rms) < diarize_threshold`
   (default 1.3), tags the word as `[overlap]` — the channels are too
   close in energy to disambiguate.

Each `segments[].speaker` is set by majority vote across that segment's
words. Cost: 1 ASR pass (~5 s / minute of audio at Qwen3-ASR 1.7B on
M5 Max). Tradeoff: words spoken simultaneously on both channels are
dropped (the mix-down collapses two voices, ASR can't separate, the
word is tagged `[overlap]` and effectively lost).

#### 4.3 `energy_tripass` (3-pass stereo, recommended for high-quality stereo)

Same input requirements as `energy`. The server runs ASR three times:

- L channel mono → speaker_L's words (deterministic attribution)
- R channel mono → speaker_R's words (deterministic attribution)
- (L+R)/2 mix mono → canonical text with full-context decoding

Each per-channel pass also chains the configured ForcedAligner companion
(e.g. `Qwen3-ForcedAligner-0.6B-4bit`) so `words[]` is populated for the
merge step.

The merge:

1. Walks the mix pass word stream as authoritative text (the mix pass
   sees both speakers, so boundary words like 你/您, 对/兑 disambiguate
   accurately).
2. For each mix word at time `t`, looks for the same text in L's stream
   within ±0.3 s. Same for R. Then:
   - matched only in L → `speaker = left_speaker`
   - matched only in R → `speaker = right_speaker`
   - matched in both → fall back to energy ratio
   - matched in neither → fall back to energy ratio
3. Recovers per-channel words the mix pass dropped (e.g. words that
   were stomped on by simultaneous speech). A recovered word is only
   accepted if its originating channel carries at least 55 % of the
   total per-window RMS — this filters out cross-talk-leak ghost
   transcriptions where the channel was actually silent.
4. Re-segments by speaker turn so SRT/VTT cue boundaries follow the
   conversation instead of presenting one giant segment.

Cost: 3× ASR latency (the underlying MLX engine is single-threaded so
the three passes run sequentially; ~15 s / minute on Qwen3-ASR 1.7B at
M5 Max). Benefit: no `[overlap]` tag, simultaneous-speech words are
recovered from the correct channel, and speaker attribution is
deterministic — not heuristic.

Known caveat: per-channel ASR on a quiet channel can hallucinate
characters in low-signal regions. Empirically observed on a real
two-leg call: the salesperson's channel grew a phantom `何总` and a
phantom `宠物用` in places where the channel was nearly silent. This is
the main reason an LLM post-processing layer is on the roadmap (filter
hallucinations + restore aligner-stripped punctuation +
semantically-correct speaker re-mapping). For now treat tripass output
as raw material that a downstream consumer can clean up.

#### 4.4 `pyannote` (mono / multi-speaker)

For recordings where each channel does not carry exactly one speaker —
conference room single mic, podcast on shared mic, mono telephony with
multiple participants. Wraps the
[`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1)
pipeline.

Server-side prerequisites (already done on m5max):

- `pip install pyannote.audio torch torchaudio` (≈ 400 MB on arm64)
- HuggingFace gated license accepted on the server's HF token account
  for three repos:
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/segmentation-3.0`
  - `pyannote/speaker-diarization-community-1` (added in pyannote 4.x)
- HF token discoverable by the server (`hf auth login` cache at
  `~/.cache/huggingface/token` is honored; `HF_TOKEN` env also works)

Speaker labels are assigned by pyannote in order of first emission:
`SPEAKER_00` → `speakers[0]`, `SPEAKER_01` → `speakers[1]`, etc. The
first person to talk in the audio becomes `speakers[0]` — pyannote does
not know who is the salesperson and who is the customer, only the
acoustic identity of each voice. If your canonical names imply roles
(`销售`, `客户`), be aware the mapping is by acoustic order, not
semantic role. Roles can be re-attached downstream from the transcript
text.

The pipeline is loaded lazily on the first request (3–5 s one-time
load) and reused thereafter via a module-level singleton. 8 kHz
telephony audio is upsampled to 16 kHz inside the wrapper. Stereo input
is down-mixed to mono.

Cost: 1 ASR pass + 1 pyannote pass. Pyannote runs on CPU (no GPU/Metal
acceleration in the current wrapper), so a 158 s 8 kHz call takes
roughly 8–12 s for the pyannote pass alone.

#### 4.5 `auto` (default)

See §2 for the dispatch logic. The recommended use is to let the
caller set `auto` plus whatever intent fields make sense
(`left_speaker`/`right_speaker` for stereo recordings, `speakers` /
`num_speakers` for multi-speaker mono), and let the server pick the
backend.

### 5. Server-side per-model quality preference

The per-model setting `default_diarize_quality` (persisted in
`~/.omlx/model_settings.json`, exposed via the admin endpoint
`PUT /admin/api/models/{model_id}/settings` with field
`default_diarize_quality`) controls whether the `auto` dispatcher picks
`energy_tripass` instead of `energy` on stereo+L/R input.

Values: `"standard"` (or `null`) and `"high"`. Default is `null`,
meaning `auto` keeps the cheap `energy` backend.

A per-request explicit `diarize_backend=energy` or
`diarize_backend=energy_tripass` always overrides the preference — the
caller is the source of truth when they declare intent.

Production default on m5max as of this writing: `null` (standard). The
3× cost of tripass is opt-in by the caller, not silently default.

### 6. End-to-end examples (tested against m5max)

The sample below is `med.wav`, a 158-second stereo 8 kHz FreeSWITCH
two-leg recording of a sales call. The salesperson is on the L channel,
the customer on the R channel.

#### Example 1 — Plain transcription (no diarize)

```bash
curl -sS -X POST http://m5max:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer <api_key>" \
  -F "model=Qwen3-ASR-1.7B-audio8-text4-mlx" \
  -F "file=@med.wav" \
  -F "language=zh"
```

Response (`json`):

```json
{
  "text": "喂，你好。哎，你好，请问是合作是吧？……",
  "language": "chinese",
  "duration": 2.07,
  "segments": null
}
```

The `duration` field reports inference time, not audio length.

#### Example 2 — `energy` (1-pass stereo)

```bash
curl -sS -X POST http://m5max:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer <api_key>" \
  -F "model=Qwen3-ASR-1.7B-audio8-text4-mlx" \
  -F "file=@med.wav" \
  -F "language=zh" \
  -F "diarize_backend=energy" \
  -F "left_speaker=销售" \
  -F "right_speaker=客户"
```

Response shape (`segments[].words[]` has the speaker labels):

```json
{
  "text": "喂，你好。哎，你好……",
  "language": "chinese",
  "segments": [{
    "text": "喂，你好。……",
    "start": 0.0,
    "end": 158.74,
    "speaker": "销售",
    "words": [
      {"word": "喂", "start": 8.16, "end": 8.32, "speaker": "客户"},
      {"word": "你", "start": 8.32, "end": 8.40, "speaker": "客户"},
      {"word": "好", "start": 8.40, "end": 8.64, "speaker": "客户"},
      {"word": "哎", "start": 9.52, "end": 9.68, "speaker": "销售"},
      …,
      {"word": "是", "start": 11.20, "end": 11.28, "speaker": "overlap"}
    ]
  }]
}
```

#### Example 3 — `energy_tripass` (3-pass stereo)

```bash
curl -sS -X POST http://m5max:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer <api_key>" \
  -F "model=Qwen3-ASR-1.7B-audio8-text4-mlx" \
  -F "file=@med.wav" \
  -F "language=zh" \
  -F "diarize_backend=energy_tripass" \
  -F "left_speaker=销售" \
  -F "right_speaker=客户"
```

Response shape (segments are now per speaker turn, no `[overlap]`):

```json
{
  "text": "喂你好哎你好请问是合何总作是吧……",
  "language": "chinese",
  "segments": [
    {"speaker": "客户", "start": 8.16, "end": 8.64, "text": "喂你好", "words": [...]},
    {"speaker": "销售", "start": 9.52, "end": 11.28, "text": "哎你好请问是合何总", "words": [...]},
    {"speaker": "客户", "start": 11.20, "end": 11.20, "text": "作", "words": [...]},
    {"speaker": "销售", "start": 11.28, "end": 11.44, "text": "是吧", "words": [...]},
    …
  ]
}
```

Note that the forced-aligner strips punctuation, so the segment-level
`text` is bare characters. The top-level `text` field is built from the
segment texts. Both will gain punctuation back once the LLM
post-processing layer ships.

#### Example 4 — `pyannote` (mono / multi-speaker)

```bash
curl -sS -X POST http://m5max:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer <api_key>" \
  -F "model=Qwen3-ASR-1.7B-audio8-text4-mlx" \
  -F "file=@meeting.wav" \
  -F "language=zh" \
  -F "diarize_backend=pyannote" \
  -F "num_speakers=3" \
  -F "speakers=Alice,Bob,Charlie"
```

`SPEAKER_00` in pyannote's output maps to `Alice`, `SPEAKER_01` to
`Bob`, etc. — by order of first emission.

#### Example 5 — `auto` with intent signals

```bash
# Stereo with L/R → server picks energy (or energy_tripass if
# default_diarize_quality=high).
curl ... -F "diarize_backend=auto" \
         -F "left_speaker=销售" -F "right_speaker=客户"

# Multi-speaker mono → server picks pyannote.
curl ... -F "diarize_backend=auto" -F "speakers=A,B,C" -F "num_speakers=3"

# No intent signals → no diarization, plain transcription.
curl ... -F "diarize_backend=auto"
```

#### Example 6 — Subtitle output with speaker prefix

```bash
curl ... -F "diarize_backend=energy" -F "left_speaker=销售" \
         -F "right_speaker=客户" -F "response_format=srt"
```

SRT cues include a speaker prefix (`销售: text`); VTT cues use the
voice tag (`<v 销售>text</v>`).

### 7. Error codes

| HTTP | When | Body shape |
|---|---|---|
| `400` | unknown `diarize_backend`; energy without L/R; mono audio sent to energy; invalid `default_diarize_quality` | `{"error": {"message": "...", "type": "..."}}` |
| `404` | unknown `model` | `{"error": {"message": "Model 'X' not found. Available: ..."}}` |
| `503` | pyannote unavailable (no install / HF license missing / HF_TOKEN missing) | `{"error": {"message": "pyannote diarization unavailable: <wrapper's actionable hint>"}}` |
| `500` | unexpected ASR / pyannote / merge failure | `{"error": {"message": "..."}}` |

`503` from `pyannote` always carries the actionable hint verbatim from
the wrapper — read the message body before paging anyone.

### 8. Performance reference (med.wav, 158 s stereo 8 kHz, M5 Max)

| Backend | Wall time | Output character count | `[overlap]` words | Notes |
|---|---|---|---|---|
| `none` | ~2 s | 703 | n/a | baseline ASR |
| `energy` | ~3 s | 700 (1 segment, 585 words) | 11 | 1× ASR + RMS pass |
| `energy_tripass` | ~9 s | 605 (50 speaker-turn segments) | 0 | 3× ASR + merge |
| `pyannote` | ~12 s | 700 | n/a | 1× ASR + pyannote CPU pass |

The "output character count" diverges across backends because each
backend strips or reattaches text differently. `energy_tripass`
character count is lower than `energy` because the merge currently
strips the punctuation that the original mix-pass `text` had — this is
the punctuation-restoration gap the LLM post-processing layer will
close.

### 9. Known caveats and roadmap

1. **Quiet-channel hallucination on `energy_tripass`.** Per-channel
   ASR can hallucinate characters in low-signal regions of the input.
   Mitigate by post-processing (LLM filter) or by client-side text
   sanity heuristics.

2. **Punctuation loss on `energy_tripass`.** The forced-aligner emits
   bare characters in `words[]`; the merge step reassembles text from
   those without restoring the punctuation the ASR's segment text had.
   This will be addressed by the LLM post-processing layer.

3. **Pyannote semantic role.** `SPEAKER_00` → `speakers[0]` is by
   acoustic order, not semantic role. If the caller cares about
   role assignment, downstream cleanup is required (e.g. an LLM step
   that reads the transcript and reassigns `销售` / `客户`).

4. **3× cost on `energy_tripass`.** The MLX engine runs the three
   passes sequentially. For batch processing this is fine; for
   near-realtime use prefer `energy` or process in parallel via
   multiple HTTP requests against the same engine pool.

5. **LLM post-processing layer (planned).** A new optional form field
   `postprocess_model` (and a per-model `default_postprocess_model`
   setting) will let the server chain an LLM cleanup pass after
   diarization. It will filter hallucinations, restore punctuation,
   re-map speakers by semantic context, and merge fragmented segments
   into coherent turns. Server-side because the model is already
   loaded on the same host.

---

## 中文

### 1. 接口与认证

```
POST  <base_url>/v1/audio/transcriptions
```

| 环境 | `<base_url>` |
|---|---|
| 生产 (m5max) | `http://m5max:8000/v1` |
| 开发 / staging (m2max) | `http://m2max:8000/v1` |

认证用 API key, 通过 `Authorization: Bearer <key>` header 传 (这个 header
是兼容 OpenAI 调用风格写的, 服务端内部当 `X-API-Key` 处理)。

```
Authorization: Bearer <api_key>
```

请求体是 `multipart/form-data` — 下面所有参数都是 form 字段。

### 2. Backend 一眼速览

`diarize_backend` 字段决定每个词怎么归到说话人。5 个取值:

| Backend | 音频形态 | ASR 成本 | 说话人归属方式 | 什么时候用 |
|---|---|---|---|---|
| `none` | 任何 | 1× | 不归 | 纯转录, 不需要说话人信息 |
| `energy` | 立体声, 每声道一人 | 1× | 按 word 时间的 L/R RMS 比 | 立体声 + 已知声道对应说话人, 便宜的默认 |
| `energy_tripass` | 立体声, 每声道一人 | 3× | 按声道单独 ASR + mix 路 merge | 立体声场景下追求 100% 词覆盖 (含同时说话也不漏) |
| `pyannote` | 单声道 / 多人 / 单麦 | 1× ASR + 1× pyannote | speaker-diarization-3.1 模型 | 单声道录音 / 会议 / 共享麦 podcast |
| `auto` (默认) | 调度器 | 取决 | 取决 | 让服务端按输入形态 + intent 信号自动选 |

`auto` 调度逻辑:

- 如果同时给了 `left_speaker` 和 `right_speaker` → `energy` (或者当服务端
  per-model 偏好 `default_diarize_quality = "high"` 时升 `energy_tripass`,
  见 §5)
- 否则只要给了 `speakers` / `num_speakers` / 非默认值的 `min_speakers` /
  `max_speakers` 其中之一 → `pyannote`
- 否则 → 不做 diarize (纯转录)

### 3. Form 字段参考

#### 3.1 每次都必传

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 模型 alias, 比如 `Qwen3-ASR-1.7B-audio8-text4-mlx` |
| `file` | file | 音频文件 (wav / mp3 / m4a / mp4 / mov / mkv — 视频容器走 ffmpeg 重打包) |

#### 3.2 转录通用选项

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `language` | string | (自动) | `zh`, `en`, … 推荐显式指定 |
| `word_timestamps` | bool | `false` | diarize backend 非 `none` 时自动启 |
| `response_format` | string | `json` | `json` / `verbose_json` / `text` / `srt` / `vtt` |
| `prompt` | string | | ASR-specific bias prompt, 直接透传给模型 |
| `temperature` | float | `0` | 默认 greedy; >0 仅当 ASR 在准静音输入上循环时启用 |
| `max_tokens` | int | (模型默认) | 长音频时调高 (例如 VibeVoice ASR 默认 8192 会截) |

完整的 sampling tail (`top_p`, `top_k`, `min_p`,
`repetition_penalty`, `repetition_context_size`, `frequency_penalty`,
`frequency_context_size`, `presence_penalty`, `presence_context_size`)
也都透传 — 参考上游 Qwen3-ASR 文档。

#### 3.3 Diarize 专属字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `diarize_backend` | string | `auto` / `energy` / `energy_tripass` / `pyannote` / `none` |
| `left_speaker` | string | `energy` / `energy_tripass` 必填, L 声道词的标签 |
| `right_speaker` | string | `energy` / `energy_tripass` 必填, R 声道词的标签 |
| `diarize_threshold` | float | 仅 `energy`. `max(L,R)/min(L,R)` 低于该值的词标 `[overlap]`. 默认 1.3 |
| `speakers` | string | 仅 `pyannote`. 逗号分隔的 canonical name 列表. 首次出现的 SPEAKER_00 映射到 `speakers[0]`, 依此类推 |
| `num_speakers` | int | `pyannote`. 强制 N 个说话人 |
| `min_speakers` | int | `pyannote`. 软下界 (默认 2; 非默认值会被 `auto`-route 当 intent 信号) |
| `max_speakers` | int | `pyannote`. 软上界 (默认 8) |

### 4. 各 Backend 详细行为

#### 4.1 `none` 和被 `auto` 解析为 none 的情况

标准转录。响应包含 `text`, `segments[].text`, 以及
`segments[].words[]` (后者仅在 `word_timestamps=true` 显式或被其它模式
隐式启动时才有). 任何 word 或 segment 上都没有 `speaker` 字段。

#### 4.2 `energy` (1-pass 立体声)

要求**立体声输入** (两个声道, 每个声道一个说话人 — 典型场景是
FreeSWITCH 双 leg 通话录音, 或独立麦克风的 podcast). 服务端:

1. 把 L+R 按 0.5 系数混到单声道, 单次 ASR 跑混音。
2. 对每个 ASR 输出的词, 在原 stereo buffer 的
   `[word.start - 0.05s, word.end + 0.05s]` 窗口上计算两个声道的 RMS,
   按较大的那个声道决定说话人。
3. 如果 `max(L_rms, R_rms) / min(L_rms, R_rms) < diarize_threshold`
   (默认 1.3), 这个词标 `[overlap]` — 能量太接近, 无法消歧。

`segments[].speaker` 按该 segment 内所有词的 majority vote 决定. 成本:
1 次 ASR (M5 Max 上 Qwen3-ASR 1.7B 约 5s/分钟音频). 取舍: 双方同时说话
的词会被丢 (混音把两个声音合到一起, ASR 分不开, 最终这个词被标
`[overlap]` 实际上字也没出来).

#### 4.3 `energy_tripass` (3-pass 立体声, 推荐用于高质量立体声场景)

输入要求跟 `energy` 一样. 服务端跑 3 次 ASR:

- L 声道 mono → 左说话人的词 (确定性归属)
- R 声道 mono → 右说话人的词 (确定性归属)
- (L+R)/2 混音 mono → 权威文本 (完整上下文解码)

每个 per-channel pass 也会跟一次配置好的 ForcedAligner companion
(例如 `Qwen3-ForcedAligner-0.6B-4bit`), 补 `words[]` 给 merge 用。

Merge:

1. 把 mix 路的词流当权威文本走 (mix 路能看到双方, 边界字像
   你/您, 对/兑 的消歧最准).
2. 对每个 mix 词时间 `t`, 在 L 流里找 ±0.3s 窗口的同字, R 同理. 然后:
   - 只在 L 匹配 → `speaker = left_speaker`
   - 只在 R 匹配 → `speaker = right_speaker`
   - 两边都匹配 → 退回 RMS 比兜底
   - 两边都不匹配 → 退回 RMS 比兜底
3. 捞回 mix 路漏掉的词 (比如被同时说话淹没的词). 捞回的词只有当源声道
   能量占该窗口总 RMS 的 ≥55% 才采纳 — 过滤 cross-talk 漏过去的 ghost
   转录 (实际上那个声道当时是静音的).
4. 按 speaker turn 重新分 segment, SRT/VTT cue 边界跟对话翻页, 不再
   一个 segment 装到底。

成本: 3× ASR 时间 (底层 MLX 引擎单线程, 3 路串行; M5 Max Qwen3-ASR 1.7B
约 15s/分钟). 收益: 没有 `[overlap]` 标签, 同时说话的词从对的声道捞回,
speaker 归属是确定性的, 不是启发式。

已知 caveat: 单声道 ASR 在低信号段会幻觉. 实测在真实双 leg 通话上,
销售那条声道在几乎静音的位置长出了凭空的 `何总` 和 `宠物用` —
这正是 LLM 后处理层规划中的主要原因 (过滤幻觉 + 补回 aligner 撕掉
的标点 + 按语义重映射 speaker). 当前阶段把 tripass 输出当下游需要清洗
的原始素材看待。

#### 4.4 `pyannote` (单声道 / 多人)

适合每个声道不是恰好一人的场景 — 会议室单麦, 共享麦 podcast, 多人
单声道电话. 封装的是
[`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1)
pipeline.

服务端前置要求 (m5max 上已经做完):

- `pip install pyannote.audio torch torchaudio` (arm64 上约 400 MB)
- HuggingFace gated license 已用服务器的 HF token 账户接受了 3 个 repo:
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/segmentation-3.0`
  - `pyannote/speaker-diarization-community-1` (pyannote 4.x 新增)
- 服务端能找到 HF token (`hf auth login` 写到
  `~/.cache/huggingface/token` 也可; `HF_TOKEN` env 也行)

说话人标签按 pyannote 首次出现的顺序映射: `SPEAKER_00` →
`speakers[0]`, `SPEAKER_01` → `speakers[1]`, etc. 第一个开口说话的人
就是 `speakers[0]` — pyannote 不知道谁是销售谁是客户, 它只认识每个声音
的声学身份. 如果你的 canonical name 里隐含了角色 (`销售`, `客户`),
请注意映射是按声学顺序, 不是语义角色. 角色重映射要在下游对转录文本
做后处理。

Pipeline 第一次请求时懒加载 (一次性 3–5s), 之后 module-level 单例复用.
8 kHz 电话音频在 wrapper 内升采样到 16 kHz. 立体声会被下混到单声道。

成本: 1× ASR + 1× pyannote. Pyannote 跑在 CPU 上 (当前 wrapper 没接
GPU/Metal 加速), 所以一条 158s 8 kHz 通话光 pyannote pass 大约 8-12s.

#### 4.5 `auto` (默认)

调度逻辑见 §2. 推荐用法是 caller 设 `auto` + 把对应的 intent 字段填上
(立体声录音填 `left_speaker`/`right_speaker`, 多人单声道填
`speakers`/`num_speakers`), 让服务端选 backend。

### 5. 服务端 per-model 质量偏好

per-model 设置 `default_diarize_quality` (持久化在
`~/.omlx/model_settings.json`, 通过 admin endpoint
`PUT /admin/api/models/{model_id}/settings` 字段
`default_diarize_quality` 暴露) 控制 `auto` 调度器在 stereo+L/R 输入
时是否选 `energy_tripass` 代替 `energy`。

取值: `"standard"` (或 `null`) 和 `"high"`. 默认 `null`, 意思 `auto`
保持便宜的 `energy` backend。

每次请求显式传 `diarize_backend=energy` 或
`diarize_backend=energy_tripass` 永远会覆盖这个偏好 — caller 显式
declare intent 时它就是 source of truth。

文档当前时点 m5max 生产默认: `null` (standard). tripass 的 3× 成本
是 caller opt-in 的, 不是隐式默认。

### 6. 端到端例子 (m5max 上实测)

下面用的样本是 `med.wav`, 一段 158 秒立体声 8 kHz FreeSWITCH 双 leg
销售通话录音. 销售在 L 声道, 客户在 R 声道.

#### 例 1 — 纯转录 (不开 diarize)

```bash
curl -sS -X POST http://m5max:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer <api_key>" \
  -F "model=Qwen3-ASR-1.7B-audio8-text4-mlx" \
  -F "file=@med.wav" \
  -F "language=zh"
```

响应 (`json`):

```json
{
  "text": "喂，你好。哎，你好，请问是合作是吧？……",
  "language": "chinese",
  "duration": 2.07,
  "segments": null
}
```

`duration` 字段是推理时间, 不是音频时长.

#### 例 2 — `energy` (1-pass 立体声)

```bash
curl -sS -X POST http://m5max:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer <api_key>" \
  -F "model=Qwen3-ASR-1.7B-audio8-text4-mlx" \
  -F "file=@med.wav" \
  -F "language=zh" \
  -F "diarize_backend=energy" \
  -F "left_speaker=销售" \
  -F "right_speaker=客户"
```

响应形状 (`segments[].words[]` 上带 speaker 标签):

```json
{
  "text": "喂，你好。哎，你好……",
  "language": "chinese",
  "segments": [{
    "text": "喂，你好。……",
    "start": 0.0,
    "end": 158.74,
    "speaker": "销售",
    "words": [
      {"word": "喂", "start": 8.16, "end": 8.32, "speaker": "客户"},
      {"word": "你", "start": 8.32, "end": 8.40, "speaker": "客户"},
      {"word": "好", "start": 8.40, "end": 8.64, "speaker": "客户"},
      {"word": "哎", "start": 9.52, "end": 9.68, "speaker": "销售"},
      …,
      {"word": "是", "start": 11.20, "end": 11.28, "speaker": "overlap"}
    ]
  }]
}
```

#### 例 3 — `energy_tripass` (3-pass 立体声)

```bash
curl -sS -X POST http://m5max:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer <api_key>" \
  -F "model=Qwen3-ASR-1.7B-audio8-text4-mlx" \
  -F "file=@med.wav" \
  -F "language=zh" \
  -F "diarize_backend=energy_tripass" \
  -F "left_speaker=销售" \
  -F "right_speaker=客户"
```

响应形状 (segments 现在按 speaker turn 切, 没有 `[overlap]`):

```json
{
  "text": "喂你好哎你好请问是合何总作是吧……",
  "language": "chinese",
  "segments": [
    {"speaker": "客户", "start": 8.16, "end": 8.64, "text": "喂你好", "words": [...]},
    {"speaker": "销售", "start": 9.52, "end": 11.28, "text": "哎你好请问是合何总", "words": [...]},
    {"speaker": "客户", "start": 11.20, "end": 11.20, "text": "作", "words": [...]},
    {"speaker": "销售", "start": 11.28, "end": 11.44, "text": "是吧", "words": [...]},
    …
  ]
}
```

注意 forced-aligner 会撕掉标点, 所以 segment 级 `text` 是裸字符.
top-level `text` 是 segment text 拼出来的. 等 LLM 后处理层上线后两者
都会把标点补回来。

#### 例 4 — `pyannote` (单声道 / 多人)

```bash
curl -sS -X POST http://m5max:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer <api_key>" \
  -F "model=Qwen3-ASR-1.7B-audio8-text4-mlx" \
  -F "file=@meeting.wav" \
  -F "language=zh" \
  -F "diarize_backend=pyannote" \
  -F "num_speakers=3" \
  -F "speakers=Alice,Bob,Charlie"
```

pyannote 输出里的 `SPEAKER_00` 映射到 `Alice`, `SPEAKER_01` 到 `Bob`,
依此类推 — 按首次出现顺序。

#### 例 5 — `auto` 配 intent 信号

```bash
# 立体声 + L/R → 服务端选 energy (或在 default_diarize_quality=high 时
# 升到 energy_tripass)
curl ... -F "diarize_backend=auto" \
         -F "left_speaker=销售" -F "right_speaker=客户"

# 多人单声道 → 服务端选 pyannote
curl ... -F "diarize_backend=auto" -F "speakers=A,B,C" -F "num_speakers=3"

# 没 intent 信号 → 不 diarize, 纯转录
curl ... -F "diarize_backend=auto"
```

#### 例 6 — 字幕输出带 speaker 前缀

```bash
curl ... -F "diarize_backend=energy" -F "left_speaker=销售" \
         -F "right_speaker=客户" -F "response_format=srt"
```

SRT cue 带 speaker 前缀 (`销售: text`); VTT 用 voice 标签
(`<v 销售>text</v>`)。

### 7. 错误码

| HTTP | 触发时机 | Body 形状 |
|---|---|---|
| `400` | 未知 `diarize_backend`; energy 但没给 L/R; 给 energy 的是单声道; 非法的 `default_diarize_quality` | `{"error": {"message": "...", "type": "..."}}` |
| `404` | `model` 不存在 | `{"error": {"message": "Model 'X' not found. Available: ..."}}` |
| `503` | pyannote 不可用 (没装 / HF license 没接受 / HF_TOKEN 缺失) | `{"error": {"message": "pyannote diarization unavailable: <wrapper 的可执行提示>"}}` |
| `500` | 意外的 ASR / pyannote / merge 失败 | `{"error": {"message": "..."}}` |

`pyannote` 的 `503` 一定带 wrapper 原文 actionable hint — 先读 body
再决定要不要找人。

### 8. 性能参考 (med.wav, 158s 立体声 8 kHz, M5 Max)

| Backend | 墙钟时间 | 输出字符数 | `[overlap]` 词数 | 备注 |
|---|---|---|---|---|
| `none` | ~2s | 703 | n/a | 基线 ASR |
| `energy` | ~3s | 700 (1 segment, 585 words) | 11 | 1× ASR + RMS |
| `energy_tripass` | ~9s | 605 (50 个 speaker turn segments) | 0 | 3× ASR + merge |
| `pyannote` | ~12s | 700 | n/a | 1× ASR + pyannote CPU pass |

"输出字符数" 不同 backend 不一样是因为每个 backend 撕 / 补 text 的方式
不同. `energy_tripass` 字符数比 `energy` 少是因为 merge 撕掉了原 mix
pass 的标点 — 这正是 LLM 后处理层要补回的标点缺失.

### 9. 已知 caveat 和路线图

1. **`energy_tripass` 安静声道幻觉.** 单声道 ASR 在输入的低信号段会
   幻觉. 后处理 (LLM 过滤) 或客户端文本合理性启发式 mitigate.

2. **`energy_tripass` 标点丢失.** Forced-aligner 在 `words[]` 里只出
   裸字符; merge 拼回 text 时没把 ASR segment text 里的标点补回. 这个
   等 LLM 后处理层解决.

3. **Pyannote 没语义角色.** `SPEAKER_00` → `speakers[0]` 是声学顺序,
   不是语义角色. caller 如果在意角色归属, 要下游处理 (比如 LLM 读
   transcript 重新分 `销售` / `客户`).

4. **`energy_tripass` 3× 成本.** MLX 引擎 3 路串行跑. batch 场景没问
   题; near-realtime 优先 `energy` 或者用同一个 engine pool 并发多个
   HTTP 请求.

5. **LLM 后处理层 (计划中).** 新的可选 form 字段 `postprocess_model`
   (加上 per-model `default_postprocess_model` 设置) 会让服务端在
   diarize 之后串一次 LLM 清洗. 过滤幻觉 + 补回标点 + 按语义重映射
   speaker + 合并碎 segment 成顺畅 turn. 服务端而不是客户端做, 因为
   LLM 模型已经在同一台服务器上加载着了.
