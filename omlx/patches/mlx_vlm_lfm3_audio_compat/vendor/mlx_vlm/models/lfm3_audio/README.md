# LFM3-Audio on MLX

LFM3-Audio-2B is an LFM3 language model that listens and speaks. A
log-mel front end, three strided convolutions and a 17-layer LFM2 encoder turn
16 kHz speech into one backbone embedding per 80 ms. The model writes text
tokens, or speech frames of 12 codebook codes that a bidirectional depth
decoder samples in three MaskGIT passes. The bundled Liquid codec
(`audio_tokenizer/`; earlier checkpoints named it `audio_detokenizer/`)
decodes the frames to 24 kHz audio. The checkpoint loads directly; no remote
code or PyTorch is needed. The implementation adapts
the Apache-2.0 reference code shipped with the checkpoint; see
[LICENSE](../lfm3/LICENSE).

```python
from mlx_vlm import generate, load
from mlx_vlm.generate import generate_audio
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("path/to/LFM3-Audio-2B")

def chat(system, user=""):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# Speech recognition ("Perform ASR." lets the model pick the language)
prompt = apply_chat_template(
    processor, model.config, chat("Perform ASR in English."), num_audios=1
)
print(generate(model, processor, prompt, audio=["speech.wav"], temperature=0).text)

# Text to speech
prompt = apply_chat_template(
    processor, model.config,
    chat("Perform TTS in American English using a female voice.", "Hello there!"),
)
result = generate_audio(model, processor, prompt, output_audio_path="tts.wav")

# Speech-to-speech chat: interleaved text and speech
prompt = apply_chat_template(
    processor, model.config, chat("Respond with interleaved text and audio."),
    num_audios=1,
)
result = generate_audio(
    model, processor, prompt, audio=["question.wav"], max_tokens=1024,
    output_audio_path="answer.wav",
)
print(result.text)
```

From the command line:

```sh
# ASR
python -m mlx_vlm.generate --model path/to/LFM3-Audio-2B \
  --system "Perform ASR in English." --prompt "" --audio speech.wav --temperature 0
# TTS
python -m mlx_vlm.generate --model path/to/LFM3-Audio-2B \
  --system "Perform TTS in American English using a female voice." \
  --prompt "Hello there!" --output-modality audio --output tts.wav
# Speech-to-speech chat
python -m mlx_vlm.generate --model path/to/LFM3-Audio-2B \
  --system "Respond with interleaved text and audio." --prompt "" \
  --audio question.wav --output-modality audio --max-tokens 1024 --output answer.wav
```

`python -m mlx_vlm generate_audio ...` is the same as
`python -m mlx_vlm generate --output-modality audio ...`. Speech output needs
`--output` (a `.wav` file) and does not work with `--chat`.

The checkpoint's README lists the TTS voices and languages and more system
prompts (translation, audio questions, tool calls). Audio questions and tool
calls use the plain assistant prompt above, with no `<think></think>` prefill.

## Generation

`generate_audio` runs one sampling loop over text and speech, the reference
`generate` state machine. By default it starts in text and the model opens
speech itself: `<|audio_start|>` for TTS, `<|mixed_start|>` for speech chat.
`generation_mode="interleaved"` appends `<|mixed_start|>`: the answer
alternates 6 text tokens and 12 speech frames until `<|text_end|>`, then
speaks the rest. `generation_mode="audio"` appends `<|audio_start|>` and
speaks until an end-of-speech frame. A prompt that already ends in either
marker starts in that mode. `generation_mode="text"` forbids both markers, so
the answer stays in text. If the model answers in text only, the audio is
empty. `max_tokens` counts slots: a text token or a speech frame
(80 ms). The result holds the text, `audio_tokens` [12, frames], and the
waveform at 24 kHz.

Speech codes default to the model card's `audio_temperature=1.0`,
`audio_top_k=128` and `audio_repetition_penalty=1.1` over the last 4096
frames (`Model.audio_sampling`). Text is greedy unless
`temperature` is set, with `repetition_penalty` over the last 32 text tokens.
Pass these to `generate_audio`, or with `--gen-kwargs` on the command line.
`model.stream_slots(...)` yields each slot as it is sampled. The MaskGIT
reveal order is random even when decoding greedily; `frame_order` fixes it.

Plain `generate` is text only: it stops if the model starts speech
(`<|audio_start|>` or `<|mixed_start|>`), and never samples the speech
placeholders. The model card instead passes `suppress_mixed=True` and
`suppress_audio_out=True` for ASR and other text-only output; `generate_audio`
and `stream_slots` accept both. LFM3-Audio picks the voice from the system
prompt.

Earlier speech in a conversation is given as codes: put one `<|audio_out|>`
per clip in the text and pass `audio_out_codes`, one [frames, 12] array per
clip (`result.audio_tokens.T`); the processor adds each clip's end-of-speech
frame. The codec encoder is not ported, so speech history cannot be given as
waveforms. Batches hold one prompt, and streaming input and output are not
implemented, including the card's LiveASR (`Perform live ASR.`, English only).

## Conversion and quantization

`python -m mlx_vlm convert -q` quantizes only the language backbone:
`Model.quant_predicate` keeps the audio encoder, projector, depth decoder and
codebook embedding in bf16, and the codec folder is copied unchanged. On
the earlier preview weights, quantizing the depth decoder to 4 bits made its
greedy codes agree with float32 56% of the time, and the audio tower and
projector moved the speech features by 28%, to save 0.12 and 0.16 GB. Mixed `--quant-predicate` recipes are not
supported for LFM3 in this tree (they find no `down_proj`); use plain `-q`.

Use 8 bits by default; 4 bits is the small option and loses accuracy.
Measured on an M3 Ultra against float32:

| Weights | Size | Last logits error, top-20 | Test clip WER (two prompts) | 418 s clip WER | Speech codes as float32 | Tokens/s | Speech slots/s |
|---|---|---|---|---|---|---|---|
| bf16 | 3.96 GB | 7.0e-3, 20/20 | 0 / 0 | 8.7% | 92% | 130 | 90 |
| 8-bit | 2.33 GB | 1.2e-2, 20/20 | 0.091 / 0 | 0 | 91% | 195 | 114 |
| 4-bit | 1.47 GB | 1.4e-1, 18/20 | 0.182 / 0.091 | 8.8% | 68% | 227 | 117 |

WER is against the float32 transcript. "Speech codes as float32" is the
share of depth-decoder greedy codes that match float32 on the same speech
prompt. On the 418 s clip, float32 and 8 bits write the whole transcript and
then repeat its last 133 words (1812 tokens); the reference implementation
picks the same token at every one of those positions. bf16 and 4 bits happen
to stop without repeating, hence their 8.7% and 8.8% against float32 (0.07%
and 0.14% against a Whisper transcript). Long transcripts need `max_tokens`
of about 1900. TTS of three sentences with the default sampling transcribes
back exactly at every precision. Speeds are best of three on a shared
machine; prompts run at about 3000 tokens per second. 6-bit and mxfp4 were
not measured on these weights.

## Validation

On an Apple M3 Ultra with MLX 0.32.4, against float32 outputs of the reference
implementation for `LiquidAI/LFM3-Audio-2B`:

- Prompts of all six fixture conversations tokenize to the reference ids.
- Float32 last logits: relative error 2.4e-6 on the ASR prompt (114 tokens)
  and 2.2e-6 on a 418 s clip (5244 tokens), with the same top-20; 1.7e-6 on
  two multi-turn speech-chat prompts with prior speech frames, and 1.3e-6
  when the earlier answer is cut off mid-speech.
- Greedy ASR: identical tokens on both reference ASR prompts; greedy
  text-only chat about a clip: identical tokens (48), with the suppress
  flags or with `generation_mode="text"`.
- With the reference's MaskGIT orders, float32 TTS and speech chat (48, 232
  and 157 slots) give the same text tokens, speech codes and mode switches
  slot for slot. The decoded TTS waveform differs by 2.4e-6 (relative).
- bf16 last logits differ from float32 by 0.70% (the reference's own bf16
  on CPU: 0.71%). Greedy ASR gives the float32 tokens on both prompts; the
  reference's own bf16 changes one word of the first. TTS transcribes back
  to its text. Over eight speech-chat answers sampled with the default
  settings, each ends with an end-of-speech frame (the longest is 169
  frames), and the speech transcribes to its text with 2.5% WER on average.

These checks pass the clips as float arrays. A file path is read with
`mlx_audio`, which rounds float WAV samples to 16-bit steps (up to 4.9e-5);
that can flip a near-tied token.
