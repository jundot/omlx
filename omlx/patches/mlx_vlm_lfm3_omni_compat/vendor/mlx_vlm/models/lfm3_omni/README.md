# LFM3-Omni on MLX

LFM3-Omni-2B is [LFM3-Audio](../lfm3_audio/README.md) with the
[LFM3-VL](../lfm3/README.md) vision tower and projector: it reads images,
videos, speech and text, and answers in text, speech, or both. The audio
encoder, depth decoder, Liquid codec (`audio_tokenizer/`) and the text/speech
generation loop are LFM3-Audio's; the vision path is LFM3-VL's. The checkpoint
loads directly; no remote code or PyTorch is needed. The implementation adapts
the Apache-2.0 reference code shipped with the checkpoint; see
[LICENSE](../lfm3/LICENSE).

`LiquidAI/d1-omni-2B` shares the LFM3-Omni `model_type` but is a decision
model: it loads as `d1_lfm3_*` and answers through `model.system_one(...)`
or `python -m mlx_vlm.decide`, not `generate`; see
[d1 on MLX](../d1_lfm3_vl/README.md).

```python
from mlx_vlm import generate, load
from mlx_vlm.generate import generate_audio
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("path/to/LFM3-Omni-2B")

def chat(system, user="", **media):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return apply_chat_template(processor, model.config, messages, **media)

# Image question
prompt = chat(
    "Answer questions about the provided image. Base every answer only on what "
    "can be seen. Respond with text.",
    "What shapes and words do you see?",
    num_images=1,
)
print(generate(model, processor, prompt, image=["shapes.png"], temperature=0).text)

# Video question (frames are sampled at 2 fps)
prompt = chat(
    "Answer questions about the video using what you see and hear. Respond with text.",
    "What happens in this video?",
    video="clip.mp4",
)
print(generate(model, processor, prompt, video=["clip.mp4"], temperature=0).text)

# Speech recognition
prompt = chat("Perform ASR in English.", num_audios=1)
print(generate(model, processor, prompt, audio=["speech.wav"], temperature=0).text)

# Text to speech
prompt = chat(
    "Perform TTS in American English using a female voice.",
    "The quick brown fox jumps over the lazy dog.",
)
generate_audio(model, processor, prompt, output_audio_path="tts.wav")

# Speech chat about an image: interleaved text and speech
prompt = chat(
    "Respond with interleaved text and audio.",
    "What is on the sign?",
    num_images=1,
    num_audios=1,
)
result = generate_audio(
    model, processor, prompt, image=["sign.png"], audio=["question.wav"],
    max_tokens=1024, output_audio_path="answer.wav",
)
print(result.text)
```

From the command line:

```sh
# Image or video question (--video clip.mp4 for a video)
python -m mlx_vlm.generate --model path/to/LFM3-Omni-2B --temperature 0 \
  --system "Answer questions about the provided image. Base every answer only on what can be seen. Respond with text." \
  --image shapes.png --prompt "What shapes and words do you see?"
# ASR
python -m mlx_vlm.generate --model path/to/LFM3-Omni-2B --temperature 0 \
  --system "Perform ASR in English." --prompt "" --audio speech.wav
# TTS
python -m mlx_vlm.generate --model path/to/LFM3-Omni-2B \
  --system "Perform TTS in American English using a female voice." \
  --prompt "Hello there!" --output-modality audio --output tts.wav
# Speech chat with an image
python -m mlx_vlm.generate --model path/to/LFM3-Omni-2B \
  --system "Respond with interleaved text and audio." --prompt "" \
  --image sign.png --audio question.wav --output-modality audio \
  --max-tokens 1024 --output answer.wav
```

`python -m mlx_vlm generate_audio ...` is the same as
`python -m mlx_vlm generate --output-modality audio ...`. Speech output needs
`--output` (a `.wav` file) and does not work with `--chat`.

## Prompts and generation

The prompt ends with `<think></think>`, so the model starts in text and
chooses its output modality itself. Without an instruction it often opens
speech chat (`<|mixed_start|>`), which ends plain `generate` with no text:
for text answers, say "Respond with text." in the system prompt, as above.
`generate_audio` follows whatever the model chooses; its options
(`generation_mode`, `audio_temperature=0.4`, `audio_repetition_penalty=1.2`,
`frame_order`, speech history as `audio_out_codes`) are LFM3-Audio's, see its
[README](../lfm3_audio/README.md#generation).

The processor wraps every image, video and audio clip in one
`<|vision_start|>…<|vision_end|>` span, whether or not the prompt already has
it; a video adds a span and a `<t seconds>` timestamp per pair of frames.
Video frames need their fps and frame indices: direct processor calls should
pass `video_metadata`, and the video loader picks the frames with the
reference's rule and passes their source indices and fps.

Not implemented: a video's soundtrack (`use_audio_in_video`; pass the audio
as a separate clip instead), streaming input or output, the codec encoder
(earlier speech can only be given as codes), and `enable_thinking`, which the
checkpoint does not support.

## Conversion and quantization

`convert -q` quantizes only the language backbone. Its multimodal skip keeps
the vision tower, vision projector and audio encoder in bf16,
`Model.quant_predicate` keeps the audio projector, depth decoder and codebook
embedding, and the codec folder is copied unchanged.

Measured against the float32 reference on the fixture prompts (M3 Ultra):

| Variant | Weights | Last logits | 32 greedy tokens, image / ASR / video | Depth argmax on own hidden | Decode tok/s |
|---|---|---|---|---|---|
| bf16 | 4.33 GB | 0.35–1.06% | differs at 5 / identical / differs at 23 | 93% | 133 |
| 8-bit | 2.71 GB | 0.57–1.01% | differs at 5 / 6 / identical | 92% | 187 |
| 4-bit | 1.85 GB | 5.5–8.9% | differs at 14 / 6 / 4 | 64% | 210 |

Use 8 bits by default: the image answers stay correct, and ASR misspells one
name (WER 0.048 against float32). At 4 bits the image answers stay correct
but ASR loses words (WER 0.190, "Hor ripe" for "Hallway"). Three TTS
sentences transcribe back exactly at every precision; speech-chat quality at
4 bits is not measured. Speeds are best of three on a shared machine. Mixed
`--quant-predicate` recipes are not supported (LFM3 has no `down_proj`); use
plain `-q`. On the earlier preview weights, 6 bits saved only 0.43 GB over 8,
and quantizing the vision tower was not worth it: 8 bits saved 0.15 GB and
moved its projector output 1.5× the bf16 noise; 4 bits moved it by 51%.

A single pass over long audio drops content: on a 418 s clip every variant,
float32 included, transcribes about 60% of the words, so split long
recordings.

## Validation

On an Apple M3 Ultra with MLX 0.32.4, against float32 outputs of the reference
implementation for `LiquidAI/LFM3-Omni-2B` on six fixture prompts (two
images; speech; an image and a clip; an 8-frame video; TTS; speech chat):

| Check | float32 | bf16 (reference's own bf16) |
|---|---|---|
| Prompt ids, grids, feature masks | identical | identical |
| Vision tower output, image / video | 4.2e-6 / 5.8e-6 | 2.2% / 3.0% (1.9% / 2.2%) |
| Vision projector, image / video | 4.9e-6 / 7.9e-6 | 2.4% / 4.7% (2.6% / 3.5%) |
| Audio projector, two clips | 3.7e-6 / 1.4e-6 | 1.1% / 0.9% (1.0%) |
| Last logits, image / ASR / mixed / video | 1.1e-6 / 2.2e-6 / 9.8e-7 / 9.5e-7 | 0.36% / 1.06% / 0.35% / 0.38% (0.34% / 0.44% / 0.39% / 0.33%) |
| Last logits top-20 | identical, in order | 20 / 20 / 20 / 19 of 20 (20 / 20 / 20 / 20) |
| 32 greedy tokens, image / ASR / video | identical | differs at 5 / identical / differs at 23 (identical / identical / differs at 23, same text) |
| TTS and speech chat, 24 and 48 slots, reference MaskGIT orders | identical modes, tokens and codes | same modes; same TTS text, different speech-chat text; depth logits 0.53% / 0.56% (0.49%) |
| Decoded TTS waveform | 2.5e-6 | |

Errors are relative (‖mlx − ref‖ / ‖ref‖). Pixel inputs match the reference
to 1.2e-7 for images and 1/255 for video frames (resize rounding); log-mel
features to 5.7e-4 (FFT rounding near the clamp). bf16 speech codes diverge
from float32 after a few frames, as the reference's own bf16 does. From the
command line in bf16, the image, video, ASR, TTS (transcribed back exactly)
and speech-chat-with-image examples above answer correctly.
