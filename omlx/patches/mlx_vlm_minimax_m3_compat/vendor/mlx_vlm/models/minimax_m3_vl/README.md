# MiniMax M3

MiniMax M3 is supported as a native multimodal model in mlx-vlm. The
implementation covers text, image, and video prompts, MiniMax Sparse Attention
(MSA), MiniMax thinking tags, MiniMax XML-style tool calls, MXFP8 config loading,
and EAGLE-3 speculative decoding with the released drafter.

## Convert

Use `mlx_vlm.convert`, not `mlx_lm.convert`, because the model type is
`minimax_m3_vl`.

```sh
mlx_vlm.convert \
  --hf-path MiniMaxAI/MiniMax-M3 \
  --mlx-path ~/MiniMax-M3-4bit \
  --quantize --q-bits 4 \
  --trust-remote-code
```

For the released MXFP8 checkpoint, keep the original quantization format:

```sh
mlx_vlm.convert \
  --hf-path MiniMaxAI/MiniMax-M3-MXFP8 \
  --mlx-path ~/MiniMax-M3-MXFP8 \
  --trust-remote-code
```

Local converted MiniMax M3 folders load their bundled processor code
automatically. Pass `--trust-remote-code` when loading directly from Hugging
Face or when converting from the Hub.

## Generate

Text:

```sh
mlx_vlm.generate \
  --model ~/MiniMax-M3-4bit \
  --prompt "Write a poem on LLMs" \
  --max-tokens 256
```

Image:

```sh
mlx_vlm.generate \
  --model ~/MiniMax-M3-4bit \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

Video:

```sh
mlx_vlm.generate \
  --model ~/MiniMax-M3-4bit \
  --video ./clip.mp4 \
  --fps 2.0 \
  --prompt "Summarize this video." \
  --max-tokens 256
```

MiniMax M3 supports template-level thinking control:

```sh
mlx_vlm.generate \
  --model ~/MiniMax-M3-4bit \
  --thinking-mode enabled \
  --prompt "Solve this step by step: 17 * 23" \
  --max-tokens 512
```

`--enable-thinking` is accepted as a shortcut for `--thinking-mode enabled`.

## EAGLE-3

The released EAGLE-3 drafter is auto-detected from its architecture, but passing
`--draft-kind eagle3` is explicit and stable:

```sh
mlx_vlm.convert \
  --hf-path Inferact/MiniMax-M3-EAGLE3 \
  --mlx-path ~/MiniMax-M3-EAGLE3

mlx_vlm.generate \
  --model ~/MiniMax-M3-4bit \
  --draft-model ~/MiniMax-M3-EAGLE3 \
  --draft-kind eagle3 \
  --draft-block-size 3 \
  --prompt "Explain MiniMax Sparse Attention in one paragraph." \
  --temperature 0 \
  --max-tokens 256
```

## Notes

- Unlike the current experimental GGUF path, this implementation loads the
  native multimodal checkpoint format and supports text, image, and video
  prompts. It also implements MSA in MLX; it does not vendor MiniMax's CUDA/SM100
  fused kernel.
- MSA is implemented functionally in MLX with block scoring, top-k sparse block
  selection, local block handling, and a MiniMax index-key side cache. The CUDA
  kernels in `MiniMax-AI/MSA` target NVIDIA CUDA/SM100 and are not vendored.
- Decode uses dense attention until the selected sparse KV window is smaller
  than the full cache. With the public M3 config this means short prompts stay
  dense; compact sparse decode starts to help on long contexts where the
  selected `sparse_topk_blocks * sparse_block_size` tokens are at most half of
  the cache by default.
- The public MiniMax M3 BF16 checkpoint config advertises MTP metadata, but its
  weight index does not publish `mtp` or `nextn` weights. This branch does not
  include MiniMax M3 native MTP support; use the released EAGLE-3 drafter for
  speculative decoding.
- Long prompts may need `--prefill-step-size 512` or lower to reduce peak memory.
- On MLX CUDA, use `--quantize-activations` for MXFP8/NVFP4 checkpoints. On
  Apple Silicon, activation quantization is not required for the MiniMax M3
  4-bit path.
