# Native Qwen Image 2.1 components (experimental)

This is the third unit of the image-model series, dependent on the structural
inspector. It does not register an engine or make an image model serveable.

The only accepted runtime layout is `native-bf16-f32-v1`, with canonical component
paths and the fixed Turbo recipe. The encoder loader validates all 399 tensor
names, shapes and storage dtypes; the denoiser validates all 265 and its fixed
architecture. These checks are separate from the inspector's structural result.

`encode_prompt(bundle, prompt)` tokenizes a fresh prompt locally using the recorded
Qwen template, removes the 14-token system prefix, and executes all 36 encoder
layers in FP32. It rejects blank prompts and encoded sequences over 512 tokens
instead of silently truncating. It streams layer weights rather than keeping
the 8B encoder resident. The retained language-model head is not evaluated.
This prioritizes memory and numerical clarity; it is not a throughput claim.

`load_transformer(bundle)` strict-loads the BF16 denoiser. Its forward uses causal
text attention, image attention over text and image tokens, the three-axis rotary
positions and shared modulation. Cancellation is checked at encoder/DiT layer
boundaries. This is cooperative CPU control: it does not interrupt a submitted
Metal operation and is not a serving lifecycle guarantee.

The adapted denoiser equations carry their original MIT/Apache attribution in
`omlx/image/LICENSE`, included in package data. No MFLUX package is imported.

## Validation interface

```
python -m tools.validate_image_conditioning BUNDLE INDEPENDENT_REFERENCE_DIRECTORY REPORT.json
```

Reference files must come from the separate source-layout execution used in the
conversion study. The tool checks token IDs, every encoder layer and final
conditioning for three prompts, including Chinese text. It never regenerates its
own expected arrays. Gates: relative L2 < 0.005 and cosine > 0.99999.

Small tests check prompt rejection, rotary complex arithmetic and cancellation.
Real-checkpoint results and exact dependency versions belong in the PR description.
A passing encoder comparison does not establish VAE, full-image or server safety.

Remaining work: VAE and full fresh-prompt runner (PR 4), independent VAE numerical
validation, total memory measurements, then engine lifecycle and API activation.
Do not activate this module in model discovery before those gates pass.
