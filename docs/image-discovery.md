# Image pipeline discovery boundaries

Image generation is not supported by this change. An image bundle containing
text encoder, denoiser and VAE directories must not appear as three chat models.
Discovery skips the bundle before descending into its components, including
when the configured model directory is the bundle itself.

Recognized boundaries are:

- An explicit `omlx-pipeline.json` entry, even if it is malformed or unsupported.
- Legacy `pipeline.json` identifying `qwen_image_2_1_turbo_text_to_image`.
- A `model_index.json` declaring a recognized Qwen image pipeline class.
- A known Qwen image-only component class, or the custom
  `qwen3_vl_image_text_encoder` model type, when supplied on its own.

The guard also runs before GGUF registration and on resolved Hugging Face cache
snapshots. It logs that image serving is unavailable. It neither deletes files
nor loads model weights. Directory names such as `transformer` or `vae`, an
unrelated `model_index.json`, and an ordinary language-model encoder config do
not establish image ownership. Consequently an unmarked encoder packaged as an
ordinary language model cannot be distinguished automatically.

The release format uses `model_index.json`; the explicit oMLX sidecar remains
a legacy research boundary only. A future inspector/runtime can validate the
model index and component metadata, but this guard does not
introduce a serving schema, image engine or inference dependency. It complements
architecture support checks: an individually loadable encoder can still belong
to an unsupported image pipeline.
