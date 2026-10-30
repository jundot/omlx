# Image bundle inspection (experimental)

Image pipelines are not serveable by this change. Discovery skips explicit
`omlx-pipeline.json` boundaries (even malformed ones), recognized Qwen image
pipeline metadata, and known Qwen image-only component classes. It does not
classify directories by name or suppress ordinary chat/GGUF models. A standalone
text encoder without pipeline ownership metadata remains indistinguishable from
a standalone language model.

## Read-only command

```
omlx inspect-image /path/to/bundle --json
omlx inspect-image /path/to/bundle --json --verify-hashes
```

The command uses only Python's standard library. It reads bounded metadata and
safetensors headers; weight payloads are read only for explicit SHA-256 checking.
Exit 0 means structural validity (and matched manifest hashes when requested).
Missing hash references give a nonzero integrity-check exit. Manifest hashes
establish consistency, not trusted provenance. No report enables serving.

The primary release contract uses the community component directory layout:

```text
model_index.json
generation_config.json
PROVENANCE.json
checksums.json                 # Optional reference hashes for weight files
text_encoder/{config.json, model*.safetensors, model.safetensors.index.json}
transformer/{config.json, model*.safetensors, model.safetensors.index.json}
vae/{config.json, model.safetensors, temporal_auxiliary.safetensors}
processor/{tokenizer.json, tokenizer_config.json, ...}
scheduler/scheduler_config.json
```

`model_index.json` declares `QwenImage21Pipeline`, descriptive component class
pairs, `mlx_format: true` and `mlx_layout: native-bf16-f32-v1`. Class names are
never dynamically imported. The layout discriminator is essential: a community
MLX repository may use different tensor names or packed quantization. Ordinary
Diffusers loading is not implied by these labels.

`generation_config.json` records recipe `qwen-image-2.1-turbo-euler-8-v1`,
`num_inference_steps: 8`, `guidance_scale: 1.0`, `height: 1024`, `width: 1024`,
and `sigmas: [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534,
0.414568, 0.0]`. `checksums.json` is a map of weight paths to SHA-256 digests.
Provenance is independent of loader configuration; it does not enable serving.

`omlx-pipeline.json` is accepted only for backward compatibility with the
research export, and is not required or generated for new releases. If a model
index exists, malformed or unsupported metadata never falls back to that legacy
sidecar. Low-bit quantization remains unsupported; 4-bit/8-bit names alone cannot
make a bundle compatible. Future quantization must declare per-component settings.

Paths must resolve inside the selected bundle. Internal links are allowed;
physical weight bytes count each inode once. Direct HF-cache symlinks to external
blob directories are not supported. Components retain their original configs and
weight indexes. Hash keys are canonical bundle-relative weight paths. Only
BF16/F32 tensors are accepted by this experimental layout inspector.

For the existing research export, from the checkout:

```
python -m tools.migrate_image_bundle /path/to/bundle
```

Migration checks the legacy pipeline, canonical component paths and fixed eight-step
recipe, then exclusively creates provenance, generation config and model index
(in that order). It publishes the index last and rolls back its new metadata on
failure, without replacing existing files or changing weights. It does not compute
reference hashes; explicit hash verification needs a separately prepared checksums.json.
It is not a GGUF converter. Its recipe identifier is reserved for the recorded
conditioning/sampling implementation; the inspector does not execute that recipe.

## Series placement and validation

This is PR 2 of the image-model series, stacked on PR 1 (#4692). It provides
structural inspection; architecture-specific required tensor names/shapes and
cross-component compatibility checks belong with the native loader in PR 3.
A structurally valid bundle is explicitly reported as runtime/numerically
unvalidated, with serving disabled. No attached numerical report can change that.

Offline fixtures cover malformed/bounded JSON, unsupported schema/recipe/layout,
truncated tensor headers, impossible shapes/offsets, overlaps, escaping paths,
FIFO rejection, index mismatches, hash mismatches, internal-link size accounting,
metadata/weight mutation, cancellation, and non-overwriting migration. The CLI
also runs under `python -S` without site packages or inference imports.

The restored real bundle passed inspection and explicit full-file SHA-256
verification: 399 encoder, 265 denoiser and 238 VAE tensors, totaling
31,287,920,248 weight bytes. The MLX files are retained; original source-weight
copies were removed only after byte/layout parity, original tensor-digest
comparison and native MLX load checks passed. These are artifact checks, not
fresh-prompt generation or serving validation.

Run focused checks with:

```
python -m pytest tests/test_model_discovery.py tests/test_image_discovery.py tests/test_image_import.py tests/test_image_model_index.py -q
python -S -m unittest discover -s tests -p test_image_import.py -q
```

Integrity checking covers weight files. Config/tokenizer files are bounded and
checked for mutation across inspection, but are not authenticated. This command
assumes a locally managed directory, not a malicious process concurrently
replacing parent directories. Runtime loading must independently revalidate
inputs; inspection is neither a lease nor an immutable filesystem snapshot.

Subsequent PRs add native conditioning/DiT, VAE and standalone generation,
then lifecycle, serving, and web/Mac integration. This PR adds no inference
framework, API route, engine registration, downloader or image UI.
