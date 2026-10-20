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

Version 1 sidecar example:

```json
{
  "schema_version": 1,
  "pipeline": "qwen_image_2_1_turbo_text_to_image",
  "layout": "native-bf16-f32-v1",
  "recipe": "qwen-image-2.1-turbo-euler-8-v1",
  "processor": "processor",
  "components": {
    "text_encoder": {"path": "text_encoder"},
    "transformer": {"path": "transformer"},
    "vae": {"path": "vae", "auxiliary": ["temporal_auxiliary.safetensors"]}
  },
  "sha256": {}
}
```

Paths must resolve inside the selected bundle. Internal links are allowed;
physical weight bytes count each inode once. Direct HF-cache symlinks to external
blob directories are not supported. Components retain their original configs and
weight indexes. Hash keys are canonical bundle-relative weight paths. Only
BF16/F32 tensors are accepted by this experimental layout inspector.

For the existing research export, from the checkout:

```
python -m tools.migrate_image_bundle /path/to/bundle
```

Migration checks the legacy pipeline and fixed eight-step recipe, then atomically
creates the sidecar without replacing an existing one or changing any weights.
It is not a GGUF converter. Its recipe identifier is reserved for the recorded
conditioning/sampling implementation; the inspector does not execute that recipe.

## Review units and evidence (2026-10-20)

Base: `8b7e7018` from refreshed upstream/main, including GGUF/Splash discovery.

1. Pipeline boundary recognition and discovery guards, with independent discovery
   regression tests. The direct-root/flat component and GGUF-precedence cases fail
   on the base (three failures) and pass with the guard.
2. Structural inspector, CLI and research metadata migration, with offline tests.
   This is an experimental part of planned PR 2, not completion of every acceptance
   case in the research plan.

Validation in the existing Python 3.12 environment:

```
python -m pytest tests/test_model_discovery.py tests/test_image_discovery.py tests/test_image_import.py -q
# 250 passed
python -S -m unittest discover -s tests -p test_image_import.py -q
# 17 passed
```

Before authorized cleanup, real-export inspection counted 399 encoder, 265 DiT
and 238 VAE tensors and 31,287,920,248 physical weight bytes. Runtime/numerical
status remained unvalidated. Real-export full hashes were not recomputed. The
19 generated safetensors files were subsequently deleted; source checkpoints,
metadata, scripts and historical validation evidence were retained in the
original research directory. Re-export is necessary before further real-weight
validation. This worktree does not contain those model assets.

## Remaining gates

- Exact required tensor names/shapes and cross-component dimension/tokenizer
  compatibility, rather than only internally consistent safetensors/indexes.
- Structured per-component diagnostic codes beyond the current invalid-bundle
  diagnostic and file/tensor details.
- Deterministic mutation/race and cancellation tests; concurrent adversarial path
  replacement is not prevented by current containment checks. Runtime loading
  must independently revalidate inputs.
- Full planned discovery matrix and actual GGUF fixtures beyond the existing
  upstream suite and new precedence tests.
- Clean environment validation against all refreshed upstream dependency pins;
  the local existing environment was reused without installing packages.
- Native numerics, VAE oracle, fresh-prompt generation, full peak memory,
  lifecycle/cancellation, API and client integration remain future work.

No inference dependency, API route, engine registration, UI, downloader or model
conversion was added. Do not describe these commits as image-serving support or
as completion of the full review acceptance matrix.
