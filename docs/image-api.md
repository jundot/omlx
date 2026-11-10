# Image generation (experimental)

The initial native image backend serves the versioned Qwen-Image-2.1 Turbo
bundle described in [image-import.md](image-import.md). Put the complete MLX
pipeline folder in a model directory; its components are never chat models.
No additional inference package is required. Structural inspection does not
prove numerical compatibility: exact runtime configuration and tensor checks
still run when generating.

Send an authenticated `POST /v1/images/generations` with a model ID or custom
alias, a nonempty `prompt`, and optionally a uint32 `seed`. The only supported
recipe is one 1024×1024 PNG, eight Euler steps, guidance 1.0 and the fixed sigma
schedule. `n=1`, `size="1024x1024"`, and `response_format="b64_json"` are the
only accepted options. Other fields are rejected. The response contains
`created` and `data[0].b64_json`, plus the chosen `data[0].seed` for replay.

The route uses the existing inference API-key policy and physical-model pool
lease. It rejects chat models, while chat endpoints reject image models.
Image models cannot be the chat default, be reclassified with a type override,
or create/apply chat profiles. Existing profile and Claude tier aliases for
image models are omitted from `/v1/models`; direct custom aliases are supported.

One request owns each image engine. A competing generation or insufficient
memory headroom returns 503. A disconnect or pool abort cancels the iterator,
waits for submitted GPU work to drain, closes its resources and then releases
the pool lease. Manual unload uses this same abort path; TTL/eviction cannot
release an active lease. No queue, streaming, image edits, URL output, arbitrary
size, recipe customization or batch generation is provided.

Model management lists the bundle once, including malformed manifests with
structural errors, without presenting its internal components as separate
models. Native and web presentation follow in separate dependent PRs.
