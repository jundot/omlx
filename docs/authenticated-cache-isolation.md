# Authenticated API-key KV cache isolation

Set `auth.cache_isolation` to `"api_key"` in `settings.json` and restart the server to isolate reusable KV state between verified API keys. The default is `"shared"`, preserving existing reuse.

The main key and each sub-key receive a stable, domain-separated SHA-256 namespace derived after authentication. Requests capture this identity before entering scheduler or worker threads. A client-supplied tenant header cannot select another namespace. Requests using the same credential can reuse their KV state; rotating a credential starts a new namespace.

The namespace applies from token zero and composes with image keys, including segmented media keys. It is also carried through SpecPrefill draft/static-prefix state, exact split-GDN checkpoints, and supported MTP prefix sidecars. Existing shared entries remain on disk and cannot satisfy isolated requests.

This option covers reusable KV state in the batched LLM and standard VLM engines. It does not provide separate memory budgets, scheduling, Responses history or vision preprocessing caches. Standalone DFlash, diffusion and distributed inference are unsupported and fail closed. Disabling verification or permitting unauthenticated inference is incompatible with this mode. No admin UI toggle is added; configuration is manual.

Local validation covers concurrent authentication, EngineCore worker handoff, hot/cold prefix reuse and sidecar/static-prefix key ownership on MLX CPU. Production model/GPU qualification has not been performed. Source and changes are recorded in `provenance/authenticated-cache-isolation.json`.
