# ECO paged attention

An optional inference extension for official MLX **0.32.2** on Apple Silicon.
Build with `uv pip install ./extensions/eco-paged-attention` in the intended
environment. Xcode Metal tools, CMake and a C++ compiler are required.

The kernel reads pages shaped `[pages, kv_heads, page_size, D]` for single-token
queries, with D in 64/128/256 and query/KV head ratios 1/2/3/4/5/6/8/12/16.
Matching float32/float16/bfloat16 inputs are supported. These are activation/KV
dtypes; weight quantization is handled by MLX-LM before this entry point.
The package owns its primitive, two-pass Metal kernels and temporary buffers.
The reduction uses the MIT-licensed MLX SDPA header at build time. MLX core does
not need source changes. The C++ array-descriptor bridge depends on MLX internal
ABI; the exact dependency pin must be requalified before changing MLX versions.

`paged_attention` checks shapes, dtypes and metadata, including positive context
lengths and physical page bounds. Metadata checks synchronize with the CPU.
The private `_paged_attention` path is for the cache manager, which owns and
validates page mappings; it omits the CPU metadata read. Live page ownership must
outlast scheduled reads. This entry point is not a public unchecked-input API.

The candidate server enables it with `--native-paged-kv-pages 128 --native-paged-attention-mode mlx`.
Prefill and custom masks use gather plus SDPA. Causal decode uses page lengths
and a cache-generated mask contract. Unsupported model geometry is rejected at
opt-in initialization. Contiguous caching and gather-only paging remain usable
without installing this package.

The current scope is local inference. CPU execution, training/autodiff, vmap,
compiled transforms, distributed execution and speculative decoding are outside
this candidate's qualification. See the branch tests and benchmark for numerical,
stream, cache lifecycle and warm decode evidence.
