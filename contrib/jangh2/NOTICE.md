# Implementation provenance

The bridge/reference code was written for this task. It calls installed
dependencies rather than vendoring their implementations. Upstream packages
retain their original licenses; nothing here relicenses model weights.

The contribution includes a self-contained Metal source generator, bounded CPU
packed-row cache, model-scoped chat-policy hooks, exact CPU row-copy
optimisations, transactional launcher cleanup, output diagnostics, opt-in
grouped packed expert kernels, exact bounded GPU PLE row caching, model-aware
tuning, workload/cancellation tests, fixed-token whole-model comparison, and
profiling and reproducibility tooling. It does not vendor upstream JANGH
native kernel files; the existing optional native integration is retained.
No speculative decoding/MTP is enabled. Generated Metal sources use MLX's
application-level custom-kernel interface; no MLX core fork.

Primary references inspected on 21 October 2026:

- Model format/config: https://huggingface.co/OsaurusAI/Allosaurus-v0.1-125B-A6B-JANGH2/blob/main/config.json
- JANGH numeric format and fixed codebooks: https://github.com/jjang-ai/jangq/blob/main/jang-tools/jang_tools/jangh/format.py
- Native expert implementation: https://github.com/jjang-ai/jangq/blob/main/jang-tools/jang_tools/jangh/switch.py
- Existing installer (MLX-LM-only SwitchGLU check): https://github.com/jjang-ai/jangq/blob/main/jang-tools/jang_tools/jangh/install.py
- MLX-VLM loader: https://github.com/Blaizzy/mlx-vlm/blob/v0.7.8/mlx_vlm/utils.py
- MLX-VLM expert classes: https://github.com/Blaizzy/mlx-vlm/blob/v0.7.8/mlx_vlm/models/switch_layers.py
- MLX-VLM PLE language model and storage: https://raw.githubusercontent.com/Blaizzy/mlx-vlm/v0.7.8/mlx_vlm/models/qwen4_exp/language.py
  and https://raw.githubusercontent.com/Blaizzy/mlx-vlm/v0.7.8/mlx_vlm/models/qwen4_exp/ple_storage.py
- Qwen4 compatibility implementation: https://github.com/jundot/omlx/tree/v0.7.1/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp
- MLX module parameter semantics: https://github.com/ml-explore/mlx/blob/v0.32.4/python/mlx/nn/layers/base.py
- MLX custom-kernel interfaces: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html,
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html,
  https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html,
  https://ml-explore.github.io/mlx/build/html/usage/indexing.html
- MLX custom-kernel backend behaviour: https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/mlx/backend/metal/custom_kernel.cpp
- JANGH kernel reference: https://raw.githubusercontent.com/jjang-ai/jangq/main/jang-tools/jang_tools/jangh/kernels.py

The indexed PLE reader and its tests were written for this task. They do not
copy or replace the upstream architecture. They keep stock n-gram hashing and
use the installed MLX affine dequantizer for selected rows.

## Scheduler adapter extension

`scheduler_compat.py` also converts `mlx_vlm`-native caches for non-JANGH2
VLM checkpoints served by the same process (Qwen3.5 family: Qwen3.6/Qwen3.8
hybrids), which stock mlx-lm cannot batch, and retains native hybrid cache
objects and their original list through `_wrap_caches`. The conversion mirrors
mlx-lm's `to_batch_cache()` using MLX-VLM's own batch cache classes:
- https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.2/mlx_lm/generate.py
- https://raw.githubusercontent.com/Blaizzy/mlx-vlm/v0.7.8/mlx_vlm/models/cache.py

Additional primary references inspected:
- https://raw.githubusercontent.com/Blaizzy/mlx-vlm/v0.7.8/mlx_vlm/prompt_utils.py
- https://raw.githubusercontent.com/jundot/omlx/refs/pull/364/head/omlx/engine/vlm.py
- https://raw.githubusercontent.com/jundot/omlx/refs/pull/364/head/omlx/scheduler.py
- https://raw.githubusercontent.com/jundot/omlx/refs/pull/364/head/omlx/server.py
- https://raw.githubusercontent.com/jundot/omlx/refs/pull/364/head/omlx/api/thinking.py
- https://huggingface.co/OsaurusAI/Allosaurus-v0.1-125B-A6B-JANGH2

Pull-request references describe reviewed interfaces, not a guarantee that
future PR revisions remain compatible. Local check_chat.py validates the
installed API.

References to `main` identify inspected interfaces, not an immutable
deployment pin. The default implementation does not fetch or execute anything
from those URLs. Optional native-wheel instructions pin jang 2.5.50, and the
adapter independently verifies its codebook values. The full installed native
wheel was not executed in this Linux development environment.

CPU checks and host C++ helper execution do not qualify Apple GPU kernels. See
VERIFICATION.md for exactly what was and was not executed.

## Reference arithmetic alignment

The packed kernels apply scales before accumulation and use the FP32
reference GEMV's column assignment and reduction order. The fused activation
uses the same precise exponential as MLX's elementwise exponential. These
changes prevent small rounding differences from changing later expert routing.
The installed MLX 0.32.4 headers were inspected for those arithmetic contracts:
- https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/gemv.h
- https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/unary_ops.h

Grouped indexing deliberately coalesces duplicate destinations; indexed updates
must not rely on an arbitrary winner. Device-cache hit detection still requires
a scalar CPU rendezvous. CPU tests and host-C++ syntax checks are not Metal
validation.
