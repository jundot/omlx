# Custom kernels

oMLX has optional native Metal kernels in `omlx/custom_kernels`. They make some models faster. If a kernel is not built, the model uses the normal MLX path instead. That path works but can be much slower and use more memory.

## Which installs have them

| Install | Kernels |
|---|---|
| macOS app (DMG) | included |
| Homebrew | only with `--with-custom-kernel` |
| `make install`, `make dev` | built |
| `make install-no-kernels`, `pip install -e .` | not built |

Building them needs full Xcode with the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`).

## Layout

Kernels are grouped by what they do, not by model.

```
omlx/custom_kernels/
  attention/       attention kernels
  sparse_index/    indexer and top-k kernels
  quant_matmul/    quantized matmul kernels, ANE offload
  lowbit/          1-bit, 2-bit and T5 kernels
  moe_staging/     MoE offload read buffers (native part is in quant_matmul)
  include/omlx/    shared C++ helpers
  _loader.py       loads each extension
  nax.py           M5 tensor unit (NAX) check
```

| Package | What goes here | Used by |
|---|---|---|
| `attention` | attention over KV (sparse, block-mask, FA-256, decode SDPA) | GLM, DeepSeek V4/V4.1, Qwen3.5/3.6, Qwen4 |
| `sparse_index` | choosing which KV blocks to attend to (indexer scores, top-k) | GLM, DeepSeek V4, Qwen4, MiniMax M3 |
| `quant_matmul` | matmuls on quantized weights, MoE combine, ANE offload | Qwen3.5/3.6, GLM, DeepSeek V4/V4.1, MiMo, K2 |
| `lowbit` | 1-bit, 2-bit and T5 decode formats | Bonsai |

Each package has:

- `fast.py`: the Python functions to call.
- `csrc/`: the C++ and Metal source.
- `_ext*.so` and `omlx_<package>_kernels.metallib`: the build output.

ANE offload is off by default. See [Qwen ANE/GPU prefill](experimental/qwen35_ane_prefill.md).

Some kernels are not in this directory. Small JIT kernels (`mx.fast.metal_kernel`) live next to the model code in `omlx/patches`. They compile at runtime and need no build.

## Build

```bash
make kernels
```

This removes old builds, builds every package and checks that each one loads.

- Build output is not in git. A new checkout or worktree needs its own `make kernels`.
- Run `make kernels` again after you change anything in `csrc/`.
- The M5 (NAX) kernels are built only with macOS SDK 26.2 or newer.

## Check what loaded

```bash
python -c "from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())"
```

The same data is in `GET /api/status` under `custom_kernels`.

## How loading works

- If a package is not built, its kernels are skipped and the model uses MLX.
- If a build fails to load, the server log shows a warning.
- Code checks `fast.has_symbol(name)` before it uses a kernel.

## Adding a kernel

1. Put it in the package that matches what it does.
2. Name it after what it does, not after a model.
3. Add the source to `csrc/`, bind it in `csrc/bindings.cpp` and list it in `setup.py`.
4. Add a wrapper in `fast.py`.
5. Call it from the model patch only when `has_symbol()` is true.
6. Add a test that compares it with MLX and skips when the kernel is missing.

A wrapper in `fast.py` and its caller look like this:

```python
# quant_matmul/fast.py
def my_qmm(x, w, *, stream=None):
    if _ext is None or not hasattr(_ext, "my_qmm"):
        raise RuntimeError("my_qmm native kernel is unavailable")
    return _ext.my_qmm(x, w, **_native_stream_kwargs(stream))


# model patch
from omlx.custom_kernels.quant_matmul import fast

if fast.has_symbol("my_qmm"):
    y = fast.my_qmm(x, w)
else:
    y = mx.matmul(x, w)
```

## Testing

CI does not build the kernels, so kernel tests skip there. Run them on a Mac after `make kernels`. Test NAX paths on an M5.

## Troubleshooting

- After the kernel files move, run `make clean-kernels` and then `make kernels`.
- If a package shows `available: False`, read its `import_error`.
