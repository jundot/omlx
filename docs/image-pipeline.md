# Standalone Qwen Image 2.1 Turbo generation

PR 4 of the series adds a complete local prompt-to-PNG path on top of the native
conditioning/DiT components. No engine, route, or model discovery activation is
included. Execute one pipeline at a time in a dedicated process.

```
python -m tools.generate_image BUNDLE 'A small yellow sailboat on a calm turquoise lake, mountains in soft morning light, watercolor illustration.' output.png --seed 19
```

The scope is one 1024×1024 PNG, fixed eight-step Euler schedule and guidance 1.
Fresh text conditioning runs in FP32, denoising in BF16 and VAE decode in FP32.
Noise uses local NumPy PCG64 state; seed identity does not imply parity with other
runtimes. Existing output files are not overwritten. Unsupported options are not
exposed. VAE architecture, normalization and all image tensor names/shapes/dtypes
are validated before weight loading. Unused temporal tensors remain in the
artifact and are not interpreted as video support.

## Evidence, 2026-10-20

M3 Max / 64 GB, Python 3.12, MLX 0.32.4, pinned mlx-vlm 0.7.8, transformers 5.17.0.
`MLX_ENABLE_TF32=0`. One fresh-prompt run, seed 19:

| Phase | Seconds |
|---|---:|
| Fresh encoding | 23.710 |
| Denoiser load | 1.971 |
| Eight denoising steps | 51.237 |
| VAE load and decode | 5.838 |
| Total including cleanup between stages and PNG encoding | 83.068 |

MLX peak allocation was 15,619,400,728 bytes, covering this entire process run,
including encoding. This is allocator peak, not RSS or a warmed throughput
benchmark. The 1024-square RGB output visibly contains the requested yellow
sailboat, turquoise lake and mountains. It is not an image-quality evaluation.

The independent PyTorch CPU decoder oracle uses NCHW functional operations and
transposes the verified exported kernels back to OIHW. It imports no MLX model
code. On deterministic 2×2 latents, the midpoint, all five up blocks and final
32×32 pixels passed the predeclared FP32 gates (relative L2 < 0.005 and cosine >
0.99999). Final pixel relative L2 was 6.3128e-6, cosine 0.99999999998.

Reproduce in separate reference and native environments:

```
python -m tools.reference_image_vae BUNDLE reference.npz
python -m tools.validate_image_vae BUNDLE reference.npz report.json
```

PyTorch is reference tooling only, not a new serving dependency. This check
establishes independent execution of the decoder equations, not official
Diffusers end-to-end parity. Source/export inverse-layout byte checks passed
before source-weight cleanup. Broader sizes, seeds, precision combinations and
image/video editing are unsupported or unvalidated.

Cancellation checks and synchronous GPU drain are present, but server lifecycle,
leases, shared-executor fairness and chat coexistence are PR 5 gates. The
standalone pipeline must not be directly dispatched as one long server executor
call. Serving activation belongs to PR 6 after lifecycle validation.
