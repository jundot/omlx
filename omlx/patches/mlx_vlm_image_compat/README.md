# mlx-vlm image compatibility

oMLX delegates image model detection, per-model sampling defaults and input
validation to mlx-vlm.

This patch only forces the native Z-Image tokenizer to load locally with
`trust_remote_code=False`, preserving oMLX's safe model-loading default.
