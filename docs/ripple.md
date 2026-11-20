# Ripple-65B Preview

oMLX recognizes the official BF16 `ricursive` checkpoint and builds its MLX architecture locally. Select the model in the manager to serve it, or use it as the source in oQ Quantization. On a 128 GB Mac calibration can use a temporary 4-bit proxy. oQe still applies activation-weighted rounding. Final code refinement is skipped when the source needs a proxy and no streamed refinement implementation is available.

The MLX architecture is adapted from the Apache-2.0 [OrcaRouter implementation](https://huggingface.co/orcarouter/Ripple-65B-abliterated-MLX/blob/f05d1fd49e6ff40388db462f1379140dedd2768a/Ripple-65B-abliterated-MLX-64GB/ricursive.py) of [Ricursive Ripple](https://huggingface.co/ricursiveAI/Ripple-65B-Preview).
