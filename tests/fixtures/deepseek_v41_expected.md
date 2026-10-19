# DeepSeek V4.1 expected outputs

`deepseek_v41_expected.npz` contains 43 small NumPy arrays, not model weights or executable reference code. The archive SHA-256 is `4bb00e3e792b71ac70586648c34cceef954b3d134c3d9a6a5127714753564571`.

The prefill/decode (`prefill*`) and DSpark (`dspark_*`) arrays were recorded from DeepSeek's MIT-licensed [DeepSeek-V4.1-Flash-1017 reference implementation at 1bae0e1088993b9b82de08f59de5be5a34e42170](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) with `record_deepseek_v41_expected.py`. The Engram (`engram_*`, `hash_*`), vision (`vision_*`) and image geometry (`image*`) arrays come from the earlier [reference at df42c109f1defefcbfcedbe7d905718a12266e40](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/df42c109f1defefcbfcedbe7d905718a12266e40); `engram.py`, `vision.py`, `image_processor.py` and `kernel.py` are byte-identical in both revisions.

The tests reconstruct synthetic weights with `load_reference_weights`: NumPy PCG64 seeded by the case seed plus CRC32 of each original PyTorch parameter name. Expert weights are generated separately before stacking; vision `.ffn.` maps to reference `.mlp.`. Configurations, input seeds, normalization adjustments and comparison tolerances are in `test_deepseek_v41.py`.

PyTorch ran in FP32 on CPU. The recording script replaces the TileLang kernels the FP32 model reaches (activation quantization, FP4 activation quantization, sparse attention, Sinkhorn) with torch ports that follow `kernel.py` operation by operation. No MLX model output was used as an expected value.

Between the two revisions `model.py` changes in two places. 1017 quantizes and dequantizes the attention output before the `wo_a` einsum (`act_quant(o, ...)`, also in DSpark attention). 1017 also republishes an index-K owner's cache on every Indexer call (`owns_k`); the df42c109 recording had applied that same correction by hand. With the `act_quant(o)` lines removed, the script reproduces the df42c109 arrays to within 6e-7 (`check` mode), which validates the kernel ports.

These vectors check parity with the reference arithmetic on random small weights; they do not evaluate real-checkpoint quality. Changes to model arithmetic must be checked against an independent reference run before replacing expected values.
