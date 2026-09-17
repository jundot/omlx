# DeepSeek V4.1 expected outputs

`deepseek_v41_expected.npz` contains 43 small NumPy arrays (23,093 bytes), not model weights or executable reference code. They were recorded from DeepSeek's MIT-licensed [reference implementation at df42c109f1defefcbfcedbe7d905718a12266e40](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/df42c109f1defefcbfcedbe7d905718a12266e40). The archive SHA-256 is `2eb45f6bd71cda9a99dd58e6efb263858d7b09e7b12c5193ccab71f096d41f31`.

The tests reconstruct synthetic weights with `load_reference_weights`: NumPy PCG64 seeded by the case seed plus CRC32 of each original PyTorch parameter name. Expert weights are generated separately before stacking; vision `.ffn.` maps to reference `.mlp.`. Configurations, input seeds, normalization adjustments and comparison tolerances are in `test_deepseek_v41.py`.

The archive covers prefill lengths 1/3/8/17 followed by three decode steps, Engram hashes across image boundaries, Engram gating, ViT and aligner outputs, image patch geometry/normalization, and three DSpark target/draft steps. PyTorch ran in FP32 on CPU with the small `cpu_kernels` substitutions in the test file replacing TileLang activation quantization, sparse attention and Sinkhorn operations. No MLX model output was used as an expected value.

For prefill/decode and DSpark, the reference indexer was made to republish its own K cache before each call. The pinned source otherwise retains another layer's shared index-K pointer on an incomplete compression group. This correction is part of the recorded oracle; these vectors do not claim parity with that reference bug or evaluate real-checkpoint quality.

To regenerate, use the linked reference revision and the configurations, parameter generator and CPU substitutions above. Record the final prefill/decode logits, target hidden states and draft tuple, and the separate vision/Engram outputs in the same assertion order. Changes to model arithmetic must be checked against that independent reference before replacing expected values.
