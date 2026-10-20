# ParoQuant targets with batched DFlash

The DFlash toggle supports ParoQuant Qwen3, Qwen3-MoE, and Qwen3.5-family dense/MoE checkpoints (`qwen3`, `qwen3_moe`, `qwen3_5`, `qwen3_5_moe`). Qwen3.6/3.8 exports using those model types follow the same rules. This requires four-bit ParoQuant weights, group size 32/64/128, positive model dimensions, and a positive rotation count when specified. Other ParoQuant architectures remain unavailable through this toggle.

Install the optional loader with `pip install 'omlx[paroquant]'`, register the target and a DFlash draft trained for that target's base model, and select the draft in the model's DFlash settings. Matching tensor dimensions alone do not establish that a draft is suitable for a target.

The batched engines use the existing ParoQuant loader, preserving its rotated projections. DFlash attaches to the shared batched verify path. No separate `dflash-mlx` target loader or text-only reload/fallback engine is introduced. TurboQuant, IndexCache, and native MTP retain their own compatibility gates.

## Validation scope

The regression tests exercise actual ParoQuant checkpoint loading, rotated dense and MoE projections, hidden-state capture, and every acceptance count in a four-token verify block. They cover all four text architectures with group-size/rotation-count pairs 32/2, 64/4, and 128/8. Separate small Qwen3.5 dense/MoE fixtures cover the mlx-vlm language-model capture and rollback hooks. Tokenizer construction is replaced only in the small checkpoint round-trip tests.

```sh
python -m pytest tests/test_paroquant_dflash.py tests/test_model_loading.py tests/test_admin_api_key.py
```

A current-backend local check with Qwen3.8-27B-PARO and its matching DFlash2 draft matched ordinary greedy decoding on two short prompts, twelve generated tokens each. Draft calls were observed. This is a bounded text-generation check, not a throughput benchmark or evidence for long-context, concurrent-serving, image-quality, or broad task parity. Image inputs and other full-size checkpoints were not rerun for this change.
