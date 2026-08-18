# mlxfast-challenge → oMLX port: correctness ledger

Per-submission record for the `perf/mlx-fast-laguna` port of the
[Layr-Labs/mlxfast-challenge](https://github.com/Layr-Labs/mlxfast-challenge)
Laguna XS 2.1 DFlash Swift optimizations (`Sources/MLXFastModel/`) into oMLX's
Python MLX Laguna path (`omlx/patches/laguna/laguna_model.py`).

Each commit on this branch that ports a challenge submission is labeled with
that submission's UUID (`Validate submission <uuid>`, matching the organizer's
own commit labels) and MUST update this doc with the submission's
token/bit-exactness status. The port bar is **bit-exact parity against the
stock vendored model** plus a measured win before anything ships default-on;
anything that is NOT bit-exact, or that carries a token-correctness risk, is
recorded in the Concern register with the challenge commit that introduced it.

Challenge baseline read at `layr-labs/main` head `d9459e4`; the optimization
features entered the frozen baseline through the organizer's "Validate
submission" commits, mapped below.

## Submission registry (all 98, chronological)

98 `Validate`/`Accept submission` commits touched `Sources/MLXFastModel/`.
They span three model eras: **DeepSeek V4 Flash** (#1–31), **Gemma 4 31B**
(#32–92), and **Laguna XS 2.1** (#93–98) — the challenge migrated models
twice, and the Laguna migration (`4799830`) removed the earlier eras' model
code. The six Laguna-era submissions are the current challenge's validated
optimizations and are ported on this branch; the 92 earlier-era submissions
are documented here, not reproduced (reasons below the table).

| # | Date | Era | Submission | Challenge commit | Key change keywords | Status |
|---|---|---|---|---|---|---|
| 1 | 2026-07-01 | DeepSeek | `ade3937f-e1c6-4955-8074-a26d025aa80f` | `6f33a5e` | — | 📋 documented |
| 2 | 2026-07-02 | DeepSeek | `c155dace-ccbd-4627-97c4-b592d61e1690` | `b03ce62` | RoPE, prefetch, resident, sink, stager | 📋 documented |
| 3 | 2026-07-02 | DeepSeek | `0ddfcd37-4a0c-483f-a9b6-7c053197bcb8` | `458a1a7` | cache, warm | 📋 documented |
| 4 | 2026-07-02 | DeepSeek | `9e384a4e-bad0-4f27-bcb8-9a3e453536ea` | `57de4b2` | prefetch, sink | 📋 documented |
| 5 | 2026-07-02 | DeepSeek | `68cd907e-ba43-4c1f-a54d-fba887f7d715` | `696c56d` | compiledSinkhorn | 📋 documented |
| 6 | 2026-07-02 | DeepSeek | `c674aecd-3124-433c-b371-9593119dbb6c` | `929a696` | prefetch, sink | 📋 documented |
| 7 | 2026-07-02 | DeepSeek | `4fb0b2e5-5f4a-4ad5-aa71-0850f5e94bcf` | `3c372f4` | prefetch, sink, warm | 📋 documented |
| 8 | 2026-07-02 | DeepSeek | `eac7cf0f-9eb5-4b02-92f9-c702c6466ca2` | `bf4076f` | prefetch, resident, sink | 📋 documented |
| 9 | 2026-07-02 | DeepSeek | `53b6b82a-8b6c-42f1-ba46-67e7523ec1c9` | `c53f285` | RoPE, prefetch, resident, sink, stager | 📋 documented |
| 10 | 2026-07-03 | DeepSeek | `ac5841c0-c5f9-4ffc-a46e-3f8cf879ba56` | `a9378f9` | compiledSinkhorn, prefetch, resident, sink | 📋 documented |
| 11 | 2026-07-03 | DeepSeek | `8ef14a14-50f0-4011-b755-9da39d8858a6` | `6652490` | compiledCollapseTail, compiledExpand, gatherQuantizedMM | 📋 documented |
| 12 | 2026-07-03 | DeepSeek | `28ade6b4-89c8-47d9-8b92-08c89b937a2f` | `6afd0b7` | — | 📋 documented |
| 13 | 2026-07-03 | DeepSeek | `81a4c176-48d0-4030-9a53-4da8a9c27e55` | `9c7d77f` | — | 📋 documented |
| 14 | 2026-07-03 | DeepSeek | `9d7b587c-62f6-4255-b3ff-a5ca58614231` | `7cd6337` | — | 📋 documented |
| 15 | 2026-07-03 | DeepSeek | `d076ec7b-0af7-43a0-9c29-7324fc4ea48f` | `d595125` | resident | 📋 documented |
| 16 | 2026-07-03 | DeepSeek | `c6bf76ea-42b2-4b0a-8b55-2b91c7aef1ef` | `0e22935` | prefetch, resident, sink | 📋 documented |
| 17 | 2026-07-04 | DeepSeek | `c5303961-46e7-42af-8805-c7aa9c4c83d5` | `b585590` | gatherQuantizedMM, resident, sink | 📋 documented |
| 18 | 2026-07-04 | DeepSeek | `e8601677-d060-4036-94b0-d5cd84ff8972` | `931fdd4` | compiledHeadTail | 📋 documented |
| 19 | 2026-07-04 | DeepSeek | `ad182bfc-78ef-4f76-ab4e-84041b75ce16` | `a788dfc` | resident, sink | 📋 documented |
| 20 | 2026-07-04 | DeepSeek | `19d83b6f-2525-426e-8bf2-6fbc7ae47fa6` | `4348e8f` | prefetch | 📋 documented |
| 21 | 2026-07-04 | DeepSeek | `d5c35da9-5070-40c8-b77a-45ced5029e70` | `2130f0a` | warm | 📋 documented |
| 22 | 2026-07-05 | DeepSeek | `06e21771-d387-4050-b5ab-09979df5c678` | `cf6b2c2` | prefetch, stager | 📋 documented |
| 23 | 2026-07-05 | DeepSeek | `e15d6404-e46e-43f6-a5aa-b8ed46d143e2` | `f4a9715` | stager | 📋 documented |
| 24 | 2026-07-05 | DeepSeek | `9212271b-1c20-48c8-a8b3-37d6818e07cc` | `9b0cb68` | prefetch | 📋 documented |
| 25 | 2026-07-05 | DeepSeek | `ac76a987-d9d8-4f2e-8da9-fd90248a1ae3` | `24f758e` | — | 📋 documented |
| 26 | 2026-07-05 | DeepSeek | `c073533b-d6f5-4a88-ab95-44d1e6168171` | `f07f304` | prefetch | 📋 documented |
| 27 | 2026-07-05 | DeepSeek | `848a9295-da87-4e20-8114-f50f500d5de1` | `53a6cff` | prefetch | 📋 documented |
| 28 | 2026-07-05 | DeepSeek | `44cdd322-8cc8-4311-af0d-79adc11aaecb` | `0bd5b31` | RoPE, gatherQuantizedMM, prefetch, resident, stager | 📋 documented |
| 29 | 2026-07-05 | DeepSeek | `c5aa58e2-b5dd-415f-811e-9c7089ab872a` | `34e4c46` | stager | 📋 documented |
| 30 | 2026-07-05 | DeepSeek | `5260a8ed-e36b-46ed-81e1-1d7c0327933e` | `0a478e6` | — | 📋 documented |
| 31 | 2026-07-06 | DeepSeek | `37e6bd05-1310-4199-af40-5831cb9c59e2` | `cbe8c70` | stager | 📋 documented |
| 32 | 2026-07-11 | Gemma | `3e3c22f6-9f3c-470f-8923-98521293eb90` | `e332a43` | cache, warm | 📋 documented |
| 33 | 2026-07-11 | Gemma | `466f6610-d42c-46f9-8b58-0fdc1960b0f5` | `8821eb2` | MLX_COMPILED_DECODE, compile, cache, warm | 📋 documented |
| 34 | 2026-07-11 | Gemma | `21860797-32f0-42b8-b844-81ae18dcb1f3` | `fcd5f38` | MLX_MAX_MB_PER_BUFFER | 📋 documented (N1) |
| 35 | 2026-07-11 | Gemma | `548bf8cb-cc24-4967-907c-766982296cf0` | `3bd825e` | MLX_COMPILED_DECODE, fusedMLP, Quantized, RoPE | 📋 documented |
| 36 | 2026-07-11 | Gemma | `2c508e80-75d7-44bf-977e-88947653fbe9` | `9184d7d` | MLX_COMPILED_MLP_TAIL, fusedMLPTail | 📋 documented |
| 37 | 2026-07-11 | Gemma | `33cc707e-64cd-4ea9-8903-cc8159a1d3d2` | `7ea7af0` | — | 📋 documented |
| 38 | 2026-07-11 | Gemma | `36ca559a-968a-4543-ae2d-5bffe8d81385` | `7e875d1` | fusedGateUp, fusedGateUpPostTail, fusedMLPTail | 📋 documented |
| 39 | 2026-07-11 | Gemma | `7056bdf3-2321-451e-b62b-033751e9c098` | `c0eb2ab` | fusedGateUp, RoPE, warm | 📋 documented |
| 40 | 2026-07-11 | Gemma | `47f96c12-2c32-4283-9640-b34c1b20f711` | `d4e4ec3` | fusedGateUpActivation, fusedGateUpPostTail | 📋 documented |
| 41 | 2026-07-11 | Gemma | `5c9695ec-be97-48d0-8e5e-1d953a52fb91` | `2f9bc8b` | fusedGateUp, fusedGateUpActivation | 📋 documented |
| 42 | 2026-07-11 | Gemma | `2eb0801b-eb19-4070-aa37-993e5d310d85` | `930b85b` | fusedQKV | 📋 documented |
| 43 | 2026-07-11 | Gemma | `375905c3-ff2d-41a1-8b04-f97a78f077a3` | `b8e0cdc` | fusedQK | 📋 documented |
| 44 | 2026-07-11 | Gemma | `f360283f-3539-4502-9b04-73acde32eeb6` | `8b0cc20` | — | 📋 documented |
| 45 | 2026-07-12 | Gemma | `6e187d4e-3ac5-4cd5-b18d-eb476841bdd7` | `6235825` | fusedAttentionRMS, rope | 📋 documented |
| 46 | 2026-07-12 | Gemma | `a0acab75-a7ca-4fa8-aab0-846eceb67163` | `bddf831` | fusedAttentionRMS, rope | 📋 documented |
| 47 | 2026-07-12 | Gemma | `b03b5d53-f3c1-4c02-95b6-f5334cec988e` | `990492c` | — | 📋 documented |
| 48 | 2026-07-12 | Gemma | `c5d655e8-9fc7-44dc-9d39-2e65c45c123b` | `917d85c` | fusedAttentionToMLPBoundary, fusedPreFFNNormalized | 📋 documented |
| 49 | 2026-07-12 | Gemma | `adf7088c-a1f4-43b8-bd9a-9e9295640b75` | `5ac409e` | fusedMLPToNextBoundary | 📋 documented |
| 50 | 2026-07-12 | Gemma | `c0ad69c4-4f78-4d8c-aa50-729e591ef780` | `7d43852` | — | 📋 documented |
| 51 | 2026-07-12 | Gemma | `dac5704d-ea79-4553-878c-e17a595a1608` | `c299429` | — | 📋 documented |
| 52 | 2026-07-12 | Gemma | `1464373d-1f57-4586-9922-fd9f9c214366` | `05dec99` | DARKBLOOM_TIED_HEAD_PACKED, fused | 📋 documented |
| 53 | 2026-07-12 | Gemma | `58ba0704-c108-4917-a2d4-cfbb547a67e0` | `266f9ff` | combined attention prefill | 📋 documented |
| 54 | 2026-07-12 | Gemma | `cdd16b85-39a9-4ab4-a3f1-92ed177a3aab` | `2425530` | DARKBLOOM_TIED_HEAD_PACKED/QMV, fused | 📋 documented |
| 55 | 2026-07-12 | Gemma | `2fb6ab18-3206-4ed9-9eb1-09cdfc82f0a2` | `22b9cc8` | DARKBLOOM_COMBINED_KV_CACHE, compiledDecodeStep, fusedAttentionRMS | 📋 documented |
| 56 | 2026-07-12 | Gemma | `53591b11-bdd6-4d9f-a35b-799452693bfe` | `7237ab4` | DARKBLOOM_COMBINED_QKV_PREFILL_PREP, fusedAttentionRMS | 📋 documented |
| 57 | 2026-07-12 | Gemma | `9573b836-e0fc-44e0-87e3-c0e0e9858354` | `7572706` | DARKBLOOM_DOWN_COTILED_FIXED | 📋 documented |
| 58 | 2026-07-12 | Gemma | `c8616262-ddf5-4dfb-a922-8a4d475e8a72` | `04aa097` | DARKBLOOM_PACKED_GATE_UP_INDICES, fused | 📋 documented |
| 59 | 2026-07-12 | Gemma | `a9a0c797-70b2-4154-af37-4a8df4bc0fb0` | `cd6f6a0` | DARKBLOOM_DOWN_COTILED_FIXED | 📋 documented |
| 60 | 2026-07-12 | Gemma | `3a649737-5e61-434d-809a-9b1b8f2a1537` | `3fc1d5e` | DARKBLOOM_OUTPUT_COTILED, resident | 📋 documented |
| 61 | 2026-07-13 | Gemma | `5a24ed01-3dc3-4466-b25d-c4c8e46d606b` | `05f6ab7` | MLX_MAX_MB_PER_BUFFER, MLX_MAX_OPS_PER_BUFFER | 📋 documented (N1) |
| 62 | 2026-07-13 | Gemma | `505266a3-210f-422f-a3e3-e49bf7f17ee8` | `752af5c` | DARKBLOOM_GATE_UP_COTILED_FIXED, fused | 📋 documented |
| 63 | 2026-07-13 | Gemma | `aef5ac54-ee8b-4fc4-9c8b-4c87503ee9dc` | `aaca6d9` | DARKBLOOM_STAGED_SLIDING_PREFILL_ATTENTION | 📋 documented |
| 64 | 2026-07-13 | Gemma | `b742a0fc-4e0b-48a7-9e89-d672522e4cd1` | `4fdba66` | DARKBLOOM_OUTPUT_COTILED_FULL/SLIDING | 📋 documented |
| 65 | 2026-07-13 | Gemma | `0effa041-c173-40c1-b411-8758097e617c` | `bb3ccb9` | MLX_MAX_MB_PER_BUFFER | 📋 documented (N1) |
| 66 | 2026-07-13 | Gemma | `7c52e8c6-86fb-4cef-a3b1-3c9a9fd16ca3` | `6eae78e` | DARKBLOOM_DIRECT_PREFILL_ATTENTION_RMS_ROPE, rope | 📋 documented |
| 67 | 2026-07-13 | Gemma | `c2e4c014-44bb-4066-8632-ae94092fb8c1` | `a0f41e6` | DARKBLOOM_COMPILED_DECODE, DARKBLOOM_PROMPT_LOOKUP, exact-two vectors, fused | 📋 documented |
| 68 | 2026-07-13 | Gemma | `ae4f803d-2a4c-4c79-8c0e-58cea8fd4980` | `9c87390` | exact-two middle, prompt lookup | 📋 documented |
| 69 | 2026-07-17 | Gemma | `553fbfe1-d6fb-4502-90a4-98561bb73101` | `f310c09` | co-tiled payloads, DARKBLOOM_QKV_COTILED, staged prefill, packed indices | 📋 documented |
| 70 | 2026-07-17 | Gemma | `d16feaf2-8b76-4848-ae91-e3b16119cdb6` | `05f6ffd` | DARKBLOOM_FUSED_DECODE_EMBED, DARKBLOOM_LAST_LAYER_TAIL_PRUNE, DARKBLOOM_DECODE_KV_DIRECT_WRITE | 📋 documented |
| 71 | 2026-07-17 | Gemma | `e05c3ce3-c5e8-4e24-a883-21ee75f88ae6` | `d7ca977` | DARKBLOOM_PREFILL_GELU_EPILOGUE, MLX_MTL_CONST, fusedMLPTail | 📋 documented |
| 72 | 2026-07-17 | Gemma | `5ee12c07-11f0-4b89-8766-ce032817412f` | `3c55e15` | DARKBLOOM_PREFILL_GELU_EPILOGUE_BN | 📋 documented |
| 73 | 2026-07-17 | Gemma | `a20a49cc-cc11-4c8e-9985-d28c9ae89680` | `ba824f0` | DARKBLOOM_GATEUP_COTILE_TAIL, DARKBLOOM_KV_SKIP_SUFFIX_ZEROFILL, DARKBLOOM_LAST_LAYER_Q_PRUNE | 📋 documented |
| 74 | 2026-07-17 | Gemma | `2d655b99-511e-4c60-93a7-34fad3af4055` | `872d24c` | DARKBLOOM_PREFILL_BN, MLX_METAL_GPU_ARCH, fused | 📋 documented |
| 75 | 2026-07-17 | Gemma | `5f04db68-269e-4e93-b5fb-b76aca066e15` | `d16f2dc` | DARKBLOOM_PREFILL_BN, fused | 📋 documented |
| 76 | 2026-07-17 | Gemma | `9fb36ab0-bd90-4ccc-89ad-17fe92567503` | `43eec7f` | DARKBLOOM_PREFILL_CHUNK_EVAL | 📋 documented |
| 77 | 2026-07-18 | Gemma | `744e6dd5-f704-4a91-beec-2d1450bdb697` | `5c1dcf3` | — | 📋 documented |
| 78 | 2026-07-18 | Gemma | `dfdf6fb1-117e-4c94-afcd-9f91994958d0` | `d428954` | DARKBLOOM_TIED_HEAD_EIGHT_SIMDGROUPS | 📋 documented |
| 79 | 2026-07-19 | Gemma | `7be2aae2-e983-4f82-9fb4-f42559e83d47` | `c321ee8` | DARKBLOOM_INIT_DRAIN_ALLOCATOR_CACHE | 📋 documented (N1) |
| 80 | 2026-07-19 | Gemma | `30975d9c-e427-420a-ab58-f524aacdb98b` | `c118b23` | fused (fast-engine consolidation) | 📋 documented |
| 81 | 2026-07-19 | Gemma | `d391d953-5e5f-4d77-bb94-a98cbea9611c` | `75e7f67` | DARKBLOOM_MTP_EXACT_PAIR_ASYNC_LAYER_GROUP | 📋 documented |
| 82 | 2026-07-19 | Gemma | `f118bc24-92a4-47db-9e3c-a22e3d8dac8a` | `18d854a` | DARKBLOOM_MTP_EXACT_FOUR, DARKBLOOM_MTP_ADAPTIVE_EXACT_FOUR, MTPRuntime | 📋 documented |
| 83 | 2026-07-19 | Gemma | `842209fc-dcc8-4014-90c9-a903b790498e` | `9bc83cc` | DARKBLOOM_MTP_EXACT_FOUR_ATTENTION | 📋 documented |
| 84 | 2026-07-19 | Gemma | `70b6e29b-e6bb-4551-a394-8cb9627ce984` | `c1031bc` | DARKBLOOM_MTP_LEADING_MARGIN_SELECTOR | 📋 documented |
| 85 | 2026-07-19 | Gemma | `1ee42a4a-033e-4e8b-a310-46e323db9beb` | `c07b9f4` | DARKBLOOM_MTP_SECONDARY_MARGIN_SELECTOR | 📋 documented |
| 86 | 2026-07-20 | Gemma | `22c14d2e-b109-4aa5-9ddc-afc6cfe92b3d` | `d9b9910` | DARKBLOOM_MTP_DRAFTER_ASYNC_EVAL | 📋 documented |
| 87 | 2026-07-20 | Gemma | `3a954b0a-0a0c-4376-be27-0d70bc314314` | `afc332f` | DARKBLOOM_MTP_EXACT_FOUR_OUTPUT_DOWN_PAIRS | 📋 documented |
| 88 | 2026-07-21 | Gemma | `22d93af1-41d1-45c6-a0a7-6388289fa59a` | `fd2369e` | DARKBLOOM_MTP_HOISTED_DRAFT_MASKS | 📋 documented |
| 89 | 2026-07-21 | Gemma | `a08522b7-877a-444d-8922-42f665dfa344` | `2c73d9a` | DARKBLOOM_MTP_ASSISTANT_POSITION_DELTA, RoPE | 📋 documented |
| 90 | 2026-07-21 | Gemma | `1f4015e7-76e1-47cf-a76c-1d33c6bd670a` | `73e8446` | DARKBLOOM_MTP_EXACT_FOUR_FUSED_ARGMAX | 📋 documented |
| 91 | 2026-07-21 | Gemma | `cc840d67-e845-435c-aee5-6424774d106d` | `45b662f` | DARKBLOOM_MTP_EXACT_PAIR_FUSED_ARGMAX | 📋 documented |
| 92 | 2026-07-21 | Gemma | `54045bac-7ac3-4088-ace7-0921a91d3c16` | `424129e` | — | 📋 documented |
| 93 | 2026-07-23 | Laguna | `8b4de42b-d6bd-4da8-814d-b0b3ae6cf2f2` | `c9e1043` | compiled softplus gate, compiled SiLU product | ✅ ported (`2a6fbe92`) |
| 94 | 2026-07-23 | Laguna | `613aaf69-9016-4d57-b799-bdd22d51c5c9` | `62c6697` | fused routed/shared gate-up banks (fused QKV documented) | ✅ ported (`35592b93`) |
| 95 | 2026-07-23 | Laguna | `8adb56be-8f8f-4611-8914-8daf052b5f21` | `f8848e0` | compiled top-k normalize (two-output router tail → C1) | ✅ ported (`f48c5323`) |
| 96 | 2026-07-24 | Laguna | `9a37e4dc-b518-446c-a3f0-e4e90a581674` | `b424bc8` | compiled weighted expert combine | ✅ ported (`6181a829`) |
| 97 | 2026-07-24 | Laguna | `eb76e2b8-de50-44d5-9137-953c6e40d28e` | `4d9eecb` | folded-normalized combine (equivalent, pinned) | ✅ ported (`90c997ed`) |
| 98 | 2026-07-24 | Laguna | `dc738a8d-a8b9-4187-abc3-68f61099fb67` | `7e61f8d` | residual-variant expert combines | ✅ ported (`4b27cc88`) |

### Why submissions #1–92 are documented, not reproduced

These 92 `Validate/Accept submission` commits target the challenge's two
**earlier** model generations — DeepSeek V4 Flash (#1–31) and Gemma 4 31B
(#32–92) — which the organizer itself replaced with Poolside Laguna XS 2.1
(`4799830` "Migrate serial track: Gemma 4 31B → Poolside Laguna XS 2.1").
Consequently:

- **Not part of the current challenge model surface.** The Laguna migration
  and later cleanups (`fe9d166` prune dead A/B variants, `51b9dd2` dead code
  removal, `461af5b` remove Gemma-MTP-era code) deleted the DeepSeek/Gemma
  runtime code these submissions optimized; the surviving head has no
  `DeepSeek*` or `Gemma4FastEngine` code.
- **Different architecture than oMLX's equivalent model paths.** oMLX serves
  DeepSeek V4 Flash and Gemma 4 through the stock mlx-lm architectures plus
  oMLX patches — not the challenge's reimplementations (expert stagers,
  prefetchers, fast-engine fusions, co-tiled/packed kernel layouts, MTP
  exact-two/four vector kernels). A faithful reproduction would be a
  re-implementation of removed code against a different runtime.
- **Model-agnostic techniques already covered by the Concern/considered
  notes:** compiled decode wiring and `MLX_MAX_MB_PER_BUFFER`/`MLX_MAX_OPS_PER_BUFFER`
  (rows 33–35, 61, 65 → N1), init-time allocator drain (row 79 → N1), and
  warmup (rows 3, 21, 32 → N3).

If a future oMLX effort targets the DeepSeek V4 Flash or Gemma 4 31B paths
directly, rows 1–92 are the per-submission starting points (challenge commit
+ key flags).

## Concern register (token/bit-exactness issues, with challenge commits)

### C1 — Compiled two-output router tail is not bit-exact in Python MLX

- **Submission / challenge commit:** `8adb56be-8f8f-4611-8914-8daf052b5f21` / `f8848e0` (`lagunaCompiledRouterTail`).
- **Optimization:** compile `[sigmoid(logits), -(sigmoid(logits)+bias)]` into one kernel (four elementwise launches per router call, 39 per token).
- **Token-exactness issue:** a compiled function returning TWO outputs that consume the same `sigmoid(a)` intermediate diverges from eager at ULP (5.96e-8–1.19e-7, deterministic, isolated to the multi-consumer shape). Single-output compiled fusions reusing the same sigmoid are bit-exact, so the trigger is the two-output shape. It feeds `argpartition` expert selection — a ULP flip at a near-tie boundary changes WHICH experts are gathered (a different forward, not a small perturbation), i.e. the challenge's own documented correctness cliff ("Rank is the wrong metric").
- **Mitigation:** kept eager in the port; pinned by `test_two_output_compiled_tail_diverges_documented` (fails if MLX ever fixes it).
- **MLX-version dependence:** property of MLX 0.32.0 compiled kernels; re-verify after any MLX bump.

### C2 — `logits_last_only` head slicing is ULP-divergent (frame divergence)

- **Challenge commit:** `4799830` (`lagunaLastTokenHidden`) — the Laguna migration, not a submission; recorded for completeness.
- **Optimization:** slice post-norm hidden to the last position before `lm_head` so prefill never computes the `[L-1, vocab]` slab.
- **Token-exactness note (not a bug):** a `[1,1,H]` head matmul is ULP-divergent from the `[B,L,H]` full matmul (measured ~1.8e-7) — the same matmul-width **frame divergence** the challenge contract documents. The DFlash reference layer tolerates it; the real-checkpoint greedy trajectory is token-identical with and without the slice. oMLX's DFlash target path already implements it (`logits_last_only`), pinned by `test_target_ops_logits_last_only_slices_before_lm_head`.

## How to update

Every ported submission appends/updates its registry row with the port commit
and a one-line evidence summary. Anything that fails either half of the bar
(bit-exactness or a measured win) is documented in the Concern register before
the corresponding code lands. Pre-Laguna submissions stay documented with
their era reason unless an oMLX effort targets that model path.
