# GLM-5.3-Flash 提速移植计划（源自 Qwen4 Flash-Next 方案）

把 jundot 在 Qwen4Exp 上验证的 prefill/decode 提速思路（#3903、#3797）移植到
GLM-5.3-Flash（`glm5_next`）。架构对照：GLM-5.3-Flash 与 Qwen4 同为
线性注意力（Kimi Delta 变体）+ DSA 块稀疏注意力（kpool 压缩 + top-k 选块）+
MoE + MTP(nextn) + 超连接（deepseek_v4 HyperConnection）的混合架构。

## 阶段划分

| 阶段 | 内容 | 对应 Qwen4 提交 | 状态 |
|---|---|---|---|
| P1 | prefill 稠密前缀旁路（选中块数≤预算的 query 行直接稠密因果注意力） | #3903 QSA dense prefix | 进行中 |
| P2 | 宽步 prefill 调度（长 prompt 后续 chunk 提高行数以喂饱 MoE GEMM） | #3903 wide prefill step | 待办 |
| P3 | 线性注意力 prefill 融合内核（conv+SiLU+L2 预处理 / norm-gate 尾部融合，按 KDA 几何移植） | #3903 GDN prefill fused kernels | 待办 |
| P4 | verify 路径接线（moe_verify_gather / packed_linear 评估接入） | #3797 | 待办 |

topk 免排序一项不适用：GLM-5.3-Flash 的选择路径没有 `mx.sort`（#3903 改的
`dspark_fp32_topk_indices` 内核只被 Qwen4 QSA 的收集路径消费）。

## P1 设计：稠密前缀旁路

`Glm5NextIndexer` 的语义：query 在 cache 位置 `pos` 的候选块 = 完整窗口块
`floor((pos+1)/kp)` 个；选择数为 `select_k = min(index_topk // kp, P)`；
`index_kpool_always_select_tail` 补齐当前部分窗口的尾部 token。因此当
候选数 ≤ `select_k` 时，选中集合 = 完整因果前缀 `[0..pos]`，top-k 选择是
恒等操作。稠密前缀最后一行满足 `floor((pos+1)/kp) ≤ select_max`，即

    D = (index_topk // kp + 1) * kp - 1
    dense_rows = min(L, max(0, D - past_len))

稠密行走 `mx.fast.scaled_dot_product_attention`（MLA latent 空间，GQA 64→1，
复用 embed_q/unembed_out 对偶投影，不展开逐头 K/V）；其余行照常走
indexer+稀疏路径（`Glm5NextIndexer` 增加 `score_from`，pool 推进覆盖全部行，
只对尾部行打分选块）。

失效即回退（fail-closed）条件：`L ≤ 8`（decode/verify 块不动）、
`kp > 1` 时必须 `index_kpool_always_select_tail`（`kp == 1` 无部分窗口问题）、
KV cache 为普通 `KVCache` 且无 left padding、PoolingCache 非批量（`_processed`
非 list）、mask 为 None 或行对齐的数组（掩码按行列切片复用）。批量请求与
`BatchKVCache`/`BatchPoolingCache` 保持原路径。

Kill switch：环境变量 `OMLX_GLM53_DENSE_PREFIX=0`（模块全局
`_DENSE_PREFIX_BYPASS`，调用时读取，测试可 monkeypatch）。

验证：`tests/test_mlx_vlm_glm5_next_compat.py`
- `test_dense_prefix_bypass_matches_reference`：tiny 模型（index_topk=4, kp=4 → D=7）
  单次 prefill 13 行，旁路开/关 logits 一致；并断言旁路确实触发。
- `test_chunked_prefill_with_bypass_matches_single`：分块 [9,4] 与整体 prefill 一致。
- `test_indexer_score_from_matches_suffix`：`score_from` 切片的 topk 与全量结果尾部逐位一致。
- `test_dense_prefix_rows_gating`：几何/门控公式与 fail-closed 条件单测。

```bash
python -m pytest -q tests/test_mlx_vlm_glm5_next_compat.py
```

## 进度日志

- 2026-09（P1 开始）：完成 glm5_next 语言模型、indexer、PoolingCache、测试基建调研；
  确认 topk 免排序不适用；P1 设计定稿。
- 2026-09（P1 完成）：`language.py` 落地 `score_from`、`_dense_prefix_rows`、
  `_dense_flat`、`_finish`；`__call__` 拆为门控 + `_forward`。新增 4 条测试全绿：
  等价性（含稠密行仍进 pool、后续 decode 一致）、分块不变量、`score_from`
  逐位切片、门控几何（含变长批 array offset 拒绝——由
  `test_variable_length_batch_matches_single_request_greedy_tokens` 抓出并修复）。
  回归对比：compat 套件失败集与基线逐条一致（6 条既有失败），MTP 套件同样
  与基线一致（12 条既有失败，`keys_and_values` handler 环境问题）。
  改动文件：`omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py`、
  `tests/test_mlx_vlm_glm5_next_compat.py`。
- 2026-09（P2 完成）：调度器接入 GLM-5.3 宽步 prefill。新增
  `_detect_glm53_wide_prefill_step`（glm5_next + native
  `glm_dsa_sparse_mla_attention` + NAX + ≥64GB → 8192）、
  `_base_prefill_step_size` 宽步块（GLM 无 SSD n-gram gather，参照
  GLM-5.2 自适应先例**自首个 chunk 起宽**，非分页时对齐 8192 网格收尾）、
  `_enlarge_block_size_for_arrays_cache` 目标纳入宽步。新增 3 条调度测试；
  `tests/test_scheduler.py` 全量 359 通过。
  注意：测试必须用 `.venv/bin/python`（homebrew python 的 mlx_lm 版本过旧，
  会产生与代码无关的收集/断言失败）。P1 在 venv 下复跑 40/40、MTP 92/92。
- P3 完成：KDA 线性注意力 prefill 融合内核。新增
  `omlx/patches/glm53_kda_prework.py`：`omlx_glm53_kda_prefill_prework`
  （conv4+SiLU+fp32 sum-L2+q 缩放+conv-state 滚动，单 kernel）与
  `omlx_glm53_kda_norm_gate_fused`（fp32 RMSNormGated 全链，单次回写 bf16）。
  与 Qwen4 供体的几何差异：GLM `_l2norm` 是 **sum**（非 mean）且全程 fp32、
  forget gate 是逐通道向量 g[B,T,H,D]（非标量）、conv K=4。门控接在
  `Glm5NextLinearAttention.__call__`（B==1 / mask None / S≥64 / fail-closed
  eligibility：bf16、生产几何、ArraysCache 无 padding、非投机、无 history）。
  **关键修复**：`glm5_next_vlm_runtime._patch_linear_attention` 整体替换
  `__call__`（MTP 捕获路径），会静默抹掉 vendor 侧融合门控——已在替换体顶部
  加同款门控（`gdn_sink is None` 时短路，verify 捕获路径不受影响），并加回归
  测试 `test_kda_fused_prefill_survives_mtp_runtime_patch`。该 bug 是测试套件
  顺序污染（oq roundtrip 测试 sticky apply）暴露的真实生产缺陷：GLM-5.3 生产
  必开 MTP，不修则 P3 在生产永不生效。
  新增 11 条测试：prework/norm-gate 内核 **位级一致**（`mx.array_equal`，
  含 S<3 状态滚动边界）、端到端等价（logits/conv 态/递归态/续 decode）、
  分块不变量、eligibility 门控、MTP runtime 存活。compat 50/50、
  MTP+scheduler 451/451、ruff 干净。
- 部署（Studio）：正式代码以 `git diff 3f2d07e8..ba0dec48` 全量补丁应用于
  Mac Studio `~/omlx`（工作树干净，可 `git checkout .` 原子回退）。Studio 测试
  闸门 501/501（scheduler+compat+MTP，venv mlx 0.32.2）。Studio 为 M3 Ultra
  非 NAX：P2 宽步不 engage（NAX-gated），4096 floor 生效；本次 A/B 实测
  P1+P3。旧代码 live 基线（2026-09-25 上午，重启前）：pp1024=912.7 /
  pp4096=1032.0 / pp8192=1287.5 tps。A/B：`~/ab_glm53_port.sh`
  （OFF=两个 kill-switch 关，ON=默认；ROUNDS=2，pp=1024/4096/8192 tg=128，
  中位数对比）。
- **基线勘误（live A/B 后）**：memory.md 中 pp4096≈1032/1287 的"GLM 基线"实为
  Qwen3.8-Flash-Next-oQ8e-mtp 的日志（ane-config 行核对）；GLM-5.3-Flash 在本机
  的真实历史基线为 pp1024≈357 / pp4096≈440 / pp8192≈415 tps（09-24 多轮 +
  本次 OFF 组吻合 <0.5%）。
- **Studio live A/B（OFF=双 kill-switch，ON=默认，2 轮中位数，2026-09-26 02:25-02:33）**：
  pp1024 359.1→354.2（-1.4%，噪声区）；pp4096 441.5→448.1（**+1.5%**）；
  pp8192 414.9→439.1（**+5.8%**，两轮 ON 均高于两轮 OFF）。ON 组 engage 日志
  确认 KDA 融合生效；chunk 明细：pp8192 首块 9566→8924ms（-6.7%）、次块
  9986→9617ms（-3.7%）。
- HC 融合无需移植：GLM 的 `deepseek_v4/hyper_connection.py` 已自带 fused
  sinkhorn+collapse Metal kernel + `@mx.compile` expand（供体 hc_fused 非空白区）。
- 追加实验：**非 NAX 强制宽步实测成功**（`OMLX_GLM53_WIDE_STEP_FORCE=8192`，
  scheduler env 覆盖，默认关；commit 5036bbc5）。requested_step=8192 确认生效，
  2 轮干净分离（02:52-02:55）：

  | pp | OFF | ON(P1+P3) | WIDE(+8192) | 累计 vs OFF |
  |---|---|---|---|---|
  | 1024 | 359.1 | 354.2 | **370.3** | **+3.1%** |
  | 4096 | 441.5 | 448.1 | **453.2** | **+2.6%** |
  | 8192 | 414.9 | 439.1 | **450.2** | **+8.5%** |

  宽步在 P1+P3 之上再叠 +2.5%（pp8192）——早期 8192 回滚（+0.8%）是在没有
  dense-prefix 摊薄首块成本的旧形态下测的，新形态下成立。生产形态：Studio 以
  `OMLX_GLM53_WIDE_STEP_FORCE=8192` 常驻。
- **阶段结论（vs 20-30% 目标）**：三层叠加实测 +8.5%（pp8192）/+2.6%（pp4096）/
  +3.1%（pp1024），可复现。供体 20-30% 的主体增益来自 NAX tensor units 与宽步
  MoE GEMM 的硬件路径，非 NAX M3 Ultra 上无对应接线面；剩余瓶颈为 MoE/MLA
  kernel 本体（需全新 kernel 设计，非移植范畴）。

- P4 调研结论（verify 路径接线）：**无适用接线面**。
  ① `qwen35_packed_linear`：`enabled()` 门控要求模块名含 qwen3_5/qwen35 且
  **不含 moe**、4-bit、M5/NAX tensor units——GLM-5.3-Flash 是 oQ8 MoE、Studio
  非 NAX，三重排除。② `moe_verify_gather`：仅由 `qwen35_moe_gate_up.fused_switch`
  调用，`_FAMILY_TOKENS`（qwen3_5/qwen3_6/qwen4_exp/laguna/hy_v3）不含
  glm5_next；GLM MoE 走自家 affine-block gather（`glm5-next-affine8`，
  bit-exact、实测 pp8192 +1.6%），其权重布局与 gather_qmm 不同，移植
  expert-order 调度 = 新 kernel 设计而非接线，且供体自身收益量级有限。
  ③ verify-qmm mma 已全局生效；dspark verify 本就绕过 verify-qmm。
  结论：P4 以“不适用”关闭；若日后做 affine 布局的 expert-order verify kernel，
  另立实验。
