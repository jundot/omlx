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
- 待办 P3：KDA 线性注意力 prefill 融合内核（conv+SiLU+L2 预处理 /
  RMSNormGated 尾部）。
