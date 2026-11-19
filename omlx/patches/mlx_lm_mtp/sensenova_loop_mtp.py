# SPDX-License-Identifier: Apache-2.0
"""SenseNova Loop MTP 模块 —— Loop 感知的 cache 回滚与 MTP 特化。

本模块承载 SenseNova Loop 的 MTP 特化逻辑（按槽回滚、跳过来触达槽），
从通用 ``qwen35_model.py`` 中独立出来。它覆盖
``qwen3_5.TextModel.mtp_partial_rollback`` 为 Loop 感知版，并在
非 Loop 模型上**回退到原始通用版**（行为与分离前完全一致）。

关键背景（13:40 修复，2026-11-02）：
  Loop 架构下 verify 前向只执行执行计划（``_loop_plan``）的**前段**
  （主片段），后续重复分片槽**未被触达**（线性层无 stash、KV offset==0）。
  若不按槽处理而按物理层处理，长度检查永不通过，整条链回退标准步进，
  投机解码完全失效。

设计原则（关注点分离）：
  ``qwen35_model.py`` 的 ``mtp_partial_rollback`` 是**通用版**（identity
  槽布局，零 Loop 概念）；本模块用 Loop 感知版**包装**它：
    - Loop 模型  → Loop 感知逻辑（按槽、跳未触达槽）；
    - 非 Loop 模型 → 回退原始通用版（零影响）。

  依赖方向：sensenova_loop_mtp → qwen35_model（经 cls 方法读取），
  qwen35_model 对 SenseNova 零感知。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .sensenova_loop_model import mtp_slot_layout

logger = logging.getLogger(__name__)

_MARKER = "_omlx_sensenova_loop_mtp"

# 原始通用版 mtp_partial_rollback（apply() 覆盖前保存，作为非 Loop 回退）。
# 若 apply() 尚未运行或原始版缺失，则为 None（防御性）。
_FALLBACK = None


def _slot_layout(model: Any) -> Optional[tuple]:
    """取 Loop 槽布局 ``(n_slots, phys_fn)``；非 Loop 模型返回 None。"""
    try:
        return mtp_slot_layout(model)
    except Exception:
        return None


def _loop_rollback(model, cache, accepted: int, num_drafts: int, layout: tuple) -> bool:
    """Loop 感知的 cache 回滚核心（按槽处理，跳过未触达槽）。"""
    n_slots, phys_fn = layout
    layers = model.layers
    if len(cache) != n_slots:
        return False
    trim_n = num_drafts - accepted
    if trim_n <= 0:
        return True
    keep = 1 + accepted  # confirmed token + accepted drafts

    # --- 第一遍：验证触达槽都具备回滚状态 ---
    for slot in range(n_slots):
        phys = phys_fn(slot)
        if phys is None:
            continue  # 前向未消费的死槽（槽 >= 物理层数），跳过
        layer = layers[phys]
        c = cache[slot]
        if getattr(layer, "is_linear", False):
            rs = getattr(c, "rollback_state", None)
            ds = getattr(c, "_mtp_draft_stash", None)
            if rs is None:
                # 未触达线性层：无 stash 则跳过；异常态（有 stash 无状态）报错
                if ds is not None:
                    return False
                continue
            if ds is None:
                return False
        else:
            # KV 槽：offset==0 表示未触达，跳过；否则必须可 trim
            off = getattr(c, "offset", 0)
            if off is None or off <= 0:
                continue
            if not (hasattr(c, "is_trimmable") and c.is_trimmable()):
                return False

    # --- 第二遍：执行回滚 ---
    for slot in range(n_slots):
        phys = phys_fn(slot)
        if phys is None:
            continue  # 死槽
        layer = layers[phys]
        c = cache[slot]
        if getattr(layer, "is_linear", False):
            if getattr(c, "rollback_state", None) is None:
                continue  # 未触达线性层，跳过
            conv_0, ssm_0 = c.rollback_state
            qkv_s, a_s, b_s = c._mtp_draft_stash
            _, conv_m, ssm_m = layer.linear_attn._process_chunk(
                qkv_s[:, :keep],
                a_s[:, :keep],
                b_s[:, :keep],
                conv_0,
                ssm_0,
                None,
            )
            c[0] = conv_m
            c[1] = ssm_m
            c.rollback_state = None
            c._mtp_draft_stash = None
        else:
            off = getattr(c, "offset", 0)
            if off is None or off <= 0:
                continue  # 未触达 KV 槽，跳过
            c.trim(trim_n)
    return True


def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
    """Loop 感知包装版（挂到 TextModel 上，替代通用版）。

    - Loop 模型（``_loop_plan`` 存在）→ Loop 感知逻辑；
    - 非 Loop 模型 → 回退原始通用版（与分离前行为一致）。
    """
    model = self.model
    layout = _slot_layout(model)
    if layout is not None:
        return _loop_rollback(model, cache, accepted, num_drafts, layout)
    # 非 Loop：回退原始通用版（若无原始版，防御性返回 False）
    if _FALLBACK is not None:
        return _FALLBACK(self, cache, accepted, num_drafts)
    return False


def apply() -> bool:
    """用 Loop 感知包装版覆盖 TextModel.mtp_partial_rollback（幂等）。

    覆盖前保存原始通用版为 ``_FALLBACK``（供非 Loop 回退）。幂等性经
    ``cls.__dict__`` 的 marker 检查。外层 ``q35.Model`` 的转发版不动
    （它经 ``self.language_model`` 转发，最终仍调 TextModel 的包装版）。

    仅当 mlx_lm.models.qwen3_5 可导入且 TextModel 已有 mtp_partial_rollback
    时才覆盖；否则跳过（防御性，返回 False）。

    同时包装 ``ModelArgs.from_dict``（读 mhc/loop_settings 实例属性）与
    ``Model.__init__``（末尾调 configure_model 注入 Loop+MHC）——这是把
    SenseNova Loop/MHC 特化从 ``mlx_lm/models/qwen3_5.py`` 抽出的编排侧。
    """
    global _FALLBACK
    try:
        from mlx_lm.models import qwen3_5 as q35
    except Exception as e:  # pragma: no cover
        logger.debug("qwen3_5 not importable; skipping sensenova_loop_mtp: %s", e)
        return False

    cls = q35.TextModel
    current = cls.__dict__.get("mtp_partial_rollback")
    if current is None:
        return False  # 原始通用版未安装（qwen35_model 未 apply），跳过

    # 幂等：已标记则跳过（但即使跳过，也要重新验证/恢复 _wrap_forward ——
    # 重载时 __call__ 可能被其他补丁覆盖，需要 self-healing 重新包装）。
    already_marked = getattr(current, _MARKER, False)
    if not already_marked:
        # 保存原始通用版作为非 Loop 回退
        _FALLBACK = current
        cls.mtp_partial_rollback = mtp_partial_rollback
        setattr(cls.mtp_partial_rollback, _MARKER, True)

    # --- SenseNova Loop/MHC 配置注入编排（从库文件 qwen3_5.py 抽出）---
    # 幂等：只包装一次。若已由其他路径包装（如旧版库文件自带），跳过。
    from . import sensenova_loop_engine as _eng
    _eng._wrap_model_args(q35)
    _eng._wrap_model_init(q35)
    _eng._wrap_make_cache(q35)
    _eng._wrap_sanitize(q35)
    # Loop 前向展开：必须包装 *替换后* 的 Qwen3_5TextModel.__call__（qwen35
    # 已在其前替换为返回 pre-norm hidden 的版本），约定保持一致。
    # _wrap_forward 内部已 self-healing：即使 marker 在，若 __call__ 被覆盖
    # 则重新包装。
    _eng._wrap_forward(q35)

    if not already_marked:
        logger.info("SenseNova Loop MTP patch applied (wrap mtp_partial_rollback)")
    return True
