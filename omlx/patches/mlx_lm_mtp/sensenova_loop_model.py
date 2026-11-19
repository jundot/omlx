# SPDX-License-Identifier: Apache-2.0
"""SenseNova Loop 模型模块 —— 商汤 Loop 架构的执行计划与 MTP 槽布局。

本模块把 SenseNova Loop 的特化逻辑从通用 Qwen3.5 模型补丁
（``qwen35_model.py``）中独立出来，使原生 Qwen 模型文件不再被
商汤 Loop 概念污染。

SenseNova Loop 架构（共享权重循环）的关键事实：
  - 模型物理层数为 ``len(model.layers)``，但推理时按一个**执行计划**
    （``_loop_plan``）重复访问中间层，因此 cache 按**槽**组织：
    槽数 == len(_loop_plan)，可能大于物理层数。
  - 物理层状态由各自槽的 cache 独立承载，MTP rollback 必须按槽处理
    而非按物理层，否则长度检查永不通过。

本模块导出：
  - ``_execution_plan(loop_settings, num_layers)`` —— 从物理层执行顺序
    生成槽→物理层映射元组。
  - ``install_loop(text_model, loop_settings)`` —— 在 MLX TextModel 上
    安装循环执行计划（设置 ``_loop_plan`` 与 ssm_idx/fa_idx）。
  - ``is_loop_model(model)`` —— 判断某内部模型是否为 Loop 架构。
  - ``mtp_slot_layout(model)`` —— 返回 ``(n_slots, phys_fn)`` 供 MTP
    rollback 使用；非 Loop 模型返回 None。

设计原则（关注点分离）：
  ``qwen35_model.py`` 只提供一个通用的 ``_mtp_slot_layout`` 扩展点
  （默认 identity，零 Loop 概念）；本模块通过
  ``sensenova_loop_mtp.apply()`` 注册 Loop 感知版本。依赖方向是
  sensenova_loop_mtp → qwen35_model（hook 注册），qwen35_model 对
  SenseNova 零感知。
"""

from __future__ import annotations

import mlx.nn as nn
from typing import Any, Callable, Optional, Tuple


def _execution_plan(loop_settings: Optional[dict], num_layers: int) -> Optional[Tuple[int, ...]]:
    """物理层执行顺序（移植 modeling_sensenova_loop.execution_plan）。

    无 loop 返回 None；否则返回::

        tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, num_layers))

    其中 ``[start, end)`` 是循环区间，``count`` 是每个循环分片的访问次数。
    """
    if not loop_settings:
        return None
    start = loop_settings.get("start")
    end = loop_settings.get("end")
    count = loop_settings.get("eval_count")
    if loop_settings.get("counts") != [count]:
        raise ValueError("Only fixed-count Loop checkpoints are supported")
    if not (0 <= start < end <= num_layers and count >= 1):
        raise ValueError("Invalid Loop interval or eval_count")
    return tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, num_layers))


def install_loop(text_model: Any, loop_settings: Optional[dict]) -> None:
    """安装循环执行计划（复刻 _configure_text_model 的 loop 部分）。

    ``text_model`` 是 MLX TextModel，其 ``.model`` 是内部
    ``Qwen3_5TextModel``（含 ``.layers``）。
    """
    if not loop_settings:
        return
    tm = text_model.model
    plan = _execution_plan(loop_settings, len(tm.layers))
    if plan is None:
        return
    tm._loop_plan = plan
    # 重置并指向执行计划中第一个对应类型的槽
    tm.ssm_idx = None
    tm.fa_idx = None
    for slot, phys in enumerate(plan):
        layer = tm.layers[phys]
        if tm.ssm_idx is None and layer.is_linear:
            tm.ssm_idx = slot
        if tm.fa_idx is None and not layer.is_linear:
            tm.fa_idx = slot
        if tm.ssm_idx is not None and tm.fa_idx is not None:
            break


def is_loop_model(model: Any) -> bool:
    """判断某内部模型（含 ``.layers``）是否为 SenseNova Loop 架构。"""
    return getattr(model, "_loop_plan", None) is not None


def mtp_slot_layout(model: Any) -> Optional[Tuple[int, Callable[[int], int]]]:
    """返回 MTP rollback 用的槽布局 ``(n_slots, phys_fn)``。

    - ``n_slots`` —— cache 槽数（== len(_loop_plan)，Loop 模型）；
    - ``phys_fn(slot)`` —— 槽 → 物理层索引映射。

    非 Loop 模型返回 None，调用方回退到通用 identity 布局
    （``n_slots = len(model.layers)``）。
    """
    plan = getattr(model, "_loop_plan", None)
    if plan is None:
        return None
    # 2026-11-10 loop-aware 前向（sensenova_loop_engine._wrap_forward）消费
    # 完整 ``_loop_plan``：``for slot, phys in enumerate(plan)``，槽 slot 的
    # cache 由物理层 ``plan[slot]`` 写入（Loop 重复分片：层 4-27 在槽 4-27
    # 与槽 28-51 各访问一次，物理层 28-31 只在槽 52-55 访问一次）。
    #
    # 因此 rollback 的 phys_fn 必须返回 ``plan[slot]`` —— 与 11-08 旧假设
    # （前向只用 zip(layers,cache) 消费 n_phys 槽）不同。旧版 phys_fn 返回
    # ``slot if slot < n_phys else None`` 在 loop-aware 前向下错误：槽 28-55
    # 被误判为死槽跳过回滚，KV/ssm_state 残留污染 → MTP-on 输出错误（off
    # 正确、on 有错）。
    #
    # 历史教训：11-08 曾用 plan[slot] 但当时前向只消费 32 槽（identity），
    # slot 28-31 → 层 4-7 导致 rollback 用错层重放（层 4 vs 层 28 权重不同）
    # → GDN ssm_state 错乱。现在前向已 loop-aware，plan[slot] 重新正确。
    return (
        len(plan),
        lambda slot: plan[slot],
    )


# ---------------------------------------------------------------------------
# MHC (multi-head cross/cycle) —— 移植自 modeling_sensenova_loop.py 的 mx 版。
#
# 与 Loop 一样，MHC 是 SenseNova Loop 架构的特化：在 ``[start, end)`` 物理层
# 之间用 ``streams`` 条残差流承载多流路由（pre 门控 + post 缩放 + sinkhorn
# 混合）。这些函数/类只被 ``install_mhc`` 使用，与通用 Qwen3.5 零耦合。
#
# 依赖方向：install_mhc → 上述 MHC 定义；qwen3_5.py 对 MHC 零感知。
# ---------------------------------------------------------------------------

def _mhc_sinkhorn(logits, iterations):
    """Normalize routing logits across rows and columns in log space (mx 移植)."""
    import mlx.core as mx

    logits = logits.astype(mx.float32)
    for _ in range(iterations):
        logits = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits - mx.logsumexp(logits, axis=-2, keepdims=True)
    return mx.exp(logits)


def _mhc_enter(hidden, physical, settings):
    """Expand to residual streams at the first mHC layer, or restore their shape."""
    import mlx.core as mx

    if physical == settings["start"]:
        return mx.contiguous(
            mx.broadcast_to(
                hidden[..., None, :],
                (*hidden.shape[:-1], settings["streams"], hidden.shape[-1]),
            )
        )
    return hidden.reshape(*hidden.shape[:-1], settings["streams"], -1)


def _mhc_leave(residual, physical, settings):
    """Reduce streams at the last mHC layer; otherwise carry them forward."""
    import mlx.core as mx

    if physical == settings["end"] - 1:
        return mx.mean(residual, axis=-2)
    return mx.reshape(residual, (*residual.shape[:-2], -1))


class MHCRouter(nn.Module):
    """多流残差路由：pre 门控 + post 缩放 + sinkhorn 混合（mx 移植，FP32 权重）。

    复刻 modeling_sensenova_loop.MHCRouter 的 mx 语义：权重/Bias 以 FP32 存储
    （模型整体可能被 cast 到 BF16，但路由参数必须保持 FP32）。

    必须继承 ``nn.Module``：mlx 的 ``load_weights`` 按模块树遍历参数，
    若为普通类则 ``mhc_attn.weight/bias`` 不会被注册为子模块参数，
    ``load_weights`` 会报 ``Received N parameters not in model``。
    """

    def __init__(self, dim, settings):
        super().__init__()
        import math
        import mlx.core as mx

        self.streams = settings["streams"]
        self.iterations = settings["iterations"]
        self.fixed_identity = settings["fixed_identity"]
        self.pre_scale = settings.get("pre_scale", 1.0)
        streams = self.streams
        output_size = 2 * streams + (0 if self.fixed_identity else streams * streams)
        # 权重/Bias 以 FP32 存储
        self.weight = mx.zeros((output_size, streams * dim), dtype=mx.float32)
        self.bias = mx.zeros((output_size,), dtype=mx.float32)
        # 初始化 bias（复刻 torch 逻辑）
        init_probs = {2: (0.25, 0.75), 4: (0.0625, 0.1875, 0.25, 0.5)}[streams]
        b = self.bias
        for i, p in enumerate(init_probs):
            b[i] = float(mx.log(mx.array(p) / (1 - mx.array(p))))
        if not self.fixed_identity:
            diag = math.log((1 - (streams - 1) / 16) * 16)
            b[2 * streams:] = (mx.eye(streams) * diag).reshape(-1)
        self.bias = b

    def __call__(self, residual):
        return self._row_call(residual)

    def _row_call(self, residual):
        import mlx.core as mx

        flat = mx.reshape(
            residual.astype(mx.float32), (*residual.shape[:-2], -1)
        )
        normalized = flat * mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + 1e-6)
        logits = normalized @ self.weight.T + self.bias
        streams = self.streams
        pre = mx.sigmoid(logits[..., :streams])
        post = 2 * mx.sigmoid(logits[..., streams:2 * streams])
        if self.fixed_identity:
            mix = mx.broadcast_to(
                mx.eye(streams, dtype=mx.float32),
                (*logits.shape[:-1], streams, streams),
            )
        else:
            mix = _mhc_sinkhorn(
                logits[..., 2 * streams:].reshape(*logits.shape[:-1], streams, streams),
                self.iterations,
            )
        # 与原版一致：branch 保持在 residual.dtype (BF16)，仅 logits/weight/bias 计算用 FP32
        branch = mx.sum(pre.astype(residual.dtype)[..., None] * residual, axis=-2)
        if self.pre_scale != 1.0:
            branch = branch * self.pre_scale
        return branch, post, mix

    @staticmethod
    def merge(residual, branch, post, mix):
        return MHCRouter._merge_rows(residual, branch, post, mix)

    @staticmethod
    def _merge_rows(residual, branch, post, mix):
        import mlx.core as mx

        # 与原版一致：结果保持在 residual.dtype (BF16)，mix/post 计算后再转回
        mixed = mx.matmul(mix.astype(residual.dtype), residual)
        update = post.astype(branch.dtype)[..., None] * branch[..., None, :]
        return mixed + update


def _mhc_patch_decoder_layer(cls: Any) -> Any:
    """给 DecoderLayer 注入 MHC 调用（复刻 MHCDecoderLayer 逻辑）。"""
    import mlx.core as mx
    _orig_call = cls.__call__

    def _patched_call(self, x, mask=None, cache=None, n_confirmed=0):
        settings = getattr(self, "_mhc_settings", None)
        if settings is None:
            return _orig_call(self, x, mask, cache, n_confirmed=n_confirmed)
        physical = self._mhc_physical
        residual = _mhc_enter(x, physical, settings)
        branch, post, mix = self.mhc_attn(residual)
        branch = self.input_layernorm(
            mx.contiguous(branch.astype(self.input_layernorm.weight.dtype))
        )
        if self.is_linear:
            if n_confirmed:
                r = self.linear_attn(branch, mask, cache, n_confirmed=n_confirmed)
            else:
                r = self.linear_attn(branch, mask, cache)
        else:
            r = self.self_attn(branch, mask, cache)
        residual = MHCRouter.merge(residual, r, post, mix)
        branch, post, mix = self.mhc_ffn(residual)
        branch = self.mlp(
            self.post_attention_layernorm(
                mx.contiguous(branch.astype(self.post_attention_layernorm.weight.dtype))
            )
        )
        return _mhc_leave(MHCRouter.merge(residual, branch, post, mix), physical, settings)

    cls.__call__ = _patched_call
    return cls


def install_mhc(model: Any, mhc_settings: Optional[dict]) -> None:
    """在模型上安装 MHC 模块（复刻 _configure_text_model 的 mhc 部分）。"""
    import mlx.core as mx

    if not mhc_settings:
        return
    start, end = mhc_settings["start"], mhc_settings["end"]
    for i in range(start, min(end, len(model.layers))):
        layer = model.layers[i]
        layer._mhc_physical = i
        layer._mhc_settings = dict(mhc_settings)
        layer.mhc_attn = MHCRouter(model.args.hidden_size, mhc_settings)
        layer.mhc_ffn = MHCRouter(model.args.hidden_size, mhc_settings)
        if not hasattr(layer, "_mhc_patched"):
            _mhc_patch_decoder_layer(type(layer))
            layer._mhc_patched = True
