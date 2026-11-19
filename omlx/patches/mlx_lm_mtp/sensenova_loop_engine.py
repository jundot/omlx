from __future__ import annotations

import logging
from typing import Any

from .sensenova_loop_model import install_loop, install_mhc

logger = logging.getLogger(__name__)


def configure_model(model: Any, args: Any) -> None:
    """从 ``ModelArgs`` 统一安装 Loop + MHC（编排入口）。

    在 ``Model.__init__`` 末尾调用：读 ``args.mhc`` / ``args.loop_settings``
    （由 sensenova_loop_mtp 包装的 ``ModelArgs.from_dict`` 附加为实例属性），
    分别调用 :func:`install_mhc` / :func:`install_loop`。

    非 SenseNova 模型（两字段均为 None）时为零操作，与通用 Qwen3.5 行为一致。
    """
    install_mhc(model, getattr(args, "mhc", None))
    install_loop(model, getattr(args, "loop_settings", None))
# ---------------------------------------------------------------------------
# SenseNova Loop/MHC 配置注入 —— 从 mlx_lm/models/qwen3_5.py 抽出的编排侧。
#
# 干净库文件（20k）的 ``ModelArgs`` 只有 model_type/text_config；
# ``Model.__init__`` 只建 language_model，不装 Loop/MHC。这里包装：
#   - ``ModelArgs.from_dict`` → 把 mhc/loop_settings 附加为实例属性；
#   - ``Model.__init__`` → 末尾调 ``sensenova_loop_model.configure_model``，
#     统一 install_mhc + install_loop。
# 非 SenseNova 模型（两字段 None）零操作，行为与通用 Qwen3.5 一致。
# ---------------------------------------------------------------------------

_MODEL_ARGS_WRAPPED = "_omlx_sensenova_loop_model_args_wrapped"
_MODEL_INIT_WRAPPED = "_omlx_sensenova_loop_model_init_wrapped"
_MAKE_CACHE_WRAPPED = "_omlx_sensenova_loop_make_cache_wrapped"
_FORWARD_WRAPPED = "_omlx_sensenova_loop_forward_wrapped"


def _wrap_make_cache(q35: Any) -> None:
    """包装 TextModel.make_cache 为 Loop 感知（复刻污染版的 loop 分支）。

    干净库文件（20k）的 ``make_cache`` 无 loop 分支，返回 ``len(layers)`` 槽
    （物理层数）。但 Loop 模型需要 ``len(_loop_plan)`` 槽（展开后，可能 >
    物理层数）；loop 感知的 ``mtp_partial_rollback`` 里 ``len(cache) != n_slots``
    会直接 False，导致退化为通用回滚（13:40 修复前的 bug）。

    本包装先查 ``_loop_plan``：Loop 模型返回展开槽数，非 Loop 模型调原始版
    （行为与通用 Qwen3.5 完全一致）。
    """
    cls = q35.TextModel
    if getattr(cls, _MAKE_CACHE_WRAPPED, False):
        return
    original_make_cache = cls.make_cache

    def make_cache(self):
        loop_plan = getattr(self.model, "_loop_plan", None)
        if loop_plan is not None:
            from mlx_lm.models.cache import ArraysCache, KVCache

            return [
                ArraysCache(size=2)
                if self.model.layers[phys].is_linear
                else KVCache()
                for phys in loop_plan
            ]
        return original_make_cache(self)

    cls.make_cache = make_cache
    setattr(cls, _MAKE_CACHE_WRAPPED, True)
    logger.debug("SenseNova Loop TextModel.make_cache wrapped (loop-aware)")


def _wrap_forward(q35: Any) -> None:
    """包装 ``Qwen3_5TextModel.__call__`` 为 Loop 感知（消费 ``_loop_plan``）。

    干净库文件（20k）的 ``Qwen3_5TextModel.__call__`` 按
    ``zip(pipeline_layers, cache)`` 顺序执行，不读 ``_loop_plan``。但
    ``install_loop`` 已把 ``_loop_plan`` 设为展开计划（可能 > 物理层数）。
    若前向不消费，Loop 重复分片槽永远不被触达，Loop 权重按标准 GDN 顺序
    解码 → 答非所问。本包装：

    - ``_loop_plan`` 存在（Loop 模型）→ 按 ``for slot, phys in enumerate(loop_plan)``
      复用物理层，每个 visit 独立 cache 槽（复刻污染版 qwen3_5.py L448-470）；
    - 非 Loop 模型 → 转发原始版（零影响）。

    必须在 ``qwen35_model`` 替换 ``Qwen3_5TextModel.__call__`` *之后* 执行
    （返回 pre-norm hidden 的约定保持一致），故由 sensenova_loop_mtp.apply()
    在 qwen35_model.apply() 成功后调用。
    """
    cls = q35.Qwen3_5TextModel
    # Self-healing: 即使 marker 已存在，也验证当前 __call__ 是否仍是我们的
    # loop 感知版。若被其他补丁（重载时 qwen35_verify_qmm / dflash 等）覆盖，
    # marker 在但 __call__ 已丢 —— 必须重新包装，否则新模型实例用非 loop
    # 前向 → 答非所问。marker 只作为快速路径，不作为跳过依据。
    current_call = cls.__dict__.get("__call__")
    if getattr(current_call, "_omlx_loop_fwd_marker", False):
        return
    original_call = cls.__call__

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        n_confirmed: int = 0,
        capture_layer_ids=None,
        capture_sink=None,
    ):
        loop_plan = getattr(self, "_loop_plan", None)
        if loop_plan is None:
            return original_call(
                self,
                inputs,
                cache,
                input_embeddings,
                n_confirmed,
                capture_layer_ids,
                capture_sink,
            )

        import mlx.core as mx

        # Loop 模式：按执行计划复用物理层，每个 visit 独立 cache 槽。
        if getattr(self, "pipeline_size", 1) > 1:
            raise ValueError("Loop execution does not support model parallelism")
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(loop_plan)

        create_attention_mask = q35.create_attention_mask
        create_ssm_mask = q35.create_ssm_mask
        fa_mask = None
        ssm_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        wanted = set(capture_layer_ids or ())
        captured = {}
        for slot, phys in enumerate(loop_plan):
            layer = self.layers[phys]
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(
                hidden_states, mask=mask, cache=cache[slot], n_confirmed=n_confirmed
            )
            if slot in wanted:
                captured[slot] = hidden_states
        if capture_sink is not None:
            capture_sink.extend(captured[i] for i in capture_layer_ids or ())

        # 与 qwen35_model 的 Qwen3_5TextModel.__call__ 约定一致：返回
        # pre-norm hidden（外层 TextModel.__call__ 应用 norm 产 logits）。
        return hidden_states

    __call__._omlx_loop_fwd_marker = True
    cls.__call__ = __call__
    setattr(cls, _FORWARD_WRAPPED, True)
    logger.info("SenseNova Loop Qwen3_5TextModel.__call__ wrapped (loop-aware forward)")


def _wrap_model_args(q35: Any) -> None:
    """包装 ModelArgs.from_dict，把 mhc/loop_settings 附加为实例属性。"""
    cls = q35.ModelArgs
    if getattr(cls, _MODEL_ARGS_WRAPPED, False):
        return
    original_from_dict = cls.from_dict.__func__

    @classmethod
    def from_dict(cls_, params):
        instance = original_from_dict(cls_, params)
        # 顶层 keys 里读 mhc/loop_settings（若存在）附加为实例属性。
        # BaseModelArgs.from_dict 会忽略未知 key，必须显式读出。
        if isinstance(params, dict):
            instance.mhc = params.get("mhc")
            instance.loop_settings = params.get("loop_settings")
        else:
            instance.mhc = getattr(instance, "mhc", None)
            instance.loop_settings = getattr(instance, "loop_settings", None)
        return instance

    cls.from_dict = from_dict
    setattr(cls, _MODEL_ARGS_WRAPPED, True)
    logger.debug("SenseNova Loop ModelArgs.from_dict wrapped (mhc/loop_settings)")


def _wrap_model_init(q35: Any) -> None:
    """包装 Model.__init__，末尾调 configure_model 注入 Loop+MHC。"""
    cls = q35.Model
    if getattr(cls, _MODEL_INIT_WRAPPED, False):
        return
    original_init = cls.__init__

    def __init__(self, args):
        original_init(self, args)
        # 统一安装 Loop + MHC（非 SenseNova 零操作）。
        configure_model(self.language_model, args)

    cls.__init__ = __init__
    setattr(cls, _MODEL_INIT_WRAPPED, True)
    logger.debug("SenseNova Loop Model.__init__ wrapped (configure_model)")


_SANITIZE_WRAPPED = "_omlx_sensenova_loop_sanitize_wrapped"


def _wrap_sanitize(q35: Any) -> None:
    """包装 Model.sanitize：修正 lm_head 路径（SenseNova 特有）。

    qwen3_5 的 Model.sanitize 把 ``model.language_model.<X>`` 统一替换为
    ``language_model.model.<X>``（加 ``.model.``）。但 SenseNova 的
    ``lm_head`` 挂在 ``language_model.lm_head``（TextModel 直接属性），
    不是 ``language_model.model.lm_head``。若不修正，lm_head 权重会被
    重映射到不存在的路径，报 "Received parameters not in model"。

    同时处理 ``model.language_model.loop_gate``：适配器循环逻辑用
    ``_loop_plan``（由 loop_settings 生成），不消费 loop_gate 权重；
    若 config 保留 loop_settings 但模型无 loop_gate 模块，剥离之。
    """
    cls = q35.Model
    if getattr(cls, _SANITIZE_WRAPPED, False):
        return
    original_sanitize = cls.sanitize

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            # SenseNova lm_head：language_model.lm_head（无 .model.）
            if key.startswith("model.language_model.lm_head"):
                key = "language_model.lm_head" + key[len("model.language_model.lm_head"):]
            # loop_gate：适配器不消费，剥离
            elif "loop_gate" in key:
                continue
            # 其余交给原版 sanitize 逻辑（保留 model.language_model 前缀由原版处理）
            sanitized[key] = value
        return original_sanitize(self, sanitized)

    cls.sanitize = sanitize
    setattr(cls, _SANITIZE_WRAPPED, True)
    logger.debug("SenseNova Loop Model.sanitize wrapped (lm_head path fix)")
