# SPDX-License-Identifier: Apache-2.0
"""LLM prompt extension for the video generation path.

Wan2.2 officially recommends prompt extension (its --use_prompt_extend
flag): a terse prompt like "a person stretching" adheres poorly because
the model's text encoder gets too little to ground the motion on, and it
falls back to a high-frequency near-miss action. Expanding the prompt
into an explicit, detailed description of the SUBJECT, the ACTION/motion,
the SCENE, the CAMERA and the LIGHTING -- before it reaches the worker --
is the single highest-leverage fix for instruction following.

This runs SERVER-SIDE (the worker venv is mflux-only and cannot reach the
engine pool) by calling a configured chat model through the engine pool.
It is best-effort: any failure falls back to the original prompt so a
rewrite problem never blocks video generation.
"""

from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

# Strip a reasoning model's <think>...</think> block from the raw output so
# only the rewritten prompt survives (recommended model is a small
# non-reasoning LLM, but guard anyway).
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")

# Output ceiling: enough for a rich paragraph, short enough that a runaway
# generation can't bloat the worker prompt or stall the request.
_MAX_NEW_TOKENS = 512
_DEFAULT_TIMEOUT_S = 60.0

_EN_SYS_PROMPT = (
    "You are a prompt engineer for a text-to-video diffusion model. Rewrite "
    "the user's brief prompt into ONE vivid, concrete English paragraph that "
    "the video model can follow precisely. Rules:\n"
    "- Preserve the user's original intent, subject and setting exactly. Do "
    "not invent a different scene or contradict anything they specified.\n"
    "- ELABORATE THE ACTION in explicit physical detail: describe the motion "
    "step by step (which body parts move, in which direction, at what speed, "
    "with what posture). This is the most important part -- a vague verb "
    "produces the wrong motion.\n"
    "- Add helpful cinematic detail: scene and background, camera framing and "
    "movement, lighting and mood, visual style. Keep it grounded, not flowery.\n"
    "- Keep it under ~80 words. Output ONLY the rewritten prompt, as a single "
    "paragraph, with no preamble, quotes, labels or explanation."
)

_ZH_SYS_PROMPT = (
    "你是文生视频扩散模型的提示词工程师。把用户的简短提示词改写成一段具体、"
    "画面感强的中文描述, 让视频模型能精确执行。规则:\n"
    "- 完全保留用户的原意, 主体和场景, 不要编造不同的画面或与用户指定的内容矛盾.\n"
    "- 重点详述动作: 一步步描述这个动作怎么发生 (哪些身体部位如何运动, 朝哪个方向, "
    "速度快慢, 什么姿态). 这是最关键的部分 -- 含糊的动词会让模型做出错误的动作.\n"
    "- 补充有用的影视化细节: 场景背景, 镜头取景与运镜, 光线与氛围, 视觉风格. "
    "写实为主, 不要堆砌辞藻.\n"
    "- 控制在 80 字以内. 只输出改写后的提示词, 一段话, 不要前言, 引号, 标签或解释."
)


def _clean_rewrite(raw: str) -> str:
    """Extract the rewritten prompt from a model's raw chat output."""
    text = _THINK_RE.sub("", raw or "").strip()
    # Some models still prefix a label or wrap in quotes; trim conservatively.
    text = text.strip().strip('"').strip("'").strip()
    return text


def _system_prompt_for(prompt: str) -> str:
    """Pick the system prompt by the input language so the rewrite stays in
    the user's language (Wan's text encoder is multilingual; preserving the
    language avoids translation drift)."""
    return _ZH_SYS_PROMPT if _CJK_RE.search(prompt or "") else _EN_SYS_PROMPT


async def extend_video_prompt(
    prompt: str,
    *,
    model_id: str,
    engine_pool,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> tuple[str, bool]:
    """Expand ``prompt`` via ``model_id`` through the engine pool.

    Returns ``(final_prompt, extended)``. ``extended`` is True only when a
    non-empty rewrite was produced; on any error / empty result the original
    prompt is returned with ``extended=False`` so the caller can proceed
    unconditionally. Never raises.
    """
    if not model_id or not (prompt or "").strip():
        return prompt, False
    try:
        engine = await asyncio.wait_for(
            engine_pool.get_engine(model_id), timeout=timeout_s
        )
        messages = [
            {"role": "system", "content": _system_prompt_for(prompt)},
            {"role": "user", "content": prompt.strip()},
        ]
        output = await asyncio.wait_for(
            engine.chat(
                messages=messages,
                max_tokens=_MAX_NEW_TOKENS,
                temperature=0.7,
            ),
            timeout=timeout_s,
        )
        rewritten = _clean_rewrite(getattr(output, "text", "") or "")
        if not rewritten:
            logger.warning(
                "Video prompt extension via %s produced empty output; "
                "using original prompt",
                model_id,
            )
            return prompt, False
        logger.info(
            "Video prompt extended via %s: %d -> %d chars",
            model_id,
            len(prompt),
            len(rewritten),
        )
        return rewritten, True
    except asyncio.TimeoutError:
        logger.warning(
            "Video prompt extension via %s timed out after %.0fs; using "
            "original prompt",
            model_id,
            timeout_s,
        )
        return prompt, False
    except Exception as e:
        logger.warning(
            "Video prompt extension via %s failed (%s: %s); using original "
            "prompt",
            model_id,
            type(e).__name__,
            e,
        )
        return prompt, False
