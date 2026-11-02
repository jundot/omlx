# SPDX-License-Identifier: Apache-2.0
"""Faster Ming-3.0 decode loop for mlx-audio's bailingmm model.

mlx-audio builds every decode step eagerly: one LM step, then ten Euler steps
of an 8-block DiT with CFG, then the 8-block aggregator, for each 80 ms
latent. It also blocks on the stop head every step, so the CPU and GPU take
turns. This patch keeps the same math and changes how it runs:

- The CFM sampler and the one-latent aggregator are compiled. Weights are
  compile inputs, so a stranded compile cache entry does not pin them.
- Rotary tables, timestep embeddings and Euler step sizes are computed once.
  The condition and history embeddings are computed once per decode step.
- The loop runs one step ahead: it queues CFM_k, the aggregator, LM_{k+1} and
  stop_{k+1}, then blocks on stop_k, which was queued one iteration earlier.

Noise is drawn outside the compiled function in mlx-audio's order, so the
output matches the eager loop for the same seed. The exception is bf16
checkpoints: their DiT stacks run on the fused kernels in
``ming3_dit_kernels``, which round differently. The decode-step pipeline
follows ming-tts-serve (MIT) ``src/ming_model.zig``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import mlx.core as mx

from . import ming3_dit_kernels as fused

try:
    from mlx_audio.tts.models.bailingmm import bailingmm
except ImportError:  # mlx-audio is optional.
    bailingmm = None

logger = logging.getLogger(__name__)

_STATE_ATTR = "_omlx_ming3_state"


class _CachedRope:
    """Stand-in for ``RotaryEmbedding`` with fixed cos/sin tables."""

    def __init__(self, rotary, seq_len: int):
        self._rotary = rotary
        pos = mx.arange(0, seq_len, dtype=rotary.inv_freq.dtype)
        freqs = mx.outer(pos, rotary.inv_freq)
        self.cos = mx.cos(freqs)[None, None, :, :]
        self.sin = mx.sin(freqs)[None, None, :, :]
        mx.eval(self.cos, self.sin)

    def __call__(self, q: mx.array, k: mx.array, offset: int = 0):
        rotary = self._rotary
        return (
            rotary._apply(q, self.cos, self.sin),
            rotary._apply(k, self.cos, self.sin),
        )


class _Ming3State:
    """Per-model tables and compiled functions."""

    def __init__(self, model):
        self.model = model
        self.dit_rope = _CachedRope(model.flowloss.cfm.model.rotary_embed, 3)
        agg = model.linear_proj_audio
        self.agg_rope = _CachedRope(agg.rotary_embed, 2)
        self.agg_cls = agg.word_embedder(mx.zeros((1, 1), dtype=mx.int32))
        mx.eval(self.agg_cls)
        gpu = mx.default_device() == mx.gpu
        self.fused_cfm = gpu and fused.cfm_supported(model.flowloss.cfm.model)
        self.fused_agg = gpu and fused.stack_supported(agg.blocks)
        self._cfm: dict[tuple, Any] = {}
        self._agg = None

    def aggregate(self, latent: mx.array) -> mx.array:
        if self._agg is None:
            agg = self.model.linear_proj_audio
            rope = self.agg_rope
            cls = self.agg_cls
            use_fused = self.fused_agg

            def agg_fn(x):
                x = agg.x_embedder(x)
                x = mx.concatenate([cls, x], axis=1)
                for block in agg.blocks:
                    if use_fused and x.dtype == mx.bfloat16:
                        x = fused.dit_block(block, x, rope.cos, rope.sin)
                    else:
                        x = block(x, None, rope)
                x = agg.final_layer(x)
                return x[:, :1, :]

            self._agg = mx.compile(agg_fn, inputs=agg.state)
        return self._agg(latent)

    def cfm(self, cfg: float, steps: int, dtype):
        """Compiled ``cfm(z, hist, noise, sde, shift)``.

        ``shift`` is sigma * sqrt(temperature) in the solver dtype. It is an
        input so that per-request temperatures reuse one compiled graph.
        """
        key = (float(cfg), int(steps), dtype)
        fn = self._cfm.get(key)
        if fn is None:
            fn = self._build_cfm(*key)
            self._cfm[key] = fn
        return fn

    def _build_cfm(self, cfg, steps, dtype):
        dit = self.model.flowloss.cfm.model
        rope = self.dit_rope

        # CFM.sample with use_epss and sway coefficient -1, in the solver dtype.
        t = bailingmm.get_epss_timesteps(steps, dtype=dtype)
        t = t + -1.0 * (mx.cos(math.pi / 2 * t) - 1 + t)
        t_emb, dts, sqs = [], [], []
        for i in range(1, t.shape[0]):
            t0 = t[i - 1]
            dt = t[i] - t0
            t_emb.append(dit.t_embedder(mx.full((2,), t0, dtype=dtype))[:, None, :])
            dts.append(dt.astype(dtype))
            sqs.append(mx.abs(dt) ** 0.5)
        mx.eval(t_emb, dts, sqs)

        if self.fused_cfm and dtype == mx.bfloat16 and steps > 0:
            return self._build_fused_cfm(dit, t_emb, dts, sqs, cfg, steps)

        def cfm_fn(z, hist, noise, sde, shift):
            c_embed = dit.c_embedder(
                mx.concatenate([z, mx.zeros_like(z)], axis=0), train=False
            )
            x_history = dit.x_embedder(mx.concatenate([hist, hist], axis=0))
            y = noise.astype(dtype)
            for i in range(steps):
                x_now = dit.x_embedder(mx.concatenate([y, y], axis=0))
                x_full = mx.concatenate([x_history, x_now], axis=1)
                x_full = mx.concatenate([t_emb[i] + c_embed, x_full], axis=1)
                for block in dit.blocks:
                    x_full = block(x_full, None, rope)
                out = dit.final_layer(x_full)[:, -1:, :]
                pred, null_pred = mx.split(out, 2, axis=0)
                f0 = (pred + (pred - null_pred) * cfg).astype(dtype)
                y = y + dts[i] * f0
                y = y + shift * sqs[i] * sde[i]
            return y

        return mx.compile(cfm_fn, inputs=dit.state)

    def _build_fused_cfm(self, dit, t_emb, dts, sqs, cfg, steps):
        rope = self.dit_rope
        dtt = mx.stack(dts)
        sq = mx.stack(sqs)
        mx.eval(dtt, sq)

        def cfm_fn(z, hist, noise, sde, shift):
            c_embed = dit.c_embedder(
                mx.concatenate([z, mx.zeros_like(z)], axis=0), train=False
            )
            x_history = dit.x_embedder(hist)
            y = noise
            for i in range(steps):
                h = fused.cfm_embed(y, dit.x_embedder, t_emb[i], c_embed, x_history)
                for block in dit.blocks:
                    h = fused.dit_block(block, h, rope.cos, rope.sin)
                y = fused.cfm_final(h, dit.final_layer, y, dtt, sq, sde, shift, i, cfg)
            return y

        return mx.compile(cfm_fn, inputs=dit.state)


def _state(model) -> _Ming3State:
    state = getattr(model, _STATE_ATTR, None)
    if state is None:
        state = _Ming3State(model)
        setattr(model, _STATE_ATTR, state)
    return state


def release_ming3_state(model: Any) -> None:
    """Drop compiled functions; call on the thread that ran them."""
    if model is not None and getattr(model, _STATE_ATTR, None) is not None:
        setattr(model, _STATE_ATTR, None)


def _pipelined_sample(
    self,
    prompt: str,
    text: str,
    instruction: str | None = None,
    prompt_waveform: mx.array | None = None,
    prompt_text: str | None = None,
    max_decode_steps: int | None = None,
    cfg: float = 2.0,
    sigma: float = 0.25,
    temperature: float = 0.0,
    flow_steps: int = 10,
    prompt_latent: mx.array | None = None,
    spk_emb=None,
):
    if prompt_text is None:
        prompt_latent = None
    elif prompt_latent is None and prompt_waveform is not None:
        prompt_latent = self._encode_prompt_latent(prompt_waveform)

    _, inputs_embeds = self._prepare_input_embed(
        prompt=prompt,
        text=text,
        instruction=instruction,
        prompt_latent=prompt_latent,
        prompt_text=prompt_text,
        spk_emb=spk_emb,
    )

    cache = [bailingmm.KVCache() for _ in range(self.llm_args.num_hidden_layers)]
    if max_decode_steps is None:
        max_decode_steps = 1000
    max_len = bailingmm.MING3_MAX_CACHE_LEN
    budget = (max_len - inputs_embeds.shape[1]) // self.patch_size
    if budget <= 0:
        raise ValueError(
            f"Ming-3.0 prompt has {inputs_embeds.shape[1]} positions and "
            f"exhausts the {max_len}-position cache. Shorten the text, "
            "instruction, or reference audio."
        )
    max_decode_steps = min(max_decode_steps, budget)
    if max_decode_steps <= 0:
        return

    state = _state(self)

    def stop_prob(z):
        return mx.softmax(self.stop_head(z), axis=-1)[0, 0, 1]

    z = self._forward_llm_hidden(inputs_embeds=inputs_embeds, cache=cache)[:, -1:, :]
    stop = stop_prob(z)

    latent_history = mx.zeros(
        (1, self.history_patch_size, self.latent_dim), dtype=z.dtype
    )
    if prompt_latent is not None:
        prompt_latent = prompt_latent.astype(z.dtype)
        start = self.history_patch_size - prompt_latent.shape[1]
        if start < 0:
            latent_history = prompt_latent[:, -start:, :]
        else:
            latent_history = mx.concatenate(
                [latent_history[:, :start, :], prompt_latent], axis=1
            )

    cfm = state.cfm(cfg, flow_steps, z.dtype)
    # Solver.integrate computes sigma * temperature**0.5 as a Python float.
    shift = mx.array(sigma * (temperature**0.5), dtype=z.dtype)
    channels = self.flowloss.z_channels
    for step in range(max_decode_steps):
        # Same draws, shapes and order as FlowLoss.sample in Ming-3.0 mode.
        noise = mx.random.normal((z.shape[0], self.patch_size, channels), dtype=z.dtype)
        sde_noise = mx.random.normal((flow_steps, *noise.shape), dtype=z.dtype)
        latent = cfm(z, latent_history, noise, sde_noise, shift)
        last_step = step + 1 == max_decode_steps
        if not last_step:
            z_next = self._forward_llm_hidden(
                inputs_embeds=state.aggregate(latent), cache=cache
            )[:, -1:, :]
            stop_next = stop_prob(z_next)
            mx.async_eval(latent, z_next, stop_next)
        else:
            mx.async_eval(latent)
        is_last = step > 3 and stop.item() > 0.5
        yield latent, is_last
        if is_last or last_step:
            break
        latent_history = mx.concatenate(
            [latent_history[:, self.patch_size :, :], latent], axis=1
        )
        z, stop = z_next, stop_next


def apply_ming3_patches() -> bool:
    """Install the Ming-3.0 decode loop on mlx-audio's bailingmm Model."""
    if bailingmm is None:
        return False

    model_cls = bailingmm.Model
    original = model_cls.sample
    if getattr(original, "_omlx_ming3", False):
        return True

    def sample(self, *args, **kwargs):
        # Ming Omni models decode 4-latent patches with a streaming VAE.
        if not self.ming3:
            return original(self, *args, **kwargs)
        return _pipelined_sample(self, *args, **kwargs)

    sample._omlx_ming3 = True
    sample._omlx_original = original
    model_cls.sample = sample
    logger.debug("Ming-3.0 decode loop installed")
    return True
