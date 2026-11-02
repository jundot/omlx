# SPDX-License-Identifier: Apache-2.0
"""Ming-3.0 speed and upstream-parity patches for mlx-audio's bailingmm model.

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

The other changes align Ming-3.0 with the upstream inclusionAI code:

- Text fragments count kana and Hangul like upstream ``get_semantic_length``.
- Voice clones get the CAM++ speaker embedding from the reference clip.
- The reference latent is sampled with the Oobleck posterior
  ``std = softplus(scale) + 1e-4``.
- A built-in voice combined with an instruction is a request error.

Ming-3.0 Tiny (mlx-audio ``lingv3_tts``) runs the same DiT, 16 blocks deep,
under a float32 Euler loop that dominates its decode step. Its
``FlowLoss.sample`` is compiled the same way, with the fused DiT blocks on
bf16 checkpoints.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ..exceptions import InvalidRequestError
from . import ming3_dit_kernels as fused

try:
    from mlx_audio.tts.models.bailingmm import bailingmm
    from mlx_audio.utils import load_audio, resample_audio
except ImportError:  # mlx-audio is optional.
    bailingmm = None

try:
    from mlx_audio.codec.models.stepaudio2.convert import (
        _map_stepaudio2_campplus_key,
    )
    from mlx_audio.codec.models.stepaudio2.speaker import StepAudio2CAMPPlus
except ImportError:  # Only voice clones need CAM++.
    StepAudio2CAMPPlus = None

try:
    from mlx_audio.tts.models.lingv3_tts import flow as tiny_flow
except ImportError:  # Older mlx-audio pins have no Ming-3.0 Tiny.
    tiny_flow = None

logger = logging.getLogger(__name__)

_STATE_ATTR = "_omlx_ming3_state"
_SPEAKER_CACHE_SIZE = 8

_LATIN = "A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ"
# One unit per CJK, kana or Hangul character and per Latin word.
_SEMANTIC_UNIT_RE = re.compile(
    "[一-鿿ぁ-ゖゝ-ゟァ-ヺー-ヿ" "가-힣]" f"|[{_LATIN}]+(?:['’][{_LATIN}]+)*"
)

# Folded CAM++ convolutions in ONNX graph order. Their initializers are the
# anonymous ``onnx::Conv_<n>`` weight/bias pairs, sorted by n.
_CAMPPLUS_CONVS = (
    ["head.conv1"]
    + [
        f"head.layer{layer}.{block}.{module}"
        for layer in (1, 2)
        for block, module in (
            (0, "conv1"),
            (0, "conv2"),
            (0, "shortcut.0"),
            (1, "conv1"),
            (1, "conv2"),
        )
    ]
    + ["head.conv2", "tdnn.linear"]
    + [
        f"blocks.{block}.layers.{layer}.linear1"
        for block, count in enumerate((12, 24, 16))
        for layer in range(count)
    ]
    + ["transits.2.linear"]
)


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


def _dit_rows(dit, rope, current, x_history, c_embed, t_emb, *, use_fused, pair):
    """DiT rows ``[t + c, history, latent]`` after every block, for one Euler step.

    ``pair`` repeats the latent for the two CFG halves on the MLX path. The
    fused EMBED kernel always writes both halves from one ``x_history`` row.
    """
    if use_fused:
        h = fused.cfm_embed(current, dit.x_embedder, t_emb, c_embed, x_history)
        for block in dit.blocks:
            h = fused.dit_block(block, h, rope.cos, rope.sin)
        return h
    x_now = dit.x_embedder(
        mx.concatenate([current, current], axis=0) if pair else current
    )
    h = mx.concatenate([x_history, x_now], axis=1)
    h = mx.concatenate([t_emb + c_embed, h], axis=1)
    for block in dit.blocks:
        h = block(h, None, rope)
    return h


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
        self._speaker_encoder = _load_speaker_encoder(model)
        self._speakers: OrderedDict = OrderedDict()

    def speaker_embedding(self, ref_audio) -> mx.array | None:
        """CAM++ embedding [1, 192] of a reference clip, cached by content."""
        if self._speaker_encoder is None:
            return None

        if isinstance(ref_audio, (str, Path)):
            wave = np.asarray(load_audio(str(ref_audio), sample_rate=16000))
        else:
            wave = np.asarray(ref_audio, dtype=np.float32)
            if wave.ndim == 2:
                # generate() takes [1, T]; other 2-D input is [T, channels].
                wave = wave[0] if wave.shape[0] == 1 else wave.mean(axis=-1)
            wave = np.asarray(resample_audio(wave, self.model.sample_rate, 16000))
        wave = wave.astype(np.float32)
        key = hashlib.sha256(wave.tobytes()).hexdigest()
        embedding = self._speakers.get(key)
        if embedding is not None:
            self._speakers.move_to_end(key)
            return embedding
        # Upstream extracts CAM++ on the CPU (onnxruntime). inference() takes
        # 16 kHz audio and computes the mean-normalized Kaldi fbank.
        with mx.stream(mx.cpu):
            embedding = self._speaker_encoder.inference(mx.array(wave))
            mx.eval(embedding)
        self._speakers[key] = embedding
        if len(self._speakers) > _SPEAKER_CACHE_SIZE:
            self._speakers.popitem(last=False)
        return embedding

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
                h = _dit_rows(
                    dit,
                    rope,
                    y,
                    x_history,
                    c_embed,
                    t_emb[i],
                    use_fused=False,
                    pair=True,
                )
                out = dit.final_layer(h)[:, -1:, :]
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
                h = _dit_rows(
                    dit,
                    rope,
                    y,
                    x_history,
                    c_embed,
                    t_emb[i],
                    use_fused=True,
                    pair=True,
                )
                y = fused.cfm_final(h, dit.final_layer, y, dtt, sq, sde, shift, i, cfg)
            return y

        return mx.compile(cfm_fn, inputs=dit.state)


class _TinyFlowState:
    """Compiled float32 Euler samplers for one Ming-3.0 Tiny ``FlowLoss``."""

    def __init__(self, flowloss):
        dit = flowloss.cfm.model
        self.dit = dit
        self.rope = _CachedRope(dit.rotary_embed, 3)
        gpu = mx.default_device() == mx.gpu
        self.fused = gpu and fused.cfm_supported(dit)
        self._fns: dict[tuple, Any] = {}

    def sampler(self, cfg: float, steps: int, dtype):
        """Compiled ``fn(hidden, history, state, noises, shift)``.

        ``shift`` is sigma * sqrt(temperature) in float32 and ``noises`` are
        the per-step SDE draws, drawn in the eager loop's order. Seeded calls
        match the eager loop except on the fused bf16 path.
        """
        key = (float(cfg), int(steps), dtype)
        fn = self._fns.get(key)
        if fn is None:
            fn = self._build(*key)
            self._fns[key] = fn
        return fn

    def _build(self, cfg, steps, dtype):
        dit = self.dit
        rope = self.rope
        # The lingv3_tts FlowLoss.sample schedule, in float32.
        times = tiny_flow.get_epss_timesteps(steps, mx.float32)
        times = times - (mx.cos(math.pi / 2 * times) - 1 + times)
        t_emb = [
            dit.t_embedder(mx.full((1,), times[i], dtype=dtype))[:, None, :]
            for i in range(steps)
        ]
        dts = [times[i + 1] - times[i] for i in range(steps)]
        sqs = [mx.sqrt(mx.abs(dt)) for dt in dts]
        mx.eval(t_emb, dts, sqs)
        # A cfg of 1 runs only the conditional branch, and a cfg near 0 only
        # the null one.
        two_branches = not (cfg == 1 or cfg < 1e-5)
        use_fused = self.fused and two_branches and dtype == mx.bfloat16
        if use_fused:
            # EMBED reads one timestep row per CFG half.
            t_emb = [mx.concatenate([te, te], axis=0) for te in t_emb]
            mx.eval(t_emb)

        def fn(hidden, history, state, noises, shift):
            cond = hidden.astype(dtype)
            past = history.astype(dtype)
            if two_branches:
                c_embed = dit.c_embedder(
                    mx.concatenate([cond, mx.zeros_like(cond)], axis=0), train=False
                )
                x_history = dit.x_embedder(
                    past if use_fused else mx.concatenate([past, past], axis=0)
                )
            else:
                c_embed = dit.c_embedder(
                    cond if cfg == 1 else mx.zeros_like(cond), train=False
                )
                x_history = dit.x_embedder(past)
            for i in range(steps):
                h = _dit_rows(
                    dit,
                    rope,
                    state.astype(dtype),
                    x_history,
                    c_embed,
                    t_emb[i],
                    use_fused=use_fused,
                    pair=two_branches,
                )
                # The MLX path keeps every row so its rounding matches upstream.
                rows = h[:, -1:, :] if use_fused else h
                out = dit.final_layer(rows)[:, -1:, :]
                if two_branches:
                    prediction, unconditional = mx.split(out, 2, axis=0)
                    prediction = prediction + (prediction - unconditional) * cfg
                else:
                    prediction = out
                state = state + dts[i] * prediction.astype(mx.float32)
                # The upstream solver skips the noise after the last step.
                if i < steps - 1:
                    state = state + shift * sqs[i] * noises[i]
            return state

        return mx.compile(fn, inputs=dit.state)


def _tiny_flow_sample(
    self,
    hidden,
    history,
    *,
    cfg=2.0,
    sigma=0.25,
    temperature=0.0,
    steps=10,
):
    """lingv3_tts ``FlowLoss.sample`` for one-latent patches and history."""
    if cfg < 0 or temperature < 0 or steps < 1 or sigma < 0:
        raise ValueError(
            "Flow CFG, sigma and temperature must be nonnegative and steps positive"
        )
    state = mx.random.normal(
        (hidden.shape[0], self.latent_dim, 1), dtype=mx.float32
    ).transpose(0, 2, 1)
    # Same draws and order as the eager loop, including the unused last one.
    noises = mx.stack(
        [mx.random.normal(state.shape, dtype=mx.float32) for _ in range(steps)]
    )
    flow_state = getattr(self, _STATE_ATTR, None)
    if flow_state is None:
        flow_state = _TinyFlowState(self)
        setattr(self, _STATE_ATTR, flow_state)
    dtype = self.cfm.model.x_embedder.weight.dtype
    fn = flow_state.sampler(cfg, steps, dtype)
    shift = mx.array(sigma * math.sqrt(temperature), dtype=mx.float32)
    return fn(hidden, history, state, noises, shift)


def _semantic_length(text: str) -> int:
    return len(_SEMANTIC_UNIT_RE.findall(text))


def _protobuf_fields(buf):
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        wire = key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == 2:
            size, pos = _varint(buf, pos)
            value, pos = buf[pos : pos + size], pos + size
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield key >> 3, wire, value


def _varint(buf, pos):
    value = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _packed_varints(buf):
    pos = 0
    while pos < len(buf):
        value, pos = _varint(buf, pos)
        yield value


def _onnx_float_initializers(path: Path) -> dict[str, np.ndarray]:
    """Float32 ``graph.initializer`` tensors of an ONNX file, without onnx."""
    tensors = {}
    for field, wire, graph in _protobuf_fields(memoryview(path.read_bytes())):
        if field != 7 or wire != 2:  # ModelProto.graph
            continue
        for gfield, gwire, tensor in _protobuf_fields(graph):
            if gfield != 5 or gwire != 2:  # GraphProto.initializer
                continue
            dims, dtype, name, raw, floats = [], 0, "", None, bytearray()
            for tfield, twire, value in _protobuf_fields(tensor):
                if tfield == 1:
                    if twire == 0:
                        dims.append(value)
                    else:
                        dims.extend(_packed_varints(value))
                elif tfield == 2:
                    dtype = value
                elif tfield == 4:  # float_data, packed or one value per field
                    floats += value
                elif tfield == 8:
                    name = bytes(value).decode()
                elif tfield == 9:
                    raw = value
                elif tfield == 14:
                    raise ValueError(f"{path.name} uses external tensor data")
            if dtype == 1:  # FLOAT
                data = raw if raw is not None else floats
                tensors[name] = np.frombuffer(data, dtype="<f4").reshape(dims)
    return tensors


def _campplus_tensors(model_dir: Path) -> dict[str, mx.array] | None:
    converted = model_dir / "campplus.safetensors"
    if converted.exists():
        return mx.load(str(converted))
    onnx = model_dir / "campplus.onnx"
    if onnx.exists():
        return {k: mx.array(v) for k, v in _onnx_float_initializers(onnx).items()}
    return None


def _campplus_from_tensors(raw: dict[str, mx.array]):
    """mlx-audio's native CAM++ loaded with a checkpoint's CAM++ initializers."""
    conv_ids = sorted(
        int(key.rsplit("_", 1)[1]) for key in raw if key.startswith("onnx::Conv_")
    )
    if len(conv_ids) != 2 * len(_CAMPPLUS_CONVS):
        raise ValueError(
            f"Unexpected CAM++ layout: {len(conv_ids)} anonymous conv tensors"
        )
    named = {}
    for i, base in enumerate(_CAMPPLUS_CONVS):
        named[f"{base}.weight"] = raw[f"onnx::Conv_{conv_ids[2 * i]}"]
        named[f"{base}.bias"] = raw[f"onnx::Conv_{conv_ids[2 * i + 1]}"]
    for key, value in raw.items():
        if not key.startswith("onnx::"):
            named[_map_stepaudio2_campplus_key(key)] = value

    net = StepAudio2CAMPPlus()
    expected = dict(tree_flatten(net.parameters()))
    weights = []
    for key, value in named.items():
        if key not in expected:
            continue
        if value.shape != expected[key].shape:
            # Torch conv layouts to MLX.
            value = (
                value.transpose(0, 2, 3, 1) if value.ndim == 4 else value.swapaxes(1, 2)
            )
        weights.append((key, value))
    net.load_weights(weights, strict=True)
    net.eval()
    mx.eval(net.parameters())
    return net


def _load_speaker_encoder(model):
    """CAM++ for voice clones, or None when the checkpoint has no usable copy.

    Cloning still works without it, as in stock mlx-audio, so a missing or
    unreadable file does not fail the model load.
    """
    model_path = getattr(model.config, "model_path", None)
    if StepAudio2CAMPPlus is None:
        logger.warning("mlx-audio has no CAM++ module; voice clones run without it")
        return None
    try:
        raw = _campplus_tensors(Path(model_path)) if model_path else None
        if raw is not None:
            return _campplus_from_tensors(raw)
        reason = "no campplus weights"
    except Exception as exc:
        reason = f"unreadable campplus weights ({exc})"
    logger.warning(
        "Ming-3.0 checkpoint has %s; voice clones will run without a speaker "
        "embedding",
        reason,
    )
    return None


def _state(model) -> _Ming3State:
    state = getattr(model, _STATE_ATTR, None)
    if state is None:
        state = _Ming3State(model)
        setattr(model, _STATE_ATTR, state)
    return state


def prepare_ming3_state(model: Any) -> None:
    """Build the per-model state at load time.

    CAM++ construction draws its initial weights from the global RNG. At
    request time that would shift the noise of a seeded generation.
    """
    if getattr(model, "ming3", False):
        _state(model)


def release_ming3_state(model: Any) -> None:
    """Drop compiled functions; call on the thread that ran them."""
    # Ming-3.0 keeps its state on the model, Ming-3.0 Tiny on its FlowLoss.
    for owner in (model, getattr(model, "flowloss", None)):
        if owner is not None and getattr(owner, _STATE_ATTR, None) is not None:
            setattr(owner, _STATE_ATTR, None)


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


def _encode_prompt_latent(self, prompt_waveform: mx.array) -> mx.array:
    waveform = self._pad_prompt_waveform(prompt_waveform)
    h = self.audio.encoder(waveform).transpose(0, 2, 1)
    mean, scale = mx.split(h, 2, axis=1)
    # diffusers OobleckDiagonalGaussianDistribution, as in upstream encode_latent.
    std = nn.softplus(scale) + 1e-4
    latent = mean + std * mx.random.normal(mean.shape, dtype=mean.dtype)
    return latent.transpose(0, 2, 1)


def _wrap_generate(original):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def generate(self, *args, **kwargs):
        if not self.ming3:
            return original(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        arguments = bound.arguments
        extra = arguments.setdefault("kwargs", {})
        voice = arguments.get("voice")
        ref_audio = arguments.get("ref_audio")
        if (
            voice
            and ref_audio is None
            and arguments.get("instruct") is not None
            and bailingmm._MING3_VOICE_RE.match(str(voice))
        ):
            raise InvalidRequestError(
                f"Ming-3.0 voice '{voice}' cannot be combined with instructions",
                field="instructions",
            )
        if (
            ref_audio is not None
            and arguments.get("ref_text") is not None
            and extra.get("spk_emb") is None
            and not extra.get("use_zero_spk_emb", False)
        ):
            embedding = _state(self).speaker_embedding(ref_audio)
            if embedding is not None:
                extra["spk_emb"] = embedding
        return original(*bound.args, **bound.kwargs)

    return generate


def _apply_tiny_patch() -> None:
    flow_cls = tiny_flow.FlowLoss
    original = flow_cls.sample
    if getattr(original, "_omlx_ming3", False):
        return

    def sample(self, hidden, history, *, patch_size=1, **kwargs):
        # The compiled DiT runs one row each of condition, history and latent.
        if patch_size != 1 or hidden.shape[:2] != (1, 1) or history.shape[:2] != (1, 1):
            return original(self, hidden, history, patch_size=patch_size, **kwargs)
        return _tiny_flow_sample(self, hidden, history, **kwargs)

    sample._omlx_ming3 = True
    sample._omlx_original = original
    flow_cls.sample = sample


def apply_ming3_patches() -> bool:
    """Install the Ming-3.0 and Ming-3.0 Tiny patches.

    Returns whether the Ming-3.0 (bailingmm) patches are installed.
    """
    if tiny_flow is not None:
        _apply_tiny_patch()
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

    original_encode = model_cls._encode_prompt_latent

    def encode_prompt_latent(self, prompt_waveform):
        if not self.ming3:
            return original_encode(self, prompt_waveform)
        return _encode_prompt_latent(self, prompt_waveform)

    sample._omlx_ming3 = True
    sample._omlx_original = original
    model_cls.sample = sample
    model_cls._encode_prompt_latent = encode_prompt_latent
    model_cls.generate = _wrap_generate(model_cls.generate)
    # _split_fragments looks this up at call time; only Ming-3.0 splits text.
    bailingmm._semantic_length = _semantic_length
    logger.debug("Ming-3.0 patches installed")
    return True
