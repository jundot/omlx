# Qwen Image 2.1 text-to-image equations, native layout v1.
# Adapted from Qwen/Hugging Face (Apache-2.0) and mflux-community/mflux (MIT).
# See LICENSE in this directory for attribution and license terms.
# No package registry, downloads, or serving registration.
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class ZeroRMS(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((4096,))

    def __call__(self, x):
        v = x.astype(mx.float32)
        return (
            v
            * mx.rsqrt(mx.mean(v * v, axis=-1, keepdims=True) + 1e-6)
            * (1 + self.weight.astype(mx.float32))
        ).astype(x.dtype)


class TextProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_norm = ZeroRMS()
        self.in_layer = nn.Linear(4096, 4096, bias=False)
        self.out_layer = nn.Linear(4096, 4096, bias=False)

    def __call__(self, x):
        return self.out_layer(nn.gelu_approx(self.in_layer(self.text_norm(x))))


class TimestepEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(256, 4096, bias=False)
        self.linear_2 = nn.Linear(4096, 4096, bias=False)

    def __call__(self, x):
        return self.linear_2(nn.silu(self.linear_1(x)))


class TimeEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.timestep_embedder = TimestepEmbedder()

    def __call__(self, t, dtype):
        freq = mx.array(np.exp(-np.log(10000) * np.arange(128, dtype=np.float32) / 128))
        phase = t.astype(mx.float32)[:, None] * 1000 * freq[None]
        return self.timestep_embedder(
            mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1).astype(dtype)
        )


def rope(text_len, height, width):
    text = np.arange(text_len)
    axes = [
        np.concatenate([text, np.full(height * width, text_len)]),
        np.concatenate(
            [text, np.repeat(np.arange(-(height - height // 2), height // 2), width)]
        ),
        np.concatenate(
            [text, np.tile(np.arange(-(width - width // 2), width // 2), height)]
        ),
    ]
    phases = [
        np.outer(
            p.astype(np.float32),
            1 / (10000 ** (np.arange(0, d, 2, dtype=np.float32) / d)),
        )
        for p, d in zip(axes, [16, 56, 56])
    ]
    phase = np.concatenate(phases, axis=-1)
    return mx.array(np.cos(phase)), mx.array(np.sin(phase))


def rotate(x, cos, sin):
    v = x.astype(mx.float32).reshape(*x.shape[:-1], 64, 2)
    r, i = v[..., 0], v[..., 1]
    c, s = cos[None, :, None], sin[None, :, None]
    return (
        mx.stack([r * c - i * s, r * s + i * c], axis=-1)
        .reshape(x.shape)
        .astype(x.dtype)
    )


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4096, 4096, bias=False)
        self.to_k = nn.Linear(4096, 4096, bias=False)
        self.to_v = nn.Linear(4096, 4096, bias=False)
        self.to_out = [nn.Linear(4096, 4096, bias=False)]
        self.norm_q = nn.RMSNorm(128, eps=1e-6)
        self.norm_k = nn.RMSNorm(128, eps=1e-6)

    def __call__(self, x, cos, sin, text_len):
        b, n, _ = x.shape
        q = rotate(
            self.norm_q(self.to_q(x).reshape(b, n, 32, 128)), cos, sin
        ).transpose(0, 2, 1, 3)
        k = rotate(
            self.norm_k(self.to_k(x).reshape(b, n, 32, 128)), cos, sin
        ).transpose(0, 2, 1, 3)
        v = self.to_v(x).reshape(b, n, 32, 128).transpose(0, 2, 1, 3)
        a = mx.fast.scaled_dot_product_attention(
            q[:, :, :text_len],
            k[:, :, :text_len],
            v[:, :, :text_len],
            scale=128**-0.5,
            mask="causal",
        )
        z = mx.fast.scaled_dot_product_attention(
            q[:, :, text_len:], k, v, scale=128**-0.5
        )
        joined = (
            mx.concatenate([a, z], axis=2).transpose(0, 2, 1, 3).reshape(b, n, 4096)
        )
        return self.to_out[0](joined)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up = nn.Linear(4096, 24576, bias=False)
        self.out = nn.Linear(12288, 4096, bias=False)

    def __call__(self, x):
        gate, up = mx.split(self.gate_up(x), 2, axis=-1)
        return self.out(nn.silu(gate) * up)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.img_mlp = MLP()

    def __call__(self, x, mods, cos, sin, text_len):
        scale1, gate1, scale2, gate2 = mx.split(mods, 4, axis=-1)
        x = x + mx.tanh(gate1) * self.attn(
            mx.fast.layer_norm(x, None, None, 1e-6) * (1 + scale1), cos, sin, text_len
        )
        return x + mx.tanh(gate2) * self.img_mlp(
            mx.fast.layer_norm(x, None, None, 1e-6) * (1 + scale2)
        )


class OutputNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4096, 4096, bias=False)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_in = TextProjection()
        self.img_in = nn.Linear(64, 4096, bias=False)
        self.time_text_embed = TimeEmbed()
        self.modulation = nn.Linear(4096, 16384, bias=False)
        self.transformer_blocks = [Block() for _ in range(32)]
        self.norm_out = OutputNorm()
        self.proj_out = nn.Linear(4096, 64, bias=False)

    def denoise_steps(self, latents, embeds, sigma, height, width, *, cancel=None):
        from .conditioning import check_cancel

        if not (
            type(height) is int
            and type(width) is int
            and 1 <= height <= 64
            and 1 <= width <= 64
        ):
            raise ValueError("unsupported latent grid")
        if (
            latents.shape != (1, height * width, 64)
            or embeds.ndim != 3
            or embeds.shape[0] != 1
            or embeds.shape[2] != 4096
            or not 1 <= embeds.shape[1] <= 512
        ):
            raise ValueError("incompatible latents or conditioning")
        if not math.isfinite(sigma) or not 0 <= sigma <= 1:
            raise ValueError("sigma must be in [0, 1]")
        check_cancel(cancel)
        text_len = embeds.shape[1]
        n = latents.shape[1]
        t = self.time_text_embed(
            mx.array([sigma, 0.0], dtype=mx.float32), latents.dtype
        )
        mod = self.modulation(nn.silu(t))
        mods = mx.concatenate(
            [
                mx.broadcast_to(mod[1][None, None], (1, text_len, 16384)),
                mx.broadcast_to(mod[0][None, None], (1, n, 16384)),
            ],
            axis=1,
        )
        cos, sin = rope(text_len, height, width)
        x = mx.concatenate([self.txt_in(embeds), self.img_in(latents)], axis=1)
        for i, block in enumerate(self.transformer_blocks):
            check_cancel(cancel)
            x = block(x, mods, cos, sin, text_len)
            mx.eval(x)
            yield ("denoiser_block", i + 1, 32)
        scale = self.norm_out.linear(nn.silu(t))[0][None, None]
        return self.proj_out(
            mx.fast.layer_norm(x[:, text_len:], None, None, 1e-6) * (1 + scale)
        )

    def __call__(self, latents, embeds, sigma, height, width, *, cancel=None):
        from .conditioning import run_steps

        return run_steps(
            self.denoise_steps(latents, embeds, sigma, height, width, cancel=cancel)
        )
