import mlx.core as mx
from mlx import nn

# Standalone validation-only native VAE, adapted from mflux-community/mflux.
# Upstream MIT license and pinned source revision: see LICENSE in this directory.
# Qwen/Hugging Face portions retain Apache-2.0 notices below.
# Changes: local imports; config supplied by exported artifact; no MFLUX dependency.


class Qwen21VAEConfig:
    @staticmethod
    def resolve(config):
        config = dict(config)
        config["temporal_downsample"] = config.get(
            "temporal_downsample", config.get("temperal_downsample")
        )
        return config


class Qwen21AvgDown(nn.Module):
    # Parameterless shortcut of the residual down blocks: average pooling over the
    # (temporal, spatial) factor, then a channel-group mean that maps in -> out channels.
    # Single-frame inputs with factor_t=2 are front-padded with a zero frame (causal pad),
    # exactly like the reference.

    def __init__(
        self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if in_channels * self.factor % out_channels != 0:
            raise ValueError(
                f"in_channels ({in_channels}) * factor ({self.factor}) must be divisible by out_channels ({out_channels})"
            )
        self.group_size = in_channels * self.factor // out_channels

    def __call__(self, x: mx.array) -> mx.array:
        b, c, h, w = x.shape
        hs, ws = self.factor_s, self.factor_s
        x = mx.expand_dims(x, 2)  # (B, C, 1, H, W)
        if self.factor_t == 2:
            x = mx.pad(x, [(0, 0), (0, 0), (1, 0), (0, 0), (0, 0)])
        x = x.reshape(b, c, 1, self.factor_t, h // hs, hs, w // ws, ws)
        x = mx.transpose(
            x, (0, 1, 3, 5, 7, 2, 4, 6)
        )  # (B, C, ft, fsh, fsw, T', H', W')
        x = x.reshape(b, c * self.factor, h // hs, w // ws)
        x = x.reshape(b, self.out_channels, self.group_size, h // hs, w // ws)
        return mx.mean(x, axis=2)


class Qwen21DupUp(nn.Module):
    # Parameterless shortcut of the residual up blocks: channel repeat + rearrange into
    # nearest (temporal, spatial) upsampling. The duplicated first temporal slot is always
    # dropped so a single-frame input stays single-frame.

    def __init__(
        self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if out_channels * self.factor % in_channels != 0:
            raise ValueError(
                f"out_channels ({out_channels}) * factor ({self.factor}) must be divisible by in_channels ({in_channels})"
            )
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x: mx.array) -> mx.array:
        b, _, h, w = x.shape
        ft, fsh, fsw = self.factor_t, self.factor_s, self.factor_s
        x = mx.repeat(x, self.repeats, axis=1)
        x = x.reshape(b, self.out_channels, ft, fsh, fsw, 1, h, w)
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))  # (B, out, T, ft, H, fsh, W, fsw)
        x = x.reshape(b, self.out_channels, ft, h * fsh, w * fsw)
        x = x[:, :, ft - 1 :, :, :]
        return mx.squeeze(x, 2)


# Copyright 2026 The Qwen Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AutoencoderKLQwenImage21; see LICENSE in this directory.


class VAEOperations:
    @staticmethod
    def conv(layer: nn.Module, x: mx.array, text_mode: bool) -> mx.array:
        if isinstance(layer, nn.Identity):
            return x
        if text_mode:
            return layer(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        return layer(x)


class ChannelNorm(nn.Module):
    def __init__(self, channels: int, text_mode: bool = False):
        super().__init__()
        self.gamma = mx.ones((channels,))
        self.text_mode = text_mode
        self.scale = float(channels) ** 0.5

    def __call__(self, x: mx.array) -> mx.array:
        axis = 1 if self.text_mode else -1
        value = x.astype(mx.float32)
        norm = mx.maximum(
            mx.sqrt(mx.sum(value * value, axis=axis, keepdims=True)), 1e-12
        )
        value = value / norm
        if self.text_mode:
            return (value * self.scale * self.gamma[None, :, None, None]).astype(
                x.dtype
            )
        return value.astype(x.dtype) * self.scale * self.gamma


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, text_mode: bool = False):
        super().__init__()
        self.text_mode = text_mode
        self.norm1 = ChannelNorm(in_dim, text_mode)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm2 = ChannelNorm(out_dim, text_mode)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = VAEOperations.conv(self.conv_shortcut, x, self.text_mode)
        x = VAEOperations.conv(self.conv1, nn.silu(self.norm1(x)), self.text_mode)
        x = VAEOperations.conv(self.conv2, nn.silu(self.norm2(x)), self.text_mode)
        return x + shortcut if self.text_mode else shortcut + x


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, text_mode: bool = False):
        super().__init__()
        self.text_mode = text_mode
        self.norm = ChannelNorm(dim, text_mode)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        normalized = self.norm(x)
        if self.text_mode:
            normalized = normalized.transpose(0, 2, 3, 1)
        b, h, w, c = normalized.shape
        qkv = self.to_qkv(normalized)
        # Keep the text path's explicit attention arithmetic and the edit path's SDPA.
        if self.text_mode:
            qkv = qkv.reshape(b, h * w, 3, c)
            q, k, v = qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :]
            scale = 1.0 / mx.sqrt(mx.array(float(c)))
            scores = mx.matmul(q, k.transpose(0, 2, 1)) * scale
            output = mx.matmul(mx.softmax(scores, axis=-1), v)
        else:
            q, k, v = mx.split(qkv.reshape(b, 1, h * w, 3 * c), 3, axis=-1)
            output = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5)
        output = self.proj(output.reshape(b, h, w, c))
        return output.transpose(0, 3, 1, 2) + x if self.text_mode else x + output


class MidBlock(nn.Module):
    def __init__(self, dim: int, text_mode: bool = False):
        super().__init__()
        self.resnets = [
            ResidualBlock(dim, dim, text_mode),
            ResidualBlock(dim, dim, text_mode),
        ]
        self.attentions = [AttentionBlock(dim, text_mode)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class Resample(nn.Module):
    def __init__(
        self,
        dim: int,
        up: bool,
        temporal: bool,
        text_mode: bool = False,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.up = up
        self.text_mode = text_mode
        self.resample = [
            nn.Identity(),
            nn.Conv2d(
                dim, out_dim or dim, 3, stride=1 if up else 2, padding=1 if up else 0
            ),
        ]
        # Only edit checkpoints retain unused first-frame temporal convolution weights.
        if temporal and not text_mode:
            self.time_conv = nn.Conv2d(dim, dim * 2 if up else dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        if self.text_mode:
            x = x.transpose(0, 2, 3, 1)
        if self.up:
            x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        else:
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        x = self.resample[1](x)
        return x.transpose(0, 3, 1, 2) if self.text_mode else x


class DownBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_blocks: int,
        down: bool,
        temporal: bool,
        text_mode: bool = False,
    ):
        super().__init__()
        self.text_mode = text_mode
        self.resnets = [
            ResidualBlock(in_dim if i == 0 else out_dim, out_dim, text_mode)
            for i in range(num_blocks)
        ]
        self.downsampler = (
            Resample(out_dim, up=False, temporal=temporal, text_mode=text_mode)
            if down
            else None
        )
        self.avg_shortcut = Qwen21AvgDown(
            in_dim, out_dim, 2 if temporal else 1, 2 if down else 1
        )

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = self._shortcut(x)
        for block in self.resnets:
            x = block(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + shortcut

    def _shortcut(self, x: mx.array) -> mx.array:
        shortcut = self.avg_shortcut(x if self.text_mode else x.transpose(0, 3, 1, 2))
        return shortcut if self.text_mode else shortcut.transpose(0, 2, 3, 1)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_blocks: int,
        up: bool,
        temporal: bool,
        text_mode: bool = False,
    ):
        super().__init__()
        self.text_mode = text_mode
        self.resnets = [
            ResidualBlock(in_dim if i == 0 else out_dim, out_dim, text_mode)
            for i in range(num_blocks + 1)
        ]
        self.upsampler = (
            Resample(out_dim, up=True, temporal=temporal, text_mode=text_mode)
            if up
            else None
        )
        self.avg_shortcut = (
            Qwen21DupUp(in_dim, out_dim, 2 if temporal else 1) if up else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        original = x
        for block in self.resnets:
            x = block(x)
        if self.upsampler is not None:
            shortcut = self.avg_shortcut(
                original if self.text_mode else original.transpose(0, 3, 1, 2)
            )
            if not self.text_mode:
                shortcut = shortcut.transpose(0, 2, 3, 1)
            x = self.upsampler(x) + shortcut
        return x


class Encoder(nn.Module):
    def __init__(self, config: dict, text_mode: bool = False):
        super().__init__()
        self.text_mode = text_mode
        dims = [config["base_dim"] * factor for factor in [1, *config["dim_mult"]]]
        self.conv_in = nn.Conv2d(config["in_channels"], dims[0], 3, padding=1)
        temporal = config["temporal_downsample"]
        self.down_blocks = [
            DownBlock(
                a,
                b,
                config["num_res_blocks"],
                i < len(dims) - 2,
                temporal[i] if i < len(temporal) else False,
                text_mode,
            )
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))
        ]
        self.mid_block = MidBlock(dims[-1], text_mode)
        self.norm_out = ChannelNorm(dims[-1], text_mode)
        self.conv_out = nn.Conv2d(dims[-1], config["z_dim"] * 2, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = VAEOperations.conv(self.conv_in, x, self.text_mode)
        for block in self.down_blocks:
            x = block(x)
            if not self.text_mode:
                mx.eval(x)
        return VAEOperations.conv(
            self.conv_out, nn.silu(self.norm_out(self.mid_block(x))), self.text_mode
        )


class Decoder(nn.Module):
    def __init__(self, config: dict, text_mode: bool = False):
        super().__init__()
        self.text_mode = text_mode
        dim_mult = config["dim_mult"]
        dims = [
            config["decoder_base_dim"] * factor
            for factor in [dim_mult[-1], *reversed(dim_mult)]
        ]
        self.conv_in = nn.Conv2d(config["z_dim"], dims[0], 3, padding=1)
        self.mid_block = MidBlock(dims[0], text_mode)
        temporal = list(reversed(config["temporal_downsample"]))
        self.up_blocks = [
            UpBlock(
                a,
                b,
                config["num_res_blocks"],
                i < len(dims) - 2,
                temporal[i] if i < len(temporal) else False,
                text_mode,
            )
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))
        ]
        self.norm_out = ChannelNorm(dims[-1], text_mode)
        self.conv_out = nn.Conv2d(dims[-1], config["out_channels"], 3, padding=1)

    def __call__(self, x: mx.array, *, cancel=None) -> mx.array:
        from .conditioning import check_cancel

        check_cancel(cancel)
        x = self.mid_block(VAEOperations.conv(self.conv_in, x, self.text_mode))
        for block in self.up_blocks:
            check_cancel(cancel)
            x = block(x)
            mx.eval(x)
        return VAEOperations.conv(
            self.conv_out, nn.silu(self.norm_out(x)), self.text_mode
        )


class Qwen21VAECore(nn.Module):
    spatial_scale = 16
    latent_channels = 64

    def __init__(self, config: dict | None = None, *, text_mode: bool = False):
        super().__init__()
        config = Qwen21VAEConfig.resolve(config)
        if not config["is_residual"] or config["patch_size"] is not None:
            raise ValueError(
                "Qwen-Image-2.1 requires the residual, unpatched image VAE."
            )
        self.encoder = Encoder(config, text_mode)
        self.decoder = Decoder(config, text_mode)
        self.latent_channels = config["z_dim"]
        self.spatial_scale = 2 ** (len(config["dim_mult"]) - 1)
        self.quant_conv = nn.Conv2d(
            self.latent_channels * 2, self.latent_channels * 2, 1
        )
        self.post_quant_conv = nn.Conv2d(self.latent_channels, self.latent_channels, 1)
        self._mean = tuple(config["latents_mean"])
        self._std = tuple(config["latents_std"])
        if (
            len(self._mean) != self.latent_channels
            or len(self._std) != self.latent_channels
        ):
            raise ValueError("VAE latent mean and std must match z_dim.")

    @staticmethod
    def _single_frame(value: mx.array) -> mx.array:
        if value.ndim == 5:
            if value.shape[2] != 1:
                raise ValueError("Qwen-Image-2.1 supports single-frame images only.")
            value = value[:, :, 0]
        if value.ndim != 4:
            raise ValueError("Qwen-Image-2.1 expects NCHW or NC1HW images.")
        return value


class Qwen21VAE(Qwen21VAECore):

    def __init__(self, config: dict | None = None):
        super().__init__(config, text_mode=True)

    def decode(self, latents: mx.array, *, cancel=None) -> mx.array:
        latents = self._single_frame(latents)
        mean = mx.array(self._mean, dtype=mx.float32).reshape(
            1, self.latent_channels, 1, 1
        )
        std = mx.array(self._std, dtype=mx.float32).reshape(
            1, self.latent_channels, 1, 1
        )
        latents = latents * std + mean
        latents = VAEOperations.conv(self.post_quant_conv, latents, True)
        return self.decoder(latents, cancel=cancel)[:, :3]

    def decode_steps(self, latents, *, cancel=None):
        from .conditioning import check_cancel

        check_cancel(cancel)
        latents = self._single_frame(latents)
        mean = mx.array(self._mean, dtype=mx.float32)[None, :, None, None]
        std = mx.array(self._std, dtype=mx.float32)[None, :, None, None]
        x = VAEOperations.conv(self.post_quant_conv, latents * std + mean, True)
        x = self.decoder.mid_block(VAEOperations.conv(self.decoder.conv_in, x, True))
        mx.eval(x)
        yield ("vae_stage", 1, 6)
        for i, block in enumerate(self.decoder.up_blocks):
            check_cancel(cancel)
            x = block(x)
            mx.eval(x)
            yield ("vae_stage", i + 2, 6)
        return VAEOperations.conv(
            self.decoder.conv_out, nn.silu(self.decoder.norm_out(x)), True
        )[:, :3]
