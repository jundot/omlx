import itertools
import math

import mlx.nn as nn
import numpy as np

from .config import VisionConfig


def _prime_factors(n: int):
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 2
    if n > 1:
        factors.append(n)
    return factors


def _linear_sum_assignment(cost: np.ndarray):
    """Minimal-cost injective assignment of rows -> distinct columns (R <= C).
    Dependency-free stand-in for scipy.optimize.linear_sum_assignment (the grids are tiny).
    """
    R, C = cost.shape
    best_cost, best = None, None
    for perm in itertools.permutations(range(C), R):
        s = sum(cost[r, perm[r]] for r in range(R))
        if best_cost is None or s < best_cost:
            best_cost, best = s, perm
    return list(range(R)), list(best)


def plan_out_scales(temporal_patch_size, patch_size, n_layers, n_channels):
    """Per-layer (t, h, w, c) grids for the HMLP encoder — faithful port of the reference."""
    h = np.cumprod(np.array(_prime_factors(patch_size)[::-1], dtype=np.int64))
    t = np.cumprod(np.array(_prime_factors(temporal_patch_size)[::-1], dtype=np.int64))
    h_ch = np.ceil(h**2 * n_channels / 64).astype(np.int64) * 64
    t_ch = np.ceil(h[-1] ** 2 * n_channels * t / 64).astype(np.int64) * 64

    base = np.array([[1, 1, 1, n_channels]], dtype=np.int64)
    spatial = np.stack([np.ones_like(h), h, h, h_ch], axis=1)
    temporal = np.stack(
        [t, np.full_like(t, h[-1]), np.full_like(t, h[-1]), t_ch], axis=1
    )
    scales = np.concatenate([base, spatial, temporal], axis=0)

    size_reduction = np.prod(scales[:, :-1], axis=1).astype(np.float64)
    total_elements = patch_size * patch_size * temporal_patch_size * n_channels
    log_ideal = np.linspace(0.0, math.log(total_elements), n_layers + 1)
    cost = np.abs(log_ideal[:, None] - np.log(size_reduction)[None, :])

    if n_layers + 1 >= scales.shape[0]:
        idxs = np.argmin(cost, axis=1)
    else:
        _, idxs = _linear_sum_assignment(cost)
        idxs = np.array(idxs)
    idxs[0] = 0
    idxs[-1] = scales.shape[0] - 1
    return scales[idxs]


def fold_timespace_to_depth(x, t_fold, hw_fold):
    """(B, T, H, W, C) -> (B, T//t, H//hw, W//hw, C * t * hw * hw)."""
    B, T, H, W, C = x.shape
    tn, hn, wn = T // t_fold, H // hw_fold, W // hw_fold
    x = x.reshape(B, tn, t_fold, hn, hw_fold, wn, hw_fold, C)
    x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    return x.reshape(B, tn, hn, wn, t_fold * hw_fold * hw_fold * C)


class InklingVisionEncoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, t_fold, hw_fold, add_norm, eps):
        super().__init__()
        self.t_fold = t_fold
        self.hw_fold = hw_fold
        self.add_norm = add_norm
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        if add_norm:
            self.layer_norm = nn.RMSNorm(output_dim, eps=eps)

    def __call__(self, x):
        if self.hw_fold > 1 or self.t_fold > 1:
            x = fold_timespace_to_depth(x, self.t_fold, self.hw_fold)
        x = self.projection(x)
        if self.add_norm:
            x = nn.gelu(self.layer_norm(x))
        return x


class VisionModel(nn.Module):
    """Hierarchical-MLP (HMLP) patchifier: progressively folds space/time into channels and
    projects each patch to one LM-space soft token. No attention."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        n_layers = config.n_layers
        scales = plan_out_scales(
            config.temporal_patch_size,
            config.patch_size,
            n_layers,
            config.num_channels,
        )
        n_last = n_layers - 1
        self.encoder_layers = []
        for i in range(len(scales) - 1):
            s, e = scales[i], scales[i + 1]
            shuffle_mult = int((e[0] // s[0]) * (e[1] // s[1]) * (e[2] // s[2]))
            output_dim = config.text_hidden_size if i == n_last else int(e[3])
            self.encoder_layers.append(
                InklingVisionEncoderLayer(
                    input_dim=int(s[3]) * shuffle_mult,
                    output_dim=output_dim,
                    t_fold=int(e[0] // s[0]),
                    hw_fold=int(e[1] // s[1]),
                    add_norm=i != n_last,
                    eps=config.rms_norm_eps,
                )
            )
        self.final_norm = nn.RMSNorm(config.text_hidden_size, eps=config.rms_norm_eps)

    def __call__(self, pixel_values):
        """pixel_values: [num_patches, T, H, W, C] -> [num_patches, text_hidden_size]."""
        num_patches = pixel_values.shape[0]
        h = pixel_values
        for layer in self.encoder_layers:
            h = layer(h)
        h = self.final_norm(h)
        return h.reshape(num_patches, -1)
