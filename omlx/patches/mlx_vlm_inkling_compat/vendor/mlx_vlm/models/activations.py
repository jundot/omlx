from functools import partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def swiglu(gate, x):
    return nn.silu(gate) * x


def xielu(x, alpha_p, alpha_n, beta, eps):
    alpha_p = nn.softplus(alpha_p)
    alpha_n = beta + nn.softplus(alpha_n)
    return mx.where(
        x > 0,
        alpha_p * mx.square(x) + beta * x,
        (mx.expm1(mx.minimum(x, eps)) - x) * alpha_n + beta * x,
    )


class XieLU(nn.Module):
    def __init__(
        self,
        alpha_p_init=0.8,
        alpha_n_init=0.8,
        beta=0.5,
        eps=-1e-6,
    ):
        super().__init__()
        alpha_p_tensor = mx.array(alpha_p_init)
        alpha_n_tensor = mx.array(alpha_n_init - beta)
        self.alpha_p = mx.log(mx.exp(alpha_p_tensor) - 1)
        self.alpha_n = mx.log(mx.exp(alpha_n_tensor) - 1)

        self.beta = mx.array(beta)
        self.eps = mx.array(eps)

    def __call__(self, x: mx.array) -> mx.array:
        return xielu(x, self.alpha_p, self.alpha_n, self.beta, self.eps)
