import mlx.nn as nn

from .activations import swiglu


class SwiGLUMLP(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DeepseekMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class GELUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class FastGELUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = nn.GELU(approx="fast")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class TanhGELUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act = nn.GELU(approx="tanh")

    def __call__(self, x):
        return self.fc2(self.act(self.fc1(x)))
