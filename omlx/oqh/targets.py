"""Which tensors an oQh build quantizes, and the rotation each one reads.

Every linear reading one producer output shares one sign vector: the producer
rotates its output once. In families that run tensor parallel, linears whose
input axis is split across ranks (attention and GDN output, down projections)
use blocks that fit one rank.
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass

from . import format as fmt

_LAYER = re.compile(r"^(?P<prefix>.*?)layers\.(?P<idx>\d+)\.(?P<rest>.+)$")

# Producer of each target, by the path below ``layers.<i>``.
_QWEN_ROLES = {
    "linear_attn.in_proj_qkv": "in",
    "linear_attn.in_proj_z": "in",
    "linear_attn.in_proj_a": "in",
    "linear_attn.in_proj_b": "in",
    "self_attn.q_proj": "in",
    "self_attn.k_proj": "in",
    "self_attn.v_proj": "in",
    "linear_attn.out_proj": "own",
    "self_attn.o_proj": "own",
    "mlp.gate_proj": "post",
    "mlp.up_proj": "post",
    "mlp.down_proj": "own",
    "mlp.shared_expert.gate_proj": "post",
    "mlp.shared_expert.up_proj": "post",
    "mlp.shared_expert.down_proj": "own",
    "mlp.switch_mlp.gate_proj": "post",
    "mlp.switch_mlp.up_proj": "post",
    "mlp.switch_mlp.down_proj": "own",
}
_ROUTED_ROLES = {
    k: v for k, v in _QWEN_ROLES.items() if k.startswith("mlp.switch_mlp.")
}
# DeepSeek V4.1 names its routed experts w1 (gate), w3 (up) and w2 (down).
_V41_ROUTED_ROLES = {
    "ffn.experts.w1": "post",
    "ffn.experts.w3": "post",
    "ffn.experts.w2": "own",
}

# Narrowest rotation block the producer kernels take.
_MIN_BLOCK = 128


def _split(path: str):
    path = path.removesuffix(".weight")
    m = _LAYER.match(path)
    if m is None or "mtp" in m.group("prefix"):
        return None
    return m


@dataclass(frozen=True)
class Scheme:
    """The oQh targets of one model family."""

    roles: dict
    tensor_parallel: bool

    def role(self, path: str) -> str | None:
        m = _split(path)
        return None if m is None else self.roles.get(m.group("rest"))

    def rotation(self, path: str, in_dim: int) -> tuple[str, int]:
        """(signs key, block) of a target's input rotation."""
        m = _split(path)
        rest = m.group("rest")
        layer = f"layers.{m.group('idx')}"
        if self.roles[rest] == "own":
            block = fmt.block_for(in_dim, row_parallel=self.tensor_parallel)
            return f"{layer}.{rest}", block
        return f"{layer}.{self.roles[rest]}", fmt.block_for(in_dim)

    def is_target(self, path: str, shape: tuple | None = None) -> bool:
        """A decoder linear the oQh kernels can serve.

        The GEMV covers 8 output rows per threadgroup and the producers rotate
        blocks of at least 128 inputs.
        """
        if self.role(path) is None:
            return False
        if shape is None:
            return True
        return (
            len(shape) >= 2
            and shape[-2] % 8 == 0
            and shape[-1] % _MIN_BLOCK == 0
            and self.rotation(path, shape[-1])[1] >= _MIN_BLOCK
        )


# Qwen3.5, Qwen3.6 and Qwen3.8 checkpoints all use qwen3_5 / qwen3_5_moe.
QWEN = Scheme(_QWEN_ROLES, tensor_parallel=True)
# Qwen3.8-Flash-Next keeps its trunk affine: its hyper-connection, attention and
# GDN kernels read affine weights. It does not run tensor parallel.
QWEN4_EXP = Scheme(_ROUTED_ROLES, tensor_parallel=False)
# GLM-5.3 keeps its trunk affine for the same reason, and runs tensor parallel.
GLM5_NEXT = Scheme(_ROUTED_ROLES, tensor_parallel=True)
# DeepSeek V4.1 likewise: its trunk, Engram tables and shared expert keep
# their oQ formats.
DEEPSEEK_V41 = Scheme(_V41_ROUTED_ROLES, tensor_parallel=True)
_SCHEMES = {
    "deepseek_v41": DEEPSEEK_V41,
    "qwen3_5": QWEN,
    "qwen3_5_moe": QWEN,
    "qwen4_exp": QWEN4_EXP,
    "glm5_next": GLM5_NEXT,
}


def scheme_for(config: dict) -> Scheme | None:
    text = config.get("text_config") or {}
    for model_type in (config.get("model_type"), text.get("model_type")):
        name = str(model_type or "").replace("-", "_").removesuffix("_text")
        if name in _SCHEMES:
            return _SCHEMES[name]
    return None


def supports(config: dict) -> bool:
    return scheme_for(config) is not None


# Tie order among targets of the same width: down projections first measured best
# on Qwen3.5-9B, then the mixers, then gate / up.
_REINVEST_RANK = {
    "down_proj": 0,
    "gate_proj": 2,
    "up_proj": 2,
    "w2": 0,
    "w1": 2,
    "w3": 2,
}
_BITS_LADDER = (2, 3, 4, 5, 6, 8)


def _reinvest_rank(path: str) -> int:
    return _REINVEST_RANK.get(path.rsplit(".", 1)[-1], 1)


def _affine_bytes(shape: tuple, spec: dict) -> int:
    n = 1
    for d in shape:
        n *= d
    rows = n // shape[-1]
    groups = shape[-1] // int(spec["group_size"])
    return (n * int(spec["bits"]) + 7) // 8 + rows * groups * 4


def _partner(path: str, bits: dict) -> str | None:
    for a, b in (
        ("gate_proj", "up_proj"),
        ("up_proj", "gate_proj"),
        ("w1", "w3"),
        ("w3", "w1"),
    ):
        if path.endswith(a):
            other = path[: -len(a)] + b
            return other if other in bits else None
    return None


def reinvest(named_shapes: dict, planned: dict, layer_scores: dict) -> dict[str, int]:
    """{path: bits} of the oQh targets within the bytes of their affine plan.

    planned: {path: {"bits", "group_size"}} the main pass would write.
    At equal width a row scale rounds coarser than affine group scales, so the
    bytes the group scales free must buy bits where the error is largest: the
    narrowest target gets one more bit first (water-filling), ties broken by
    _REINVEST_RANK, then the most sensitive layer. Gate and up of one MLP at
    the same width rise together, so they still fuse into one projection.
    """
    bits = {p: int(v["bits"]) for p, v in planned.items()}
    slack = sum(
        _affine_bytes(named_shapes[p], v) - price(named_shapes[p], bits[p])
        for p, v in planned.items()
    )

    def key(path: str) -> tuple:
        layer = int(_split(path).group("idx"))
        score = float(layer_scores.get(str(layer), 0.0))
        return (bits[path], _reinvest_rank(path), -score, layer, path)

    heap = [key(p) for p in bits]
    heapq.heapify(heap)
    while heap:
        entry = heapq.heappop(heap)
        p = entry[-1]
        if entry[0] != bits[p]:
            continue  # Raised with its partner since this entry was pushed.
        wider = next((b for b in _BITS_LADDER if b > bits[p]), None)
        if wider is None:
            continue
        partner = _partner(p, bits)
        group = (p, partner) if partner and bits[partner] == bits[p] else (p,)
        delta = sum(
            price(named_shapes[q], wider) - price(named_shapes[q], bits[q])
            for q in group
        )
        if delta > slack:
            continue
        slack -= delta
        for q in group:
            bits[q] = wider
            heapq.heappush(heap, key(q))
    return bits


def price(shape: tuple, bits: int) -> int:
    """Serialized bytes of a target at ``bits``: codes, row scales and signs."""
    n = 1
    for d in shape:
        n *= d
    rows = n // shape[-1]
    return (n * bits + 7) // 8 + 2 * rows + 2 * shape[-1]
