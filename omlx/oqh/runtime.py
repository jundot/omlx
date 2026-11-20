"""oQh runtime: producer fusion without changing any caller.

Linears that read one producer output share signs and a block. The RMSNorm in
front of them computes the plain and the rotated output in one kernel, returns
the plain one (routers and every other reader see the usual value) and leaves
the rotated one in the oQh memo keyed by the returned array, where the linears
pick it up. MLP and SwitchGLU forwards are replaced on the instance only: gate
and up run as one paired projection, and the activation writes the rotated
input of down. Other models of the same class in the process are untouched.
"""

from __future__ import annotations

import functools
import itertools

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

from ..custom_kernels.oqh import fast
from .modules import (
    OQH_DENSE,
    OQH_MODULES,
    OQH_SWITCH,
    QMV_MAX_ROWS,
    exact_rows,
    hand_off,
    remember,
)

_KEYS = itertools.count(1)
_SUBCLASSES: dict[tuple[type, type], type] = {}


def _rebase(module: nn.Module, mixin: type) -> None:
    """Give one instance the mixin's forward without touching its class."""
    base = type(module)
    if issubclass(base, mixin):
        return
    cls = _SUBCLASSES.get((base, mixin))
    if cls is None:
        cls = type(f"Oqh{base.__name__}", (mixin, base), {})
        _SUBCLASSES[(base, mixin)] = cls
    module.__class__ = cls


def _set(obj, name, value) -> None:
    # Plain attributes stay out of the parameter tree and the weight files.
    object.__setattr__(obj, name, value)


class _DualNorm:
    """RMSNorm that also writes the rotated input of its oQh readers."""

    def __call__(self, x):
        rot, plain = fast.rmsnorm_had(x, self["weight"], self._oqh_signs, self.eps)
        remember(plain, rot, self._oqh_key)
        return plain


class _DualGatedNorm:
    """GDN gated RMSNorm that also hands the rotated input to its oQh out_proj."""

    def __call__(self, x, gate=None):
        out_proj = self._oqh_reader
        if (
            gate is None
            or x.ndim < 2
            or x.shape != gate.shape
            or x.shape[-1] * x.shape[-2] != out_proj.input_dims
        ):
            return super().__call__(x, gate)
        rot, plain = fast.gnorm_had(
            x, gate, self["weight"], out_proj.signs, self.eps, out_proj.block
        )
        hand_off(self._oqh_handoff, rot)
        return plain


class _FusedMLP:
    """down(silu(gate(x)) * up(x)) with gate and up in one paired projection."""

    def __call__(self, x):
        gate, up, down = self.gate_proj, self.up_proj, self.down_proj
        x = gate.rotated_input(x)
        if gate.gemv_rows(x):
            act = fast.act_split_had(gate.pair_rotated(x, up), down.signs, down.block)
        else:
            gu = (gate.matmul_rotated(x), up.matmul_rotated(x))
            act = fast.act_pair_had(*gu, down.signs, down.block)
        return down.matmul_rotated(act)


class _FusedSwitchGLU:
    """SwitchGLU with gate and up in one paired routed projection.

    oqh_rotated_input and oqh_experts also serve MoE blocks that sort and
    combine the routes themselves.
    """

    # GLM-5.3's activation clamps gate and up at this limit.
    _oqh_limit = None

    def oqh_rotated_input(self, x):
        return self.gate_proj.rotated_input(x)

    def oqh_experts(self, x, indices, sorted_indices: bool = False):
        """Expert outputs of rotated rows."""
        gate, up, down = self.gate_proj, self.up_proj, self.down_proj
        limit = self._oqh_limit
        if gate.gemv_rows(x, sorted_indices):
            y = gate.pair_rotated(x, indices, up)
            act = fast.act_split_had(y, down.signs, down.block, limit)
        else:
            gu = [m.matmul_rotated(x, indices, sorted_indices) for m in (gate, up)]
            act = fast.act_pair_had(*gu, down.signs, down.block, limit)
        return down.matmul_rotated(act, indices, sorted_indices)

    def __call__(
        self,
        x,
        indices,
        weights=None,
        shared=None,
        residual=None,
        scores=None,
        weighted_sum=False,
    ):
        """mlx-lm, mlx-vlm and DeepSeek-V4-layout call forms.

        DeepSeek-V4-layout callers (GLM-5.3) pass ``scores`` and sum the
        unsummed [..., top_k, D] result themselves.
        """
        x = mx.expand_dims(self.oqh_rotated_input(x), (-2, -3))
        if self.training:
            indices = mx.stop_gradient(indices)
        # Decode and MTP verify keep the per-route GEMV of one-token decode.
        tokens = indices.size // indices.shape[-1]
        do_sort = indices.size >= 64 and tokens > QMV_MAX_ROWS
        idx, inv_order = indices, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        x = self.oqh_experts(x, idx, do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        x = x.squeeze(-2)
        if weights is None and shared is None and residual is None:
            return x
        # mlx-vlm's SwitchGLU folds the router weights and shared experts in.
        return self._combine(x, weights, shared, residual)


def _same_rotation(a, b) -> bool:
    return a.block == b.block and bool(mx.array_equal(a.signs, b.signs).item())


def _pairs(gate, up) -> bool:
    """Whether gate and up can run as one paired projection."""
    return (
        type(gate) is type(up)
        and gate.bits == up.bits
        and gate.input_dims == up.input_dims
        and gate.output_dims == up.output_dims
        and _same_rotation(gate, up)
        and "bias" not in gate
        and "bias" not in up
    )


def _install_mlp(mlp) -> list:
    """Pair gate and up of an MLP or SwitchGLU; returns the oQh readers of its input."""
    names = ("gate_proj", "up_proj", "down_proj")
    mods = [getattr(mlp, n, None) for n in names]
    readers = [m for m in mods[:2] if isinstance(m, OQH_MODULES)]
    if not all(isinstance(m, OQH_MODULES) for m in mods) or not _pairs(*mods[:2]):
        return readers
    _set(mlp, "_oqh_installed", True)
    switch = isinstance(mods[0], OQH_SWITCH)
    _rebase(mlp, _FusedSwitchGLU if switch else _FusedMLP)
    limit = getattr(getattr(mlp, "activation", None), "limit", None)
    if switch and limit is not None:
        _set(mlp, "_oqh_limit", limit)
    return readers


def _is_oqh_glu(module) -> bool:
    return not getattr(module, "_oqh_installed", False) and all(
        isinstance(getattr(module, n, None), OQH_MODULES)
        for n in ("gate_proj", "up_proj", "down_proj")
    )


def _is_oqh_expert_set(module) -> bool:
    """DeepSeek V4.1 Expert: w1 and w3 read one input, w2 the activation."""
    return not getattr(module, "_oqh_installed", False) and all(
        isinstance(getattr(module, n, None), OQH_MODULES) for n in ("w1", "w3", "w2")
    )


def _verify_switch_glu(original):
    """Run oQh experts in row-exact MTP verify through the decode arithmetic."""

    @functools.wraps(original)
    def switch_glu(self, switch_mlp, x, indices):
        experts = getattr(switch_mlp, "oqh_experts", None)
        if experts is None or x.ndim != 3:
            return original(self, switch_mlp, x, indices)
        batch, length, width = x.shape
        rows = switch_mlp.oqh_rotated_input(x).reshape(batch * length, 1, 1, width)
        out = experts(rows, indices.reshape(batch * length, -1))
        return out.squeeze(-2).reshape(batch, length, indices.shape[-1], -1)

    switch_glu._omlx_oqh = True
    return switch_glu


def _exact_forward(original):
    """Row-exact forward: decode-sized oQh linears keep the one-row arithmetic.

    Lightning MTP runs both decode and verify through it, so verify rows match
    one-token decode.
    """

    @functools.wraps(original)
    def call(self, *args, **kwargs):
        with exact_rows():
            return original(self, *args, **kwargs)

    call._omlx_oqh = True
    return call


def _verify_feed_forward(original):
    """Run paired oQh MLPs in row-exact MTP verify through the decode arithmetic.

    Up to QMV_MAX_ROWS rows, their forward runs one GEMV per row, as in decode.
    """

    @functools.wraps(original)
    def feed_forward(self, module, x):
        if isinstance(module, _FusedMLP):
            return module(x)
        return original(self, module, x)

    feed_forward._omlx_oqh = True
    return feed_forward


def _bind_readers(norm, readers) -> None:
    """Give readers of one input a shared memo key, fused into the norm if possible."""
    groups: list[list] = []
    for m in readers:
        for group in groups:
            if _same_rotation(group[0], m):
                group.append(m)
                break
        else:
            groups.append([m])
    for group in groups:
        key = next(_KEYS)
        for m in group:
            _set(m, "_oqh_key", key)
    first = groups[0][0] if len(groups) == 1 else None
    width = None if first is None else first.input_dims
    if (
        first is None
        or not isinstance(norm, nn.RMSNorm)
        or width % 256
        or width > 8192
        or first.block != 256
    ):
        return
    _set(norm, "_oqh_signs", first.signs.astype(norm["weight"].dtype))
    _set(norm, "_oqh_key", first._oqh_key)
    _rebase(norm, _DualNorm)


def _readers(module, names) -> list:
    return [
        m
        for m in (getattr(module, n, None) for n in names)
        if isinstance(m, OQH_MODULES)
    ]


_MIXER_INPUTS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "q_proj",
    "k_proj",
    "v_proj",
)


def _bind_gated_norm(norm, out_proj) -> None:
    """The GDN gated norm rotates the input of out_proj in the same kernel.

    Its callers reshape the heads before out_proj, so the rotated copy goes
    through the module handoff, not the memo.
    """
    if not isinstance(out_proj, OQH_DENSE) or not isinstance(norm, nn.Module):
        return
    weight = norm.get("weight")
    if weight is None or weight.size != 128 or out_proj.input_dims % 128:
        return
    key = next(_KEYS)
    _set(norm, "_oqh_reader", out_proj)
    _set(norm, "_oqh_handoff", key)
    _set(out_proj, "_oqh_handoff", key)
    _rebase(norm, _DualGatedNorm)


def _install_qwen_layer(layer) -> None:
    mixer = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
    if mixer is not None:
        readers = _readers(mixer, _MIXER_INPUTS)
        if readers:
            _bind_readers(layer.input_layernorm, readers)
        _bind_gated_norm(getattr(mixer, "norm", None), getattr(mixer, "out_proj", None))
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return
    switch = getattr(mlp, "switch_mlp", None)
    if switch is not None:
        readers = _install_mlp(switch)
        shared = getattr(mlp, "shared_expert", None)
        if shared is not None:
            readers += _install_mlp(shared)
    else:
        readers = _install_mlp(mlp)
    if readers:
        _bind_readers(layer.post_attention_layernorm, readers)


def install(model: nn.Module) -> int:
    """Wire producer fusion into every decoder layer that holds oQh linears.

    Layers with Qwen3.5 norms get the norm fusion first; any other MLP or
    SwitchGLU of oQh linears (Qwen4-Exp experts) gets gate | up fusion only.
    """
    count = 0
    for _, module in model.named_modules():
        if getattr(module, "_oqh_installed", False):
            continue
        if not (
            hasattr(module, "input_layernorm")
            and hasattr(module, "post_attention_layernorm")
        ):
            continue
        if not any(isinstance(m, OQH_MODULES) for _, m in module.named_modules()):
            continue
        _install_qwen_layer(module)
        _set(module, "_oqh_installed", True)
        count += 1
    for _, module in model.named_modules():
        if _is_oqh_glu(module) and _install_mlp(module):
            count += 1
        elif _is_oqh_expert_set(module):
            # Its forward weights routes before w2, so only the rotation of
            # the shared input is shared.
            _bind_readers(None, [module.w1, module.w3])
            _set(module, "_oqh_installed", True)
            count += 1
    if count:
        verifier = Qwen3_5BatchInvariantForward
        for name, wrap in (
            ("__call__", _exact_forward),
            ("_switch_glu", _verify_switch_glu),
            ("_feed_forward", _verify_feed_forward),
        ):
            original = getattr(verifier, name)
            if not getattr(original, "_omlx_oqh", False):
                setattr(verifier, name, wrap(original))
    return count


def release_wrapped(model: nn.Module) -> int:
    """Give fused MLPs whose projections are not all oQh linears their own forward.

    A wrapper around an oQh projection, such as a LoRA adapter, has no
    matmul_rotated; the class forward calls each projection, and the norm
    fusion still hands the rotated input to the wrapped oQh linear.
    """
    count = 0
    names = ("gate_proj", "up_proj", "down_proj")
    for _, module in model.named_modules():
        cls = type(module)
        if not issubclass(cls, (_FusedMLP, _FusedSwitchGLU)):
            continue
        if all(isinstance(getattr(module, n, None), OQH_MODULES) for n in names):
            continue
        # _rebase built cls as (mixin, original class).
        module.__class__ = cls.__bases__[1]
        count += 1
    return count
