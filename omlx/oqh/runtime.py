"""oQh runtime: producer fusion without changing any caller.

Linears that read one producer output share signs and a block. The RMSNorm in
front of them computes the plain and the rotated output in one kernel, returns
the plain one (routers and every other reader see the usual value) and leaves
the rotated one in the oQh memo keyed by the returned array, where the linears
pick it up. MLP and SwitchGLU forwards are replaced on the instance only: gate
and up run as one fused projection, and the activation writes the rotated
input of down. Other models of the same class in the process are untouched.
"""

from __future__ import annotations

import functools
import itertools
import weakref

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

from ..custom_kernels.oqh import fast
from .modules import (
    OQH_MODULES,
    QMV_MAX_ROWS,
    OqhLinear,
    OqhSwitchLinear,
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
        rot, plain = fast.rmsnorm_had(
            x, self["weight"], self._oqh_signs, self.eps, self._oqh_block, dual=True
        )
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
            x,
            gate,
            self["weight"],
            out_proj.signs,
            self.eps,
            out_proj.block,
            dual=True,
        )
        hand_off(self._oqh_handoff, rot)
        return plain


class _FusedMLP:
    """down(silu(gate(x)) * up(x)) with gate | up as one projection."""

    def __call__(self, x):
        gu = self._oqh_gate_up
        y = gu.matmul_rotated(gu.rotated_input(x))
        down = self.down_proj
        return down.matmul_rotated(fast.act_split_had(y, down.signs, down.block))


class _FusedSwitchGLU:
    """SwitchGLU with gate | up as one routed projection.

    oqh_rotated_input and oqh_experts also serve MoE blocks that sort and
    combine the routes themselves.
    """

    @property
    def oqh_gate_up(self):
        """gate | up as one routed projection, [E, 2N, ...] (gate rows first)."""
        return self._oqh_gate_up

    def oqh_rotated_input(self, x):
        return self._oqh_gate_up.rotated_input(x)

    def oqh_experts(self, x, indices, sorted_indices: bool = False):
        """Expert outputs of rotated rows."""
        y = self._oqh_gate_up.matmul_rotated(x, indices, sorted_indices)
        down = self.down_proj
        # GLM-5.3's activation clamps gate and up at its limit.
        limit = getattr(self.activation, "limit", None)
        act = fast.act_split_had(y, down.signs, down.block, limit)
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


def _fuse_gate_up(mlp) -> nn.Module | None:
    """Fuse gate and up rows; the originals become views of the fused buffers."""
    gate, up = mlp.gate_proj, mlp.up_proj
    if (
        type(gate) is not type(up)
        or gate.bits != up.bits
        or gate.input_dims != up.input_dims
        or not _same_rotation(gate, up)
        or "bias" in gate
        or "bias" in up
    ):
        return None
    n = gate.output_dims
    if isinstance(gate, OqhLinear):
        fused = OqhLinear(gate.input_dims, 2 * n, gate.bits, gate.block)
        axis = 0
    else:
        fused = OqhSwitchLinear(
            gate.input_dims, 2 * n, gate.num_experts, gate.bits, gate.block
        )
        axis = 1
    fused.weight = mx.concatenate([gate["weight"], up["weight"]], axis=axis)
    fused.rscale = mx.concatenate([gate["rscale"], up["rscale"]], axis=axis)
    fused.oqh_signs = gate["oqh_signs"]
    mx.eval(fused.parameters())
    lo, hi = slice(0, n), slice(n, 2 * n)
    for mod, part in ((gate, lo), (up, hi)):
        index = (part,) if axis == 0 else (slice(None), part)
        mod.weight = fused["weight"][index]
        mod.rscale = fused["rscale"][index]
    mx.eval(gate.parameters(), up.parameters())
    on_split = _refuse_after_split(weakref.ref(mlp))
    for mod in (gate, up):
        _set(mod, "_oqh_on_split", on_split)
    return fused


def _refuse_after_split(ref):
    """Fuse gate and up again once a tensor-parallel split has cut both."""

    def refuse():
        mlp = ref()
        if mlp is None:
            return
        gate, up = mlp.gate_proj, mlp.up_proj
        if gate.output_dims == up.output_dims != mlp._oqh_gate_up.output_dims // 2:
            _set(mlp, "_oqh_gate_up", _fuse_gate_up(mlp))

    return refuse


def _install_mlp(mlp) -> list:
    """Fuse one MLP or SwitchGLU; returns the oQh modules that read its input."""
    names = ("gate_proj", "up_proj", "down_proj")
    mods = [getattr(mlp, n, None) for n in names]
    if not all(isinstance(m, OQH_MODULES) for m in mods):
        return [m for m in mods[:2] if isinstance(m, OQH_MODULES)]
    fused = _fuse_gate_up(mlp)
    if fused is None:
        return mods[:2]
    _set(mlp, "_oqh_gate_up", fused)
    _set(mlp, "_oqh_installed", True)
    _rebase(mlp, _FusedSwitchGLU if isinstance(fused, OqhSwitchLinear) else _FusedMLP)
    return [fused]


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
        or first.block < 256
    ):
        return
    _set(norm, "_oqh_signs", first.signs.astype(norm["weight"].dtype))
    _set(norm, "_oqh_block", first.block)
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
    if not isinstance(out_proj, OqhLinear) or not isinstance(norm, nn.Module):
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
    original = Qwen3_5BatchInvariantForward._switch_glu
    if count and not getattr(original, "_omlx_oqh", False):
        Qwen3_5BatchInvariantForward._switch_glu = _verify_switch_glu(original)
    return count
