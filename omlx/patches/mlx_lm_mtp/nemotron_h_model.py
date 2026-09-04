"""oMLX Native-MTP patch for nemotron_h (hybrid Mamba2 + attention + MoE).

Adds multi-token-prediction speculative decoding for nemotron_h, mirroring the
qwen3_5 (hybrid GatedDeltaNet) + deepseek_v4 (eh_proj MTP head) patches.

Verified head recipe (mtp_probe.py, 2026-06-06): fuse `[enorm(embed_next) ‖ hnorm(hidden)]`
through eh_proj, where `hidden` is the main model's POST-norm_f hidden; then an attention
block + an MoE block + final_layernorm + shared lm_head. ~52% draft acceptance.

Patches (all idempotent, self-healing via a marker attr):
  - NemotronHMamba2Mixer.__call__: accept n_confirmed; split confirmed/draft, snapshot
    (conv,ssm) -> cache.rollback_state at the boundary.
  - NemotronHBlock.__call__: thread n_confirmed to the Mamba mixer.
  - NemotronHModel.__call__ / Model.__call__: thread n_confirmed, add return_hidden
    (returns post-norm_f hidden for the MTP head).
  - Model: attach .mtp (MTPModule) when active; add mtp_forward / make_mtp_cache.
  - Model.sanitize: keep + remap + stack mtp.* weights when .mtp is present.

This module is import-installed by patches/mlx_lm_mtp/__init__.py and gated by
utils/model_loading.py `_is_mtp_compatible` (model_type.startswith("nemotron_h")).
"""
import contextlib
import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger("omlx.mtp.nemotron_h")
_MARKER = "_omlx_nh_mtp"

# MTP activation: in-package this defers to omlx.patches.mlx_lm_mtp.is_mtp_active;
# standalone (quant/tests) it falls back to a module-level flag.
_MTP_ACTIVE_STANDALONE = False


def set_mtp_active(value: bool) -> None:
    global _MTP_ACTIVE_STANDALONE
    _MTP_ACTIVE_STANDALONE = bool(value)


def _is_active() -> bool:
    try:
        from . import is_mtp_active  # type: ignore
        return is_mtp_active()
    except Exception:
        return _MTP_ACTIVE_STANDALONE


def _is_ours(cls, attr):
    return getattr(cls.__dict__.get(attr, None), _MARKER, False)


def apply() -> bool:
    try:
        from mlx_lm.models import nemotron_h as nh
    except Exception as e:  # pragma: no cover
        logger.debug("nemotron_h not importable; skipping MTP patch: %s", e)
        return False
    # Upstream may eventually ship mtp_forward; defer to it.
    if hasattr(nh.Model, "mtp_forward") and not _is_ours(nh.Model, "mtp_forward"):
        return True
    _patch_args(nh)
    _register_mtp_module(nh)
    _patch_mamba_mixer(nh)
    _patch_block(nh)
    _patch_backbone(nh)
    _patch_model(nh)
    return True


def _patch_args(nh):
    """nemotron_h ModelArgs doesn't declare num_nextn_predict_layers, and
    BaseModelArgs.from_dict drops unknown keys. Surface it as an instance attr
    so the patched Model.__init__ can decide whether to attach the MTP head."""
    A = nh.ModelArgs
    if getattr(A, "_omlx_nh_mtp_args", False):
        return
    orig = A.from_dict.__func__  # unwrap classmethod

    def patched_from_dict(cls, params):
        inst = orig(cls, params)
        inst.num_nextn_predict_layers = int(params.get("num_nextn_predict_layers", 0) or 0)
        return inst

    A.from_dict = classmethod(patched_from_dict)
    A._omlx_nh_mtp_args = True


# --------------------------------------------------------------------------- #
# MTP head module
# --------------------------------------------------------------------------- #
def _register_mtp_module(nh):
    if hasattr(nh, "MTPModule") and getattr(nh.MTPModule, _MARKER, False):
        return
    NemotronHAttention = nh.NemotronHAttention
    NemotronHMoE = nh.NemotronHMoE
    create_attention_mask = nh.create_attention_mask

    class _MTPLayer0(nn.Module):
        """Fusion + attention sublayer. Param paths mirror disk `mtp.layers.0.*`."""

        def __init__(self, args):
            super().__init__()
            H = args.hidden_size
            eps = args.layer_norm_epsilon
            self.eh_proj = nn.Linear(2 * H, H, bias=False)
            self.enorm = nn.RMSNorm(H, eps=eps)
            self.hnorm = nn.RMSNorm(H, eps=eps)
            self.norm = nn.RMSNorm(H, eps=eps)
            self.mixer = NemotronHAttention(args)

    class _MTPLayer1(nn.Module):
        """MoE sublayer + final norm. Param paths mirror disk `mtp.layers.1.*`."""

        def __init__(self, args):
            super().__init__()
            H = args.hidden_size
            eps = args.layer_norm_epsilon
            self.norm = nn.RMSNorm(H, eps=eps)
            self.mixer = NemotronHMoE(args)
            self.final_layernorm = nn.RMSNorm(H, eps=eps)

    class MTPModule(nn.Module):
        """nemotron_h MTP predict-layer (pattern '*E'): fuse -> attn block -> moe block -> norm.
        Structured so parameter paths exactly match the on-disk `mtp.layers.{0,1}.*` names
        (only the routed experts are stacked, in sanitize)."""

        def __init__(self, args):
            super().__init__()
            self.layers = [_MTPLayer0(args), _MTPLayer1(args)]

        def __call__(self, hidden, next_ids, embed_tokens, cache):
            # hidden: (B,1,H) post-norm_f from backbone; next_ids: (B,1)
            l0, l1 = self.layers
            e = l0.enorm(embed_tokens(next_ids))
            h = l0.hnorm(hidden)
            fused = l0.eh_proj(mx.concatenate([e, h], axis=-1))
            kv = cache[0] if cache else None
            mask = create_attention_mask(fused, kv) if fused.shape[1] > 1 else None
            x = fused + l0.mixer(l0.norm(fused), mask=mask, cache=kv)
            x = x + l1.mixer(l1.norm(x))
            return l1.final_layernorm(x)

    MTPModule._omlx_nh_mtp = True
    nh.MTPModule = MTPModule


# --------------------------------------------------------------------------- #
# Mamba2 mixer: n_confirmed split + rollback snapshot
# --------------------------------------------------------------------------- #
def _patch_mamba_mixer(nh):
    Mixer = nh.NemotronHMamba2Mixer
    if _is_ours(Mixer, "__call__"):
        return
    orig = Mixer.__call__  # processes a full chunk, chaining state via cache[0]/cache[1]

    def __call__(self, hidden_states, mask, cache=None, n_confirmed=0):
        S = hidden_states.shape[1]
        if cache is not None and 0 < n_confirmed < S:
            mc = mask[:, :n_confirmed] if mask is not None else None
            md = mask[:, n_confirmed:] if mask is not None else None
            out_c = orig(self, hidden_states[:, :n_confirmed], mc, cache)
            # snapshot confirmed (conv,ssm) state before processing drafts
            cache.rollback_state = (cache[0], cache[1])
            out_d = orig(self, hidden_states[:, n_confirmed:], md, cache)
            return mx.concatenate([out_c, out_d], axis=1)
        return orig(self, hidden_states, mask, cache)

    __call__._omlx_nh_mtp = True
    Mixer.__call__ = __call__


def _patch_block(nh):
    Block = nh.NemotronHBlock
    if _is_ours(Block, "__call__"):
        return

    def __call__(self, x, mask=None, cache=None, n_confirmed=0):
        h = self.norm(x)
        if self.block_type == "M":
            h = self.mixer(h, mask, cache, n_confirmed=n_confirmed)
        elif self.block_type == "*":
            h = self.mixer(h, mask=mask, cache=cache)
        else:
            h = self.mixer(h)
        return x + h

    __call__._omlx_nh_mtp = True
    Block.__call__ = __call__


def _patch_backbone(nh):
    NemotronHModel = nh.NemotronHModel
    if _is_ours(NemotronHModel, "__call__"):
        return
    create_attention_mask = nh.create_attention_mask
    create_ssm_mask = nh.create_ssm_mask

    def __call__(self, inputs, cache=None, n_confirmed=0):
        h = self.embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        attn_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        cc = 0
        for layer in self.layers:
            if layer.block_type in ("M", "*"):
                c = cache[cc]
                cc += 1
            else:
                c = None
            mask = attn_mask if layer.block_type == "*" else ssm_mask
            if layer.block_type == "M":
                h = layer(h, mask=mask, cache=c, n_confirmed=n_confirmed)
            else:
                h = layer(h, mask=mask, cache=c)
        return self.norm_f(h)

    __call__._omlx_nh_mtp = True
    NemotronHModel.__call__ = __call__


# --------------------------------------------------------------------------- #
# Model: __call__(return_hidden, n_confirmed) + mtp_forward + make_mtp_cache + sanitize
# --------------------------------------------------------------------------- #
def _patch_model(nh):
    Model = nh.Model
    KVCache = nh.KVCache

    # --- __init__ wrap: attach .mtp when active ---
    if not getattr(Model, "_omlx_nh_mtp_init", False):
        orig_init = Model.__init__

        def __init__(self, args):
            orig_init(self, args)
            n = int(getattr(args, "num_nextn_predict_layers", 0) or 0)
            active = bool(n > 0 and _is_active())
            # omlx >= 0.5 batch_generator gates the whole MTP decode path on
            # this per-instance marker (see set_mtp_active docstring); without
            # it drafting is silently skipped even with the head attached.
            self._omlx_mtp_decode_enabled = active
            if active:
                self.mtp = nh.MTPModule(args)

        Model.__init__ = __init__
        Model._omlx_nh_mtp_init = True

    # --- __call__: return_hidden + n_confirmed ---
    if not _is_ours(Model, "__call__"):
        def __call__(self, inputs, cache=None, return_hidden=False, n_confirmed=0):
            hidden = self.backbone(inputs, cache=cache, n_confirmed=n_confirmed)  # post-norm_f
            logits = self.lm_head(hidden)
            if return_hidden:
                return logits, hidden
            return logits

        __call__._omlx_nh_mtp = True
        Model.__call__ = __call__

    # --- mtp_forward / make_mtp_cache ---
    def mtp_forward(self, hidden, next_ids, mtp_cache):
        out = self.mtp(hidden, next_ids, self.backbone.embeddings, mtp_cache)
        return self.lm_head(out)

    def make_mtp_cache(self):
        return [KVCache()] if hasattr(self, "mtp") else []

    mtp_forward._omlx_nh_mtp = True
    Model.mtp_forward = mtp_forward
    Model.make_mtp_cache = make_mtp_cache

    # --- sanitize: keep + remap + stack mtp.* when .mtp present ---
    if not _is_ours(Model, "sanitize"):
        orig_sanitize = Model.sanitize

        def sanitize(self, weights):
            has_mtp = hasattr(self, "mtp")
            has_mtp_w = any(k.startswith("mtp.") for k in weights)
            if has_mtp and not has_mtp_w:
                # config declares an MTP head but the checkpoint stripped the weights
                with contextlib.suppress(Exception):
                    del self.mtp
                has_mtp = False

            mtp_w = {k: v for k, v in weights.items() if k.startswith("mtp.")}
            base = {k: v for k, v in weights.items() if not k.startswith("mtp.")}
            out = orig_sanitize(self, base)   # stacks main experts, strips its own mtp (none here)

            if has_mtp and mtp_w:
                E = self.args.n_routed_experts
                ep = "mtp.layers.1.mixer.experts"
                # MTPModule param paths already mirror disk names; only stack routed experts.
                for k, v in mtp_w.items():
                    if f"{ep}." not in k:
                        out[k] = v
                if f"{ep}.0.up_proj.weight" in mtp_w:
                    out["mtp.layers.1.mixer.switch_mlp.fc1.weight"] = mx.stack(
                        [mtp_w[f"{ep}.{e}.up_proj.weight"] for e in range(E)]
                    )
                    out["mtp.layers.1.mixer.switch_mlp.fc2.weight"] = mx.stack(
                        [mtp_w[f"{ep}.{e}.down_proj.weight"] for e in range(E)]
                    )
            return out

        sanitize._omlx_nh_mtp = True
        Model.sanitize = sanitize
