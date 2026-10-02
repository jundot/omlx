from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .audio import AudioModel
from .config import ModelConfig
from .language import LanguageModel, fuse_qkvr, shared_experts_to_dense
from .vision import VisionModel


def masked_scatter(input_tensor, mask, source):
    """MLX port of torch.Tensor.masked_scatter: fill mask==True slots with source in order."""
    mask = mask.astype(mx.bool_)
    if not mask.any():
        return input_tensor
    shape = input_tensor.shape
    flat = input_tensor.flatten()
    mask_flat = mask.flatten()
    source_flat = source.flatten()
    positions = mx.cumsum(mask_flat.astype(mx.int32)) - 1
    positions = mx.clip(positions, 0, source_flat.size - 1)
    selected = source_flat[positions]
    return mx.where(mask_flat, selected, flat).reshape(shape)


def _split_gate_up(v):
    """De-interleave a fused [..., 2*inter, hidden] gate/up weight into (gate, up).

    Return contiguous halves for performance; `gather_mm / `gather_qmm` are much slower when fed strided views
    """
    *lead, two_i, hidden = v.shape
    w = v.reshape(*lead, two_i // 2, 2, hidden)
    return mx.contiguous(w[..., 0, :]), mx.contiguous(w[..., 1, :])


def _to_mlx_sconv_weight(v):
    """Convert PyTorch Conv1d weights while keeping MLX weights idempotent."""
    if getattr(v, "ndim", 0) == 3 and v.shape[1] == 1 and v.shape[2] != 1:
        return v.transpose(0, 2, 1)
    return v


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        config.vision_config.text_hidden_size = config.text_config.hidden_size
        config.audio_config.text_hidden_size = config.text_config.hidden_size
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionModel(config.vision_config)
        self.audio_tower = AudioModel(config.audio_config)

    def get_image_features(self, pixel_values):
        return self.vision_tower(pixel_values)

    def get_audio_features(self, audio_input_ids, audio_input_ids_mask=None):
        if audio_input_ids_mask is not None:
            flat = audio_input_ids.reshape(-1, audio_input_ids.shape[-1])
            keep = np.nonzero(np.array(audio_input_ids_mask).reshape(-1))[0]
            frames = flat[mx.array(keep)]
        else:
            frames = audio_input_ids.reshape(-1, audio_input_ids.shape[-1])
        return self.audio_tower(frames)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        h = self.language_model.model.embed(input_ids)

        if pixel_values is not None:
            feats = self.get_image_features(pixel_values).astype(h.dtype)
            mask = mx.broadcast_to(
                (input_ids == self.config.image_token_id)[..., None], h.shape
            )
            h = masked_scatter(h, mask, feats)

        audio_input_ids = kwargs.get("audio_input_ids", None)
        if audio_input_ids is not None:
            feats = self.get_audio_features(
                audio_input_ids, kwargs.get("audio_input_ids_mask", None)
            ).astype(h.dtype)
            mask = mx.broadcast_to(
                (input_ids == self.config.audio_token_id)[..., None], h.shape
            )
            h = masked_scatter(h, mask, feats)
        return InputEmbeddingsFeatures(inputs_embeds=h)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
        **kwargs,
    ):
        spec_kwargs = {
            k: kwargs.pop(k)
            for k in ("return_hidden", "return_shared_kv", "skip_logits")
            if k in kwargs
        }
        embeds = self.get_input_embeddings(
            input_ids, pixel_values=pixel_values, **kwargs
        )
        return self.language_model(
            inputs_embeds=embeds.inputs_embeds, cache=cache, **spec_kwargs
        )

    _ATTN = {
        "wq_du": "q_proj",
        "wk_dv": "k_proj",
        "wv_dv": "v_proj",
        "wr_du": "r_proj",
        "wo_ud": "o_proj",
    }

    def _map_llm_layer(self, base, sub, v):
        out = {}
        if sub.startswith("attn."):
            name, leaf = sub[len("attn.") :].rsplit(".", 1)
            if name in self._ATTN:
                out[base + f"self_attn.{self._ATTN[name]}.{leaf}"] = v
            elif name in ("q_norm", "k_norm"):
                out[base + f"self_attn.{name}.{leaf}"] = v
            elif name in ("k_sconv", "v_sconv"):
                if leaf == "weight":
                    v = _to_mlx_sconv_weight(v)
                out[base + f"self_attn.{name}.conv.{leaf}"] = v
            elif name == "rel_logits_proj":
                out[base + "self_attn.rel_proj"] = v
            else:
                out[base + "self_attn." + name + "." + leaf] = v
        elif sub == "attn_norm.weight":
            out[base + "input_layernorm.weight"] = v
        elif sub == "mlp_norm.weight":
            out[base + "post_attention_layernorm.weight"] = v
        elif sub == "attn_sconv.weight":
            out[base + "attn_sconv.conv.weight"] = _to_mlx_sconv_weight(v)
        elif sub == "mlp_sconv.weight":
            out[base + "mlp_sconv.conv.weight"] = _to_mlx_sconv_weight(v)
        elif sub.startswith("mlp."):
            m = sub[len("mlp.") :]
            p = base + "mlp."
            if m == "gate.weight":
                out[p + "gate_weight"] = v
            elif m == "gate.bias":
                out[p + "e_score_correction_bias"] = v
            elif m in ("gate.global_scale", "global_scale"):
                out[p + "global_scale"] = v
            elif m == "shared_experts.shared_w13_weight":
                g, u = _split_gate_up(v)
                out[p + "shared_experts.gate_proj.weight"] = g
                out[p + "shared_experts.up_proj.weight"] = u
            elif m == "shared_experts.shared_w2_weight":
                out[p + "shared_experts.down_proj.weight"] = v
            elif m == "w13_dn.weight":
                g, u = _split_gate_up(v)
                out[p + "gate_proj.weight"] = g
                out[p + "up_proj.weight"] = u
            elif m == "w2_md.weight":
                out[p + "down_proj.weight"] = v
            elif m.startswith("experts."):
                out[p + "switch_mlp." + m[len("experts.") :]] = v
            else:
                out[p + m] = v
        else:
            out[base + sub] = v
        return out

    def _map_experts(self, i, buf):
        """Emit MLX-native switch-MLP weights for one MoE layer.

        ``buf`` maps ``"w13"``/``"w2"`` to a dict of the raw checkpoint tensors
        (``weight`` plus, when NVFP4-quantized, ``scale``/``scale2``). NVFP4
        experts are stored as ModelOpt two-level nvfp4: uint8-packed E2M1 codes
        (bit-identical to MLX's uint32 layout via ``view``), per-block E4M3
        ``scale`` (already MLX's ``.scales`` byte layout), and a per-expert
        ``scale2``. We pass the codes and block scales through untouched and
        route ``scale2`` to ``InklingSwitchGLU`` (``gate_scale``/``out_scale``).
        Layer ``dense_mlp_idx`` (the first MoE layer) is excluded from
        quantization and arrives as bf16, handled by the plain-split branch.
        """
        out = {}
        p = f"language_model.model.layers.{i}.mlp.switch_mlp."
        w13, w2 = buf["w13"], buf["w2"]
        quantized = w13["weight"].dtype == mx.uint8
        if quantized:
            gw, uw = _split_gate_up(w13["weight"].view(mx.uint32))
            gs, us = _split_gate_up(w13["scale"])
            out[p + "gate_proj.weight"] = gw
            out[p + "gate_proj.scales"] = gs
            out[p + "up_proj.weight"] = uw
            out[p + "up_proj.scales"] = us
            out[p + "down_proj.weight"] = w2["weight"].view(mx.uint32)
            out[p + "down_proj.scales"] = w2["scale"]
            s13 = w13["scale2"].astype(mx.float32)
            s2 = w2["scale2"].astype(mx.float32)
            out[p + "gate_scale"] = s13
            out[p + "out_scale"] = s13 * s2
        else:
            gw, uw = _split_gate_up(w13["weight"])
            out[p + "gate_proj.weight"] = gw
            out[p + "up_proj.weight"] = uw
            out[p + "down_proj.weight"] = w2["weight"]
            n = gw.shape[0]
            out[p + "gate_scale"] = mx.ones((n,))
            out[p + "out_scale"] = mx.ones((n,))
        return out

    def sanitize(self, weights):
        out = {}
        experts = {}  # layer index -> {"w13"|"w2": {"weight"|"scale"|"scale2": array}}
        split_experts = set()
        for k, v in weights.items():
            if ".mtp" in k or k.startswith("model.mtp") or k.endswith("training_args"):
                # OMLX: single-checkpoint Lightning MTP keeps and remaps
                # the head weights when the runtime patch installs a hook
                # (upstream always drops them). None → upstream behavior.
                hook = globals().get("_OMLX_MTP_SANITIZE_HOOK")
                if hook is not None and not k.endswith("training_args"):
                    out.update(hook(k, v))
                continue
            # Routed-expert tensors (base weight + NVFP4 sidecars) are buffered and
            # folded per layer once both w13 and w2 are seen; input_amax /
            # original_shape are calibration metadata with no inference use.
            if ".mlp.experts.w13_weight" in k or ".mlp.experts.w2_weight" in k:
                head, tail = k.split(".mlp.experts.")
                i = head.split("layers.")[1]
                which = "w13" if tail.startswith("w13") else "w2"
                leaf = tail.split("_weight", 1)[1].lstrip(".") or "weight"
                if leaf in ("input_amax", "original_shape"):
                    continue
                experts.setdefault(i, {}).setdefault(which, {})[leaf] = v
                continue
            if k.startswith("model.llm.embed."):
                leaf = k[len("model.llm.embed.") :]
                out["language_model.model.embed_tokens." + leaf] = v
            elif k.startswith("model.llm.unembed."):
                leaf = k[len("model.llm.unembed.") :]
                out["language_model.lm_head." + leaf] = v
            elif k in ("model.llm.embed_norm.weight", "model.llm.norm.weight"):
                out["language_model.model." + k[len("model.llm.") :]] = v
            elif k.startswith("model.llm.layers."):
                i, sub = k[len("model.llm.layers.") :].split(".", 1)
                if sub.startswith("mlp.experts."):
                    split_experts.add(i)
                out.update(
                    self._map_llm_layer(f"language_model.model.layers.{i}.", sub, v)
                )
            elif k.startswith("model.visual."):
                sub = k[len("model.visual.") :]
                if sub.startswith("layers.linear_"):
                    j, leaf = sub[len("layers.linear_") :].split(".", 1)
                    out[f"vision_tower.encoder_layers.{j}.projection.{leaf}"] = v
                elif sub.startswith("layers.norm_"):
                    j, leaf = sub[len("layers.norm_") :].split(".", 1)
                    out[f"vision_tower.encoder_layers.{j}.layer_norm.{leaf}"] = v
                else:
                    out["vision_tower." + sub] = v
            elif k.startswith("model.audio."):
                sub = k[len("model.audio.") :]
                if sub.startswith("encoder."):
                    leaf = sub[len("encoder.") :]
                    out["audio_tower.embed_audio_tokens." + leaf] = v
                elif sub == "final_norm.weight":
                    out["audio_tower.norm.weight"] = v
                else:
                    out["audio_tower." + sub] = v
            else:
                out[k] = v
        for i, buf in experts.items():
            out.update(self._map_experts(i, buf))
        for i in split_experts:
            p = f"language_model.model.layers.{i}.mlp.switch_mlp."
            gate = out.get(p + "gate_proj.weight")
            if gate is None:
                continue
            n = gate.shape[0]
            out.setdefault(p + "gate_scale", mx.ones((n,)))
            out.setdefault(p + "out_scale", mx.ones((n,)))
        model_config = getattr(self, "config", None)
        return fuse_qkvr(
            shared_experts_to_dense(out),
            getattr(model_config, "text_config", None),
        )

    def load_weights(self, weights, strict=True):
        """Apply load-time fusions to already-mapped MLX weights.

        mlx-vlm skips ``sanitize()`` for safetensors already marked as MLX.
        Transforming the fully aggregated weight list here also handles legacy
        oQ shards where Q/K/V/R tensors for one layer live in different files.
        """
        if isinstance(weights, str):
            return super().load_weights(weights, strict=strict)
        transformed = fuse_qkvr(
            shared_experts_to_dense(dict(weights)), self.config.text_config
        )
        return super().load_weights(list(transformed.items()), strict=strict)

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def layers(self):
        return self.language_model.model.layers
