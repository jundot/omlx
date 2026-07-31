"""DSpark drafter config — loaded from the HF checkpoint's config.json.

Adapted from ARahim3/mlx-dspark (MIT); see THIRD_PARTY_NOTICES.md.

Supports two drafter families with a shared inference path:
  - gemma4  (gemma4_text): k_eq_v attention, v_norm, partial/proportional rope,
            sandwich norms + layer_scalar, gelu-tanh MLP, logit softcap.
  - qwen3   (qwen3):       standard GQA (separate v_proj, no v_norm), default rope,
            Llama-style 2-norm layer, silu MLP, no softcap. Also covers qwen3_5-flavored
            backbones (model_type qwen3_5) via two config-driven
            knobs: gated_q_proj (q_proj emits [q ‖ gate], attn out × sigmoid(gate)) and
            rope_dims (partial rotary).
Only the fields the MLX inference path needs are pulled out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DSparkConfig:
    family: str = "gemma4"  # "gemma4" | "qwen3"

    # core dims
    hidden_size: int = 3840
    vocab_size: int = 262144
    # Some speculators checkpoints predict through a compact draft vocabulary
    # and carry d2t/t2d maps.  ``None`` means the native DeepSpec full vocab.
    draft_vocab_size: int | None = None
    num_hidden_layers: int = 5
    intermediate_size: int = 15360
    rms_norm_eps: float = 1e-6

    # attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    attention_k_eq_v: bool = True
    attention_bias: bool = False

    # rope
    rope_theta: float = 1_000_000.0
    partial_rotary_factor: float = 0.25
    rope_type: str = "proportional"
    # qwen3_5 drafters rope only the first rope_dims of head_dim (partial rotary);
    # None = full head_dim (all other families).
    rope_dims: int | None = None

    # qwen3_5 gated attention: q_proj emits [q ‖ gate] per head (2× out-features),
    # attention output is multiplied by sigmoid(gate) before o_proj.
    gated_q_proj: bool = False

    # qwen3_5 stores every RMSNorm weight as an additive offset from one (Gemma-style
    # (1+w)·x̂ — the reference vLLM patches call this offset_rms_norm). load_drafter adds
    # 1.0 to all RMSNorm weights at load so plain nn.RMSNorm modules compute the right
    # thing. Applying them un-offset multiplies the context fusion by ~0 and silently
    # collapses acceptance to ~1.25 (measured; d0 15% → 90% with the offset).
    offset_rms_norm: bool = False

    # dspark specifics
    block_size: int = 7
    mask_token_id: int = 4
    target_layer_ids: list[int] = field(default_factory=lambda: [5, 17, 29, 41, 46])
    num_target_layers: int = 48

    # markov + confidence
    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True
    speculators_vocab_mapping: bool = False
    max_proposal_tokens: int | None = None

    # Optional GIDD log-SNR conditioning used by some DeepSpec drafters.
    # At inference the per-position pattern is fixed — anchor (block pos 0) at
    # max_log_snr, every masked position at min_log_snr — so the resulting additive
    # embedding is a constant per block position (see model.LogSnrEmbed).
    log_snr_conditioning: bool = False
    min_log_snr: float = -9.0
    max_log_snr: float = 9.0

    # logits
    final_logit_softcapping: float | None = 30.0
    pad_token_id: int = 0

    # ---- family-derived knobs (set in from_json) ----
    mlp_activation: str = "gelu_tanh"  # "gelu_tanh" | "silu"
    norm_style: str = "gemma"  # "gemma" (sandwich+scalar) | "qwen" (llama 2-norm)
    use_v_norm: bool = True  # gemma: RMSNormNoScale v_norm; qwen: none
    attention_scaling: float | None = None  # None -> 1/sqrt(attn_head_dim)

    @property
    def attn_head_dim(self) -> int:
        """Head dim used by the drafter's own attention."""
        return self.global_head_dim if self.family == "gemma4" else self.head_dim

    @property
    def n_kv_heads(self) -> int:
        if self.family == "gemma4" and self.attention_k_eq_v:
            return self.num_global_key_value_heads
        return self.num_key_value_heads

    @property
    def scaling(self) -> float:
        if self.attention_scaling is not None:
            return self.attention_scaling
        return self.attn_head_dim**-0.5 if self.family == "qwen3" else 1.0

    @property
    def rope_parameters(self) -> dict:
        return {
            "rope_type": self.rope_type,
            "partial_rotary_factor": self.partial_rotary_factor,
        }

    @classmethod
    def from_json(cls, path: str | Path) -> DSparkConfig:
        with open(path) as f:
            c = json.load(f)
        mt = c.get("model_type", "")

        # Standard vLLM ``speculators`` DSpark layout.  It uses the same Qwen3
        # backbone as the native drafter, but nests that config and predicts a
        # compact draft vocabulary through explicit d2t/t2d maps.
        if "speculators_config" in c or "speculators_model_type" in c:
            tc = c.get("transformer_layer_config") or {}
            required = (
                "hidden_size",
                "vocab_size",
                "num_hidden_layers",
                "intermediate_size",
                "num_attention_heads",
            )
            missing = [key for key in required if key not in tc]
            taps = c.get("aux_hidden_state_layer_ids") or []
            if missing or not taps or "draft_vocab_size" not in c:
                raise ValueError(
                    f"{path}: incomplete speculators DSpark config "
                    f"(missing transformer fields={missing}, taps={bool(taps)}, "
                    f"draft_vocab_size={'draft_vocab_size' in c})"
                )
            rp = tc.get("rope_parameters") or {}
            head_dim = int(
                tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"])
            )
            partial = float(rp.get("partial_rotary_factor", 1.0))
            proposals = None
            for method in (c.get("speculators_config") or {}).get(
                "proposal_methods", []
            ):
                if (
                    isinstance(method, dict)
                    and method.get("speculative_tokens") is not None
                ):
                    proposals = max(proposals or 0, int(method["speculative_tokens"]))
            return cls(
                family="qwen3",
                hidden_size=int(tc["hidden_size"]),
                vocab_size=int(tc["vocab_size"]),
                draft_vocab_size=int(c["draft_vocab_size"]),
                num_hidden_layers=int(tc["num_hidden_layers"]),
                intermediate_size=int(tc["intermediate_size"]),
                rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
                num_attention_heads=int(tc["num_attention_heads"]),
                num_key_value_heads=int(tc.get("num_key_value_heads", 8)),
                head_dim=head_dim,
                attention_k_eq_v=False,
                attention_bias=bool(tc.get("attention_bias", False)),
                rope_theta=float(
                    rp.get("rope_theta", tc.get("rope_theta", 1_000_000.0))
                ),
                rope_type="default",
                rope_dims=(int(head_dim * partial) if partial < 1.0 else None),
                gated_q_proj=False,
                offset_rms_norm=False,
                block_size=int(c["block_size"]),
                mask_token_id=int(c["mask_token_id"]),
                target_layer_ids=[int(x) for x in taps],
                num_target_layers=int(c.get("num_target_layers") or max(taps) + 1),
                markov_rank=int(c.get("markov_rank") or 0),
                markov_head_type=str(c.get("markov_head_type") or "vanilla"),
                enable_confidence_head=bool(c.get("enable_confidence_head", False)),
                confidence_head_with_markov=bool(
                    c.get("confidence_head_with_markov", True)
                ),
                final_logit_softcapping=None,
                pad_token_id=int(tc.get("pad_token_id") or 0),
                mlp_activation="silu",
                norm_style="qwen",
                use_v_norm=False,
                speculators_vocab_mapping=True,
                max_proposal_tokens=proposals,
            )
        if "block_size" not in c and any(k.startswith("dspark_") for k in c):
            raise ValueError(
                f"{path}: this looks like a full target model with an embedded DSpark drafter "
                f"(dspark_* fields in the target config), not a standalone drafter checkpoint. "
                f"Configure a standalone DeepSpec drafter checkpoint instead."
            )

        if "qwen3" in mt:
            family = "qwen3"
        elif "gemma4" in mt:
            family = "gemma4"
        else:
            raise ValueError(
                f"{path}: unsupported drafter family (model_type={mt!r}). Supported drafter "
                f"backbones: qwen3, gemma4 (gemma4_text)."
            )

        required = (
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "block_size",
            "mask_token_id",
            "target_layer_ids",
        )
        missing = [k for k in required if k not in c]
        if missing:
            raise ValueError(
                f"{path}: config is missing required DeepSpec drafter fields {missing} — this "
                f"does not look like a DeepSpec-format DSpark drafter checkpoint."
            )

        if family == "qwen3":
            rp = c.get("rope_parameters") or {}
            head_dim = c.get("head_dim", c["hidden_size"] // c["num_attention_heads"])
            # qwen3_5-flavored backbones may declare gated q_proj and
            # partial rotary in the same DeepSpec layout; plain qwen3 configs carry neither,
            # so both knobs default to the classic behavior. The config's mrope fields are a
            # text-only no-op (equal position ids collapse mrope to standard rope — mlx-lm's
            # own qwen3_5 text module does the same). Careful: a drafter config is a deepcopy
            # of the TARGET's, so several fields here describe the target and not the drafter
            # (see the rope note below, and `attn_output_gate`/`linear_*`, which the drafter's
            # own weight shapes contradict). Trust weight shapes and provenance, not fields.
            gated_q = bool(c.get("enable_qwen35_gated_q_proj", False))
            # qwen3_5 house style: norm weights stored offset-from-one (see field docs).
            offset_norms = mt == "qwen3_5"
            # `partial_rotary_factor` means partial rotary only on the qwen3_5-NATIVE fork
            # (architectures "Qwen35DSparkModel": Qwen3Next-style gated
            # attention + offset norms — the same code path that ropes a head_dim slice).
            # DeepSpec's stock trainer builds the drafter config as a deepcopy of the
            # TARGET's, so every stock head for a Qwen3.5/3.6 target carries the field as
            # inherited noise while its rope is full head_dim: the reference builds rope
            # with transformers' Qwen3RotaryEmbedding (whose default init keys off head_dim
            # alone) and its apply_rotary_pos_emb multiplies q at full head_dim width — a
            # quarter-width cos would not even broadcast. Honoring the field there ropes a
            # quarter of each head and quietly costs acceptance with no error anywhere
            # (measured on satgeze/Qwen3.5-0.8B-DSpark: accept 1.29 -> 1.59 code,
            # 1.36 -> 1.78 chat, once the rope went back to full width).
            qwen35_native = gated_q or offset_norms
            rope_dims = (
                int(head_dim * float(rp.get("partial_rotary_factor", 1.0)))
                if qwen35_native
                else head_dim
            )
            if c.get("log_snr_conditioning"):
                lo, hi = c.get("min_log_snr"), c.get("max_log_snr")
                if lo is None or hi is None or not (float(hi) > float(lo)):
                    raise ValueError(
                        f"{path}: log_snr_conditioning is enabled but min/max_log_snr are "
                        f"missing or not ordered (min={lo!r}, max={hi!r}) — the featurization "
                        f"divides by (max - min), so a drafter converted without them would "
                        f"draft from silently-wrong embeddings."
                    )
            return cls(
                family="qwen3",
                hidden_size=c["hidden_size"],
                vocab_size=c["vocab_size"],
                num_hidden_layers=c["num_hidden_layers"],
                intermediate_size=c["intermediate_size"],
                rms_norm_eps=c.get("rms_norm_eps", 1e-6),
                num_attention_heads=c["num_attention_heads"],
                num_key_value_heads=c.get("num_key_value_heads", 8),
                head_dim=head_dim,
                attention_k_eq_v=False,
                attention_bias=c.get("attention_bias", False),
                rope_theta=rp.get("rope_theta", c.get("rope_theta", 1_000_000.0)),
                rope_type="default",
                rope_dims=(rope_dims if rope_dims != head_dim else None),
                gated_q_proj=gated_q,
                offset_rms_norm=offset_norms,
                block_size=c["block_size"],
                mask_token_id=c["mask_token_id"],
                target_layer_ids=list(c["target_layer_ids"]),
                num_target_layers=c.get("num_target_layers", 36),
                markov_rank=c.get("markov_rank", 256),
                markov_head_type=c.get("markov_head_type", "vanilla"),
                enable_confidence_head=c.get("enable_confidence_head", True),
                confidence_head_with_markov=c.get("confidence_head_with_markov", True),
                final_logit_softcapping=c.get("final_logit_softcapping", None),
                pad_token_id=c.get("pad_token_id") or 0,
                mlp_activation="silu",
                norm_style="qwen",
                use_v_norm=False,
                log_snr_conditioning=bool(c.get("log_snr_conditioning", False)),
                min_log_snr=float(c.get("min_log_snr", -9.0)),
                max_log_snr=float(c.get("max_log_snr", 9.0)),
            )

        rope = (c.get("rope_parameters") or {}).get("full_attention", {}) or {}
        return cls(
            family="gemma4",
            hidden_size=c["hidden_size"],
            vocab_size=c["vocab_size"],
            num_hidden_layers=c["num_hidden_layers"],
            intermediate_size=c["intermediate_size"],
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c.get("num_key_value_heads", 8),
            num_global_key_value_heads=c.get("num_global_key_value_heads", 1),
            head_dim=c.get("head_dim", 256),
            global_head_dim=c.get("global_head_dim", 512),
            attention_k_eq_v=c.get("attention_k_eq_v", True),
            attention_bias=c.get("attention_bias", False),
            rope_theta=rope.get("rope_theta", 1_000_000.0),
            partial_rotary_factor=rope.get("partial_rotary_factor", 0.25),
            rope_type=rope.get("rope_type", "proportional"),
            block_size=c["block_size"],
            mask_token_id=c["mask_token_id"],
            target_layer_ids=list(c["target_layer_ids"]),
            num_target_layers=c.get("num_target_layers", 48),
            markov_rank=c.get("markov_rank", 256),
            markov_head_type=c.get("markov_head_type", "vanilla"),
            enable_confidence_head=c.get("enable_confidence_head", True),
            confidence_head_with_markov=c.get("confidence_head_with_markov", True),
            final_logit_softcapping=c.get("final_logit_softcapping", 30.0),
            pad_token_id=c.get("pad_token_id", 0),
            mlp_activation="gelu_tanh",
            norm_style="gemma",
            use_v_norm=True,
        )
