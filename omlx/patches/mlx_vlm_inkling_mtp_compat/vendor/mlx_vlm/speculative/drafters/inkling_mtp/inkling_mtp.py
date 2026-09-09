from dataclasses import replace
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ....models.cache import ArraysCache, CacheList, KVCache
from ....models.inkling.inkling import _split_gate_up
from ....models.inkling.language import (
    InklingDecoderLayer,
    _restore_cache_state,
    _snapshot_cache_state,
    fuse_qkvr,
    shared_experts_to_dense,
)
from .config import InklingMTPConfig

_ATTN = {
    "wq_du": "q_proj",
    "wk_dv": "k_proj",
    "wv_dv": "v_proj",
    "wr_du": "r_proj",
    "wo_ud": "o_proj",
}


def _map_transformer_block(sub: str, v: mx.array) -> dict:
    """Map a checkpoint ``transformer_block.<sub>`` weight onto the MLX
    ``InklingDecoderLayer`` parameter names (mirrors the target model's
    per-layer mapping so the reused decoder layer loads unchanged)."""
    out = {}
    p = "transformer_block."
    if sub.startswith("attn."):
        name, leaf = sub[len("attn.") :].rsplit(".", 1)
        if name in _ATTN:
            out[p + f"self_attn.{_ATTN[name]}.weight"] = v
        elif name in ("q_norm", "k_norm"):
            out[p + f"self_attn.{name}.weight"] = v
        elif name in ("k_sconv", "v_sconv"):
            out[p + f"self_attn.{name}.conv.weight"] = v.transpose(0, 2, 1)
        elif name == "rel_logits_proj":
            out[p + "self_attn.rel_proj"] = v
        else:
            out[p + "self_attn." + name + "." + leaf] = v
    elif sub == "attn_norm.weight":
        out[p + "input_layernorm.weight"] = v
    elif sub == "mlp_norm.weight":
        out[p + "post_attention_layernorm.weight"] = v
    elif sub == "attn_sconv.weight":
        out[p + "attn_sconv.conv.weight"] = v.transpose(0, 2, 1)
    elif sub == "mlp_sconv.weight":
        out[p + "mlp_sconv.conv.weight"] = v.transpose(0, 2, 1)
    elif sub.startswith("mlp."):
        m = sub[len("mlp.") :]
        mp = p + "mlp."
        if m == "w13_dn.weight":
            g, u = _split_gate_up(v)
            out[mp + "gate_proj.weight"] = g
            out[mp + "up_proj.weight"] = u
        elif m == "w2_md.weight":
            out[mp + "down_proj.weight"] = v
        elif m in ("gate.global_scale", "global_scale"):
            out[mp + "global_scale"] = v
        else:
            out[mp + m] = v
    else:
        out[p + sub] = v
    return out


class InklingMTPBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )
        self.transformer_block = InklingDecoderLayer(config, layer_idx)


class InklingMTPDraftModel(nn.Module):
    """DeepSeek-style multi-token-prediction drafter for Inkling.

    Each MTP block combines the previous hidden state and the next-token
    embedding through per-block RMSNorms and a ``2H -> H`` projection, then a
    reused ``InklingDecoderLayer``. The token embedding, final norm scaling and
    LM head come from the target model at ``bind`` time; only the final RMSNorm
    is carried in the drafter checkpoint."""

    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True
    supports_ragged_batch_acceptance = False

    def __init__(self, config: InklingMTPConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("InklingMTPConfig.text_config must be set")

        hidden = text_config.hidden_size
        n = int(config.num_mtp_layers)
        local = set(config.mtp_local_layer_ids or [])
        layer_config = replace(
            text_config,
            num_hidden_layers=n,
            layer_types=[
                "hybrid_sliding" if i in local else "hybrid" for i in range(n)
            ],
            mlp_layer_types=["dense"] * n,
            local_layer_ids=None,
        )
        self.blocks = [InklingMTPBlock(layer_config, i) for i in range(n)]
        self.norm = nn.RMSNorm(hidden, eps=text_config.rms_norm_eps)

        self._mup = float(getattr(text_config, "logits_mup_width_multiplier", 1.0))
        self._uv = getattr(text_config, "unpadded_vocab_size", None)

        self._input_embed = None
        self._lm_head_fn = None
        self._cache: List[CacheList] = []
        self._snapshot = None
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._round_appended = 0
        self._draft_round = 0

        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model) -> "InklingMTPDraftModel":
        inner = None
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        if inner is None:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )

        self._input_embed = inner.embed_tokens
        lm = getattr(target_model, "language_model", target_model)
        self._lm_head_fn = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self._input_embed.as_linear
        )
        return self

    def make_cache(self, left_padding: Optional[List[int]] = None) -> List[CacheList]:
        return [CacheList(KVCache(), ArraysCache(4)) for _ in self.blocks]

    def reset(
        self, target_model, left_padding: Optional[List[int]] = None
    ) -> List[CacheList]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self._draft_round = 0
        self._cache = self.make_cache(left_padding)
        self._snapshot = None
        self._seed_token = None
        self._seed_hidden = None
        self._round_appended = 0
        return self._cache

    def draft_eval_state(self):
        state = [self._seed_token, self._seed_hidden]
        for cache in self._cache:
            for subcache in cache.caches:
                if getattr(subcache, "keys", False) is None:
                    continue
                state.append(subcache.state)
        return state

    def set_shared_kv(
        self,
        shared_kv_states: dict,
        kv_offset,
        position=None,
        kv_valid_len=None,
        left_padding=None,
    ) -> None:
        del shared_kv_states, kv_offset, position, kv_valid_len, left_padding

    def _block_logits(self, hidden: mx.array) -> mx.array:
        logits = self._lm_head_fn(self.norm(hidden) / self._mup)
        if self._uv is not None and self._uv < logits.shape[-1]:
            logits = logits[..., : self._uv]
        return logits

    def _forward_seq(
        self,
        tokens: mx.array,
        hidden: mx.array,
        block_idx: int,
        token_dtype: mx.Dtype,
    ) -> mx.array:
        block = self.blocks[block_idx]
        token_embed = self._input_embed(tokens.astype(token_dtype))
        h = mx.concatenate(
            [block.hidden_norm(hidden), block.embed_norm(token_embed)], axis=-1
        )
        h = block.input_proj(h)
        return block.transformer_block(h, cache=self._cache[block_idx])

    def _forward_token(
        self, tok: mx.array, hidden: mx.array, token_dtype: mx.Dtype
    ) -> mx.array:
        block_idx = min(self._round_appended, len(self.blocks) - 1)
        h = self._forward_seq(tok, hidden, block_idx, token_dtype)
        self._round_appended += 1
        return h

    def _set_seed_from_hidden(self, hidden: mx.array, sampler, greedy: bool) -> None:
        logits = self._block_logits(hidden)
        self._seed_token = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
        self._seed_hidden = hidden

    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus_token,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        if input_ids.shape[1] == 0:
            return
        if isinstance(bonus_token, int):
            bonus = mx.array([[bonus_token]], dtype=token_dtype)
        else:
            bonus = bonus_token[:, None].astype(token_dtype)

        shifted = mx.concatenate([input_ids[:, 1:].astype(token_dtype), bonus], axis=1)
        self._round_appended = 0
        h = self._forward_seq(shifted, hidden[:, : shifted.shape[1], :], 0, token_dtype)
        self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> mx.array:
        del cache
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block() "
                "so the drafter can use the target embeddings and LM head."
            )

        self._snapshot = _snapshot_cache_state(self._cache)
        self._round_appended = 0

        if isinstance(last_bonus, int):
            tok = mx.array([[last_bonus]], dtype=token_dtype)
        else:
            tok = last_bonus[:, None].astype(token_dtype)

        h_prev = hidden
        tokens: List[mx.array] = []
        if self._seed_token is not None and self._seed_hidden is not None:
            tok = self._seed_token.astype(token_dtype)
            h_prev = self._seed_hidden
            tokens.append(tok)
            self._seed_token = None
            self._seed_hidden = None

        while len(tokens) < block_size - 1:
            h_prev = self._forward_token(tok, h_prev, token_dtype)
            logits = self._block_logits(h_prev)
            tok = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
            if tok.ndim == 1:
                tok = tok[:, None]
            tokens.append(tok)

        self._draft_round += 1
        return mx.concatenate(tokens, axis=1)

    def accept_verified_tokens(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: int,
        new_tokens: List[int],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        if self._snapshot is not None:
            _restore_cache_state(self._cache, self._snapshot)
            self._snapshot = None
        self._round_appended = 0

        token_chunks = []
        hidden_chunks = []
        for i in range(int(accepted)):
            token_chunks.append(draft_tokens[:, i : i + 1])
            hidden_chunks.append(verify_hidden[:, i : i + 1, :])
        if new_tokens:
            token_chunks.append(mx.array([[int(new_tokens[-1])]], dtype=token_dtype))
            hidden_chunks.append(verify_hidden[:, int(accepted) : int(accepted) + 1, :])

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            h = self._forward_seq(tokens, hiddens, 0, token_dtype)
            self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)

    def accept_verified_tokens_batch(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: List[int],
        new_tokens: List[List[int]],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        if len(accepted) <= 1:
            self.accept_verified_tokens(
                verify_hidden,
                draft_tokens,
                int(accepted[0]),
                new_tokens[0],
                sampler,
                token_dtype,
                greedy,
            )
            return
        raise NotImplementedError(
            "Inkling MTP batched acceptance is not implemented; run with batch size 1."
        )

    def filter_batch(self, keep) -> None:
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)
        for cache in self._cache:
            cache_filter = getattr(cache, "filter", None)
            if callable(cache_filter):
                cache_filter(keep)
        if self._seed_token is not None:
            self._seed_token = self._seed_token[keep]
        if self._seed_hidden is not None:
            self._seed_hidden = self._seed_hidden[keep]

    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            key = k[len("mtp.") :] if k.startswith("mtp.") else k
            if key.startswith("layers."):
                i, sub = key[len("layers.") :].split(".", 1)
                base = f"blocks.{i}."
                if sub in (
                    "embed_norm.weight",
                    "hidden_norm.weight",
                    "input_proj.weight",
                ):
                    out[base + sub] = v
                elif sub.startswith("transformer_block."):
                    tb = sub[len("transformer_block.") :]
                    for nk, nv in _map_transformer_block(tb, v).items():
                        out[base + nk] = nv
                else:
                    out[base + sub] = v
            else:
                out[key] = v
        return fuse_qkvr(shared_experts_to_dense(out))
