# SPDX-License-Identifier: Apache-2.0
"""Tiny real-collective MiniMax decode used by the cluster diagnostic."""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any

from .performance import execution_profile
from .runtime_optimizations import install_runtime_optimizations

_CONFIG = {
    "model_type": "minimax_m3_vl",
    "text_config": {
        "num_hidden_layers": 4,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 32,
        "shared_intermediate_size": 32,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "max_position_embeddings": 512,
    },
}


class _RecordingModel:
    """Expose the real adapter while recording which rank requested logits."""

    _omlx_supports_rank_zero_logits = True

    def __init__(self, model: Any) -> None:
        self._model = model
        self.model = model.model
        self.skip_logits_calls: list[bool] = []

    @property
    def _omlx_output_vocab_size(self) -> int:
        return int(self._model._omlx_output_vocab_size)

    def __call__(
        self,
        inputs: Any,
        cache: Any = None,
        skip_logits: bool = False,
    ) -> Any:
        self.skip_logits_calls.append(bool(skip_logits))
        return self._model(
            inputs,
            cache=cache,
            skip_logits=skip_logits,
        )


class _DecodeBatch:
    """The state consumed by the pinned MLX-LM GenerationBatch._step."""

    def __init__(self, mx: Any, model: Any, cache: list[Any]) -> None:
        self.model = model
        self.uids = [1]
        self.prompt_cache = cache
        self.tokens = [[]]
        self.samplers = [None]
        self.fallback_sampler = lambda value: mx.argmax(value, axis=-1)
        self.logits_processors = [[]]
        self._current_tokens = None
        self._current_logprobs: list[Any] = []
        self._next_tokens = mx.array([1], dtype=mx.uint32)
        self._next_logprobs: list[Any] = []
        self._token_context: list[Any] = []


def main() -> int:
    try:
        import mlx.core as mx
        from mlx_lm.utils import _get_classes

        from omlx.patches.minimax_m3_mlx_lm import (
            apply_minimax_m3_mlx_lm_patch,
        )

        group = mx.distributed.init(backend="ring", strict=True)
        if group.size() != 2:
            raise RuntimeError("MiniMax decode smoke requires exactly two ranks")

        apply_minimax_m3_mlx_lm_patch()
        mlx_generate = importlib.import_module("mlx_lm.generate")
        model_cls, args_cls = _get_classes(_CONFIG)
        mx.random.seed(7)
        raw_model = model_cls(args_cls.from_dict(_CONFIG))
        raw_model.model.pipeline(group)
        cache = raw_model.make_cache()
        if not cache:
            raise RuntimeError("MiniMax decode smoke did not construct a cache")

        model = _RecordingModel(raw_model)
        batch = _DecodeBatch(mx, model, cache)
        settings = execution_profile("balanced", sampling_rank_only=True)
        with install_runtime_optimizations(
            model,
            group,
            settings,
            batchable=True,
        ) as capabilities:
            if not capabilities["rank_zero_logits"]["active"]:
                raise RuntimeError(
                    "rank-zero logits capability was not active: "
                    f"{capabilities['rank_zero_logits']}"
                )
            for _ in range(3):
                mlx_generate.GenerationBatch._step(batch)
            mx.eval(batch._next_tokens, batch._next_logprobs)
            next_token = int(batch._next_tokens[0].item())
            tokens = mx.distributed.all_gather(
                mx.array([next_token], dtype=mx.uint32)
            )
            mx.eval(tokens)
            token_values = [int(token) for token in tokens.tolist()]
            if len(set(token_values)) != 1:
                raise RuntimeError(f"pipeline ranks chose different tokens: {token_values}")

            rank = int(group.rank())
            expected_skip = rank != 0
            if model.skip_logits_calls != [expected_skip] * 3:
                raise RuntimeError(
                    f"rank {rank} skip_logits calls were "
                    f"{model.skip_logits_calls}, expected {[expected_skip] * 3}"
                )
            record = {
                "type": "minimax_decode_result",
                "model_type": "minimax_m3_vl",
                "rank": rank,
                "size": int(group.size()),
                "steps": len(model.skip_logits_calls),
                "skip_logits": expected_skip,
                "local_layer_count": int(raw_model.model.num_layers),
                "local_cache_count": len(cache),
                "logprobs_width": int(batch._next_logprobs[0].shape[-1]),
                "next_token": next_token,
            }
            print(json.dumps(record, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            "oMLX MiniMax decode smoke failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
