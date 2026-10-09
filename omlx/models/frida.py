# SPDX-License-Identifier: Apache-2.0
"""Adapter for the pinned FRIDA-Decisions 0.4.0 native MLX backend."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .decision import ChunkSize, DecisionContextLengthError, DecisionRequestError


@dataclass
class FridaPlan:
    # All fields are CPU data. Plans survive pool eviction and precision reloads.
    questions: dict
    parsed: Any
    candidates: list
    tokenized: Any
    state_truncated: bool


class FridaModel:
    has_vision = False

    def __init__(
        self, model_path: str, trust_remote_code: bool = False, precision: str = "fp32"
    ):
        if precision not in ("fp32", "bf16"):
            raise ValueError("frida_precision must be fp32 or bf16")
        self.model_path = model_path
        self.precision = precision
        self.judge = None

    def load(self) -> None:
        from frida_decisions.mlx_backend import MlxJudge

        from ..model_discovery import decision_kind

        if decision_kind(Path(self.model_path)) != "frida":
            raise ValueError("Incomplete or unsupported FRIDA checkpoint")

        self.judge = MlxJudge.from_pretrained(
            self.model_path,
            dtype=mx.float32 if self.precision == "fp32" else mx.bfloat16,
            state_max=384,
            rows_per_forward=1,
            state_cache_mb=0,
            compile_encoder=True,
        )

    def close(self) -> None:
        if self.judge is not None:
            if self.judge.state_cache is not None:
                self.judge.state_cache.clear()
            self.judge.state_cache = None
            self.judge._forward_encoder = None
            self.judge._forward_cached_encoder = None
            self.judge.model = None
            self.judge = None

    def encode(self, request: dict, truncate: bool = True) -> FridaPlan:
        from frida_decisions.protocol import RequestError

        if request.get("images"):
            raise DecisionRequestError("FRIDA does not support images")
        raw = {key: request[key] for key in ("state", "questions")}
        judge = self.judge
        try:
            parsed, candidates, tok = judge.compile(raw)
        except RequestError as error:
            raise DecisionRequestError(str(error)) from error
        state_truncated = judge.text.count(candidates[0].state) > len(tok.state)
        if not truncate:
            cfg = judge.config
            if state_truncated:
                raise DecisionContextLengthError("FRIDA state exceeds 384 tokens")
            for candidate in candidates:
                if judge.text.count(candidate.instruction) > cfg.instruction_max_tokens:
                    raise DecisionContextLengthError(
                        "FRIDA instructions including suffix exceed 96 tokens"
                    )
                if judge.text.count(candidate.text) > cfg.option_max_tokens - 1:
                    raise DecisionContextLengthError(
                        "FRIDA option text exceeds 255 tokens plus EOS"
                    )
        return FridaPlan(request["questions"], parsed, candidates, tok, state_truncated)

    def run(self, plan: FridaPlan, chunk_size: ChunkSize):
        """Yield whole encoder rows; bidirectional attention cannot be chunked."""
        from frida_decisions.mlx_modeling import StateCache
        from frida_decisions.packing import (
            build_rows,
            pack_queries,
            pack_rows,
            state_buckets,
        )
        from frida_decisions.protocol import aggregate

        judge = self.judge
        tok, cfg = plan.tokenized, judge.config
        rows, _ = build_rows([tok], cfg)
        margins, tokens = [], 0
        ks = vs = hidden = values = None
        # Never attach a cache to the loaded judge, even during this request.
        cache = StateCache(512 * 2**20) if len(rows) > 1 else None
        try:
            if cache is None:
                for row in rows:
                    batch = pack_rows([row], cfg)
                    margins.extend(judge._forward_packed(batch))
                    cost = sum(batch.lengths)
                    tokens += cost
                    yield cost
            else:
                n = len(tok.state)
                buckets, allowed = state_buckets(n, cfg)
                ids = np.full((1, buckets.shape[0]), cfg.pad_token_id, dtype=np.int32)
                ids[0, :n] = tok.state
                ks, vs = judge.model.encode_state(
                    mx.array(ids), mx.array(buckets[None]), mx.array(allowed[None]), n
                )
                cache.put(tuple(tok.state), ks, vs)
                mx.eval(ks, vs)
                tokens += n
                # State encoding is a complete forward and also yields the GPU.
                yield n
                queries = pack_queries(tok, cfg)
                r = queries.readout
                for index, row in enumerate(queries.rows):
                    selected = r.row == index
                    slots = r.slot[selected]
                    offset = int(slots[0])
                    count = int(slots[-1]) - offset + 1
                    hidden = judge._forward_cached_encoder(
                        mx.array(queries.input_ids[index : index + 1], dtype=mx.int32),
                        mx.array(queries.buckets[index : index + 1]),
                        mx.array(queries.allowed[index : index + 1]),
                        ks,
                        vs,
                    )
                    values = judge.model.margins(
                        hidden,
                        mx.array(r.row[selected] - index, dtype=mx.int32),
                        mx.array(r.col[selected], dtype=mx.int32),
                        mx.array(slots - offset, dtype=mx.int32),
                        count,
                    )
                    mx.eval(values)
                    margins.extend(values.tolist())
                    del hidden, values
                    tokens += len(row.ids)
                    yield len(row.ids)
            return {
                "answers": aggregate(plan.parsed, plan.candidates, margins),
                "input_tokens": tokens,
                "usage": {
                    "state_tokens": len(tok.state),
                    "state_truncated": plan.state_truncated,
                },
            }
        finally:
            if cache is not None:
                cache.clear()
            ks = vs = hidden = values = None
            # Drop even an oversized state rejected by the bounded cache.
