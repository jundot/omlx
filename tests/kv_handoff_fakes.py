# SPDX-License-Identifier: Apache-2.0
"""A stand-in for the MCDMA vLLM connector: real MLX KV laid out as vLLM pages, served over a loopback link."""

from __future__ import annotations

import json
import threading
import zlib

import mlx.core as mx
import numpy as np
from mlx_lm.models import qwen2
from mlx_lm.models.cache import make_prompt_cache
from rdma_loopback import PythonWordOps

from omlx.cluster.rdma.mailbox import ServiceMailbox
from omlx.remote_prefill import wire

BLOCK = 16


def build_model() -> qwen2.Model:
    args = qwen2.ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=3,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=256,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    mx.random.seed(99)
    model = qwen2.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def prefill(model: qwen2.Model, tokens: list[int], caches=None):
    """Caches holding `tokens`, computed locally."""
    caches = caches if caches is not None else make_prompt_cache(model)
    model(mx.array(tokens)[None], cache=caches)
    mx.eval([cache.state for cache in caches])
    return caches


def greedy(model: qwen2.Model, caches, suffix: list[int], steps: int) -> list[int]:
    """Feed `suffix`, then take `steps` greedy tokens."""
    logits = model(mx.array(suffix)[None], cache=caches)
    out = []
    for _ in range(steps):
        token = int(mx.argmax(logits[0, -1]).item())
        out.append(token)
        logits = model(mx.array([[token]]), cache=caches)
    return out


def kv(cache):
    """A cache's filled keys and values."""
    return cache.keys[..., : cache.offset, :], cache.values[..., : cache.offset, :]


def _bits(array) -> np.ndarray:
    return np.array(array.view(mx.uint16))


def export_rank(
    caches, first: int, end: int, *, rank: int = 0, ranks: int = 1
) -> tuple[list[dict], list[tuple[int, int, bytes]]]:
    """This rank's layers as vLLM FlashAttention pages [blocks, heads, block, 2 * head_dim]."""
    layers, pages = [], []
    for index, cache in enumerate(caches):
        keys, values = kv(cache)
        heads, dim = keys.shape[1], keys.shape[3]
        share = heads // ranks
        mine = slice(rank * share, (rank + 1) * share)
        k = _bits(keys[0, mine, first:end])
        v = _bits(values[0, mine, first:end])
        tokens = end - first
        rows = -(-tokens // BLOCK)
        packed = np.zeros((rows, share, BLOCK, 2 * dim), dtype=np.uint16)
        for token in range(tokens):
            block, slot = divmod(token, BLOCK)
            packed[block, :, slot, :dim] = k[:, token]
            packed[block, :, slot, dim:] = v[:, token]
        layers.append(
            {
                "index": index,
                "kind": "attention",
                "shape": [rows, share, BLOCK, 2 * dim],
                "dims": ["block", "head", "token", "kv_head_dim"],
                "dtype": "bfloat16",
                "heads": share,
                "total_heads": heads,
                "head_size": dim,
            }
        )
        pages.append(packed)
    return layers, [(index, packed) for index, packed in enumerate(pages)]


def manifest(
    tokens: list[int], first: int, layers: list[dict], frames: int, *, rank=0, ranks=1
) -> dict:
    return {
        "protocol": 1,
        "handoff": "",
        "model": "tiny",
        "prompt_tokens": len(tokens),
        "first_token": first,
        "token_sha256": wire.token_sha256(tokens),
        "block_size": BLOCK,
        "tp_rank": rank,
        "tp_size": ranks,
        "layers": layers,
        "frames": frames,
    }


def frames_of(pages: list[tuple[int, np.ndarray]], rows_per_frame: int):
    """(layer, row_start, rows, bytes) frames of at most `rows_per_frame` page rows."""
    out = []
    for index, packed in pages:
        for start in range(0, packed.shape[0], rows_per_frame):
            chunk = packed[start : start + rows_per_frame]
            out.append((index, start, chunk.shape[0], chunk.tobytes()))
    return out


class FakeProducer:
    """Answers OPEN, PULL and CLOSE for one handoff from prepared frames, on a thread."""

    def __init__(
        self,
        link,
        manifest_body: dict,
        frames: list[tuple[int, int, int, bytes]],
        *,
        waits: int = 0,
        refuse: str = "",
        flip_frame: int | None = None,
    ) -> None:
        self.service = ServiceMailbox.attach(
            link.name, link.socket_path, PythonWordOps(), mailbox_path=link.mailbox_path
        )
        self.manifest = manifest_body
        self.frames = frames
        self.waits = waits
        self.refuse = refuse
        self.flip_frame = flip_frame
        self.checksum = None
        self.closed = False
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _reply(self, seq, kind, handoff, payload=b"", **fields) -> None:
        header = wire.Header(kind, handoff, nbytes=len(payload), **fields)
        self.service.reply(seq, (wire.pack(header), payload))

    def _serve(self) -> None:
        try:
            while True:
                got = self.service.next_request(timeout_s=20)
                if got is None:
                    return
                seq, payload = got
                header = wire.unpack(payload)
                if self.refuse:
                    self._reply(seq, wire.ERROR, header.handoff, self.refuse.encode())
                    return
                if header.kind == wire.OPEN:
                    self.checksum = json.loads(bytes(wire.body(payload, header)))[
                        "checksum"
                    ]
                    if self.waits:
                        self.waits -= 1
                        self._reply(seq, wire.WAIT, header.handoff)
                        continue
                    body = json.dumps(
                        {**self.manifest, "handoff": header.handoff.hex()}
                    )
                    self._reply(
                        seq,
                        wire.MANIFEST,
                        header.handoff,
                        body.encode(),
                        frames=len(self.frames),
                    )
                elif header.kind == wire.PULL:
                    layer, start, rows, data = self.frames[header.frame]
                    crc = zlib.crc32(data)
                    if header.frame == self.flip_frame:
                        data = bytes([data[0] ^ 1]) + data[1:]
                    self._reply(
                        seq,
                        wire.DATA,
                        header.handoff,
                        data,
                        frame=header.frame,
                        frames=len(self.frames),
                        layer=layer,
                        flags=wire.CHECKED,
                        row_start=start,
                        rows=rows,
                        crc=crc,
                    )
                elif header.kind == wire.CLOSE:
                    self.closed = True
                    self._reply(seq, wire.ACK, header.handoff)
                    return
        except BaseException as exc:
            self.error = exc

    def join(self) -> None:
        self._thread.join(timeout=20)
        self.service.close()
