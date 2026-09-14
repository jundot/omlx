# SPDX-License-Identifier: Apache-2.0
"""Optional, lossy inspection of cached token blocks (never a restore format).

Only CPU data belongs here. Tokenizer snapshots are created once per model;
rendering uses an independent batch decoder, never a streaming detokenizer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any

FORMAT_VERSION = 1
RENDERER_VERSION = 2


def display_text(value: str) -> str:
    """Escape controls and annotation delimiters in untrusted decoded text."""
    # Classify distinct characters only; str.translate handles the long text
    # in C. Avoid a Python callback for every character of every cache block.
    escapes = {
        ord(char): (
            f"\\u{ord(char):04x}" if ord(char) <= 0xFFFF else f"\\U{ord(char):08x}"
        )
        for char in set(value)
        if (unicodedata.category(char).startswith("C") and char not in "\n\t")
        or char in "⟦⟧"
    }
    return value.translate(escapes) if escapes else value


@dataclass(frozen=True)
class BlockInspection:
    token_ids: tuple[int, ...]
    token_start: int
    parent_hash: str | None
    media: tuple[dict[str, Any], ...] = ()

    @property
    def estimated_bytes(self) -> int:
        # Conservative allowance for JSON IDs, rendered text and descriptors.
        return 1024 + 64 * len(self.token_ids) + len(json.dumps(self.media)) * 2


class InspectionRenderer:
    """A CPU-only tokenizer snapshot owned by the cache, not the request."""

    def __init__(self, tokenizer: Any, model_name: str):
        self.model_name = model_name
        self._lock = threading.Lock()
        self._decoder = None
        self._lookup = None
        self._backend = False
        self._special: dict[int, str] = {}
        self.identity = {"class": type(tokenizer).__name__, "fingerprint": None}
        if tokenizer is None:
            return
        # mlx-lm wraps the HF tokenizer. Do not copy its streaming state.
        tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None and hasattr(tokenizer, "to_str"):
            backend = tokenizer
        if backend is not None:
            from tokenizers import Tokenizer

            serialized = backend.to_str()
            self._decoder = Tokenizer.from_str(serialized)
            self._lookup = self._decoder.id_to_token
            self._backend = True
            self.identity["fingerprint"] = (
                "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            )
            added = json.loads(serialized).get("added_tokens", [])
            self._special = {
                token["id"]: token["content"] for token in added if token.get("special")
            }
        else:
            self._decoder = copy.deepcopy(tokenizer)
            self._lookup = getattr(self._decoder, "convert_ids_to_tokens", None)
        self._special.update(
            zip(
                getattr(tokenizer, "all_special_ids", ()),
                getattr(tokenizer, "all_special_tokens", ()),
            )
        )

    def _decode(self, ids: list[int]) -> str:
        if self._decoder is None:
            return "".join(f"⟦undecoded token: id={token}⟧" for token in ids)
        try:
            kwargs = {"skip_special_tokens": False}
            if not self._backend:
                kwargs["clean_up_tokenization_spaces"] = False
            try:
                decoded = self._decoder.decode(ids, **kwargs)
            except TypeError:
                decoded = self._decoder.decode(ids, skip_special_tokens=False)
            if not decoded:
                return f"⟦empty decoding: {len(ids)} tokens; see .tokens⟧"
            return display_text(decoded)
        except Exception:
            return "".join(f"⟦undecodable token: id={token}⟧" for token in ids)

    def _body(self, ids: tuple[int, ...]) -> str:
        parts: list[str] = []
        run: list[int] = []

        def flush() -> None:
            if run:
                parts.append(self._decode(run))
                run.clear()

        i = 0
        while i < len(ids):
            token = ids[i]
            spelling = self._special.get(token)
            if spelling is not None:
                flush()
                end = i + 1
                # Only registered special tokens are candidates. Literal user
                # text resembling an image marker must never be collapsed.
                media = next(
                    (
                        kind
                        for kind in ("image", "audio", "video")
                        if kind in spelling.lower()
                    ),
                    None,
                )
                if media:
                    while end < len(ids) and ids[end] == token:
                        end += 1
                if media and end - i > 1:
                    parts.append(
                        f"⟦{media} marker {display_text(spelling)} × {end - i}; "
                        f"block positions [{i}, {end})⟧"
                    )
                else:
                    parts.append(display_text(spelling))
                i = end
                continue
            if self._lookup is not None:
                try:
                    known = self._lookup(token) is not None
                except Exception:
                    known = False
                if not known:
                    flush()
                    parts.append(f"⟦unknown token: id={token}⟧")
                    i += 1
                    continue
            run.append(token)
            i += 1
        flush()
        return "".join(parts)

    def render(self, block_hash: bytes, block: BlockInspection) -> tuple[bytes, bytes]:
        """Return versioned JSON and annotated UTF-8, without loading media."""
        payload = {
            "format_version": FORMAT_VERSION,
            "renderer_version": RENDERER_VERSION,
            "block_hash": block_hash.hex(),
            "parent_hash": block.parent_hash,
            "model": self.model_name,
            "tokenizer": self.identity,
            "token_start": block.token_start,
            "token_count": len(block.token_ids),
            "token_ids": block.token_ids,
            "media": block.media,
        }
        with self._lock:
            body = self._body(block.token_ids)
        # Model identifiers are labels, not decoded content. Keep the header
        # on one line even if an identifier contains whitespace or controls.
        model_label = (
            display_text(self.model_name).replace("\n", "\\n").replace("\t", "\\t")
        )
        lines = [
            f"oMLX cache inspection — annotated, lossy (renderer v{RENDERER_VERSION})",
            f"Model: {model_label}",
            f"Block: {block_hash.hex()}",
            f"Parent block: {block.parent_hash or 'none (root)'}",
            f"Tokens in this block: {len(block.token_ids)}",
            f"Token range: [{block.token_start}, {block.token_start + len(block.token_ids)})",
            "Token positions are zero-based; start is inclusive, end is exclusive.",
            "⟦…⟧ marks oMLX annotations; literal delimiters and controls are escaped.",
            "Block boundaries may split characters; .tokens contains the exact IDs.",
        ]
        if "\ufffd" in body:
            lines.append(
                "⟦decoding contains replacement characters; possibly an incomplete boundary⟧"
            )
        for descriptor in block.media:
            lines.append(
                "⟦media context: "
                + display_text(json.dumps(descriptor, ensure_ascii=False))
                + "⟧"
            )
        lines.extend(("", "--- decoded content ---", body))
        return (
            (
                json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
            ).encode("utf-8"),
            ("\n".join(lines) + "\n").encode("utf-8"),
        )


def media_context_for_block(
    descriptors: tuple[dict[str, Any], ...], start: int, end: int
) -> tuple[dict[str, Any], ...]:
    """Select known cache-key contexts, not guessed image token spans."""
    selected = []
    for descriptor in descriptors:
        scope_start = descriptor["key_start"]
        scope_end = descriptor.get("key_end")
        if scope_start < end and (scope_end is None or scope_end > start):
            # Do not retain a future message boundary from the request that
            # happened to write this shared block. Describe this block only.
            selected.append(
                {
                    **descriptor,
                    "key_start": max(start, scope_start),
                    "key_end": min(end, scope_end) if scope_end is not None else end,
                }
            )
    return tuple(selected)


def image_context_descriptors(
    sizes: list[tuple[int, int]],
    image_hash: str | None,
    key_ranges: list[tuple[int, str]],
    image_counts: list[int],
) -> tuple[dict[str, Any], ...]:
    """Describe existing image cache keys without hashes or pixel processing.

    These are cumulative image contexts, NOT exact placeholder spans. Later
    images must not leak into metadata for a shared earlier prefix.
    """
    if not sizes or not image_hash:
        return ()
    if not key_ranges or len(key_ranges) != len(image_counts):
        key_ranges = [(0, image_hash)]
        image_counts = [len(sizes)]
    descriptors = []
    consumed = 0
    for index, ((start, fingerprint), count) in enumerate(
        zip(key_ranges, image_counts)
    ):
        consumed += count
        descriptors.append(
            {
                "kind": "image",
                "scope": "cumulative_cache_key_context",
                "key_start": start,
                "key_end": (
                    key_ranges[index + 1][0] if index + 1 < len(key_ranges) else None
                ),
                "fingerprint": "sha256:" + fingerprint,
                "input_dimensions": [list(size) for size in sizes[:consumed]],
                "token_span": None,
            }
        )
    return tuple(descriptors)
