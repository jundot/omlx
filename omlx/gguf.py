# SPDX-License-Identifier: Apache-2.0
"""GGUF model files and the model folders oMLX derives from them.

oMLX serves a GGUF only through an external engine. Each GGUF model gets a
generated folder under ``<base path>/gguf`` with a config.json derived from
the file's header and links to the file and its vision projectors, so
discovery, settings and the engines handle it like any other model folder.

A GGUF of the qwen35 or flint tokenizer profile also gets a tokenizer.json
derived from its header, as Splash (https://github.com/incoai/splash,
Apache-2.0) derives it in install/gguf.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    normalizers,
    pre_tokenizers,
    processors,
)
from tokenizers import models as token_models

logger = logging.getLogger(__name__)

GGUF_MAGIC = b"GGUF"
SPLIT_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE
)
# Splash selects split files through a VARIANT/ folder and :VARIANT.
SPLIT_VARIANT = "split"
# Vision projector precisions in the order a GGUF is paired with them.
PROJECTOR_PREFERENCE = ("bf16", "f16", "f32")

# Header sanity bounds: well above real files, far below a corrupt length.
_MAX_KEYS = 1 << 20
_MAX_STRING = 64 << 20
_MAX_ARRAY = 1 << 26

_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_STRING = 8
_ARRAY = 9

# Hugging Face model types of the GGUF architectures oMLX names like their
# safetensors checkpoints; any other architecture keeps its GGUF name.
_HF_MODEL_TYPES = {
    "qwen35": "qwen3_5",
    "qwen35moe": "qwen3_5_moe",
    "qwen4exp": "qwen4_exp",
    # Inkling 2's two GGUF layouts (llama.cpp's inkling2_loop, Unsloth's
    # inkling2) state the same Hugging Face text model type.
    "inkling2_loop": "inkling2_loop",
    "inkling2": "inkling2_loop",
}
# Inkling 2's loop, as each GGUF layout keys it: its first layer, the layer
# past its last and its passes. llama.cpp's converter writes inkling2_loop,
# whose layer count is its weight layers; Unsloth's own layout (inkling2, the
# root files of unsloth/Inkling-2-Mini-GGUF) states its layer counts per
# layer application, the loop's core layers once a pass.
_INKLING2_LOOP_KEYS = {
    "inkling2_loop": ("loop.start", "loop.end", "loop.count"),
    "inkling2": ("loop.core_start", "loop.core_end", "num_loops"),
}

# The generated folder's files besides the links to the source files.
CONFIG_FILE = "config.json"
_CHAT_TEMPLATE_FILE = "chat_template.jinja"
_GENERATION_CONFIG_FILE = "generation_config.json"
# Rebuild generated folders with Flint's tokenizer and attention layout.
_FOLDER_LAYOUT = 3

# The qwen35 pre-tokenizer: Qwen2's letter class extended to combining
# marks, then byte-level BPE.
_QWEN35_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# The flint pre-tokenizer, as Flint's own tokenizer.json splits text: digits
# in runs of up to three, runs of Han, hiragana and katakana, then o200k's
# pattern.
_FLINT_PATTERNS = (
    r"\p{N}{1,3}",
    "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
)
# The GGUF tokenizer profiles (tokenizer.ggml.pre) oMLX derives a tokenizer
# for, as Splash does: the patterns that split text, in order, and whether
# text is NFC-normalized first.
# The inkling2 profile, as Inkling 2's own tokenizer.json splits text before
# its byte-level BPE, without a normalizer: o200k's pattern alone, digits in
# runs of up to three.
_INKLING2_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# Each tokenizer profile (tokenizer.ggml.pre) oMLX derives a tokenizer for:
# the patterns that split text, each isolating its matches, in order; whether
# text is NFC-normalized before them; and whether its BPE takes a piece its
# vocabulary holds whole as that token before any merge (ignore_merges), as
# o200k's does and Inkling 2's tokenizer.json states.
_TOKENIZER_PROFILES: dict[str, tuple[tuple[str, ...], bool, bool]] = {
    "qwen35": ((_QWEN35_PATTERN,), True, False),
    "flint": (_FLINT_PATTERNS, False, False),
    "inkling2": ((_INKLING2_PATTERN,), False, True),
}
# GGUF token types: control tokens are special added tokens, user-defined
# ones added tokens that are not special.
_CONTROL_TOKEN = 3
_USER_DEFINED_TOKEN = 4
_TOKENIZER_ARRAYS = frozenset(
    {"tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"}
)
# Per-layer arrays the derived config reads: which layers attend through a
# sliding window (one boolean per layer, so cheap to keep), and Inkling 2's
# head counts, which its layouts state per layer application.
_LAYER_ARRAYS = frozenset(
    {
        "flint.attention.sliding_window_pattern",
        "inkling2_loop.attention.head_count",
        "inkling2_loop.attention.head_count_kv",
        "inkling2.attention.head_count",
        "inkling2.attention.head_count_kv",
    }
)


class GgufError(ValueError):
    """Raised for a file that is not a readable GGUF."""


@dataclass(frozen=True)
class ArrayInfo:
    """An array value read without its elements."""

    length: int


@dataclass(frozen=True)
class GgufModel:
    """One GGUF model in a folder: its parts and the projectors beside it."""

    model_id: str
    parts: tuple[Path, ...]
    projectors: tuple[Path, ...]
    source_folder: Path

    @property
    def path(self) -> Path:
        return self.parts[0]


def file_kind(name: str) -> str:
    """'mmproj', 'imatrix' or 'gguf' for a .gguf file name."""
    base = name.rsplit("/", 1)[-1].lower()
    if "mmproj" in base:
        return "mmproj"
    if "imatrix" in base:
        return "imatrix"
    return "gguf"


def projector_rank(name: str) -> tuple[int, str]:
    """Sort key putting the projector a GGUF pairs with first."""
    base = name.rsplit("/", 1)[-1].lower()
    for rank, tag in enumerate(PROJECTOR_PREFERENCE):
        if re.search(rf"(^|[-_.]){tag}([-_.]|$)", base):
            return rank, name
    return len(PROJECTOR_PREFERENCE), name


def _root_ggufs(folder: Path) -> list[Path]:
    try:
        entries = sorted(folder.iterdir())
    except OSError:
        return []
    return [
        path
        for path in entries
        if path.name.lower().endswith(".gguf")
        and not path.name.startswith(".")
        and path.is_file()
    ]


def is_split_variant(name: str) -> bool:
    """Whether a file is a split GGUF in its VARIANT/ folder."""
    folder, _, base = name.rpartition("/")
    split = SPLIT_RE.match(base)
    return bool(
        folder
        and "/" not in folder
        and not folder.startswith(".")
        and split
        and (
            split["prefix"].lower().endswith(f"-{folder.lower()}")
            or re.fullmatch(r"(?:UD-)?[A-Za-z0-9_]+-PLE\d+", folder, re.IGNORECASE)
        )
        and file_kind(base) == "gguf"
    )


def split_file_groups(names: list[str]) -> dict[str, str]:
    """Map split files to one model, including separately named table shards."""
    groups: dict[str, str] = {}
    folders: dict[str, list[str]] = {}
    for name in names:
        folder, _, base = name.rpartition("/")
        if base.startswith(".") or file_kind(base) != "gguf":
            continue
        split = SPLIT_RE.match(base)
        if split and (not folder or is_split_variant(name)):
            groups[name] = f"{folder}/{split['prefix']}" if folder else split["prefix"]
        if is_split_variant(name):
            folders.setdefault(folder, []).append(name)
    for folder, parts in folders.items():
        splits = [SPLIT_RE.match(name.rsplit("/", 1)[-1]) for name in parts]
        if len(parts) < 2:
            continue
        # A PLE table can have another prefix; its shard number keeps it in order.
        ordered = sorted(zip(parts, splits), key=lambda item: int(item[1]["index"]))
        count = int(ordered[0][1]["count"])
        if {int(split["count"]) for _, split in ordered} != {count} or [
            int(split["index"]) for _, split in ordered
        ] != list(range(1, count + 1)):
            continue
        prefix = ordered[0][1]["prefix"]
        if not prefix.lower().endswith(f"-{folder.lower()}"):
            prefix = f"{prefix}-{folder}"
        groups.update((name, f"{folder}/{prefix}") for name in parts)
    return groups


def model_ids(names: list[str]) -> dict[str, str]:
    """Map the GGUF model files among repo paths to the ID each serves under."""
    groups = split_file_groups(names)
    ids: dict[str, str] = {}
    for name in names:
        if name in groups:
            ids[name] = groups[name].rsplit("/", 1)[-1]
        elif (
            "/" not in name
            and not name.startswith(".")
            and name.lower().endswith(".gguf")
            and file_kind(name) == "gguf"
        ):
            ids[name] = name[: -len(".gguf")]
    return ids


def _variant_ggufs(folder: Path) -> list[Path]:
    try:
        variants = sorted(path for path in folder.iterdir() if path.is_dir())
    except OSError:
        return []
    return [
        path
        for variant in variants
        for path in _root_ggufs(variant)
        if is_split_variant(f"{variant.name}/{path.name}")
    ]


def _model_ggufs(folder: Path) -> list[Path]:
    return _root_ggufs(folder) + _variant_ggufs(folder)


def holds_checkpoint(folder: Path) -> bool:
    """Whether a subfolder of ``folder`` is a model checkpoint."""
    try:
        return any(
            (path / "config.json").is_file()
            for path in folder.iterdir()
            if path.is_dir()
        )
    except OSError:
        return False


def find_models(folder: Path) -> list[GgufModel]:
    """The GGUF models in ``folder``, including split VARIANT/ folders.

    A split GGUF is one model; PLE variants also include their folder name.
    Projectors, imatrix files and MTP heads are not models.
    """
    files = _model_ggufs(folder)
    groups = split_file_groups([path.relative_to(folder).as_posix() for path in files])
    projectors = tuple(
        sorted(
            (path for path in files if file_kind(path.name) == "mmproj"),
            key=lambda path: projector_rank(path.name),
        )
    )
    split_parts: dict[str, list[Path]] = {}
    models: list[GgufModel] = []
    for path in files:
        if file_kind(path.name) != "gguf":
            continue
        split = SPLIT_RE.match(path.name)
        if split:
            group = groups[path.relative_to(folder).as_posix()]
            split_parts.setdefault(group, []).append(path)
            continue
        models.append(
            GgufModel(path.name[: -len(".gguf")], (path,), projectors, folder)
        )
    for group, parts in split_parts.items():
        prefix = group.rsplit("/", 1)[-1]
        parts.sort(key=lambda path: int(SPLIT_RE.match(path.name)["index"]))
        splits = [SPLIT_RE.match(part.name) for part in parts]
        count = int(splits[0]["count"])
        if {int(split["count"]) for split in splits} != {count} or [
            int(split["index"]) for split in splits
        ] != list(range(1, count + 1)):
            logger.warning(
                "Skipping incomplete split GGUF %s in %s: %d of %d parts",
                prefix,
                folder,
                len(parts),
                count,
            )
            continue
        models.append(GgufModel(prefix, tuple(parts), projectors, folder))
    return sorted(models, key=lambda model: model.model_id)


def is_gguf_folder(folder: Path) -> bool:
    """True for a folder of GGUF models rather than a safetensors checkpoint."""
    try:
        if any(folder.glob("*.safetensors")):
            return False
    except OSError:
        return False
    if any(file_kind(path.name) == "gguf" for path in _root_ggufs(folder)):
        return True
    # Split variants do not make an organization folder of checkpoints a repo.
    return bool(_variant_ggufs(folder)) and not holds_checkpoint(folder)


# -- header -------------------------------------------------------------------


class _Reader:
    def __init__(self, stream: BinaryIO):
        self._stream = stream

    def take(self, size: int) -> bytes:
        data = self._stream.read(size)
        if len(data) != size:
            raise GgufError("truncated GGUF header")
        return data

    def scalar(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def length(self, bound: int) -> int:
        value = self.scalar("<Q")
        if value > bound:
            raise GgufError("GGUF header length out of range")
        return value

    def string(self) -> str:
        return self.take(self.length(_MAX_STRING)).decode("utf-8", "replace")

    def skip_strings(self, count: int) -> None:
        # A 250K-entry vocabulary spends its time in per-string method calls.
        read, seek = self._stream.read, self._stream.seek
        unpack = struct.Struct("<Q").unpack
        for _ in range(count):
            data = read(8)
            if len(data) != 8:
                raise GgufError("truncated GGUF header")
            (size,) = unpack(data)
            if size > _MAX_STRING:
                raise GgufError("GGUF header length out of range")
            seek(size, os.SEEK_CUR)

    def value(self, kind: int, keep: bool):
        if kind in _SCALARS:
            return self.scalar(_SCALARS[kind])
        if kind == _STRING:
            return self.string()
        if kind != _ARRAY:
            raise GgufError(f"unknown GGUF value type {kind}")
        item = self.scalar("<I")
        count = self.length(_MAX_ARRAY)
        if keep:
            return [self.value(item, keep=True) for _ in range(count)]
        if item == _STRING:
            self.skip_strings(count)
        elif item in _SCALARS:
            self._stream.seek(struct.calcsize(_SCALARS[item]) * count, os.SEEK_CUR)
        else:
            raise GgufError(f"unsupported GGUF array of type {item}")
        return ArrayInfo(count)


def read_metadata(
    path: Path, keep_arrays: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Key-value metadata of a GGUF header.

    Arrays read as ``ArrayInfo`` unless their key is in ``keep_arrays``,
    which spares reading a 250K-entry vocabulary when only its size matters.
    """
    with open(path, "rb") as stream:
        reader = _Reader(stream)
        if reader.take(4) != GGUF_MAGIC:
            raise GgufError(f"{path.name} is not a GGUF file")
        version = reader.scalar("<I")
        if version < 2:
            raise GgufError(f"{path.name} uses GGUF version {version}")
        reader.scalar("<Q")  # tensor count
        count = reader.length(_MAX_KEYS)
        metadata: dict[str, Any] = {}
        for _ in range(count):
            key = reader.string()
            kind = reader.scalar("<I")
            metadata[key] = reader.value(kind, keep=key in keep_arrays)
    return metadata


def _int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _shared_layer_size(
    metadata: dict[str, Any], arch: str, name: str, applications: int | None
) -> int | None:
    """The size at the header's ``name``, which llama.cpp states once or as
    one value per layer application (Inkling 2's head counts): the one value
    every application shares."""
    values = metadata.get(f"{arch}.{name}")
    if not isinstance(values, list):
        return _int(values)
    if applications is None or len(values) != applications:
        return None
    shared = set(values)
    if len(shared) != 1:
        return None
    value = shared.pop()
    return value if _int(value) is not None else None


def _inkling2_loop(metadata: dict[str, Any], arch: str) -> dict[str, int] | None:
    """Inkling 2's loop, as its release states it: the core layers that run
    once per pass."""
    start_key, end_key, passes_key = _INKLING2_LOOP_KEYS[arch]
    start = _int(metadata.get(f"{arch}.{start_key}"))
    end = _int(metadata.get(f"{arch}.{end_key}"))
    passes = _int(metadata.get(f"{arch}.{passes_key}"))
    if start is None or end is None or passes is None:
        return None
    if not 0 <= start < end or passes < 1:
        return None
    return {"core_start": start, "core_end": end, "num_passes": passes}


def _inkling2_weight_layers(
    arch: str, applications: int, loop: dict[str, int] | None
) -> int:
    """Inkling 2's weight layers: fewer than the header's applications when
    its layout counts the loop's core layers once per pass (inkling2)."""
    if arch != "inkling2" or loop is None:
        return applications
    repeated = (loop["num_passes"] - 1) * (loop["core_end"] - loop["core_start"])
    if repeated <= 0 or repeated >= applications:
        return applications
    return applications - repeated


def derive_config(
    metadata: dict[str, Any], projector: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A config.json for a GGUF model from its header, in the Hugging Face layout."""
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise GgufError("GGUF header names no architecture")

    def field(name: str) -> int | None:
        return _int(metadata.get(f"{arch}.{name}"))

    layers = field("block_count")
    if layers is not None:
        # MTP layers are not part of the model the engines run.
        layers -= field("nextn_predict_layers") or 0
    applications = layers
    loop_config = None
    if layers is not None and arch in _INKLING2_LOOP_KEYS:
        loop_config = _inkling2_loop(metadata, arch)
        if loop_config is not None:
            layers = _inkling2_weight_layers(arch, layers, loop_config)
    tokens = metadata.get("tokenizer.ggml.tokens")
    if isinstance(tokens, ArrayInfo):
        vocab_size = tokens.length
    else:
        vocab_size = len(tokens) if isinstance(tokens, list) else None
    padded = field("vocab_size")
    if (
        arch in _INKLING2_LOOP_KEYS
        and padded is not None
        and vocab_size is not None
        and padded >= vocab_size
    ):
        # Inkling 2 pads its vocabulary to the model's rows; the token table
        # holds the tokens its tokenizer emits.
        vocab_size = padded
    sizes = {
        "hidden_size": field("embedding_length"),
        "num_hidden_layers": layers,
        "max_position_embeddings": field("context_length"),
        "num_attention_heads": _shared_layer_size(
            metadata, arch, "attention.head_count", applications
        ),
        "num_key_value_heads": _shared_layer_size(
            metadata, arch, "attention.head_count_kv", applications
        ),
        "head_dim": field("attention.key_length"),
        "full_attention_interval": field("full_attention_interval"),
        "num_experts": field("expert_count"),
        "num_experts_per_tok": field("expert_used_count"),
        "vocab_size": vocab_size,
    }
    hf_type = _HF_MODEL_TYPES.get(arch)
    text: dict[str, Any] = {
        key: value for key, value in sizes.items() if value is not None
    }
    window = field("attention.sliding_window")
    pattern = metadata.get(f"{arch}.attention.sliding_window_pattern")
    if (
        window
        and isinstance(pattern, list)
        and layers is not None
        and applications is not None
        and len(pattern) >= applications
        and all(isinstance(slides, bool) for slides in pattern)
    ):
        # Only the full layers' KV grows with the context (Flint, Inkling 2).
        text["sliding_window"] = window
        order = list(range(layers))
        if arch == "inkling2" and loop_config is not None and applications > layers:
            # The trailing layers follow the loop's repeated applications.
            repeated = applications - layers
            shift_at = loop_config["core_end"]
            order = [
                layer if layer < shift_at else layer + repeated
                for layer in range(layers)
            ]
        text["layer_types"] = [
            "sliding_attention" if pattern[application] else "full_attention"
            for application in order
        ]
    if loop_config is not None:
        text["loop_config"] = loop_config
    text["model_type"] = f"{hf_type}_text" if hf_type else arch
    config: dict[str, Any] = {"model_type": hf_type or arch, "text_config": text}
    if projector is not None:
        vision = {"model_type": projector.get("general.architecture") or "clip"}
        if isinstance(projector.get("clip.projector_type"), str):
            vision["projector_type"] = projector["clip.projector_type"]
        config["vision_config"] = vision
    return config


def _tokenizer_profile(
    metadata: dict[str, Any],
) -> tuple[tuple[str, ...], bool, bool] | None:
    """The tokenizer profile oMLX derives a tokenizer for, or None."""
    if metadata.get("tokenizer.ggml.model") != "gpt2":
        return None
    return _TOKENIZER_PROFILES.get(str(metadata.get("tokenizer.ggml.pre")))


def tokenizer_files(metadata: dict[str, Any]) -> dict[str, str]:
    """tokenizer.json and tokenizer_config.json of a GGUF whose tokenizer
    profile oMLX derives (``_TOKENIZER_PROFILES``).

    ``metadata`` must hold the tokenizer arrays (``_TOKENIZER_ARRAYS``).
    """
    profile = _tokenizer_profile(metadata)
    if profile is None:
        raise GgufError("unsupported GGUF tokenizer profile")
    patterns, nfc, ignore_merges = profile
    tokens = metadata["tokenizer.ggml.tokens"]
    types = metadata["tokenizer.ggml.token_type"]
    merges = metadata["tokenizer.ggml.merges"]
    if len(types) != len(tokens) or len(set(tokens)) != len(tokens):
        raise GgufError("invalid GGUF tokenizer vocabulary")
    pairs = [tuple(merge.split(" ", 1)) for merge in merges]
    if any(len(pair) != 2 for pair in pairs):
        raise GgufError("invalid GGUF BPE merge")
    backend = Tokenizer(
        token_models.BPE(
            {token: i for i, token in enumerate(tokens)},
            pairs,
            ignore_merges=ignore_merges,
        )
    )
    if nfc:
        backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            *(
                pre_tokenizers.Split(Regex(pattern), behavior="isolated")
                for pattern in patterns
            ),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=False, trim_offsets=False
            ),
        ]
    )
    backend.post_processor = processors.ByteLevel(
        add_prefix_space=False, use_regex=False, trim_offsets=False
    )
    backend.decoder = decoders.ByteLevel()
    # Index of each added token -> whether it is special.
    added = {
        index: kind == _CONTROL_TOKEN
        for index, kind in enumerate(types)
        if kind in (_CONTROL_TOKEN, _USER_DEFINED_TOKEN)
    }
    backend.add_tokens(
        [
            AddedToken(tokens[index], normalized=False, special=special)
            for index, special in added.items()
        ]
    )

    def special_token(name: str) -> str | None:
        index = _int(metadata.get(f"tokenizer.ggml.{name}_token_id"))
        return tokens[index] if index is not None and 0 <= index < len(tokens) else None

    arch = metadata.get("general.architecture")
    config = {
        "tokenizer_class": "TokenizersBackend",
        "model_max_length": _int(metadata.get(f"{arch}.context_length")),
        "clean_up_tokenization_spaces": False,
        "bos_token": special_token("bos"),
        "eos_token": special_token("eos"),
        "pad_token": special_token("padding"),
        "added_tokens_decoder": {
            str(index): {
                "content": tokens[index],
                "special": special,
                "normalized": False,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
            }
            for index, special in added.items()
        },
    }
    return {
        "tokenizer.json": backend.to_str(),
        "tokenizer_config.json": json.dumps(config),
    }


def _generation_config(metadata: dict[str, Any]) -> dict[str, Any]:
    names = {"temp": "temperature", "top_p": "top_p", "top_k": "top_k"}
    config = {}
    for key, name in names.items():
        value = metadata.get(f"general.sampling.{key}")
        if isinstance(value, float):
            config[name] = round(value, 6)
        elif _int(value) is not None:
            config[name] = value
    return config


# -- generated folders ----------------------------------------------------------


def _folder_name(model: GgufModel) -> str:
    """Name that changes whenever the model's files or source folder change."""
    # The config records the source folder, so another path to it rebuilds.
    digest = hashlib.sha1(f"layout {_FOLDER_LAYOUT}\n{model.source_folder}\n".encode())
    for path in (*model.parts, *model.projectors):
        stat = path.stat()
        line = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        digest.update(line.encode())
    return f"{model.model_id}-{digest.hexdigest()[:12]}"


def model_size(model: GgufModel) -> int:
    """Bytes of the parts and the projector an engine loads beside them."""
    size = sum(part.stat().st_size for part in model.parts)
    if model.projectors:
        size += model.projectors[0].stat().st_size
    return size


def prepare_model_folder(model: GgufModel, root: Path) -> Path:
    """The generated folder for ``model``, built on first use."""
    folder = root / _folder_name(model)
    if (folder / CONFIG_FILE).is_file():
        return folder
    metadata = read_metadata(model.path, keep_arrays=_LAYER_ARRAYS)
    tokenizer = {}
    if _tokenizer_profile(metadata) is not None:
        metadata = read_metadata(
            model.path, keep_arrays=_TOKENIZER_ARRAYS | _LAYER_ARRAYS
        )
        tokenizer = tokenizer_files(metadata)
    projector = read_metadata(model.projectors[0]) if model.projectors else None
    config = derive_config(metadata, projector)
    config["omlx_gguf"] = {
        "source": str(model.path),
        "parts": [
            f"{SPLIT_VARIANT}/{part.name}" if len(model.parts) > 1 else part.name
            for part in model.parts
        ],
        "projectors": [path.name for path in model.projectors],
    }
    config["omlx_gguf"]["directory"] = str(model.source_folder)
    config["omlx_gguf"]["source_parts"] = [
        part.relative_to(model.source_folder).as_posix() for part in model.parts
    ]
    partial = root / f".{folder.name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for part, name in zip(model.parts, config["omlx_gguf"]["parts"], strict=True):
        link = partial / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(part.resolve())
    for path in model.projectors:
        (partial / path.name).symlink_to(path.resolve())
    (partial / CONFIG_FILE).write_text(json.dumps(config, indent=2))
    template = metadata.get("tokenizer.chat_template")
    if isinstance(template, str) and template.strip():
        (partial / _CHAT_TEMPLATE_FILE).write_text(template)
    generation = _generation_config(metadata)
    if generation:
        (partial / _GENERATION_CONFIG_FILE).write_text(json.dumps(generation))
    for name, text in tokenizer.items():
        (partial / name).write_text(text)
    try:
        partial.rename(folder)
    except OSError:
        # Another discovery built it first.
        shutil.rmtree(partial, ignore_errors=True)
    return folder


def prune_model_folders(root: Path, keep: set[str]) -> None:
    """Remove generated folders, by name, whose model is gone or changed."""
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.is_dir() and entry.name not in keep:
            shutil.rmtree(entry, ignore_errors=True)


def source_info(model_path: str | Path) -> dict[str, Any] | None:
    """The ``omlx_gguf`` record of a generated folder, or None."""
    try:
        config = json.loads((Path(model_path) / CONFIG_FILE).read_text())
    except (OSError, ValueError):
        return None
    info = config.get("omlx_gguf") if isinstance(config, dict) else None
    return info if isinstance(info, dict) else None
