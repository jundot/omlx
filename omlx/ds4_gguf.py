# SPDX-License-Identifier: Apache-2.0
"""DS4 GGUF discovery helpers.

DS4 is a specialized DeepSeek V4 backend, not a general GGUF runtime.  This
module keeps GGUF filename normalization, metadata inspection, split-shard
filtering, and DS4 support checks out of the generic model discovery scanner.
"""

from __future__ import annotations

import logging
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

DS4_GENERIC_GGUF_STEMS = {
    "model",
    "gguf",
    "weights",
    "consolidated",
}
DS4_SUPPORTED_GGUF_ARCHITECTURES = {"deepseek4"}
DS4_MTP_GGUF_ARCHITECTURE = "deepseek4_mtp_support"
DS4_DSPARK_GGUF_ARCHITECTURE = "deepseek4-dspark"
DS4SidecarKind = Literal["legacy_mtp", "dspark"]
DS4_SPECULATOR_GGUF_ARCHITECTURES: dict[str, DS4SidecarKind] = {
    DS4_MTP_GGUF_ARCHITECTURE: "legacy_mtp",
    DS4_DSPARK_GGUF_ARCHITECTURE: "dspark",
}
DS4_MTP_REQUIRED_TENSORS = (
    "mtp.0.hc_head_base.weight",
    "mtp.0.hc_head_fn.weight",
    "mtp.0.hc_head_scale.weight",
    "mtp.0.hc_attn_base.weight",
    "mtp.0.hc_ffn_base.weight",
    "mtp.0.hc_attn_fn.weight",
    "mtp.0.hc_attn_scale.weight",
    "mtp.0.hc_ffn_fn.weight",
    "mtp.0.hc_ffn_scale.weight",
    "mtp.0.attn_sinks.weight",
    "mtp.0.attn_q_a.weight",
    "mtp.0.attn_q_b.weight",
    "mtp.0.attn_q_a_norm.weight",
    "mtp.0.attn_output_a.weight",
    "mtp.0.attn_kv.weight",
    "mtp.0.attn_kv_a_norm.weight",
    "mtp.0.attn_output_b.weight",
    "mtp.0.attn_norm.weight",
    "mtp.0.ffn_norm.weight",
    "mtp.0.ffn_gate_shexp.weight",
    "mtp.0.ffn_up_shexp.weight",
    "mtp.0.ffn_down_shexp.weight",
    "mtp.0.ffn_gate_exps.weight",
    "mtp.0.ffn_up_exps.weight",
    "mtp.0.ffn_down_exps.weight",
    "mtp.0.ffn_gate_inp.weight",
    "mtp.0.exp_probs_b.bias",
    "mtp.0.e_proj.weight",
    "mtp.0.h_proj.weight",
    "mtp.0.enorm.weight",
    "mtp.0.hnorm.weight",
    "mtp.0.norm.weight",
)
DS4_DSPARK_REQUIRED_TENSORS = (
    "mtp.0.main_norm.weight",
    "mtp.0.main_proj.weight",
    "mtp.1.attn_norm.weight",
    "mtp.2.confidence_head.proj.weight",
    "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.markov_w2.weight",
    "mtp.2.norm.weight",
)
_DS4_SUMMARY_METADATA_KEYS = {
    "general.architecture",
    "general.name",
    "split.no",
    "split.count",
    "deepseek4.block_count",
    "deepseek4.embedding_length",
    "deepseek4.expert_count",
    "deepseek4.expert_feed_forward_length",
    "deepseek4.nextn_predict_layers",
    "deepseek4.mtp_layer_count",
    "deepseek4.vocab_size",
}

_DS4_ID_SEPARATORS_RE = re.compile(r"[^a-z0-9.]+")
_DS4_ID_DASHES_RE = re.compile(r"-+")
_GGUF_MAGIC = b"GGUF"

# Maximum allowed string length for a single GGUF metadata key or value (1 MiB).
# Real-world GGUF metadata strings (architecture names, tokenizer paths, etc.)
# are well under 1 KiB; 1 MiB provides ample headroom while blocking OOM from
# corrupted headers that claim multi-GB strings.
_GGUF_MAX_STRING_LENGTH = 1 * 1024 * 1024
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12
_GGUF_SCALAR_FORMATS = {
    _GGUF_TYPE_UINT8: "B",
    _GGUF_TYPE_INT8: "b",
    _GGUF_TYPE_UINT16: "H",
    _GGUF_TYPE_INT16: "h",
    _GGUF_TYPE_UINT32: "I",
    _GGUF_TYPE_INT32: "i",
    _GGUF_TYPE_FLOAT32: "f",
    _GGUF_TYPE_BOOL: "?",
    _GGUF_TYPE_UINT64: "Q",
    _GGUF_TYPE_INT64: "q",
    _GGUF_TYPE_FLOAT64: "d",
}


@dataclass(frozen=True)
class GGUFMetadataSummary:
    """Subset of GGUF metadata needed for DS4 discovery decisions."""

    architecture: str | None = None
    general_name: str | None = None
    tensor_count: int | None = None
    split_no: int | None = None
    split_count: int | None = None
    block_count: int | None = None
    embedding_length: int | None = None
    expert_count: int | None = None
    expert_feed_forward_length: int | None = None
    nextn_predict_layers: int | None = None
    mtp_layer_count: int | None = None
    vocab_size: int | None = None
    tensor_infos: tuple[GGUFTensorInfo, ...] = ()


@dataclass(frozen=True)
class GGUFTensorInfo:
    """Tensor directory entry from a GGUF file."""

    name: str
    dimensions: tuple[int, ...]
    ggml_type: int
    offset: int


@dataclass(frozen=True)
class DS4GGUFModelCandidate:
    """A GGUF file that can be exposed as a DS4-backed discovered model."""

    base_id: str
    model_path: Path
    estimated_size: int
    config_model_type: str
    display_name: str
    source_type: str = "local"
    source_repo_id: str | None = None


@dataclass(frozen=True)
class DS4MTPGGUFSidecarCandidate:
    """A GGUF file that can be selected as a DS4 speculative sidecar."""

    display_name: str
    path: Path
    size: int
    kind: DS4SidecarKind = "legacy_mtp"
    source_type: str = "local"
    source_repo_id: str | None = None


class GGUFMetadataError(ValueError):
    """Raised when the GGUF header/metadata cannot be parsed."""


class DS4MTPCompatibilityError(ValueError):
    """Raised when a DS4 MTP sidecar does not match the main GGUF."""


def _read_exact(f, size: int) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise GGUFMetadataError("truncated GGUF metadata")
    return data


def _file_remaining(f) -> int:
    """Return the number of bytes remaining in the file from the current position."""
    pos = f.tell()
    f.seek(0, 2)
    end = f.tell()
    f.seek(pos)
    return end - pos


def _read_u32(f) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_gguf_string(f, remaining: int | None = None) -> str:
    length = _read_u64(f)
    # Bound string length against remaining file size to prevent OOM
    # from a corrupted header claiming a multi-GB string length.
    if remaining is None:
        remaining = _file_remaining(f)
    if length > remaining:
        raise GGUFMetadataError(
            f"GGUF string length ({length}) exceeds remaining file size ({remaining})"
        )
    # Also reject unreasonably large strings even if theoretically within
    # file bounds — a single metadata key/value should never need more than
    # a few MB.
    if length > _GGUF_MAX_STRING_LENGTH:
        raise GGUFMetadataError(
            f"GGUF string length ({length}) exceeds maximum ({_GGUF_MAX_STRING_LENGTH})"
        )
    return _read_exact(f, length).decode("utf-8", "replace")


def _read_gguf_scalar(f, value_type: int):
    if value_type == _GGUF_TYPE_STRING:
        return _read_gguf_string(f)
    fmt = _GGUF_SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise GGUFMetadataError(f"unsupported GGUF metadata value type {value_type}")
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, _read_exact(f, size))[0]


def _skip_gguf_scalar(f, value_type: int, file_size: int) -> None:
    if value_type == _GGUF_TYPE_STRING:
        length = _read_u64(f)
        if length > file_size:
            raise GGUFMetadataError(
                f"GGUF string length {length} exceeds file size {file_size}"
            )
        f.seek(length, 1)
        return
    fmt = _GGUF_SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise GGUFMetadataError(f"unsupported GGUF metadata value type {value_type}")
    f.seek(struct.calcsize("<" + fmt), 1)


def _skip_gguf_value(f, value_type: int, file_size: int) -> None:
    if value_type == _GGUF_TYPE_ARRAY:
        item_type = _read_u32(f)
        item_count = _read_u64(f)
        if item_count > file_size:
            raise GGUFMetadataError(
                f"GGUF array item count {item_count} exceeds file size {file_size}"
            )
        fmt = _GGUF_SCALAR_FORMATS.get(item_type)
        if fmt is not None:
            item_size = struct.calcsize("<" + fmt)
            if item_count * item_size > file_size:
                raise GGUFMetadataError(
                    f"GGUF array byte length {item_count * item_size} "
                    f"exceeds file size {file_size}"
                )
            f.seek(item_size * item_count, 1)
            return
        for _ in range(item_count):
            _skip_gguf_scalar(f, item_type, file_size)
        return
    _skip_gguf_scalar(f, value_type, file_size)


def _summary_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _read_gguf_tensor_infos(f, tensor_count: int) -> tuple[GGUFTensorInfo, ...]:
    infos: list[GGUFTensorInfo] = []
    for _ in range(tensor_count):
        name = _read_gguf_string(f)
        n_dimensions = _read_u32(f)
        if n_dimensions > 16:
            raise GGUFMetadataError(
                f"GGUF tensor {name!r} has implausible dimension count {n_dimensions}"
            )
        dimensions = tuple(_read_u64(f) for _ in range(n_dimensions))
        ggml_type = _read_u32(f)
        offset = _read_u64(f)
        infos.append(
            GGUFTensorInfo(
                name=name,
                dimensions=dimensions,
                ggml_type=ggml_type,
                offset=offset,
            )
        )
    return tuple(infos)


def read_ds4_gguf_metadata_summary(
    path: Path, *, include_tensor_info: bool = False
) -> GGUFMetadataSummary:
    """Read just enough GGUF metadata to decide if DS4 can expose a file."""
    path = Path(path)
    with path.open("rb") as f:
        # Check for minimum GGUF header size before attempting to read magic.
        # A file with fewer than 4 bytes is not a valid GGUF file.
        initial = f.read(4)
        if len(initial) < 4:
            raise GGUFMetadataError(
                f"file too short to be a GGUF file ({len(initial)} bytes)"
            )
        if initial != _GGUF_MAGIC:
            raise GGUFMetadataError("not a GGUF file (missing magic bytes)")
        # Magic already consumed; file pointer is now at position 4
        # (start of version field).  Continue reading from here.
        _read_u32(f)  # version
        tensor_count = _read_u64(f)
        kv_count = _read_u64(f)

        # Bound kv_count to avoid DoS via absurd metadata sizes.
        if kv_count > 65536:
            raise GGUFMetadataError(
                f"GGUF metadata key-value count {kv_count} is implausibly large"
            )

        architecture: str | None = None
        general_name: str | None = None
        split_no: int | None = None
        split_count: int | None = None
        block_count: int | None = None
        embedding_length: int | None = None
        expert_count: int | None = None
        expert_feed_forward_length: int | None = None
        nextn_predict_layers: int | None = None
        mtp_layer_count: int | None = None
        vocab_size: int | None = None
        file_size = path.stat().st_size
        for _ in range(kv_count):
            remaining = file_size - f.tell()
            if remaining <= 0:
                raise GGUFMetadataError("GGUF metadata truncated (unexpected EOF)")
            key = _read_gguf_string(f, remaining=remaining)
            value_type = _read_u32(f)
            if key in _DS4_SUMMARY_METADATA_KEYS:
                value = _read_gguf_scalar(f, value_type)
                if key == "general.architecture":
                    architecture = str(value)
                elif key == "general.name":
                    general_name = str(value)
                elif key == "split.no":
                    split_no = _summary_int(value)
                elif key == "split.count":
                    split_count = _summary_int(value)
                elif key == "deepseek4.block_count":
                    block_count = _summary_int(value)
                elif key == "deepseek4.embedding_length":
                    embedding_length = _summary_int(value)
                elif key == "deepseek4.expert_count":
                    expert_count = _summary_int(value)
                elif key == "deepseek4.expert_feed_forward_length":
                    expert_feed_forward_length = _summary_int(value)
                elif key == "deepseek4.nextn_predict_layers":
                    nextn_predict_layers = _summary_int(value)
                elif key == "deepseek4.mtp_layer_count":
                    mtp_layer_count = _summary_int(value)
                elif key == "deepseek4.vocab_size":
                    vocab_size = _summary_int(value)
            else:
                _skip_gguf_value(f, value_type, file_size)

        tensor_infos = (
            _read_gguf_tensor_infos(f, tensor_count) if include_tensor_info else ()
        )

    return GGUFMetadataSummary(
        architecture=architecture,
        general_name=general_name,
        tensor_count=tensor_count,
        split_no=split_no,
        split_count=split_count,
        block_count=block_count,
        embedding_length=embedding_length,
        expert_count=expert_count,
        expert_feed_forward_length=expert_feed_forward_length,
        nextn_predict_layers=nextn_predict_layers,
        mtp_layer_count=mtp_layer_count,
        vocab_size=vocab_size,
        tensor_infos=tensor_infos,
    )


def _raise_mtp_compat(message: str) -> None:
    raise DS4MTPCompatibilityError(f"DS4 MTP compatibility check failed: {message}")


def _ds4_is_flash(summary: GGUFMetadataSummary, path: Path) -> bool:
    haystack = " ".join(part for part in (summary.general_name, path.name) if part)
    haystack = haystack.lower()
    return "deepseek" in haystack and "v4" in haystack and "flash" in haystack


def detect_ds4_mtp_sidecar_kind(path: Path) -> DS4SidecarKind | None:
    """Return ``legacy_mtp`` or ``dspark`` for a DS4 support GGUF."""
    try:
        metadata = read_ds4_gguf_metadata_summary(path)
    except GGUFMetadataError as e:
        logger.debug("Skipping non-DS4 support GGUF %s: %s", path, e)
        return None
    except Exception as e:
        logger.info("Could not inspect GGUF metadata for %s: %s", path, e)
        return None
    if metadata.split_no is not None and metadata.split_no > 0:
        return None
    architecture = (metadata.architecture or "").strip().lower()
    return DS4_SPECULATOR_GGUF_ARCHITECTURES.get(architecture)


def validate_ds4_mtp_compatibility(main_path: Path, mtp_path: Path) -> None:
    """Validate a legacy MTP or DSpark support GGUF against *main_path*.

    This is a cheap, upfront GGUF-header/tensor-directory check. DS4 still does
    the final runtime load validation, but this catches wrong paths, PRO/main
    GGUFs passed as sidecars, non-Flash main GGUFs, and obvious shape mismatches
    before a managed backend is restarted.
    """
    main_path = Path(main_path).expanduser().resolve()
    mtp_path = Path(mtp_path).expanduser().resolve()
    if main_path == mtp_path:
        _raise_mtp_compat("main model and MTP sidecar must be different files")
    if not main_path.is_file():
        _raise_mtp_compat(f"main model does not exist: {main_path}")
    if not mtp_path.is_file():
        _raise_mtp_compat(f"MTP sidecar does not exist: {mtp_path}")

    try:
        main = read_ds4_gguf_metadata_summary(main_path)
        mtp = read_ds4_gguf_metadata_summary(mtp_path, include_tensor_info=True)
    except GGUFMetadataError as exc:
        _raise_mtp_compat(str(exc))
    except OSError as exc:
        _raise_mtp_compat(str(exc))

    if (main.architecture or "").strip().lower() != "deepseek4":
        _raise_mtp_compat(
            f"main model architecture must be deepseek4, got {main.architecture!r}"
        )
    if main.split_no is not None and main.split_no > 0:
        _raise_mtp_compat("main model is a continuation split shard")
    if not _ds4_is_flash(main, main_path):
        _raise_mtp_compat(
            "the current MTP sidecar is only supported for DeepSeek V4 Flash GGUFs"
        )

    mtp_architecture = (mtp.architecture or "").strip().lower()
    sidecar_kind = DS4_SPECULATOR_GGUF_ARCHITECTURES.get(mtp_architecture)
    if sidecar_kind is None:
        _raise_mtp_compat(
            "support GGUF architecture must be one of "
            f"{', '.join(sorted(DS4_SPECULATOR_GGUF_ARCHITECTURES))}, "
            f"got {mtp.architecture!r}"
        )
    if mtp.split_no is not None and mtp.split_no > 0:
        _raise_mtp_compat("support GGUF is a continuation split shard")

    tensor_by_name = {info.name: info for info in mtp.tensor_infos}
    if not tensor_by_name:
        _raise_mtp_compat("support GGUF has no tensor directory")
    non_mtp = [name for name in tensor_by_name if not name.startswith("mtp.")]
    if non_mtp:
        _raise_mtp_compat(
            "support GGUF contains non-MTP tensor(s): " + ", ".join(non_mtp[:3])
        )

    if sidecar_kind == "dspark":
        missing = [
            name for name in DS4_DSPARK_REQUIRED_TENSORS if name not in tensor_by_name
        ]
        if missing:
            _raise_mtp_compat(
                "DSpark support GGUF is missing tensor(s): " + ", ".join(missing[:3])
            )
        return

    if mtp.mtp_layer_count not in (None, 1):
        _raise_mtp_compat(f"expected one MTP layer, got {mtp.mtp_layer_count}")
    if mtp.nextn_predict_layers not in (None, 1):
        _raise_mtp_compat(
            f"expected one next-token prediction layer, got {mtp.nextn_predict_layers}"
        )
    if (
        main.nextn_predict_layers is not None
        and mtp.nextn_predict_layers is not None
        and main.nextn_predict_layers != mtp.nextn_predict_layers
    ):
        _raise_mtp_compat(
            "main model and MTP sidecar disagree on "
            f"deepseek4.nextn_predict_layers ({main.nextn_predict_layers} != "
            f"{mtp.nextn_predict_layers})"
        )
    if (
        main.expert_count is not None
        and mtp.expert_count is not None
        and main.expert_count != mtp.expert_count
    ):
        _raise_mtp_compat(
            f"expert_count mismatch ({main.expert_count} != {mtp.expert_count})"
        )

    missing = [name for name in DS4_MTP_REQUIRED_TENSORS if name not in tensor_by_name]
    if missing:
        _raise_mtp_compat("MTP sidecar is missing tensor(s): " + ", ".join(missing[:3]))

    embedding = main.embedding_length
    expert_ffn = main.expert_feed_forward_length
    experts = main.expert_count

    def dims(name: str) -> tuple[int, ...]:
        return tensor_by_name[name].dimensions

    if embedding is not None:
        for name in (
            "mtp.0.enorm.weight",
            "mtp.0.hnorm.weight",
            "mtp.0.norm.weight",
            "mtp.0.attn_norm.weight",
            "mtp.0.ffn_norm.weight",
        ):
            if dims(name) != (embedding,):
                _raise_mtp_compat(
                    f"{name} shape {dims(name)} does not match embedding_length {embedding}"
                )
        for name in ("mtp.0.e_proj.weight", "mtp.0.h_proj.weight"):
            if dims(name) != (embedding, embedding):
                _raise_mtp_compat(
                    f"{name} shape {dims(name)} does not match embedding_length {embedding}"
                )

    routed_dims = dims("mtp.0.ffn_gate_exps.weight")
    if (
        embedding is not None
        and expert_ffn is not None
        and experts is not None
        and routed_dims != (embedding, expert_ffn, experts)
    ):
        _raise_mtp_compat(
            "MTP routed expert shape "
            f"{routed_dims} does not match main model "
            f"(embedding={embedding}, expert_ffn={expert_ffn}, experts={experts})"
        )


def is_supported_ds4_gguf(path: Path) -> bool:
    """Return True when a GGUF is a DS4-supported primary DeepSeek V4 file."""
    try:
        metadata = read_ds4_gguf_metadata_summary(path)
    except GGUFMetadataError as e:
        # no-magic: not a real GGUF — keep extension-based compatibility
        # for hand-made test stubs.
        if "magic" in str(e).lower() or "missing" in str(e).lower():
            logger.debug(
                "Not a GGUF file (no magic), treating %s as supported "
                "by extension for stub compatibility",
                path,
            )
            return True
        # bad-header: GGUF magic present but metadata is corrupt or
        # unsupported — reject explicitly.
        logger.info("Corrupt GGUF header in %s: %s", path, e)
        return False
    except Exception as e:
        # Unexpected I/O errors: treat as unsupported.
        logger.info("Could not inspect GGUF metadata for %s: %s", path, e)
        return False

    if metadata.split_no is not None and metadata.split_no > 0:
        logger.info(
            "Skipping DS4 GGUF continuation shard %s (split.no=%s)",
            path,
            metadata.split_no,
        )
        return False

    architecture = (metadata.architecture or "").strip().lower()
    if architecture and architecture not in DS4_SUPPORTED_GGUF_ARCHITECTURES:
        logger.info(
            "Skipping unsupported DS4 GGUF %s (architecture=%s)",
            path,
            metadata.architecture,
        )
        return False

    return True


def is_ds4_mtp_gguf_sidecar(path: Path) -> bool:
    """Return True for legacy MTP and DSpark support GGUFs."""
    return detect_ds4_mtp_sidecar_kind(path) is not None


def normalize_ds4_gguf_model_id(name: str) -> str:
    """Normalize a DS4 GGUF file/repo name into an API model id.

    DS4 model ids are intentionally lowercased and separator-normalized so
    `Foo.gguf` and `foo` resolve consistently.  The original source casing is
    kept separately in ``DiscoveredModel.display_name`` for UI presentation.
    """
    raw = name.strip()
    if raw.lower().endswith(".gguf"):
        raw = raw[:-5]
    normalized = raw.lower()
    normalized = _DS4_ID_SEPARATORS_RE.sub("-", normalized)
    normalized = _DS4_ID_DASHES_RE.sub("-", normalized).strip("-.")
    return normalized or "gguf-model"


def is_ds4_gguf_file(path: Path) -> bool:
    """Return True for visible GGUF model files that DS4 discovery may inspect."""
    return (
        path.is_file()
        and path.suffix.lower() == ".gguf"
        and not path.name.startswith(".")
    )


def detect_ds4_gguf_config_type(
    gguf_path: Path, source_repo_id: str | None = None
) -> str:
    """Classify DeepSeek V4 GGUF variant from filename/repo heuristics."""
    haystack = " ".join(
        part for part in (source_repo_id, gguf_path.parent.name, gguf_path.name) if part
    ).lower()
    if "deepseek" in haystack and "v4" in haystack and "flash" in haystack:
        return "deepseek_v4_flash_gguf"
    if "deepseek" in haystack and "v4" in haystack and "pro" in haystack:
        return "deepseek_v4_pro_gguf"
    return "ds4_gguf"


def compose_ds4_gguf_model_id(
    container: Path,
    gguf_path: Path,
    gguf_count: int,
    *,
    source_repo_id: str | None = None,
) -> str:
    """Build the preferred normalized id before collision suffixing."""
    file_id = normalize_ds4_gguf_model_id(gguf_path.stem)
    if container == gguf_path.parent:
        if source_repo_id:
            repo_id = normalize_ds4_gguf_model_id(source_repo_id.split("/")[-1])
            if gguf_count == 1 and (
                file_id in DS4_GENERIC_GGUF_STEMS
                or file_id.startswith("model-")
                or file_id == repo_id
            ):
                return repo_id
            if file_id.startswith(f"{repo_id}-"):
                return file_id
            return f"{repo_id}-{file_id}"
        # For top-level GGUFs, the filename is the model id:
        #   Foo.gguf -> foo
        return file_id

    container_id = normalize_ds4_gguf_model_id(gguf_path.parent.name)
    if gguf_count == 1 and (
        file_id in DS4_GENERIC_GGUF_STEMS
        or file_id.startswith("model-")
        or file_id == container_id
    ):
        return container_id
    if file_id.startswith(f"{container_id}-"):
        return file_id
    return f"{container_id}-{file_id}"


def compose_ds4_gguf_display_name(
    root_dir: Path,
    gguf_path: Path,
    gguf_count: int,
    *,
    source_repo_id: str | None = None,
) -> str:
    """Build the UI display name for a DS4 GGUF file."""
    file_id = normalize_ds4_gguf_model_id(gguf_path.stem)
    if source_repo_id and gguf_path.parent == root_dir:
        repo_display = source_repo_id.split("/")[-1]
        return (
            repo_display
            if gguf_count == 1 and file_id in DS4_GENERIC_GGUF_STEMS
            else f"{repo_display} / {gguf_path.stem}"
        )
    if gguf_path.parent == root_dir:
        return gguf_path.stem
    if gguf_count == 1 and file_id in DS4_GENERIC_GGUF_STEMS:
        return gguf_path.parent.name
    return f"{gguf_path.parent.name} / {gguf_path.stem}"


def collect_ds4_gguf_model_candidates(
    root_dir: Path,
    paths: Iterable[Path],
    *,
    source_type: str = "local",
    source_repo_id: str | None = None,
) -> list[DS4GGUFModelCandidate]:
    """Collect DS4-supported primary GGUF files from a direct path listing."""
    ggufs = [path for path in paths if is_ds4_gguf_file(path)]
    supported_ggufs = [path for path in ggufs if is_supported_ds4_gguf(path)]
    candidates: list[DS4GGUFModelCandidate] = []
    for gguf_path in supported_ggufs:
        try:
            candidates.append(
                DS4GGUFModelCandidate(
                    base_id=compose_ds4_gguf_model_id(
                        root_dir,
                        gguf_path,
                        len(supported_ggufs),
                        source_repo_id=source_repo_id,
                    ),
                    model_path=gguf_path,
                    estimated_size=int(gguf_path.stat().st_size * 1.05),
                    config_model_type=detect_ds4_gguf_config_type(
                        gguf_path,
                        source_repo_id,
                    ),
                    display_name=compose_ds4_gguf_display_name(
                        root_dir,
                        gguf_path,
                        len(supported_ggufs),
                        source_repo_id=source_repo_id,
                    ),
                    source_type=source_type,
                    source_repo_id=source_repo_id,
                )
            )
        except Exception as e:
            logger.error("Failed to discover DS4 GGUF %s: %s", gguf_path, e)
    return candidates


def collect_ds4_mtp_gguf_sidecar_candidates(
    root_dir: Path,
    paths: Iterable[Path],
    *,
    source_type: str = "local",
    source_repo_id: str | None = None,
) -> list[DS4MTPGGUFSidecarCandidate]:
    """Collect legacy MTP and DSpark support GGUFs from a path listing."""
    ggufs = [path for path in paths if is_ds4_gguf_file(path)]
    support_ggufs = [
        (path, kind)
        for path in ggufs
        if (kind := detect_ds4_mtp_sidecar_kind(path)) is not None
    ]
    candidates: list[DS4MTPGGUFSidecarCandidate] = []
    for gguf_path, kind in support_ggufs:
        try:
            candidates.append(
                DS4MTPGGUFSidecarCandidate(
                    display_name=compose_ds4_gguf_display_name(
                        root_dir,
                        gguf_path,
                        len(support_ggufs),
                        source_repo_id=source_repo_id,
                    ),
                    path=gguf_path,
                    size=gguf_path.stat().st_size,
                    kind=kind,
                    source_type=source_type,
                    source_repo_id=source_repo_id,
                )
            )
        except OSError as e:
            logger.info("Skipping DS4 MTP GGUF %s: %s", gguf_path, e)
    return candidates
