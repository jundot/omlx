# SPDX-License-Identifier: Apache-2.0
"""HuggingFace downloader catalog helpers for DS4-GGUF filters."""

from datetime import datetime, timezone

_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}

# DS4 v1 targets DeepSeek V4 Flash/Pro GGUF repos. HuggingFace exposes these
# as base-model tags (the same values used by the web UI's `other=` filter).
DS4_GGUF_BASE_MODEL_FILTERS = (
    "base_model:quantized:deepseek-ai/DeepSeek-V4-Flash",
    "base_model:finetune:deepseek-ai/DeepSeek-V4-Flash",
    "base_model:quantized:deepseek-ai/DeepSeek-V4-Pro",
    "base_model:finetune:deepseek-ai/DeepSeek-V4-Pro",
)


def calc_safetensors_disk_size(safetensors: dict) -> int:
    """Calculate actual disk size in bytes from safetensors parameters."""
    params = safetensors.get("parameters", {})
    if not params:
        return 0
    return sum(count * _DTYPE_BYTES.get(dtype, 1) for dtype, count in params.items())


def format_model_size(size_bytes: int) -> str:
    """Format model size in bytes to a human-readable string."""
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    return f"{size_bytes / 1024**3:.1f} GB"


def format_param_count(total_params: int) -> str:
    """Format parameter count to a human-readable string."""
    if total_params >= 1e12:
        return f"{total_params / 1e12:.1f}T"
    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"
    if total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"
    return str(total_params)


def get_param_count(safetensors: dict) -> int:
    """Get total parameter count from safetensors metadata."""
    params = safetensors.get("parameters", {})
    if not params:
        return 0
    return sum(params.values())


def hf_tags(model) -> list[str]:
    """Return string tags from a HF model/info object or test dict."""
    tags = model.get("tags") if isinstance(model, dict) else getattr(model, "tags", None)
    if not isinstance(tags, (list, tuple, set)):
        return []
    return [tag for tag in tags if isinstance(tag, str)]


def hf_siblings(model) -> list:
    """Return HF sibling/file entries without triggering MagicMock iteration."""
    siblings = (
        model.get("siblings") if isinstance(model, dict) else getattr(model, "siblings", None)
    )
    return list(siblings) if isinstance(siblings, (list, tuple)) else []


def hf_sibling_name(sibling) -> str:
    """Return a normalized filename from a HF sibling/file entry."""
    if isinstance(sibling, dict):
        value = sibling.get("rfilename") or sibling.get("name") or sibling.get("Name")
    else:
        value = getattr(sibling, "rfilename", None) or getattr(sibling, "name", None)
    return value if isinstance(value, str) else ""


def hf_sibling_size(sibling) -> int:
    """Return a normalized byte size from a HF sibling/file entry."""
    if isinstance(sibling, dict):
        value = sibling.get("size") or sibling.get("Size") or 0
    else:
        value = getattr(sibling, "size", 0) or 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def hf_gguf_metadata(model):
    """Return HF GGUF metadata from a model object."""
    gguf = model.get("gguf") if isinstance(model, dict) else getattr(model, "gguf", None)
    if gguf is None or isinstance(gguf, (bool, int, float, str)):
        return None
    if not isinstance(gguf, dict):
        return None
    return gguf


def has_gguf_metadata(model) -> bool:
    """Return whether HF explicitly reports GGUF metadata for this repo."""
    gguf = hf_gguf_metadata(model)
    return isinstance(gguf, dict) and bool(gguf)


def hf_gguf_total_size(model) -> int:
    """Return total GGUF byte size from HF list metadata when available."""
    gguf = hf_gguf_metadata(model)
    if not isinstance(gguf, dict):
        return 0
    for key in ("totalFileSize", "total_file_size", "totalSize", "size"):
        value = gguf.get(key) if isinstance(gguf, dict) else getattr(gguf, key, None)
        if not isinstance(value, (int, float, str)):
            continue
        try:
            size = int(value)
        except (TypeError, ValueError):
            continue
        if size > 0:
            return size
    return 0


def gguf_size_from_siblings(model) -> int:
    """Calculate total size of repo GGUF files from HF GGUF/file metadata."""
    metadata_size = hf_gguf_total_size(model)
    if metadata_size > 0:
        return metadata_size
    total = 0
    for sibling in hf_siblings(model):
        if hf_sibling_name(sibling).lower().endswith(".gguf"):
            total += hf_sibling_size(sibling)
    return total


def has_gguf_files(model) -> bool:
    """Best-effort GGUF repo detection from HF metadata, files, or tags."""
    if has_gguf_metadata(model):
        return True
    if any(hf_sibling_name(s).lower().endswith(".gguf") for s in hf_siblings(model)):
        return True
    return any(tag.lower() == "gguf" for tag in hf_tags(model))


def hf_timestamp(model, *names: str) -> float:
    """Return a sortable POSIX timestamp from HF model datetime metadata."""
    for name in names:
        value = model.get(name) if isinstance(model, dict) else getattr(model, name, None)
        if value is None:
            continue
        if isinstance(value, datetime):
            dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            try:
                return datetime.fromisoformat(text).timestamp()
            except ValueError:
                continue
    return 0.0


def resolve_hf_catalog_filters(
    *,
    mlx_only: bool,
    show_mlx: bool | None,
    show_ds4_gguf: bool | None,
) -> tuple[bool, bool]:
    """Resolve new OR-filter flags while preserving legacy mlx_only callers."""
    if show_mlx is None and show_ds4_gguf is None:
        # Legacy mode: `mlx_only` controls how narrowly the MLX/safetensors
        # query is scoped; it never used to include GGUF candidates.
        return True, False
    return bool(show_mlx), bool(show_ds4_gguf)


def merge_catalog_results(results: list[dict]) -> list[dict]:
    """Merge duplicate repos returned by multiple backend/base-model filters."""
    merged: dict[str, dict] = {}
    for item in results:
        repo_id = item.get("repo_id")
        if not repo_id:
            continue
        existing = merged.get(repo_id)
        if existing is None:
            merged[repo_id] = item
            continue
        backends = set(existing.get("backends") or [existing.get("backend")])
        backends.update(item.get("backends") or [item.get("backend")])
        backends.discard(None)
        existing["backends"] = sorted(backends)
        if len(backends) > 1:
            existing["backend"] = "mlx+ds4"
            existing["backend_label"] = "MLX + DS4-GGUF"
            existing["format"] = "mixed"
        if not existing.get("size") and item.get("size"):
            existing["size"] = item["size"]
            existing["size_formatted"] = item.get("size_formatted", "")
        for field in ("created_at", "updated_at"):
            existing[field] = max(existing.get(field, 0) or 0, item.get(field, 0) or 0)
        existing.setdefault("base_model_filters", [])
        for filter_name in item.get("base_model_filters", []):
            if filter_name not in existing["base_model_filters"]:
                existing["base_model_filters"].append(filter_name)
    return list(merged.values())


def hf_catalog_item(
    model,
    *,
    backend: str,
    name: str,
    base_model_filter: str | None = None,
) -> dict:
    """Normalize one HF model listing entry for admin downloader catalog APIs."""
    params = None
    params_formatted = None
    size = 0
    if backend == "ds4":
        size = gguf_size_from_siblings(model)
    else:
        safetensors = getattr(model, "safetensors", None)
        if safetensors and safetensors.get("parameters"):
            params = get_param_count(safetensors)
            params_formatted = format_param_count(params) if params > 0 else None
            size = calc_safetensors_disk_size(safetensors)
            if params and params <= 0:
                params = None

    if backend == "ds4":
        backends = ["ds4"]
        backend_label = "DS4-GGUF"
        model_format = "gguf"
        compatibility_status = "unverified"
        compatibility_note = (
            "DeepSeek V4 GGUF candidate; compatibility is unverified and DS4 "
            "will validate the downloaded repo at launch."
        )
    else:
        backends = ["mlx"]
        backend_label = "MLX"
        model_format = "safetensors"
        compatibility_status = "verified"
        compatibility_note = "MLX library model."

    item = {
        "repo_id": model.id,
        "name": name,
        "downloads": model.downloads or 0,
        "likes": model.likes or 0,
        "trending_score": model.trending_score or 0,
        "created_at": hf_timestamp(model, "created_at", "createdAt"),
        "updated_at": hf_timestamp(
            model,
            "last_modified",
            "lastModified",
            "updated_at",
            "updatedAt",
        ),
        "size": size,
        "size_formatted": format_model_size(size) if size > 0 else "",
        "params": params,
        "params_formatted": params_formatted,
        "backend": backend,
        "backends": backends,
        "backend_label": backend_label,
        "format": model_format,
        "compatibility_status": compatibility_status,
        "compatibility_note": compatibility_note,
    }
    if base_model_filter:
        item["base_model_filters"] = [base_model_filter]
    return item
