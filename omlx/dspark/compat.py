"""dSpark checkpoint discovery and target/drafter compatibility checks.

This module deliberately performs no model import and never executes remote code.
It is safe to use from model discovery and the admin compatibility endpoint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

DSparkFormat = Literal["deepspec", "speculators", "higgs_sidecar"]
PairingMode = Literal["exact", "verified_cross_target"]
SUPPORTED_TARGET_TYPES = {
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_6",
    "qwen3_6_text",
    "qwen3_6_moe",
    "qwen3_6_moe_text",
}


@dataclass(frozen=True)
class DSparkProbe:
    path: str
    format: DSparkFormat
    architecture: str
    vocab_size: int | None
    hidden_size: int | None
    target_layers: int | None
    target_layer_ids: tuple[int, ...]
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    block_size: int | None
    markov_rank: int | None
    owns_embedding: bool
    owns_output_head: bool
    tensor_count: int
    weight_bytes: int
    target_fingerprint: str | None = None
    source_revision: str | None = None
    quantization: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DSparkCompatibility:
    compatible: bool
    format: str | None
    pairing_mode: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    blocked_reasons: tuple[str, ...] = field(default_factory=tuple)
    target_fingerprint: str | None = None
    drafter_fingerprint: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    probe: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {path.name}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {path.name}: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def _architectures(config: dict[str, Any]) -> list[str]:
    value = config.get("architectures") or []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def detect_format(config: dict[str, Any]) -> DSparkFormat:
    arches = [a.lower() for a in _architectures(config)]
    if "speculators_config" in config or any("dsparkdraftmodel" in a for a in arches):
        return "speculators"
    if any("qwen3dsparkmodel" in a or "deepspec" in a for a in arches):
        return "deepspec"
    if "dflash_config" in config or any("dflashdraftmodel" in a for a in arches):
        if config.get("markov_rank") or config.get("markov_head_type"):
            return "speculators"
        return "higgs_sidecar"
    if "dspark_config" in config:
        return "deepspec"
    raise ValueError("checkpoint is not a recognized dSpark helper format")


def _nested_int(config: dict[str, Any], key: str) -> int | None:
    candidates = [
        config,
        config.get("dspark_config") or {},
        config.get("dflash_config") or {},
        config.get("speculators_config") or {},
        config.get("transformer_layer_config") or {},
    ]
    for block in candidates:
        value = block.get(key) if isinstance(block, dict) else None
        if isinstance(value, int):
            return value
    return None


def _target_taps(config: dict[str, Any]) -> tuple[int, ...]:
    for block in (
        config,
        config.get("dspark_config") or {},
        config.get("dflash_config") or {},
        config.get("speculators_config") or {},
    ):
        value = block.get("target_layer_ids") if isinstance(block, dict) else None
        if isinstance(value, list) and all(isinstance(v, int) for v in value):
            return tuple(value)
    value = config.get("aux_hidden_state_layer_ids")
    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return tuple(value)
    return ()


def _tensor_inventory(root: Path) -> tuple[int, int]:
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise ValueError(f"dSpark checkpoint has no safetensors weights: {root}")
    total_bytes = sum(path.stat().st_size for path in files)
    try:
        from safetensors import safe_open

        count = 0
        for path in files:
            with safe_open(path, framework="numpy") as handle:
                count += len(handle.keys())
        return count, total_bytes
    except Exception:
        # Header inspection is best-effort in the lightweight admin process;
        # the model loader performs a strict inventory before allocation.
        return 0, total_bytes


def _tensor_names(root: Path) -> set[str]:
    try:
        from safetensors import safe_open

        names: set[str] = set()
        for path in sorted(root.glob("*.safetensors")):
            with safe_open(path, framework="numpy") as handle:
                names.update(handle.keys())
        return names
    except Exception:
        return set()


def model_fingerprint(config_or_root: dict[str, Any] | str | Path) -> str:
    if isinstance(config_or_root, dict):
        config = config_or_root
    else:
        root = Path(config_or_root).expanduser().resolve()
        config = _read_json(root / "config.json")
    text = (
        config.get("text_config")
        if isinstance(config.get("text_config"), dict)
        else config
    )
    stable = {
        key: text.get(key)
        for key in (
            "model_type",
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "full_attention_interval",
        )
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def probe_drafter(path: str | Path, requested_format: str = "auto") -> DSparkProbe:
    root = Path(path).expanduser().resolve()
    config = _read_json(root / "config.json")
    detected = detect_format(config)
    if requested_format != "auto" and requested_format != detected:
        raise ValueError(
            f"configured dspark_format={requested_format!r}, detected {detected!r}"
        )
    manifest_path = root / "dspark_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    tensor_count, weight_bytes = _tensor_inventory(root)
    tensor_names = _tensor_names(root)
    arches = _architectures(config)
    owns_embedding = detected == "deepspec" or any(
        name == "embed_tokens.weight" or name.startswith("embed_tokens.")
        for name in tensor_names
    )
    owns_output_head = detected == "deepspec" or any(
        name == "lm_head.weight" or name.startswith("lm_head.") for name in tensor_names
    )
    return DSparkProbe(
        path=str(root),
        format=detected,
        architecture=arches[0] if arches else "",
        vocab_size=_nested_int(config, "vocab_size"),
        hidden_size=_nested_int(config, "hidden_size"),
        target_layers=(
            _nested_int(config, "num_target_layers")
            or manifest.get("target_num_hidden_layers")
        ),
        target_layer_ids=_target_taps(config),
        num_attention_heads=_nested_int(config, "num_attention_heads"),
        num_key_value_heads=_nested_int(config, "num_key_value_heads"),
        head_dim=_nested_int(config, "head_dim"),
        block_size=_nested_int(config, "block_size"),
        markov_rank=_nested_int(config, "markov_rank"),
        owns_embedding=bool(manifest.get("owns_embedding", owns_embedding)),
        owns_output_head=bool(manifest.get("owns_output_head", owns_output_head)),
        tensor_count=tensor_count,
        weight_bytes=weight_bytes,
        target_fingerprint=manifest.get("target_fingerprint"),
        source_revision=manifest.get("source_revision"),
        quantization=(
            manifest.get("quantization")
            or config.get("quantization")
            or config.get("quantization_config")
        ),
    )


def _target_config(root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_json(Path(root).expanduser().resolve() / "config.json")
    text = config.get("text_config")
    return config, text if isinstance(text, dict) else config


def validate_pair(
    target_path: str | Path,
    drafter_path: str | Path,
    *,
    requested_format: str = "auto",
    pairing_mode: PairingMode = "exact",
) -> DSparkCompatibility:
    warnings: list[str] = []
    blocked: list[str] = []
    try:
        probe = probe_drafter(drafter_path, requested_format)
        _, target = _target_config(target_path)
    except ValueError as exc:
        return DSparkCompatibility(
            compatible=False,
            format=None,
            pairing_mode=pairing_mode,
            blocked_reasons=(str(exc),),
        )

    target_type = str(target.get("model_type") or "").lower()
    if target_type not in SUPPORTED_TARGET_TYPES:
        blocked.append(
            f"unsupported dSpark target model_type: {target_type or 'missing'}"
        )

    checks = (
        ("vocab_size", probe.vocab_size, target.get("vocab_size")),
        ("hidden_size", probe.hidden_size, target.get("hidden_size")),
        ("num_target_layers", probe.target_layers, target.get("num_hidden_layers")),
    )
    for name, draft_value, target_value in checks:
        if (
            draft_value is not None
            and target_value is not None
            and draft_value != target_value
        ):
            blocked.append(
                f"{name} mismatch: drafter={draft_value}, target={target_value}"
            )
    target_layers = target.get("num_hidden_layers")
    if probe.target_layer_ids and isinstance(target_layers, int):
        invalid = [
            tap for tap in probe.target_layer_ids if tap < 0 or tap >= target_layers
        ]
        if invalid:
            blocked.append(f"target_layer_ids out of range: {invalid}")

    target_fp = model_fingerprint(target)
    if probe.target_fingerprint:
        if probe.target_fingerprint != target_fp:
            if pairing_mode == "exact":
                blocked.append(
                    "drafter target fingerprint does not match selected target"
                )
            else:
                warnings.append(
                    "cross-target pairing; startup smoke verification required"
                )
    elif pairing_mode == "exact":
        blocked.append(
            "drafter has no target_fingerprint; prepare a managed checkpoint first"
        )
    else:
        warnings.append(
            "drafter has no target fingerprint; using verified cross-target mode"
        )

    return DSparkCompatibility(
        compatible=not blocked,
        format=probe.format,
        pairing_mode=pairing_mode,
        warnings=tuple(warnings),
        blocked_reasons=tuple(blocked),
        target_fingerprint=target_fp,
        drafter_fingerprint=probe.target_fingerprint,
        capabilities={
            "sampling": True,
            "markov": probe.markov_rank is not None,
            # Requests share the native continuous Scheduler and admission
            # controls.  Target verification is currently row-isolated so a
            # rejection can roll back only that request's hybrid cache.
            "continuous_batching": True,
            "continuous_scheduling": True,
            "batched_target_verify": True,
            "turboquant": True,
            "specprefill": True,
            "specprefill_mode": "native_target_only_cap_zero",
            "target_only_cap_zero": True,
            "shares_target_embedding": not probe.owns_embedding,
            "shares_target_output_head": not probe.owns_output_head,
            "compact_vocabulary": bool(
                probe.owns_output_head and probe.format == "speculators"
            ),
        },
        probe=probe.to_dict(),
    )
