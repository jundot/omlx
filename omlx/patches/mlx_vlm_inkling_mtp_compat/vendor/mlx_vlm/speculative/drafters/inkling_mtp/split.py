import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

import mlx.core as mx
from safetensors import safe_open

from ....utils import get_model_path
from .inkling_mtp import InklingMTPDraftModel

_MTP_PREFIX = "model.mtp."
_NORM_KEY = "model.llm.norm.weight"


def _wanted(key: str) -> bool:
    return key.startswith(_MTP_PREFIX) or key == _NORM_KEY


def _strip(key: str) -> str:
    if key.startswith(_MTP_PREFIX):
        return key[len(_MTP_PREFIX) :]
    if key == _NORM_KEY:
        return "norm.weight"
    return key


def _safetensor_files(model_path: Path) -> list:
    return [
        Path(path)
        for path in glob.glob(str(model_path / "*.safetensors"))
        if not path.endswith("consolidated.safetensors")
    ]


def _weight_map(model_path: Path) -> Dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        return json.load(f).get("weight_map", {})


def _iter_keys(model_path: Path) -> Iterable[tuple]:
    weight_map = _weight_map(model_path)
    if weight_map:
        by_file: Dict[str, list] = {}
        for key, filename in weight_map.items():
            if _wanted(key):
                by_file.setdefault(filename, []).append(key)
        if by_file:
            for filename, keys in by_file.items():
                yield model_path / filename, keys
            return

    for file in _safetensor_files(model_path):
        with safe_open(file, framework="mlx") as f:
            keys = [key for key in f.keys() if _wanted(key)]
        if keys:
            yield file, keys


def _is_mlx_safetensors(file: Path) -> bool:
    with safe_open(file, framework="mlx") as f:
        metadata = f.metadata() or {}
    return metadata.get("format") == "mlx"


def _load_selected_tensors(file: Path, keys: list) -> Dict[str, mx.array]:
    tensors = {}
    try:
        with safe_open(file, framework="mlx") as f:
            for key in keys:
                tensors[_strip(key)] = mx.array(f.get_tensor(key))
    except TypeError:
        shard = mx.load(str(file))
        tensors = {_strip(key): shard[key] for key in keys}
    return tensors


def split_inkling_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write Inkling native MTP tensors into a standalone drafter folder."""
    source_path = get_model_path(
        source, revision=revision, force_download=force_download
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(source_path / "config.json") as f:
        source_config = json.load(f)

    text_config = dict(source_config.get("text_config") or {})
    if not text_config:
        raise ValueError(f"{source_path} does not contain a text_config.")

    mtp_config = source_config.get("mtp_config") or {}
    text_config.setdefault(
        "num_mtp_layers",
        mtp_config.get("num_nextn_predict_layers")
        or text_config.get("num_nextn_predict_layers"),
    )
    text_config.setdefault(
        "mtp_local_layer_ids",
        mtp_config.get("local_layer_ids") or text_config.get("local_layer_ids"),
    )

    selected = {}
    source_is_mlx = False
    for file, keys in _iter_keys(source_path):
        source_is_mlx = source_is_mlx or _is_mlx_safetensors(file)
        selected.update(_load_selected_tensors(file, keys))

    if not selected:
        raise ValueError(f"No {_MTP_PREFIX}* tensors found in {source_path}.")

    if not source_is_mlx:
        selected = InklingMTPDraftModel.sanitize(None, selected)

    mx.save_safetensors(
        str(output_path / "model.safetensors"),
        selected,
        metadata={"format": "mlx"},
    )

    depth = (
        text_config.get("num_mtp_layers")
        or text_config.get("num_nextn_predict_layers")
        or 1
    )
    draft_config = {
        "model_type": "inkling_mtp",
        "text_config": text_config,
        "num_mtp_layers": int(depth),
        "mtp_local_layer_ids": text_config.get("mtp_local_layer_ids"),
        "block_size": int(block_size or int(depth) + 2),
        "tie_word_embeddings": bool(text_config.get("tie_word_embeddings", False)),
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(dict(sorted(draft_config.items())), f, indent=2)

    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        src = source_path / name
        if src.exists():
            shutil.copy(src, output_path / name)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split Inkling native MTP tensors into a standalone MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_inkling_mtp(**vars(args))
    print(f"Wrote Inkling MTP drafter to {output}")


if __name__ == "__main__":
    main()
