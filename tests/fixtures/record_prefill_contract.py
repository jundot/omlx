"""Record a model family's Prefill Accelerator contract fixture.

A fixture holds a tiny checkpoint, one split prompt and every array that
crossed the link. tests/test_prefill_helper.py checks that oMLX still sends
those arrays; a helper built from other code (apps/omlx-cuda-helper) loads the
checkpoint and must reproduce them.

  python tests/fixtures/record_prefill_contract.py <family> [--force]

Record again after raising the family adapter's ``contract``; --force replaces
a fixture of the same contract version.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from tokenizers import Tokenizer, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import test_prefill_helper as cases  # noqa: E402

# DeepSeek V4.1 field -> its name in a checkpoint's text_config.
_DEEPSEEK_V41_NAMES = {
    "dim": "hidden_size",
    "moe_inter_dim": "moe_intermediate_size",
    "n_layers": "num_hidden_layers",
    "n_heads": "num_attention_heads",
    "n_activated_experts": "num_experts_per_tok",
    "score_func": "scoring_func",
    "route_scale": "routed_scaling_factor",
    "rope_head_dim": "qk_rope_head_dim",
    "norm_eps": "rms_norm_eps",
    "window_size": "sliding_window",
    "kv_source_layers": "kv_source_layer_ids",
    "index_source_layers": "index_source_layer_ids",
    "candidate_source_layer": "candidate_source_layer_id",
    "engram_pad_id": "engram_pad_token_id",
    "n_mtp_layers": "num_nextn_predict_layers",
    "max_seq_len": "max_position_embeddings",
}
# Settings of a loaded model that a checkpoint does not store.
_DEEPSEEK_V41_RUNTIME = (
    "model_type",
    "preserve_mtp",
    "ced_prefill",
    "a8_prefill",
    "a8_prefill_min_tokens",
)


def _deepseek_v41_config(lm):
    """config.json of a converted checkpoint without quantized modules."""
    text = asdict(lm._config)
    for name in _DEEPSEEK_V41_RUNTIME:
        del text[name]
    text["rope_scaling"] = {
        "factor": text.pop("rope_factor"),
        "beta_fast": text.pop("beta_fast"),
        "beta_slow": text.pop("beta_slow"),
        "original_max_position_embeddings": text.pop("original_seq_len"),
    }
    return {
        "model_type": "deepseek_v41",
        "text_config": {_DEEPSEEK_V41_NAMES.get(k, k): v for k, v in text.items()},
        "omlx_deepseek_v41": {
            "version": 1,
            "quantized_modules": {},
            "engram_tables": {},
        },
    }


def _identity_tokenizer(size):
    """tokenizer.json whose Engram compressed ids equal the token ids."""
    vocabulary = {f"t{i}": i for i in range(size)}
    return Tokenizer(models.WordLevel(vocabulary, unk_token="t0")).to_str()


# family -> the tiny model's config.json, in the checkpoint's layout.
CONFIGS = {
    "qwen4_exp": lambda lm: {
        "model_type": "qwen4_exp",
        "text_config": asdict(lm.model.args),
    },
    "minimax_m3_vl": lambda lm: {
        "model_type": "minimax_m3_vl",
        "text_config": asdict(lm.args),
    },
    "glm5_next": lambda lm: {
        "model_type": "glm5_next",
        "text_config": asdict(lm.args),
    },
    "qwen3_5": lambda lm: {
        "model_type": "qwen3_5",
        "text_config": {
            **asdict(lm.args),
            "layer_types": [
                "linear_attention" if layer.is_linear else "full_attention"
                for layer in lm.model.layers
            ],
        },
    },
    "deepseek_v41": _deepseek_v41_config,
}
# family -> other files of the checkpoint folder a helper reads (name -> text).
SIDECARS = {
    "deepseek_v41": lambda lm: {
        "tokenizer.json": _identity_tokenizer(lm._config.vocab_size)
    },
}


def _meta_text(meta: dict) -> str:
    """JSON with one line per key and one line per manifest entry."""
    lines = [
        f"  {json.dumps(k)}: {json.dumps(v)}"
        for k, v in meta.items()
        if k != "manifest"
    ]
    entries = ",\n".join(f"    {json.dumps(entry)}" for entry in meta["manifest"])
    lines.append(f'  "manifest": [\n{entries}\n  ]')
    return "{\n" + ",\n".join(lines) + "\n}\n"


def record(family: str, force: bool) -> None:
    adapter, _, k, prefix = cases._CONTRACT_CASES[family]
    folder = cases._CONTRACT_DIR / family
    meta_path = folder / "meta.json"
    if meta_path.exists() and not force:
        recorded = json.loads(meta_path.read_text())["contract"]
        if adapter.contract <= recorded:
            sys.exit(
                f"{family} already has a fixture for contract {recorded}; raise "
                "the adapter's contract or pass --force"
            )
    host, ids, arrays = cases._contract_case(family)
    (folder / "model").mkdir(parents=True, exist_ok=True)
    weights = {prefix + n: v for n, v in tree_flatten(host.parameters())}
    mx.save_safetensors(str(folder / "model" / "model.safetensors"), weights)
    config = CONFIGS[family](host)
    (folder / "model" / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name, text in SIDECARS.get(family, lambda lm: {})(host).items():
        (folder / "model" / name).write_text(text)
    np.savez(folder / "case.npz", **{name: data for name, (data, _) in arrays.items()})
    meta = {
        "family": family,
        "contract": adapter.contract,
        "layers": k,
        "model_layers": adapter.layer_count(host),
        "ids": ids.reshape(-1).tolist(),
        "sizes": cases._CONTRACT_SIZES,
        "prefix_chunks": cases._CONTRACT_PREFIX,
        # Float arrays are compared within this tolerance, the rest exactly.
        "rtol": 1e-4,
        "atol": 1e-5,
        "manifest": cases._contract_manifest(arrays),
    }
    meta_path.write_text(_meta_text(meta))
    print(f"recorded {family} contract {adapter.contract}: {len(arrays)} arrays")


if __name__ == "__main__":
    record(sys.argv[1], "--force" in sys.argv[2:])
