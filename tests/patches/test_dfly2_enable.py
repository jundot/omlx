"""Loading an official checkpoint on disk into a live draft runtime."""

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.deepseek_v41_dfly2.enable import enable_dfly2, load_checkpoint_tensors

DIM = 32
HEAD_DIM = 16
HEADS = 2
Q_LORA = 16
O_GROUPS = 2
O_LORA = 8
INTER = 16
EXPERTS = 4
DRAFT_LAYERS = (28, 32, 36)


def _config():
    return SimpleNamespace(
        dim=DIM,
        n_heads=HEADS,
        head_dim=HEAD_DIM,
        rope_head_dim=8,
        q_lora_rank=Q_LORA,
        o_groups=O_GROUPS,
        o_lora_rank=O_LORA,
        moe_intermediate_size=INTER,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
        hc_mult=1,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        hc_eps=1e-6,
        hc_sinkhorn_iters=4,
        route_norm=True,
        router_scaling_factor=1.5,
        moe_router_score_func="sqrtsoftplus",
        swiglu_limit=0.25,
    )


def _stage(prefix: str, rng) -> dict[str, mx.array]:
    def r(*shape):
        return mx.array(rng.standard_normal(shape).astype("float32") * 0.1)

    tensors = {
        f"{prefix}attn.wq_a.weight": r(Q_LORA, DIM),
        f"{prefix}attn.q_norm.weight": mx.ones((Q_LORA,)),
        f"{prefix}attn.wq_b.weight": r(HEADS * HEAD_DIM, Q_LORA),
        f"{prefix}attn.wkv.weight": r(HEAD_DIM, DIM),
        f"{prefix}attn.kv_norm.weight": mx.ones((HEAD_DIM,)),
        f"{prefix}attn.wo_a.weight": r(O_GROUPS * O_LORA, HEADS * HEAD_DIM // O_GROUPS),
        f"{prefix}attn.wo_b.weight": r(DIM, O_GROUPS * O_LORA),
        f"{prefix}attn.attn_sink": mx.zeros((HEADS,)),
        f"{prefix}attn_norm.weight": mx.ones((DIM,)),
        f"{prefix}ffn_norm.weight": mx.ones((DIM,)),
        f"{prefix}ffn.gate.weight": r(EXPERTS, DIM),
        f"{prefix}ffn.gate.bias": mx.zeros((EXPERTS,)),
        f"{prefix}ffn.gate.bias_vl": mx.zeros((EXPERTS,)),
    }
    for expert in range(EXPERTS):
        tensors[f"{prefix}ffn.experts.{expert}.w1.weight"] = r(INTER, DIM)
        tensors[f"{prefix}ffn.experts.{expert}.w2.weight"] = r(DIM, INTER)
        tensors[f"{prefix}ffn.experts.{expert}.w3.weight"] = r(INTER, DIM)
    for name in ("w1", "w2", "w3"):
        shape = (INTER, DIM) if name in ("w1", "w3") else (DIM, INTER)
        tensors[f"{prefix}ffn.shared_experts.{name}.weight"] = r(*shape)
    return tensors


def _checkpoint(tmp_path: Path) -> Path:
    import numpy as np

    rng = np.random.default_rng(0)
    tensors = {}
    for layer in range(len(DRAFT_LAYERS)):
        tensors.update(_stage(f"model.layers.{layer}.", rng))
    tensors["model.mtp_fusion.enorm.weight"] = mx.ones((DIM,))
    tensors["model.mtp_fusion.hnorm.weight"] = mx.ones((DIM,))
    tensors["model.mtp_fusion.eh_proj.weight"] = mx.array(rng.standard_normal((DIM, 2 * DIM)).astype("float32") * 0.1)
    tensors["model.mtp_fusion.gate"] = mx.array([0.5])
    tensors["model.hidden_correction.hidden_norm.weight"] = mx.ones((DIM,))
    tensors["model.hidden_correction.embed_norm.weight"] = mx.ones((DIM,))
    tensors["model.hidden_correction.gate_proj.weight"] = mx.array(rng.standard_normal((INTER, 2 * DIM)).astype("float32") * 0.1)
    tensors["model.hidden_correction.up_proj.weight"] = mx.array(rng.standard_normal((INTER, 2 * DIM)).astype("float32") * 0.1)
    tensors["model.hidden_correction.down_proj.weight"] = mx.array(rng.standard_normal((DIM, INTER)).astype("float32") * 0.1)
    tensors["model.final_norm.weight"] = mx.ones((DIM,))

    shard = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(tmp_path / shard), tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "dfly_config": {
                    "block_size": 6,
                    "mask_token_id": 128799,
                    "index_topk": 512,
                    "target_kv_layers": list(DRAFT_LAYERS),
                    "target_index_layers": list(DRAFT_LAYERS),
                },
            }
        )
    )
    return tmp_path


def test_checkpoint_loads_every_indexed_tensor(tmp_path):
    directory = _checkpoint(tmp_path)
    tensors, config = load_checkpoint_tensors(directory)
    assert config["model_type"] == "deepseek_v41"
    assert f"model.layers.{len(DRAFT_LAYERS) - 1}.ffn.shared_experts.w2.weight" in tensors
    assert "model.hidden_correction.down_proj.weight" in tensors
    assert all(not name.startswith("model.layers.3.") for name in tensors)


def test_enable_builds_a_runtime_that_drafts_a_block(tmp_path):
    directory = _checkpoint(tmp_path)

    class Host:
        _config = _config()

        def __init__(self):
            self.embed = lambda ids: mx.zeros(ids.shape + (DIM,))

    host = Host()
    runtime = enable_dfly2(host, directory)
    assert host.dfly is runtime
    assert len(runtime.layers) == len(DRAFT_LAYERS)

    history = [
        SimpleNamespace(
            latents=mx.zeros((1, 3, HEAD_DIM)),
            valid=mx.ones((1, 3), dtype=mx.bool_),
        )
        for _ in DRAFT_LAYERS
    ]
    block = runtime.block_hidden(
        host.embed, mx.zeros((1, 1, DIM)), mx.array([7], dtype=mx.int32), 3, history, 6
    )
    mx.eval(block)
    assert block.shape == (1, 6, DIM)
    assert bool(mx.all(mx.isfinite(block)))


def test_missing_tensor_fails_at_load_not_at_first_forward(tmp_path):
    directory = _checkpoint(tmp_path)
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].pop("model.final_norm.weight")
    index_path.write_text(json.dumps(index))
    with pytest.raises(KeyError, match="final_norm"):
        load_checkpoint_tensors(directory)
