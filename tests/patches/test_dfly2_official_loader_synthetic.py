import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from omlx.patches.deepseek_v41_dfly2.official_loader import OfficialDFly2Checkpoint


def test_loader_reads_only_indexed_selected_synthetic_tensors(tmp_path):
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    key_a = "model.layers.0.attn.wq_a.weight"
    key_b = "model.mtp_fusion.eh_proj.weight"
    save_file({key_a: np.arange(6, dtype=np.float32).reshape(2, 3)}, tmp_path / first)
    save_file({key_b: np.array([[-1.0, 0.5]], dtype=np.float32)}, tmp_path / second)

    checkpoint = OfficialDFly2Checkpoint(
        config={"model_type": "deepseek_v41", "dfly_config": {"block_size": 6}},
        weight_map={key_a: first, key_b: second},
    )
    plan = checkpoint.plan_for_prefixes(("model.layers.0.attn.wq_a.", "model.mtp_fusion."))
    exact = checkpoint.plan_for_tensor_names((key_b, key_a))
    loaded = checkpoint.load_available_shards(tmp_path, plan)

    assert plan.shards == (first, second)
    assert exact.tensor_names == (key_a, key_b)
    assert exact.shards == (first, second)
    np.testing.assert_array_equal(loaded.tensors[key_a], np.arange(6, dtype=np.float32).reshape(2, 3))
    np.testing.assert_array_equal(loaded.tensors[key_b], np.array([[-1.0, 0.5]], dtype=np.float32))
