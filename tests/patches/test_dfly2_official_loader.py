import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from omlx.patches.deepseek_v41_dfly2.official_loader import OfficialDFly2Checkpoint, required_prefixes


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "checkpoint_metadata" / "config.json"
INDEX = ROOT / "checkpoint_metadata" / "model.safetensors.index.json"


def test_official_plan_resolves_three_layers_and_correction_shard():
    checkpoint = OfficialDFly2Checkpoint.from_metadata(CONFIG, INDEX)
    plan = checkpoint.plan_for_prefixes(required_prefixes(layer_ids=(0,), include_fusion=True))
    assert plan.shards == (
        "model-00001-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    )
    assert "model.layers.0.attn.wq_a.weight" in plan.tensor_names
    assert "model.mtp_fusion.eh_proj.weight" in plan.tensor_names
    assert plan.missing_prefixes == ()


def test_checkpoint_rejects_missing_required_prefix():
    checkpoint = OfficialDFly2Checkpoint.from_metadata(CONFIG, INDEX)
    with pytest.raises(KeyError, match="not present"):
        checkpoint.plan_for_prefixes(("model.no_such_module.",))


def test_checkpoint_reports_unavailable_shards_before_loading_weights(tmp_path):
    checkpoint = OfficialDFly2Checkpoint.from_metadata(CONFIG, INDEX)
    plan = checkpoint.plan_for_prefixes(required_prefixes(layer_ids=(0,), include_fusion=True))
    with pytest.raises(FileNotFoundError, match="missing official shard"):
        checkpoint.load_available_shards(tmp_path, plan)
