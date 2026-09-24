"""Adapter boundaries keep memory planning portable and fail closed."""

import subprocess
import sys
from pathlib import Path

import pytest

from omlx.prefill.models import model_geometry_from_args


@pytest.mark.parametrize("model_type", [None, [], {}, 1, "unregistered_model"])
def test_unknown_argument_schema_has_no_memory_estimate(model_type):
    assert model_geometry_from_args({"model_type": model_type}) is None


def test_geometry_and_memory_planning_do_not_import_gpu_runtime():
    code = """
import importlib.abc
import sys
class NoGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mlx', 'mlx_lm'):
            raise AssertionError('Planning imported GPU runtime: ' + fullname)
sys.meta_path.insert(0, NoGPU())
from omlx.prefill.models import model_geometry_from_args
from omlx.prefill.memory import estimate_batched_prefill_memory
geometry = model_geometry_from_args(dict(
    model_type='hy_v3', num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=8, hidden_size=32, intermediate_size=64,
    vocab_size=128, num_experts=4, num_experts_per_tok=2,
    expert_hidden_dim=32, num_shared_experts=1, first_k_dense_replace=1,
))
assert geometry is not None
cost = estimate_batched_prefill_memory(geometry, batch_size=2, query_tokens=8, max_prompt_tokens=16)
assert cost.additional_peak_bytes > 0
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        timeout=30,
    )
