# SPDX-License-Identifier: Apache-2.0
"""oQ must give trust_remote_code to the tokenizer, not only to the model.

A checkpoint whose tokenizer needs custom code (a tokenization_*.py with no
tokenizer.json) loads the model but not the tokenizer when only the model call
carries the flag. transformers then falls back to a base config and raises
``'PreTrainedConfig' object has no attribute 'max_position_embeddings'``, which
oQ reports as "sensitivity measurement produced no scores".
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import omlx.oq as oq

CONFIG = {"model_type": "llama", "num_hidden_layers": 2}


@pytest.fixture
def loader(monkeypatch):
    """Stub the pre-load helpers so only the loader call is exercised."""
    import omlx.utils.model_loading as model_loading

    monkeypatch.setattr(
        model_loading, "_checkpoint_has_mtp_weights", lambda *a, **k: False
    )
    monkeypatch.setattr(model_loading, "_has_mtp_heads", lambda *a, **k: False)
    monkeypatch.setattr(
        model_loading, "maybe_apply_pre_load_patches", lambda *a, **k: None
    )
    monkeypatch.setattr(oq, "_calibration_model_settings", lambda *a, **k: None)
    monkeypatch.setattr(oq, "_measure_sensitivity_from_model", lambda *a, **k: {0: 1.0})
    monkeypatch.setattr(oq, "_is_vlm_load", lambda *a, **k: False)

    lm_load = MagicMock(return_value=(MagicMock(), MagicMock()))
    monkeypatch.setattr(model_loading, "lm_load_compat", lm_load)
    return lm_load


@pytest.mark.parametrize("trust", [True, False])
def test_sensitivity_load_forwards_trust_to_tokenizer(loader, trust):
    oq._measure_sensitivity("/models/some-model", CONFIG, 4, trust_remote_code=trust)

    assert loader.call_count == 1
    kwargs = loader.call_args.kwargs
    assert kwargs["trust_remote_code"] is trust
    assert kwargs["tokenizer_config"] == {"trust_remote_code": trust}


def test_sensitivity_returns_no_scores_when_tokenizer_rejects_custom_code(loader):
    """The failure mode this fix addresses, reported as an empty score map."""
    loader.side_effect = AttributeError(
        "'PreTrainedConfig' object has no attribute 'max_position_embeddings'"
    )

    assert oq._measure_sensitivity("/models/some-model", CONFIG, 4) == {}


def _tokenizer_call_sites():
    """Every call in oq.py that builds a tokenizer, with its keywords."""
    tree = ast.parse(Path(oq.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in ("lm_load", "load_tokenizer"):
            continue
        yield node


def test_every_tokenizer_call_site_passes_trust_remote_code():
    """Guard the call sites the unit tests above do not reach.

    oQ builds tokenizers in the VLM branches and in the streaming imatrix and
    sensitivity paths too. Each of those functions already has
    trust_remote_code in scope, so a new site that drops it is a regression.
    """
    missing = []
    for node in _tokenizer_call_sites():
        keywords = {k.arg for k in node.keywords}
        # load_tokenizer takes the tokenizer config positionally
        has_config = "tokenizer_config" in keywords or (
            node.func.id == "load_tokenizer" and len(node.args) >= 2
        )
        if not has_config:
            missing.append(f"{node.func.id} at line {node.lineno}")

    assert not missing, "tokenizer built without trust_remote_code: " + ", ".join(
        missing
    )
