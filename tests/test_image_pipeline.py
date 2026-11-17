# SPDX-License-Identifier: Apache-2.0
import threading

import numpy as np
import pytest

from omlx.image.pipeline import SIGMAS, generate_png, initial_noise


def test_fixed_recipe_noise_is_local_and_reproducible():
    before = np.random.get_state()
    a = initial_noise(42)
    np.testing.assert_array_equal(a, initial_noise(42))
    assert not np.array_equal(a, initial_noise(43))
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert a.shape == (1, 4096, 64) and a.dtype == np.float32
    assert len(SIGMAS) == 9 and all(a > b for a, b in zip(SIGMAS, SIGMAS[1:]))
    assert SIGMAS[0] == 1 and SIGMAS[-1] == 0


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_seed_validation(seed):
    with pytest.raises(ValueError):
        initial_noise(seed)


def test_pre_cancelled_pipeline_rejects_before_encoding(tmp_path, monkeypatch):
    import omlx.image.pipeline as module

    event = threading.Event()
    event.set()
    monkeypatch.setattr(module, "native_bundle", lambda root: root)
    monkeypatch.setattr(
        module, "inspect_bundle", lambda root: {"structural_status": "valid"}
    )
    monkeypatch.setattr(
        module, "encode_prompt_steps", lambda *a, **k: pytest.fail("encoding started")
    )
    with pytest.raises(InterruptedError):
        generate_png(tmp_path, "prompt", 42, cancel=event)


def test_encoder_failure_still_drains(tmp_path, monkeypatch):
    import omlx.image.pipeline as module

    calls = []
    monkeypatch.setattr(module, "native_bundle", lambda root: root)
    monkeypatch.setattr(
        module, "inspect_bundle", lambda root: {"structural_status": "valid"}
    )
    monkeypatch.setattr(module, "release_gpu", lambda: calls.append("drain"))

    def fail(*a, **k):
        raise ValueError("bad encoder")

    monkeypatch.setattr(module, "encode_prompt_steps", fail)
    with pytest.raises(ValueError, match="encoder"):
        generate_png(tmp_path, "prompt", 42)
    assert calls == ["drain"]
