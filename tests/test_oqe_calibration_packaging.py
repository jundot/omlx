# SPDX-License-Identifier: Apache-2.0
"""Regression tests for shipping the oQe imatrix calibration corpus.

`enhanced=True` (oQe) quantization calibrates its importance matrix on the
built-in ``oqe_calibration_data.json`` corpus. That file lives in the ``omlx``
package but was missing from ``[tool.setuptools.package-data]``, so every wheel
build dropped it and oQe silently fell back to a different, smaller corpus.
These tests fail if the corpus stops being declared or stops being importable.
"""

import json
import tomllib
from importlib.resources import files
from pathlib import Path

import pytest

# Keys the oQe calibration path reads out of the corpus (omlx.oq).
_EXPECTED_KEYS = {
    "tool_calling",
    "chat",
    "mixed",
    "reasoning",
    "code",
    "en",
    "ko",
    "zh",
    "ja",
    "bartowski",
}


def _pyproject_path() -> Path:
    return Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.mark.skipif(
    not _pyproject_path().is_file(),
    reason="pyproject.toml not available (installed without source tree)",
)
def test_oqe_corpus_declared_in_package_data():
    data = tomllib.loads(_pyproject_path().read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]["omlx"]
    assert "oqe_calibration_data.json" in package_data, (
        "oqe_calibration_data.json must be in [tool.setuptools.package-data] "
        "so it ships in the wheel; otherwise oQe silently mis-calibrates."
    )


def test_oqe_corpus_shipped_and_loadable():
    resource = files("omlx").joinpath("oqe_calibration_data.json")
    assert resource.is_file(), (
        "oqe_calibration_data.json is not present in the installed omlx "
        "package; enhanced quantization cannot calibrate correctly."
    )
    corpus = json.loads(resource.read_text(encoding="utf-8"))
    assert isinstance(corpus, dict)
    present = _EXPECTED_KEYS & set(corpus)
    assert present, f"oQe corpus contains none of the expected keys: {_EXPECTED_KEYS}"


def test_oq_corpus_still_shipped():
    resource = files("omlx").joinpath("oq_calibration_data.json")
    assert resource.is_file()


def test_missing_oqe_corpus_raises_instead_of_silent_fallback(monkeypatch):
    """When the oQe corpus is absent, calibration must fail loudly rather than
    silently fall back to a different corpus and mis-calibrate the imatrix."""
    import omlx.oq as oq

    real_exists = Path.exists

    def fake_exists(self):
        if self.name == "oqe_calibration_data.json":
            return False
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    with pytest.raises(FileNotFoundError, match="oQe calibration corpus"):
        oq._load_calibration_data(tokenizer=None, dataset=oq._OQE_CALIB_DATASET)
