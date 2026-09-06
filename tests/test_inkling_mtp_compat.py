"""Regression tests for the vendored inkling_mtp drafter compat layer.

oMLX's mlx-vlm pin (78b96eb) predates the ``inkling_mtp`` drafter, so an
Inkling drafter checkpoint resolves to ``dflash`` and then fails to import.
These tests pin the two discovery surfaces the compat layer installs.
"""

from pathlib import Path


def _compat_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "omlx/patches/mlx_vlm_inkling_mtp_compat"


def test_vendored_drafter_package_is_present():
    pkg = _compat_dir() / "vendor/mlx_vlm/speculative/drafters/inkling_mtp"
    for name in ("__init__.py", "config.py", "inkling_mtp.py"):
        assert (pkg / name).is_file(), f"missing vendored {name}"

    # load_model resolves the class through these two names.
    init = (pkg / "__init__.py").read_text()
    assert "InklingMTPDraftModel as Model" in init
    assert "InklingMTPConfig as ModelConfig" in init


def test_vendor_path_is_appended_not_prepended():
    """A pin bump shipping the drafter upstream must win over the vendor."""
    body = (_compat_dir() / "__init__.py").read_text()
    assert "package_path.append(path_str)" in body
    assert "package_path.insert" not in body


def test_drafter_kind_registration_defers_to_upstream():
    body = (_compat_dir() / "__init__.py").read_text()
    assert 'table.setdefault(_MODEL_TYPE, _DRAFT_KIND)' in body
    assert '_MODEL_TYPE = "inkling_mtp"' in body
    assert '_DRAFT_KIND = "mtp"' in body


def test_model_loading_applies_drafter_patch_unchained():
    """The drafter patch must not hang off the model compat's return value.

    ``apply_mlx_vlm_inkling_compat_patch`` returns False once applied, so
    chaining would skip the drafter patch on every load after the first.
    """
    root = Path(__file__).resolve().parents[1]
    body = (root / "omlx/utils/model_loading.py").read_text()

    assert "apply_mlx_vlm_inkling_mtp_compat_patch" in body
    model_at = body.index("if apply_mlx_vlm_inkling_compat_patch():")
    drafter_at = body.index("if apply_mlx_vlm_inkling_mtp_compat_patch():")
    assert model_at < drafter_at, "drafter patch must apply after the model compat"
