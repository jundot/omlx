# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the Homebrew formula's macOS 27 beta workarounds.

macOS 27 betas broke `brew install omlx` in several ways (issue #2110):

- dyld now requires the LC_SYMTAB string pool in Mach-O libraries to be
  8-byte aligned, so prebuilt Rust wheels (e.g. tokenizers) fail dlopen.
- The beta `strip` binary corrupts dynamic offsets in Mach-O libraries
  (llvm/llvm-project#203678), so Cargo/maturin release stripping and
  Homebrew's post-install clean pass must be kept away from the dylibs.
- CMake's default Python discovery can pick a newer unlinked system
  Python instead of the formula's venv when building custom kernels.
- The custom-kernel verification ran from the build directory, where the
  raw omlx/ source tree shadows the installed package.

The formula is Ruby, so these are text-level assertions that the guards
stay present in Formula/omlx.rb.
"""

from pathlib import Path

import pytest

FORMULA_PATH = Path(__file__).resolve().parents[1] / "Formula" / "omlx.rb"

MACOS_27_GUARD = 'MacOS.version >= "27"'


@pytest.fixture(scope="module")
def formula() -> str:
    return FORMULA_PATH.read_text()


class TestMacOS27Workarounds:
    def test_tokenizers_built_from_source_on_macos_27(self, formula):
        """Rust wheels with 4-byte-aligned LINKEDIT must be rebuilt natively."""
        assert MACOS_27_GUARD in formula
        assert 'no_binary += ",tokenizers"' in formula

    def test_base_no_binary_list_unconditional(self, formula):
        """Older macOS keeps the existing source-build list unchanged."""
        assert 'no_binary = "cohere_melody,pydantic-core,rpds-py,tiktoken"' in formula
        assert '"--no-binary", no_binary' in formula

    def test_release_stripping_disabled_on_macos_27(self, formula):
        """The beta strip binary corrupts dylibs; Cargo/maturin must not strip."""
        assert 'ENV["CARGO_PROFILE_RELEASE_STRIP"] = "false"' in formula
        assert 'ENV["MATURIN_STRIP"] = "false"' in formula

    def test_homebrew_clean_pass_skipped_on_macos_27(self, formula):
        """Homebrew's clean pass also runs strip over the venv's dylibs."""
        assert f'skip_clean "libexec" if {MACOS_27_GUARD}' in formula


class TestCustomKernelBuild:
    def test_cmake_pinned_to_venv_python(self, formula):
        """CMake must not discover a stray system Python for kernel builds."""
        assert '-DPython_EXECUTABLE=#{libexec}/bin/python' in formula

    def test_kernel_verification_not_shadowed_by_buildpath(self, formula):
        """Import check must run outside buildpath's raw omlx/ source tree."""
        assert "Dir.chdir(libexec)" in formula
