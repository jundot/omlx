# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Homebrew formula and its release automation.

The release workflow must update only the formula's top-level source URL and
checksum. Resource blocks have independent checksums that must survive version
bumps (issues #2151 and #2173).

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
- Later pip steps (mlx-audio, python-multipart) ran without --no-binary,
  so a prebuilt wheel could clobber a source-built package, and pip's
  wheel cache could resurrect a dylib built before the strip guards.

The formula and workflow use Ruby and shell syntax, so these are text-level
assertions that the guards stay present.
"""

from pathlib import Path

import pytest

FORMULA_PATH = Path(__file__).resolve().parents[1] / "Formula" / "omlx.rb"
WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "update-formula.yml"
)

MACOS_27_GUARD = 'MacOS.version >= "27"'
SPACY_MODEL_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"


@pytest.fixture(scope="module")
def formula() -> str:
    return FORMULA_PATH.read_text()


@pytest.fixture(scope="module")
def formula_update_workflow() -> str:
    return WORKFLOW_PATH.read_text()


class TestFormulaReleaseUpdate:
    def test_source_sha_update_is_scoped_to_top_level(self, formula_update_workflow):
        """Release bumps must not replace checksums inside resource blocks."""
        sha_update = next(
            line.strip()
            for line in formula_update_workflow.splitlines()
            if line.lstrip().startswith("sed -i") and "steps.sha.outputs.sha256" in line
        )

        assert "s|^  sha256" in sha_update
        assert '"$|  sha256' in sha_update

    def test_spacy_model_checksum_is_independent(self, formula):
        """The bundled spaCy wheel checksum must survive source version bumps."""
        resource_start = formula.index('resource "en-core-web-sm" do')
        resource_end = formula.index("\n  end", resource_start)
        resource_block = formula[resource_start:resource_end]

        assert f'sha256 "{SPACY_MODEL_SHA256}"' in resource_block


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
        assert "on_macos do" in formula
        assert f'skip_clean "libexec" if {MACOS_27_GUARD}' in formula

    def test_pip_cache_bypassed_on_macos_27(self, formula):
        """Pip reuses locally built wheels even under --no-binary, so a
        wheel cached before the strip guards existed stays corrupted."""
        assert 'pip_flags << "--no-cache-dir"' in formula


class TestSharedPipFlags:
    def test_shared_pip_install_array(self, formula):
        """All pip steps must share the --no-binary/--no-cache-dir flags."""
        assert (
            'pip_install = [libexec/"bin/pip", "install", *pip_flags,'
            ' "--no-binary", no_binary]' in formula
        )
        assert "system(*pip_install, install_spec)" in formula

    def test_audio_extra_is_the_only_mlx_audio_source(self, formula):
        """spaCy for the model check comes only from omlx's [audio] extra, and a
        separate mlx-audio resource pins a second revision that drifts (#4942)."""
        assert '"#{buildpath}[audio,grammar]" : "#{buildpath}[audio]"' in formula
        assert 'resource "mlx-audio"' not in formula

    def test_no_bare_pip_install_besides_spacy_wheel(self, formula):
        """The only direct pip invocation is the --no-deps local spaCy model
        wheel; any new bare `pip install` would bypass the shared flags."""
        assert formula.count('bin/pip", "install"') == 2
        assert 'system libexec/"bin/pip", "install", "--no-deps"' in formula


class TestCustomKernelBuild:
    def test_formula_covers_every_native_kernel_package(self, formula):
        """The formula on main also builds the current release tarball, so it
        reads the package list from the installed oMLX instead of a copy."""
        assert "CUSTOM_KERNELS" not in formula
        assert 'glob("*/csrc")' in formula
        assert "from omlx.custom_kernels import native_kernel_status" in formula

    def test_cmake_pinned_to_venv_python(self, formula):
        """CMake must not discover a stray system Python for kernel builds."""
        assert "-DPython_EXECUTABLE=#{libexec}/bin/python" in formula

    def test_kernel_verification_not_shadowed_by_buildpath(self, formula):
        """Import check must run outside buildpath's raw omlx/ source tree."""
        assert "Dir.chdir(libexec)" in formula

    def test_metal_toolchain_checked_before_kernel_build(self, formula):
        block = formula.index('if build.with?("custom-kernel")')
        guard = formula.index('quiet_system("/usr/bin/xcrun", "metal", "--version")', block)
        assert guard < formula.index('ENV["OMLX_WITH_CUSTOM_KERNEL"]', block)

    def test_metal_guard_executes_the_compiler(self, formula):
        """`xcrun -f metal` succeeds whenever the shim exists, which it does even
        when the Metal toolchain behind it cannot run, so it cannot be the guard."""
        assert 'quiet_system("/usr/bin/xcrun", "-f", "metal")' not in formula
