# SPDX-License-Identifier: Apache-2.0
"""Source-ref resolution for the dashboard branch badge."""

from __future__ import annotations

import subprocess

import pytest

from omlx import build_ref as build_ref_module
from omlx.build_ref import _clean, _from_git, _from_stamp, build_ref


@pytest.fixture(autouse=True)
def _clear_cache():
    build_ref.cache_clear()
    yield
    build_ref.cache_clear()


def _git_repo(path, branch="test-branch"):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", branch, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "f.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "init"], check=True)
    return path


# ----------------------------------------------------------------- _clean --


def test_clean_rejects_empty_and_none():
    assert _clean(None) is None
    assert _clean("") is None
    assert _clean("   \n  ") is None


def test_clean_rejects_bare_head_so_detached_falls_through():
    assert _clean("HEAD\n") is None


def test_clean_rejects_control_characters():
    assert _clean("main\x07evil") is None


def test_clean_takes_first_line_and_strips():
    assert _clean("  local-develop \nnoise\n") == "local-develop"


def test_clean_truncates_long_refs():
    assert _clean("b" * 200) == "b" * 64


# ------------------------------------------------------------- precedence --


def test_env_var_wins_over_stamp_and_git(tmp_path, monkeypatch):
    stamp = tmp_path / "_build_ref.txt"
    stamp.write_text("from-stamp", encoding="utf-8")
    monkeypatch.setattr(build_ref_module, "_STAMP", stamp)
    monkeypatch.setattr(build_ref_module, "_REPO_ROOT", _git_repo(tmp_path / "repo"))
    monkeypatch.setenv("OMLX_BUILD_REF", "from-env")

    assert build_ref() == "from-env"


def test_stamp_wins_over_git_when_env_unset(tmp_path, monkeypatch):
    stamp = tmp_path / "_build_ref.txt"
    stamp.write_text("from-stamp\n", encoding="utf-8")
    monkeypatch.setattr(build_ref_module, "_STAMP", stamp)
    monkeypatch.setattr(build_ref_module, "_REPO_ROOT", _git_repo(tmp_path / "repo"))
    monkeypatch.delenv("OMLX_BUILD_REF", raising=False)

    assert build_ref() == "from-stamp"


def test_git_used_when_env_and_stamp_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(build_ref_module, "_STAMP", tmp_path / "missing.txt")
    monkeypatch.setattr(build_ref_module, "_REPO_ROOT", _git_repo(tmp_path / "repo"))
    monkeypatch.delenv("OMLX_BUILD_REF", raising=False)

    assert build_ref() == "test-branch"


def test_returns_none_when_nothing_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(build_ref_module, "_STAMP", tmp_path / "missing.txt")
    monkeypatch.setattr(build_ref_module, "_REPO_ROOT", tmp_path / "not-a-repo")
    monkeypatch.delenv("OMLX_BUILD_REF", raising=False)

    assert build_ref() is None


# ---------------------------------------------------------------- sources --


def test_from_stamp_survives_an_unreadable_path(tmp_path):
    assert _from_stamp(tmp_path / "nope.txt") is None


def test_from_git_returns_none_outside_a_repo(tmp_path):
    assert _from_git(tmp_path) is None


def test_from_git_reports_the_branch_name(tmp_path):
    repo = _git_repo(tmp_path / "repo", branch="local-develop")
    assert _from_git(repo) == "local-develop"


def test_from_git_reports_a_tag_checkout_as_the_short_sha(tmp_path):
    repo = _git_repo(tmp_path / "repo")
    subprocess.run(["git", "-C", str(repo), "tag", "v9.9.9"], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "v9.9.9"], check=True)

    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # Detached HEAD: --abbrev-ref prints "HEAD", so the short SHA is returned.
    assert _from_git(repo) == head


def test_build_ref_is_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(build_ref_module, "_STAMP", tmp_path / "missing.txt")
    monkeypatch.setattr(build_ref_module, "_REPO_ROOT", tmp_path / "not-a-repo")
    monkeypatch.setenv("OMLX_BUILD_REF", "first")
    assert build_ref() == "first"

    monkeypatch.setenv("OMLX_BUILD_REF", "second")
    assert build_ref() == "first"

    build_ref.cache_clear()
    assert build_ref() == "second"
