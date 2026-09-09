# SPDX-License-Identifier: Apache-2.0
"""Tests for the git pin drift startup diagnostic."""

import json

from omlx.utils.pin_drift import PinDrift, find_git_pin_drift, log_git_pin_drift

PINNED = "78b96eb5462141447b9a6b4943ef553891da56dd"
STALE = "086ab9d5d575fec64d8d8ad907ce000007c25c1a"
VLM_REQ = f"mlx-vlm@ git+https://github.com/Blaizzy/mlx-vlm@{PINNED}"


def _vcs_json(commit):
    return json.dumps(
        {
            "url": "https://github.com/Blaizzy/mlx-vlm",
            "vcs_info": {"commit_id": commit, "vcs": "git"},
        }
    )


class TestFindGitPinDrift:
    def test_matching_commit_reports_nothing(self):
        drift = find_git_pin_drift([VLM_REQ], lambda name: _vcs_json(PINNED))
        assert drift == []

    def test_moved_pin_reports_drift(self):
        drift = find_git_pin_drift([VLM_REQ], lambda name: _vcs_json(STALE))
        assert drift == [
            PinDrift(
                name="mlx-vlm",
                requirement=VLM_REQ,
                pinned_commit=PINNED,
                installed_commit=STALE,
            )
        ]

    def test_commit_comparison_is_case_insensitive(self):
        drift = find_git_pin_drift([VLM_REQ], lambda name: _vcs_json(PINNED.upper()))
        assert drift == []

    def test_missing_package_is_skipped(self):
        """Extras that were never installed (e.g. the audio extra) look up
        as None and must not warn."""
        drift = find_git_pin_drift([VLM_REQ], lambda name: None)
        assert drift == []

    def test_index_install_without_direct_url_is_skipped(self):
        # _read_installed_direct_url returns None for index installs; the
        # lambda models that. Nothing to compare, nothing to report.
        drift = find_git_pin_drift([VLM_REQ], lambda name: None)
        assert drift == []

    def test_archive_install_is_skipped(self):
        """Homebrew builds deps from tarball resources; direct_url.json has
        archive_info, not vcs_info. There is no commit to compare, so no
        warning even when the tarball is something else entirely."""
        archive = json.dumps(
            {
                "url": "file:///tmp/mlx-vlm-0.6.3.tar.gz",
                "archive_info": {"hash": "sha256=abc"},
            }
        )
        drift = find_git_pin_drift([VLM_REQ], lambda name: archive)
        assert drift == []

    def test_non_git_requirements_are_ignored(self):
        reqs = ["numpy>=1.24.0,<2.4", "regex", 'mcp>=1.2; extra == "mcp"']
        drift = find_git_pin_drift(reqs, lambda name: _vcs_json(STALE))
        assert drift == []

    def test_branch_pin_without_commit_is_ignored(self):
        req = "mlx-vlm@ git+https://github.com/Blaizzy/mlx-vlm@main"
        drift = find_git_pin_drift([req], lambda name: _vcs_json(STALE))
        assert drift == []

    def test_url_fragment_does_not_hide_the_commit(self):
        req = (
            f"mlx-vlm@ git+https://github.com/Blaizzy/mlx-vlm@{PINNED}"
            "#egg=mlx-vlm"
        )
        drift = find_git_pin_drift([req], lambda name: _vcs_json(STALE))
        assert len(drift) == 1

    def test_extra_marker_pin_is_still_checked_when_installed(self):
        """Markers are not evaluated: an installed audio-extra pin with a
        drifted commit must warn."""
        req = (
            "mlx-audio[tts,stt,sts]@ git+https://github.com/Blaizzy/mlx-audio"
            f"@{PINNED} ; extra == \"audio\""
        )
        drift = find_git_pin_drift([req], lambda name: _vcs_json(STALE))
        assert len(drift) == 1
        assert drift[0].name == "mlx-audio"
        # The requirement string is echoed verbatim so the warning's pip
        # command reinstalls the same extras.
        assert drift[0].requirement == req

    def test_malformed_direct_url_is_tolerated(self):
        for bad in ["not json", "[]", json.dumps({"vcs_info": "nope"})]:
            assert find_git_pin_drift([VLM_REQ], lambda name: bad) == []

    def test_malformed_requirement_is_tolerated(self):
        drift = find_git_pin_drift(
            ["===not-a-requirement===", VLM_REQ],
            lambda name: _vcs_json(STALE),
        )
        assert len(drift) == 1

    def test_missing_commit_id_is_skipped(self):
        text = json.dumps({"vcs_info": {"vcs": "git"}})
        assert find_git_pin_drift([VLM_REQ], lambda name: text) == []


class TestLogGitPinDrift:
    def test_runs_against_real_metadata(self):
        # Whatever this machine's venv looks like, the check must complete
        # and return a list.
        assert isinstance(log_git_pin_drift(), list)

    def test_unknown_distribution_never_raises(self):
        assert log_git_pin_drift("definitely-not-a-real-dist") == []

    def test_drift_logs_actionable_warning(self, monkeypatch, caplog):
        import importlib.metadata as im

        from omlx.utils import pin_drift

        monkeypatch.setattr(
            pin_drift, "_read_installed_direct_url", lambda name: _vcs_json(STALE)
        )
        monkeypatch.setattr(im, "requires", lambda dist: [VLM_REQ])

        with caplog.at_level("WARNING", logger="omlx.utils.pin_drift"):
            drift = log_git_pin_drift()

        assert len(drift) == 1
        message = caplog.text
        assert STALE[:12] in message
        assert PINNED[:12] in message
        assert f'pip install --force-reinstall --no-deps "{VLM_REQ}"' in message
