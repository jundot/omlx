# SPDX-License-Identifier: Apache-2.0
"""
CLI tests for oMLX.

Tests CLI argument parsing, command setup, and help text.
Note: Configuration validation tests are in test_config.py.
"""

import argparse
import subprocess
import sys
from unittest.mock import patch, MagicMock

import pytest


class TestCLIModule:
    """Tests for CLI module existence and basic functionality."""

    def test_cli_module_importable(self):
        """Test that CLI module can be imported."""
        from omlx import cli
        assert hasattr(cli, "main")

    def test_cli_has_serve_command(self):
        """Test that CLI has serve command setup."""
        from omlx import cli
        # The module should have the main entry point
        assert callable(cli.main)


class TestCLIHelp:
    """Tests for CLI help functionality."""

    def test_main_help(self):
        """Test main CLI help output."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should succeed with help
        assert result.returncode == 0
        # Should show available commands
        assert "serve" in result.stdout.lower()

    def test_serve_help(self):
        """Test serve command help output."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should succeed with help
        assert result.returncode == 0
        # Should show serve options
        stdout_lower = result.stdout.lower()
        assert "host" in stdout_lower
        assert "port" in stdout_lower
        assert "model-dir" in stdout_lower


class TestCLIEntryPoint:
    """Tests for CLI entry point functionality."""

    def test_module_runnable(self):
        """Test that CLI module is runnable."""
        # Should not crash when running with --help
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_invalid_command_error(self):
        """Test error handling for invalid command."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "invalid_command"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should fail with non-zero exit code
        assert result.returncode != 0

    def test_no_command_shows_help(self):
        """Test that no command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should exit with non-zero (no command provided)
        assert result.returncode != 0


class TestServeCommandOptions:
    """Tests for serve command options via help output."""

    def test_serve_has_model_dir_option(self):
        """Test that serve command has --model-dir option."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--model-dir" in result.stdout

    def test_serve_has_max_model_memory_option(self):
        """Test that serve command has --max-model-memory option."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--max-model-memory" in result.stdout

    def test_serve_no_model_specific_options(self):
        """Test that serve command does not have model-specific options (managed via admin page)."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # These options are now managed via admin page, not CLI
        assert "--pin" not in result.stdout
        assert "--default-model" not in result.stdout
        assert "--max-tokens" not in result.stdout
        assert "--temperature" not in result.stdout
        assert "--top-p" not in result.stdout
        assert "--top-k" not in result.stdout
        assert "--force-sampling" not in result.stdout

    def test_serve_has_host_port_options(self):
        """Test that serve command has --host and --port options."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_serve_has_scheduler_options(self):
        """Test that serve command has scheduler options."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--max-num-seqs" in result.stdout
        assert "--completion-batch-size" in result.stdout

    def test_serve_has_cache_options(self):
        """Test that serve command has cache options."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--paged-ssd-cache-dir" in result.stdout
        assert "--paged-ssd-cache-max-size" in result.stdout
        assert "--no-cache" in result.stdout

    def test_serve_has_mcp_option(self):
        """Test that serve command has --mcp-config option."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--mcp-config" in result.stdout

    def test_serve_has_base_path_option(self):
        """Test that serve command has --base-path option."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--base-path" in result.stdout

    def test_serve_has_api_key_option(self):
        """Test that serve command has --api-key option."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--api-key" in result.stdout



class TestLaunchCommandOptions:
    """Tests for launch command options via help output."""

    def test_launch_has_host_port_options(self):
        """Test that launch command has --host and --port options."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_launch_has_model_option(self):
        """Test that launch command has --model option."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--model" in result.stdout


class TestServeCommandFunctions:
    """Tests for serve command function."""

    def test_serve_command_exists(self):
        """Test that serve_command function exists."""
        from omlx.cli import serve_command
        assert callable(serve_command)

    def test_serve_model_dir_optional_with_default(self):
        """Test that serve --model-dir is optional with default ~/.omlx/models."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should show that model-dir has a default
        assert "default" in result.stdout.lower()
        # Help text should mention ~/.omlx/models or similar
        assert ".omlx" in result.stdout or "model" in result.stdout.lower()



class TestCLIDocstrings:
    """Tests for CLI module docstrings and descriptions."""

    def test_main_has_description(self):
        """Test that main help has description."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should have some description
        assert "omlx" in result.stdout.lower() or "llm" in result.stdout.lower()

    def test_serve_has_description(self):
        """Test that serve command has description."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should describe multi-model serving
        assert "multi-model" in result.stdout.lower() or "server" in result.stdout.lower()
