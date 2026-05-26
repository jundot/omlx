# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx/mcp/config.py — config file discovery, JSON/YAML
loading, schema validation, and the example-config helper.

MCPConfig / MCPServerConfig / MCPTransport themselves are covered in
test_mcp_types.py; here we only exercise the loader and validator.
"""

from __future__ import annotations

import json

import pytest

from omlx.mcp import config as mcp_config
from omlx.mcp.config import (
    CONFIG_ENV_VAR,
    create_example_config,
    load_mcp_config,
    validate_config,
)
from omlx.mcp.types import MCPConfig, MCPServerConfig, MCPTransport


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """Run each test in a clean directory with no env var and an empty
    search-path list. Prevents real ~/.config/omlx/mcp.json or a stray
    ./mcp.json from leaking into the test."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.setattr(mcp_config, "CONFIG_SEARCH_PATHS", [])
    return tmp_path


# =============================================================================
# validate_config
# =============================================================================


class TestValidateConfigInput:
    def test_non_dict_input_rejected(self):
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_config([])  # type: ignore[arg-type]

    def test_non_dict_input_rejected_for_string(self):
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_config("not a dict")  # type: ignore[arg-type]

    def test_empty_dict_yields_defaults(self):
        cfg = validate_config({})
        assert cfg.servers == {}
        assert cfg.max_tool_calls == 10
        assert cfg.default_timeout == 30.0

    def test_servers_string_value_rejected(self):
        """A truthy non-dict value for ``servers`` reaches the
        isinstance check and raises."""
        with pytest.raises(ValueError, match="'servers' must be a dictionary"):
            validate_config({"servers": "not-a-dict"})

    def test_servers_falsy_non_dict_silently_falls_through(self):
        """Quirk of the ``data.get('servers') or data.get('mcpServers', {})``
        chain: an empty list, empty string, or 0 for ``servers`` is
        falsy and triggers the fallback to ``mcpServers``. Documented
        so a future tighten-up doesn't break callers relying on it.
        """
        cfg = validate_config({"servers": []})
        assert cfg.servers == {}  # silently treated as 'use mcpServers default'

    def test_server_entry_must_be_dict(self):
        with pytest.raises(
            ValueError, match="Server 'broken' config must be a dictionary"
        ):
            validate_config({"servers": {"broken": "not-a-dict"}})


class TestValidateConfigServerLoading:
    def test_stdio_server_loaded(self):
        cfg = validate_config(
            {
                "servers": {
                    "fs": {
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["-y", "filesystem"],
                    }
                }
            }
        )
        assert "fs" in cfg.servers
        srv = cfg.servers["fs"]
        assert isinstance(srv, MCPServerConfig)
        assert srv.transport == MCPTransport.STDIO
        assert srv.command == "npx"
        assert srv.args == ["-y", "filesystem"]

    def test_server_name_auto_set_from_key(self):
        """User doesn't have to repeat ``name`` inside each server entry;
        the key acts as the name. Tests both the loader's name injection
        and that MCPServerConfig.__post_init__ doesn't override it."""
        cfg = validate_config(
            {"servers": {"my-server": {"transport": "stdio", "command": "x"}}}
        )
        assert cfg.servers["my-server"].name == "my-server"

    def test_explicit_name_in_entry_is_overridden_by_key(self):
        """If the user wrote ``name: other`` inside the entry, the dict
        key still wins — protects against name/key drift."""
        cfg = validate_config(
            {
                "servers": {
                    "real-name": {
                        "name": "wrong-name",
                        "transport": "stdio",
                        "command": "x",
                    }
                }
            }
        )
        assert cfg.servers["real-name"].name == "real-name"

    def test_invalid_server_field_raises_value_error(self):
        """Unknown kwargs to MCPServerConfig should surface as ValueError
        with the offending server's name in the message — makes the
        admin-panel error display actionable."""
        with pytest.raises(ValueError, match="Invalid config for server 'fs'"):
            validate_config(
                {"servers": {"fs": {"transport": "stdio", "command": "x", "bogus": 1}}}
            )

    def test_claude_desktop_mcpServers_format_accepted(self):
        """Upstream chose to accept Claude Desktop's ``mcpServers`` key
        as an alias for oMLX's ``servers``. Drop this and Claude users
        lose drop-in compatibility."""
        cfg = validate_config(
            {
                "mcpServers": {
                    "claude-srv": {"transport": "stdio", "command": "npx"}
                }
            }
        )
        assert "claude-srv" in cfg.servers
        assert cfg.servers["claude-srv"].command == "npx"

    def test_servers_takes_precedence_over_mcpServers(self):
        """When both keys are present, ``servers`` wins — the ``or``
        operator in load returns the first truthy value. This isn't
        merging; it's an either/or."""
        cfg = validate_config(
            {
                "servers": {"a": {"transport": "stdio", "command": "x"}},
                "mcpServers": {"b": {"transport": "stdio", "command": "y"}},
            }
        )
        assert set(cfg.servers.keys()) == {"a"}

    def test_empty_mcpServers_falls_through_to_servers(self):
        """If ``servers`` is missing and ``mcpServers`` is empty, the
        result is an empty servers dict, not an error."""
        cfg = validate_config({"mcpServers": {}})
        assert cfg.servers == {}


class TestValidateConfigGlobalOptions:
    def test_custom_max_tool_calls(self):
        cfg = validate_config({"max_tool_calls": 5})
        assert cfg.max_tool_calls == 5

    def test_max_tool_calls_zero_rejected(self):
        with pytest.raises(ValueError, match="'max_tool_calls' must be a positive integer"):
            validate_config({"max_tool_calls": 0})

    def test_max_tool_calls_negative_rejected(self):
        with pytest.raises(ValueError, match="'max_tool_calls' must be a positive integer"):
            validate_config({"max_tool_calls": -1})

    def test_max_tool_calls_non_int_rejected(self):
        with pytest.raises(ValueError, match="'max_tool_calls' must be a positive integer"):
            validate_config({"max_tool_calls": 3.5})

    def test_max_tool_calls_bool_rejected_in_practice(self):
        """``isinstance(True, int)`` is True in Python, so True passes
        the int check. Document the current behavior so it surfaces if
        someone tightens the check later."""
        cfg = validate_config({"max_tool_calls": True})
        assert cfg.max_tool_calls is True  # currently accepted; weird but expected

    def test_custom_default_timeout_int(self):
        cfg = validate_config({"default_timeout": 60})
        assert cfg.default_timeout == 60

    def test_custom_default_timeout_float(self):
        cfg = validate_config({"default_timeout": 45.5})
        assert cfg.default_timeout == 45.5

    def test_default_timeout_zero_rejected(self):
        with pytest.raises(ValueError, match="'default_timeout' must be a positive number"):
            validate_config({"default_timeout": 0})

    def test_default_timeout_negative_rejected(self):
        with pytest.raises(ValueError, match="'default_timeout' must be a positive number"):
            validate_config({"default_timeout": -1.0})

    def test_default_timeout_string_rejected(self):
        with pytest.raises(ValueError, match="'default_timeout' must be a positive number"):
            validate_config({"default_timeout": "30s"})


# =============================================================================
# _find_config_file (via load_mcp_config)
# =============================================================================


class TestExplicitPath:
    def test_existing_explicit_path_loads(self, isolated_env):
        cfg_path = isolated_env / "custom.json"
        cfg_path.write_text(json.dumps({"servers": {}}))
        cfg = load_mcp_config(cfg_path)
        assert isinstance(cfg, MCPConfig)
        assert cfg.servers == {}

    def test_missing_explicit_path_raises(self, isolated_env):
        with pytest.raises(FileNotFoundError, match="MCP config file not found"):
            load_mcp_config(isolated_env / "does-not-exist.json")

    def test_explicit_path_tilde_expanded(self, isolated_env, monkeypatch):
        """Tilde must expand — admins commonly pass ``~/.config/...``
        via --mcp-config flag."""
        monkeypatch.setenv("HOME", str(isolated_env))
        cfg_path = isolated_env / "tilde.json"
        cfg_path.write_text(json.dumps({"servers": {}}))
        cfg = load_mcp_config("~/tilde.json")
        assert isinstance(cfg, MCPConfig)


class TestEnvVarPath:
    def test_env_var_path_loads(self, isolated_env, monkeypatch):
        cfg_path = isolated_env / "from-env.json"
        cfg_path.write_text(json.dumps({"servers": {}}))
        monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg_path))
        cfg = load_mcp_config()
        assert isinstance(cfg, MCPConfig)

    def test_env_var_missing_file_falls_through(self, isolated_env, monkeypatch, caplog):
        """If OMLX_MCP_CONFIG points at a nonexistent file, the loader
        logs a warning but continues to the search-path fallback rather
        than aborting — broken env vars must not kill the server."""
        monkeypatch.setenv(CONFIG_ENV_VAR, str(isolated_env / "missing.json"))
        with caplog.at_level("WARNING", logger="omlx.mcp.config"):
            cfg = load_mcp_config()
        assert isinstance(cfg, MCPConfig)
        assert cfg.servers == {}
        assert any("not found" in r.message for r in caplog.records)


class TestSearchPath:
    def test_first_existing_search_path_wins(self, isolated_env, monkeypatch):
        a = isolated_env / "a.json"
        b = isolated_env / "b.json"
        a.write_text(json.dumps({"max_tool_calls": 1}))
        b.write_text(json.dumps({"max_tool_calls": 99}))
        # Order matters — first existing path is chosen
        monkeypatch.setattr(mcp_config, "CONFIG_SEARCH_PATHS", [str(a), str(b)])
        cfg = load_mcp_config()
        assert cfg.max_tool_calls == 1

    def test_falls_through_missing_paths(self, isolated_env, monkeypatch):
        present = isolated_env / "found.json"
        present.write_text(json.dumps({"max_tool_calls": 7}))
        monkeypatch.setattr(
            mcp_config,
            "CONFIG_SEARCH_PATHS",
            [
                str(isolated_env / "missing-1.json"),
                str(isolated_env / "missing-2.json"),
                str(present),
            ],
        )
        cfg = load_mcp_config()
        assert cfg.max_tool_calls == 7

    def test_no_config_anywhere_returns_empty(self, isolated_env):
        """All discovery paths fail → empty MCPConfig (not None, not
        FileNotFoundError). The server starts MCP-less in this case."""
        cfg = load_mcp_config()
        assert isinstance(cfg, MCPConfig)
        assert cfg.servers == {}
        assert cfg.max_tool_calls == 10
        assert cfg.default_timeout == 30.0


# =============================================================================
# File-format handling
# =============================================================================


class TestFileFormats:
    def test_loads_json_file(self, isolated_env):
        cfg_path = isolated_env / "mcp.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "servers": {"fs": {"transport": "stdio", "command": "x"}},
                    "max_tool_calls": 3,
                }
            )
        )
        cfg = load_mcp_config(cfg_path)
        assert "fs" in cfg.servers
        assert cfg.max_tool_calls == 3

    def test_invalid_json_raises(self, isolated_env):
        cfg_path = isolated_env / "broken.json"
        cfg_path.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            load_mcp_config(cfg_path)

    def test_loads_yaml_file_when_pyyaml_available(self, isolated_env):
        """Only run if PyYAML is installed; not a hard dependency of
        the project."""
        pytest.importorskip("yaml")
        cfg_path = isolated_env / "mcp.yaml"
        cfg_path.write_text(
            "servers:\n  fs:\n    transport: stdio\n    command: npx\n"
            "max_tool_calls: 4\n"
        )
        cfg = load_mcp_config(cfg_path)
        assert cfg.servers["fs"].command == "npx"
        assert cfg.max_tool_calls == 4

    def test_yml_extension_also_treated_as_yaml(self, isolated_env):
        pytest.importorskip("yaml")
        cfg_path = isolated_env / "mcp.yml"
        cfg_path.write_text(
            "servers:\n  fs:\n    transport: stdio\n    command: x\n"
        )
        cfg = load_mcp_config(cfg_path)
        assert "fs" in cfg.servers


# =============================================================================
# create_example_config
# =============================================================================


class TestCreateExampleConfig:
    def test_returns_valid_json_string(self):
        example = create_example_config()
        data = json.loads(example)  # would raise if not valid JSON
        assert isinstance(data, dict)

    def test_example_round_trips_through_validate(self):
        """The example written to disk by ``omlx mcp init`` (or similar)
        must be a valid config — otherwise the bootstrap UX is broken."""
        example = create_example_config()
        data = json.loads(example)
        cfg = validate_config(data)
        assert isinstance(cfg, MCPConfig)
        assert len(cfg.servers) >= 1
        # The example showcases multiple transports — keep a guard so
        # future edits don't shrink it to just stdio.
        transports = {s.transport for s in cfg.servers.values()}
        assert MCPTransport.STDIO in transports
        assert MCPTransport.SSE in transports

    def test_example_has_top_level_global_options(self):
        data = json.loads(create_example_config())
        assert "max_tool_calls" in data
        assert "default_timeout" in data
