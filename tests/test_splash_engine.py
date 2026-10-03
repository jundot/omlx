# SPDX-License-Identifier: Apache-2.0
"""Tests for the Splash engine and its discovery, compatibility and pool wiring."""

import asyncio
import json
import os
import signal
import stat
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from omlx.engine import splash as splash_mod
from omlx.engine.splash import (
    SplashBuild,
    SplashEngine,
    find_splash_builds,
    is_splash_compatible,
    select_splash_build,
    splash_effort,
    splash_messages,
)
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import InvalidRequestError, ModelLoadingError
from omlx.model_discovery import discover_models, is_splash_package

QWEN38_27B_TEXT = dict(splash_mod.SPLASH_FAMILIES["Qwen3.8-27B"])
MLX_4BIT = {"mode": "affine", "bits": 4, "group_size": 64}


def _write_package(path: Path, fmt: str = "splash-packed-q4") -> Path:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"format": {"name": fmt}}))
    (path / "target.bin").write_bytes(b"0" * 4096)
    tokenizer = path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    return path


def _write_mlx_model(path: Path, text_config=None, quantization=None) -> Path:
    path.mkdir(parents=True)
    config = {
        "model_type": "qwen3_5",
        "text_config": QWEN38_27B_TEXT if text_config is None else text_config,
        "quantization": MLX_4BIT if quantization is None else quantization,
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors").write_bytes(b"0" * 1024)
    return path


def _cache_repo(hub: Path, repo_id: str) -> None:
    snapshot = hub / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "abc"
    snapshot.mkdir(parents=True)


def _fake_executable(path: Path, body: str = "exit 0\n") -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def hub(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(splash_mod, "_hf_hub_cache", lambda: hub)
    return hub


# What `splash serve --help` lists, per release: 1.1 added upstream MLX
# checkpoints together with --language-only.
SPLASH_1_1_OPTIONS = "--host --port --model --language-only --no-webui --max-context"
SPLASH_1_0_OPTIONS = "--host --port --model --no-webui --max-context"


def _fake_launcher(path: Path, options: str, body: str = "exit 0\n") -> Path:
    help_text = (
        f'if [ "$1" = serve ] && [ "$2" = --help ]; then echo "{options}"; exit 0; fi\n'
    )
    return _fake_executable(path, help_text + body)


def _build(options: str = SPLASH_1_1_OPTIONS) -> SplashBuild:
    return SplashBuild(
        "splash", "Splash", Path("/bin/splash"), frozenset(options.split())
    )


@pytest.fixture
def launchers(tmp_path, monkeypatch):
    """Install fake launchers by name; no other Splash install is found."""
    installed: dict[str, Path] = {}
    monkeypatch.delenv(splash_mod.SPLASH_PATH_ENV, raising=False)
    monkeypatch.setattr(splash_mod, "_find_launcher", installed.get)

    def install(name, options=SPLASH_1_1_OPTIONS, body="exit 0\n"):
        directory = tmp_path / "bin"
        directory.mkdir(exist_ok=True)
        installed[name] = _fake_launcher(directory / name, options, body)
        return installed[name]

    return install


@pytest.fixture
def splash_installed(launchers):
    return launchers("splash")


# -- discovery ---------------------------------------------------------------


class TestDiscovery:
    @pytest.mark.asyncio
    async def test_model_manager_lists_and_deletes_packages(
        self, tmp_path, monkeypatch
    ):
        from omlx.admin import routes

        _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        _write_mlx_model(tmp_path / "Qwen3.8-27B-4bit")
        (tmp_path / "not-a-model").mkdir()
        settings = MagicMock()
        settings.model.get_model_dirs.return_value = [tmp_path]
        monkeypatch.setattr(routes, "_get_global_settings", lambda: settings)
        monkeypatch.setattr(routes, "_get_engine_pool", lambda: None)

        listed = (await routes.list_hf_models(is_admin=True))["models"]
        assert sorted(m["name"] for m in listed) == [
            "Qwen3.8-27B-4bit",
            "Qwen3.8-27B-Splash",
        ]

        await routes.delete_hf_model("Qwen3.8-27B-Splash", is_admin=True)
        assert not (tmp_path / "incoai" / "Qwen3.8-27B-Splash").exists()

    def test_package_detected_by_manifest_format(self, tmp_path):
        package = _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        other = tmp_path / "other"
        other.mkdir()
        (other / "manifest.json").write_text(json.dumps({"format": {"name": "gguf"}}))

        assert is_splash_package(package)
        assert not is_splash_package(other)
        assert not is_splash_package(tmp_path / "missing")

    def test_packages_register_on_the_splash_engine(self, tmp_path):
        _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        _write_package(tmp_path / "flat-package")

        models = discover_models(tmp_path)

        for model_id in ("Qwen3.8-27B-Splash", "flat-package"):
            info = models[model_id]
            assert info.engine_type == "splash"
            assert info.model_type == "llm"
            assert info.config_model_type == "qwen3_5"
            assert info.estimated_size >= 4096

    def test_hf_cache_package_keeps_its_repo_id(self, tmp_path):
        entry = tmp_path / "models--incoai--Qwen3.8-27B-Splash"
        _write_package(entry / "snapshots" / "abc")
        (entry / "refs").mkdir()
        (entry / "refs" / "main").write_text("abc")

        models = discover_models(tmp_path)

        info = next(iter(models.values()))
        assert info.engine_type == "splash"
        assert info.source_type == "hf_cache"
        assert info.source_repo_id == "incoai/Qwen3.8-27B-Splash"


# -- compatibility -------------------------------------------------------------


class TestCompatibility:
    def test_supported_cached_checkpoint(self, tmp_path, hub, splash_installed):
        model = _write_mlx_model(tmp_path / "mlx-community" / "Qwen3.8-27B-4bit")
        _cache_repo(hub, "mlx-community/Qwen3.8-27B-4bit")

        assert is_splash_compatible(model) == (True, "")

    def test_hf_cache_entry_uses_source_repo_id(self, tmp_path, hub, splash_installed):
        model = _write_mlx_model(tmp_path / "snapshots" / "abc")
        _cache_repo(hub, "mlx-community/Qwen3.8-27B-4bit")

        ok, _ = is_splash_compatible(model, "mlx-community/Qwen3.8-27B-4bit")

        assert ok

    @pytest.mark.parametrize(
        "text_config, quantization, reason",
        [
            ({**QWEN38_27B_TEXT, "num_hidden_layers": 48}, None, "serves"),
            (None, {"mode": "affine", "bits": 8, "group_size": 64}, "4-bit"),
            (None, {"mode": "mxfp4", "bits": 4, "group_size": 32}, "4-bit"),
        ],
    )
    def test_rejects_unsupported_checkpoints(
        self, tmp_path, hub, splash_installed, text_config, quantization, reason
    ):
        model = _write_mlx_model(tmp_path / "org" / "model", text_config, quantization)
        _cache_repo(hub, "org/model")

        ok, message = is_splash_compatible(model)

        assert not ok
        assert reason in message

    def test_rejects_models_missing_from_hf_cache(
        self, tmp_path, hub, splash_installed
    ):
        model = _write_mlx_model(tmp_path / "mlx-community" / "Qwen3.8-27B-4bit")

        ok, message = is_splash_compatible(model)

        assert not ok
        assert "Hugging Face cache" in message

    def test_rejects_when_splash_is_missing(self, tmp_path, hub, launchers):
        model = _write_mlx_model(tmp_path / "mlx-community" / "Qwen3.8-27B-4bit")
        _cache_repo(hub, "mlx-community/Qwen3.8-27B-4bit")

        ok, message = is_splash_compatible(model)

        assert not ok
        assert message == splash_mod.SPLASH_INSTALL_HINT
        assert "incoai/tap/splash" in message and "M1/M2" in message

    def test_old_build_serves_packages_but_not_checkpoints(
        self, tmp_path, hub, launchers
    ):
        launchers("splash-m1", SPLASH_1_0_OPTIONS)
        model = _write_mlx_model(tmp_path / "mlx-community" / "Qwen3.8-27B-4bit")
        package = _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        _cache_repo(hub, "mlx-community/Qwen3.8-27B-4bit")
        _cache_repo(hub, "incoai/Qwen3.8-27B-Splash")

        ok, message = is_splash_compatible(model)

        assert not ok
        assert "Splash 1.1" in message
        assert is_splash_compatible(package) == (True, "")

    def test_requested_build_must_be_installed(self, tmp_path, hub, splash_installed):
        package = _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        _cache_repo(hub, "incoai/Qwen3.8-27B-Splash")

        ok, message = is_splash_compatible(package, build_id="splash-m1")

        assert not ok
        assert "'splash-m1' is not installed" in message

    def test_package_needs_no_config(self, tmp_path, hub, splash_installed):
        package = _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        _cache_repo(hub, "incoai/Qwen3.8-27B-Splash")

        assert is_splash_compatible(package) == (True, "")


# -- builds ------------------------------------------------------------------------


class TestBuilds:
    def test_builds_report_their_serve_options(self, launchers):
        launchers("splash", SPLASH_1_1_OPTIONS)
        launchers("splash-m1", SPLASH_1_0_OPTIONS)

        builds = {b.id: b for b in find_splash_builds()}

        assert builds["splash"].loads_mlx_checkpoints
        assert not builds["splash-m1"].loads_mlx_checkpoints
        assert builds["splash-m1"].label == "Splash M1"

    @pytest.mark.parametrize(
        "m1_or_m2, expected", [(True, "splash-m1"), (False, "splash")]
    )
    def test_automatic_choice_follows_the_chip(
        self, launchers, monkeypatch, m1_or_m2, expected
    ):
        launchers("splash")
        launchers("splash-m1")
        monkeypatch.setattr(splash_mod, "_is_m1_or_m2", lambda: m1_or_m2)

        assert select_splash_build(find_splash_builds()).id == expected

    def test_automatic_choice_falls_back_to_the_other_build(
        self, launchers, monkeypatch
    ):
        launchers("splash")
        monkeypatch.setattr(splash_mod, "_is_m1_or_m2", lambda: True)

        assert select_splash_build(find_splash_builds()).id == "splash"

    def test_explicit_path_comes_first(self, tmp_path, launchers, monkeypatch):
        launchers("splash")
        custom = _fake_launcher(tmp_path / "checkout-splash", SPLASH_1_1_OPTIONS)
        monkeypatch.setenv(splash_mod.SPLASH_PATH_ENV, str(custom))

        build = select_splash_build(find_splash_builds())

        assert build.id == splash_mod.CUSTOM_BUILD
        assert build.path == custom

    def test_requested_build_wins(self, launchers, monkeypatch):
        launchers("splash")
        launchers("splash-m1")
        monkeypatch.setattr(splash_mod, "_is_m1_or_m2", lambda: True)
        builds = find_splash_builds()

        assert select_splash_build(builds, "splash").id == "splash"
        assert select_splash_build(builds, "custom") is None


# -- request translation ---------------------------------------------------------


class TestRequestTranslation:
    @pytest.mark.parametrize(
        "kwargs, effort",
        [
            ({"enable_thinking": False}, "none"),
            ({"reasoning_effort": "high"}, "xhigh"),
            ({"reasoning_effort": "Minimal"}, "low"),
            ({}, None),
            (None, None),
        ],
    )
    def test_effort_maps_to_splash_levels(self, kwargs, effort):
        assert splash_effort(kwargs) == effort

    def test_tool_call_arguments_become_json_strings(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "f", "arguments": {"a": 1}}}
                ],
            },
        ]

        converted = splash_messages(messages)

        call = converted[1]["tool_calls"][0]
        assert call["type"] == "function"
        assert json.loads(call["function"]["arguments"]) == {"a": 1}
        assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"a": 1}

    def test_serve_command_passes_context_limit(self, tmp_path):
        engine = SplashEngine(
            str(tmp_path / "mlx-community" / "Qwen3.8-27B-4bit"),
            _build(),
            max_context=10**6,
        )

        command = engine._serve_command(9000)

        assert command[:4] == [
            "/bin/splash",
            "serve",
            "--model",
            "mlx-community/Qwen3.8-27B-4bit",
        ]
        assert "--language-only" in command
        assert command[-2:] == ["--max-context", str(splash_mod.SPLASH_MAX_CONTEXT)]

    def test_packages_start_without_language_only(self, tmp_path):
        package = _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        engine = SplashEngine(str(package), _build())

        assert "--language-only" not in engine._serve_command(9000)

    def test_serve_command_omits_options_an_old_build_lacks(self, tmp_path):
        engine = SplashEngine(
            str(tmp_path / "incoai" / "Qwen3.8-27B-Splash"), _build(SPLASH_1_0_OPTIONS)
        )

        command = engine._serve_command(9000)

        assert "--language-only" not in command
        assert "--no-webui" in command

    def test_env_hides_key_from_argv_and_drops_bundled_python(self, monkeypatch):
        monkeypatch.setenv("PYTHONHOME", "/bundle")
        monkeypatch.setenv("PYTHONPATH", "/bundle/lib")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        env = splash_mod._splash_env("secret")

        assert env["SPLASH_API_KEY"] == "secret"
        assert "PYTHONHOME" not in env and "PYTHONPATH" not in env
        assert env["PATH"].split(":") == [
            *splash_mod._BREW_BIN_DIRS,
            "/usr/bin",
            "/bin",
        ]

    def test_serve_command_max_cache_disk(self, tmp_path):
        build_with_disk = _build(SPLASH_1_1_OPTIONS + " --max-cache-disk")
        build_without_disk = _build(SPLASH_1_1_OPTIONS)

        # when cache_disk_bytes > 0 and build supports it: included
        engine = SplashEngine(
            str(tmp_path / "model"),
            build_with_disk,
            cache_disk_bytes=1024 * 1024,
        )
        assert engine._serve_command(9000)[-2:] == [
            "--max-cache-disk",
            str(1024 * 1024),
        ]

        # omits when bytes is 0
        engine_zero = SplashEngine(
            str(tmp_path / "model"),
            build_with_disk,
            cache_disk_bytes=0,
        )
        assert "--max-cache-disk" not in engine_zero._serve_command(9000)

        # omits when build does not list the option
        engine_unsupported = SplashEngine(
            str(tmp_path / "model"),
            build_without_disk,
            cache_disk_bytes=1024 * 1024,
        )
        assert engine_unsupported._cache_disk_bytes == 0
        assert "--max-cache-disk" not in engine_unsupported._serve_command(9000)

    def test_env_tmpdir(self, monkeypatch):
        monkeypatch.setenv("TMPDIR", "/inherited/tmp")
        env_with_tmp = splash_mod._splash_env("secret", tmpdir=Path("/custom/tmp"))
        assert env_with_tmp["TMPDIR"] == "/custom/tmp"

        env_default = splash_mod._splash_env("secret")
        assert env_default.get("TMPDIR") == "/inherited/tmp"

    def test_request_body_response_format(self, tmp_path):
        engine = SplashEngine(str(tmp_path / "model"), _build())
        rf = {"type": "json_object"}
        body = engine._request_body(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            kwargs={"response_format": rf},
        )
        assert body["response_format"] == rf

        body_without = engine._request_body(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            kwargs={},
        )
        assert "response_format" not in body_without


# -- streaming -------------------------------------------------------------------


def _sse(*chunks) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _engine_with_transport(tmp_path, handler) -> SplashEngine:
    engine = SplashEngine(str(tmp_path / "incoai" / "Qwen3.8-27B-Splash"), _build())
    engine._client = httpx.AsyncClient(
        base_url="http://splash", transport=httpx.MockTransport(handler)
    )
    return engine


class TestStreaming:
    async def test_reasoning_content_and_tool_calls(self, tmp_path):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            body = _sse(
                {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
                {"choices": [{"delta": {"content": "Hi"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "get",
                                            "arguments": '{"a"',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": ": 1}"}}
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    },
                },
            )
            return httpx.Response(200, content=body)

        engine = _engine_with_transport(tmp_path, handler)
        outputs = [
            o
            async for o in engine.stream_chat(
                [{"role": "user", "content": "hi"}],
                top_k=100,
                min_p=0.1,
                chat_template_kwargs={"reasoning_effort": "high"},
                tools=[{"type": "function", "function": {"name": "get"}}],
            )
        ]

        final = outputs[-1]
        assert final.finished
        assert final.text == "<think>hmm</think>Hi"
        assert final.finish_reason == "tool_calls"
        assert final.tool_calls == [
            {"id": "call_1", "name": "get", "arguments": '{"a": 1}'}
        ]
        assert (final.prompt_tokens, final.completion_tokens, final.cached_tokens) == (
            7,
            3,
            4,
        )
        assert all(not o.finished for o in outputs[:-1])
        assert outputs[0].first_token_at is not None
        body = seen["body"]
        assert body["top_k"] == splash_mod.SPLASH_MAX_TOP_K
        assert "min_p" not in body
        assert body["reasoning_effort"] == "xhigh"
        assert body["stream"] is True
        assert body["return_progress"] is True
        assert engine.has_active_requests() is False

    async def test_unfinished_reasoning_is_closed(self, tmp_path):
        def handler(request):
            return httpx.Response(
                200, content=_sse({"choices": [{"delta": {"reasoning_content": "x"}}]})
            )

        engine = _engine_with_transport(tmp_path, handler)
        final = await engine.chat([{"role": "user", "content": "hi"}])

        assert final.text == "<think>x</think>"
        assert final.new_text == final.text

    async def test_client_errors_become_invalid_requests(self, tmp_path):
        def handler(request):
            return httpx.Response(400, json={"error": {"message": "bad top_p"}})

        engine = _engine_with_transport(tmp_path, handler)

        with pytest.raises(InvalidRequestError, match="bad top_p"):
            await engine.chat([{"role": "user", "content": "hi"}])
        assert engine.has_active_requests() is False

    async def test_server_errors_raise(self, tmp_path):
        def handler(request):
            return httpx.Response(503, text="engine failure")

        engine = _engine_with_transport(tmp_path, handler)

        with pytest.raises(RuntimeError, match="503"):
            await engine.chat([{"role": "user", "content": "hi"}])

    async def test_live_requests_during_stream_and_on_error(self, tmp_path):
        prefill_checked = False

        class StreamingBody(httpx.AsyncByteStream):
            def __init__(self, check_prefill):
                self.check_prefill = check_prefill

            async def __aiter__(self):
                chunk1 = {
                    "id": "1",
                    "choices": [],
                    "prompt_progress": {
                        "total": 100,
                        "cache": 20,
                        "processed": 60,
                        "time_ms": 200.0,
                    },
                }
                yield f"data: {json.dumps(chunk1)}\n\n".encode()
                self.check_prefill()
                chunk2 = {"choices": [{"delta": {"content": "Hello world"}}]}
                yield f"data: {json.dumps(chunk2)}\n\n".encode()
                yield b"data: [DONE]\n\n"

        seen = {}

        def check_prefill():
            nonlocal prefill_checked
            live = engine.get_live_requests()
            assert len(live["prefilling"]) == 1
            assert len(live["generating"]) == 0
            row = live["prefilling"][0]
            assert row["total"] == 100
            assert row["processed"] == 60
            assert row["speed"] == (60 - 20) / 0.2
            assert row["eta"] == (100 - 60) / 200.0
            assert row["elapsed"] >= 0.0
            assert row["detail"] is None
            prefill_checked = True

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, stream=StreamingBody(check_prefill))

        engine = _engine_with_transport(tmp_path, handler)
        gen_checked = False
        async for out in engine.stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=128
        ):
            if not out.finished:
                live = engine.get_live_requests()
                assert len(live["prefilling"]) == 0
                assert len(live["generating"]) == 1
                row = live["generating"][0]
                assert row["generated_tokens"] > 0
                assert row["prompt_tokens"] == 100
                assert row["max_tokens"] == 128
                assert row["elapsed_seconds"] >= 0.0
                assert row["tokens_per_second"] >= 0.0
                assert row["last_activity_age_seconds"] >= 0.0
                gen_checked = True

        assert prefill_checked is True
        assert gen_checked is True
        assert seen["body"].get("return_progress") is True
        assert engine.get_live_requests() == {"prefilling": [], "generating": []}

        # Verify both lists are empty when the stream raises
        def failing_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="internal error")

        engine._client = httpx.AsyncClient(
            base_url="http://splash", transport=httpx.MockTransport(failing_handler)
        )
        with pytest.raises(RuntimeError):
            async for _ in engine.stream_chat([{"role": "user", "content": "hi"}]):
                pass
        assert engine.get_live_requests() == {"prefilling": [], "generating": []}

    async def test_prompt_progress_empty_choices_produces_no_text(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            body = _sse(
                {
                    "choices": [],
                    "prompt_progress": {
                        "total": 50,
                        "cache": 10,
                        "processed": 50,
                        "time_ms": 100.0,
                    },
                }
            )
            return httpx.Response(200, content=body)

        engine = _engine_with_transport(tmp_path, handler)
        outputs = [
            o async for o in engine.stream_chat([{"role": "user", "content": "hi"}])
        ]
        assert len(outputs) == 1
        assert outputs[0].finished is True
        assert outputs[0].text == ""
        assert outputs[0].new_text == ""

    async def test_plain_completions_are_rejected(self, tmp_path):
        engine = SplashEngine(str(tmp_path / "org" / "model"), _build())

        with pytest.raises(InvalidRequestError, match="chat requests only"):
            await engine.generate("Once upon a time")


# -- process lifecycle -------------------------------------------------------------

# A stand-in for `splash serve`: serves /v1/models to requests that carry the
# API key from SPLASH_API_KEY, like the real server.
FAKE_SERVER = textwrap.dedent("""\
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    model, port = args[args.index("--model") + 1], int(args[args.index("--port") + 1])
    key = os.environ["SPLASH_API_KEY"]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer " + key:
                self.send_response(401); self.end_headers(); return
            body = json.dumps({"data": [{"id": model}]}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
        def log_message(self, *a):
            pass
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """)


@pytest.fixture
def lifecycle(tmp_path, monkeypatch, launchers):
    monkeypatch.setattr(splash_mod, "_log_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(splash_mod, "STARTUP_POLL_SECONDS", 0.05)
    monkeypatch.setattr(
        SplashEngine,
        "_load_tokenizer",
        lambda self: setattr(self, "_tokenizer", MagicMock()),
    )

    def install(body: str) -> SplashBuild:
        launchers("splash", SPLASH_1_1_OPTIONS, body)
        return find_splash_builds()[0]

    return install


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    return True


class TestLifecycle:
    async def test_start_serves_and_stop_ends_the_process_group(
        self, tmp_path, lifecycle
    ):
        script = tmp_path / "server.py"
        script.write_text(FAKE_SERVER)
        build = lifecycle(f'exec "{sys.executable}" "{script}" "$@"\n')
        engine = SplashEngine(str(tmp_path / "incoai" / "Qwen3.8-27B-Splash"), build)

        await engine.start()
        group = engine._process.pid
        try:
            assert engine.get_stats()["running"] is True
            assert await engine._serves_model()
        finally:
            await engine.stop()

        assert not _group_alive(group)
        assert engine.get_stats()["running"] is False

    async def test_exit_during_start_reports_the_log(self, tmp_path, lifecycle):
        build = lifecycle(
            'echo "Installing collected packages: numpy"\n'
            'echo "error: the engine\'s device check failed" >&2\nexit 1\n'
        )
        engine = SplashEngine(str(tmp_path / "org" / "model"), build)

        with pytest.raises(RuntimeError, match="device check failed") as error:
            await engine.start()
        assert "Installing" not in str(error.value)
        assert engine._process is None

    async def test_watchdog_stops_splash_when_its_parent_dies(self, tmp_path):
        # The watchdog polls its parent's PID; a PID that is gone must end
        # the command it supervises.
        dead_parent = os.fork()
        if dead_parent == 0:
            os._exit(0)
        os.waitpid(dead_parent, 0)
        command = splash_mod._watchdog_command(["/bin/sleep", "60"])
        command[command.index(str(os.getpid()))] = str(dead_parent)
        process = __import__("subprocess").Popen(command, start_new_session=True)
        try:
            assert process.wait(timeout=10) is not None
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
        assert not _group_alive(process.pid)

    async def test_start_creates_cache_disk_dir_and_passes_tmpdir(
        self, tmp_path, lifecycle
    ):
        cache_dir = tmp_path / "cache" / "splash"
        assert not cache_dir.exists()
        script = tmp_path / "server.py"
        script.write_text(FAKE_SERVER)
        build_options = SPLASH_1_1_OPTIONS + " --max-cache-disk"
        launcher = lifecycle(f'exec "{sys.executable}" "{script}" "$@"\n')
        build = SplashBuild(
            launcher.id, launcher.label, launcher.path, frozenset(build_options.split())
        )
        engine = SplashEngine(
            str(tmp_path / "incoai" / "Qwen3.8-27B-Splash"),
            build,
            cache_disk_bytes=100 * 1024 * 1024,
            cache_disk_dir=cache_dir,
        )

        await engine.start()
        try:
            assert cache_dir.exists()
            assert engine.get_stats()["cache_disk_bytes"] == 100 * 1024 * 1024
        finally:
            await engine.stop()

    def test_stats_cache_disk_bytes_zero_when_not_passed(self, tmp_path):
        build_with_disk = _build(SPLASH_1_1_OPTIONS + " --max-cache-disk")
        build_without_disk = _build(SPLASH_1_1_OPTIONS)

        engine_zero = SplashEngine(
            str(tmp_path / "model"), build_with_disk, cache_disk_bytes=0
        )
        assert engine_zero.get_stats()["cache_disk_bytes"] == 0

        engine_unsupported = SplashEngine(
            str(tmp_path / "model"), build_without_disk, cache_disk_bytes=1024
        )
        assert engine_unsupported.get_stats()["cache_disk_bytes"] == 0


# -- engine pool -----------------------------------------------------------------------


def _entry(path: Path, engine_type: str = "batched", repo=None) -> EngineEntry:
    return EngineEntry(
        model_id=path.name,
        model_path=str(path),
        model_type="llm",
        engine_type=engine_type,
        estimated_size=1024,
        source_repo_id=repo,
    )


class TestEnginePool:
    def test_packages_always_use_splash(self, tmp_path, hub, launchers):
        launchers("splash-m1", SPLASH_1_0_OPTIONS)
        package = _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        _cache_repo(hub, "incoai/Qwen3.8-27B-Splash")

        pool = EnginePool()
        engine = pool._splash_engine_for(_entry(package, "splash"), None, "splash")

        assert isinstance(engine, SplashEngine)
        assert engine.model_name == "incoai/Qwen3.8-27B-Splash"
        assert engine.get_stats()["splash_build"] == "Splash M1"

    def test_package_without_splash_fails_with_install_hint(
        self, tmp_path, hub, launchers
    ):
        package = _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        _cache_repo(hub, "incoai/Qwen3.8-27B-Splash")

        pool = EnginePool()
        with pytest.raises(ModelLoadingError, match="incoai/tap/splash"):
            pool._splash_engine_for(_entry(package, "splash"), None, "splash")

    def test_mlx_models_need_the_setting(self, tmp_path):
        model = _write_mlx_model(tmp_path / "mlx-community" / "Qwen3.8-27B-4bit")

        pool = EnginePool()
        assert pool._splash_engine_for(_entry(model), None, "batched") is None
        settings = MagicMock(splash_enabled=False)
        assert pool._splash_engine_for(_entry(model), settings, "batched") is None

    def test_enabled_mlx_model_uses_splash(self, tmp_path, hub, splash_installed):
        model = _write_mlx_model(tmp_path / "mlx-community" / "Qwen3.8-27B-4bit")
        _cache_repo(hub, "mlx-community/Qwen3.8-27B-4bit")
        settings = MagicMock(
            splash_enabled=True, splash_build=None, max_context_window=65536
        )

        pool = EnginePool()
        engine = pool._splash_engine_for(_entry(model), settings, "vlm")

        assert isinstance(engine, SplashEngine)
        assert engine._max_context == 65536

    def test_enabled_incompatible_model_fails_to_load(
        self, tmp_path, hub, splash_installed
    ):
        model = _write_mlx_model(tmp_path / "mlx-community" / "Qwen3.8-27B-4bit")
        settings = MagicMock(splash_enabled=True, splash_build=None)

        pool = EnginePool()
        with pytest.raises(ModelLoadingError, match="Hugging Face cache"):
            pool._splash_engine_for(_entry(model), settings, "batched")

    def test_ssd_cache_forwarded_to_splash_engine(self, tmp_path, hub, launchers):
        launchers("splash-m1", SPLASH_1_0_OPTIONS + " --max-cache-disk")
        package = _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        _cache_repo(hub, "incoai/Qwen3.8-27B-Splash")

        # When paged_ssd_cache_dir is set and hot_cache_only is False:
        pool = EnginePool()
        pool._scheduler_config.paged_ssd_cache_dir = str(tmp_path / "ssd")
        pool._scheduler_config.paged_ssd_cache_max_size = 50 * 1024 * 1024
        pool._scheduler_config.hot_cache_only = False

        engine = pool._splash_engine_for(_entry(package, "splash"), None, "splash")
        assert engine._cache_disk_bytes == 50 * 1024 * 1024
        assert engine._cache_disk_dir == Path(tmp_path / "ssd") / "splash"

        # When build lacks --max-cache-disk option:
        launchers("splash-m1", SPLASH_1_0_OPTIONS)
        engine_unsupported = pool._splash_engine_for(
            _entry(package, "splash"), None, "splash"
        )
        assert engine_unsupported._cache_disk_bytes == 0
        assert engine_unsupported._cache_disk_dir == Path(tmp_path / "ssd") / "splash"

        # When paged_ssd_cache_dir is None:
        pool_no_cache = EnginePool()
        pool_no_cache._scheduler_config.paged_ssd_cache_dir = None
        pool_no_cache._scheduler_config.hot_cache_only = False

        engine_no_cache = pool_no_cache._splash_engine_for(
            _entry(package, "splash"), None, "splash"
        )
        assert engine_no_cache._cache_disk_bytes == 0
        assert engine_no_cache._cache_disk_dir is None

        # When hot_cache_only is True:
        pool_hot_only = EnginePool()
        pool_hot_only._scheduler_config.paged_ssd_cache_dir = str(tmp_path / "ssd")
        pool_hot_only._scheduler_config.paged_ssd_cache_max_size = 50 * 1024 * 1024
        pool_hot_only._scheduler_config.hot_cache_only = True

        engine_hot_only = pool_hot_only._splash_engine_for(
            _entry(package, "splash"), None, "splash"
        )
        assert engine_hot_only._cache_disk_bytes == 0
        assert engine_hot_only._cache_disk_dir is None

    def test_package_path_passes_the_missing_model_check(self, tmp_path):
        package = _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        pool = EnginePool()
        entry = _entry(package, "splash")
        pool._entries = {entry.model_id: entry}

        pool._raise_if_model_path_missing_locked(entry.model_id, entry)

        assert entry.model_id in pool._entries

    async def test_load_and_unload_skip_mlx_memory_barrier(
        self, tmp_path, hub, splash_installed
    ):
        _write_package(tmp_path / "models" / "incoai" / "Qwen3.8-27B-Splash")
        _cache_repo(hub, "incoai/Qwen3.8-27B-Splash")
        pool = EnginePool()
        pool._get_final_ceiling = lambda: 0
        pool.discover_models(str(tmp_path / "models"))
        model_id = "Qwen3.8-27B-Splash"

        async def fake_start(self):
            self._tokenizer = MagicMock()

        with (
            patch.object(SplashEngine, "start", fake_start),
            patch.object(SplashEngine, "stop", AsyncMock()) as stop,
        ):
            await pool._load_engine(model_id)
            entry = pool.get_entry(model_id)
            assert isinstance(entry.engine, SplashEngine)
            assert entry.actual_size == entry.estimated_size

            with patch("omlx.engine_pool.mx") as mx:
                await pool._unload_engine(model_id)

        stop.assert_awaited_once()
        mx.get_active_memory.assert_not_called()
        assert entry.engine is None
        assert pool._current_model_memory == 0


# -- native response_format ---------------------------------------------------


class TestNativeResponseFormat:
    def test_native_response_format_pydantic_json_schema(self):
        from omlx.api.openai_models import ResponseFormat, ResponseFormatJsonSchema
        from omlx.server import _native_response_format

        engine = MagicMock(supports_native_response_format=True)
        rf = ResponseFormat(
            type="json_schema",
            json_schema=ResponseFormatJsonSchema(
                name="person",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                strict=True,
            ),
        )

        wire = _native_response_format(engine, rf)
        assert wire is not None
        assert wire["type"] == "json_schema"
        assert wire["json_schema"]["name"] == "person"
        assert "schema" in wire["json_schema"]
        assert "schema_" not in wire["json_schema"]
        assert wire["json_schema"]["schema"] == {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        assert wire["json_schema"]["strict"] is True

    def test_native_response_format_engine_without_flag(self):
        from omlx.api.openai_models import ResponseFormat
        from omlx.server import _native_response_format

        engine = MagicMock(spec=[])
        rf = ResponseFormat(type="json_object")
        assert _native_response_format(engine, rf) is None

        engine_false = MagicMock(supports_native_response_format=False)
        assert _native_response_format(engine_false, rf) is None

    def test_native_response_format_none_when_structured_outputs(self):
        from omlx.api.openai_models import ResponseFormat
        from omlx.server import _native_response_format

        engine = MagicMock(supports_native_response_format=True)
        rf = ResponseFormat(type="json_object")
        assert (
            _native_response_format(engine, rf, structured_outputs={"json_schema": {}})
            is None
        )

    def test_native_response_format_none_when_format_none(self):
        from omlx.server import _native_response_format

        engine = MagicMock(supports_native_response_format=True)
        assert _native_response_format(engine, None) is None

    def test_native_response_format_dict_returns_copy(self):
        from omlx.server import _native_response_format

        engine = MagicMock(supports_native_response_format=True)
        rf = {"type": "json_object"}
        res = _native_response_format(engine, rf)
        assert res == rf
        assert res is not rf

    def test_chat_completions_forwards_native_response_format(self, monkeypatch):
        from types import SimpleNamespace

        from fastapi import HTTPException
        from fastapi.testclient import TestClient

        import omlx.server as srv

        captured = {}

        async def fake_preflight(*args, **kwargs):
            captured.update(kwargs)
            raise HTTPException(status_code=418, detail="Captured")

        engine = MagicMock()
        engine.supports_native_response_format = True
        engine.model_type = "qwen3_5"
        engine.is_diffusion_model = False
        engine.start = AsyncMock()
        engine.preflight_chat = AsyncMock(side_effect=fake_preflight)
        engine.count_chat_tokens.return_value = 10

        pool = MagicMock()
        pool.preload_pinned_models = AsyncMock()
        pool.check_ttl_expirations = AsyncMock()
        pool.shutdown = AsyncMock()
        pool.get_entry.return_value = SimpleNamespace(
            config_model_type="qwen3_5",
            preserve_thinking_default=False,
        )

        monkeypatch.setattr(srv._server_state, "engine_pool", pool)
        monkeypatch.setattr(srv, "get_engine_for_model", AsyncMock(return_value=engine))
        monkeypatch.setattr(srv, "resolve_model_id", lambda name: name)
        monkeypatch.setattr(srv, "validate_context_window", lambda *a, **k: None)
        from omlx.model_settings import ModelSettings

        monkeypatch.setattr(
            srv,
            "get_model_settings_for_request",
            lambda name: ModelSettings(cache_reasoning_output=False),
        )
        monkeypatch.setitem(
            srv.app.dependency_overrides, srv.verify_inference_api_key, lambda: True
        )

        with TestClient(srv.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "splash-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "response_format": {"type": "json_object"},
                },
            )

        assert response.status_code == 418
        assert captured.get("response_format") == {"type": "json_object"}
        assert "Warning" not in response.headers

    def test_chat_completions_fallback_without_native_support(self, monkeypatch):
        from types import SimpleNamespace

        from fastapi import HTTPException
        from fastapi.testclient import TestClient

        import omlx.server as srv
        from omlx.model_settings import ModelSettings

        captured = {}

        async def fake_preflight(*args, **kwargs):
            captured.update(kwargs)
            raise HTTPException(status_code=418, detail="Captured")

        engine = MagicMock(supports_native_response_format=False)
        engine.model_type = "qwen3_5"
        engine.is_diffusion_model = False
        engine.start = AsyncMock()
        engine.preflight_chat = AsyncMock(side_effect=fake_preflight)
        engine.count_chat_tokens.return_value = 10

        pool = MagicMock()
        pool.preload_pinned_models = AsyncMock()
        pool.check_ttl_expirations = AsyncMock()
        pool.shutdown = AsyncMock()
        pool.get_entry.return_value = SimpleNamespace(
            config_model_type="qwen3_5",
            preserve_thinking_default=False,
        )

        monkeypatch.setattr(srv._server_state, "engine_pool", pool)
        monkeypatch.setattr(srv, "get_engine_for_model", AsyncMock(return_value=engine))
        monkeypatch.setattr(srv, "resolve_model_id", lambda name: name)
        monkeypatch.setattr(srv, "validate_context_window", lambda *a, **k: None)
        monkeypatch.setattr(
            srv,
            "get_model_settings_for_request",
            lambda name: ModelSettings(cache_reasoning_output=False),
        )
        monkeypatch.setitem(
            srv.app.dependency_overrides, srv.verify_inference_api_key, lambda: True
        )

        with TestClient(srv.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "other-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "response_format": {"type": "json_object"},
                },
            )

        assert response.status_code == 418
        assert "response_format" not in captured

    def test_responses_stream_forwards_native_response_format(self, monkeypatch):
        from types import SimpleNamespace

        from fastapi.testclient import TestClient

        import omlx.server as srv
        from omlx.engine.base import GenerationOutput
        from omlx.model_settings import ModelSettings

        captured = {}

        async def fake_stream_chat(messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            yield GenerationOutput(
                text='{"name": "test"}',
                new_text='{"name": "test"}',
                prompt_tokens=4,
                completion_tokens=6,
                finish_reason="stop",
                finished=True,
            )

        engine = MagicMock()
        engine.tokenizer = None
        engine.supports_native_response_format = True
        engine.model_type = "qwen3_5"
        engine.is_diffusion_model = False
        engine.start = AsyncMock()
        engine.preflight_chat = AsyncMock(return_value=None)
        engine.stream_chat = fake_stream_chat
        engine.count_chat_tokens.return_value = 10

        pool = MagicMock()
        pool.preload_pinned_models = AsyncMock()
        pool.check_ttl_expirations = AsyncMock()
        pool.shutdown = AsyncMock()
        pool.get_entry.return_value = SimpleNamespace(
            config_model_type="qwen3_5",
            preserve_thinking_default=False,
        )

        monkeypatch.setattr(srv._server_state, "engine_pool", pool)
        monkeypatch.setattr(srv, "get_engine_for_model", AsyncMock(return_value=engine))
        monkeypatch.setattr(srv, "resolve_model_id", lambda name: name)
        monkeypatch.setattr(srv, "validate_context_window", lambda *a, **k: None)
        monkeypatch.setattr(
            srv,
            "get_model_settings_for_request",
            lambda name: ModelSettings(cache_reasoning_output=False),
        )
        monkeypatch.setitem(
            srv.app.dependency_overrides, srv.verify_inference_api_key, lambda: True
        )

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        with TestClient(srv.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "splash-model",
                    "input": "hi",
                    "stream": True,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "user_schema",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                },
            )

        assert response.status_code == 200
        assert "Warning" not in response.headers
        rf = captured["kwargs"].get("response_format")
        assert rf is not None
        assert "schema" in (rf.get("json_schema") or rf)
        messages = captured["messages"]
        assert len(messages) == 1
        assert messages[0]["content"] == "hi"
        assert not any("JSON" in str(m.get("content", "")) for m in messages)

    def test_responses_chat_forwards_native_response_format(self, monkeypatch):
        from types import SimpleNamespace

        from fastapi.testclient import TestClient

        import omlx.server as srv
        from omlx.api.responses_utils import ResponseStore
        from omlx.engine.base import GenerationOutput
        from omlx.model_settings import ModelSettings

        captured = {}

        async def fake_chat(messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return GenerationOutput(
                text='{"name": "test"}',
                prompt_tokens=4,
                completion_tokens=6,
                finish_reason="stop",
                finished=True,
            )

        engine = MagicMock()
        engine.tokenizer = None
        engine.supports_native_response_format = True
        engine.model_type = "qwen3_5"
        engine.is_diffusion_model = False
        engine.start = AsyncMock()
        engine.preflight_chat = AsyncMock(return_value=None)
        engine.chat = AsyncMock(side_effect=fake_chat)
        engine.count_chat_tokens.return_value = 10

        pool = MagicMock()
        pool.preload_pinned_models = AsyncMock()
        pool.check_ttl_expirations = AsyncMock()
        pool.shutdown = AsyncMock()
        pool.get_entry.return_value = SimpleNamespace(
            config_model_type="qwen3_5",
            preserve_thinking_default=False,
        )

        monkeypatch.setattr(srv._server_state, "engine_pool", pool)
        monkeypatch.setattr(srv._server_state, "responses_store", ResponseStore())
        monkeypatch.setattr(srv, "get_engine_for_model", AsyncMock(return_value=engine))
        monkeypatch.setattr(srv, "resolve_model_id", lambda name: name)
        monkeypatch.setattr(srv, "validate_context_window", lambda *a, **k: None)
        monkeypatch.setattr(
            srv,
            "get_model_settings_for_request",
            lambda name: ModelSettings(cache_reasoning_output=False),
        )
        monkeypatch.setitem(
            srv.app.dependency_overrides, srv.verify_inference_api_key, lambda: True
        )

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        with TestClient(srv.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "splash-model",
                    "input": "hi",
                    "stream": False,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "user_schema",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                },
            )

        assert response.status_code == 200
        assert "Warning" not in response.headers
        rf = captured["kwargs"].get("response_format")
        assert rf is not None
        assert "schema" in (rf.get("json_schema") or rf)
        messages = captured["messages"]
        assert len(messages) == 1
        assert messages[0]["content"] == "hi"
        assert not any("JSON" in str(m.get("content", "")) for m in messages)


class TestProcessMemory:
    def test_list_process_group_pids(self):
        from omlx.utils.proc_memory import list_process_group_pids

        current_pgid = os.getpgid(0)
        pids = list_process_group_pids(current_pgid)
        assert os.getpid() in pids

        # Invalid pgids return empty list
        assert list_process_group_pids(-1) == []
        assert list_process_group_pids(99999999) == []

    def test_process_and_peak_memory_bytes(self, tmp_path, monkeypatch):
        engine = _engine_with_transport(tmp_path, lambda r: httpx.Response(200))
        # Not running: memory returns 0
        assert engine.process_memory_bytes() == 0
        assert engine.peak_memory_bytes() == 0

        # Running process
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.pid = 4200
        engine._process = fake_proc

        monkeypatch.setattr(
            "omlx.engine.splash.list_process_group_pids",
            lambda pgid: [4200, 4201] if pgid == 4200 else [],
        )
        monkeypatch.setattr(
            "omlx.engine.splash.get_phys_footprint",
            lambda pid: {4200: 50 * 1024 * 1024, 4201: 70 * 1024 * 1024}.get(pid, 0),
        )
        monkeypatch.setattr(
            "omlx.engine.splash.get_lifetime_max_phys_footprint",
            lambda pid: {4200: 80 * 1024 * 1024, 4201: 90 * 1024 * 1024}.get(pid, 0),
        )

        assert engine.process_memory_bytes() == 120 * 1024 * 1024
        assert engine.peak_memory_bytes() == 90 * 1024 * 1024
        assert engine.get_stats()["process_memory_bytes"] == 120 * 1024 * 1024

    def test_mapped_weights_floor_memory(self, tmp_path, monkeypatch):
        # Splash 1.1 maps its weights; the kernel charges them to no process.
        package = _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        (package / "target.bin").write_bytes(b"0" * 1_000_000)
        engine = SplashEngine(str(package), _build())
        engine._process = MagicMock(pid=4200, **{"poll.return_value": None})
        monkeypatch.setattr(splash_mod, "list_process_group_pids", lambda pgid: [4200])
        monkeypatch.setattr(splash_mod, "get_phys_footprint", lambda pid: 1000)
        monkeypatch.setattr(
            splash_mod, "get_lifetime_max_phys_footprint", lambda pid: 2000
        )
        weights = sum(f.stat().st_size for f in package.rglob("*") if f.is_file())

        assert engine.process_memory_bytes() == weights
        assert engine.peak_memory_bytes() == weights

    def test_process_memory_prefers_splash_actual_memory(self, tmp_path, monkeypatch):
        package = _write_package(tmp_path / "incoai" / "Qwen3.8-27B-Splash")
        (package / "target.bin").write_bytes(b"0" * 100_000)
        engine = SplashEngine(str(package), _build())
        engine._process = MagicMock(pid=4200, **{"poll.return_value": None})
        monkeypatch.setattr(splash_mod, "list_process_group_pids", lambda pgid: [4200])
        monkeypatch.setattr(splash_mod, "get_phys_footprint", lambda pid: 200_000)
        monkeypatch.setattr(
            splash_mod, "get_lifetime_max_phys_footprint", lambda pid: 300_000
        )

        # Without snapshot: floored by max(footprint, weights)
        assert engine.process_memory_bytes() == 200_000
        assert engine.peak_memory_bytes() == 300_000

        # With snapshot having larger memory_actual
        engine._status = {
            "memory_actual": {
                "current_bytes": 500_000,
                "peak_bytes": 700_000,
            }
        }
        assert engine.process_memory_bytes() == 500_000
        assert engine.peak_memory_bytes() == 700_000

        # With snapshot having smaller memory_actual: footprint still wins
        engine._status = {
            "memory_actual": {
                "current_bytes": 150_000,
                "peak_bytes": 250_000,
            }
        }
        assert engine.process_memory_bytes() == 200_000
        assert engine.peak_memory_bytes() == 300_000

    def test_benchmark_endpoint_requires_started_engine(self, tmp_path):
        engine = _engine_with_transport(tmp_path, lambda r: httpx.Response(200))
        with pytest.raises(RuntimeError, match="not started"):
            engine.benchmark_endpoint()

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.pid = 4200
        engine._process = fake_proc
        engine._base_url = "http://127.0.0.1:8123"
        engine._api_key = "secret"
        engine._repo_id = "test/repo"

        assert engine.benchmark_endpoint() == {
            "base_url": "http://127.0.0.1:8123/v1",
            "api_key": "secret",
            "model": "test/repo",
            "extra_body": {"reasoning_effort": "none"},
        }

    def test_engine_pool_status_reports_external_process_memory(self):
        pool = EnginePool()
        engine1 = MagicMock()
        engine1.process_memory_bytes.return_value = 300 * 1024 * 1024
        entry1 = EngineEntry(
            model_id="model-1",
            model_path="/tmp/model1",
            model_type="llm",
            engine_type="batched",
            estimated_size=100,
            engine=engine1,
        )

        engine2 = MagicMock()
        engine2.process_memory_bytes.return_value = 200 * 1024 * 1024
        entry2 = EngineEntry(
            model_id="model-2",
            model_path="/tmp/model2",
            model_type="llm",
            engine_type="batched",
            estimated_size=100,
            engine=engine2,
        )

        pool._entries = {"model-1": entry1, "model-2": entry2}
        status = pool.get_status()

        assert status["external_process_memory"] == 500 * 1024 * 1024
        models_by_id = {m["id"]: m for m in status["models"]}
        assert models_by_id["model-1"]["actual_size"] == 300 * 1024 * 1024
        assert models_by_id["model-2"]["actual_size"] == 200 * 1024 * 1024


class TestSplashBenchmark:
    async def test_warmup_calibrates_template_overhead(self):
        from omlx.admin.benchmark import _CHAT_CALIBRATION_TOKENS, _ChatBenchmark
        from omlx.admin.external_api import StreamStats

        engine = MagicMock()
        engine.benchmark_endpoint.return_value = {
            "base_url": "http://x/v1",
            "model": "m",
        }
        engine.count_chat_tokens.return_value = 40
        engine.tokenizer.encode.side_effect = lambda text, add_special_tokens: list(
            range(10_000)
        )
        engine.tokenizer.decode.side_effect = lambda ids: " ".join(["t"] * len(ids))
        bench = _ChatBenchmark(engine, "code_python")
        stats = MagicMock(spec=StreamStats, prompt_tokens=_CHAT_CALIBRATION_TOKENS - 40)
        bench.client.stream_chat_completion = AsyncMock(return_value=stats)

        await bench.warmup(32)

        # The engine's template is 40 tokens shorter than the tokenizer's.
        assert bench.overhead == 0

    def test_chat_benchmark_for_engine_detection(self):
        from omlx.admin.benchmark import _ChatBenchmark

        mlx_engine = MagicMock(spec=[])
        assert _ChatBenchmark.for_engine(mlx_engine, "code_python") is None

        splash_engine = MagicMock()
        splash_engine.benchmark_endpoint.return_value = {
            "base_url": "http://127.0.0.1:8123/v1",
            "api_key": "key",
            "model": "splash-model",
        }
        splash_engine.count_chat_tokens.return_value = 4
        splash_engine.tokenizer = MagicMock()

        bench = _ChatBenchmark.for_engine(splash_engine, "code_python")
        assert bench is not None
        assert bench.overhead == 4

    def test_chat_benchmark_prompt(self):
        from omlx.admin.benchmark import _ChatBenchmark

        engine = MagicMock()
        engine.benchmark_endpoint.return_value = {
            "base_url": "http://127.0.0.1:8123/v1",
            "api_key": "key",
            "model": "splash-model",
        }
        engine.count_chat_tokens.return_value = 2
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: list(
            range(len(text))
        )
        tokenizer.decode.side_effect = lambda ids: f"tokens:{len(ids)}"
        engine.tokenizer = tokenizer

        bench = _ChatBenchmark(engine, "code_python")
        prompt = bench.prompt(100)
        assert prompt == "tokens:98"

    async def test_chat_benchmark_methods(self):
        from omlx.admin.benchmark import _ChatBenchmark
        from omlx.admin.external_api import StreamStats

        engine = MagicMock()
        engine.benchmark_endpoint.return_value = {
            "base_url": "http://127.0.0.1:8123/v1",
            "api_key": "key",
            "model": "splash-model",
        }
        engine.peak_memory_bytes.return_value = 500 * 1024 * 1024
        engine.count_chat_tokens.return_value = 0
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "prompt"
        engine.tokenizer = tokenizer

        bench = _ChatBenchmark(engine, "code_python")
        mock_client = MagicMock()
        mock_client.stream_chat_completion = AsyncMock(
            return_value=StreamStats(
                prompt_tokens=50,
                completion_tokens=10,
                cached_tokens=0,
                start_time=1.0,
                first_content_time=1.2,
                last_content_time=1.5,
                end_time=1.5,
                text="out",
                content_observed=True,
            )
        )
        mock_client.aclose = AsyncMock()
        bench.client = mock_client

        await bench.warmup(10)
        mock_client.stream_chat_completion.assert_awaited_once()

        single = await bench.single(50, 10)
        assert single["prompt_tokens"] == 50
        assert single["peak_memory_bytes"] == 500 * 1024 * 1024

        batch = await bench.batch(2, 10)
        assert batch["prompt_tokens"] == 50
        assert batch["batch_size"] == 2
        assert batch["peak_memory_bytes"] == 500 * 1024 * 1024

        await bench.aclose()
        mock_client.aclose.assert_awaited_once()

    async def test_run_benchmark_against_splash_endpoint(self, monkeypatch):
        from types import SimpleNamespace

        from omlx.admin.benchmark import (
            BenchmarkRequest,
            BenchmarkRun,
            run_benchmark,
        )
        from omlx.admin.external_api import StreamStats

        def fake_stream(**kwargs):
            return StreamStats(
                prompt_tokens=1000,
                completion_tokens=100,
                cached_tokens=0,
                start_time=0.0,
                first_content_time=0.5,
                last_content_time=1.5,
                end_time=1.5,
                text="benchmark output",
                content_observed=True,
            )

        mock_client = MagicMock()
        mock_client.stream_chat_completion = AsyncMock(side_effect=fake_stream)
        mock_client.aclose = AsyncMock()

        monkeypatch.setattr(
            "omlx.admin.benchmark.ExternalAPIClient",
            lambda config: mock_client,
        )

        mock_engine = MagicMock()
        mock_engine.benchmark_endpoint.return_value = {
            "base_url": "http://127.0.0.1:8123/v1",
            "api_key": "key",
            "model": "splash-model",
            "extra_body": {"reasoning_effort": "none"},
        }
        mock_engine.peak_memory_bytes.return_value = 999 * 1024 * 1024
        mock_engine.count_chat_tokens.return_value = 3
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(2000))
        tokenizer.decode.return_value = "benchmark-prompt"
        mock_engine.tokenizer = tokenizer

        pool = MagicMock()
        pool.get_loaded_model_ids.return_value = []
        pool.get_engine = AsyncMock(return_value=mock_engine)
        pool._unload_engine = AsyncMock()
        pool.get_entry.return_value = SimpleNamespace(
            model_type="qwen3_5",
            model_settings=None,
            model_path=Path("/tmp/model"),
        )

        run = BenchmarkRun(
            bench_id="splash-run-1",
            request=BenchmarkRequest(
                model_id="splash-model",
                prompt_lengths=[1024],
                batch_sizes=[2],
            ),
        )

        await run_benchmark(run, pool)

        mock_client.aclose.assert_awaited_once()

        single_result = next(r for r in run.results if r["test_type"] == "single")
        assert single_result["requested_pp"] == 1024
        assert single_result["pp"] == 1000
        assert single_result["peak_memory_bytes"] == 999 * 1024 * 1024

        batch_result = next(r for r in run.results if r["test_type"] == "batch")
        assert batch_result["requested_pp"] == 1024
        assert batch_result["pp"] == 1000
        assert batch_result["batch_size"] == 2
        assert batch_result["peak_memory_bytes"] == 999 * 1024 * 1024

        # Upload must be skipped for splash
        assert run.upload_state["phase"] == "skipped"
        assert run.upload_state["skipped_reason"] == "splash"
        event_types = [e["type"] for e in run.events]
        assert "upload_skipped" in event_types
        skipped_event = next(e for e in run.events if e["type"] == "upload_skipped")
        assert skipped_event["reason"] == "splash"


class TestSplashRuntimeCache:
    def test_get_runtime_cache_stats_mapping(self, tmp_path):
        engine = _engine_with_transport(tmp_path, lambda r: httpx.Response(200))
        assert engine.get_runtime_cache_stats() is None

        status_dict = {
            "block_tokens": 256,
            "kv": {
                "block_tokens": 256,
                "pages_total": 100,
                "pages_cache": 42,
                "pages_free": 58,
                "resident_backing_bytes": 10240,
            },
            "state": {
                "entries": 5,
                "bytes": 50000,
                "evictions": 3,
                "hits": 10,
                "misses": 2,
            },
            "disk": {
                "capacity_bytes": 1000000,
                "used_bytes": 200000,
                "kv_blocks": 50,
                "kv_bytes": 150000,
                "kv_demotions": 15,
                "kv_restores": 8,
            },
            "cache": {
                "lookups": 30,
                "hits": 20,
                "cold_misses": 10,
                "hit_rate": 0.6667,
                "kv_hit_tokens": 4000,
                "kv_disk_hit_tokens": 1500,
                "reused_tokens": 5500,
            },
            "memory_actual": {
                "current_bytes": 1234567,
                "peak_bytes": 2345678,
            },
        }
        engine._status = status_dict
        stats = engine.get_runtime_cache_stats()
        assert stats is not None
        assert stats["block_size"] == 256
        assert stats["indexed_blocks"] == 42
        assert stats["ssd_cache"] == {
            "num_files": 1,
            "total_size_bytes": 200000,
            "max_size_bytes": 1000000,
            "hits": 20,
            "misses": 10,
            "evictions": 3,
            "loads": 8,
            "saves": 15,
        }
        assert stats["prefix_cache"] == {"block_size": 256}
        assert stats["cache_rates"]["cumulative"]["prefix_hits"] == 20
        assert stats["cache_rates"]["cumulative"]["prefix_misses"] == 10
        assert stats["cache_rates"]["cumulative"]["evictions"] == 3
        assert stats["cache_rates"]["cumulative"]["ssd_disk_loads"] == 8
        assert stats["cache_rates"]["cumulative"]["ssd_saves"] == 15
        assert stats["splash_cache"] == {
            "reused_tokens": 5500,
            "kv_hit_tokens": 4000,
            "kv_disk_hit_tokens": 1500,
            "state_entries": 5,
            "state_bytes": 50000,
        }

        # Tolerates missing disk and state
        engine._status = {
            "block_tokens": 128,
            "kv": {"pages_cache": 7},
            "cache": {"hits": 4, "cold_misses": 1},
        }
        stats_min = engine.get_runtime_cache_stats()
        assert stats_min is not None
        assert stats_min["block_size"] == 128
        assert stats_min["indexed_blocks"] == 7
        assert stats_min["ssd_cache"]["num_files"] == 0
        assert stats_min["ssd_cache"]["total_size_bytes"] == 0
        assert stats_min["ssd_cache"]["max_size_bytes"] == 0
        assert stats_min["ssd_cache"]["evictions"] == 0
        assert stats_min["ssd_cache"]["loads"] == 0
        assert stats_min["ssd_cache"]["saves"] == 0
        assert stats_min["splash_cache"] == {
            "reused_tokens": 0,
            "kv_hit_tokens": 0,
            "kv_disk_hit_tokens": 0,
            "state_entries": 0,
            "state_bytes": 0,
        }

    @pytest.mark.asyncio
    async def test_status_poller_and_stop_cancellation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(splash_mod, "STATUS_POLL_SECONDS", 0.01)
        status_payload = {
            "kv": {"block_tokens": 256, "pages_cache": 15},
            "disk": {"used_bytes": 1024},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/status":
                return httpx.Response(200, json=status_payload)
            return httpx.Response(404)

        engine = _engine_with_transport(tmp_path, handler)
        assert engine._status is None

        engine._status_task = asyncio.create_task(engine._poll_status())
        await asyncio.sleep(0.03)

        assert engine._status == status_payload
        assert engine._status_task is not None
        assert not engine._status_task.done()

        await engine.stop()
        assert engine._status_task is None

    def test_build_runtime_cache_observability_includes_splash_and_mlx(self, tmp_path):
        from types import SimpleNamespace

        from omlx.admin import routes as admin_routes

        cache_dir = tmp_path / "ssd_cache"
        settings = SimpleNamespace(
            base_path=tmp_path,
            cache=SimpleNamespace(
                ssd_cache_max_size="auto",
                get_ssd_cache_dir=lambda base_path: cache_dir,
                get_ssd_cache_max_size_bytes=lambda base_path: 0,
            ),
        )

        class FakeMlxEngine:
            scheduler = None

            def get_runtime_cache_stats(self):
                return {
                    "block_size": 512,
                    "indexed_blocks": 25,
                    "ssd_cache": {
                        "num_files": 2,
                        "total_size_bytes": 300,
                        "max_size_bytes": 1000,
                        "hot_cache_max_bytes": 2048,
                        "hot_cache_size_bytes": 512,
                        "hot_cache_entries": 2,
                        "hits": 10,
                        "misses": 5,
                    },
                    "cache_rates": {"cumulative": {"prefix_hits": 10}},
                }

        splash_engine = _engine_with_transport(tmp_path, lambda r: httpx.Response(200))
        splash_engine._status = {
            "kv": {"block_tokens": 256, "pages_cache": 40},
            "disk": {
                "capacity_bytes": 2000,
                "used_bytes": 500,
                "kv_demotions": 3,
                "kv_restores": 2,
            },
            "cache": {
                "hits": 15,
                "cold_misses": 5,
                "reused_tokens": 1200,
                "kv_hit_tokens": 800,
                "kv_disk_hit_tokens": 400,
            },
            "state": {"entries": 6, "bytes": 24000, "evictions": 1},
        }

        class FakePool:
            def __init__(self, entries):
                self._entries = entries

            def get_status(self):
                return {
                    "models": [
                        {"id": "mlx-model", "loaded": True},
                        {"id": "splash-model", "loaded": True},
                    ]
                }

        pool = FakePool(
            {
                "mlx-model": SimpleNamespace(engine=FakeMlxEngine()),
                "splash-model": SimpleNamespace(engine=splash_engine),
            }
        )

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            payload = admin_routes._build_runtime_cache_observability(settings)

        assert len(payload["models"]) == 2
        mlx_row = next(m for m in payload["models"] if m["id"] == "mlx-model")
        splash_row = next(m for m in payload["models"] if m["id"] == "splash-model")

        assert mlx_row.get("cache_tier") != "splash"
        assert "splash_cache" not in mlx_row
        assert mlx_row["total_size_bytes"] == 300
        assert mlx_row["num_files"] == 2

        assert splash_row["cache_tier"] == "splash"
        assert splash_row["splash_cache"] == {
            "reused_tokens": 1200,
            "kv_hit_tokens": 800,
            "kv_disk_hit_tokens": 400,
            "state_entries": 6,
            "state_bytes": 24000,
        }
        assert splash_row["total_size_bytes"] == 500
        assert splash_row["num_files"] == 1
        assert splash_row["block_size"] == 256
        assert splash_row["indexed_blocks"] == 40

        # Total size bytes sums both models
        assert payload["total_size_bytes"] == 300 + 500
        assert payload["total_num_files"] == 2 + 1
