# SPDX-License-Identifier: Apache-2.0
"""Tests for the Sushi external engine and its pool and settings wiring."""

import asyncio
import json
import os
import stat
import sys
import textwrap
from types import SimpleNamespace

import pytest

from omlx.engine import splash as splash_mod
from omlx.engine.splash import EXTERNAL_ENGINES, SplashEngine
from omlx.engine.sushi import (
    SushiEngine,
    SushiOptions,
    SushiTarget,
    find_sushi_binary,
    inspect_sushi_binary,
    is_sushi_pack,
    parse_size_bytes,
    pick_effort,
    resolve_sushi_target,
    sushi_memory_estimate,
)
from omlx.settings import IntegrationSettings

QWEN38_EFFORTS = ["off", "low", "medium", "xhigh"]

FAKE_SUSHI = textwrap.dedent(
    r'''
    #!{python}
    """A stand-in for `sushi`: --version, and `serve` with a tiny OpenAI API."""
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    args = sys.argv[1:]
    if args == ["--version"]:
        print("sushi 1.0.0 (test)")
        sys.exit(0)
    assert args[0] == "serve", args
    port = int(args[args.index("--port") + 1])
    model = args[args.index("--model") + 1]
    key = os.environ[args[args.index("--api-key-env") + 1]]
    with open(os.environ["FAKE_SUSHI_ARGV"], "w") as f:
        json.dump(args, f)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self._json(200, {"data": [{"id": model, "loaded": True,
                    "max_model_len": 500000,
                    "reasoning_efforts": ["off", "low", "medium", "xhigh"]}]})
            elif self.path == "/props":
                self._json(200, {"memory": {"active_bytes": 1234},
                    "settings": {"prefix_cache": {"ram_enabled": True,
                    "mem_bytes": 0, "disk_bytes": 100, "disk_used_bytes": 7,
                    "disk_entries": 1}}})
            else:
                self._json(404, {})

        def do_POST(self):
            assert self.headers["Authorization"] == f"Bearer {key}"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with open(os.environ["FAKE_SUSHI_BODY"], "w") as f:
                json.dump(body, f)
            if body.get("reasoning_effort") not in (None, "off", "low", "medium", "xhigh"):
                return self._json(400, {"error": "bad effort"})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 2}}},
            ]
            for c in chunks:
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
    '''
).lstrip()


@pytest.fixture
def fake_sushi(tmp_path):
    path = tmp_path / "bin" / "sushi"
    path.parent.mkdir()
    path.write_text(FAKE_SUSHI.replace("{python}", sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def sushi_pack(tmp_path):
    model = tmp_path / "models" / "beamster" / "Qwen3.8-Flash-Next-Sushi-4bpw"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "language_model_only": True,
                "expert_quant": {"format": "exl3", "k": 4, "codebook": "mcg"},
                "text_config": {"max_position_embeddings": 1048576},
            }
        )
    )
    (model / "model-00001.safetensors").write_bytes(b"\0" * 4096)
    return model


def test_registered():
    assert EXTERNAL_ENGINES["sushi"] == "Sushi"
    assert issubclass(SushiEngine, SplashEngine)
    assert SushiEngine.external_engine_id == "sushi"


def test_is_sushi_pack():
    assert is_sushi_pack({"expert_quant": {"format": "exl3"}})
    assert is_sushi_pack({"text_config": {"expert_quant": {"format": "EXL3"}}})
    assert not is_sushi_pack({"model_type": "qwen3"})


@pytest.mark.parametrize(
    "requested, expected",
    [
        (None, None),
        ("off", "off"),
        ("none", "off"),
        ("low", "low"),
        ("minimal", "low"),
        ("high", "xhigh"),
        ("max", "xhigh"),
        ("on", "low"),
        ("bogus", None),
    ],
)
def test_pick_effort_qwen(requested, expected):
    assert pick_effort(requested, QWEN38_EFFORTS) == expected


def test_pick_effort_glm_has_no_off():
    assert pick_effort("off", ["low", "high", "max"]) is None
    assert pick_effort("medium", ["low", "high", "max"]) == "high"


def test_parse_size_bytes():
    assert parse_size_bytes("20GB") == 20 * 1024**3
    assert parse_size_bytes("512MB") == 512 * 1024**2
    assert parse_size_bytes(None) == 0
    assert parse_size_bytes("nope") == 0


def test_memory_estimate(sushi_pack):
    est = sushi_memory_estimate(sushi_pack, 500_000, "8")
    assert est >= 4096 + 16_640 * 500_000
    assert sushi_memory_estimate(sushi_pack, 500_000, "4") < est


def test_find_binary(fake_sushi):
    binary, reason = find_sushi_binary(str(fake_sushi))
    assert binary is not None, reason
    assert binary.version.startswith("1.0.0")
    missing, reason = inspect_sushi_binary(fake_sushi.parent / "nope", auto_detected=False)
    assert missing is None and "not an executable" in reason


def test_resolve_target(fake_sushi, sushi_pack, tmp_path):
    target, reason = resolve_sushi_target(sushi_pack, str(fake_sushi))
    assert target is not None, reason
    assert target.model_type == "qwen4_exp" and target.exl3
    other = tmp_path / "plain"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "llama"}))
    target, reason = resolve_sushi_target(other, str(fake_sushi))
    assert target is None and "llama" in reason


def test_serve_command_matches_recommended_launch(fake_sushi, sushi_pack, tmp_path):
    binary, _ = find_sushi_binary(str(fake_sushi))
    engine = SushiEngine(
        model_name=str(sushi_pack),
        model_id="beamster/Qwen3.8-Flash-Next-Sushi-4bpw",
        target=SushiTarget(binary, "qwen4_exp", True),
        log_dir=tmp_path / "logs",
        options=SushiOptions(),
    )
    cmd = engine._serve_command(4321)
    joined = " ".join(cmd)
    for flag in (
        "--mtp",
        "--kv-quant 8",
        "--mtp-head-kv-quant",
        "--ctx-size 500000",
        "--prefill-chunk 2048",
        "--max-tokens 64000",
        "--prefix-cache-disk 20GB",
        "--prefix-cache-entries 1",
        "--temp 1",
        "--port 4321",
        f"--parent-pid {os.getpid()}",
    ):
        assert flag in joined, flag
    assert "--prefix-cache-mem" not in cmd


def test_settings_roundtrip():
    s = IntegrationSettings()
    assert s.sushi_ctx_size == 500000 and s.sushi_kv_quant == "8"
    s.sushi_path = "/x/sushi"
    s.sushi_extra_args = "--mtp-typical 0.2"
    back = IntegrationSettings.from_dict(s.to_dict())
    assert back.sushi_path == "/x/sushi"
    assert back.sushi_extra_args == "--mtp-typical 0.2"
    assert back.sushi_prefix_cache_disk == "20GB"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_engine_end_to_end(fake_sushi, sushi_pack, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SUSHI_ARGV", str(tmp_path / "argv.json"))
    monkeypatch.setenv("FAKE_SUSHI_BODY", str(tmp_path / "body.json"))
    monkeypatch.setattr(splash_mod, "STARTUP_POLL_SECONDS", 0.1)
    # Linux reports a zombie group leader as alive until it is reaped; macOS
    # does not. Keep stop() from waiting out the full timeout there.
    monkeypatch.setattr(splash_mod, "STOP_TIMEOUT_SECONDS", 1)
    binary, _ = find_sushi_binary(str(fake_sushi))
    engine = SushiEngine(
        model_name=str(sushi_pack),
        model_id="beamster/Qwen3.8-Flash-Next-Sushi-4bpw",
        target=SushiTarget(binary, "qwen4_exp", True),
        log_dir=tmp_path / "logs",
        options=SushiOptions(extra_args=["--mtp-typical", "0.2"]),
        memory_budget_bytes=99,
    )
    # No tokenizer files in the fake pack: counts fall back to estimates.
    monkeypatch.setattr(
        SushiEngine, "_load_tokenizer", lambda self: setattr(self, "_tokenizer", None)
    )

    async def run():
        await engine.start()
        try:
            assert engine.max_context_tokens == 500000
            outputs = [
                o
                async for o in engine.stream_chat(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=16,
                    chat_template_kwargs={"reasoning_effort": "high"},
                    thinking_budget=512,
                    _request_id="req-1",
                )
            ]
            final = outputs[-1]
            assert final.finished
            assert final.text == "<think>hmm</think>Hello world"
            assert final.prompt_tokens == 5 and final.cached_tokens == 2
            body = json.loads((tmp_path / "body.json").read_text())
            assert body["reasoning_effort"] == "xhigh"
            assert body["reasoning_budget_tokens"] == 512
            assert "thinking_token_budget" not in body
            assert body["model"] == str(sushi_pack)
            await engine._refresh_status()
            assert engine._reported_memory("current_bytes") == 1234
            stats = engine.get_runtime_cache_stats()
            assert stats["sushi_cache"]["disk_used_bytes"] == 7
            assert engine.reserved_bytes() == 99
            assert engine.get_stats()["engine_type"] == "sushi"
        finally:
            await engine.stop()
        assert engine.runtime_failed_reason is None

    asyncio.run(run())
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[-2:] == ["--mtp-typical", "0.2"]


def test_sushi_pack_picks_sushi_without_settings(sushi_pack, tmp_path):
    from omlx.engine_pool import EnginePool

    pool = EnginePool.__new__(EnginePool)
    pool._entries = {
        "pack": SimpleNamespace(model_path=str(sushi_pack), model_format="mlx"),
        "gguf": SimpleNamespace(model_path=str(tmp_path / "x.gguf"), model_format="gguf"),
        "plain": SimpleNamespace(model_path=str(tmp_path), model_format="mlx"),
    }
    assert pool.external_engine_id("pack", None) == "sushi"
    assert pool.external_engine_id("gguf", None) == "splash"
    assert pool.external_engine_id("plain", None) is None
    assert pool.external_engine_id("pack", SimpleNamespace(external_engine="splash")) == "splash"
