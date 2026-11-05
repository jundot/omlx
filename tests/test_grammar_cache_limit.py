# SPDX-License-Identifier: Apache-2.0
"""Compiled-grammar cache budget (issue #4321).

xgrammar's ``GrammarCompiler`` was created with ``cache_limit_bytes=-1`` —
xgrammar's unbounded default — at every oMLX call site, so every JSON schema
the server had not seen before pinned its compiled grammar for as long as
the engine stayed loaded.  A workload with many distinct schemas drove the
process footprint into the prefill memory guard, which then rejected
otherwise small requests with HTTP 400 ``prefill_memory_exceeded``; the
retained bytes are not MLX memory, so the enforcer's eviction cannot
reclaim them.

These tests pin the bound where the engines resolve it, at the
``create_grammar_compiler`` default, and at both engine call sites, plus the
k2_horizon exception that keeps its pre-existing 64 MiB cap.
"""

import inspect
import subprocess
import sys
import types
from pathlib import Path

import pytest

from omlx.api import grammar

_REPO_ROOT = Path(__file__).resolve().parents[1]
ENV = "OMLX_GRAMMAR_CACHE_MAX_SIZE"
K2_LIMIT = 64 * 1024**2
_MAX_SANE_LIMIT = 2 * 1024**3

# "Use the module's own bound" / "leave the variable unset".  Resolved inside
# the test body so this file still collects against a checkout that predates
# the bound, where every case below is red rather than an uncollectable error.
_DEFAULT = object()
_UNSET = object()


class _PlainObject:
    """Stands in for a tokenizer / model; ``resolve_vocab_size`` says None."""


class _RecordingCompiler:
    def __init__(self, tokenizer_info, *, cache_limit_bytes):
        self.tokenizer_info = tokenizer_info
        self.cache_limit_bytes = cache_limit_bytes


class _FakeTokenizerInfo:
    @classmethod
    def from_huggingface(cls, hf_tokenizer, **kwargs):
        return cls()


@pytest.fixture
def fake_xgrammar(monkeypatch):
    """xgrammar is not installed in every environment; record its arguments."""
    module = types.ModuleType("xgrammar")
    module.TokenizerInfo = _FakeTokenizerInfo
    module.GrammarCompiler = _RecordingCompiler
    module.CompiledGrammar = object
    monkeypatch.setitem(sys.modules, "xgrammar", module)
    return module


def _resolve(monkeypatch, env, expected):
    """Apply the row's environment and unwind the ``_DEFAULT`` placeholder."""
    if env is _UNSET:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, env)
    if expected is _DEFAULT:
        return grammar.DEFAULT_GRAMMAR_CACHE_MAX_BYTES
    return expected


# ---------------------------------------------------------------------------
# The budget every engine call site resolves.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type,env,expected",
    [
        (None, _UNSET, _DEFAULT),  # the bounded default
        ("qwen3", "512MB", 512 * 1024**2),  # env override
        ("qwen3", "0", 0),  # caching off
        ("qwen3", "-1", -1),  # explicit opt-in to unbounded
        ("qwen3", "garbage", _DEFAULT),  # unusable value: unparsable ...
        ("qwen3", "-2", _DEFAULT),  # ... and out of range
        ("k2_horizon", _UNSET, K2_LIMIT),  # k2_horizon's existing cap ...
        ("k2_horizon", "2GB", K2_LIMIT),  # ... which no setting widens
    ],
)
def test_resolved_budget(monkeypatch, model_type, env, expected):
    expected = _resolve(monkeypatch, env, expected)
    assert grammar.grammar_cache_limit_bytes(model_type) == expected


# ---------------------------------------------------------------------------
# The default itself has to be bounded, so a caller that passes no limit --
# and the signature a reviewer reads -- cannot reintroduce the leak.
# ---------------------------------------------------------------------------


def test_default_is_a_finite_conservative_constant(monkeypatch, fake_xgrammar):
    monkeypatch.delenv(ENV, raising=False)
    default = grammar.DEFAULT_GRAMMAR_CACHE_MAX_BYTES
    assert 0 < default <= _MAX_SANE_LIMIT
    assert grammar.grammar_cache_limit_bytes() == default
    signature = inspect.signature(grammar.create_grammar_compiler)
    assert signature.parameters["cache_limit_bytes"].default == default
    compiler = grammar.create_grammar_compiler(_PlainObject(), _PlainObject())
    assert compiler.cache_limit_bytes == default


def test_explicit_limit_still_wins(fake_xgrammar):
    """-1 remains a supported opt-in; it just is not the default any more."""
    compiler = grammar.create_grammar_compiler(
        _PlainObject(), _PlainObject(), cache_limit_bytes=-1
    )
    assert compiler.cache_limit_bytes == -1


# ---------------------------------------------------------------------------
# Engine call sites.
#
# These have to import ``omlx.engine.*``, which imports ``omlx.scheduler``,
# which patches ``mlx_lm.generate.PromptProcessingBatch.prompt`` at import
# time and stashes the original in ``_omlx_base_prompt``.  Importing an engine
# into the pytest process is therefore not something this module can undo:
# evicting ``omlx.*`` from ``sys.modules`` afterwards makes a later test
# re-execute that module body, patch the already-patched function, and recurse
# forever in ``_patched_ppb_prompt``.  Leaving the patch in place instead
# hides the mlx_vlm import failures a whole file of later tests is supposed to
# hit.  Either way the session is contaminated, so the probe runs in a
# throw-away interpreter: the import, the patch and the stubs below all die
# with the child process.
#
# The pinned mlx_vlm is v0.7.4; a local 0.6.3 checkout cannot satisfy the
# compat modules ``omlx.engine`` imports, so the child stubs only what is
# missing, and only when the real import fails -- a healthy environment keeps
# the real modules.
# ---------------------------------------------------------------------------

_CALL_SITE_PROBE = """
import importlib
import sys
import types
from importlib.machinery import ModuleSpec

kind, model_type, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])


def _stub_if_missing(name, **attrs):
    try:
        importlib.import_module(name)
    except Exception:
        module = types.ModuleType(name)
        module.__spec__ = ModuleSpec(name, None)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_if_missing(
    "omlx.patches.vlm_batch_kv_capacity",
    apply_batch_kv_capacity_patch=lambda *a, **k: None,
)
_stub_if_missing(
    "mlx_vlm.embedding_loader", load_embedding_model=lambda *a, **k: None
)

from omlx.api import grammar


class _Config:
    def __init__(self, value):
        self.model_type = value


class _Model:
    def __init__(self, value):
        self.config = _Config(value)


class _Tokenizer:
    pass


calls = []


def _record(tokenizer, model, **kwargs):
    calls.append(kwargs)
    return object()


grammar.create_grammar_compiler = _record

if kind == "vlm":
    engine_module = importlib.import_module("omlx.engine.vlm")
    engine_class = engine_module.VLMBatchedEngine
else:
    engine_module = importlib.import_module("omlx.engine.batched")
    engine_class = engine_module.BatchedEngine

engine = engine_class.__new__(engine_class)
engine._grammar_compiler = None
engine._grammar_compiler_init_attempted = False
engine._tokenizer = _Tokenizer()
engine._model_name = "test-model"
if kind == "vlm":
    engine._vlm_model = _Model(model_type)
else:
    engine._model = _Model(model_type)

# ``model_type`` is a property on both engines, derived from the model config.
assert engine.model_type == model_type, engine.model_type
assert engine.grammar_compiler is not None, (
    "grammar_compiler returned None; the engine swallowed the failure"
)
assert calls == [{"cache_limit_bytes": expected}], calls
print("ok")
"""


def _probe_call_site(kind, model_type, expected_bytes):
    """Run one engine's ``grammar_compiler`` in a separate interpreter."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _CALL_SITE_PROBE,
            kind,
            model_type,
            str(expected_bytes),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"probe {kind}/{model_type} failed\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    "kind,engine_model_type,env,expected",
    [
        ("batched", "qwen3", _UNSET, _DEFAULT),
        ("vlm", "gemma4", _UNSET, _DEFAULT),
        ("vlm", "gemma4", "512MB", 512 * 1024**2),
        ("batched", "k2_horizon", "2GB", K2_LIMIT),
    ],
)
def test_engine_call_sites_pass_the_resolved_budget(
    monkeypatch, kind, engine_model_type, env, expected
):
    _probe_call_site(kind, engine_model_type, _resolve(monkeypatch, env, expected))
