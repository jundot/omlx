"""Model format discovery must work in a fresh process without MLX/GPU access."""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NO_GPU_IMPORTS = """
import importlib.abc
class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'mlx_vlm', 'mlx_embeddings'}:
            raise AssertionError('Unexpected GPU runtime import: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
"""


def run_fresh(source):
    # Isolation avoids a test-session import masking eager package side effects.
    script = (
        f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
        + NO_GPU_IMPORTS
        + textwrap.dedent(source)
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module",
    [
        "omlx.models",
        "omlx.models.jangh2",
        "omlx.models.jangh2.metadata",
        "omlx.models.jangh2.checkpoint",
        "omlx.models.jangh2.performance",
    ],
)
def test_format_discovery_never_imports_gpu_libraries(module):
    run_fresh(f"""
        import importlib
        importlib.import_module({module!r})
        from omlx.models.jangh2 import is_jangh2_config
        from omlx.models.jangh2.metadata import canonical
        from omlx.models.jangh2.performance import PerformanceOptions
        assert is_jangh2_config({{'jangtq': {{}}}})
        assert not is_jangh2_config({{'model_type': 'llama'}})
        assert canonical('model.layers.0.weight') == 'language_model.model.layers.0.weight'
        assert PerformanceOptions().prefill == 'legacy'
        assert 'omlx.models.llm' not in sys.modules
        assert 'omlx.models.embedding' not in sys.modules
        assert not any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules)
    """)


@pytest.mark.parametrize(
    "symbol, target",
    [
        ("MLXLanguageModel", "omlx.models.llm"),
        ("MLXEmbeddingModel", "omlx.models.embedding"),
        ("EmbeddingOutput", "omlx.models.embedding"),
    ],
)
def test_public_wrappers_resolve_only_when_requested(symbol, target):
    run_fresh(f"""
        import importlib.util
        import types
        calls = []
        exported = object()
        class WrapperLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in ('omlx.models.llm', 'omlx.models.embedding'):
                    calls.append(fullname)
                    return importlib.util.spec_from_loader(fullname, self)
            def create_module(self, spec):
                return types.ModuleType(spec.name)
            def exec_module(self, module):
                setattr(module, {symbol!r}, exported)
        sys.meta_path.insert(0, WrapperLoader())
        import omlx.models as models
        assert set(models.__all__) == {{'MLXLanguageModel', 'MLXEmbeddingModel', 'EmbeddingOutput'}}
        assert set(models.__all__) <= set(dir(models))
        assert calls == []
        from omlx.models import {symbol} as resolved
        assert resolved is exported
        assert getattr(models, {symbol!r}) is exported
        assert calls == [{target!r}]
    """)


def test_unknown_attribute_raises_without_importing_wrappers():
    run_fresh("""
        import omlx.models as models
        try:
            models.no_such_model
        except AttributeError as error:
            assert "'omlx.models'" in str(error)
            assert 'no_such_model' in str(error)
        else:
            raise AssertionError('Unknown public attribute was accepted')
        assert 'omlx.models.llm' not in sys.modules
        assert 'omlx.models.embedding' not in sys.modules
    """)
