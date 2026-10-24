"""Load one shared runtime implementation in installed or legacy checkouts.

The native oMLX package never imports this helper. Standalone diagnostics may
run with the old PR virtualenv, whose oMLX package does not include JANGH2 yet;
load only the shared module file in that case, preserving the old runtime.
"""
from importlib import import_module, util
from pathlib import Path
import sys


def _native_source_root():
    """Inspect the top-level package without importing legacy GPU wrappers."""
    try:
        spec = util.find_spec('omlx')
    except (ModuleNotFoundError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            root = Path(location) / 'models' / 'jangh2'
            if (root / '__init__.py').is_file():
                return root
    return None


def shared_source_root():
    """Locate the packaged runtime or its source tree for fingerprinting."""
    native = _native_source_root()
    if native is not None:
        return native
    return Path(__file__).resolve().parents[3] / 'omlx' / 'models' / 'jangh2'


def shared_module(name):
    target = f'omlx.models.jangh2.{name}'
    if _native_source_root() is not None:
        return import_module(target)
    qualified_name = f'{__package__}.{name}'
    source = shared_source_root() / f'{name}.py'
    spec = util.spec_from_file_location(qualified_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot find shared JANGH2 runtime: {source}')
    module = util.module_from_spec(spec)
    previous = sys.modules.get(qualified_name)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(qualified_name, None)
        else:
            sys.modules[qualified_name] = previous
        raise
    return module
