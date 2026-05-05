# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 monkey-patch for mlx-lm v0.31.3.

Brings PR 1192 (https://github.com/ml-explore/mlx-lm/pull/1192) into omlx
without modifying the pinned mlx-lm. The patch:

1. Injects ``PoolingCache`` and ``BatchPoolingCache`` into
   ``mlx_lm.models.cache``.
2. Registers ``mlx_lm.models.hyper_connection`` and
   ``mlx_lm.models.deepseek_v4`` modules in ``sys.modules`` so the
   built-in ``importlib.import_module(f"mlx_lm.models.{model_type}")``
   path used by ``_get_classes`` finds them.
3. Replaces ``mlx_lm.utils.load_model`` with a copy that handles
   ``F8_E8M0`` dtype fallback and the DeepSeek V4 ``fp8`` quant_method.
4. Replaces ``mlx_lm.generate._make_cache`` with a copy aware of
   ``PoolingCache`` → ``BatchPoolingCache`` conversion.
5. Wraps ``mlx_lm.tokenizer_utils.AutoTokenizer`` with a fallback that
   retries with an empty ``PreTrainedConfig()`` when transformers does
   not yet recognize the ``deepseek_v4`` model_type (PR 45643 was
   merged 2026-05-02 but is missing from transformers <=5.7.0). This
   adopts PR 1189's tokenizer-fallback strategy.
6. Registers omlx-side cache handlers for the two new cache classes so
   prefix-cache / SSD-cache state extraction does not silently fall
   through to ``DefaultCacheHandler``.

The whole patch is gated on ``model_type == "deepseek_v4"`` in
``config.json``; other models pay zero cost.

Once mlx-lm merges PR 1192 upstream this package can be removed in a
single delete (along with the conditional dispatch in
``omlx/utils/model_loading.py`` and ``omlx/engine/batched.py``).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PR_HEAD_SHA = "5c10538136b9038b9626c134612b08afc18d697a"
PR_URL = "https://github.com/ml-explore/mlx-lm/pull/1192"

_APPLIED = False


def _inject_cache_extras() -> None:
    """Add PoolingCache + BatchPoolingCache as attributes of mlx_lm.models.cache.

    Same setattr pattern used by turboquant_attention.py for
    ``scaled_dot_product_attention``. Idempotent.
    """
    import mlx_lm.models.cache as _cache_mod

    if hasattr(_cache_mod, "PoolingCache") and hasattr(_cache_mod, "BatchPoolingCache"):
        return

    from . import cache_extras as _extras

    _cache_mod.PoolingCache = _extras.PoolingCache
    _cache_mod.BatchPoolingCache = _extras.BatchPoolingCache
    # Also expose at __dict__ level so callers that reload the module
    # after the patch (e.g. ``from mlx_lm.models.cache import PoolingCache``)
    # see them too.
    _cache_mod.__dict__["PoolingCache"] = _extras.PoolingCache
    _cache_mod.__dict__["BatchPoolingCache"] = _extras.BatchPoolingCache

    # Reattach the classes' __module__ so any class-name introspection
    # (e.g. type(c).__module__) matches what mlx-lm code expects.
    _extras.PoolingCache.__module__ = "mlx_lm.models.cache"
    _extras.BatchPoolingCache.__module__ = "mlx_lm.models.cache"

    logger.info("PoolingCache / BatchPoolingCache injected into mlx_lm.models.cache")


def _register_module(qualname: str, file_name: str) -> None:
    """Load a local file as if it were ``qualname`` (e.g. mlx_lm.models.deepseek_v4).

    Sets ``__package__`` to ``mlx_lm.models`` so relative imports inside
    the loaded file (``from .cache import PoolingCache``,
    ``from .hyper_connection import HyperConnection``) resolve through
    the real mlx_lm package — *not* through omlx.

    Idempotent.
    """
    if qualname in sys.modules:
        return

    here = Path(__file__).parent
    file_path = here / file_name
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[qualname] = module
    spec.loader.exec_module(module)
    logger.info("Registered %s from %s", qualname, file_path.name)


def _register_cache_handlers() -> None:
    """Register PoolingCache / BatchPoolingCache handlers in omlx CacheTypeRegistry."""
    from omlx.cache.type_registry import CacheTypeRegistry

    from .cache_handlers import BatchPoolingCacheHandler, PoolingCacheHandler

    CacheTypeRegistry.register(PoolingCacheHandler())
    CacheTypeRegistry.register(BatchPoolingCacheHandler())
    logger.info("PoolingCacheHandler + BatchPoolingCacheHandler registered")


def apply_deepseek_v4_patch() -> bool:
    """Apply the DeepSeek V4 patch to mlx-lm. Idempotent.

    Must run *before* ``mlx_lm.load()`` encounters a deepseek_v4 model.

    Returns ``True`` if the patch was freshly applied, ``False`` if already
    applied or if mlx-lm is not importable.
    """
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        logger.debug("mlx_lm not importable — deepseek_v4 patch skipped")
        return False

    # Order matters:
    # 1. Inject PoolingCache / BatchPoolingCache so the deepseek_v4 module
    #    can ``from .cache import PoolingCache`` at exec time.
    _inject_cache_extras()

    # 2. Register hyper_connection BEFORE deepseek_v4 (deepseek_v4 imports
    #    HyperConnection / HyperHead / hc_expand from it).
    _register_module("mlx_lm.models.hyper_connection", "hyper_connection.py")

    # 3. Register deepseek_v4 itself.
    _register_module("mlx_lm.models.deepseek_v4", "deepseek_v4_model.py")

    # 4. Patch utils.load_model (F8_E8M0 fallback + fp8 quant branch).
    from .utils_patch import apply_utils_patch

    apply_utils_patch()

    # 5. Patch generate._make_cache (PoolingCache → BatchPoolingCache).
    from .generate_patch import apply_generate_patch

    apply_generate_patch()

    # 6. Wrap tokenizer_utils.AutoTokenizer (deepseek_v4 fallback for
    #    transformers releases that pre-date PR 45643 / 2026-05-02).
    from .tokenizer_patch import apply_tokenizer_patch

    apply_tokenizer_patch()

    # 7. Register omlx-side cache handlers.
    _register_cache_handlers()

    _APPLIED = True
    logger.info("DeepSeek V4 patch applied (PR 1192 head %s)", PR_HEAD_SHA[:8])
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_deepseek_v4_patch", "is_applied", "PR_HEAD_SHA", "PR_URL"]
