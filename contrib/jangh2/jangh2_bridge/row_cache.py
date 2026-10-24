"""Compatibility alias for the shared JANGH2 row_cache implementation."""
import sys
from ._shared import shared_module

sys.modules[__name__] = shared_module("row_cache")
