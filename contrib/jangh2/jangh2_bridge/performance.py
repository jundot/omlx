"""Compatibility alias for the shared JANGH2 performance implementation."""
import sys
from ._shared import shared_module

sys.modules[__name__] = shared_module("performance")
