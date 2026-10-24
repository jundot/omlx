"""Compatibility alias for the shared JANGH2 metadata implementation."""
import sys
from ._shared import shared_module

sys.modules[__name__] = shared_module("metadata")
