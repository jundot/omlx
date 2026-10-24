"""Compatibility alias for the shared JANGH2 reference_ops implementation."""
import sys
from ._shared import shared_module

sys.modules[__name__] = shared_module("reference_ops")
