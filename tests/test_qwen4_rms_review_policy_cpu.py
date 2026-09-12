import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "omlx/patches/qwen35_gdn_prework.py"

class RMSReviewPolicyTests(unittest.TestCase):
    def test_rms_defaults_off_and_explicit_optin_is_respected(self):
        node = next(n for n in ast.parse(SOURCE.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_qwen4_unified_rms_enabled")
        namespace = {"os": os}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        enabled = namespace[node.name]
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(enabled())
            for value in ("0", "false", "no", "off", " OFF "):
                os.environ["OMLX_QWEN4_UNIFIED_GDN_RMS_VERIFY"] = value
                self.assertFalse(enabled(), value)
            os.environ["OMLX_QWEN4_UNIFIED_GDN_RMS_VERIFY"] = "1"
            self.assertTrue(enabled())
