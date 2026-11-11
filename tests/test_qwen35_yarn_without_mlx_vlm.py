# SPDX-License-Identifier: Apache-2.0
"""The YaRN patch on a CUDA cluster worker, whose venv ships without mlx-vlm (#4739)."""

import subprocess
import sys
import textwrap


def test_yarn_patch_imports_and_applies_without_mlx_vlm():
    # A fresh interpreter: the patch is process-global and applies once.
    code = textwrap.dedent(
        """
        import sys

        # `import mlx_vlm...` now raises ImportError, as on the worker.
        sys.modules["mlx_vlm"] = None

        import mlx_lm.utils as lm_utils

        original = lm_utils.load_config

        # The inference worker imports this on every load.
        import omlx.utils.model_loading  # noqa: F401
        from omlx.patches import qwen35_yarn

        qwen35_yarn.apply_qwen35_yarn_patch()
        # mlx-lm models still get the YaRN config hook.
        assert lm_utils.load_config is not original
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
