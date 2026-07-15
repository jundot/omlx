import os
import sys

from setuptools import setup


CUSTOM_KERNEL_FLAG = "--with-custom-kernel"
TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_CUSTOM_KERNEL_DEPLOYMENT_TARGET = "15.0"


def _with_custom_kernel() -> bool:
    if CUSTOM_KERNEL_FLAG in sys.argv:
        sys.argv.remove(CUSTOM_KERNEL_FLAG)
        return True
    return os.environ.get("OMLX_WITH_CUSTOM_KERNEL", "").strip().lower() in TRUTHY


def _custom_kernel_build_kwargs() -> dict:
    if not _with_custom_kernel():
        return {}

    target = (
        os.environ.get("OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET")
        or os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        or DEFAULT_CUSTOM_KERNEL_DEPLOYMENT_TARGET
    )
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", target)
    cmake_args = os.environ.get("CMAKE_ARGS", "").strip()
    if "CMAKE_OSX_DEPLOYMENT_TARGET" not in cmake_args:
        target_arg = f"-DCMAKE_OSX_DEPLOYMENT_TARGET={target}"
        os.environ["CMAKE_ARGS"] = (
            f"{cmake_args} {target_arg}".strip() if cmake_args else target_arg
        )

    from mlx import extension

    return {
        "ext_modules": [
            extension.CMakeExtension(
                "omlx.custom_kernels.bonsai._ext",
                sourcedir="omlx/custom_kernels/bonsai/csrc",
            ),
            extension.CMakeExtension(
                "omlx.custom_kernels.glm_moe_dsa._ext",
                sourcedir="omlx/custom_kernels/glm_moe_dsa/csrc",
            ),
            extension.CMakeExtension(
                "omlx.custom_kernels.minimax_m3._ext",
                sourcedir="omlx/custom_kernels/minimax_m3/csrc",
            ),
            extension.CMakeExtension(
                "omlx.custom_kernels.qwen35_prefill._ext",
                sourcedir="omlx/custom_kernels/qwen35_prefill/csrc",
            ),
        ],
        "cmdclass": {"build_ext": extension.CMakeBuild},
    }


if __name__ == "__main__":
    setup(**_custom_kernel_build_kwargs())
