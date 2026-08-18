import os
import sys
from pathlib import Path

from setuptools import setup

CUSTOM_KERNEL_FLAG = "--with-custom-kernel"
TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_CUSTOM_KERNEL_DEPLOYMENT_TARGET = "15.0"
DOTENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_dotenv(path: Path = DOTENV_PATH) -> None:
    """Seed os.environ from a gitignored .env so build flags persist locally.

    Lets a checkout opt into OMLX_WITH_CUSTOM_KERNEL (and the deployment
    target / CMAKE_ARGS overrides below) without exporting anything in the
    shell or touching a tracked file.  Real environment variables win, so
    CI and one-off `OMLX_WITH_CUSTOM_KERNEL=1 pip install` keep working.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip("\"'"))


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
        cmake_args = os.environ["CMAKE_ARGS"]

    # CMake otherwise chooses the first framework Python on PATH, which can
    # differ from the interpreter running pip (and lack nanobind / MLX).  The
    # extensions must use the active environment's ABI and CMake packages.
    python_args = " ".join(
        (
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        )
    )
    if "Python_EXECUTABLE" not in cmake_args:
        os.environ["CMAKE_ARGS"] = f"{cmake_args} {python_args}".strip()

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
    _load_dotenv()
    setup(**_custom_kernel_build_kwargs())
