import os
import subprocess
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

    extension._HOST_SOURCE_SUFFIXES.add(".mm")

    common = "omlx/custom_kernels/common/csrc"
    bonsai = "omlx/custom_kernels/bonsai/csrc"
    decode = "omlx/custom_kernels/decode_fast/csrc"
    glm = "omlx/custom_kernels/glm_moe_dsa/csrc"
    minimax = "omlx/custom_kernels/minimax_m3/csrc"
    qwen = "omlx/custom_kernels/qwen35_prefill/csrc"

    qwen_libraries = [
        extension.MetalLibrary(
            "omlx_qwen35_prefill_kernels",
            sources=[
                f"{qwen}/qwen35_attention.metal",
                f"{qwen}/qwen35_qmm.metal",
                f"{qwen}/qwen35_oq_a8.metal",
                f"{qwen}/qwen35_ane.metal",
            ],
        ),
    ]
    sdk_version = subprocess.check_output(
        ["xcrun", "-sdk", "macosx", "--show-sdk-version"], text=True
    ).strip()
    if tuple(int(part) for part in sdk_version.split(".")) >= (26, 2):
        qwen_libraries.append(
            extension.MetalLibrary(
                "omlx_qwen35_prefill_kernels_nax",
                sources=[
                    f"{qwen}/qwen35_qmm_nax.metal",
                    f"{qwen}/qwen35_oq_a8_nax.metal",
                    f"{qwen}/qwen35_oq_q8_a8_nax.metal",
                ],
                deployment_target="26.2",
            )
        )

    return {
        "ext_modules": [
            extension.MetalExtension(
                "omlx.custom_kernels.bonsai._ext",
                sources=[f"{bonsai}/bindings.cpp", f"{bonsai}/bonsai_kernels.cpp"],
                include_dirs=[bonsai, common],
                metal_libraries=[
                    extension.MetalLibrary(
                        "omlx_bonsai_kernels",
                        sources=[
                            f"{bonsai}/bonsai_quantized.metal",
                            f"{bonsai}/spec_decode.metal",
                        ],
                    ),
                ],
            ),
            extension.MetalExtension(
                "omlx.custom_kernels.decode_fast._ext",
                sources=[
                    f"{decode}/bindings.cpp",
                    f"{decode}/sdpa_decode.cpp",
                ],
                include_dirs=[decode],
                metal_libraries=[
                    extension.MetalLibrary(
                        "omlx_decode_fast_kernels",
                        sources=[f"{decode}/sdpa_decode.metal"],
                    ),
                ],
            ),
            extension.MetalExtension(
                "omlx.custom_kernels.glm_moe_dsa._ext",
                sources=[
                    f"{glm}/bindings.cpp",
                    f"{glm}/dsa_indexer.cpp",
                    f"{glm}/deepseek_v4_sparse_attention.cpp",
                    f"{glm}/deepseek_v41_packed_attention.cpp",
                    f"{glm}/deepseek_v41_grouped_expert.cpp",
                    f"{glm}/dspark_gemm.cpp",
                    f"{glm}/dspark_qmv.cpp",
                    f"{glm}/exact_block_attention.cpp",
                    f"{glm}/fused_moe.cpp",
                    f"{glm}/qwen4_qsa_sparse_gqa.cpp",
                    f"{glm}/sparse_mla.cpp",
                ],
                include_dirs=[glm, common],
                metal_libraries=[
                    extension.MetalLibrary(
                        "omlx_glm_kernels",
                        sources=[
                            f"{glm}/deepseek_moe.metal",
                            f"{glm}/deepseek_v4_sparse_attention.metal",
                            f"{glm}/deepseek_v41_packed_attention.metal",
                            f"{glm}/dspark_gemm.metal",
                            f"{glm}/dspark_qmv.metal",
                            f"{glm}/dsa_indexer.metal",
                            f"{glm}/exact_block_attention.metal",
                            f"{glm}/fused_moe.metal",
                            f"{glm}/qwen4_qsa_sparse_gqa.metal",
                            f"{glm}/sparse_mla.metal",
                        ],
                    ),
                ],
            ),
            extension.MetalExtension(
                "omlx.custom_kernels.minimax_m3._ext",
                sources=[f"{minimax}/bindings.cpp", f"{minimax}/minimax_msa.cpp"],
                include_dirs=[minimax],
                metal_libraries=[
                    extension.MetalLibrary(
                        "omlx_minimax_m3_kernels",
                        sources=[f"{minimax}/minimax_msa.metal"],
                    ),
                ],
            ),
            extension.MetalExtension(
                "omlx.custom_kernels.qwen35_prefill._ext",
                sources=[
                    f"{qwen}/bindings.cpp",
                    f"{qwen}/qwen35_prefill.cpp",
                    f"{qwen}/qwen35_oq_a8.cpp",
                    f"{qwen}/qwen35_ane.mm",
                ],
                include_dirs=[qwen, common],
                metal_libraries=qwen_libraries,
                extra_link_args=[
                    "SHELL:-framework Accelerate",
                    "SHELL:-framework Foundation",
                    "SHELL:-framework IOSurface",
                    "SHELL:-framework Metal",
                ],
            ),
        ],
        "cmdclass": {
            "build_ext": extension.BuildExtension.with_options(generate_stubs=False),
        },
    }


if __name__ == "__main__":
    setup(**_custom_kernel_build_kwargs())
