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

    root = "omlx/custom_kernels"
    shared = f"{root}/include"
    attention = f"{root}/attention/csrc"
    lowbit = f"{root}/lowbit/csrc"
    quant = f"{root}/quant_matmul/csrc"
    sparse = f"{root}/sparse_index/csrc"

    quant_libraries = [
        extension.MetalLibrary(
            "omlx_quant_matmul_kernels",
            sources=[
                f"{quant}/affine_qmm.metal",
                f"{quant}/gather_qmm.metal",
                f"{quant}/head_flat_qmm.metal",
                f"{quant}/moe_weighted_sum.metal",
                f"{quant}/mxfp8_qmv.metal",
                f"{quant}/oq_a8.metal",
                f"{quant}/ane/ane_hybrid.metal",
            ],
        ),
    ]
    sdk_version = subprocess.check_output(
        ["xcrun", "-sdk", "macosx", "--show-sdk-version"], text=True
    ).strip()
    if tuple(int(part) for part in sdk_version.split(".")) >= (26, 2):
        quant_libraries.append(
            extension.MetalLibrary(
                "omlx_quant_matmul_kernels_nax",
                sources=[
                    f"{quant}/affine_qmm_nax.metal",
                    f"{quant}/oq_a8_nax.metal",
                    f"{quant}/oq_q8_a8_nax.metal",
                ],
                deployment_target="26.2",
            )
        )

    return {
        "ext_modules": [
            extension.MetalExtension(
                "omlx.custom_kernels.attention._ext",
                sources=[
                    f"{attention}/bindings.cpp",
                    f"{attention}/block_mask_attention.cpp",
                    f"{attention}/compressed_sparse_attention.cpp",
                    f"{attention}/fa256_attention.cpp",
                    f"{attention}/packed_sparse_attention.cpp",
                    f"{attention}/sdpa_decode.cpp",
                    f"{attention}/sparse_gqa_attention.cpp",
                    f"{attention}/sparse_mla.cpp",
                    f"{attention}/verify_gemm.cpp",
                ],
                include_dirs=[attention, shared],
                metal_libraries=[
                    extension.MetalLibrary(
                        "omlx_attention_kernels",
                        sources=[
                            f"{attention}/block_mask_attention.metal",
                            f"{attention}/compressed_sparse_attention.metal",
                            f"{attention}/fa256_attention.metal",
                            f"{attention}/packed_sparse_attention.metal",
                            f"{attention}/sdpa_decode.metal",
                            f"{attention}/sparse_gqa_attention.metal",
                            f"{attention}/sparse_mla.metal",
                            f"{attention}/verify_gemm.metal",
                        ],
                    ),
                ],
            ),
            extension.MetalExtension(
                "omlx.custom_kernels.lowbit._ext",
                sources=[f"{lowbit}/bindings.cpp", f"{lowbit}/lowbit_qmv.cpp"],
                include_dirs=[lowbit, shared],
                metal_libraries=[
                    extension.MetalLibrary(
                        "omlx_lowbit_kernels",
                        sources=[
                            f"{lowbit}/lowbit_qmv.metal",
                        ],
                    ),
                ],
            ),
            extension.MetalExtension(
                "omlx.custom_kernels.quant_matmul._ext",
                sources=[
                    f"{quant}/bindings.cpp",
                    f"{quant}/affine_qmm.cpp",
                    f"{quant}/command_buffer_caps.cpp",
                    f"{quant}/command_queue.cpp",
                    f"{quant}/gather_qmm.cpp",
                    f"{quant}/grouped_expert.cpp",
                    f"{quant}/head_flat_qmm.cpp",
                    f"{quant}/moe_staging.cpp",
                    f"{quant}/moe_weighted_sum.cpp",
                    f"{quant}/mxfp8_qmv.cpp",
                    f"{quant}/oq_a8.cpp",
                    f"{quant}/ane/ane_hybrid.mm",
                ],
                include_dirs=[quant, shared],
                metal_libraries=quant_libraries,
                extra_link_args=[
                    "SHELL:-framework Accelerate",
                    "SHELL:-framework Foundation",
                    "SHELL:-framework IOSurface",
                    "SHELL:-framework Metal",
                ],
            ),
            extension.MetalExtension(
                "omlx.custom_kernels.sparse_index._ext",
                sources=[
                    f"{sparse}/bindings.cpp",
                    f"{sparse}/dsa_indexer.cpp",
                    f"{sparse}/msa_block_topk.cpp",
                ],
                include_dirs=[sparse, shared],
                metal_libraries=[
                    extension.MetalLibrary(
                        "omlx_sparse_index_kernels",
                        sources=[
                            f"{sparse}/dsa_indexer.metal",
                            f"{sparse}/msa_block_topk.metal",
                        ],
                    ),
                ],
            ),
        ],
        "cmdclass": {
            "build_ext": extension.BuildExtension.with_options(generate_stubs=False),
        },
    }


if __name__ == "__main__":
    setup(**_custom_kernel_build_kwargs())
