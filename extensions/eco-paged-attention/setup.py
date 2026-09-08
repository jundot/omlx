from mlx import extension
from setuptools import setup

setup(
    name="eco-paged-attention",
    version="0.2.0.dev2",
    packages=["eco_paged_attention"],
    install_requires=["mlx==0.32.2"],
    ext_modules=[extension.CMakeExtension("eco_paged_attention._ext")],
    cmdclass={"build_ext": extension.CMakeBuild},
    package_data={"eco_paged_attention": ["*.so", "*.dylib", "*.metallib"]},
    zip_safe=False,
)
