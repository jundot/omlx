cmake_minimum_required(VERSION 3.27)
project(qwen4_moe_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# No custom Metal kernels: this extension only wraps mmap'd buffers as mx.arrays;
# all compute stays on stock mlx::core::gather_qmm. NB_DOMAIN mlx is load-bearing
# -- it must match the domain the mlx wheel was built with or the mx.array type
# caster is isolated and abi_probe fails (fast.py then disables the native path).
nanobind_add_module(_ext
  NB_STATIC STABLE_ABI LTO NOMINSIZE
  NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_include_directories(_ext PRIVATE ${MLX_INCLUDE_DIRS})
target_link_libraries(_ext PRIVATE mlx
  "-framework Metal" "-framework Foundation" "-framework IOSurface")

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE
    -Wl,-rpath,@loader_path
    -Wl,-rpath,@loader_path/../../../mlx/lib)  # resolve the mlx wheel's libmlx.dylib
endif()
