cmake_minimum_required(VERSION 3.27)
project(omlx_moe_staging LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)
nanobind_add_module(_ext NB_STATIC NB_DOMAIN mlx staging.cpp)
target_link_libraries(_ext PRIVATE mlx)
target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path/../../../mlx/lib)
