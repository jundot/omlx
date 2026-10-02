# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.27)
project(omlx_decode_fast_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

add_library(omlx_decode_fast_kernel_ops)
target_sources(omlx_decode_fast_kernel_ops PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/sdpa_decode.cpp)
target_include_directories(omlx_decode_fast_kernel_ops PUBLIC
  ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(omlx_decode_fast_kernel_ops PUBLIC mlx)

if(MLX_BUILD_METAL)
  set(SDPA_METAL ${CMAKE_CURRENT_LIST_DIR}/sdpa_decode.metal)
  set(SDPA_AIR ${CMAKE_CURRENT_BINARY_DIR}/sdpa_decode.air)
  set(SDPA_METALLIB
      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/omlx_decode_fast_kernels.metallib)
  set(SDPA_INCLUDE_ARGS -I${CMAKE_CURRENT_LIST_DIR})
  foreach(INCLUDE_DIR IN LISTS MLX_INCLUDE_DIRS)
    list(APPEND SDPA_INCLUDE_ARGS -I${INCLUDE_DIR})
  endforeach()
  set(SDPA_METAL_FLAGS -x metal -Wall -Wextra -fno-fast-math
      -Wno-c++17-extensions -Wno-c++20-extensions)
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    list(APPEND SDPA_METAL_FLAGS
         "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  add_custom_command(
    OUTPUT ${SDPA_AIR}
    COMMAND xcrun -sdk macosx metal ${SDPA_METAL_FLAGS} -c ${SDPA_METAL}
            ${SDPA_INCLUDE_ARGS} -o ${SDPA_AIR}
    DEPENDS ${SDPA_METAL} ${CMAKE_CURRENT_LIST_DIR}/sdpa_decode_kernels.h
    VERBATIM)
  add_custom_command(
    OUTPUT ${SDPA_METALLIB}
    COMMAND xcrun -sdk macosx metal ${SDPA_AIR} -o ${SDPA_METALLIB}
    DEPENDS ${SDPA_AIR} VERBATIM)
  add_custom_target(omlx_decode_fast_kernels_metallib DEPENDS ${SDPA_METALLIB})
  add_dependencies(omlx_decode_fast_kernel_ops
                   omlx_decode_fast_kernels_metallib)
endif()

nanobind_add_module(_ext NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE omlx_decode_fast_kernel_ops)
if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path
                      -Wl,-rpath,@loader_path/../../../mlx/lib)
endif()
