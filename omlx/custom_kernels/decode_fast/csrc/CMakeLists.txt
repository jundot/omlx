# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.27)

project(omlx_decode_fast_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

add_library(omlx_decode_fast_kernel_ops)
target_sources(
  omlx_decode_fast_kernel_ops
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/decode_fast.cpp
  ${CMAKE_CURRENT_LIST_DIR}/sdpa_decode.cpp
  ${CMAKE_CURRENT_LIST_DIR}/rope_append.cpp
  ${CMAKE_CURRENT_LIST_DIR}/sparse_attn_decode.cpp)
target_include_directories(
  omlx_decode_fast_kernel_ops
  PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(omlx_decode_fast_kernel_ops PUBLIC mlx)

if(MLX_BUILD_METAL)
  set(OMLX_DECODE_FAST_METAL_SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/rms_residual.metal
      ${CMAKE_CURRENT_LIST_DIR}/sdpa_decode.metal
      ${CMAKE_CURRENT_LIST_DIR}/rope_append.metal
      ${CMAKE_CURRENT_LIST_DIR}/sparse_attn_decode.metal)
  file(GLOB_RECURSE OMLX_DECODE_FAST_METAL_HEADERS CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_LIST_DIR}/*.h)
  set(OMLX_DECODE_FAST_METAL_FLAGS
      -x
      metal
      -Wall
      -Wextra
      -fno-fast-math
      -Wno-c++17-extensions
      -Wno-c++20-extensions)
  if(MLX_METAL_DEBUG)
    set(OMLX_DECODE_FAST_METAL_FLAGS
        ${OMLX_DECODE_FAST_METAL_FLAGS} -gline-tables-only -frecord-sources)
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    set(OMLX_DECODE_FAST_METAL_FLAGS
        ${OMLX_DECODE_FAST_METAL_FLAGS}
        "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  set(OMLX_DECODE_FAST_METAL_INCLUDE_ARGS)
  list(APPEND OMLX_DECODE_FAST_METAL_INCLUDE_ARGS
       -I${CMAKE_CURRENT_LIST_DIR})
  foreach(INCLUDE_DIR IN LISTS MLX_INCLUDE_DIRS)
    list(APPEND OMLX_DECODE_FAST_METAL_INCLUDE_ARGS -I${INCLUDE_DIR})
  endforeach()
  set(OMLX_DECODE_FAST_METAL_AIR)
  foreach(SRCFILE IN LISTS OMLX_DECODE_FAST_METAL_SOURCES)
    get_filename_component(TARGET_NAME ${SRCFILE} NAME_WE)
    set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.air)
    add_custom_command(
      OUTPUT ${AIR_FILE}
      COMMAND xcrun -sdk macosx metal ${OMLX_DECODE_FAST_METAL_FLAGS} -c
              ${SRCFILE} ${OMLX_DECODE_FAST_METAL_INCLUDE_ARGS} -o
              ${AIR_FILE}
      DEPENDS ${SRCFILE} ${OMLX_DECODE_FAST_METAL_HEADERS}
      COMMENT "Building ${TARGET_NAME}.air"
      VERBATIM)
    list(APPEND OMLX_DECODE_FAST_METAL_AIR ${AIR_FILE})
  endforeach()

  set(OMLX_DECODE_FAST_METALLIB
      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/omlx_decode_fast_kernels.metallib)
  add_custom_command(
    OUTPUT ${OMLX_DECODE_FAST_METALLIB}
    COMMAND xcrun -sdk macosx metal ${OMLX_DECODE_FAST_METAL_AIR} -o
            ${OMLX_DECODE_FAST_METALLIB}
    DEPENDS ${OMLX_DECODE_FAST_METAL_AIR}
    COMMENT "Building omlx_decode_fast_kernels.metallib"
    VERBATIM)
  add_custom_target(omlx_decode_fast_kernels_metallib
                    DEPENDS ${OMLX_DECODE_FAST_METALLIB})

  add_dependencies(omlx_decode_fast_kernel_ops
                   omlx_decode_fast_kernels_metallib)
endif()

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE omlx_decode_fast_kernel_ops)

if(BUILD_SHARED_LIBS)
  # @loader_path resolves the sibling kernel_ops dylib; the second entry
  # resolves the mlx wheel's libmlx.dylib (issue #2233).
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path
                      -Wl,-rpath,@loader_path/../../../mlx/lib)
endif()
