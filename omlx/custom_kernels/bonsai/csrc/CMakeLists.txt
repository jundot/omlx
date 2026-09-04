cmake_minimum_required(VERSION 3.27)

project(omlx_bonsai_kernels LANGUAGES CXX)

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
list(GET MLX_INCLUDE_DIRS 0 OMLX_BONSAI_MLX_INCLUDE_DIR)

# Vendored Bonsai-patched headers.  quantized.h in this directory shadows the
# stock mlx copy so the 1-bit qmv_fast and qmv_wide templates are available.
set(OMLX_BONSAI_CSRC_DIR ${CMAKE_CURRENT_LIST_DIR})
set(OMLX_CUSTOM_KERNEL_COMMON_DIR
    ${CMAKE_CURRENT_LIST_DIR}/../../common/csrc)

# ── C++ dispatch library ────────────────────────────────────────────────────
add_library(omlx_bonsai_kernel_ops)
target_sources(
  omlx_bonsai_kernel_ops
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/bonsai_kernels.cpp)
target_include_directories(
  omlx_bonsai_kernel_ops
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}        # vendored quantized.h shadows mlx's
    ${OMLX_CUSTOM_KERNEL_COMMON_DIR}
    ${OMLX_BONSAI_MLX_INCLUDE_DIR})
target_compile_definitions(
  omlx_bonsai_kernel_ops
  PUBLIC
    OMLX_BONSAI_KERNEL_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
    OMLX_BONSAI_MLX_INCLUDE_DIR="${OMLX_BONSAI_MLX_INCLUDE_DIR}")
target_link_libraries(omlx_bonsai_kernel_ops PUBLIC mlx)

# ── Metal kernels ────────────────────────────────────────────────────────────
if(MLX_BUILD_METAL)
  set(OMLX_BONSAI_METAL_SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/bonsai_quantized.metal
      ${CMAKE_CURRENT_LIST_DIR}/spec_decode.metal)

  # Headers — include vendored csrc dir BEFORE the mlx system include so
  # our quantized.h (with 1-bit + qmv_wide) shadows the stock mlx copy.
  file(GLOB_RECURSE OMLX_BONSAI_METAL_HEADERS CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_LIST_DIR}/*.h
       ${OMLX_CUSTOM_KERNEL_COMMON_DIR}/mlx/**/*.h)

  set(OMLX_BONSAI_METAL_INCLUDE_DIRS
      ${CMAKE_CURRENT_LIST_DIR}      # vendored quantized.h first
      ${OMLX_CUSTOM_KERNEL_COMMON_DIR}
      ${MLX_INCLUDE_DIRS})

  set(OMLX_BONSAI_METAL_FLAGS
      -x metal
      -Wall -Wextra
      -fno-fast-math
      -Wno-c++17-extensions
      -Wno-c++20-extensions)

  if(MLX_METAL_DEBUG)
    list(APPEND OMLX_BONSAI_METAL_FLAGS -gline-tables-only -frecord-sources)
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    list(APPEND OMLX_BONSAI_METAL_FLAGS
         "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()

  set(OMLX_BONSAI_METAL_INCLUDE_ARGS)
  foreach(INCLUDE_DIR IN LISTS OMLX_BONSAI_METAL_INCLUDE_DIRS)
    list(APPEND OMLX_BONSAI_METAL_INCLUDE_ARGS -I${INCLUDE_DIR})
  endforeach()

  set(OMLX_BONSAI_AIR_FILES)
  foreach(SRCFILE IN LISTS OMLX_BONSAI_METAL_SOURCES)
    get_filename_component(TARGET_NAME ${SRCFILE} NAME_WE)
    set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.air)
    add_custom_command(
      OUTPUT ${AIR_FILE}
      COMMAND xcrun -sdk macosx metal
              ${OMLX_BONSAI_METAL_FLAGS}
              -c ${SRCFILE}
              ${OMLX_BONSAI_METAL_INCLUDE_ARGS}
              -o ${AIR_FILE}
      DEPENDS ${SRCFILE} ${OMLX_BONSAI_METAL_HEADERS}
      COMMENT "Building ${TARGET_NAME}.air"
      VERBATIM)
    list(APPEND OMLX_BONSAI_AIR_FILES ${AIR_FILE})
  endforeach()

  set(OMLX_BONSAI_METALLIB
      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/omlx_bonsai_kernels.metallib)
  add_custom_command(
    OUTPUT ${OMLX_BONSAI_METALLIB}
    COMMAND xcrun -sdk macosx metal ${OMLX_BONSAI_AIR_FILES}
            -o ${OMLX_BONSAI_METALLIB}
    DEPENDS ${OMLX_BONSAI_AIR_FILES}
    COMMENT "Linking omlx_bonsai_kernels.metallib"
    VERBATIM)
  add_custom_target(omlx_bonsai_metallib DEPENDS ${OMLX_BONSAI_METALLIB})
  add_dependencies(omlx_bonsai_kernel_ops omlx_bonsai_metallib)
endif()

# ── Python extension (nanobind) ──────────────────────────────────────────────
nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE omlx_bonsai_kernel_ops)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
