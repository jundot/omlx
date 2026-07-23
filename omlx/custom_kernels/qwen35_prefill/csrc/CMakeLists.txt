cmake_minimum_required(VERSION 3.27)

project(omlx_qwen35_prefill_kernels LANGUAGES CXX)

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
list(GET MLX_INCLUDE_DIRS 0 OMLX_QWEN35_PREFILL_MLX_INCLUDE_DIR)

set(OMLX_CUSTOM_KERNEL_COMMON_DIR
    ${CMAKE_CURRENT_LIST_DIR}/../../common/csrc)

add_library(omlx_qwen35_prefill_kernel_ops)
target_sources(
  omlx_qwen35_prefill_kernel_ops
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/qwen35_prefill.cpp)
target_include_directories(
  omlx_qwen35_prefill_kernel_ops
  PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${OMLX_CUSTOM_KERNEL_COMMON_DIR})
target_compile_definitions(
  omlx_qwen35_prefill_kernel_ops
  PUBLIC OMLX_QWEN35_PREFILL_KERNEL_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
         OMLX_QWEN35_PREFILL_MLX_INCLUDE_DIR="${OMLX_QWEN35_PREFILL_MLX_INCLUDE_DIR}")
target_link_libraries(omlx_qwen35_prefill_kernel_ops PUBLIC mlx)

if(MLX_BUILD_METAL)
  set(OMLX_QWEN35_PREFILL_METAL_SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/qwen35_attention.metal
      ${CMAKE_CURRENT_LIST_DIR}/qwen35_qmm.metal)
  file(GLOB_RECURSE OMLX_QWEN35_PREFILL_METAL_HEADERS CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_LIST_DIR}/*.h
       ${OMLX_CUSTOM_KERNEL_COMMON_DIR}/kernels/*.h
       ${OMLX_CUSTOM_KERNEL_COMMON_DIR}/mlx/**/*.h)
  set(OMLX_QWEN35_PREFILL_METAL_FLAGS
      -x
      metal
      -Wall
      -Wextra
      -fno-fast-math
      -Wno-c++17-extensions
      -Wno-c++20-extensions)
  if(MLX_METAL_DEBUG)
    set(OMLX_QWEN35_PREFILL_METAL_FLAGS
        ${OMLX_QWEN35_PREFILL_METAL_FLAGS} -gline-tables-only -frecord-sources)
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    set(OMLX_QWEN35_PREFILL_METAL_FLAGS
        ${OMLX_QWEN35_PREFILL_METAL_FLAGS}
        "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  set(OMLX_QWEN35_PREFILL_METAL_INCLUDE_DIRS
      ${CMAKE_CURRENT_LIST_DIR}
      ${OMLX_CUSTOM_KERNEL_COMMON_DIR}
      ${MLX_INCLUDE_DIRS})
  set(OMLX_QWEN35_PREFILL_METAL_INCLUDE_ARGS)
  foreach(INCLUDE_DIR IN LISTS OMLX_QWEN35_PREFILL_METAL_INCLUDE_DIRS)
    list(APPEND OMLX_QWEN35_PREFILL_METAL_INCLUDE_ARGS -I${INCLUDE_DIR})
  endforeach()
  set(OMLX_QWEN35_PREFILL_METAL_AIR)
  foreach(SRCFILE IN LISTS OMLX_QWEN35_PREFILL_METAL_SOURCES)
    get_filename_component(TARGET_NAME ${SRCFILE} NAME_WE)
    set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.air)
    add_custom_command(
      OUTPUT ${AIR_FILE}
      COMMAND xcrun -sdk macosx metal ${OMLX_QWEN35_PREFILL_METAL_FLAGS} -c
              ${SRCFILE} ${OMLX_QWEN35_PREFILL_METAL_INCLUDE_ARGS} -o
              ${AIR_FILE}
      DEPENDS ${SRCFILE} ${OMLX_QWEN35_PREFILL_METAL_HEADERS}
      COMMENT "Building ${TARGET_NAME}.air"
      VERBATIM)
    list(APPEND OMLX_QWEN35_PREFILL_METAL_AIR ${AIR_FILE})
  endforeach()

  set(OMLX_QWEN35_PREFILL_METALLIB
      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/omlx_qwen35_prefill_kernels.metallib)
  add_custom_command(
    OUTPUT ${OMLX_QWEN35_PREFILL_METALLIB}
    COMMAND xcrun -sdk macosx metal ${OMLX_QWEN35_PREFILL_METAL_AIR} -o
            ${OMLX_QWEN35_PREFILL_METALLIB}
    DEPENDS ${OMLX_QWEN35_PREFILL_METAL_AIR}
    COMMENT "Building omlx_qwen35_prefill_kernels.metallib"
    VERBATIM)
  add_custom_target(omlx_qwen35_prefill_kernels_metallib
                    DEPENDS ${OMLX_QWEN35_PREFILL_METALLIB})

  add_dependencies(omlx_qwen35_prefill_kernel_ops
                   omlx_qwen35_prefill_kernels_metallib)

  # NAX (M5 tensor-unit) kernels live in a separate metallib compiled with
  # -mmacosx-version-min=26.2: the MetalPerformancePrimitives tensor-ops API
  # needs macOS SDK 26.2+, and keeping them out of the classic metallib
  # preserves its deployment target. The C++ op only loads this library when
  # the runtime reports NAX support and degrades to the classic kernels when
  # the file is absent.
  execute_process(
    COMMAND xcrun -sdk macosx --show-sdk-version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE OMLX_MACOS_SDK_VERSION
    ERROR_QUIET)
  if(OMLX_MACOS_SDK_VERSION VERSION_GREATER_EQUAL 26.2)
    set(OMLX_QWEN35_PREFILL_NAX_METAL_FLAGS
        -x
        metal
        -Wall
        -Wextra
        -fno-fast-math
        -Wno-c++17-extensions
        -Wno-c++20-extensions
        -mmacosx-version-min=26.2)
    if(MLX_METAL_DEBUG)
      set(OMLX_QWEN35_PREFILL_NAX_METAL_FLAGS
          ${OMLX_QWEN35_PREFILL_NAX_METAL_FLAGS} -gline-tables-only
          -frecord-sources)
    endif()
    set(OMLX_QWEN35_PREFILL_NAX_AIR
        ${CMAKE_CURRENT_BINARY_DIR}/qwen35_qmm_nax.air)
    add_custom_command(
      OUTPUT ${OMLX_QWEN35_PREFILL_NAX_AIR}
      COMMAND
        xcrun -sdk macosx metal ${OMLX_QWEN35_PREFILL_NAX_METAL_FLAGS} -c
        ${CMAKE_CURRENT_LIST_DIR}/qwen35_qmm_nax.metal
        ${OMLX_QWEN35_PREFILL_METAL_INCLUDE_ARGS} -o
        ${OMLX_QWEN35_PREFILL_NAX_AIR}
      DEPENDS ${CMAKE_CURRENT_LIST_DIR}/qwen35_qmm_nax.metal
              ${OMLX_QWEN35_PREFILL_METAL_HEADERS}
      COMMENT "Building qwen35_qmm_nax.air"
      VERBATIM)
    set(OMLX_QWEN35_PREFILL_NAX_METALLIB
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/omlx_qwen35_prefill_kernels_nax.metallib)
    add_custom_command(
      OUTPUT ${OMLX_QWEN35_PREFILL_NAX_METALLIB}
      # The link step also needs the 26.2 floor: MACOSX_DEPLOYMENT_TARGET in
      # the build environment otherwise pins the link to an older AIR version
      # and air-lld silently drops the newer NAX object (empty metallib).
      COMMAND xcrun -sdk macosx metal -mmacosx-version-min=26.2
              ${OMLX_QWEN35_PREFILL_NAX_AIR} -o
              ${OMLX_QWEN35_PREFILL_NAX_METALLIB}
      DEPENDS ${OMLX_QWEN35_PREFILL_NAX_AIR}
      COMMENT "Building omlx_qwen35_prefill_kernels_nax.metallib"
      VERBATIM)
    add_custom_target(omlx_qwen35_prefill_kernels_nax_metallib
                      DEPENDS ${OMLX_QWEN35_PREFILL_NAX_METALLIB})
    add_dependencies(omlx_qwen35_prefill_kernel_ops
                     omlx_qwen35_prefill_kernels_nax_metallib)
  else()
    message(
      STATUS
        "omlx: macOS SDK ${OMLX_MACOS_SDK_VERSION} < 26.2, skipping qwen35 NAX kernels"
    )
  endif()
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
target_link_libraries(_ext PRIVATE omlx_qwen35_prefill_kernel_ops)

if(BUILD_SHARED_LIBS)
  # @loader_path resolves the sibling kernel_ops dylib. The second entry
  # resolves the mlx wheel's libmlx.dylib from the installed site-packages
  # layout (omlx/custom_kernels/<pkg> -> site-packages -> mlx/lib); without
  # it the only libmlx rpath is the link-time absolute path into the build
  # env, which is dead after install (issue #2233).
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path
                      -Wl,-rpath,@loader_path/../../../mlx/lib)
endif()
