cmake_minimum_required(VERSION 3.27)

project(omlx_minimax_m3_kernels LANGUAGES CXX)

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
list(GET MLX_INCLUDE_DIRS 0 OMLX_MINIMAX_M3_MLX_INCLUDE_DIR)

add_library(omlx_minimax_m3_kernel_ops)
target_sources(
  omlx_minimax_m3_kernel_ops
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/minimax_msa.cpp)
target_include_directories(
  omlx_minimax_m3_kernel_ops
  PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(
  omlx_minimax_m3_kernel_ops
  PUBLIC OMLX_MINIMAX_M3_KERNEL_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
         OMLX_MINIMAX_M3_MLX_INCLUDE_DIR="${OMLX_MINIMAX_M3_MLX_INCLUDE_DIR}")
target_link_libraries(omlx_minimax_m3_kernel_ops PUBLIC mlx)

if(MLX_BUILD_METAL)
  set(OMLX_MINIMAX_M3_METAL_SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/minimax_msa.metal)
  file(GLOB_RECURSE OMLX_MINIMAX_M3_METAL_HEADERS CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_LIST_DIR}/*.h)
  set(OMLX_MINIMAX_M3_METAL_FLAGS
      -x
      metal
      -Wall
      -Wextra
      -fno-fast-math
      -Wno-c++17-extensions
      -Wno-c++20-extensions)
  if(MLX_METAL_DEBUG)
    set(OMLX_MINIMAX_M3_METAL_FLAGS ${OMLX_MINIMAX_M3_METAL_FLAGS}
                                    -gline-tables-only -frecord-sources)
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    set(OMLX_MINIMAX_M3_METAL_FLAGS
        ${OMLX_MINIMAX_M3_METAL_FLAGS}
        "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  set(OMLX_MINIMAX_M3_METAL_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}
                                         ${MLX_INCLUDE_DIRS})
  set(OMLX_MINIMAX_M3_METAL_INCLUDE_ARGS)
  foreach(INCLUDE_DIR IN LISTS OMLX_MINIMAX_M3_METAL_INCLUDE_DIRS)
    list(APPEND OMLX_MINIMAX_M3_METAL_INCLUDE_ARGS -I${INCLUDE_DIR})
  endforeach()
  set(OMLX_MINIMAX_M3_METAL_AIR)
  foreach(SRCFILE IN LISTS OMLX_MINIMAX_M3_METAL_SOURCES)
    get_filename_component(TARGET_NAME ${SRCFILE} NAME_WE)
    set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.air)
    add_custom_command(
      OUTPUT ${AIR_FILE}
      COMMAND xcrun -sdk macosx metal ${OMLX_MINIMAX_M3_METAL_FLAGS} -c
              ${SRCFILE} ${OMLX_MINIMAX_M3_METAL_INCLUDE_ARGS} -o ${AIR_FILE}
      DEPENDS ${SRCFILE} ${OMLX_MINIMAX_M3_METAL_HEADERS}
      COMMENT "Building ${TARGET_NAME}.air"
      VERBATIM)
    list(APPEND OMLX_MINIMAX_M3_METAL_AIR ${AIR_FILE})
  endforeach()

  set(OMLX_MINIMAX_M3_METALLIB
      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/omlx_minimax_m3_kernels.metallib)
  add_custom_command(
    OUTPUT ${OMLX_MINIMAX_M3_METALLIB}
    COMMAND xcrun -sdk macosx metal ${OMLX_MINIMAX_M3_METAL_AIR} -o
            ${OMLX_MINIMAX_M3_METALLIB}
    DEPENDS ${OMLX_MINIMAX_M3_METAL_AIR}
    COMMENT "Building omlx_minimax_m3_kernels.metallib"
    VERBATIM)
  add_custom_target(omlx_minimax_m3_kernels_metallib
                    DEPENDS ${OMLX_MINIMAX_M3_METALLIB})

  add_dependencies(omlx_minimax_m3_kernel_ops
                   omlx_minimax_m3_kernels_metallib)
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
target_link_libraries(_ext PRIVATE omlx_minimax_m3_kernel_ops)

if(BUILD_SHARED_LIBS)
  # @loader_path resolves the sibling kernel_ops dylib. The second entry
  # resolves the mlx wheel's libmlx.dylib from the installed site-packages
  # layout (omlx/custom_kernels/<pkg> -> site-packages -> mlx/lib); without
  # it the only libmlx rpath is the link-time absolute path into the build
  # env, which is dead after install (issue #2233).
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path
                      -Wl,-rpath,@loader_path/../../../mlx/lib)
endif()
