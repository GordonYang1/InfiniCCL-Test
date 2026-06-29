cmake_minimum_required(VERSION 3.18)
project(InfiniCCLUserTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================================
# FIND INFINICCL (via its CMake config package)
# =========================================================
# This replaces the old manual wiring (explicit include dir + path to
# `libinfiniccl.so` + hand-set RPATH). `find_package` pulls in the imported
# target `InfiniCCL::infiniccl`, which already carries the public include
# directory, the library location, and the runtime search path.
#
# To locate a non-standard install, point CMake at the InfiniCCL prefix with
# any of the usual mechanisms, e.g.:
#     -DCMAKE_PREFIX_PATH=/path/to/install
#     -DInfiniCCL_ROOT=/path/to/install
# For backward compatibility, the existing `INFINICCL_INSTALL` environment
# variable is still honored as a search hint.
if(DEFINED ENV{INFINICCL_INSTALL})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{INFINICCL_INSTALL}")
endif()

find_package(InfiniCCL REQUIRED)

# =========================================================
# HARDWARE RUNTIME & MPI SETUP
# =========================================================
find_package(MPI REQUIRED)

if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(GPU_LIBS CUDA::cudart)
elseif(USE_MACA)
    add_definitions(-DUSE_MACA)

    set(MACA_PATH $ENV{MACA_PATH})
    if(NOT MACA_PATH)
        set(MACA_PATH "/opt/maca")
        message(WARNING "`MACA_PATH` environment variable not set. Defaulting to `/opt/maca/`. Please verify this path is correct.")
    endif()

    include_directories("${MACA_PATH}/include")
    find_library(MACA_LIB mcruntime HINTS "${MACA_PATH}/lib" REQUIRED)
    set(GPU_LIBS ${MACA_LIB})
endif()

# =========================================================
# DEFINE THE DIRECTORY GROUPS FOR DISCOVERY
# =========================================================
# Add or remove directory names here to organize test suites.
set(TEST_SUITES
    "pure_mpi_tests"
    "ccl_mpi_tests"
)

# =========================================================
# GLOBAL INCLUDE DIRECTORIES
# =========================================================
# Note: the InfiniCCL headers are provided transitively by the
# `InfiniCCL::infiniccl` imported target, so only the local test headers
# need to be listed here.
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# =========================================================
# AUTOMATED BUILD EXECUTION LOOP
# =========================================================
foreach(suite_dir ${TEST_SUITES})
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${suite_dir}")
        message(STATUS "Discovering tests inside directory suite: [${suite_dir}]")

        # Scrape all C++ source files out of the targeted subfolder.
        file(GLOB SUITE_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/${suite_dir}/*.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/${suite_dir}/*.cpp"
        )

        foreach(source_file ${SUITE_SOURCES})
            # Strip path and file type extension to isolate a clean target binary name.
            get_filename_component(test_name "${source_file}" NAME_WE)

            # Instantiate the test binary target.
            add_executable(${test_name} "${source_file}")

            # Bind our compiled dependencies, drivers, and runtime targets.
            # `InfiniCCL::infiniccl` brings its own include dirs and RPATH.
            target_link_libraries(${test_name}
                PRIVATE
                    InfiniCCL::infiniccl
                    ${GPU_LIBS}
                    MPI::MPI_CXX
            )

            set_target_properties(${test_name} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${suite_dir}"
            )

            message(STATUS "  -> Registered test binary: ${suite_dir}/${test_name}")
        endforeach()
    endif()
endforeach()
