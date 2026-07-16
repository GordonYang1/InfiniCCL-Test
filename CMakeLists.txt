cmake_minimum_required(VERSION 3.18)
project(InfiniCCLUserTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

option(INFINICCL_TEST_ENABLE_PURE_MPI "Enable pure MPI-path tests" ON)
option(INFINICCL_TEST_ENABLE_PURE_CCL "Enable reserved pure CCL-path tests" OFF)
option(INFINICCL_TEST_ENABLE_MIX "Enable reserved mixed-path tests" OFF)
option(INFINICCL_TEST_PIN_ALL_RANKS_TO_DEVICE0 "Pin all ranks to device 0" ON)
set(INFINICCL_TEST_CLUSTER_CONFIG "${CMAKE_SOURCE_DIR}/cluster.yaml" CACHE FILEPATH
    "Path to the local icclrun cluster.yaml used by CTest")

find_program(INFINICCL_TEST_ICCLRUN_EXECUTABLE NAMES icclrun)
if(NOT INFINICCL_TEST_ICCLRUN_EXECUTABLE)
    message(WARNING "`icclrun` was not found in PATH; CTest cases that require it will not be registered.")
endif()

set(INFINI_INSTALL_DIR $ENV{INFINICCL_INSTALL})
if(NOT INFINI_INSTALL_DIR)
    message(FATAL_ERROR "Environment variable `INFINICCL_INSTALL` is not set! Please export it before compiling.")
endif()

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
    "pure_ccl_tests"
    "mix_tests"
)

# =========================================================
# GLOBAL INCLUDE DIRECTORIES
# =========================================================
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${INFINI_INSTALL_DIR}/include"
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
            get_filename_component(test_stem "${source_file}" NAME_WE)
            string(REGEX REPLACE "_tests$" "" suite_name "${suite_dir}")
            set(test_name "${suite_name}_${test_stem}")

            # Instantiate the test binary target.
            add_executable(${test_name} "${source_file}")
            
            # Bind our compiled dependencies, drivers, and runtime targets.
            target_link_libraries(${test_name} 
                PRIVATE 
                    "${INFINI_INSTALL_DIR}/lib/libinfiniccl.so" 
                    ${GPU_LIBS} 
                    MPI::MPI_CXX
            )

            # Enforce runtime linker parameters (RPATH search mappings).
            set_target_properties(${test_name} PROPERTIES 
                OUTPUT_NAME "${test_stem}"
                BUILD_WITH_INSTALL_RPATH TRUE
                INSTALL_RPATH "${INFINI_INSTALL_DIR}/lib"
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${suite_dir}"
            )
            
            message(STATUS "  -> Registered test binary: ${suite_dir}/${test_stem}")

            if(INFINICCL_TEST_ENABLE_PURE_MPI
               AND INFINICCL_TEST_ICCLRUN_EXECUTABLE
               AND suite_dir STREQUAL "pure_mpi_tests"
               AND test_stem STREQUAL "all_reduce")
                add_test(
                    NAME pure_mpi.all_reduce.float32.sum.count1024
                    COMMAND
                        ${INFINICCL_TEST_ICCLRUN_EXECUTABLE}
                        --config ${INFINICCL_TEST_CLUSTER_CONFIG}
                        pure_mpi_tests/all_reduce
                        --dtype float32
                        --red-op sum
                        --count 1024
                        --pin-device0 $<IF:$<BOOL:${INFINICCL_TEST_PIN_ALL_RANKS_TO_DEVICE0}>,1,0>
                )
                set_tests_properties(
                    pure_mpi.all_reduce.float32.sum.count1024
                    PROPERTIES
                        LABELS "suite:pure_mpi;op:all_reduce;kind:reduction;dtype:float32;redop:sum;size:smoke"
                )
            endif()
        endforeach()
    endif()
endforeach()
