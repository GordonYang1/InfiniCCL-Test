cmake_minimum_required(VERSION 3.18)
project(InfiniCCLUserTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    "ccl_mpi_tests"
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
            get_filename_component(test_name "${source_file}" NAME_WE)
            
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
                BUILD_WITH_INSTALL_RPATH TRUE
                INSTALL_RPATH "${INFINI_INSTALL_DIR}/lib"
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${suite_dir}"
            )
            
            message(STATUS "  -> Registered test binary: ${suite_dir}/${test_name}")
        endforeach()
    endif()
endforeach()
