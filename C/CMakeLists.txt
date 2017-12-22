# We need some FindCUDA features present in cmake >= 3.7
cmake_minimum_required (VERSION 3.7)
project (MemoryReliability)

# Default build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
   message(STATUS "No build type selected, defaulting to Release")
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

# Restrict to only DEBUG | RELEASE | RELWITHDEBINFO :
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (CMAKE_BUILD_TYPE AND
      NOT uppercase_CMAKE_BUILD_TYPE MATCHES "^(DEBUG|RELEASE|RELWITHDEBINFO)$")
   message(FATAL_ERROR "Invalid value for CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

# Project options
option(ENABLE_ERROR_INJECTION "Inject randomly errors in allocated memory region" OFF)
if(ENABLE_ERROR_INJECTION)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DINJECT_ERR")
endif()

option(ENABLE_CUDA "Test GPU memory" ON)
find_package(CUDA)
if (ENABLE_CUDA AND CUDA_FOUND)
    add_definitions(-DUSE_CUDA)

    set(CUDA_PROPAGATE_HOST_FLAGS ON)

    set(CUDA_NVCC_FLAGS "-std=c++11 ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=\"sm_30,sm_35,sm_37\" -gencode arch=compute_50,code=sm_52 -gencode arch=compute_60,code=\"sm_60,sm_61\" -lineinfo")
    set(CUDA_NVCC_FLAGS_DEBUG   "-O0 -g -G")
    set(CUDA_NVCC_FLAGS_RELEASE "-O3       -res-usage")

    message(STATUS "--------------------------------------")
    message(STATUS "----[CUDA ENABLED] :                  ")
    message(STATUS "---- CUDA_VERSION:      ${CUDA_VERSION} (${CUDA_VERSION_STRING})")
    message(STATUS "---- CUDA_TOOLKIT_DIR:  ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "---- CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "---- CUDA_LIBRARIES:    ${CUDA_LIBRARIES}")
    message(STATUS "--------                      --------")
    message(STATUS "---- CUDA_HOST_COMPILER: ${CUDA_HOST_COMPILER}")
    message(STATUS "--------------------------------------")

elseif(ENABLE_CUDA AND NOT CUDA_FOUND)
    message(INFO "CUDA not found")
else()
endif()


# C Standard
set(CMAKE_C_STANDARD_REQUIRED 99)

# Compile flags
if (("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU") OR
    ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang"))
    # gcc/clang flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    # icc flags
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    # pgcc flags
else ()
    message(WARNING "Compiler not supported!")
endif ()

message(STATUS "----------------------------------------")
message(STATUS "-----------  SDCfinder    --------------")
message(STATUS "----------------------------------------")
message(STATUS "CPU TYPE:                       ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "BUILD TYPE:                     ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_C_FLAGS:                  ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_C_FLAGS_DEBUG:            ${CMAKE_C_FLAGS_DEBUG}")
message(STATUS "CMAKE_C_FLAGS_RELEASE:          ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS:         ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS_DEBUG:   ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS_RELEASE: ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
message(STATUS "CUDA_NVCC_FLAGS:                ${CUDA_NVCC_FLAGS}")
message(STATUS "CUDA_NVCC_FLAGS_DEBUG:          ${CUDA_NVCC_FLAGS_DEBUG}")
message(STATUS "CUDA_NVCC_FLAGS_RELEASE:        ${CUDA_NVCC_FLAGS_RELEASE}")
message(STATUS "----------------------------------------")

add_library(addr    ${CMAKE_CURRENT_SOURCE_DIR}/lib/addresstranslation.c)
target_include_directories(addr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(logging ${CMAKE_CURRENT_SOURCE_DIR}/lib/logging.c)
target_include_directories(logging PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(logging addr)

set(MRLIB_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/MemoryReliability_defs.c
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/pmu.c
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/tools.c)
add_library(MRlib STATIC ${MRLIB_SRC})
target_include_directories(MRlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(MRlib logging)

if (ENABLE_CUDA AND CUDA_FOUND)
    cuda_add_library(MRlib-cuda STATIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/tools.cu)
    set_target_properties(MRlib-cuda PROPERTIES CXX_STANDARD 11)
    target_include_directories(MRlib-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

    target_include_directories(MRlib PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(MRlib MRlib-cuda stdc++ ${CUDA_LIBRARIES})
endif()

add_executable(MemoryReliability MemoryReliability.c)
target_include_directories(MemoryReliability PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(MemoryReliability MRlib)
install(TARGETS MemoryReliability RUNTIME DESTINATION bin)