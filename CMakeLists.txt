cmake_minimum_required(VERSION 3.20)

project(snapdragon_x_npu_audio_enhancer
    VERSION 0.1.0
    DESCRIPTION "Low-latency Snapdragon X NPU assisted audio enhancement core"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SXNAE_BUILD_TESTS "Build unit tests" ON)

add_library(sxnae_core
    src/dsp/audio_metrics.cpp
    src/dsp/enhancer.cpp
    src/inference/npu_backend.cpp
    src/profile/service_profile.cpp
)

target_include_directories(sxnae_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(sxnae_core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(sxnae_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(sxnae_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(sxnae_demo src/main.cpp)
target_link_libraries(sxnae_demo PRIVATE sxnae_core)

if(SXNAE_BUILD_TESTS)
    enable_testing()

    add_executable(dsp_pipeline_tests tests/dsp_pipeline_tests.cpp)
    target_link_libraries(dsp_pipeline_tests PRIVATE sxnae_core)

    add_test(NAME dsp_pipeline_tests COMMAND dsp_pipeline_tests)
endif()
