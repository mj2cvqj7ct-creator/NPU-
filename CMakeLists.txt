cmake_minimum_required(VERSION 3.20)

project(SnapdragonNpuAudioEnhancer VERSION 0.1.0 LANGUAGES CXX)

option(SNAE_ENABLE_TESTS "Build DSP unit tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(snae_core
    src/dsp/audio_frame.cpp
    src/dsp/audio_metrics.cpp
    src/dsp/biquad.cpp
    src/dsp/enhancement_pipeline.cpp
    src/dsp/limiter.cpp
    src/inference/enhancement_model.cpp
    src/profile/listening_profile.cpp
)

target_include_directories(snae_core PUBLIC src)

if(MSVC)
    target_compile_options(snae_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(snae_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(SNAE_ENABLE_TESTS)
    enable_testing()

    add_executable(snae_dsp_tests tests/dsp_pipeline_tests.cpp)
    target_link_libraries(snae_dsp_tests PRIVATE snae_core)

    add_test(NAME dsp_pipeline_tests COMMAND snae_dsp_tests)
endif()
