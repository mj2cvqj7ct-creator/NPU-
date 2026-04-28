cmake_minimum_required(VERSION 3.16)

project(SnapdragonNpuAudioEnhancer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(audio_enhancer
    src/dsp/audio_pipeline.cpp
    src/inference/npu_backend.cpp
    src/io/wav_io.cpp
)

target_include_directories(audio_enhancer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(audio_enhancer
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(audio_enhancer_cli src/app/main.cpp)
target_link_libraries(audio_enhancer_cli PRIVATE audio_enhancer)

enable_testing()

add_executable(audio_pipeline_tests tests/audio_pipeline_tests.cpp)
target_link_libraries(audio_pipeline_tests PRIVATE audio_enhancer)
add_test(NAME audio_pipeline_tests COMMAND audio_pipeline_tests)
