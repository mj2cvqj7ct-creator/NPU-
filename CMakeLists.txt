cmake_minimum_required(VERSION 3.20)

project(
  SnapdragonNpuAudioEnhancer
  VERSION 0.1.0
  DESCRIPTION "Low-latency audio enhancement pipeline for ARM64 Snapdragon X NPU targets"
  LANGUAGES CXX)

option(NPU_AUDIO_ENABLE_QNN "Enable Qualcomm QNN backend selection hooks" OFF)

add_library(npu_audio_enhancer
  src/dsp/enhancer.cpp
  src/dsp/realtime_enhancer.cpp
  src/inference/backend.cpp
  src/io/wav_io.cpp)

target_compile_features(npu_audio_enhancer PUBLIC cxx_std_20)
target_include_directories(npu_audio_enhancer PUBLIC include)

if(NPU_AUDIO_ENABLE_QNN)
  target_compile_definitions(npu_audio_enhancer PUBLIC NPU_AUDIO_ENABLE_QNN=1)
endif()

if(MSVC)
  target_compile_options(npu_audio_enhancer PRIVATE /W4 /permissive-)
else()
  target_compile_options(npu_audio_enhancer PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(npu_audio_enhance tools/enhance_wav.cpp)
target_link_libraries(npu_audio_enhance PRIVATE npu_audio_enhancer)

enable_testing()

add_executable(npu_audio_tests tests/dsp_tests.cpp)
target_link_libraries(npu_audio_tests PRIVATE npu_audio_enhancer)
add_test(NAME npu_audio_tests COMMAND npu_audio_tests)
