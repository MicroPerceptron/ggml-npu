cmake_minimum_required(VERSION 3.16)
project(ggml_npu_openvino_shim LANGUAGES CXX)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_library(ggml_npu_ovc SHARED ov_compile.cpp)
target_compile_features(ggml_npu_ovc PRIVATE cxx_std_17)
target_compile_options(ggml_npu_ovc PRIVATE -O2)
target_link_libraries(ggml_npu_ovc PRIVATE openvino::runtime)

set_target_properties(
    ggml_npu_ovc
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
