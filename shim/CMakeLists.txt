cmake_minimum_required(VERSION 3.16)
project(ggml_npu_openvino_shim LANGUAGES CXX)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
include(CTest)

set(
    GGML_NPU_SHIM_OUTPUT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
    CACHE PATH
    "Directory for the runtime compiler shim"
)

add_library(ggml_npu_ovc SHARED ov_compile.cpp)
target_compile_features(ggml_npu_ovc PRIVATE cxx_std_17)
target_compile_options(ggml_npu_ovc PRIVATE -O2)
target_link_libraries(ggml_npu_ovc PRIVATE openvino::runtime)

if(BUILD_TESTING)
    add_executable(ggml_npu_quant_graph_test quant_graph_test.cpp)
    target_compile_features(ggml_npu_quant_graph_test PRIVATE cxx_std_17)
    target_link_libraries(ggml_npu_quant_graph_test PRIVATE ggml_npu_ovc)
    add_test(NAME quant_graph_descriptors COMMAND ggml_npu_quant_graph_test)

    # Explicit benchmark target: compiling a native NPU artifact is hardware/toolchain dependent,
    # so this is built with tests but not run by CTest.
    add_executable(ggml_npu_quant_compile_bench quant_compile_bench.cpp)
    target_compile_features(ggml_npu_quant_compile_bench PRIVATE cxx_std_17)
    target_link_libraries(ggml_npu_quant_compile_bench PRIVATE ggml_npu_ovc openvino::runtime)
endif()

set_target_properties(
    ggml_npu_ovc
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${GGML_NPU_SHIM_OUTPUT_DIR}"
)
