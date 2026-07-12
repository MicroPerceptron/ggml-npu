#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <vector>

#include <openvino/openvino.hpp>

struct npu_constant_desc {
    int32_t numeric_kind, bit_width, storage_bits, signedness;
    int32_t block_elems, group_elems, packing;
    int32_t scale_encoding, offset_encoding, exactness;
    const uint8_t* values;
    size_t values_len;
    const uint8_t* scale_base;
    size_t scale_base_len;
    const uint8_t* scale_factor;
    size_t scale_factor_len;
    const uint8_t* offset_base;
    size_t offset_base_len;
    const uint8_t* offset_factor;
    size_t offset_factor_len;
};

extern "C" int ggml_npu_ov_compile_matmul_constant_v2(
    int32_t, int32_t, int32_t, const npu_constant_desc*, const char*, uint8_t**, size_t*);
extern "C" void ggml_npu_ov_free(uint8_t*);

static bool benchmark_artifact(
    const char* name, const uint8_t* blob, size_t length, float expected_output) {
    if (!std::getenv("GGML_NPU_RUN_ARTIFACT")) return true;
    try {
        std::string bytes(reinterpret_cast<const char*>(blob), length);
        std::istringstream stream(bytes, std::ios::binary);
        ov::Core core;
        auto compiled = core.import_model(stream, "NPU");
        auto request = compiled.create_infer_request();
        std::vector<float> input(256, 1.0f);
        request.set_input_tensor(ov::Tensor(ov::element::f32, ov::Shape{1, 256}, input.data()));
        for (int warmup = 0; warmup < 5; ++warmup) request.infer();
        std::vector<double> samples;
        samples.reserve(100);
        for (int iteration = 0; iteration < 100; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            request.infer();
            samples.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count());
        }
        const auto output = request.get_output_tensor();
        const float value = output.data<const float>()[0];
        std::sort(samples.begin(), samples.end());
        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        std::printf(
            "%s warm_infer_mean_us=%.3f warm_infer_p50_us=%.3f output=%g\n",
            name, mean, samples[samples.size() / 2], value);
        return std::isfinite(value) && std::abs(value - expected_output) <= 0.01f;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s infer failed: %s\n", name, error.what());
        return false;
    }
}

static int compile_and_report(
    const char* name, const npu_constant_desc& descriptor, float expected_output) {
    uint8_t* blob = nullptr;
    size_t length = 0;
    const auto start = std::chrono::steady_clock::now();
    const int status = ggml_npu_ov_compile_matmul_constant_v2(
        1, 256, 1, &descriptor, nullptr, &blob, &length);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start);
    std::printf(
        "%s status=%d compile_ms=%.3f artifact_bytes=%zu\n",
        name, status, elapsed.count(), length);
    const bool inference_ok =
        status == 0 && benchmark_artifact(name, blob, length, expected_output);
    ggml_npu_ov_free(blob);
    return inference_ok ? status : -5;
}

int main() {
    const std::vector<uint8_t> canonical_values(256, 1);
    std::vector<uint8_t> canonical_scales(16, 0);
    for (size_t index = 0; index < canonical_scales.size(); index += 2) {
        canonical_scales[index] = 0x00;
        canonical_scales[index + 1] = 0x38; // IEEE f16 0.5
    }
    const std::vector<uint8_t> canonical_zeros(8, 0);
    const npu_constant_desc canonical_u4{
        1, 4, 8, 1, 32, 32, 0, 1, 1, 0,
        canonical_values.data(), canonical_values.size(),
        canonical_scales.data(), canonical_scales.size(),
        nullptr, 0,
        canonical_zeros.data(), canonical_zeros.size(),
        nullptr, 0,
    };
    const npu_constant_desc canonical_u8{
        1, 8, 8, 1, 32, 32, 0, 1, 1, 0,
        canonical_values.data(), canonical_values.size(),
        canonical_scales.data(), canonical_scales.size(),
        nullptr, 0,
        canonical_zeros.data(), canonical_zeros.size(),
        nullptr, 0,
    };

    const std::vector<uint8_t> q4_values(128, 0x10);
    const std::vector<uint8_t> q4_scale_base{0x00, 0x3c};
    const std::vector<uint8_t> q4_scale_factor(8, 1);
    const std::vector<uint8_t> q4_offset_base{0x00, 0x38};
    const std::vector<uint8_t> q4_offset_factor(8, 1);
    const npu_constant_desc q4{
        1, 4, 4, 1, 256, 32, 1, 2, 2, 1,
        q4_values.data(), q4_values.size(),
        q4_scale_base.data(), q4_scale_base.size(),
        q4_scale_factor.data(), q4_scale_factor.size(),
        q4_offset_base.data(), q4_offset_base.size(),
        q4_offset_factor.data(), q4_offset_factor.size(),
    };

    const std::vector<uint8_t> q6_values(192, 0x20);
    const std::vector<uint8_t> q6_scale_base{0x00, 0x3c};
    const std::vector<uint8_t> q6_scale_factor(16, 1);
    const npu_constant_desc q6{
        1, 6, 6, 2, 256, 16, 1, 3, 0, 1,
        q6_values.data(), q6_values.size(),
        q6_scale_base.data(), q6_scale_base.size(),
        q6_scale_factor.data(), q6_scale_factor.size(),
        nullptr, 0, nullptr, 0,
    };

    const int canonical_u4_status = compile_and_report("canonical_u4", canonical_u4, 128.0f);
    const int canonical_u8_status = compile_and_report("canonical_u8", canonical_u8, 128.0f);
    const int q4_status = compile_and_report("q4_k_native", q4, 0.0f);
    const int q6_status = compile_and_report("q6_k_native", q6, -1408.0f);
    return canonical_u4_status == 0 && canonical_u8_status == 0 &&
            q4_status == 0 && q6_status == 0 ? 0 : 1;
}
