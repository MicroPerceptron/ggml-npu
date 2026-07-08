// Runtime OpenVINO compile shim for the ggml NPU backend.
//
// A small C++ island that links OpenVINO and exposes a C ABI. The core backend is a no_std
// Rust cdylib with NO OpenVINO dependency; it `dlopen`s this shim on a cache miss to compile
// a native NPU blob for a given op shape at runtime. Only needed on Intel hosts — where the
// shim (or OpenVINO) is absent, the backend simply falls back to CPU.
//
// Build: shim/build.sh  ->  libggml_npu_ovc.so
#include <openvino/openvino.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

// One process-lifetime Core (loading plugins is expensive); thread-safe init.
static ov::Core& core() {
    static ov::Core c;
    return c;
}

extern "C" {

// Compile a ggml-convention MatMul for the NPU: A[M,K] @ B[N,K]^T -> [M,N] (transpose_b),
// with BOTH operands as inputs (shape-generic; weights are not baked). On success, writes a
// malloc'd native blob to *blob / *len (free with ggml_npu_ov_free) and returns 0.
int ggml_npu_ov_compile_matmul(int32_t M, int32_t K, int32_t N, uint8_t** blob, size_t* len) {
    if (!blob || !len || M <= 0 || K <= 0 || N <= 0) {
        return -1;
    }
    try {
        auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
        auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)N, (size_t)K});
        auto mm = std::make_shared<ov::op::v0::MatMul>(a, b, /*transpose_a=*/false, /*transpose_b=*/true);
        auto r = std::make_shared<ov::op::v0::Result>(mm);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{r}, ov::ParameterVector{a, b});

        auto compiled = core().compile_model(model, "NPU");
        std::ostringstream os(std::ios::binary);
        compiled.export_model(os);
        const std::string s = os.str();

        uint8_t* buf = static_cast<uint8_t*>(std::malloc(s.size()));
        if (!buf) {
            return -2;
        }
        std::memcpy(buf, s.data(), s.size());
        *blob = buf;
        *len = s.size();
        return 0;
    } catch (const std::exception&) {
        return -3;
    } catch (...) {
        return -4;
    }
}

void ggml_npu_ov_free(uint8_t* blob) {
    std::free(blob);
}

}  // extern "C"
