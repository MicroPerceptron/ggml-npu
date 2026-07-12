#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

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

extern "C" int ggml_npu_ov_inspect_constant_v2(
    const npu_constant_desc*, int32_t, int32_t, uint32_t*);

enum : uint32_t {
    INSPECT_U4 = 1u << 0,
    INSPECT_I8 = 1u << 1,
    INSPECT_SCALE = 1u << 2,
    INSPECT_OFFSET = 1u << 3,
    INSPECT_F32 = 1u << 4,
};

static npu_constant_desc q4_descriptor(
    const std::vector<uint8_t>& values,
    const std::vector<uint8_t>& scale_base,
    const std::vector<uint8_t>& scale_factor,
    const std::vector<uint8_t>& offset_base,
    const std::vector<uint8_t>& offset_factor) {
    return {
        1, 4, 4, 1, 256, 32, 1, 2, 2, 1,
        values.data(), values.size(),
        scale_base.data(), scale_base.size(),
        scale_factor.data(), scale_factor.size(),
        offset_base.data(), offset_base.size(),
        offset_factor.data(), offset_factor.size(),
    };
}

static npu_constant_desc q6_descriptor(
    const std::vector<uint8_t>& values,
    const std::vector<uint8_t>& scale_base,
    const std::vector<uint8_t>& scale_factor) {
    return {
        1, 6, 6, 2, 256, 16, 1, 3, 0, 1,
        values.data(), values.size(),
        scale_base.data(), scale_base.size(),
        scale_factor.data(), scale_factor.size(),
        nullptr, 0, nullptr, 0,
    };
}

int main() {
    std::vector<uint8_t> q4_values(128, 0x10);
    std::vector<uint8_t> q4_scale_base{0x00, 0x38};
    std::vector<uint8_t> q4_scale_factor(8, 1);
    std::vector<uint8_t> q4_offset_base{0x00, 0x34};
    std::vector<uint8_t> q4_offset_factor(8, 1);
    auto q4 = q4_descriptor(
        q4_values, q4_scale_base, q4_scale_factor, q4_offset_base, q4_offset_factor);
    uint32_t q4_flags = 0;
    if (ggml_npu_ov_inspect_constant_v2(&q4, 1, 256, &q4_flags) != 0 ||
        (q4_flags & (INSPECT_U4 | INSPECT_SCALE | INSPECT_OFFSET)) !=
            (INSPECT_U4 | INSPECT_SCALE | INSPECT_OFFSET) ||
        (q4_flags & (INSPECT_I8 | INSPECT_F32)) != 0) {
        std::fprintf(stderr, "Q4 descriptor graph flags: 0x%x\n", q4_flags);
        return 1;
    }

    std::vector<uint8_t> q6_values(192, 0x20);
    std::vector<uint8_t> q6_scale_base{0x00, 0x30};
    std::vector<uint8_t> q6_scale_factor(16, 1);
    auto q6 = q6_descriptor(q6_values, q6_scale_base, q6_scale_factor);
    uint32_t q6_flags = 0;
    if (ggml_npu_ov_inspect_constant_v2(&q6, 1, 256, &q6_flags) != 0 ||
        (q6_flags & (INSPECT_I8 | INSPECT_SCALE)) != (INSPECT_I8 | INSPECT_SCALE) ||
        (q6_flags & (INSPECT_U4 | INSPECT_OFFSET | INSPECT_F32)) != 0) {
        std::fprintf(stderr, "Q6 descriptor graph flags: 0x%x\n", q6_flags);
        return 2;
    }
    return 0;
}
