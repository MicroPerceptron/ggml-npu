// Runtime OpenVINO compile shim for the ggml NPU backend.
//
// A small C++ island that links OpenVINO and exposes a C ABI. The core backend is a no_std
// Rust cdylib with NO OpenVINO dependency; it `dlopen`s this shim on a cache miss to compile
// a native NPU blob for a given op shape at runtime. Only needed on Intel hosts — where the
// shim (or OpenVINO) is absent, the backend simply falls back to CPU.
//
// Build: shim/build.sh  ->  libggml_npu_ovc.so
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/group_conv.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/gather_elements.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/scatter_update.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/op/tile.hpp>
#include <openvino/op/unsqueeze.hpp>

#include <sys/stat.h>

#include <cstdint>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

// One process-lifetime Core (loading plugins is expensive); thread-safe init.
static ov::Core& core() {
    static ov::Core c;
    return c;
}

// ---------------------------------------------------------------------------
// On-disk blob cache (managed here in C++ so the Rust core stays no_std / file-I/O-free).
// ---------------------------------------------------------------------------

// Resolve + create the cache directory: $GGML_NPU_CACHE, else $XDG_CACHE_HOME/ggml-npu, else
// $HOME/.cache/ggml-npu. Returns "" if none resolvable (caching disabled — still compiles).
static std::string cache_dir() {
    std::string dir;
    if (const char* e = std::getenv("GGML_NPU_CACHE"); e && *e) {
        dir = e;
    } else if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) {
        dir = std::string(x) + "/ggml-npu";
    } else if (const char* h = std::getenv("HOME"); h && *h) {
        dir = std::string(h) + "/.cache/ggml-npu";
    }
    if (!dir.empty()) {
        ::mkdir(dir.c_str(), 0755); // best-effort; leaf only (parent assumed to exist)
    }
    return dir;
}

// Read an entire file into a malloc'd buffer (freed by the Rust caller via ggml_npu_ov_free).
static bool read_blob(const std::string& path, uint8_t** out, size_t* out_len) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        return false;
    }
    std::streamsize n = ifs.tellg();
    if (n <= 0) {
        return false;
    }
    ifs.seekg(0);
    uint8_t* buf = static_cast<uint8_t*>(std::malloc((size_t)n));
    if (!buf) {
        return false;
    }
    if (!ifs.read(reinterpret_cast<char*>(buf), n)) {
        std::free(buf);
        return false;
    }
    *out = buf;
    *out_len = (size_t)n;
    return true;
}

static size_t cache_budget_bytes() {
    const char* value = std::getenv("GGML_NPU_DISK_CACHE_MB");
    if (!value || !*value) {
        return 0;
    }
    char* end = nullptr;
    unsigned long long mib = std::strtoull(value, &end, 10);
    if (!end || *end != '\0' || mib == 0 ||
        mib > static_cast<unsigned long long>(SIZE_MAX / (1024 * 1024))) {
        return 0;
    }
    return static_cast<size_t>(mib) * 1024 * 1024;
}

static bool reserve_cache_bytes(const std::string& path, size_t incoming) {
    const size_t budget = cache_budget_bytes();
    if (budget == 0) {
        return true;
    }
    if (incoming > budget) {
        return false;
    }
    namespace fs = std::filesystem;
    std::error_code error;
    fs::path target(path);
    std::vector<fs::directory_entry> blobs;
    uintmax_t total = 0;
    for (fs::directory_iterator it(target.parent_path(), error), end; !error && it != end;
         it.increment(error)) {
        if (!it->is_regular_file(error) || it->path() == target ||
            it->path().extension() != ".blob") {
            continue;
        }
        total += it->file_size(error);
        blobs.push_back(*it);
    }
    std::sort(blobs.begin(), blobs.end(), [&error](const auto& left, const auto& right) {
        return left.last_write_time(error) < right.last_write_time(error);
    });
    for (const auto& entry : blobs) {
        if (total + incoming <= budget) {
            break;
        }
        const uintmax_t bytes = entry.file_size(error);
        if (fs::remove(entry.path(), error)) {
            total = bytes > total ? 0 : total - bytes;
        }
    }
    return total + incoming <= budget;
}

// Write blob bytes atomically (temp + rename) so a concurrent/aborted run never loads a partial.
static void write_blob(const std::string& path, const void* data, size_t len) {
    if (!reserve_cache_bytes(path, len)) {
        return;
    }
    std::string tmp = path + ".tmp";
    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs || !ofs.write(static_cast<const char*>(data), (std::streamsize)len)) {
            std::remove(tmp.c_str());
            return;
        }
    }
    std::rename(tmp.c_str(), path.c_str());
}

static std::shared_ptr<ov::Node> reshape_to(const std::shared_ptr<ov::Node>& x,
                                            std::vector<int64_t> shape) {
    auto c = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    return std::make_shared<ov::op::v1::Reshape>(x, c, false);
}

static std::shared_ptr<ov::Node> reshape_output(const ov::Output<ov::Node>& x,
                                                std::vector<int64_t> shape) {
    auto c = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    return std::make_shared<ov::op::v1::Reshape>(x, c, false);
}

enum npu_numeric_kind { NPU_NUMERIC_FLOAT = 0, NPU_NUMERIC_INTEGER = 1 };
enum npu_signedness { NPU_SIGN_NONE = 0, NPU_UNSIGNED = 1, NPU_SIGNED = 2 };
enum npu_packing { NPU_UNPACKED = 0, NPU_DENSE_LSB = 1 };
enum npu_scale_encoding {
    NPU_SCALE_NONE = 0,
    NPU_SCALE_GROUP_F16 = 1,
    NPU_SCALE_BLOCK_F16_GROUP_U8 = 2,
    NPU_SCALE_BLOCK_F16_GROUP_I8 = 3,
};
enum npu_offset_encoding {
    NPU_OFFSET_NONE = 0,
    NPU_OFFSET_ZERO_U8 = 1,
    NPU_OFFSET_NEG_BLOCK_F16_GROUP_U8 = 2,
};
enum npu_exactness { NPU_REQUANTIZED = 0, NPU_SOURCE_NUMERIC = 1, NPU_SOURCE_BYTES = 2 };

// Compiler-facing constant contract. It intentionally contains semantic fields rather than source
// format or Intel names; this shim alone maps the contract to OpenVINO graph patterns.
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

static bool expected_packed_bytes(size_t count, int32_t bits, size_t& bytes) {
    if (bits <= 0 || bits > 16 || count > (SIZE_MAX - 7) / (size_t)bits) return false;
    bytes = (count * (size_t)bits + 7) / 8;
    return true;
}

static uint16_t packed_value(const uint8_t* source, size_t index, int32_t bits) {
    const size_t bit = index * (size_t)bits;
    const size_t byte = bit / 8;
    const int shift = (int)(bit % 8);
    uint16_t value = (uint16_t)(source[byte] >> shift);
    if (shift + bits > 8) value |= (uint16_t)source[byte + 1] << (8 - shift);
    return value & (uint16_t)((1u << bits) - 1u);
}

static bool unpack_integers(const npu_constant_desc& desc, size_t count,
                            std::vector<uint8_t>& unsigned_values,
                            std::vector<int8_t>& signed_values) {
    size_t expected = 0;
    if (desc.packing == NPU_DENSE_LSB) {
        if (!expected_packed_bytes(count, desc.storage_bits, expected) ||
            desc.values_len != expected || desc.storage_bits != desc.bit_width) return false;
    } else if (desc.packing == NPU_UNPACKED) {
        expected = count;
        if (desc.values_len != expected || desc.storage_bits != 8) return false;
    } else {
        return false;
    }
    if (desc.signedness == NPU_SIGNED) {
        signed_values.resize(count);
        for (size_t i = 0; i < count; ++i) {
            uint8_t raw = desc.packing == NPU_DENSE_LSB
                ? (uint8_t)packed_value(desc.values, i, desc.bit_width)
                : desc.values[i];
            if (desc.bit_width < 8) {
                const int shift = 8 - desc.bit_width;
                signed_values[i] = (int8_t)((int8_t)(raw << shift) >> shift);
            } else {
                signed_values[i] = (int8_t)raw;
            }
        }
    } else if (desc.signedness == NPU_UNSIGNED) {
        unsigned_values.resize(count);
        for (size_t i = 0; i < count; ++i) {
            unsigned_values[i] = desc.packing == NPU_DENSE_LSB
                ? (uint8_t)packed_value(desc.values, i, desc.bit_width)
                : desc.values[i];
        }
    } else {
        return false;
    }
    return true;
}

static std::shared_ptr<ov::Node> integer_constant(const npu_constant_desc& desc,
                                                  const ov::Shape& shape, size_t count) {
    namespace v0 = ov::op::v0;
    std::vector<uint8_t> unsigned_values;
    std::vector<int8_t> signed_values;
    if (!desc.values || !unpack_integers(desc, count, unsigned_values, signed_values)) return nullptr;
    if (desc.signedness == NPU_UNSIGNED && desc.bit_width == 4) {
        return v0::Constant::create(ov::element::u4, shape, unsigned_values);
    }
    if (desc.signedness == NPU_SIGNED && desc.bit_width == 4) {
        return v0::Constant::create(ov::element::i4, shape, signed_values);
    }
    if (desc.signedness == NPU_SIGNED) {
        return v0::Constant::create(ov::element::i8, shape, signed_values);
    }
    return v0::Constant::create(ov::element::u8, shape, unsigned_values);
}

// Build the dequantized f16 `[N,K]` constant. Source-native affine metadata remains factorized:
// Q4-style values use `q * (block_scale * group_scale) - block_min * group_min`; signed values use
// `q * (block_scale * group_scale)`. Canonical fallback retains `(q-zero)*scale` byte-for-byte.
static std::shared_ptr<ov::Node> make_weight(const npu_constant_desc& desc, int64_t N, int64_t K) {
    namespace v0 = ov::op::v0;
    namespace v1 = ov::op::v1;
    if (N <= 0 || K <= 0 || !desc.values) return nullptr;
    const size_t count = (size_t)N * (size_t)K;
    if (desc.numeric_kind == NPU_NUMERIC_FLOAT) {
        if (desc.bit_width != 16 || desc.storage_bits != 16 || desc.packing != NPU_UNPACKED ||
            desc.values_len != count * 2 || desc.scale_encoding != NPU_SCALE_NONE ||
            desc.offset_encoding != NPU_OFFSET_NONE) return nullptr;
        return std::make_shared<v0::Constant>(ov::element::f16, ov::Shape{(size_t)N, (size_t)K},
                                              (const void*)desc.values);
    }
    if (desc.numeric_kind != NPU_NUMERIC_INTEGER || desc.group_elems <= 0 ||
        desc.block_elems <= 0 || K % desc.block_elems != 0 ||
        desc.block_elems % desc.group_elems != 0) {
        return nullptr;
    }
    const int64_t blocks = K / desc.block_elems;
    const int64_t groups_per_block = desc.block_elems / desc.group_elems;
    const int64_t groups = K / desc.group_elems;
    const ov::Shape wshape{
        (size_t)N, (size_t)blocks, (size_t)groups_per_block, (size_t)desc.group_elems};
    const ov::Shape gshape{(size_t)N, (size_t)blocks, (size_t)groups_per_block, 1};

    if (std::getenv("GGML_NPU_DEBUG")) {
        std::fprintf(stderr,
            "[mw2] bits=%d storage=%d signed=%d block=%d group=%d packing=%d scale=%d offset=%d exact=%d N=%ld K=%ld\n",
            desc.bit_width, desc.storage_bits, desc.signedness, desc.block_elems, desc.group_elems,
            desc.packing, desc.scale_encoding, desc.offset_encoding, desc.exactness, (long)N, (long)K);
    }

    auto integer = integer_constant(desc, wshape, count);
    if (!integer) return nullptr;
    std::shared_ptr<ov::Node> deq = std::make_shared<v0::Convert>(integer, ov::element::f16);

    if (desc.scale_encoding == NPU_SCALE_NONE) {
        if (desc.offset_encoding != NPU_OFFSET_NONE) return nullptr;
        return reshape_to(deq, {N, K});
    }

    if (desc.scale_encoding == NPU_SCALE_GROUP_F16) {
        if (!desc.scale_base || desc.scale_base_len != (size_t)N * (size_t)groups * 2) return nullptr;
        if (desc.offset_encoding == NPU_OFFSET_ZERO_U8) {
            if (!desc.offset_base || desc.offset_base_len != (size_t)N * (size_t)groups) return nullptr;
            auto zero = std::make_shared<v0::Constant>(
                ov::element::u8, gshape, (const void*)desc.offset_base);
            deq = std::make_shared<v1::Subtract>(
                deq, std::make_shared<v0::Convert>(zero, ov::element::f16));
        } else if (desc.offset_encoding != NPU_OFFSET_NONE) {
            return nullptr;
        }
        auto scale = std::make_shared<v0::Constant>(
            ov::element::f16, gshape, (const void*)desc.scale_base);
        return reshape_to(std::make_shared<v1::Multiply>(deq, scale), {N, K});
    }

    const bool signed_factor = desc.scale_encoding == NPU_SCALE_BLOCK_F16_GROUP_I8;
    if (!signed_factor && desc.scale_encoding != NPU_SCALE_BLOCK_F16_GROUP_U8) return nullptr;
    if (!desc.scale_base || desc.scale_base_len != (size_t)N * (size_t)blocks * 2 ||
        !desc.scale_factor || desc.scale_factor_len != (size_t)N * (size_t)groups) return nullptr;
    const ov::Shape bshape{(size_t)N, (size_t)blocks, 1, 1};
    auto scale_base = std::make_shared<v0::Constant>(
        ov::element::f16, bshape, (const void*)desc.scale_base);
    auto scale_factor = std::make_shared<v0::Constant>(
        signed_factor ? ov::element::i8 : ov::element::u8, gshape,
        (const void*)desc.scale_factor);
    auto scale = std::make_shared<v1::Multiply>(
        scale_base, std::make_shared<v0::Convert>(scale_factor, ov::element::f16));
    deq = std::make_shared<v1::Multiply>(deq, scale);

    if (desc.offset_encoding == NPU_OFFSET_NEG_BLOCK_F16_GROUP_U8) {
        if (!desc.offset_base || desc.offset_base_len != (size_t)N * (size_t)blocks * 2 ||
            !desc.offset_factor || desc.offset_factor_len != (size_t)N * (size_t)groups) return nullptr;
        auto offset_base = std::make_shared<v0::Constant>(
            ov::element::f16, bshape, (const void*)desc.offset_base);
        auto offset_factor = std::make_shared<v0::Constant>(
            ov::element::u8, gshape, (const void*)desc.offset_factor);
        auto offset = std::make_shared<v1::Multiply>(
            offset_base, std::make_shared<v0::Convert>(offset_factor, ov::element::f16));
        deq = std::make_shared<v1::Subtract>(deq, offset);
    } else if (desc.offset_encoding != NPU_OFFSET_NONE) {
        return nullptr;
    }
    return reshape_to(deq, {N, K});
}

// ---------------------------------------------------------------------------
// Fused-subgraph ABI: a flat op-DAG the Rust core recognizes and hands over to be compiled as ONE
// ov::Model (so the NPU runs it as one program: intra-graph fusion + pipelining, one submit). Op
// codes and struct layouts must match src/vendor/ivpu/lower/subgraph.rs.
// ---------------------------------------------------------------------------
enum { NPU_SG_MATMUL = 1, NPU_SG_MUL = 2, NPU_SG_SILU = 3, NPU_SG_ADD = 4, NPU_SG_RMSNORM = 5 };

// Value reference: r in [0, n_inputs) → external input r; r >= n_inputs → node (r - n_inputs); < 0
// → unused. All values are f16 inside the graph (inputs are Converted at the boundary).
struct npu_sg_node {
    int32_t op;
    int32_t a;         // first value ref (matmul: activation; mul: lhs; silu: operand)
    int32_t b;         // second value ref (mul: rhs) or -1
    int32_t const_idx; // baked weight index (matmul) or -1
    int32_t out_n;     // matmul output width N (else unused)
};

struct npu_sg_const {
    int32_t n, k;
    struct npu_constant_desc constant;
};

// Graph IR v2 wire ABI. Opcodes are backend-owned (not ggml enum values), allowing the Rust
// capture layer and this independent Intel lowerer to evolve without including ggml headers.
enum npu_ir_op {
    NPU_IR_MATMUL = 1, NPU_IR_MUL = 2, NPU_IR_ADD = 3, NPU_IR_RMSNORM = 4,
    NPU_IR_SILU = 5, NPU_IR_RESHAPE = 6, NPU_IR_VIEW = 7, NPU_IR_CONT = 8,
    NPU_IR_TRANSPOSE = 9, NPU_IR_CONCAT = 10, NPU_IR_COPY = 11, NPU_IR_SSM_CONV = 12,
    NPU_IR_ROPE = 13, NPU_IR_SOFTMAX = 14, NPU_IR_GET_ROWS = 15, NPU_IR_ARGSORT = 16, NPU_IR_MATMUL_ID = 17,
    NPU_IR_DIV = 18, NPU_IR_SCALE = 19, NPU_IR_CLAMP = 20,
    NPU_IR_PERMUTE = 21, NPU_IR_SWIGLU = 23, NPU_IR_SIGMOID = 24, NPU_IR_SUM_ROWS = 25,
    NPU_IR_SET_ROWS = 26, NPU_IR_FLASH_ATTN = 22,
};
enum npu_ir_role { NPU_IR_INPUT = 0, NPU_IR_CONSTANT = 1, NPU_IR_STATE = 2,
                   NPU_IR_INTERMEDIATE = 3, NPU_IR_OUTPUT = 4 };
static constexpr int NPU_IR_MAX_INPUTS = 16;
static constexpr int NPU_IR_PARAM_BYTES = 64;

struct npu_ir_tensor {
    int32_t role, dtype;
    int64_t ne[4];
    struct npu_constant_desc constant;
};

struct npu_ir_node {
    int32_t op, n_inputs;
    int32_t inputs[NPU_IR_MAX_INPUTS];
    int32_t output;
    uint8_t params[NPU_IR_PARAM_BYTES];
};

extern "C" {

// Compile a ggml-convention MatMul for the NPU: A[M,K] @ B[N,K]^T -> [M,N] (transpose_b),
// with BOTH operands as inputs (shape-generic; weights are not baked). On success, writes a
// malloc'd native blob to *blob / *len (free with ggml_npu_ov_free) and returns 0.
//
// Mixed precision: the activation A and output stay **f32** (so the backend binds ggml's own
// f32 tensor pointers with no copy), while the weight B is an **f16** input (half the bytes of
// f32, and the NPU's native math precision) — the backend dequantizes each weight once into an
// f16 shadow. A is converted to f16 in-graph so the MatMul runs in f16; the result converts back
// to f32. The NPU compiler accepts an f16 weight *input*, so this stays shape-generic (one blob
// per shape, any weights) rather than needing per-weight compiled Constants.
int ggml_npu_ov_compile_matmul(int32_t M, int32_t K, int32_t N, uint8_t** blob, size_t* len) {
    if (!blob || !len || M <= 0 || K <= 0 || N <= 0) {
        return -1;
    }
    try {
        auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
        auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{(size_t)N, (size_t)K});
        auto a16 = std::make_shared<ov::op::v0::Convert>(a, ov::element::f16);
        auto mm = std::make_shared<ov::op::v0::MatMul>(a16, b, /*transpose_a=*/false, /*transpose_b=*/true);
        auto out = std::make_shared<ov::op::v0::Convert>(mm, ov::element::f32);
        auto r = std::make_shared<ov::op::v0::Result>(out);
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

// Compile a ggml-convention MatMul with the **weight baked in as a compile-time Constant** — the
// path that gets the NPU's fused decompress-matmul (a runtime low-bit input is NOT fused; a baked
// Constant is, and runs faster than an f16 input at a fraction of the bytes).
//
// The activation A[M,K] stays an f32 **input** (bound zero-copy) and the output stays f32; only
// the weight is baked. `bits` selects the weight Constant:
//   16 -> f16 weight, no dequant       (weight = N*K f16 values, row-major [N,K])
//    8 -> i8 weight  + per-group f16 scales, dequant in-graph (Convert->Reshape->Multiply)
//    4 -> i4 weight (packed nibbles) + per-group f16 scales, same dequant
// For bits 8/4, `group` is the per-group length along K and `scales` holds N*(K/group) f16 values.
//
// The compiled native blob is cached on disk under `cache_key` (see cache_dir); a warm hit skips
// compilation entirely and just returns the stored bytes — the same NATIVE blob the Level Zero
// path loads via pfnCreate. On success writes a malloc'd blob to *blob/*len (free with
// ggml_npu_ov_free) and returns 0.
int ggml_npu_ov_compile_matmul_constant_v2(int32_t M, int32_t K, int32_t N,
                                            const struct npu_constant_desc* constant,
                                            const char* cache_key, uint8_t** blob, size_t* len) {
    if (!blob || !len || M <= 0 || K <= 0 || N <= 0 || !constant || !constant->values) {
        return -1;
    }
    *blob = nullptr;
    *len = 0;

    // Disk-cache lookup first: a hit is just a file read, no OpenVINO involvement.
    std::string path;
    if (cache_key && *cache_key) {
        std::string dir = cache_dir();
        if (!dir.empty()) {
            path = dir + "/" + cache_key + ".blob";
            if (read_blob(path, blob, len)) {
                return 0;
            }
        }
    }

    try {
        namespace v0 = ov::op::v0;
        namespace v1 = ov::op::v1;
        auto a = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
        auto a16 = std::make_shared<v0::Convert>(a, ov::element::f16);

        auto w = make_weight(*constant, N, K);
        if (!w) {
            return -1;
        }

        auto mm = std::make_shared<v0::MatMul>(a16, w, /*transpose_a=*/false, /*transpose_b=*/true);
        auto out = std::make_shared<v0::Convert>(mm, ov::element::f32);
        auto r = std::make_shared<v0::Result>(out);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{r}, ov::ParameterVector{a});

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
        if (!path.empty()) {
            write_blob(path, s.data(), s.size());
        }
        return 0;
    } catch (const std::exception&) {
        return -3;
    } catch (...) {
        return -4;
    }
}

// Compile a fused subgraph (op-DAG) as ONE ov::Model for the NPU. All boundary inputs are f32
// activations `[M, input_k[i]]` (Converted to f16 at the graph edge); every internal value is f16;
// the single output (`out_node`) is Converted back to f32. Weights are baked as Constants. The
// native blob is disk-cached under `cache_key` (see cache_dir). On success writes a malloc'd blob
// to *blob/*len (free with ggml_npu_ov_free) and returns 0.
int ggml_npu_ov_compile_subgraph_v2(int32_t M, int32_t n_inputs, const int32_t* input_k,
                                    int32_t n_nodes, const struct npu_sg_node* nodes,
                                    int32_t n_consts, const struct npu_sg_const* consts,
                                    const int32_t* out_nodes, int32_t n_outs, const char* cache_key,
                                    uint8_t** blob, size_t* len) {
    if (!blob || !len || M <= 0 || n_inputs <= 0 || n_nodes <= 0 || !input_k || !nodes) {
        return -1;
    }
    if (!out_nodes || n_outs <= 0) {
        return -1;
    }
    for (int i = 0; i < n_outs; i++) {
        if (out_nodes[i] < 0 || out_nodes[i] >= n_nodes) {
            return -1;
        }
    }
    *blob = nullptr;
    *len = 0;

    if (std::getenv("GGML_NPU_DEBUG")) {
        std::fprintf(stderr, "[sg] M=%d n_inputs=%d n_nodes=%d n_consts=%d n_outs=%d\n",
                     M, n_inputs, n_nodes, n_consts, n_outs);
        for (int i = 0; i < n_inputs; i++) std::fprintf(stderr, "[sg]  input[%d] k=%d\n", i, input_k[i]);
        for (int j = 0; j < n_nodes; j++)
            std::fprintf(stderr, "[sg]  node[%d] op=%d a=%d b=%d const=%d out_n=%d\n", j, nodes[j].op,
                         nodes[j].a, nodes[j].b, nodes[j].const_idx, nodes[j].out_n);
        for (int c = 0; c < n_consts; c++)
            std::fprintf(stderr, "[sg]  const[%d] bits=%d group=%d n=%d k=%d values=%zu exact=%d\n", c,
                         consts[c].constant.bit_width, consts[c].constant.group_elems,
                         consts[c].n, consts[c].k, consts[c].constant.values_len,
                         consts[c].constant.exactness);
        for (int i = 0; i < n_outs; i++) std::fprintf(stderr, "[sg]  out[%d] node=%d\n", i, out_nodes[i]);
    }

    std::string path;
    if (cache_key && *cache_key) {
        std::string dir = cache_dir();
        if (!dir.empty()) {
            path = dir + "/" + cache_key + ".blob";
            if (read_blob(path, blob, len)) {
                return 0;
            }
        }
    }

    try {
        namespace v0 = ov::op::v0;
        namespace v1 = ov::op::v1;

        // Parameters (one per boundary input), each Converted to f16 as its usable value. f16
        // interior is accurate: validated against a dequant reference to ~5e-4 (the fused chain
        // matches the CPU quantized kernel to ~1% with i8/f16 weights).
        ov::ParameterVector params;
        std::vector<std::shared_ptr<ov::Node>> input_val((size_t)n_inputs);
        for (int i = 0; i < n_inputs; i++) {
            if (input_k[i] <= 0) {
                return -1;
            }
            auto p = std::make_shared<v0::Parameter>(ov::element::f32,
                                                     ov::Shape{(size_t)M, (size_t)input_k[i]});
            params.push_back(p);
            input_val[i] = std::make_shared<v0::Convert>(p, ov::element::f16);
        }

        // Resolve a value ref to its f16 node: [0, n_inputs) = input, >= n_inputs = a prior node.
        // Shapes (and thus each MatMul's K) are inferred by OpenVINO from the operand nodes.
        std::vector<std::shared_ptr<ov::Node>> node_val((size_t)n_nodes);
        auto value = [&](int32_t r, std::shared_ptr<ov::Node>& out) -> bool {
            if (r >= 0 && r < n_inputs) { out = input_val[(size_t)r]; return true; }
            int32_t j = r - n_inputs;
            if (j >= 0 && j < n_nodes && node_val[(size_t)j]) { out = node_val[(size_t)j]; return true; }
            return false;
        };

        for (int j = 0; j < n_nodes; j++) {
            const npu_sg_node& nd = nodes[j];
            std::shared_ptr<ov::Node> a, b;
            switch (nd.op) {
                case NPU_SG_MATMUL: {
                    if (!value(nd.a, a) || nd.const_idx < 0 || nd.const_idx >= n_consts) return -1;
                    const npu_sg_const& c = consts[(size_t)nd.const_idx];
                    auto w = make_weight(c.constant, c.n, c.k);
                    if (!w) return -1;
                    node_val[(size_t)j] =
                        std::make_shared<v0::MatMul>(a, w, /*transpose_a=*/false, /*transpose_b=*/true);
                    break;
                }
                case NPU_SG_MUL: {
                    if (!value(nd.a, a) || !value(nd.b, b)) return -1;
                    node_val[(size_t)j] = std::make_shared<v1::Multiply>(a, b);
                    break;
                }
                case NPU_SG_ADD: {
                    if (!value(nd.a, a) || !value(nd.b, b)) return -1;
                    node_val[(size_t)j] = std::make_shared<v1::Add>(a, b);
                    break;
                }
                case NPU_SG_RMSNORM: {
                    // RMSNorm over the last dim (K): x / sqrt(mean(x²) + eps). Weight-multiply is a
                    // separate Mul node. eps arrives as f32 bits in out_n.
                    if (!value(nd.a, a)) return -1;
                    float eps;
                    std::memcpy(&eps, &nd.out_n, sizeof(float));
                    auto x2 = std::make_shared<v1::Multiply>(a, a);
                    auto axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
                    auto mean = std::make_shared<v1::ReduceMean>(x2, axis, /*keep_dims=*/true);
                    auto epsc = v0::Constant::create(ov::element::f16, ov::Shape{1}, std::vector<ov::float16>{ov::float16(eps)});
                    auto rms = std::make_shared<v0::Sqrt>(std::make_shared<v1::Add>(mean, epsc));
                    node_val[(size_t)j] = std::make_shared<v1::Divide>(a, rms);
                    break;
                }
                case NPU_SG_SILU: {
                    if (!value(nd.a, a)) return -1;
                    // SiLU(x) = x * sigmoid(x).
                    node_val[(size_t)j] =
                        std::make_shared<v1::Multiply>(a, std::make_shared<v0::Sigmoid>(a));
                    break;
                }
                default:
                    return -1;
            }
        }

        // One Result per requested output node, in the caller's order. This fixes the blob's output
        // argument order, which the Rust session binds `outs[i]` to in the same order.
        ov::ResultVector results;
        results.reserve((size_t)n_outs);
        for (int i = 0; i < n_outs; i++) {
            auto out = std::make_shared<v0::Convert>(node_val[(size_t)out_nodes[i]], ov::element::f32);
            results.push_back(std::make_shared<v0::Result>(out));
        }
        auto model = std::make_shared<ov::Model>(results, params);

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
        if (!path.empty()) {
            write_blob(path, s.data(), s.size());
        }
        return 0;
    } catch (const std::exception&) {
        return -3;
    } catch (...) {
        return -4;
    }
}

// Compile one Graph IR v2 scheduler split. The initial lowerer intentionally covers the dense and
// LFM ShortConv building blocks; unsupported opcodes fail capability probing in Rust and remain on
// CPU/GPU. Runtime tensors are f32 graph arguments with an f16 interior. Constants arrive already
// requantized and use the same fused low-bit decompression form as the mature matmul path.
int ggml_npu_ov_compile_graph_v3(int32_t n_tensors, const struct npu_ir_tensor* tensors,
                                 int32_t n_nodes, const struct npu_ir_node* nodes,
                                 int32_t n_inputs, const uint32_t* input_ids,
                                 int32_t n_outputs, const uint32_t* output_ids,
                                 const char* cache_key, uint8_t** blob, size_t* len) {
    if (!blob || !len || n_tensors <= 0 || !tensors || n_nodes <= 0 || !nodes ||
        n_inputs <= 0 || !input_ids || n_outputs <= 0 || !output_ids) {
        return -1;
    }
    *blob = nullptr;
    *len = 0;

    std::string path;
    if (cache_key && *cache_key) {
        std::string dir = cache_dir();
        if (!dir.empty()) {
            path = dir + "/" + cache_key + ".blob";
            const auto read_started = std::chrono::steady_clock::now();
            if (read_blob(path, blob, len)) {
                if (std::getenv("GGML_PERF_METRICS")) {
                    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - read_started);
                    std::fprintf(
                        stderr,
                        "[ggml-perf] {\"schema\":1,\"source\":\"ggml-npu-compiler\",\"event\":\"artifact_load\",\"persistent_key\":\"%s\",\"outcome\":\"disk_hit\",\"elapsed_ns\":%lld,\"elf_bytes\":%zu}\n",
                        cache_key, static_cast<long long>(elapsed.count()), *len);
                }
                if (std::getenv("GGML_NPU_DEBUG")) {
                    std::fprintf(stderr, "[ir2 cache] hit key=%s nodes=%d bytes=%zu\n",
                                 cache_key, n_nodes, *len);
                }
                return 0;
            }
        }
    }

    try {
        const auto compile_started = std::chrono::steady_clock::now();
        if (std::getenv("GGML_NPU_DEBUG")) {
            std::fprintf(stderr, "[ir2 compile] start key=%s nodes=%d tensors=%d\n",
                         cache_key ? cache_key : "(none)", n_nodes, n_tensors);
        }
        namespace v0 = ov::op::v0;
        namespace v1 = ov::op::v1;
        auto ov_shape = [&](int32_t id) -> ov::Shape {
            const auto& t = tensors[id];
            return {(size_t)t.ne[3], (size_t)t.ne[2], (size_t)t.ne[1], (size_t)t.ne[0]};
        };
        auto shape_i64 = [&](int32_t id) -> std::vector<int64_t> {
            const auto& t = tensors[id];
            return {t.ne[3], t.ne[2], t.ne[1], t.ne[0]};
        };
        auto elements = [&](int32_t id) -> int64_t {
            const auto& t = tensors[id];
            return t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
        };

        std::vector<ov::Output<ov::Node>> values((size_t)n_tensors);
        std::vector<bool> routed_weight((size_t)n_tensors, false);
        for (int32_t j = 0; j < n_nodes; ++j) {
            if (nodes[j].op == NPU_IR_MATMUL_ID && nodes[j].n_inputs == 3 &&
                nodes[j].inputs[0] >= 0 && nodes[j].inputs[0] < n_tensors) {
                routed_weight[(size_t)nodes[j].inputs[0]] = true;
            }
        }
        ov::ParameterVector params;
        params.reserve((size_t)n_inputs);
        for (int i = 0; i < n_inputs; ++i) {
            const int32_t id = (int32_t)input_ids[i];
            if (id < 0 || id >= n_tensors || tensors[id].role == NPU_IR_CONSTANT) return -1;
            const auto et = tensors[id].dtype == 1 ? ov::element::i32 :
                            tensors[id].dtype == 2 ? ov::element::i64 :
                            tensors[id].dtype == 3 ? ov::element::f16 : ov::element::f32;
            auto p = std::make_shared<v0::Parameter>(et, ov_shape(id));
            params.push_back(p);
            values[(size_t)id] = et == ov::element::f32
                ? ov::Output<ov::Node>(std::make_shared<v0::Convert>(p, ov::element::f16))
                : ov::Output<ov::Node>(p);
        }
        for (int32_t id = 0; id < n_tensors; ++id) {
            const auto& t = tensors[id];
            if (t.role != NPU_IR_CONSTANT) continue;
            if (!t.constant.values || t.ne[0] <= 0) return -1;
            // Routed banks are gathered while still compressed and dequantized only after the
            // runtime expert selection. Eagerly expanding the whole bank makes OpenVINO spend
            // minutes folding hundreds of MiB of unused experts.
            if (routed_weight[(size_t)id]) continue;
            const int64_t K = t.ne[0];
            const int64_t N = t.ne[1] * t.ne[2] * t.ne[3];
            auto w = make_weight(t.constant, N, K);
            if (!w) return -1;
            values[(size_t)id] = reshape_to(w, shape_i64(id));
        }

        auto get = [&](int32_t id, ov::Output<ov::Node>& out) -> bool {
            if (id < 0 || id >= n_tensors || !values[(size_t)id].get_node()) return false;
            out = values[(size_t)id];
            return true;
        };

        for (int32_t j = 0; j < n_nodes; ++j) {
            const auto& nd = nodes[j];
            if (nd.output < 0 || nd.output >= n_tensors || nd.n_inputs < 1 ||
                nd.n_inputs > NPU_IR_MAX_INPUTS) return -1;
            ov::Output<ov::Node> a, b, result;
            if (nd.op != NPU_IR_MATMUL_ID && !get(nd.inputs[0], a)) return -1;
            if (nd.n_inputs > 1 && !get(nd.inputs[1], b)) return -1;

            switch (nd.op) {
                case NPU_IR_MATMUL:
                    if (nd.n_inputs != 2) return -1;
                    result = std::make_shared<v0::MatMul>(b, a, false, true);
                    break;
                case NPU_IR_MUL:
                    if (nd.n_inputs != 2) return -1;
                    result = std::make_shared<v1::Multiply>(a, b);
                    break;
                case NPU_IR_ADD:
                    if (nd.n_inputs != 2) return -1;
                    result = std::make_shared<v1::Add>(a, b);
                    break;
                case NPU_IR_RMSNORM: {
                    float eps = 0.0f;
                    std::memcpy(&eps, nd.params, sizeof(eps));
                    auto x2 = std::make_shared<v1::Multiply>(a, a);
                    auto axis = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{3});
                    auto mean = std::make_shared<v1::ReduceMean>(x2, axis, true);
                    auto epsc = v0::Constant::create(ov::element::f16, {1},
                                                     std::vector<ov::float16>{ov::float16(eps)});
                    result = std::make_shared<v1::Divide>(
                        a, std::make_shared<v0::Sqrt>(std::make_shared<v1::Add>(mean, epsc)));
                    break;
                }
                case NPU_IR_SILU:
                    result = std::make_shared<v1::Multiply>(a, std::make_shared<v0::Sigmoid>(a));
                    break;
                case NPU_IR_RESHAPE:
                case NPU_IR_CONT:
                case NPU_IR_COPY:
                    result = reshape_to(a.get_node_shared_ptr(), shape_i64(nd.output));
                    break;
                case NPU_IR_VIEW: {
                    uint64_t byte_offset = 0;
                    std::memcpy(&byte_offset, nd.params, sizeof(byte_offset));
                    const int64_t begin = (int64_t)(byte_offset / 4); // LFM activation/state views are f32.
                    const int64_t end = begin + elements(nd.output);
                    auto flat = reshape_to(a.get_node_shared_ptr(), {-1});
                    auto starts = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{begin});
                    auto stops = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{end});
                    auto steps = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{1});
                    auto axes = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{0});
                    result = std::make_shared<ov::op::v8::Slice>(flat, starts, stops, steps, axes);
                    break;
                }
                case NPU_IR_TRANSPOSE: {
                    auto order = v0::Constant::create(ov::element::i64, {4},
                                                      std::vector<int64_t>{0, 1, 3, 2});
                    result = std::make_shared<v1::Transpose>(a, order);
                    break;
                }
                case NPU_IR_PERMUTE: {
                    int32_t axes[4] = {0, 1, 2, 3};
                    std::memcpy(axes, nd.params, sizeof(axes));
                    std::vector<int64_t> ov_axes(4);
                    for (int i = 0; i < 4; ++i) {
                        if (axes[3 - i] < 0 || axes[3 - i] > 3) return -1;
                        ov_axes[(size_t)i] = 3 - axes[3 - i];
                    }
                    auto order = v0::Constant::create(ov::element::i64, {4}, ov_axes);
                    result = std::make_shared<v1::Transpose>(a, order);
                    break;
                }
                case NPU_IR_CONCAT: {
                    if (nd.n_inputs != 2) return -1;
                    int32_t ggml_axis = 0;
                    std::memcpy(&ggml_axis, nd.params, sizeof(ggml_axis));
                    if (ggml_axis < 0 || ggml_axis > 3) return -1;
                    result = std::make_shared<v0::Concat>(ov::OutputVector{a, b}, 3 - ggml_axis);
                    break;
                }
                case NPU_IR_SSM_CONV: {
                    if (nd.n_inputs != 2) return -1;
                    const auto& sx = tensors[nd.inputs[0]];
                    const int64_t n_s = sx.ne[2], d_inner = sx.ne[1], ncs = sx.ne[0];
                    const int64_t d_conv = tensors[nd.inputs[1]].ne[0];
                    const int64_t n_t = ncs - d_conv + 1;
                    if (n_s <= 0 || d_inner <= 0 || n_t <= 0) return -1;
                    auto sx3 = reshape_to(a.get_node_shared_ptr(), {n_s, d_inner, ncs});
                    auto kernel = reshape_to(b.get_node_shared_ptr(), {d_inner, 1, 1, d_conv});
                    auto conv = std::make_shared<v1::GroupConvolution>(
                        sx3, kernel, ov::Strides{1}, ov::CoordinateDiff{0}, ov::CoordinateDiff{0}, ov::Strides{1});
                    auto order = v0::Constant::create(ov::element::i64, {3},
                                                      std::vector<int64_t>{0, 2, 1});
                    result = std::make_shared<v1::Transpose>(conv, order);
                    break;
                }
                case NPU_IR_ROPE: {
                    if (nd.n_inputs != 2) return -1;
                    int32_t n_dims = 0, mode = 0;
                    float freq_base = 0.0f, freq_scale = 0.0f, ext_factor = 0.0f, attn_factor = 0.0f;
                    std::memcpy(&n_dims, nd.params + 4, sizeof(n_dims));
                    std::memcpy(&mode, nd.params + 8, sizeof(mode));
                    std::memcpy(&freq_base, nd.params + 20, sizeof(freq_base));
                    std::memcpy(&freq_scale, nd.params + 24, sizeof(freq_scale));
                    std::memcpy(&ext_factor, nd.params + 28, sizeof(ext_factor));
                    std::memcpy(&attn_factor, nd.params + 32, sizeof(attn_factor));
                    const int64_t head_dim = tensors[nd.inputs[0]].ne[0];
                    if ((mode != 0 && mode != 2) || n_dims != head_dim || (n_dims & 1) ||
                        freq_base <= 0.0f || ext_factor != 0.0f) return -1;

                    std::vector<ov::float16> inv((size_t)n_dims / 2);
                    for (int32_t i = 0; i < n_dims / 2; ++i) {
                        inv[(size_t)i] = ov::float16(
                            std::pow(freq_base, -2.0f * (float)i / (float)n_dims) *
                            freq_scale * attn_factor);
                    }
                    ov::Output<ov::Node> positions = b;
                    if (positions.get_element_type() != ov::element::f16) {
                        positions = std::make_shared<v0::Convert>(positions, ov::element::f16);
                    }
                    positions = reshape_to(positions.get_node_shared_ptr(), {-1, 1});
                    auto inv_freq = v0::Constant::create(
                        ov::element::f16, {1, (size_t)n_dims / 2}, inv);
                    auto theta = std::make_shared<v1::Multiply>(positions, inv_freq);
                    auto cs_shape = v0::Constant::create(
                        ov::element::i64, {4}, std::vector<int64_t>{1, -1, 1, n_dims / 2});
                    ov::Output<ov::Node> cos_theta = std::make_shared<v1::Reshape>(
                        std::make_shared<v0::Cos>(theta), cs_shape, false);
                    ov::Output<ov::Node> sin_theta = std::make_shared<v1::Reshape>(
                        std::make_shared<v0::Sin>(theta), cs_shape, false);
                    auto split_axis = v0::Constant::create(ov::element::i64, {}, std::vector<int64_t>{-1});

                    if (mode == 0) {
                        auto paired_shape = v0::Constant::create(
                            ov::element::i64, {5}, std::vector<int64_t>{0, 0, 0, n_dims / 2, 2});
                        auto paired = std::make_shared<v1::Reshape>(a, paired_shape, true);
                        auto parts = std::make_shared<v1::Split>(paired, split_axis, 2);
                        auto x0 = parts->output(0);
                        auto x1 = parts->output(1);
                        auto last_axis = v0::Constant::create(
                            ov::element::i64, {1}, std::vector<int64_t>{-1});
                        auto cos_normal = std::make_shared<v0::Unsqueeze>(cos_theta, last_axis);
                        auto sin_normal = std::make_shared<v0::Unsqueeze>(sin_theta, last_axis);
                        auto y0 = std::make_shared<v1::Subtract>(
                            std::make_shared<v1::Multiply>(x0, cos_normal),
                            std::make_shared<v1::Multiply>(x1, sin_normal));
                        auto y1 = std::make_shared<v1::Add>(
                            std::make_shared<v1::Multiply>(x0, sin_normal),
                            std::make_shared<v1::Multiply>(x1, cos_normal));
                        auto interleaved = std::make_shared<v0::Concat>(ov::OutputVector{y0, y1}, -1);
                        result = reshape_to(interleaved, shape_i64(nd.output));
                    } else {
                        auto parts = std::make_shared<v1::Split>(a, split_axis, 2);
                        auto x0 = parts->output(0);
                        auto x1 = parts->output(1);
                        auto y0 = std::make_shared<v1::Subtract>(
                            std::make_shared<v1::Multiply>(x0, cos_theta),
                            std::make_shared<v1::Multiply>(x1, sin_theta));
                        auto y1 = std::make_shared<v1::Add>(
                            std::make_shared<v1::Multiply>(x0, sin_theta),
                            std::make_shared<v1::Multiply>(x1, cos_theta));
                        result = std::make_shared<v0::Concat>(ov::OutputVector{y0, y1}, -1);
                    }
                    break;
                }
                case NPU_IR_GET_ROWS: {
                    if (nd.n_inputs != 2) return -1;
                    if (b.get_element_type() != ov::element::i32 && b.get_element_type() != ov::element::i64) {
                        b = std::make_shared<v0::Convert>(b, ov::element::i32);
                    }
                    const auto& source = tensors[nd.inputs[0]];
                    const auto& output = tensors[nd.output];
                    if (source.ne[0] <= 0 || output.ne[0] != source.ne[0] ||
                        elements(nd.output) != elements(nd.inputs[1]) * source.ne[0]) {
                        return -1;
                    }
                    // ggml GET_ROWS selects independently for every outer index. Flattening all
                    // indices into an opset8 Gather cross-products the token dimension (e.g.
                    // 16 tokens x top-4 routes became 64 rows for each token). Expand each index
                    // across the row width and gather on the reversed-layout row axis instead.
                    auto index_shape = std::vector<int64_t>{
                        output.ne[3], output.ne[2], output.ne[1], 1};
                    auto per_element = reshape_output(b, index_shape);
                    auto repeats = v0::Constant::create(
                        ov::element::i64, {4},
                        std::vector<int64_t>{1, 1, 1, source.ne[0]});
                    per_element = std::make_shared<v0::Tile>(per_element, repeats);
                    result = std::make_shared<ov::op::v6::GatherElements>(a, per_element, 2);
                    break;
                }
                case NPU_IR_SET_ROWS: {
                    if (nd.n_inputs != 3) return -1;
                    ov::Output<ov::Node> base;
                    if (!get(nd.inputs[2], base)) return -1;
                    if (b.get_element_type() != ov::element::i32 && b.get_element_type() != ov::element::i64) {
                        b = std::make_shared<v0::Convert>(b, ov::element::i32);
                    }
                    b = reshape_to(b.get_node_shared_ptr(), {-1});
                    auto axis = v0::Constant::create(ov::element::i32, {}, std::vector<int32_t>{2});
                    result = std::make_shared<ov::op::v3::ScatterUpdate>(base, b, a, axis);
                    break;
                }
                case NPU_IR_SCALE: {
                    float scale = 0.0f;
                    std::memcpy(&scale, nd.params, sizeof(scale));
                    auto c = v0::Constant::create(ov::element::f16, {},
                                                  std::vector<ov::float16>{ov::float16(scale)});
                    result = std::make_shared<v1::Multiply>(a, c);
                    break;
                }
                case NPU_IR_DIV:
                    if (nd.n_inputs != 2) return -1;
                    result = std::make_shared<v1::Divide>(a, b);
                    break;
                case NPU_IR_CLAMP: {
                    float lo = 0.0f, hi = 0.0f;
                    std::memcpy(&lo, nd.params, sizeof(lo));
                    std::memcpy(&hi, nd.params + sizeof(lo), sizeof(hi));
                    // The NPU compiler's f16 Clamp parser converts infinity to zero. The graph
                    // interior is already f16, so finite f16 extrema are semantically equivalent.
                    if (!std::isfinite(lo)) lo = lo < 0 ? -65504.0f : 65504.0f;
                    if (!std::isfinite(hi)) hi = hi < 0 ? -65504.0f : 65504.0f;
                    result = std::make_shared<v0::Clamp>(a, lo, hi);
                    break;
                }
                case NPU_IR_SIGMOID:
                    result = std::make_shared<v0::Sigmoid>(a);
                    break;
                case NPU_IR_SOFTMAX: {
                    float scale = 1.0f;
                    std::memcpy(&scale, nd.params, sizeof(scale));
                    ov::Output<ov::Node> logits = a;
                    if (scale != 1.0f) {
                        auto c = v0::Constant::create(
                            ov::element::f16, {}, std::vector<ov::float16>{ov::float16(scale)});
                        logits = std::make_shared<v1::Multiply>(logits, c);
                    }
                    if (nd.n_inputs > 1) {
                        logits = std::make_shared<v1::Add>(logits, b);
                    }
                    result = std::make_shared<ov::op::v8::Softmax>(logits, 3);
                    break;
                }
                case NPU_IR_FLASH_ATTN: {
                    if (nd.n_inputs < 3 || nd.n_inputs > 4) return -1;
                    ov::Output<ov::Node> v;
                    if (!get(nd.inputs[2], v)) return -1;
                    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
                    std::memcpy(&scale, nd.params, sizeof(scale));
                    std::memcpy(&max_bias, nd.params + 4, sizeof(max_bias));
                    std::memcpy(&logit_softcap, nd.params + 8, sizeof(logit_softcap));
                    if (max_bias != 0.0f || logit_softcap != 0.0f) return -1;
                    const int64_t query_heads = tensors[nd.inputs[0]].ne[2];
                    const int64_t key_heads = tensors[nd.inputs[1]].ne[2];
                    const int64_t value_heads = tensors[nd.inputs[2]].ne[2];
                    if (query_heads <= 0 || key_heads <= 0 || value_heads <= 0 ||
                        query_heads % key_heads != 0 || query_heads % value_heads != 0) return -1;
                    if (query_heads != key_heads) {
                        auto grouped_shape = v0::Constant::create(
                            ov::element::i64, {5},
                            std::vector<int64_t>{
                                tensors[nd.inputs[1]].ne[3], key_heads, 1,
                                tensors[nd.inputs[1]].ne[1], tensors[nd.inputs[1]].ne[0]});
                        b = std::make_shared<v1::Reshape>(b, grouped_shape, false);
                        auto repeats = v0::Constant::create(
                            ov::element::i64, {5},
                            std::vector<int64_t>{1, 1, query_heads / key_heads, 1, 1});
                        b = std::make_shared<v0::Tile>(b, repeats);
                        b = reshape_to(b.get_node_shared_ptr(), {
                            tensors[nd.inputs[1]].ne[3], query_heads,
                            tensors[nd.inputs[1]].ne[1], tensors[nd.inputs[1]].ne[0]});
                    }
                    if (query_heads != value_heads) {
                        auto grouped_shape = v0::Constant::create(
                            ov::element::i64, {5},
                            std::vector<int64_t>{
                                tensors[nd.inputs[2]].ne[3], value_heads, 1,
                                tensors[nd.inputs[2]].ne[1], tensors[nd.inputs[2]].ne[0]});
                        v = std::make_shared<v1::Reshape>(v, grouped_shape, false);
                        auto repeats = v0::Constant::create(
                            ov::element::i64, {5},
                            std::vector<int64_t>{1, 1, query_heads / value_heads, 1, 1});
                        v = std::make_shared<v0::Tile>(v, repeats);
                        v = reshape_to(v.get_node_shared_ptr(), {
                            tensors[nd.inputs[2]].ne[3], query_heads,
                            tensors[nd.inputs[2]].ne[1], tensors[nd.inputs[2]].ne[0]});
                    }
                    ov::Output<ov::Node> logits = std::make_shared<v0::MatMul>(a, b, false, true);
                    if (scale != 1.0f) {
                        auto c = v0::Constant::create(
                            ov::element::f16, {}, std::vector<ov::float16>{ov::float16(scale)});
                        logits = std::make_shared<v1::Multiply>(logits, c);
                    }
                    if (nd.n_inputs == 4) {
                        ov::Output<ov::Node> mask;
                        if (!get(nd.inputs[3], mask)) return -1;
                        logits = std::make_shared<v1::Add>(logits, mask);
                    }
                    auto probs = std::make_shared<ov::op::v8::Softmax>(logits, 3);
                    const bool value_is_transposed =
                        tensors[nd.inputs[2]].ne[1] == tensors[nd.inputs[0]].ne[0];
                    result = std::make_shared<v0::MatMul>(probs, v, false, value_is_transposed);
                    break;
                }
                case NPU_IR_SUM_ROWS: {
                    auto axis = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{3});
                    result = std::make_shared<v1::ReduceSum>(a, axis, true);
                    break;
                }
                case NPU_IR_ARGSORT: {
                    const int64_t k_value = tensors[nd.output].ne[0];
                    auto k = v0::Constant::create(ov::element::i64, {}, std::vector<int64_t>{k_value});
                    auto topk = std::make_shared<ov::op::v3::TopK>(
                        a, k, 3, ov::op::v3::TopK::Mode::MAX, ov::op::v3::TopK::SortType::SORT_VALUES,
                        ov::element::i32);
                    result = topk->output(1);
                    break;
                }
                case NPU_IR_MATMUL_ID: {
                    if (nd.n_inputs != 3) return -1;
                    ov::Output<ov::Node> ids;
                    if (!get(nd.inputs[2], ids)) return -1;
                    if (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
                        ids = std::make_shared<v0::Convert>(ids, ov::element::i32);
                    }
                    const auto& wt = tensors[nd.inputs[0]];
                    const auto& constant = wt.constant;
                    const auto& it = tensors[nd.inputs[2]];
                    const int64_t K = wt.ne[0], N = wt.ne[1], E = wt.ne[2];
                    const int64_t U = it.ne[0], T = it.ne[1];
                    if (K <= 0 || N <= 0 || E <= 0 || U <= 0 || T <= 0 || !constant.values) {
                        return -1;
                    }
                    // Float f16 banks may be SourceBytes pass-through or legacy Requantized-tagged
                    // ABI (shim tests historically left exactness=0). Packed integer banks must be
                    // canonical requant (unpacked) — never packed source-native gather (#11).
                    const bool float16_bank =
                        constant.numeric_kind == NPU_NUMERIC_FLOAT && constant.bit_width == 16;
                    if (float16_bank) {
                        if (constant.exactness != NPU_REQUANTIZED
                            && constant.exactness != NPU_SOURCE_BYTES) {
                            return -1;
                        }
                    } else if (constant.exactness != NPU_REQUANTIZED) {
                        return -1;
                    }
                    auto ids_flat = reshape_output(ids, {T * U});
                    auto axis = v0::Constant::create(ov::element::i32, {}, std::vector<int32_t>{0});
                    ov::Output<ov::Node> selected;
                    if (float16_bank) {
                        auto bank = std::make_shared<v0::Constant>(
                            ov::element::f16, ov::Shape{(size_t)E, (size_t)N, (size_t)K},
                            (const void*)constant.values);
                        selected = std::make_shared<ov::op::v8::Gather>(bank, ids_flat, axis);
                        selected = reshape_to(
                            selected.get_node_shared_ptr(), {T, U, N, K});
                    } else {
                        if ((constant.bit_width != 4 && constant.bit_width != 8) ||
                            constant.storage_bits != 8 || constant.packing != NPU_UNPACKED ||
                            constant.group_elems <= 0 || K % constant.group_elems != 0 ||
                            constant.scale_encoding != NPU_SCALE_GROUP_F16 ||
                            !constant.scale_base) return -1;
                        const int64_t KG = K / constant.group_elems;
                        const ov::Shape wshape{
                            (size_t)E, (size_t)N, (size_t)KG, (size_t)constant.group_elems};
                        const ov::Shape gshape{(size_t)E, (size_t)N, (size_t)KG, 1};
                        std::shared_ptr<ov::Node> packed;
                        if (constant.bit_width == 4) {
                            if (constant.offset_encoding == NPU_OFFSET_ZERO_U8) {
                                // Dynamic Gather on sub-byte constants is not representable by the
                                // current NPU register ABI. Keep the host's already-unpacked nibbles
                                // as u8; selection still happens before f16 dequantization.
                                packed = std::make_shared<v0::Constant>(
                                    ov::element::u8, wshape, (const void*)constant.values);
                            } else {
                                packed = std::make_shared<v0::Constant>(
                                    ov::element::i8, wshape, (const void*)constant.values);
                            }
                        } else {
                            packed = std::make_shared<v0::Constant>(
                                constant.offset_encoding == NPU_OFFSET_ZERO_U8
                                    ? ov::element::u8 : ov::element::i8,
                                wshape, (const void*)constant.values);
                        }
                        auto gathered = std::make_shared<ov::op::v8::Gather>(packed, ids_flat, axis);
                        selected = std::make_shared<v0::Convert>(gathered, ov::element::f16);
                        if (constant.offset_encoding == NPU_OFFSET_ZERO_U8) {
                            if (!constant.offset_base) return -1;
                            auto zero = std::make_shared<v0::Constant>(
                                ov::element::u8, gshape, (const void*)constant.offset_base);
                            auto selected_zero =
                                std::make_shared<ov::op::v8::Gather>(zero, ids_flat, axis);
                            selected = std::make_shared<v1::Subtract>(
                                selected,
                                std::make_shared<v0::Convert>(
                                    selected_zero, ov::element::f16));
                        }
                        auto scale = std::make_shared<v0::Constant>(
                            ov::element::f16, gshape, (const void*)constant.scale_base);
                        auto selected_scale =
                            std::make_shared<ov::op::v8::Gather>(scale, ids_flat, axis);
                        selected =
                            std::make_shared<v1::Multiply>(selected, selected_scale);
                        selected = reshape_to(
                            selected.get_node_shared_ptr(), {T, U, N, K});
                    }
                    auto acts3 = reshape_to(b.get_node_shared_ptr(), {T, tensors[nd.inputs[1]].ne[1], K});
                    if (tensors[nd.inputs[1]].ne[1] == 1 && U > 1) {
                        auto shape = v0::Constant::create(ov::element::i64, {3},
                                                          std::vector<int64_t>{T, U, K});
                        acts3 = std::make_shared<ov::op::v3::Broadcast>(acts3, shape);
                    }
                    auto axis2 = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{2});
                    auto acts4 = std::make_shared<v0::Unsqueeze>(acts3, axis2);
                    result = std::make_shared<v0::MatMul>(acts4, selected, false, true);
                    break;
                }
                case NPU_IR_SWIGLU:
                    if (nd.n_inputs != 2) return -1;
                    result = std::make_shared<v1::Multiply>(
                        std::make_shared<v1::Multiply>(a, std::make_shared<v0::Sigmoid>(a)), b);
                    break;
                default:
                    return -1;
            }
            values[(size_t)nd.output] = reshape_output(result, shape_i64(nd.output));
        }

        ov::ResultVector results;
        results.reserve((size_t)n_outputs);
        for (int i = 0; i < n_outputs; ++i) {
            const int32_t id = (int32_t)output_ids[i];
            if (id < 0 || id >= n_tensors || !values[(size_t)id].get_node()) return -1;
            const auto et = tensors[id].dtype == 1 ? ov::element::i32 :
                            tensors[id].dtype == 2 ? ov::element::i64 :
                            tensors[id].dtype == 3 ? ov::element::f16 : ov::element::f32;
            ov::Output<ov::Node> out = values[(size_t)id];
            if (out.get_element_type() != et) {
                out = std::make_shared<v0::Convert>(out, et);
            }
            results.push_back(std::make_shared<v0::Result>(out));
        }
        auto model = std::make_shared<ov::Model>(results, params);
        auto compiled = std::find(routed_weight.begin(), routed_weight.end(), true) !=
                                routed_weight.end()
            ? core().compile_model(model, "NPU", ov::intel_npu::tiles(1))
            : core().compile_model(model, "NPU");
        std::ostringstream os(std::ios::binary);
        compiled.export_model(os);
        const std::string native = os.str();
        uint8_t* buf = static_cast<uint8_t*>(std::malloc(native.size()));
        if (!buf) return -2;
        std::memcpy(buf, native.data(), native.size());
        *blob = buf;
        *len = native.size();
        if (!path.empty()) write_blob(path, native.data(), native.size());
        if (std::getenv("GGML_NPU_DEBUG")) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - compile_started);
            std::fprintf(stderr, "[ir2 compile] done key=%s nodes=%d ms=%lld bytes=%zu\n",
                         cache_key ? cache_key : "(none)", n_nodes,
                         static_cast<long long>(elapsed.count()), native.size());
        }
        if (std::getenv("GGML_PERF_METRICS")) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - compile_started);
            std::fprintf(
                stderr,
                "[ggml-perf] {\"schema\":1,\"source\":\"ggml-npu-compiler\",\"event\":\"artifact_load\",\"persistent_key\":\"%s\",\"outcome\":\"compiled\",\"elapsed_ns\":%lld,\"elf_bytes\":%zu}\n",
                cache_key ? cache_key : "", static_cast<long long>(elapsed.count()), native.size());
        }
        return 0;
    } catch (const std::exception& e) {
        if (std::getenv("GGML_NPU_DEBUG")) std::fprintf(stderr, "[ir2] %s\n", e.what());
        return -3;
    } catch (...) {
        return -4;
    }
}

enum npu_constant_inspection {
    NPU_INSPECT_U4 = 1u << 0,
    NPU_INSPECT_I8 = 1u << 1,
    NPU_INSPECT_SCALE_MULTIPLY = 1u << 2,
    NPU_INSPECT_AFFINE_SUBTRACT = 1u << 3,
    NPU_INSPECT_F32_CONSTANT = 1u << 4,
};

// Test/diagnostic entry point: inspect the actual OpenVINO dequant graph before compilation.
int ggml_npu_ov_inspect_constant_v2(const struct npu_constant_desc* constant,
                                    int32_t N, int32_t K, uint32_t* flags) {
    if (!constant || !flags || N <= 0 || K <= 0) return -1;
    try {
        auto weight = make_weight(*constant, N, K);
        if (!weight) return -1;
        auto result = std::make_shared<ov::op::v0::Result>(weight);
        auto model = std::make_shared<ov::Model>(
            ov::ResultVector{result}, ov::ParameterVector{});
        uint32_t observed = 0;
        for (const auto& node : model->get_ordered_ops()) {
            if (const auto constant_node =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(node)) {
                const auto type = constant_node->get_element_type();
                if (type == ov::element::u4) observed |= NPU_INSPECT_U4;
                if (type == ov::element::i8) observed |= NPU_INSPECT_I8;
                if (type == ov::element::f32) observed |= NPU_INSPECT_F32_CONSTANT;
            }
            const std::string type_name = node->get_type_name();
            if (type_name == "Multiply") observed |= NPU_INSPECT_SCALE_MULTIPLY;
            if (type_name == "Subtract") observed |= NPU_INSPECT_AFFINE_SUBTRACT;
        }
        *flags = observed;
        return 0;
    } catch (...) {
        return -2;
    }
}

void ggml_npu_ov_free(uint8_t* blob) {
    std::free(blob);
}

}  // extern "C"
