// Runtime OpenVINO compile shim for the ggml NPU backend.
//
// A small C++ island that links OpenVINO and exposes a C ABI. The core backend is a no_std
// Rust cdylib with NO OpenVINO dependency; it `dlopen`s this shim on a cache miss to compile
// a native NPU blob for a given op shape at runtime. Only needed on Intel hosts — where the
// shim (or OpenVINO) is absent, the backend simply falls back to CPU.
//
// Build: shim/build.sh  ->  libggml_npu_ovc.so
#include <openvino/openvino.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/sqrt.hpp>

#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

// Write blob bytes atomically (temp + rename) so a concurrent/aborted run never loads a partial.
static void write_blob(const std::string& path, const void* data, size_t len) {
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

// Build the dequantized f16 weight node `[N, K]` for a baked Constant. bits 16 → f16 Constant (no
// dequant). bits 8/4 → grouped low-bit Constant `[N, K/group, group]` + per-group f16 scale, dequant
// in-graph. `zeros != nullptr` selects the **canonical asymmetric-unsigned** form the NPU compiler
// keeps compressed (u4/u8 stays low-bit in device memory, dequant fused in-kernel):
//   `(Convert(u4 weight) - Convert(u4 zero)) * f16 scale`.
// `zeros == nullptr` is the symmetric signed form (i4/i8, no zero-point). The weight/zeros arrive
// **unpacked** (one value per byte); `Constant::create` packs the sub-byte types. Returns nullptr on
// a bad request. Shared by the single-matmul and fused-subgraph compile paths.
static std::shared_ptr<ov::Node> make_weight(int32_t bits, int32_t group, int64_t N, int64_t K,
                                             const uint8_t* weight, const uint8_t* scales,
                                             const uint8_t* zeros) {
    namespace v0 = ov::op::v0;
    namespace v1 = ov::op::v1;
    if (bits == 16) {
        return std::make_shared<v0::Constant>(ov::element::f16, ov::Shape{(size_t)N, (size_t)K},
                                              (const void*)weight);
    }
    if (!scales || group <= 0 || K % group != 0) {
        return nullptr;
    }
    const int64_t KG = K / group;
    const bool asym = (zeros != nullptr);
    const ov::Shape wshape{(size_t)N, (size_t)KG, (size_t)group};
    const ov::Shape gshape{(size_t)N, (size_t)KG, 1};

    if (std::getenv("GGML_NPU_DEBUG")) {
        std::fprintf(stderr, "[mw] bits=%d group=%d N=%ld K=%ld asym=%d  w[0..7]=", bits, group, (long)N, (long)K, asym);
        for (int i = 0; i < 8 && i < N * K; i++) std::fprintf(stderr, "%u ", weight[i]);
        std::fprintf(stderr, " zp[0..3]=");
        for (int i = 0; zeros && i < 4; i++) std::fprintf(stderr, "%u ", zeros[i]);
        const uint16_t* sh = (const uint16_t*)scales;
        std::fprintf(stderr, " sc[0..3]=");
        for (int i = 0; i < 4; i++) std::fprintf(stderr, "%.4f ", (float)*(const ov::float16*)&sh[i]);
        std::fprintf(stderr, "\n");
    }

    // Weight Constant (unsigned when asymmetric, signed otherwise). int4/int8 arrive unpacked.
    std::shared_ptr<ov::Node> bc;
    if (bits == 4) {
        if (asym) {
            std::vector<uint8_t> v(weight, weight + (size_t)N * K);
            bc = v0::Constant::create(ov::element::u4, wshape, v);
        } else {
            std::vector<int8_t> v(reinterpret_cast<const int8_t*>(weight),
                                  reinterpret_cast<const int8_t*>(weight) + (size_t)N * K);
            bc = v0::Constant::create(ov::element::i4, wshape, v);
        }
    } else {
        bc = std::make_shared<v0::Constant>(asym ? ov::element::u8 : ov::element::i8, wshape,
                                            (const void*)weight);
    }
    std::shared_ptr<ov::Node> deq = std::make_shared<v0::Convert>(bc, ov::element::f16);

    if (asym) {
        std::shared_ptr<ov::Node> zc;
        if (bits == 4) {
            std::vector<uint8_t> z(zeros, zeros + (size_t)N * KG);
            zc = v0::Constant::create(ov::element::u4, gshape, z);
        } else {
            zc = std::make_shared<v0::Constant>(ov::element::u8, gshape, (const void*)zeros);
        }
        auto zcf = std::make_shared<v0::Convert>(zc, ov::element::f16);
        deq = std::make_shared<v1::Subtract>(deq, zcf);
    }
    auto sc = std::make_shared<v0::Constant>(ov::element::f16, gshape, (const void*)scales);
    return reshape_to(std::make_shared<v1::Multiply>(deq, sc), {N, K});
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
    int32_t bits, group, n, k;
    const uint8_t* weight;
    size_t weight_len;
    const uint8_t* scales;
    size_t scales_len;
    const uint8_t* zeros;  // per-group zero-points (asymmetric); null/empty → symmetric
    size_t zeros_len;
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
int ggml_npu_ov_compile_matmul_const(int32_t M, int32_t K, int32_t N, int32_t bits, int32_t group,
                                     const uint8_t* weight, size_t weight_len,
                                     const uint8_t* scales, size_t scales_len,
                                     const uint8_t* zeros, size_t zeros_len,
                                     const char* cache_key, uint8_t** blob, size_t* len) {
    (void)weight_len;
    (void)scales_len;
    if (!blob || !len || M <= 0 || K <= 0 || N <= 0 || !weight) {
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

        auto w = make_weight(bits, group, N, K, weight, scales, (zeros && zeros_len) ? zeros : nullptr);
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
int ggml_npu_ov_compile_subgraph(int32_t M, int32_t n_inputs, const int32_t* input_k,
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
            std::fprintf(stderr, "[sg]  const[%d] bits=%d group=%d n=%d k=%d wlen=%zu slen=%zu\n", c,
                         consts[c].bits, consts[c].group, consts[c].n, consts[c].k,
                         consts[c].weight_len, consts[c].scales_len);
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
                    auto w = make_weight(c.bits, c.group, c.n, c.k, c.weight, c.scales,
                                         (c.zeros && c.zeros_len) ? c.zeros : nullptr);
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

void ggml_npu_ov_free(uint8_t* blob) {
    std::free(blob);
}

}  // extern "C"
