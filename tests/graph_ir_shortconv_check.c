/* Tiny end-to-end Graph IR v2 ShortConv check:
 * Graph IR ABI -> OpenVINO compile -> native blob -> Rust Level Zero submit -> CPU comparison.
 */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

/* Symbols required by the backend under RTLD_NOW; this check never calls these ggml helpers. */
ggml_backend_buffer_t ggml_backend_buffer_init(ggml_backend_buffer_type_t b,
        struct ggml_backend_buffer_i i, void * c, size_t s) {
    (void)b; (void)i; (void)c; (void)s; return NULL;
}
int ggml_graph_n_nodes(struct ggml_cgraph * g) { (void)g; return 0; }
struct ggml_tensor * ggml_graph_node(struct ggml_cgraph * g, int i) { (void)g; (void)i; return NULL; }
bool ggml_is_contiguous(const struct ggml_tensor * t) { (void)t; return true; }
bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t b) { (void)b; return false; }
const struct ggml_type_traits * ggml_get_type_traits(enum ggml_type t) { (void)t; return NULL; }

enum { IR_INPUT = 0, IR_CONSTANT = 1, IR_STATE = 2, IR_INTERMEDIATE = 3, IR_OUTPUT = 4,
       IR_SSM_CONV = 12, IR_ROPE = 13, IR_SOFTMAX = 14, IR_GET_ROWS = 15, IR_ARGSORT = 16, IR_MATMUL_ID = 17,
       IR_FLASH_ATTN = 22, IR_SIGMOID = 24, IR_SUM_ROWS = 25, IR_SET_ROWS = 26,
       IR_MAX_INPUTS = 16 };
/* Matches shim `npu_constant_desc` / Rust `ConstantAbi` for compile_graph_v3. */
struct ir_constant {
    int32_t numeric_kind, bit_width, storage_bits, signedness;
    int32_t block_elems, group_elems, packing;
    int32_t scale_encoding, offset_encoding, exactness;
    const uint8_t * values; size_t values_len;
    const uint8_t * scale_base; size_t scale_base_len;
    const uint8_t * scale_factor; size_t scale_factor_len;
    const uint8_t * offset_base; size_t offset_base_len;
    const uint8_t * offset_factor; size_t offset_factor_len;
};
struct ir_tensor {
    int32_t role, dtype;
    int64_t ne[4];
    struct ir_constant constant;
};
struct ir_node {
    int32_t op, n_inputs, inputs[IR_MAX_INPUTS], output;
    uint8_t params[64];
};
static struct ir_constant ir_f16_constant(const uint16_t * values, size_t n_values) {
    struct ir_constant c;
    memset(&c, 0, sizeof(c));
    c.numeric_kind = 0; /* NPU_NUMERIC_FLOAT */
    c.bit_width = 16;
    c.storage_bits = 16;
    /* exactness left 0 (NPU_REQUANTIZED) for ShortConv/legacy ABI; MoE float banks may
     * override to NPU_SOURCE_BYTES (=2) to exercise the gatherable pass-through path (#11). */
    c.values = (const uint8_t *)values;
    c.values_len = n_values * sizeof(uint16_t);
    return c;
}
typedef int (*compile_fn)(int32_t, const struct ir_tensor *, int32_t, const struct ir_node *,
        int32_t, const uint32_t *, int32_t, const uint32_t *, const char *, uint8_t **, size_t *);
typedef void (*free_fn)(uint8_t *);
typedef int (*run_fn)(const uint8_t *, size_t, const float *, size_t, float *, size_t);
typedef int (*run_multi_fn)(const uint8_t *, size_t, const void * const *, const size_t *,
        size_t, float *, size_t);

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s libggml_npu_ovc.so libggml_npu.so\n", argv[0]);
        return 2;
    }
    void * shim = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!shim) {
        fprintf(stderr, "shim dlopen: %s\n", dlerror());
        return 1;
    }
    void * backend = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!backend) {
        fprintf(stderr, "backend dlopen: %s\n", dlerror());
        return 1;
    }
    compile_fn compile = (compile_fn)dlsym(shim, "ggml_npu_ov_compile_graph_v3");
    free_fn free_blob = (free_fn)dlsym(shim, "ggml_npu_ov_free");
    run_fn run = (run_fn)dlsym(backend, "ggml_npu_ivpu_selftest");
    run_multi_fn run_multi = (run_multi_fn)dlsym(backend, "ggml_npu_ivpu_selftest_multi");
    if (!compile || !free_blob || !run || !run_multi) {
        fprintf(stderr, "missing graph compile/selftest symbol\n");
        return 1;
    }

    /* f16 kernels: [1,1,1] and [0.5,0,-0.5], channel-major. */
    const uint16_t kernels[] = {0x3c00, 0x3c00, 0x3c00, 0x3800, 0x0000, 0xb800};
    struct ir_tensor tensors[3] = {
        {.role=IR_INPUT, .dtype=0, .ne={4,2,1,1}},
        {.role=IR_CONSTANT, .dtype=0, .ne={3,2,1,1},
         .constant=ir_f16_constant(kernels, 6)},
        {.role=IR_OUTPUT, .dtype=0, .ne={2,2,1,1}},
    };
    struct ir_node node = {.op=IR_SSM_CONV, .n_inputs=2, .inputs={0,1}, .output=2};
    uint32_t inputs[] = {0}, outputs[] = {2};
    uint8_t * blob = NULL;
    size_t blob_len = 0;
    int rc = compile(3, tensors, 1, &node, 1, inputs, 1, outputs,
                     "ir3-shortconv-e2e-v1", &blob, &blob_len);
    if (rc != 0 || !blob || !blob_len) {
        fprintf(stderr, "compile failed: %d\n", rc);
        return 1;
    }

    const float x[] = {1,2,3,4, 10,20,30,40};
    const float expected[] = {6,-10, 9,-10};
    float actual[4] = {0};
    rc = run(blob, blob_len, x, 8, actual, 4);
    free_blob(blob);
    if (rc != 4) {
        fprintf(stderr, "NPU run failed: %d\n", rc);
        return 1;
    }
    for (int i = 0; i < 4; ++i) {
        if (fabsf(actual[i] - expected[i]) > 0.05f) {
            fprintf(stderr, "mismatch[%d]: got %.5f expected %.5f\n", i, actual[i], expected[i]);
            return 1;
        }
    }

    /* Batched GET_ROWS must apply each token's indices to that token only. Flattening the index
     * tensor creates a token-by-all-indices cross product and was the first full-model direct
     * runtime boundary after padded ELF output sections were handled correctly. */
    struct ir_tensor gather_tensors[3] = {
        {.role=IR_INPUT, .dtype=0, .ne={1,4,2,1}},
        {.role=IR_INPUT, .dtype=1, .ne={2,2,1,1}},
        {.role=IR_OUTPUT, .dtype=0, .ne={1,2,2,1}},
    };
    struct ir_node gather_node = {
        .op=IR_GET_ROWS, .n_inputs=2, .inputs={0,1}, .output=2
    };
    uint32_t gather_inputs[] = {0,1}, gather_outputs[] = {2};
    blob = NULL; blob_len = 0;
    rc = compile(3, gather_tensors, 1, &gather_node, 2, gather_inputs, 1, gather_outputs,
                 "ir3-batched-get-rows-e2e-v1", &blob, &blob_len);
    if (rc != 0 || !blob || !blob_len) {
        fprintf(stderr, "batched GetRows compile failed: %d\n", rc);
        return 1;
    }
    const float gather_source[] = {10,20,30,40, 50,60,70,80};
    const int32_t gather_indices[] = {3,1, 0,2};
    const void * gather_args[] = {gather_source, gather_indices};
    const size_t gather_lens[] = {8,4};
    const float gather_expected[] = {40,20, 50,70};
    float gather_actual[4] = {0};
    rc = run_multi(blob, blob_len, gather_args, gather_lens, 2, gather_actual, 4);
    free_blob(blob);
    if (rc != 4) {
        fprintf(stderr, "batched GetRows NPU run failed: %d\n", rc);
        return 1;
    }
    for (int i = 0; i < 4; ++i) {
        if (fabsf(gather_actual[i] - gather_expected[i]) > 0.05f) {
            fprintf(stderr, "batched GetRows mismatch[%d]: got %.5f expected %.5f\n",
                    i, gather_actual[i], gather_expected[i]);
            return 1;
        }
    }

    /* Compiler spike for the alternative selected by the MoE feasibility work: one baked expert
     * bank and runtime top-k indices. This verifies Gather-before-MatMul lowering without retaining
     * per-expert programs; model-scale bandwidth/energy remains an explicit benchmark concern. */
    uint16_t experts[4 * 2 * 4];
    const uint16_t expert_value[] = {0x3c00, 0x4000, 0x4200, 0x4400};
    for (int e = 0; e < 4; ++e) {
        for (int i = 0; i < 8; ++i) experts[e*8 + i] = expert_value[e];
    }
    struct ir_tensor moe_tensors[5] = {
        {.role=IR_INPUT, .dtype=0, .ne={4,1,1,1}},
        {.role=IR_CONSTANT, .dtype=0, .ne={4,2,4,1},
         .constant=ir_f16_constant(experts, 4 * 2 * 4)},
        {.role=IR_INPUT, .dtype=0, .ne={4,1,1,1}},
        {.role=IR_INTERMEDIATE, .dtype=1, .ne={2,1,1,1}},
        {.role=IR_OUTPUT, .dtype=0, .ne={2,2,1,1}},
    };
    moe_tensors[1].constant.exactness = 2; /* NPU_SOURCE_BYTES float expert bank */
    struct ir_node moe_nodes[2] = {
        {.op=IR_ARGSORT, .n_inputs=1, .inputs={0}, .output=3},
        {.op=IR_MATMUL_ID, .n_inputs=3, .inputs={1,2,3}, .output=4},
    };
    uint32_t moe_inputs[] = {0,2}, moe_outputs[] = {4};
    blob = NULL; blob_len = 0;
    rc = compile(5, moe_tensors, 2, moe_nodes, 2, moe_inputs, 1, moe_outputs,
                 "ir3-moe-routing-e2e-v2", &blob, &blob_len);
    if (rc != 0 || !blob || !blob_len) {
        fprintf(stderr, "MoE Gather/MatMul compile failed: %d\n", rc);
        return 1;
    }
    const float moe_logits[] = {0,2,3,1};
    const float moe_acts[] = {1,1,1,1};
    const void * moe_args[] = {moe_logits, moe_acts};
    const size_t moe_lens[] = {4,4};
    const float moe_expected[] = {12,12,8,8};
    float moe_actual[4] = {0};
    rc = run_multi(blob, blob_len, moe_args, moe_lens, 2, moe_actual, 4);
    if (rc != 4) {
        fprintf(stderr, "MoE NPU run failed: %d\n", rc);
        return 1;
    }
    for (int i = 0; i < 4; ++i) {
        if (fabsf(moe_actual[i] - moe_expected[i]) > 0.05f) {
            fprintf(stderr, "MoE mismatch[%d]: got %.5f expected %.5f\n",
                    i, moe_actual[i], moe_expected[i]);
            return 1;
        }
    }
    free_blob(blob);

    /* MoE gating + reduction glue used beside TopK/MatMulId (#11). */
    {
        struct ir_tensor sig_tensors[2] = {
            {.role=IR_INPUT, .dtype=0, .ne={4,1,1,1}},
            {.role=IR_OUTPUT, .dtype=0, .ne={4,1,1,1}},
        };
        struct ir_node sig_node = {.op=IR_SIGMOID, .n_inputs=1, .inputs={0}, .output=1};
        uint32_t sig_inputs[] = {0}, sig_outputs[] = {1};
        blob = NULL; blob_len = 0;
        rc = compile(2, sig_tensors, 1, &sig_node, 1, sig_inputs, 1, sig_outputs,
                     "ir3-moe-sigmoid-v1", &blob, &blob_len);
        if (rc != 0 || !blob || !blob_len) {
            fprintf(stderr, "MoE Sigmoid compile failed: %d\n", rc);
            return 1;
        }
        const float sig_in[] = {0.f, 2.f, -2.f, 4.f};
        /* logistic(x) */
        const float sig_expected[] = {0.5f, 0.880797f, 0.119203f, 0.982014f};
        float sig_actual[4] = {0};
        const void * sig_args[] = {sig_in};
        const size_t sig_lens[] = {4};
        rc = run_multi(blob, blob_len, sig_args, sig_lens, 1, sig_actual, 4);
        if (rc != 4) {
            fprintf(stderr, "MoE Sigmoid NPU run failed: %d\n", rc);
            return 1;
        }
        for (int i = 0; i < 4; ++i) {
            if (fabsf(sig_actual[i] - sig_expected[i]) > 0.05f) {
                fprintf(stderr, "Sigmoid mismatch[%d]: got %.5f expected %.5f\n",
                        i, sig_actual[i], sig_expected[i]);
                return 1;
            }
        }
        free_blob(blob);

        /* SumRows reduces ggml ne[0] (OV axis 3). Input [4,2,1,1] → output [1,2,1,1]. */
        struct ir_tensor sum_tensors[2] = {
            {.role=IR_INPUT, .dtype=0, .ne={4,2,1,1}},
            {.role=IR_OUTPUT, .dtype=0, .ne={1,2,1,1}},
        };
        struct ir_node sum_node = {.op=IR_SUM_ROWS, .n_inputs=1, .inputs={0}, .output=1};
        uint32_t sum_inputs[] = {0}, sum_outputs[] = {1};
        blob = NULL; blob_len = 0;
        rc = compile(2, sum_tensors, 1, &sum_node, 1, sum_inputs, 1, sum_outputs,
                     "ir3-moe-sum-rows-v1", &blob, &blob_len);
        if (rc != 0 || !blob || !blob_len) {
            fprintf(stderr, "MoE SumRows compile failed: %d\n", rc);
            return 1;
        }
        const float sum_in[] = {1,2,3,4, 10,20,30,40};
        const float sum_expected[] = {10, 100};
        float sum_actual[2] = {0};
        const void * sum_args[] = {sum_in};
        const size_t sum_lens[] = {8};
        rc = run_multi(blob, blob_len, sum_args, sum_lens, 1, sum_actual, 2);
        if (rc != 2) {
            fprintf(stderr, "MoE SumRows NPU run failed: %d\n", rc);
            return 1;
        }
        for (int i = 0; i < 2; ++i) {
            if (fabsf(sum_actual[i] - sum_expected[i]) > 0.05f) {
                fprintf(stderr, "SumRows mismatch[%d]: got %.5f expected %.5f\n",
                        i, sum_actual[i], sum_expected[i]);
                return 1;
            }
        }
        free_blob(blob);
    }

    /* Mutable KV/state write. Execute the same artifact twice, carrying the first output into the
     * second invocation, to verify recurrent state continuity without unsafe in-place aliasing. */
    struct ir_tensor state_tensors[4] = {
        {.role=IR_INPUT, .dtype=0, .ne={2,2,1,1}},
        {.role=IR_INPUT, .dtype=2, .ne={2,1,1,1}},
        {.role=IR_STATE, .dtype=0, .ne={2,4,1,1}},
        {.role=IR_STATE, .dtype=0, .ne={2,4,1,1}},
    };
    struct ir_node state_node = {.op=IR_SET_ROWS, .n_inputs=3, .inputs={0,1,2}, .output=3};
    uint32_t state_inputs[] = {0,1,2}, state_outputs[] = {3};
    blob = NULL; blob_len = 0;
    rc = compile(4, state_tensors, 1, &state_node, 3, state_inputs, 1, state_outputs,
                 "ir3-set-rows-state-v1", &blob, &blob_len);
    if (rc != 0 || !blob || !blob_len) {
        fprintf(stderr, "mutable SetRows compile failed: %d\n", rc);
        return 1;
    }
    const float updates[] = {10,11,20,21};
    const int64_t indices[] = {1,3};
    const float state[] = {1,2,3,4,5,6,7,8};
    const float state_expected[] = {1,2,10,11,5,6,20,21};
    const void * state_args[] = {updates, indices, state};
    const size_t state_lens[] = {4,4,8};
    float state_actual[8] = {0};
    rc = run_multi(blob, blob_len, state_args, state_lens, 3, state_actual, 8);
    if (rc != 8) {
        fprintf(stderr, "mutable SetRows NPU run failed: %d\n", rc);
        return 1;
    }
    for (int i = 0; i < 8; ++i) {
        if (fabsf(state_actual[i] - state_expected[i]) > 0.05f) {
            fprintf(stderr, "SetRows mismatch[%d]: got %.5f expected %.5f\n",
                    i, state_actual[i], state_expected[i]);
            return 1;
        }
    }
    const float updates_next[] = {30,31,40,41};
    const int64_t indices_next[] = {0,2};
    const float state_expected_next[] = {30,31,10,11,40,41,20,21};
    const void * state_args_next[] = {updates_next, indices_next, state_actual};
    float state_actual_next[8] = {0};
    rc = run_multi(blob, blob_len, state_args_next, state_lens, 3, state_actual_next, 8);
    if (rc != 8) {
        fprintf(stderr, "mutable SetRows continuity run failed: %d\n", rc);
        return 1;
    }
    for (int i = 0; i < 8; ++i) {
        if (fabsf(state_actual_next[i] - state_expected_next[i]) > 0.05f) {
            fprintf(stderr, "SetRows continuity mismatch[%d]: got %.5f expected %.5f\n",
                    i, state_actual_next[i], state_expected_next[i]);
            return 1;
        }
    }
    free_blob(blob);

    /* Mutable-state gather → write round-trip (GET_ROWS of state, SET_ROWS of gathered rows). */
    {
        struct ir_tensor gw_tensors[5] = {
            {.role=IR_STATE, .dtype=0, .ne={2,4,1,1}},
            {.role=IR_INPUT, .dtype=1, .ne={2,1,1,1}},
            {.role=IR_INTERMEDIATE, .dtype=0, .ne={2,2,1,1}},
            {.role=IR_INPUT, .dtype=2, .ne={2,1,1,1}},
            {.role=IR_STATE, .dtype=0, .ne={2,4,1,1}},
        };
        struct ir_node gw_nodes[2] = {
            {.op=IR_GET_ROWS, .n_inputs=2, .inputs={0,1}, .output=2},
            {.op=IR_SET_ROWS, .n_inputs=3, .inputs={2,3,0}, .output=4},
        };
        uint32_t gw_inputs[] = {0,1,3}, gw_outputs[] = {4};
        blob = NULL; blob_len = 0;
        rc = compile(5, gw_tensors, 2, gw_nodes, 3, gw_inputs, 1, gw_outputs,
                     "ir3-state-gather-write-v1", &blob, &blob_len);
        if (rc != 0 || !blob || !blob_len) {
            fprintf(stderr, "state gather/write compile failed: %d\n", rc);
            return 1;
        }
        const float gw_state[] = {1,2,3,4,5,6,7,8};
        const int32_t gw_gidx[] = {3,1};
        const int64_t gw_widx[] = {0,2};
        /* Gather rows 3 and 1 → {7,8,3,4}; write into slots 0 and 2. */
        const float gw_expected[] = {7,8,3,4,3,4,7,8};
        const void * gw_args[] = {gw_state, gw_gidx, gw_widx};
        const size_t gw_lens[] = {8,2,4};
        float gw_actual[8] = {0};
        rc = run_multi(blob, blob_len, gw_args, gw_lens, 3, gw_actual, 8);
        if (rc != 8) {
            fprintf(stderr, "state gather/write NPU run failed: %d\n", rc);
            return 1;
        }
        for (int i = 0; i < 8; ++i) {
            if (fabsf(gw_actual[i] - gw_expected[i]) > 0.05f) {
                fprintf(stderr, "state gather/write mismatch[%d]: got %.5f expected %.5f\n",
                        i, gw_actual[i], gw_expected[i]);
                return 1;
            }
        }
        free_blob(blob);
    }

    /* Adjacent SSM_CONV → SetRows writeback. Intel NPU rejects fusing SSM_CONV+SET_ROWS into one
     * OV model; the intended claimed path is adjacent programs (FeasibilityLimit peels) (#9).
     * Gather is covered by the state gather/write fixture above; this chains conv→write. */
    {
        const uint16_t adj_kernels[] = {0x3c00, 0x3c00, 0x3c00, 0x3800, 0x0000, 0xb800};
        struct ir_tensor conv_tensors[3] = {
            {.role=IR_INPUT, .dtype=0, .ne={4,2,1,1}},
            {.role=IR_CONSTANT, .dtype=0, .ne={3,2,1,1},
             .constant=ir_f16_constant(adj_kernels, 6)},
            {.role=IR_OUTPUT, .dtype=0, .ne={2,2,1,1}},
        };
        struct ir_node conv_node = {.op=IR_SSM_CONV, .n_inputs=2, .inputs={0,1}, .output=2};
        uint32_t conv_inputs[] = {0}, conv_outputs[] = {2};
        blob = NULL; blob_len = 0;
        rc = compile(3, conv_tensors, 1, &conv_node, 1, conv_inputs, 1, conv_outputs,
                     "ir3-shortconv-adjacent-conv-v1", &blob, &blob_len);
        if (rc != 0 || !blob || !blob_len) {
            fprintf(stderr, "adjacent SSM_CONV compile failed: %d\n", rc);
            return 1;
        }
        const float conv_x[] = {1,2,3,4, 10,20,30,40};
        const float conv_expected[] = {6,-10, 9,-10};
        float conv_actual[4] = {0};
        rc = run(blob, blob_len, conv_x, 8, conv_actual, 4);
        free_blob(blob);
        if (rc != 4) {
            fprintf(stderr, "adjacent SSM_CONV NPU run failed: %d\n", rc);
            return 1;
        }
        for (int i = 0; i < 4; ++i) {
            if (fabsf(conv_actual[i] - conv_expected[i]) > 0.05f) {
                fprintf(stderr, "adjacent SSM_CONV mismatch[%d]: got %.5f expected %.5f\n",
                        i, conv_actual[i], conv_expected[i]);
                return 1;
            }
        }

        struct ir_tensor wr_tensors[4] = {
            {.role=IR_INPUT, .dtype=0, .ne={2,2,1,1}},
            {.role=IR_INPUT, .dtype=2, .ne={2,1,1,1}},
            {.role=IR_STATE, .dtype=0, .ne={2,4,1,1}},
            {.role=IR_STATE, .dtype=0, .ne={2,4,1,1}},
        };
        struct ir_node wr_node = {.op=IR_SET_ROWS, .n_inputs=3, .inputs={0,1,2}, .output=3};
        uint32_t wr_inputs[] = {0,1,2}, wr_outputs[] = {3};
        blob = NULL; blob_len = 0;
        rc = compile(4, wr_tensors, 1, &wr_node, 3, wr_inputs, 1, wr_outputs,
                     "ir3-shortconv-adjacent-write-v1", &blob, &blob_len);
        if (rc != 0 || !blob || !blob_len) {
            fprintf(stderr, "adjacent SetRows writeback compile failed: %d\n", rc);
            return 1;
        }
        const int64_t wr_idx[] = {1,3};
        const float wr_state[] = {1,2,3,4,5,6,7,8};
        /* Write conv outputs {6,-10} / {9,-10} into state rows 1 and 3. */
        const float wr_expected[] = {1,2, 6,-10, 5,6, 9,-10};
        const void * wr_args[] = {conv_actual, wr_idx, wr_state};
        const size_t wr_lens[] = {4,4,8};
        float wr_actual[8] = {0};
        rc = run_multi(blob, blob_len, wr_args, wr_lens, 3, wr_actual, 8);
        free_blob(blob);
        if (rc != 8) {
            fprintf(stderr, "adjacent SetRows writeback NPU run failed: %d\n", rc);
            return 1;
        }
        for (int i = 0; i < 8; ++i) {
            if (fabsf(wr_actual[i] - wr_expected[i]) > 0.05f) {
                fprintf(stderr, "adjacent SetRows writeback mismatch[%d]: got %.5f expected %.5f\n",
                        i, wr_actual[i], wr_expected[i]);
                return 1;
            }
        }
    }

    struct ir_tensor rope_tensors[3] = {
        {.role=IR_INPUT, .dtype=0, .ne={8,2,2,1}},
        {.role=IR_INPUT, .dtype=1, .ne={2,1,1,1}},
        {.role=IR_OUTPUT, .dtype=0, .ne={8,2,2,1}},
    };
    struct ir_node rope_node = {.op=IR_ROPE, .n_inputs=2, .inputs={0,1}, .output=2};
    int32_t n_dims = 8, rope_mode = 0;
    float freq_base = 10000.0f, freq_scale = 1.0f, attn_factor = 1.0f;
    memcpy(rope_node.params + 4, &n_dims, sizeof(n_dims));
    memcpy(rope_node.params + 8, &rope_mode, sizeof(rope_mode));
    memcpy(rope_node.params + 20, &freq_base, sizeof(freq_base));
    memcpy(rope_node.params + 24, &freq_scale, sizeof(freq_scale));
    memcpy(rope_node.params + 32, &attn_factor, sizeof(attn_factor));
    uint32_t rope_inputs[] = {0,1}, rope_outputs[] = {2};
    blob = NULL; blob_len = 0;
    rc = compile(3, rope_tensors, 1, &rope_node, 2, rope_inputs, 1, rope_outputs,
                 "ir3-rope-normal-v1", &blob, &blob_len);
    if (rc != 0 || !blob || !blob_len) {
        fprintf(stderr, "RoPE compile failed: %d\n", rc);
        return 1;
    }
    float rope_data[32];
    int32_t rope_pos[] = {0,0};
    for (int i = 0; i < 32; ++i) rope_data[i] = (float)(i + 1) / 16.0f;
    const void * rope_args[] = {rope_data, rope_pos};
    const size_t rope_lens[] = {32,2};
    float rope_actual[32] = {0};
    rc = run_multi(blob, blob_len, rope_args, rope_lens, 2, rope_actual, 32);
    if (rc != 32) {
        fprintf(stderr, "RoPE NPU run failed: %d\n", rc);
        return 1;
    }
    for (int i = 0; i < 32; ++i) {
        if (fabsf(rope_actual[i] - rope_data[i]) > 0.01f) {
            fprintf(stderr, "RoPE zero-position mismatch[%d]: got %.5f expected %.5f\n",
                    i, rope_actual[i], rope_data[i]);
            return 1;
        }
    }
    free_blob(blob);


    struct ir_tensor sm_tensors[2] = {
        {.role=IR_INPUT, .dtype=0, .ne={4,1,1,1}},
        {.role=IR_OUTPUT, .dtype=0, .ne={4,1,1,1}},
    };
    struct ir_node sm_node = {.op=IR_SOFTMAX, .n_inputs=1, .inputs={0}, .output=1};
    float sm_scale = 1.0f;
    memcpy(sm_node.params, &sm_scale, sizeof(sm_scale));
    uint32_t sm_inputs[] = {0}, sm_outputs[] = {1};
    blob = NULL; blob_len = 0;
    rc = compile(2, sm_tensors, 1, &sm_node, 1, sm_inputs, 1, sm_outputs,
                 "ir3-softmax-axis0-v1", &blob, &blob_len);
    if (rc != 0 || !blob || !blob_len) {
        fprintf(stderr, "SoftMax compile failed: %d\n", rc);
        return 1;
    }
    /* Uniform logits → uniform probs on ggml ne0 / OV axis 3. */
    const float sm_logits[4] = {0, 0, 0, 0};
    const float sm_expected[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    const void * sm_args[] = {sm_logits};
    const size_t sm_lens[] = {4};
    float sm_actual[4] = {0};
    rc = run_multi(blob, blob_len, sm_args, sm_lens, 1, sm_actual, 4);
    if (rc != 4) {
        fprintf(stderr, "SoftMax NPU run failed: %d\n", rc);
        return 1;
    }
    for (int i = 0; i < 4; ++i) {
        if (fabsf(sm_actual[i] - sm_expected[i]) > 0.02f) {
            fprintf(stderr, "SoftMax mismatch[%d]: got %.5f expected %.5f\n",
                    i, sm_actual[i], sm_expected[i]);
            return 1;
        }
    }
    free_blob(blob);

    struct ir_tensor fa_tensors[5] = {
        {.role=IR_INPUT, .dtype=0, .ne={4,2,4,1}},
        {.role=IR_INPUT, .dtype=0, .ne={4,2,2,1}},
        {.role=IR_INPUT, .dtype=0, .ne={4,2,2,1}},
        {.role=IR_INPUT, .dtype=0, .ne={2,2,1,1}},
        {.role=IR_OUTPUT, .dtype=0, .ne={4,2,4,1}},
    };
    struct ir_node fa_node = {.op=IR_FLASH_ATTN, .n_inputs=4, .inputs={0,1,2,3}, .output=4};
    float fa_scale = 0.5f;
    memcpy(fa_node.params, &fa_scale, sizeof(fa_scale));
    uint32_t fa_inputs[] = {0,1,2,3}, fa_outputs[] = {4};
    blob = NULL; blob_len = 0;
    rc = compile(5, fa_tensors, 1, &fa_node, 4, fa_inputs, 1, fa_outputs,
                 "ir3-flash-attn-gqa-v2", &blob, &blob_len);
    if (rc != 0 || !blob || !blob_len) {
        fprintf(stderr, "FlashAttention compile failed: %d\n", rc);
        return 1;
    }
    const float q_zero[32] = {0};
    const float k_zero[16] = {0};
    const float fa_v[] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    const float fa_mask[4] = {0};
    const float fa_expected[] = {
        3,4,5,6, 3,4,5,6, 3,4,5,6, 3,4,5,6,
        11,12,13,14, 11,12,13,14, 11,12,13,14, 11,12,13,14,
    };
    const void * fa_args[] = {q_zero, k_zero, fa_v, fa_mask};
    const size_t fa_lens[] = {32,16,16,4};
    float fa_actual[32] = {0};
    rc = run_multi(blob, blob_len, fa_args, fa_lens, 4, fa_actual, 32);
    if (rc != 32) {
        fprintf(stderr, "FlashAttention NPU run failed: %d\n", rc);
        return 1;
    }
    for (int i = 0; i < 32; ++i) {
        if (fabsf(fa_actual[i] - fa_expected[i]) > 0.05f) {
            fprintf(stderr, "FlashAttention mismatch[%d]: got %.5f expected %.5f\n",
                    i, fa_actual[i], fa_expected[i]);
            return 1;
        }
    }
    free_blob(blob);
    puts("GRAPH IR SHORTCONV + STATE + ROPE + ATTENTION + MOE PASS");
    return 0;
}
