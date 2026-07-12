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
       IR_SSM_CONV = 12, IR_ROPE = 13, IR_ARGSORT = 16, IR_MATMUL_ID = 17,
       IR_FLASH_ATTN = 22, IR_SET_ROWS = 26,
       IR_MAX_INPUTS = 16 };
struct ir_tensor {
    int32_t role, dtype;
    int64_t ne[4];
    int32_t bits, group;
    const uint8_t * weight; size_t weight_len;
    const uint8_t * scales; size_t scales_len;
    const uint8_t * zeros;  size_t zeros_len;
};
struct ir_node {
    int32_t op, n_inputs, inputs[IR_MAX_INPUTS], output;
    uint8_t params[64];
};
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
    compile_fn compile = (compile_fn)dlsym(shim, "ggml_npu_ov_compile_graph");
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
        {.role=IR_CONSTANT, .dtype=0, .ne={3,2,1,1}, .bits=16,
         .weight=(const uint8_t *)kernels, .weight_len=sizeof(kernels)},
        {.role=IR_OUTPUT, .dtype=0, .ne={2,2,1,1}},
    };
    struct ir_node node = {.op=IR_SSM_CONV, .n_inputs=2, .inputs={0,1}, .output=2};
    uint32_t inputs[] = {0}, outputs[] = {2};
    uint8_t * blob = NULL;
    size_t blob_len = 0;
    int rc = compile(3, tensors, 1, &node, 1, inputs, 1, outputs,
                     "ir2-shortconv-e2e-v1", &blob, &blob_len);
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
        {.role=IR_CONSTANT, .dtype=0, .ne={4,2,4,1}, .bits=16,
         .weight=(const uint8_t *)experts, .weight_len=sizeof(experts)},
        {.role=IR_INPUT, .dtype=0, .ne={4,1,1,1}},
        {.role=IR_INTERMEDIATE, .dtype=1, .ne={2,1,1,1}},
        {.role=IR_OUTPUT, .dtype=0, .ne={2,2,1,1}},
    };
    struct ir_node moe_nodes[2] = {
        {.op=IR_ARGSORT, .n_inputs=1, .inputs={0}, .output=3},
        {.op=IR_MATMUL_ID, .n_inputs=3, .inputs={1,2,3}, .output=4},
    };
    uint32_t moe_inputs[] = {0,2}, moe_outputs[] = {4};
    blob = NULL; blob_len = 0;
    rc = compile(5, moe_tensors, 2, moe_nodes, 2, moe_inputs, 1, moe_outputs,
                 "ir2-moe-routing-e2e-v2", &blob, &blob_len);
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
                 "ir2-set-rows-state-v1", &blob, &blob_len);
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
                 "ir2-rope-normal-v1", &blob, &blob_len);
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
                 "ir2-flash-attn-gqa-v2", &blob, &blob_len);
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
