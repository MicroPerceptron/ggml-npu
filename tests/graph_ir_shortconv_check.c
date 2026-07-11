/* Tiny end-to-end Graph IR v2 ShortConv check:
 * Graph IR ABI -> OpenVINO compile -> native blob -> Rust Level Zero submit -> CPU comparison.
 */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

enum { IR_INPUT = 0, IR_CONSTANT = 1, IR_OUTPUT = 4, IR_SSM_CONV = 12,
       IR_MATMUL_ID = 17, IR_MAX_INPUTS = 16 };
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
    if (!compile || !free_blob || !run) {
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
    for (size_t i = 0; i < sizeof(experts)/sizeof(experts[0]); ++i) experts[i] = 0x3c00;
    struct ir_tensor moe_tensors[4] = {
        {.role=IR_CONSTANT, .dtype=0, .ne={4,2,4,1}, .bits=16,
         .weight=(const uint8_t *)experts, .weight_len=sizeof(experts)},
        {.role=IR_INPUT, .dtype=0, .ne={4,1,1,1}},
        {.role=IR_INPUT, .dtype=1, .ne={2,1,1,1}},
        {.role=IR_OUTPUT, .dtype=0, .ne={2,2,1,1}},
    };
    struct ir_node moe_node = {.op=IR_MATMUL_ID, .n_inputs=3, .inputs={0,1,2}, .output=3};
    uint32_t moe_inputs[] = {1,2}, moe_outputs[] = {3};
    blob = NULL; blob_len = 0;
    rc = compile(4, moe_tensors, 1, &moe_node, 2, moe_inputs, 1, moe_outputs,
                 "ir2-moe-gather-e2e-v1", &blob, &blob_len);
    if (rc != 0 || !blob || !blob_len) {
        fprintf(stderr, "MoE Gather/MatMul compile failed: %d\n", rc);
        return 1;
    }
    free_blob(blob);
    puts("GRAPH IR SHORTCONV + MOE COMPILE PASS");
    return 0;
}
