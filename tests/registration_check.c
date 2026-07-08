/* registration_check.c — Phase 0 verification for the ggml-npu backend.
 *
 * Mimics ggml's dynamic-backend loader (dlopen + ggml_backend_init + api_version
 * check) WITHOUT building all of llama.cpp, then walks the four vtables to confirm
 * the backend registers as an ACCEL device and claims no ops (supports_op == false).
 *
 * Build (needs the llama.cpp headers; -rdynamic so our stub ggml symbol is visible
 * to the dlopened .so under RTLD_NOW, exactly as a real host exports ggml's symbols):
 *   cc -rdynamic -I<llama.cpp>/ggml/include -I<llama.cpp>/ggml/src \
 *      registration_check.c -o registration_check -ldl
 * Run:
 *   ./registration_check /path/to/libggml_npu.so
 */
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

/* The only ggml symbol our cdylib references. Real hosts provide it from libggml;
 * here a stub is enough (never called: supports_op==false => no allocation). */
ggml_backend_buffer_t ggml_backend_buffer_init(
        ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i iface,
        void * context,
        size_t size) {
    (void) buft; (void) iface; (void) context; (void) size;
    return NULL;
}

typedef ggml_backend_reg_t (*init_fn)(void);
typedef int (*score_fn)(void);

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s /path/to/libggml_npu.so\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];

    void * h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    init_fn init = (init_fn) dlsym(h, "ggml_backend_init");
    if (!init) {
        fprintf(stderr, "missing ggml_backend_init symbol\n");
        return 1;
    }

    ggml_backend_reg_t reg = init();
    if (!reg) {
        fprintf(stderr, "ggml_backend_init returned NULL\n");
        return 1;
    }

    int passed = 1;
    #define CHECK(cond, msg) do {                                   \
        int c_ = (cond);                                            \
        printf("  [%s] %s\n", c_ ? "ok" : "FAIL", msg);             \
        passed = passed && c_;                                      \
    } while (0)

    printf("ggml-npu Phase 0 registration check: %s\n\n", path);

    printf("api_version   = %d\n", reg->api_version);
    printf("registry name = %s\n", reg->iface.get_name(reg));
    size_t ndev = reg->iface.get_device_count(reg);
    printf("device count  = %zu\n", ndev);

    ggml_backend_dev_t dev = reg->iface.get_device(reg, 0);
    printf("device name   = %s\n", dev->iface.get_name(dev));
    printf("device desc   = %s\n", dev->iface.get_description(dev));
    int dtype = (int) dev->iface.get_type(dev);
    printf("device type   = %d\n", dtype);
    int sop = dev->iface.supports_op(dev, NULL) ? 1 : 0;
    printf("supports_op   = %d\n", sop);

    score_fn score = (score_fn) dlsym(h, "ggml_backend_score");
    if (score) {
        printf("score         = %d\n", score());
    }

    ggml_backend_buffer_type_t buft = dev->iface.get_buffer_type(dev);

    printf("\nassertions:\n");
    CHECK(reg->api_version == GGML_BACKEND_API_VERSION, "api_version == GGML_BACKEND_API_VERSION (2)");
    CHECK(ndev == 1, "exactly one device");
    CHECK(dtype == GGML_BACKEND_DEVICE_TYPE_ACCEL, "device type is ACCEL");
    CHECK(sop == 0, "supports_op returns false (pure CPU fallback)");
    CHECK(buft != NULL, "get_buffer_type is non-null");
    if (buft) {
        CHECK(buft->iface.get_alignment(buft) >= 1, "buffer type reports sane alignment");
    }

    printf("\n%s\n", passed ? "PHASE 0 PASS" : "PHASE 0 FAIL");
    return passed ? 0 : 1;
}
