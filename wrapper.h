/* Aggregated headers for bindgen.
 *
 * ggml-backend-impl.h carries the struct layouts (the four vtables) we fill.
 * ggml.h carries ggml_tensor (needed for the buffer set/get paths).
 * ggml_cgraph stays opaque here (only forward-declared in ggml.h); Phase 0 never
 * dereferences it, so we deliberately do NOT pull in the private ggml-impl.h.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
