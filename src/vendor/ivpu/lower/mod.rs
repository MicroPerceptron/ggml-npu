//! Intel op lowerings: ggml op + shape → compiled NPU artifact (OpenVINO shim) → Level Zero
//! submit. One module per op; each fetches the shape's blob from the JIT cache
//! ([`super::compile`]) and runs it via the Level Zero HAL ([`super::graph`]).

pub mod matmul;
