# Semantic identity

Persistent artifacts use full BLAKE3-256 with explicit domains and length-framed fields. Cache
namespace/version `b3v7` cleanly misses the older FNV files. Graph, subgraph, tensor, op-parameter,
shape, stride, output-order, compiler policy, and requant-content semantics are included.

Large weight/scales/zero slices are streamed directly into BLAKE3. Scalar metadata is accumulated
in a 1 KiB stack buffer to reduce tiny `Hasher::update` calls. Requantized constants have a bounded
in-process digest cache keyed by exact immutable source identity, model-buffer generation, shape,
dtype, and quantization policy. Persistent keys contain only the resulting domain-separated content
digest, never source addresses.

In-process Graph IR and fused-subgraph caches use owned exact length-framed keys. BLAKE3 is checked
first, but an equality hit also compares the complete framed bytes. `BindingKey` remains the exact
fixed-size pointer/generation key required for recorded command-list safety.

## SIMD configuration

`blake3` uses `default-features = false, features = ["pure"]`.

- x86/x86_64: the build script enables Rust SSE2, SSE4.1, and AVX2 implementations; runtime
  `cpufeatures` detection selects among them. The measured host selected AVX2, SIMD degree 8.
- AVX-512: disabled because the crate's implementation requires C/assembly under this configuration.
- little-endian AArch64: `pure` disables the crate's C NEON implementation, so the portable Rust
  backend remains. This is the explicit cost of avoiding C/build-host coupling.

Do not add portable_simd or pulp around BLAKE3 compression. The crate already owns compression
dispatch. Those tools are candidates only for separately measured field packing or requantization.

## Criterion smoke sample

One `--quick` run on the Lunar Lake host observed:

- tiny metadata: naive updates ~1.07 us, buffered frames ~0.532 us;
- 256-field Graph IR metadata: naive ~18.6 us, buffered ~12.75 us;
- 5 MiB contiguous input: built-in BLAKE3 ~2.48 ms, framed stream ~1.79 ms;
- 24 × 5 MiB routed stream: built-in chunks ~44.7 ms, framed chunks ~43.1 ms.

Quick samples are diagnostics, not performance conclusions. The metadata result supports buffering;
large-stream differences are close enough that no additional SIMD layer is justified.
