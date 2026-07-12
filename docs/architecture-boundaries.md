# Architecture boundaries

Dependencies point inward. A lower layer must not name a provider, operating system, compiler,
firmware protocol, graphics API, or foreign-handle representation owned by a higher layer.

## Layers

1. **Core semantics** — graph values and operations, byte lengths, typed address/range mechanics,
   access modes, identities, metrics, and capability descriptions.
2. **Runtime contracts** — fallible lifecycle, allocation intent, immutable bindings, artifact
   envelopes, deadlines, ownership, and completion. Artifact formats and target identities are
   opaque here; their schemas belong to adapters.
3. **OS adapters** — owned system handles, files, mappings, clocks, polling, persistence, and
   syscall encodings.
4. **External API adapters** — API-specific memory handles, import/export descriptors, queue and
   fence rules, and synchronization calls.
5. **Provider/compiler adapters** — device discovery, executable formats, target fingerprints,
   relocations, command protocols, and compilation conventions.

Provider and API adapters may depend on runtime, OS, and core modules. OS adapters may depend on
runtime/core value semantics. Runtime and core modules must not depend on adapters.

## Promotion rule

A concept moves inward only when at least two implementations can share its invariants and failure
semantics without translation by convention. Similar field names are not sufficient. In
particular, external-memory handle types, import/export capability negotiation, fence ownership,
cache maintenance, and queue-family transfer rules remain adapter-owned until proven otherwise.

The currently proven shared-allocation surface is intentionally small:

- `AllocationIdentity` identifies one backing allocation without using a process-local pointer or
  handle.
- `AccessMode` expresses read/write intent.
- `SharedMapping` exposes a live owner-controlled host mapping as `NonNull<u8> + usize`; it does not
  turn that pointer into an integer address or claim cross-device visibility.

## Address and artifact rules

`src/address.rs` defines only `Address<Domain>`, `Offset<Domain>`, checked half-open ranges,
`ByteLen`, and `Alignment`. Concrete domains and aliases live in adapters. Raw ABI/UAPI structures
retain their specified primitive fields and convert immediately after crossing the boundary.

`runtime::artifact` treats format identifiers and target identity words as opaque. The adapter for a
format owns the numeric identifier, target-word schema, validation, and decoding.

## Host environments

`src/host/` is the single composition and platform-selection boundary. Its focused services cover
files/mappings, clocks, memory status, configuration, diagnostics/process exit, allocation,
dynamic libraries, scalar host math, and artifact loading. Runtime code may accept any
`HostEnvironment`; normal builds use `host::builtin()`.

Only `src/host/mod.rs` selects a built-in implementation with `cfg(target_os)`. An unsupported
target receives a compile-time error instead of silently compiling partial stubs. To add a host:

1. implement each focused service using that host's supported APIs;
2. compose them in one `HostEnvironment`;
3. add the implementation to the centralized selection block;
4. keep raw handles, loader constants, paths, and control-request encodings private to the host or
   provider-specific host extension;
5. run the mock-host tests and generic-module boundary scan.

`src/host/mock.rs` is the injection example for host-independent tests. Linux file, mapping,
DMA-BUF, DRM PRIME/GEM, and typed ioctl mechanisms live under `src/host/linux/`. Provider control
protocols remain adapter-owned; IVPU request semantics and error translation are confined to the
explicit `vendor/ivpu/direct/linux.rs` composition adapter.

## Known deferred separations

- `src/graph.rs` still combines the graph model with the ggml capture/decoding adapter. Its module
  documentation marks that boundary; splitting the model from source-API decoding should preserve
  the current graph-cache identity and operation validation tests.
- The legacy five-verb HAL remains a compatibility façade. New execution work targets
  `runtime::RuntimeHal`.
- External-memory handles and synchronization protocols are not part of the generic interop
  contract. OS, graphics/compute API, and provider adapters should first prove ownership and
  ordering independently.
