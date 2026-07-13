# Compiler migration plan

Status: implementation scope, based on repository inspection on 2026-07-12. Statements marked
**verified** are present in the pinned source trees; statements marked **proposed** are work to be
implemented or decisions still to be made.

## Source pins and checkout

| Compiler | Pin inspected | Configured branch | Remote default branch | License |
| --- | --- | --- | --- | --- |
| Intel | `30d2bb87b3044d4b8304009876299c560d224adb` | `develop` | `develop` | Apache-2.0 |
| AMD | `af5a158e7c97cee5b375c372e76269910b159553` | `main` | `main` | Apache-2.0 WITH LLVM-exception |
| Qualcomm | `9b4b8fcea2b93c801b5de784ee750ca9350d504f` | `main` | `main` | BSD-3-Clause |

The AMD and Qualcomm default branches were read from each remote's symbolic `HEAD`, not inferred
from naming. Clone all pinned compiler trees and their own dependencies with:

```sh
git submodule update --init --recursive
```

Recursive initialization is required. Intel nests `npu_plugin_elf`, its compiler LLVM fork, the NPU
cost model, and test tooling. AMD nests board descriptions, CMake modules, AIE runtime, bootgen, and
AIE API repositories. Qualcomm nests its exact Triton and Triton-Shared sources. A non-recursive
checkout is sufficient for source review but not a supported compiler build.

## Current OpenVINO path

The following is **verified** in the current ggml-npu tree.

1. `src/graph.rs` captures a scheduler split as `GraphIr`: value role, ggml dtype, four dimensions,
   byte strides, source storage identity, all node inputs, a stable `WireOp`, and 64 bytes of
   operation parameters. The legacy matmul and fused-subgraph paths use narrower structures.
2. `src/vendor/ivpu/compile.rs` checks the selected execution runtime, loads
   `libggml_npu_ovc.so` through the host dynamic-library service, and resolves:
   `ggml_npu_ov_compile_matmul_const`, `ggml_npu_ov_compile_subgraph`,
   `ggml_npu_ov_compile_graph`, and `ggml_npu_ov_free`.
3. Rust supplies shape validation, conservative capability checks, policy gates for attention and
   MoE, conversion of ggml quantized weights into grouped 4/8-bit or f16 constants, pointer-lifetime
   control for the call, and BLAKE3-based cache names. Persistent format namespace `8` currently
   identifies `openvino-native-graph`; constant content, graph semantics, quantization, and policy
   switches are included in the key.
4. `shim/ov_compile.cpp` supplies every compiler frontend function currently missing from Rust:
   - construction and shape inference of OpenVINO `Parameter`, `Constant`, and `Result` graphs;
   - ggml-to-OpenVINO axis/layout mapping and boundary conversion (normally f32 boundary, f16
     interior);
   - asymmetric u4/u8 and symmetric i4/i8 grouped dequantization, preserving the constant pattern
     used by Intel's compressed-weight kernels;
   - lowering of matmul, elementwise, RMSNorm, SiLU/SwiGLU, reshape/view/copy, transpose/permute,
     concat, ShortConv, RoPE, gather/scatter, scale/div/clamp, softmax, FlashAttention, reductions,
     TopK routing, and routed-expert matmul;
   - the routed-expert one-tile compiler option;
   - `ov::Core` lifetime, `compile_model(model, "NPU")`, and `export_model`, producing the native
     Intel program blob;
   - exception containment, malloc/free ownership, and atomic-on-rename disk cache I/O plus its
     optional size budget.
5. `shim/CMakeLists.txt` requires OpenVINO Runtime and links `openvino::runtime`; `shim/build.sh`
   locates `OpenVINOConfig.cmake`. The Rust crate does not link OpenVINO directly.
6. `src/vendor/ivpu/lower/graph.rs` asks the compiler for a transient blob and passes the exact
   semantic key and runtime bindings to the selected session.
7. With `level-zero-compat`, `src/vendor/ivpu/session_level_zero.rs` creates a
   `ZE_GRAPH_FORMAT_NATIVE` graph, discovers blob arguments, initializes it, records bindings and
   execution, and drops the transient bytes. By default, `src/vendor/ivpu/session.rs` (Direct IVPU)
   parses the Intel ELF, validates firmware and memory requirements, applies the controlled
   relocation allowlist, allocates/imports buffers, builds HostParsedInference and commands, and
   submits through the `intel_vpu` DRM UAPI. Therefore replacing compilation does not imply
   replacing either runtime.
8. `src/runtime/artifact.rs` already defines a pointer-free outer envelope, opaque target identity,
   format identifier, binding ABI, payload/resident sizes, and compatibility checks.

OpenVINO currently provides more than graph syntax: operation validation and shape inference,
low-bit pattern recognition, Intel target selection, compiler pass orchestration, native linking,
prebuilt-kernel selection, native serialization, and useful failure isolation. The replacement must
assign each responsibility explicitly; merely translating `WireOp` names to MLIR is insufficient.

## Neutral compiler boundary

The generic API must describe semantics and ownership, not a particular compiler technology.

### Rust contract (**proposed**)

Add a `CompilerProvider` contract with four operations:

- `identity() -> CompilerIdentity`: stable provider name, adapter ABI, compiler source commit,
  toolchain/build ID, pipeline revision, and deterministic option digest.
- `capabilities(request) -> CapabilityReport`: accepted/rejected graph regions and a stable reason
  code per rejected node. This operation must not compile, touch a device, or claim support based
  only on an opcode.
- `compile(request, diagnostic_sink) -> CompileOutput`: canonical graph semantics, immutable
  constants, opaque target configuration, requested binding ABI, and deterministic options in;
  one artifact envelope plus diagnostics out.
- `release(output)`: explicit release for provider-owned bytes.

`CapabilityReport` must cover rank/dim bounds, dtype and quantization forms, strides/layout,
aliasing, mutable state and ordering, input/output limits, dynamic-shape bounds, operation
parameters, and required runtime features. `Supported` means the same request can enter `compile`;
`Unsupported` causes scheduler fallback; `Invalid` is a graph bug; `Unavailable` is a missing
compiler/toolchain.

The boundary uses length-delimited bytes and integers only. No Rust layout, C++ object, MLIR object,
exception, filesystem path, or vendor enum crosses it. Target and payload formats remain opaque to
generic code. Generic names must remain technology-neutral; vendor dialect, architecture, firmware,
kernel, and linker choices belong in adapters.

### C ABI (**proposed**)

Each dynamically loaded provider exports one versioned function-table getter. The table operates on
opaque handles and byte spans, returns stable status/domain/code triples, accepts an optional
diagnostic callback, and returns buffers paired with the provider's release function. Requirements:

- no C++ exception, allocator, `std::string`, MLIR handle, or thread-local error pointer crosses the
  ABI;
- create/destroy and compile-result release are paired and idempotence rules are documented;
- request bytes remain caller-owned only for the duration of the call;
- output bytes remain provider-owned until release unless a caller allocator is supplied;
- diagnostics carry severity, stage, stable code, node/value index, and UTF-8 text;
- providers declare `reentrant`, `serialized`, or `one-context-per-thread`; ggml-npu serializes
  compilation unless reentrancy is verified;
- cancellation is cooperative and never leaves a cacheable partial artifact.

This is an in-process API. Production compilation must not invoke a CLI, Python interpreter, shell,
or child process. CLI tools remain useful only as conformance or debugging oracles.

## Artifact and cache contract

The current `GGNPUART` envelope should remain the outer transport, with an ABI-major bump only if
the fixed header layout changes. Add a versioned, pointer-free requirements manifest before the
opaque payload containing:

- compiler identity and pipeline revision;
- payload format revision and content digest;
- exact/compatible target identity, including architecture, revision/stepping, tile/partition
  configuration, firmware/runtime ABI ranges, and required capabilities;
- ordered bindings with role, dtype, logical/allocation bytes, alignment, access, alias group, and
  optional stable name;
- scratch, executable, constant, and total resident-memory requirements;
- required prebuilt-kernel/runtime-module identities;
- linker/loader ABI and relocation policy;
- deterministic-build marker and optional separated-constant object list.

The runtime validates the manifest before parsing a vendor payload. Intel keeps its ELF and
HostParsedInference checks below that boundary; AMD keeps xclbin/instruction and XRT argument
rules; Qualcomm keeps object/shared-object, QuRT/FastRPC, VTCM, and symbol rules.

A persistent cache key is a domain-separated BLAKE3 digest over canonical graph semantics,
constant contents (never addresses), requested binding ABI, complete target configuration,
compiler commit/build ID, adapter and pipeline versions, every code-generation option, LLVM and
auxiliary-toolchain pins, prebuilt kernel/runtime-module digests, and artifact schema. Cache files
are written atomically and rejected if the embedded identity disagrees with the filename.

Determinism is a release gate: fixed inputs in a pinned container must produce byte-identical
artifacts. Disable timestamps, random symbols, unstable iteration, and absolute build paths; use
deterministic archives/linking and sorted pass inputs. If a vendor linker cannot yet meet this,
mark artifacts non-deterministic, prohibit shared persistent-cache publication, and track the
remaining differing sections. Reproducibility means clean-build equality, not merely equal output
numerics.

## What is actually shared

There is a real but narrow shared foundation:

- ggml `GraphIr` semantics, canonical serialization, capability/diagnostic schemas;
- artifact envelope, cache identity, ownership/error C ABI, test harness, and runtime lifecycle;
- upstream MLIR concepts and reusable upstream dialects such as builtin, tensor, linalg, arith,
  math, vector, affine, scf, memref, func, and LLVM;
- generic pass-manager utilities when built against the *same* pinned LLVM.

A universal MLIR module or one process-wide MLIR library is not currently credible. Intel pins its
own compiler LLVM and lowers through IE/VPU/VPUIP/VPUMI/VPUASM/NPUReg dialect families. AMD uses
AIE/AIEX/AIEVec/XLLVM, explicit tile/routing/DMA models, and Peano or xchesscc. Qualcomm is tied to
Triton's LLVM revision and uses TTX, TmTensor, Crouton, HexagonMem, HexagonTPtr, HexKL, HVX and
DMA-specific passes. Their shape semantics, address spaces, kernel ABIs, schedulers, link products,
and runtime bindings differ.

Consequently, share source-level utilities selectively, but build each provider with hidden
symbols behind its C ABI. Never link incompatible LLVM/MLIR revisions into one global namespace.
Load one provider at a time or prove namespace isolation. `GraphIr` is the common semantic request,
not a promise that every provider receives identical MLIR.

## Provider adapters

### Intel first

**Verified:** the fork is a CMake/C++17 project. It requires `OpenVINODeveloperPackage` and builds
`openvino_intel_npu_compiler`, `openvino_intel_npu_compiler_loader`, and a static compiler target.
The installed VCL 7.7 C API creates a compiler from PCI device/revision/tile count, reports compiler
identity and options, accepts serialized OpenVINO XML plus weights, emits a serializable native blob
and compatibility requirements, returns logs, supports caller allocators, and has parallel compile
tests. Internal dialects include IE, VPU, VPUIP, VPUIPDPU, VPUMI37XX/40XX, VPUASM, NPURegMapped and
NPUReg40XX. The output is Intel device ELF/native graph data. Prebuilt SHAVE kernel ELFs for
37xx/40xx/50xx are embedded by default; rebuilding them requires separate MoviTools.

**Gap:** VCL is an excellent ownership/error/lifetime reference but not the final frontend: it
accepts OpenVINO IR and the build links OpenVINO NPU abstraction/runtime libraries. The fork has no
verified public entrypoint accepting `GraphIr`, neutral serialized semantics, or an MLIR module
independent of OpenVINO.

**Proposed adapter:** add a standalone target in the fork that decodes the neutral request, performs
shape/type/constant validation, constructs the existing internal IE/VPU pipeline directly, and
returns the same ELF plus compatibility requirements. Preserve NPU dialects, pass pipelines,
compressed-constant forms, SHAVE kernels, architecture/stepping/tile options, ELF metadata,
relocations, firmware ABI, and linker behavior below the boundary. Initially compare its output to
VCL/OpenVINO as an oracle; do not translate Intel dialects into a new universal dialect.

### AMD boundary

**Verified:** this is MLIR-AIE/IRON, not an end-to-end model compiler. It targets AIE1, AIE2 and
AIE2P devices; Ryzen `npu1` covers Phoenix/Hawk Point and `npu2` covers Strix/Strix Halo/Krackan
Point, including partition variants. It requires CMake 3.30 and a matching MLIR installation,
builds `aie-opt`, `aie-translate`, and C++ `aiecc`, and exposes Python bindings. `aiecc` runs MLIR
passes/translations in-process and preserves AIE/AIEX/AIEVec routing, locks, object FIFOs, buffer
descriptors, DMA and target models. It emits per-core ELF, NPU instruction binaries, transaction
MLIR, CDO, PDI, full ELF, control packets, and xclbin, and supports external kernel objects.
Runtime expectations are XDNA plus XRT.

**Gap:** there is no stable whole-compile C ABI. The available C API covers target-model queries,
not compilation. Even C++ `aiecc` invokes Peano `opt`/`llc`/`clang` for core code and `xclbinutil`
for packaging; AIEBU and bootgen are only optionally in-process. The repository explicitly says it
complements, rather than replaces, mainstream inference tooling. A ggml graph-to-spatial-AIE
partitioner and a no-subprocess linker/packager are not verified.

**Proposed adapter:** first expose the existing in-memory pipelines and translations through the
neutral C ABI. Add in-process Peano library integration or a deliberately small embedded
LLVM-AIE codegen library, direct AIEBU/bootgen, and an in-process xclbin writer. Preserve target
model, tile partition, DMA/object-FIFO ABI, per-core kernels, instruction sequence, xclbin metadata,
and XRT positional argument order. Build an AMD-specific GraphIr lowering; do not route it through
Intel IE/VPU IR.

Compiler integration can proceed with artifact fixtures and validators before an executable AMD
backend exists. `src/vendor/xdna/mod.rs` is currently a skeleton: device discovery, XRT ownership,
buffer synchronization, xclbin registration, instruction BO, kernel arguments, submit and wait
remain independent runtime work.

### Qualcomm boundary

**Verified:** Hexagon-MLIR is a Triton/PyTorch-oriented compiler stack tested for Hexagon v73, v75,
v79 and v81 (HexKL is currently invalid for v81). It builds as Triton plugins/tools rather than a
standalone root CMake project. The pipeline is TTIR -> Triton-Shared/Linalg -> Qualcomm dialects ->
LLVM IR -> Hexagon object, with optional HTP shared-object generation. It has C++ opt/translate
tools and pybind functions such as `translate_linalg_to_obj`, but the production Python backend
still invokes `triton-shared-opt` as a subprocess. Its version hash is literally `TODO`. It emits
one or more `.o` files, optional `.so`, LLVM IR, and optionally separates constants. It embeds
Hexagon SDK runtime bitcode for HVX intrinsics, DMA, VTCM, Hexagon C API, and HexKL into generated
code. Builds require the matching Triton LLVM, Hexagon SDK 6.4.0.2, Hexagon Tools 19.0.02, QuRT
headers and optionally HexKL 1.0.0; target features include 128-byte HVX. Runtime execution in this
tree uses Hexagon/QuRT launch machinery, not a verified QNN context-binary compiler.

**Gap:** no stable standalone C ABI, neutral graph frontend, in-process Triton-to-Linalg stage,
stable compiler identity, or ggml runtime loader is verified. The fork does **not** currently emit
the QNN context binary assumed by the old skeleton comment.

**Proposed adapter:** move Triton-Shared conversion into the linked pass pipeline, expose dialect
registration/lowering/object emission through the neutral ABI, add stable identity and target
configuration, and define a deterministic in-process link path for object(s), separated constants,
runtime bitcode, and the DSP shared object. Preserve HVX/VTCM/DMA/HexKL decisions, QuRT ABI, Hexagon
architecture and shared-object symbols below the boundary. A direct GraphIr-to-Linalg/TTX lowering
is Qualcomm-specific.

This work does not duplicate an official OpenVINO Qualcomm plugin or an official Hexagon/QNN
backend. Compiler and runtime choices are independent pairs: an OpenVINO-produced artifact can use
its supported runtime, a QNN context binary can use QNN/FastRPC, and a Hexagon-MLIR object can use a
native DSP-object loader. They may coexist behind capability selection. Hexagon-MLIR supersedes
another path only after it matches that path's graph semantics, artifact contract, device coverage
and runtime performance; compiler availability alone does not do so. The current
`src/vendor/hexagon/mod.rs` remains a runtime skeleton and its QNN-context assumptions must not be
used to claim that Hexagon-MLIR artifacts are loadable.

## Intel replacement sequence and gates

1. **Freeze evidence and toolchains.** Record all submodule and nested pins, CMake presets, LLVM,
   OpenVINO oracle, compiler options, prebuilt-kernel digests, host compiler/linker, and target
   firmware identity. Produce a hermetic x86_64 compiler SDK. Gate: clean build from pins.
2. **Tiny known-answer artifact.** Through the new C ABI, compile one f16 add or 1xKxN matmul for
   the exact NPU40xx target; inspect the envelope/ELF, compare bindings and requirements against the
   OpenVINO oracle, run through Level Zero and direct DRM, and compare output to CPU. Gate: repeated
   build bytes and diagnostics are deterministic.
3. **Constant matmul parity.** Implement f16, grouped u8/u4 asymmetric, and debug signed forms with
   the existing fused-decompression semantics. Gate: numerical tolerances, blob/resident size, and
   pp/tg latency no worse than the current shim's agreed budget.
4. **Graph operation parity.** Port capability and lowering in measured increments: elementwise and
   layout; RMSNorm/SiLU/SwiGLU/FFN; ShortConv and mutable state; RoPE/softmax/attention; gather,
   scatter and routed experts. Unsupported variants must reject before compile. Gate each operation
   with property tests, malformed requests, oracle differential tests, and hardware tests.
5. **Full LFM model.** Compile and execute all intended scheduler splits for prefill and decode of
   the selected LFM model, including state mutation and routed experts, while fallback remains
   correct. Gate: fresh-process output quality, no device reset/hang, bounded compile time and
   memory, cache warm hit, and stable split coverage.
6. **Identity and reproducibility.** Switch persistent keys to the full neutral identity, implement
   envelope validation and atomic cache publication, and prove clean-build reproducibility. Old
   `b3v8` OpenVINO artifacts remain a separate namespace and are never reinterpreted.
7. **Direct-submission qualification.** Run the same artifact corpus through Level Zero and direct
   DRM. Gate: binding metadata, relocations, HPI/firmware versions, resident accounting,
   import/copy behavior, synchronization and fault recovery all pass; runtime choice does not alter
   compiler cache identity except where target requirements differ.
8. **Remove OpenVINO.** Delete `shim/ov_compile.cpp`, `shim/CMakeLists.txt`, `shim/build.sh`, the
   `GGML_NPU_OVC` loader and all `ggml_npu_ov_*` symbols only when every prior gate passes and:
   - all enabled Intel graph paths use the new provider, with explicit fallback on rejection;
   - no ggml-npu build, package, test, or runtime links/loads OpenVINO;
   - the standalone Intel compiler target itself no longer requires OpenVINO headers,
     `OpenVINODeveloperPackage`, `openvino::runtime`, or `openvino::npu_al` transitively;
   - operation coverage and LFM quality/performance gates meet the recorded baseline;
   - both Intel runtimes consume the new artifact and old cache entries are safely ignored;
   - CI builds and tests with OpenVINO absent.

The last condition makes full removal harder than merely replacing the shim: the inspected Intel
fork is itself OpenVINO-coupled. That standalone-build work is a critical blocker, not an assumed
existing capability.

## Conformance, performance, and failure gates

Required host-only tests:

- canonical GraphIr serialization, constant digesting, option ordering, malformed lengths and
  allocation limits;
- capability/compile agreement and stable rejection diagnostics;
- artifact envelope, target/firmware mismatch, binding/alias validation, relocation and resident
  limits;
- C ABI ownership, allocator failure, repeated create/destroy, concurrent calls according to the
  declared thread model, and exception/panic containment;
- deterministic rebuild in two clean directories and cross-host reproducibility where supported;
- fixture inspection for Intel ELF, AMD xclbin/instructions/per-core ELF, and Qualcomm objects,
  separated constants and runtime symbols.

Required hardware tests per supported target:

- tiny known answers, every operation variant, fused graph, mutable state, multi-output and
  zero/copy/shared-memory bindings;
- model prefill/decode correctness and quality against CPU, long-run state correctness, device-loss
  and fallback behavior;
- compile cold/warm latency, artifact and resident bytes, stage-in/out bytes, submit/device/sync
  time, dispatch count, pp/tg throughput, energy where available, and peak host memory;
- stress across cache eviction, repeated model loads, concurrent compile requests, and firmware
  compatibility.

Performance promotion requires confidence intervals and a pinned baseline. A provider is not made
default merely because one kernel is faster; end-to-end compile amortization, memory pressure,
fallback boundaries and runtime transfers must meet the release budget.

Cross-compilation is separate for provider and runtime. Compiler providers may be host x86_64
libraries while artifacts target another ISA/device. Build target code with the pinned device
toolchain, never host `PATH` discovery. AMD must pin Peano/AIEBU/bootgen/xclbin packaging and its
XRT ABI. Qualcomm must pin Triton LLVM, Hexagon SDK/tools/QuRT/HexKL and support the aarch64 host
runtime separately. Intel must pin its LLVM, NPU ELF/cost-model repositories, prebuilt kernels and
firmware compatibility. Every cross artifact records target, not build-host, identity.

## Ordered work breakdown

1. Land submodule pins and this scope; add CI verification for recursive checkout and clean pins.
2. Define and test neutral request, capability, diagnostic, identity, C ABI and artifact manifest.
3. Build a hermetic Intel fork SDK and add the standalone provider target.
4. Pass Intel tiny-artifact and constant-matmul gates.
5. Port Intel graph coverage, then pass full-LFM, cache/reproducibility and dual-runtime gates.
6. Remove OpenVINO only at the explicit criteria above.
7. In parallel after contract freeze, add AMD fixture validation and in-process packaging work;
   implement XDNA runtime separately, then join them at the artifact contract.
8. Add Qualcomm fixture validation and in-process Triton-Shared/link work; choose native
   Hexagon-object versus QNN-context runtime separately, then implement and join that runtime.
9. Promote AMD/Qualcomm providers only after their own operation, model, compatibility and
   performance matrices pass. Their incompleteness must not delay Intel removal.

## Likely later file changes (not part of this change)

Generic additions:

- `src/compiler/mod.rs`, `src/compiler/contract.rs`, `src/compiler/abi.rs`,
  `src/compiler/diagnostics.rs`, `src/compiler/cache.rs`
- `src/runtime/artifact.rs`, `src/graph.rs`, `src/identity.rs`, `src/vendor/mod.rs`
- `tests/compiler_abi.rs`, `tests/compiler_conformance.rs`, `tests/artifact_fixtures.rs`
- compiler SDK build/packaging files and CI workflows

Intel:

- add `src/vendor/ivpu/compiler/{mod.rs,ffi.rs,lower.rs,target.rs}`
- change `src/vendor/ivpu/lower/graph.rs`, `src/vendor/ivpu/lower/matmul.rs`,
  `src/vendor/ivpu/lower/subgraph.rs`, `src/vendor/ivpu/session.rs`,
  `src/vendor/ivpu/session_level_zero.rs`, and direct artifact validation as required
- retire `src/vendor/ivpu/compile.rs` and finally delete `shim/`
- in the Intel compiler fork, add a standalone provider library/C header, neutral-request decoder,
  GraphIr-to-IE lowering, diagnostics, and standalone CMake profile; refactor existing pipeline,
  linker, compatibility and prebuilt-kernel code rather than copying it

AMD:

- add `src/vendor/xdna/compiler/`, `src/vendor/xdna/runtime/`, artifact/target validators and XRT
  bindings; replace the current `src/vendor/xdna/mod.rs` skeleton
- in the AMD compiler fork, add the provider C ABI, GraphIr lowering, library form of the `aiecc`
  orchestration, and in-process core codegen/xclbin packaging

Qualcomm:

- add `src/vendor/hexagon/compiler/`, the selected runtime adapter, artifact/target validators and
  binding-name manifest support; replace the current `src/vendor/hexagon/mod.rs` skeleton
- in the Qualcomm compiler fork, add the provider C ABI, GraphIr lowering, in-process
  Triton-Shared pipeline, stable identity, and deterministic object/shared-object linker

Build integration will eventually change Cargo/build/packaging files, but compiler SDKs should be
prebuilt and dynamically loaded so ggml-npu remains `no_std`, avoids incompatible LLVM libraries in
its link image, and degrades to CPU when a provider is absent.

## Decisions still required

1. Intel frontend: expose existing internal IE construction directly, or define a small
   fork-owned semantic dialect before IE. It must not depend on OpenVINO to satisfy removal.
2. Provider loading: ordinary local `dlopen`, isolated namespaces where available, or a strict
   one-provider-per-process rule to prevent LLVM symbol collisions.
3. Compilation scheduling: one global worker, one context per provider, or verified concurrent
   contexts; current vendor APIs do not establish one common thread-safety guarantee.
4. Artifact manifest encoding and ABI evolution strategy, including separated constant objects.
5. AMD deliverable for first runtime: xclbin plus instruction blob, full ELF, or another XRT-native
   package, and the exact XDNA/XRT ABI matrix.
6. Qualcomm runtime pair: native Hexagon `.so` over FastRPC/QuRT, QNN context binary via a distinct
   compiler, or both. The inspected fork only verifies the native object/shared-object route.
7. Distribution policy for Intel prebuilt kernels, AMD proprietary-optional Vitis components, and
   Qualcomm SDK/Tools/HexKL dependencies under their respective licenses.
