# Direct runtime foundation

The dependency and promotion rules for this implementation are defined in
[`architecture-boundaries.md`](architecture-boundaries.md).

This milestone introduces an OS-neutral runtime contract and a Linux implementation without
replacing the working OpenVINO + Level Zero path. `rustix` supplies syscall mechanics only; it does
not define Intel firmware commands, graph formats, queue semantics, or compiler behavior.
Default execution lives in `src/vendor/ivpu/session.rs` (Direct IVPU); Level Zero remains an
opt-in oracle in `src/vendor/ivpu/session_level_zero.rs` behind `level-zero-compat`.

## Implemented boundary

- `src/runtime/`: fallible lifecycle traits for device, memory, program, immutable bindings,
  queue, event, deadlines, unload, and destruction.
- `src/runtime/artifact.rs`: a versioned pointer-free artifact envelope containing an opaque target
  identity, opaque format identifier, binding ABI, payload length, and resident-byte requirement.
- `src/host/`: focused host services and the single compile-time platform-selection point. Linux
  uses rustix where available; process allocation, environment, loader, exit, and C-math ABI calls
  remain isolated in the Linux host implementation.
- `src/host/linux/`: `rustix` owned files, mappings, polling, DMA-BUF, DRM PRIME/GEM, typed ioctl,
  process, loader, and monotonic-clock implementation modules.
- `src/vendor/ivpu/direct/`: feature-gated Linux UAPI subset for discovery, parameters, bounded
  GEM BO ownership, explicit command queues, validated submit, and finite BO waits.
- `src/vendor/ivpu/direct/elf.rs`: Goblin-backed NPU ELF preflight and zero-copy section, symbol,
  and relocation views.
- `src/vendor/ivpu/direct/link.rs`: bounded host-side link operations for an explicit Intel
  relocation allowlist. It does not map or mutate device memory.

Goblin is optional and enabled only by `direct-ivpu`, with defaults disabled and
`elf32`, `elf64`, and `endian_fd` enabled. `endian_fd` is Goblin's no-std parsing feature and
requires its `alloc` support; Goblin's unified parser currently also references both ELF classes,
so `elf32` must compile even though the preflight rejects every non-ELF64 device artifact before
the allocating parse. PE, Mach-O, archive, TE, and `std` support are disabled.

Goblin is not used for host dynamic loading. Host `dlopen`/`dlsym` must retain loader namespaces,
dependency and symbol resolution, constructors, TLS, and platform security behavior that parsing
an ELF file cannot reproduce.

The IVPU definitions are transcribed from `/usr/include/drm/ivpu_accel.h` supplied by Fedora
`kernel-headers-7.1.3-200.fc44.x86_64`. Layout and `_IOC` request values are asserted in tests:

- `DRM_IOCTL_IVPU_GET_PARAM = 0xc0106440`
- `DRM_IOCTL_IVPU_BO_CREATE = 0xc0186442`
- `DRM_IOCTL_IVPU_BO_INFO = 0xc0206443`
- `DRM_IOCTL_IVPU_BO_WAIT = 0xc0186446`
- `DRM_IOCTL_IVPU_CMDQ_CREATE = 0xc00c644b`
- `DRM_IOCTL_IVPU_CMDQ_DESTROY = 0x4004644c`
- `DRM_IOCTL_IVPU_CMDQ_SUBMIT = 0x4020644d`
- `DRM_IOCTL_GEM_CLOSE = 0x40086409`

The inspected source pins are:

- Fedora kernel UAPI: `kernel-headers-7.1.3-200.fc44.x86_64`.
- Linux mainline snapshot: `44696aa3a489d2baf58efa61b37833f100072bee`.
- Installed Fedora UMD: `intel-npu-driver-1.32.0-1.fc44`, corresponding to upstream tag
  `v1.32.0` (`89391631e99707269c07b87687ac828ed6823df5`).
- Release-pinned ELF loader submodule:
  `openvinotoolkit/npu_compiler_elf@82c444bcb9feb0f55fa33e18fbd711ec35426fba`.
- Public firmware command header:
  `firmware/include/api/vpu_jsm_job_cmd_api.h` at `v1.32.0`, Job Command API 4.15.

The hardware reports firmware API word 3 as `0x0004000f`, exactly matching that public 4.15
header, and word 7 as `0x000b000d`, matching NNRT 11.13. This command ABI is public and versioned
but is not Linux stable UAPI; it must remain release-pinned and rejected on a version mismatch.

## Stable versus release-private boundaries

- **Linux stable UAPI:** open-created contexts, GET_PARAM, BO create/info/map/close, BO wait,
  command-queue create/destroy, and the outer submit descriptor (handles, queue ID, command offset,
  and preemption index).
- **Public, release-versioned firmware ABI:** command-buffer header and NOP/timestamp/fence/barrier/
  copy/fill/inference command records from `vpu_jsm_job_cmd_api.h`.
- **Public, separately versioned compiler ABI:** NPU ELF extensions, HostParsedInference loading,
  section allocation, relocations, metadata, and input/output patching in `npu_compiler_elf`.
- **Not supplied by the kernel:** semantics of compiler ELF sections and relocations, native
  inference construction, and binding patching. The kernel validates handles and offsets but passes
  the command-buffer VPU address to firmware; it does not validate the firmware command stream.

One opt-in ignored hardware test now loads and executes the bounded cached 128-byte vector-add
artifact through the direct path. It compares against a Level Zero execution of the same artifact,
submits twice in one queue only after the first result passes, and enforces a two-second deadline
per wait. Metric streaming and general graph execution remain unimplemented. All hardware tests
are ignored by default.

Build the foundation with:

```sh
cargo check --features direct-ivpu
cargo test --features direct-ivpu
GGML_NPU_TEST_ARTIFACT=./artifacts/bounded-vector-add.blob \
  cargo test --lib --features bench-hardware \
  vendor::ivpu::direct::tests::tiny_known_answer_direct_submission \
  -- --ignored --exact --nocapture --test-threads=1
```

## Required before general direct execution

1. **Canonical compiler contract.** Stabilize a pointer-free canonical module independent of
   `GraphIr`, then implement in-process compiler adapters for Intel, AMD, and Qualcomm. A compiler
   subprocess is not part of the architecture.
2. **Intel ELF ingestion.** Goblin parsing now
   validates target `EM_NONE`, serialized NPU40XX platform metadata, all table/range/alignment
   bounds, symbols, relocations, and a 512 MiB resident ceiling before its bounded header-vector
   allocations. The minimal pinned HPI path now parses metadata/resource requirements, validates
   loader ABI 1.2.2 and NNRT 11.4.10 against the installed 1.3/11.13 implementation, resolves the
   public NPU40XX CMX metadata/workspace symbols, stages all allocated sections, applies static and
   binding relocations, and constructs the 384-byte HostParsedInference.
3. **Firmware ABI and negotiation.** Source command structures from a version-matched public Intel
   compiler/firmware interface. Define command headers, DPU/DMA/SHAVE descriptors, scratch and
   barrier records, and required firmware API versions. Do not infer these from captured traffic.
4. **Context and queue protocol.** Implicit-open context, bounded CMDQ create/destroy, the exact
   280-byte inference command, handle ordering, HPI descriptor, inference ID, NOP tail, explicit
   submit, and command-BO wait are proven for the tiny vector-add artifact. The test accounts
   90,112 bytes of user BOs plus the device-reported 28,389,376-byte preemption resource under a
   64 MiB ceiling.
5. **Memory and coherency.** Establish BO flag policy, cache synchronization, mapping lifetime,
   IOMMU address constraints, immutable executable memory, and a byte-budgeted resident-object
   eviction policy.
6. **Completion and recovery.** Absolute monotonic deadlines, command-BO wait, success-status
   decoding, and deterministic normal cleanup are proven. Add cancellation boundaries,
   device-loss handling, and reset recovery after partial execution.
7. **Redistributable artifact.** The local cached tiny vector-add artifact is proven, but its source,
   toolchain identity, and redistribution status must be recorded before it can be a portable CI
   fixture. A separately versioned tiny matmul remains useful for broader primitive coverage.
8. **Negative testing.** Fuzz ELF/envelope parsing, relocation overflow, binding validation, and
   firmware-version mismatch entirely without hardware before enabling any submit ioctl.

Until these are complete, `direct-ivpu` is a proven tiny-artifact execution path, not a general
alternative execution backend.

## First direct execution record

On 2026-07-11, Lunar Lake device `0x643e` revision 4 completed two serial submissions in explicit
queue 1. Firmware words matched Job API 4.15 and NNRT 11.13. The first submit-plus-wait observation
was 15.853 ms and the successful same-queue repeat was 0.212 ms; these are descriptive bring-up
observations, not benchmark results.

The 32 FP32 outputs were bit-identical between direct submissions and exactly matched the Level
Zero oracle (`max_absolute = 0`, `max_relative = 0`, configured tolerance `1e-5`). The test process
returned to 28 open FDs, RSS changed from 160,020 KiB to 160,212 KiB, swap stayed at zero, and the
driver's reported memory utilization returned to its 68,714,496-byte baseline. Kernel logs gained
no IVPU reset, fault, MMU, firmware, OOM, or hang entry; the device returned to D3hot.

## Direct queue overlap boundary

The default direct path remains synchronous by design. Graph-region planning now makes data,
mutable-state, and output dependencies explicit, but a prepared inference still owns one reusable
command BO and scheduler handoff ownership is returned at the end of each split. Submitting the
next dependent partition before command-BO completion would permit command reuse, output reads, or
external-memory ownership transfer while the device still owns the prior job.

Safe overlap requires a per-submission completion object, command-buffer slots that are not reused
until that completion, pinning resident programs and shared allocations through completion, and a
dependency DAG that can prove two regions share neither values nor mutable state. The current
driver targets do not expose an `uring_cmd` path, and host thread pools do not establish those
invariants. Until queue/event ownership supports them, direct submission intentionally performs
`submit -> finite command-BO wait -> output/ownership release`.

## Resident budget stability

The direct context snapshots its configured/host-derived resident byte budget before the first
program load. It does not recompute the limit from live `MemAvailable` while canonicalization,
artifact reads, page cache, and BO preparation are increasing process pressure. Recomputing had
allowed a claimed graph to become inadmissible during the same pass and caused cyclic eviction.
The tiered cache may consume this stable capacity input, but changing tiers must not make placement
or admission depend on transient pressure after a graph has been claimed.

## Current parity

- Discovery and identity: **direct proven**, Level Zero proven.
- Context lifetime: **direct proven through open/close**, Level Zero proven.
- Bounded host-visible memory: **direct proven for one 4 KiB BO**, Level Zero proven.
- Explicit queue lifecycle: **direct proven for one default queue**, Level Zero proven.
- Finite wait ABI: **direct proven on firmware work with a two-second absolute deadline**;
  Level Zero proven.
- ELF structure inspection and immutable binding/range validation: **direct software-proven**;
  Level Zero uses the pinned full HostParsedInference loader.
- Native load/materialization: **direct proven through bounded section BO staging, static linking,
  HPI creation, and binding-time relocation**, Level Zero proven through execution.
- Bindings: **direct proven for metadata count/size validation and two input/one output relocation
  patching**, Level Zero proven.
- Submit: **direct proven for two serial submissions of one tiny artifact in one explicit queue**;
  Level Zero proven.
- Wait/events: **direct command-BO completion and success status proven**; general event objects and
  cancellation remain unsupported. Level Zero proven.
- Cleanup/recovery: **direct deterministic cleanup proven on successful execution and injected
  allocation/device failures**, Level Zero cleanup proven; direct reset recovery remains unproven.
- Numerical parity: **direct vector-add output matched Level Zero bit-for-bit for all 32 FP32
  elements across both submissions**. Direct ShortConv/RoPE/FlashAttention/MoE remain unproven;
  Level Zero focused gates are proven.
- Existing tiny Intel artifact execution: **direct proven** for
  `ir2-8c91da4914adc7d5-n1.blob` (45,192 bytes, 43 sections, 33 symbols, 42 relocations,
  17 allocated sections totaling 8,096 logical bytes and 69,632 page-aligned BO bytes). Metadata
  reports two 128-byte inputs, one 128-byte output, four slices, no barriers, and a 2,432-byte
  mapped-inference entry. Structural and linked identities now use full domain-separated
  BLAKE3-256; three binding-time relocations were applied during the full dry run.
- Native program preparation and execution: **direct proven through command validation,
  deterministic cleanup, explicit submit, finite wait, and output readback** under an accounted
  28,479,488-byte ceiling. The observed direct output exactly matched the Level Zero oracle
  (`max_absolute = 0`, `max_relative = 0`, tolerance `1e-5`).
- General artifact caching, asynchronous events, cancellation, recovery after device loss, and
  broader numerical operation coverage remain unsupported.
