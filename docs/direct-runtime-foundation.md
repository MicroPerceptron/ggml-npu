# Direct runtime foundation

This milestone introduces an OS-neutral runtime contract and a Linux implementation without
replacing the working OpenVINO + Level Zero path. `rustix` supplies syscall mechanics only; it does
not define Intel firmware commands, graph formats, queue semantics, or compiler behavior.

## Implemented boundary

- `src/runtime/`: fallible lifecycle traits for device, memory, program, immutable bindings,
  queue, event, deadlines, unload, and destruction.
- `src/runtime/artifact.rs`: a versioned pointer-free artifact envelope containing a target
  fingerprint, native payload kind, binding ABI, payload length, and resident-byte requirement.
- `src/os/`: OS ownership, mapping, polling, and monotonic-time traits.
- `src/os/linux.rs`: `rustix` owned-FD, `open`, `mmap`/`munmap`, `poll`, and monotonic-clock
  implementation. Dynamic loading remains a separate compatibility concern.
- `src/vendor/ivpu/direct/`: feature-gated Linux UAPI subset for safe discovery and read-only
  parameter queries, plus bounded GEM BO create/info/map/close ownership wrappers.
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
- `DRM_IOCTL_IVPU_CMDQ_SUBMIT = 0x4020644d` (layout only; never invoked)
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

No command submit, graph load, native artifact execution, metric streaming, or NPU compute is
implemented or invoked. Hardware tests for parameter query, one 4 KiB BO lifecycle, and one
queue create/destroy are ignored by default.

Build the foundation with:

```sh
cargo check --features direct-ivpu
cargo test --features direct-ivpu
```

## Required before the first direct matmul

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
4. **Context and queue protocol.** Implicit-open context and bounded CMDQ create/destroy are proven.
   The exact 280-byte inference command, handle ordering, context-save address, HPI descriptor,
   inference ID, and NOP tail now pass the dry-run validator. Submission is blocked by the safety
   budget: both managed and legacy kernel paths call `ivpu_preemption_job_init`; with no user
   preemption BO they allocate the device-reported 28,389,376-byte preemption resource. The fully
   prepared user BO set is only 90,112 bytes, but the combined 28,479,488 bytes exceeds the
   mandated 8 MiB cap. No public UAPI switch disables that allocation.
5. **Memory and coherency.** Establish BO flag policy, cache synchronization, mapping lifetime,
   IOMMU address constraints, immutable executable memory, and a byte-budgeted resident-object
   eviction policy.
6. **Completion and recovery.** Add absolute monotonic deadlines, BO wait or CMDQ completion,
   status decoding, cancellation boundaries, device-loss handling, and guaranteed destruction
   after partial initialization.
7. **Tiny known-answer artifact.** Obtain or build a redistributable, version-pinned SHAVE program
   that performs a tiny operation (eventually a small matmul), record its source and toolchain
   hash, and test it with page-sized buffers and strict process/cgroup memory and time limits.
8. **Negative testing.** Fuzz ELF/envelope parsing, relocation overflow, binding validation, and
   firmware-version mismatch entirely without hardware before enabling any submit ioctl.

Until these are complete, `direct-ivpu` is discovery and resource-ownership scaffolding, not an
alternative execution backend.

## Current parity

- Discovery and identity: **direct proven**, Level Zero proven.
- Context lifetime: **direct proven through open/close**, Level Zero proven.
- Bounded host-visible memory: **direct proven for one 4 KiB BO**, Level Zero proven.
- Explicit queue lifecycle: **direct proven for one default queue**, Level Zero proven.
- Finite wait ABI: **direct implemented and software-tested, not tested on firmware work**;
  Level Zero proven.
- ELF structure inspection and immutable binding/range validation: **direct software-proven**;
  Level Zero uses the pinned full HostParsedInference loader.
- Native load/materialization: **direct proven through bounded section BO staging, static linking,
  HPI creation, and binding-time relocation**, Level Zero proven through execution.
- Bindings: **direct proven for metadata count/size validation and two input/one output relocation
  patching**, Level Zero proven.
- Submit: **direct blocked before ioctl by the 8 MiB preemption budget**, Level Zero proven.
- Wait/events: **direct BO-wait ABI implemented but no submitted event exists**, Level Zero proven.
- Cleanup/recovery: **direct deterministic cleanup proven on success, injected allocation/device
  failures, and the preemption-budget rejection**, Level Zero cleanup proven; direct reset recovery
  unproven.
- Numerical parity and ShortConv/RoPE/FlashAttention/MoE coverage: **direct unproven because no
  submission was permitted**, Level Zero focused gates proven.
- Existing tiny Intel artifact inspection: **direct proven without loading** for
  `ir2-8c91da4914adc7d5-n1.blob` (45,192 bytes, 43 sections, 33 symbols, 42 relocations,
  17 allocated sections totaling 8,096 logical bytes and 69,632 page-aligned BO bytes). Metadata
  reports two 128-byte inputs, one 128-byte output, four slices, no barriers, and a 2,432-byte
  mapped-inference entry. Structural and linked identities now use full domain-separated
  BLAKE3-256; three binding-time relocations were applied during the full dry run.
- Native program preparation: **direct proven through command validation and deterministic
  cleanup**, then intentionally rejected with
  `ExplicitQueuePreemptionExceedsBudget { user_bytes: 90112, preemption_bytes: 28389376 }`.
- Direct command submission and numerical parity: **not attempted** because the required
  preemption allocation violates the explicit safety budget.
- Relocation, native program load, submission, events, recovery after device loss, and numerical
  operations: **direct unsupported**; Level Zero currently proven by focused gates.
