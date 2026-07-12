# io_uring assessment for Intel IVPU

Assessment target:

- Fedora `kernel-devel-7.1.3-200.fc44.x86_64`
- Linux mainline snapshot `44696aa3a489d2baf58efa61b37833f100072bee`
- `drivers/accel/ivpu/ivpu_drv.c`

The IVPU driver uses `DRM_ACCEL_FOPS`. Fedora's installed `include/drm/drm_accel.h` expands that
macro to `open=accel_open`, `release=drm_release`, `unlocked_ioctl=drm_ioctl`,
`compat_ioctl=drm_compat_ioctl`, `poll=drm_poll`, `read=drm_read`, `mmap=drm_gem_mmap`, and
`llseek=noop_llseek`. There is no `.uring_cmd`, `.uring_cmd_iopoll`, or `.write`.

Linux 7.1 exposes `IORING_OP_URING_CMD` and `IORING_OP_URING_CMD128`, but these invoke a
provider-specific `file_operations::uring_cmd` callback. There is no generic io_uring ioctl opcode
in the installed UAPI. Consequently an IVPU fd cannot accept `IORING_OP_URING_CMD`.

The inherited DRM poll/read methods service the DRM event queue. IVPU job completion instead writes
the command BO's `job_status`, signals its DMA fence, and is consumed through
`DRM_IOCTL_IVPU_BO_WAIT`. The driver does not publish an IVPU completion event to `drm_poll`, so
`IORING_OP_POLL_ADD` is not a replacement for BO_WAIT either.

## Recommendation

Keep ioctl as the stable transport. io_uring around unrelated file operations cannot make NPU
submission asynchronous and would add ring-management overhead. Dispatch and firmware latency are
orders of magnitude larger than the raw syscall boundary in current traces; the useful work is:

- retain persistent contexts and command queues;
- batch/fuse graph work to reduce submission count;
- use immutable bindings and reusable command buffers;
- add a driver-supported completion event/fence fd when one becomes available;
- separate host submit time from device wait time through `MetricsSink`.

The optional `UringCommandServices` HAL extension is intentionally unimplemented. A future IVPU
driver can implement it only after adding `.uring_cmd` (and optionally `.uring_cmd_iopoll`), a
documented provider opcode, cancellation/lifetime rules, and asynchronous completion via
`io_uring_cmd_done`. No `io_uring` Rust dependency is currently justified.
