//! Live scheduler telemetry: a shared-memory sink feeding the `os0-telemetry` ring.
//!
//! Env-gated (`GGML_NPU_TELEMETRY_SHM=<path>`), off by default — when unset, `push` is one relaxed
//! atomic load and returns. When set, the region is a `/dev/shm` mmap today and can move to the
//! `NPU_HOST` `zeMemAllocHost`/dmabuf buffer later (same host-visible bytes, same code). A separate
//! process (the wasmrt host's `host_get_telemetry`, an `nputop` TUI) maps the same region and reads
//! lock-free via [`telemetry_ring::Reader`]. The producer never blocks on the reader, so telemetry
//! cannot perturb the µs/dispatch hot path.
//!
//! Two seams feed the ring: [`TelemetrySink`] (execution-cost `MetricEvent`s) and the free
//! functions [`program`]/[`rejection`] called next to the existing `trace::perf` sites in `engine`
//! (placement decisions, which carry structured data the JSON path stringifies).

use core::sync::atomic::{AtomicPtr, Ordering};

use crate::metrics::{DispatchMetrics, MetricEvent, MetricsSink};
use crate::telemetry_ring::{region_bytes, Backend, Kind, Reason, Record, Writer};

/// The live region base, published once at init with an `AtomicPtr` (per the crate's
/// atomics-over-`OnceLock` convention). Null => telemetry disabled.
static BASE: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());
const CAP: usize = 1 << 12; // 4096 slots; region_bytes(CAP) ~= 256 KiB

/// Map `GGML_NPU_TELEMETRY_SHM` if set and publish the writer base. Call once from backend init,
/// after `crate::host::builtin()` is available. Idempotent.
pub fn init_from_env() {
    if !BASE.load(Ordering::Acquire).is_null() {
        return;
    }
    let cfg = crate::host::builtin().config();
    let Some(path) = cfg.value(c"GGML_NPU_TELEMETRY_SHM") else {
        return;
    };
    // NOTE: shown with rustix directly (already a dependency, mirrors host/linux/file.rs). To stay
    // inside the os/ host boundary, route this through `HostEnvironment::file()` + `MapOptions`.
    use rustix::fs::{ftruncate, open, Mode, OFlags};
    use rustix::mm::{mmap, MapFlags, ProtFlags};
    let bytes = region_bytes(CAP);
    let Ok(fd) = open(
        path,
        OFlags::CREATE | OFlags::RDWR,
        Mode::from_bits_truncate(0o600),
    ) else {
        return;
    };
    if ftruncate(&fd, bytes as u64).is_err() {
        return;
    }
    // SAFETY: fresh mapping of `bytes` writable shared bytes; the region outlives the process.
    let base = unsafe {
        mmap(
            core::ptr::null_mut(),
            bytes,
            ProtFlags::READ | ProtFlags::WRITE,
            MapFlags::SHARED,
            &fd,
            0,
        )
    };
    let Ok(base) = base else { return };
    let base = base.cast::<u8>();
    // SAFETY: `base` maps `region_bytes(CAP)` bytes; CAP is a power of two.
    unsafe { Writer::create(base, CAP) };
    BASE.store(base, Ordering::Release);
}

#[inline]
fn writer() -> Option<Writer> {
    let base = BASE.load(Ordering::Acquire);
    // SAFETY: once published, `base` is a valid initialized region for process lifetime.
    (!base.is_null()).then(|| unsafe { Writer::from_base(base, CAP) })
}

/// One subgraph placed on the NPU. Call beside `trace::perf("npu_program", …)`.
#[inline]
pub fn program(nodes: u32, reason: &str, op: u32, unitary: bool) {
    if let Some(w) = writer() {
        w.push(Record {
            kind: Kind::NpuProgram as u16,
            backend: Backend::Npu as u8,
            flags: unitary as u8,
            a: nodes,
            b: reason_code(reason) as u32,
            c: op,
            t_ns: 0,
            seq: 0,
        });
    }
}

/// A node the NPU declined; ggml routes it to `to`. Call beside `trace::perf("placement_rejection", …)`.
#[inline]
pub fn rejection(op: u32, reason: &str, to: Backend, wire: i32) {
    if let Some(w) = writer() {
        w.push(Record {
            kind: Kind::PlacementRejection as u16,
            backend: to as u8,
            flags: 0,
            a: op,
            b: reason_code(reason) as u32,
            c: wire as u32,
            t_ns: 0,
            seq: 0,
        });
    }
}

/// Map the engine's `&'static str` reasons to the stable ring code.
fn reason_code(r: &str) -> Reason {
    match r {
        "recognized_subgraph" => Reason::RecognizedSubgraph,
        "stateful_whole_graph" => Reason::StatefulWholeGraph,
        "per_node" => Reason::PerNode,
        "glue_fallback" => Reason::GlueFallback,
        "unsupported_shape" => Reason::UnsupportedShape,
        "not_offloadable" => Reason::NotOffloadable,
        "glue_below_threshold" => Reason::GlueBelowThreshold,
        _ => Reason::Other,
    }
}

/// Execution-cost sink: register alongside the existing `PROFILE_SINK` so one `MetricEvent::Dispatch`
/// per dispatch also lands in the ring (nanoseconds fit u32 comfortably at these magnitudes).
pub struct TelemetrySink;

impl MetricsSink for TelemetrySink {
    #[inline]
    fn event(&self, event: MetricEvent) {
        let Some(w) = writer() else { return };
        if let MetricEvent::Dispatch(DispatchMetrics {
            submit_ns,
            wait_ns,
            bytes_in,
            bytes_out,
            stage_in_ns,
            copy_out_ns,
            ..
        }) = event
        {
            w.push(Record {
                kind: Kind::Dispatch as u16,
                backend: Backend::Npu as u8,
                flags: 0,
                a: submit_ns.min(u32::MAX as u64) as u32,
                b: wait_ns.min(u32::MAX as u64) as u32,
                c: (bytes_in + bytes_out).min(u32::MAX as u64) as u32,
                t_ns: stage_in_ns + submit_ns + wait_ns + copy_out_ns,
                seq: 0,
            });
        }
    }
}
