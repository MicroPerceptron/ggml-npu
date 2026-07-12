//! Env-gated runtime diagnostics for submit/wait/copy crash and placement investigations.
//!
//! Every offloaded op — a single matmul or a fused subgraph — funnels through one `session::run`,
//! so timing its three phases (host stage-in of foreign inputs, the device submit+sync round-trip,
//! host copy-out of foreign outputs) plus the owned-vs-bounced argument split gives an evidence
//! breakdown of where a token's wall-clock went: host memcpy, device latency, or
//! orchestration. This is not a benchmark statistics engine; Criterion and llama-bench own those
//! roles. Enabled by `GGML_NPU_PROFILE` (any non-empty value); the accumulated report is
//! written to stderr at process exit (via `atexit`), so it works for both the scratchpad harnesses
//! and a real `llama-cli` run.
//!
//! `no_std` + atomics-not-locks: all state is process-global atomics (compute is single-threaded per
//! backend, but the counters are lock-free regardless). When disabled, the hot path pays only one
//! relaxed atomic load per dispatch and takes no clock samples.

use core::ffi::c_void;
use core::ptr;
use core::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use crate::host::{ClockService, ConfigService, DiagnosticService, HostEnvironment};
use crate::metrics::{DispatchMetrics, MetricEvent, MetricsSink};

/// Monotonic clock in nanoseconds (vDSO-fast). Only sampled when profiling is on.
pub(super) fn now_ns() -> u64 {
    crate::host::builtin()
        .clock()
        .monotonic_now_ns()
        .unwrap_or(0)
}

// 0 = untried, 1 = off, 2 = on. Resolved once from the environment on the first dispatch.
static ENABLED: AtomicU8 = AtomicU8::new(0);
static DEBUG: AtomicU8 = AtomicU8::new(0);

/// Is `GGML_NPU_DEBUG` set? (Cached.) Gates the failure-path diagnostics.
pub(super) fn debug_on() -> bool {
    let s = DEBUG.load(Ordering::Acquire);
    if s != 0 {
        return s == 2;
    }
    let on = crate::host::builtin()
        .config()
        .is_nonempty(c"GGML_NPU_DEBUG");
    DEBUG.store(if on { 2 } else { 1 }, Ordering::Release);
    on
}

/// Write `s` to stderr when `GGML_NPU_DEBUG` is set (a `no_std` eprintln for failure diagnostics).
pub(super) fn note(s: &str) {
    if debug_on() {
        crate::host::builtin()
            .diagnostics()
            .write_stderr(s.as_bytes());
    }
}

static N_DISPATCH: AtomicU64 = AtomicU64::new(0);
static T_STAGE_IN: AtomicU64 = AtomicU64::new(0); // ns, host copy of foreign inputs into bounces
static T_DEVICE: AtomicU64 = AtomicU64::new(0); // ns, submit + NPU compute + sync (= submit + sync)
static T_SUBMIT: AtomicU64 = AtomicU64::new(0); // ns, cmd_queue_exec — async submit (host cost only)
static T_SYNC: AtomicU64 = AtomicU64::new(0); // ns, cmd_queue_sync — wait for the NPU to drain
static T_COPY_OUT: AtomicU64 = AtomicU64::new(0); // ns, host copy of foreign outputs out of bounces
static N_DIRECT_BINDINGS: AtomicU64 = AtomicU64::new(0);
static N_SHARED_BINDINGS: AtomicU64 = AtomicU64::new(0);
static N_IMPORTED_BINDINGS: AtomicU64 = AtomicU64::new(0);
static N_STAGED_OCCURRENCES: AtomicU64 = AtomicU64::new(0);
static N_STAGED_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static STAGED_BYTES: AtomicU64 = AtomicU64::new(0);
static STAGED_ALLOCATION_BYTES: AtomicU64 = AtomicU64::new(0);
static BYTES_IN: AtomicU64 = AtomicU64::new(0);
static BYTES_OUT: AtomicU64 = AtomicU64::new(0);
static RESIDENT_BYTES: AtomicU64 = AtomicU64::new(0);
static RESIDENT_HIGH_WATER: AtomicU64 = AtomicU64::new(0);
static ACCOUNTED_HIGH_WATER: AtomicU64 = AtomicU64::new(0);
static RESIDENT_LIMIT: AtomicU64 = AtomicU64::new(0);
static EVICTED_BYTES: AtomicU64 = AtomicU64::new(0);
static EXECUTABLE_BYTES: AtomicU64 = AtomicU64::new(0);
static BINDING_BYTES: AtomicU64 = AtomicU64::new(0);
static HPI_BYTES: AtomicU64 = AtomicU64::new(0);
static COMMAND_BYTES: AtomicU64 = AtomicU64::new(0);
static PREEMPTION_BYTES: AtomicU64 = AtomicU64::new(0);

/// Is profiling on? Resolves `GGML_NPU_PROFILE` once and registers the exit dump on first enable.
pub(super) fn enabled() -> bool {
    let s = ENABLED.load(Ordering::Acquire);
    if s != 0 {
        return s == 2;
    }
    let on = crate::host::builtin()
        .config()
        .is_nonempty(c"GGML_NPU_PROFILE");
    let want = if on { 2 } else { 1 };
    // Only the thread that flips 0 -> want registers the exit dump (once).
    if ENABLED
        .compare_exchange(0, want, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
        && on
    {
        crate::host::builtin()
            .diagnostics()
            .register_process_exit(dump_at_exit, ptr::null_mut());
    }
    on
}

/// Record one dispatch's phase timings (ns) and byte volumes. `t0..t3` are `now_ns` samples taken
/// around stage-in, the device round-trip, and copy-out.
/// `t0..t3` are `now_ns` samples: t0 before stage-in, t1 after stage-in (= before submit), `tmid`
/// after the async submit returns (= before sync), t2 after sync, t3 after copy-out.
#[allow(clippy::too_many_arguments)]
pub(super) fn record(
    t0: u64,
    t1: u64,
    tmid: u64,
    t2: u64,
    t3: u64,
    direct: u64,
    shared: u64,
    imported: u64,
    staged_occurrences: u64,
    staged_bytes: u64,
    bytes_in: u64,
    bytes_out: u64,
) {
    PROFILE_SINK.event(MetricEvent::Dispatch(DispatchMetrics {
        stage_in_ns: t1.saturating_sub(t0),
        submit_ns: tmid.saturating_sub(t1),
        wait_ns: t2.saturating_sub(tmid),
        copy_out_ns: t3.saturating_sub(t2),
        direct_bindings: direct,
        shared_bindings: shared,
        imported_bindings: imported,
        staged_occurrences,
        staged_bytes,
        bytes_in,
        bytes_out,
    }));
}

pub(super) fn event(event: MetricEvent) {
    if enabled() {
        PROFILE_SINK.event(event);
    }
}

pub(super) fn record_direct_memory_breakdown(
    executable_bytes: u64,
    binding_bytes: u64,
    hpi_bytes: u64,
    command_bytes: u64,
    preemption_bytes: u64,
) {
    if enabled() {
        EXECUTABLE_BYTES.store(executable_bytes, Ordering::Relaxed);
        BINDING_BYTES.store(binding_bytes, Ordering::Relaxed);
        HPI_BYTES.store(hpi_bytes, Ordering::Relaxed);
        COMMAND_BYTES.store(command_bytes, Ordering::Relaxed);
        PREEMPTION_BYTES.store(preemption_bytes, Ordering::Relaxed);
    }
}

struct ProfileSink;
static PROFILE_SINK: ProfileSink = ProfileSink;

impl MetricsSink for ProfileSink {
    fn event(&self, event: MetricEvent) {
        match event {
            MetricEvent::Dispatch(metrics) => {
                N_DISPATCH.fetch_add(1, Ordering::Relaxed);
                T_STAGE_IN.fetch_add(metrics.stage_in_ns, Ordering::Relaxed);
                T_SUBMIT.fetch_add(metrics.submit_ns, Ordering::Relaxed);
                T_SYNC.fetch_add(metrics.wait_ns, Ordering::Relaxed);
                T_DEVICE.fetch_add(
                    metrics.submit_ns.saturating_add(metrics.wait_ns),
                    Ordering::Relaxed,
                );
                T_COPY_OUT.fetch_add(metrics.copy_out_ns, Ordering::Relaxed);
                N_DIRECT_BINDINGS.fetch_add(metrics.direct_bindings, Ordering::Relaxed);
                N_SHARED_BINDINGS.fetch_add(metrics.shared_bindings, Ordering::Relaxed);
                N_IMPORTED_BINDINGS.fetch_add(metrics.imported_bindings, Ordering::Relaxed);
                N_STAGED_OCCURRENCES.fetch_add(metrics.staged_occurrences, Ordering::Relaxed);
                STAGED_BYTES.fetch_add(metrics.staged_bytes, Ordering::Relaxed);
                BYTES_IN.fetch_add(metrics.bytes_in, Ordering::Relaxed);
                BYTES_OUT.fetch_add(metrics.bytes_out, Ordering::Relaxed);
            }
            MetricEvent::StagingAllocation { bytes, .. } => {
                N_STAGED_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                STAGED_ALLOCATION_BYTES.fetch_add(bytes, Ordering::Relaxed);
            }
            MetricEvent::MemoryState {
                resident_bytes,
                resident_high_water,
                accounted_high_water,
                resident_limit,
            } => {
                RESIDENT_BYTES.store(resident_bytes, Ordering::Relaxed);
                RESIDENT_HIGH_WATER.fetch_max(resident_high_water, Ordering::Relaxed);
                ACCOUNTED_HIGH_WATER.fetch_max(accounted_high_water, Ordering::Relaxed);
                RESIDENT_LIMIT.store(resident_limit, Ordering::Relaxed);
            }
            MetricEvent::CacheEviction { bytes, .. } => {
                EVICTED_BYTES.fetch_add(bytes, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1.0e6
}
fn us_per(ns: u64, n: u64) -> f64 {
    if n == 0 {
        0.0
    } else {
        ns as f64 / n as f64 / 1.0e3
    }
}
fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

extern "C" fn dump_at_exit(_arg: *mut c_void) {
    let n = N_DISPATCH.load(Ordering::Relaxed);
    if n == 0 {
        return;
    }
    let (ti, td, to) = (
        T_STAGE_IN.load(Ordering::Relaxed),
        T_DEVICE.load(Ordering::Relaxed),
        T_COPY_OUT.load(Ordering::Relaxed),
    );
    let (tsub, tsync) = (
        T_SUBMIT.load(Ordering::Relaxed),
        T_SYNC.load(Ordering::Relaxed),
    );
    let total = ti + td + to;
    let host = ti + to;
    let direct = N_DIRECT_BINDINGS.load(Ordering::Relaxed);
    let shared = N_SHARED_BINDINGS.load(Ordering::Relaxed);
    let imported = N_IMPORTED_BINDINGS.load(Ordering::Relaxed);
    let staged = N_STAGED_OCCURRENCES.load(Ordering::Relaxed);
    let staged_allocations = N_STAGED_ALLOCATIONS.load(Ordering::Relaxed);
    let staged_bytes = STAGED_BYTES.load(Ordering::Relaxed);
    let staged_allocation_bytes = STAGED_ALLOCATION_BYTES.load(Ordering::Relaxed);
    let (bin, bout) = (
        BYTES_IN.load(Ordering::Relaxed),
        BYTES_OUT.load(Ordering::Relaxed),
    );
    let bindings = direct + shared + imported + staged;

    let s = alloc::format!(
        "\n[ggml-npu profile]  dispatches={n}   measured host+device wall={:.3} ms ({:.3} us/disp)\n\
         \x20 stage-in (host->bounce) : {:>9.3} ms  {:>8.3} us/disp  {:>5.1}%   {:.2} MB\n\
         \x20 device  (submit+sync)   : {:>9.3} ms  {:>8.3} us/disp  {:>5.1}%\n\
         \x20   |- submit (exec async): {:>9.3} ms  {:>8.3} us/disp  ({:.1}% of device)\n\
         \x20   \\- sync   (wait NPU)  : {:>9.3} ms  {:>8.3} us/disp  ({:.1}% of device)\n\
         \x20 copy-out (bounce->host) : {:>9.3} ms  {:>8.3} us/disp  {:>5.1}%   {:.2} MB\n\
         \x20 ---- host copy total    : {:>9.3} ms  ({:.1}%)     device total: {:.3} ms ({:.1}%)\n\
         \x20 bindings: {direct} direct / {shared} shared / {imported} imported / {staged} staged occurrences ({:.1}% no-copy of {bindings})\n\
         \x20 bounce allocations: {staged_allocations} unique, {:.2} MiB allocated; {:.2} MiB copied across occurrences\n\
         \x20 direct memory: resident {:.2} MiB, resident high {:.2} MiB, accounted high {:.2} MiB, limit {:.2} MiB, evicted {:.2} MiB\n\
         \x20 latest program: executable {:.2} MiB, bindings {:.2} MiB, HPI {:.2} MiB, command {:.2} MiB, preemption {:.2} MiB\n",
        ms(total),
        us_per(total, n),
        ms(ti),
        us_per(ti, n),
        pct(ti, total),
        bin as f64 / 1.048576e6,
        ms(td),
        us_per(td, n),
        pct(td, total),
        ms(tsub),
        us_per(tsub, n),
        pct(tsub, td),
        ms(tsync),
        us_per(tsync, n),
        pct(tsync, td),
        ms(to),
        us_per(to, n),
        pct(to, total),
        bout as f64 / 1.048576e6,
        ms(host),
        pct(host, total),
        ms(td),
        pct(td, total),
        pct(direct + shared + imported, bindings.max(1)),
        staged_allocation_bytes as f64 / 1.048576e6,
        staged_bytes as f64 / 1.048576e6,
        RESIDENT_BYTES.load(Ordering::Relaxed) as f64 / 1.048576e6,
        RESIDENT_HIGH_WATER.load(Ordering::Relaxed) as f64 / 1.048576e6,
        ACCOUNTED_HIGH_WATER.load(Ordering::Relaxed) as f64 / 1.048576e6,
        RESIDENT_LIMIT.load(Ordering::Relaxed) as f64 / 1.048576e6,
        EVICTED_BYTES.load(Ordering::Relaxed) as f64 / 1.048576e6,
        EXECUTABLE_BYTES.load(Ordering::Relaxed) as f64 / 1.048576e6,
        BINDING_BYTES.load(Ordering::Relaxed) as f64 / 1.048576e6,
        HPI_BYTES.load(Ordering::Relaxed) as f64 / 1.048576e6,
        COMMAND_BYTES.load(Ordering::Relaxed) as f64 / 1.048576e6,
        PREEMPTION_BYTES.load(Ordering::Relaxed) as f64 / 1.048576e6,
    );
    crate::host::builtin()
        .diagnostics()
        .write_stderr(s.as_bytes());
}
