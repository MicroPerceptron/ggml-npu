//! Env-gated performance instrumentation for the NPU dispatch hot path ([`super::session::run`]).
//!
//! Every offloaded op — a single matmul or a fused subgraph — funnels through one `session::run`,
//! so timing its three phases (host stage-in of foreign inputs, the device submit+sync round-trip,
//! host copy-out of foreign outputs) plus the owned-vs-bounced argument split gives an evidence
//! breakdown of where a token's wall-clock actually goes: host memcpy, device latency, or
//! orchestration. Enabled by `GGML_NPU_PROFILE` (any non-empty value); the accumulated report is
//! written to stderr at process exit (via `atexit`), so it works for both the scratchpad harnesses
//! and a real `llama-cli` run.
//!
//! `no_std` + atomics-not-locks: all state is process-global atomics (compute is single-threaded per
//! backend, but the counters are lock-free regardless). When disabled, the hot path pays only one
//! relaxed atomic load per dispatch and takes no clock samples.

use core::ffi::{c_char, c_void};
use core::ptr;
use core::sync::atomic::{AtomicU64, AtomicU8, Ordering};

extern "C" {
    fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    // glibc routes `atexit` through `__cxa_atexit`; the plain `atexit` symbol is not reliably
    // resolvable from a dlopen'd .so, so register the exit dump the C++-runtime way. A null
    // `dso_handle` schedules the callback at process exit.
    fn __cxa_atexit(func: extern "C" fn(*mut c_void), arg: *mut c_void, dso: *mut c_void) -> i32;
    fn getenv(name: *const c_char) -> *const c_char;
}

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}
const CLOCK_MONOTONIC: i32 = 1;

/// Monotonic clock in nanoseconds (vDSO-fast). Only sampled when profiling is on.
pub(super) fn now_ns() -> u64 {
    let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
    if unsafe { clock_gettime(CLOCK_MONOTONIC, &mut ts) } != 0 {
        return 0;
    }
    (ts.tv_sec as u64).wrapping_mul(1_000_000_000).wrapping_add(ts.tv_nsec as u64)
}

// 0 = untried, 1 = off, 2 = on. Resolved once from the environment on the first dispatch.
static ENABLED: AtomicU8 = AtomicU8::new(0);

static N_DISPATCH: AtomicU64 = AtomicU64::new(0);
static T_STAGE_IN: AtomicU64 = AtomicU64::new(0); // ns, host copy of foreign inputs into bounces
static T_DEVICE: AtomicU64 = AtomicU64::new(0); // ns, submit + NPU compute + sync
static T_COPY_OUT: AtomicU64 = AtomicU64::new(0); // ns, host copy of foreign outputs out of bounces
static N_ARGS_OWNED: AtomicU64 = AtomicU64::new(0); // bound directly (zero copy)
static N_ARGS_BOUNCED: AtomicU64 = AtomicU64::new(0); // staged through a bounce buffer
static BYTES_IN: AtomicU64 = AtomicU64::new(0);
static BYTES_OUT: AtomicU64 = AtomicU64::new(0);

/// Is profiling on? Resolves `GGML_NPU_PROFILE` once and registers the exit dump on first enable.
pub(super) fn enabled() -> bool {
    let s = ENABLED.load(Ordering::Acquire);
    if s != 0 {
        return s == 2;
    }
    let on = unsafe {
        let p = getenv(c"GGML_NPU_PROFILE".as_ptr());
        !p.is_null() && *p != 0
    };
    let want = if on { 2 } else { 1 };
    // Only the thread that flips 0 -> want registers the exit dump (once).
    if ENABLED.compare_exchange(0, want, Ordering::AcqRel, Ordering::Acquire).is_ok() && on {
        unsafe { __cxa_atexit(dump_at_exit, ptr::null_mut(), ptr::null_mut()) };
    }
    on
}

/// Record one dispatch's phase timings (ns) and byte volumes. `t0..t3` are `now_ns` samples taken
/// around stage-in, the device round-trip, and copy-out.
#[allow(clippy::too_many_arguments)]
pub(super) fn record(
    t0: u64,
    t1: u64,
    t2: u64,
    t3: u64,
    owned: u64,
    bounced: u64,
    bytes_in: u64,
    bytes_out: u64,
) {
    N_DISPATCH.fetch_add(1, Ordering::Relaxed);
    T_STAGE_IN.fetch_add(t1.saturating_sub(t0), Ordering::Relaxed);
    T_DEVICE.fetch_add(t2.saturating_sub(t1), Ordering::Relaxed);
    T_COPY_OUT.fetch_add(t3.saturating_sub(t2), Ordering::Relaxed);
    N_ARGS_OWNED.fetch_add(owned, Ordering::Relaxed);
    N_ARGS_BOUNCED.fetch_add(bounced, Ordering::Relaxed);
    BYTES_IN.fetch_add(bytes_in, Ordering::Relaxed);
    BYTES_OUT.fetch_add(bytes_out, Ordering::Relaxed);
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
    let total = ti + td + to;
    let host = ti + to;
    let (owned, bounced) = (N_ARGS_OWNED.load(Ordering::Relaxed), N_ARGS_BOUNCED.load(Ordering::Relaxed));
    let (bin, bout) = (BYTES_IN.load(Ordering::Relaxed), BYTES_OUT.load(Ordering::Relaxed));
    let args = owned + bounced;

    let s = alloc::format!(
        "\n[ggml-npu profile]  dispatches={n}   measured host+device wall={:.3} ms ({:.3} us/disp)\n\
         \x20 stage-in (host->bounce) : {:>9.3} ms  {:>8.3} us/disp  {:>5.1}%   {:.2} MB\n\
         \x20 device  (submit+sync)   : {:>9.3} ms  {:>8.3} us/disp  {:>5.1}%\n\
         \x20 copy-out (bounce->host) : {:>9.3} ms  {:>8.3} us/disp  {:>5.1}%   {:.2} MB\n\
         \x20 ---- host copy total    : {:>9.3} ms  ({:.1}%)     device total: {:.3} ms ({:.1}%)\n\
         \x20 args: {owned} owned (zero-copy) / {bounced} bounced  ({:.1}% zero-copy of {args})\n",
        ms(total),
        us_per(total, n),
        ms(ti),
        us_per(ti, n),
        pct(ti, total),
        bin as f64 / 1.048576e6,
        ms(td),
        us_per(td, n),
        pct(td, total),
        ms(to),
        us_per(to, n),
        pct(to, total),
        bout as f64 / 1.048576e6,
        ms(host),
        pct(host, total),
        ms(td),
        pct(td, total),
        pct(owned, args.max(1)),
    );
    unsafe { write(2, s.as_ptr(), s.len()) };
}
