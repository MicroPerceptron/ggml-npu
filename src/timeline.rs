//! Env-gated hetero forward-pass timeline (busy/idle + memory loads).
//!
//! Enable with `GGML_NPU_TIMELINE=<path_prefix>` (non-empty). When unset, every mark is one
//! relaxed atomic load and returns. On process exit the recorder dumps:
//! - `<prefix>.jsonl` — one event per line
//! - `<prefix>.trace.json` — Chrome Trace Event Format
//! - `<prefix>.summary.txt` — busy%/idle%/copy/load per domain
//! and prints the same summary on stderr.
//!
//! Host (llama.cpp) probes resolve `ggml_npu_timeline_*` via `get_proc_address`. Direct / L0
//! sessions attribute host stage/zero/submit/copy-out to CPU and device wait to NPU. Meta TP
//! backends use domain 3 so fence reclaim is not labeled CPU math. Does not change claim/buft
//! policy.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_void, CStr};
use core::ptr;
use core::sync::atomic::{AtomicPtr, AtomicU64, AtomicU8, Ordering};

use crate::host::{ClockService, ConfigService, DiagnosticService, HostEnvironment};
use crate::timeline_ring::{region_bytes, Domain, Event, Kind, Reader, Writer};

const CAP: usize = 1 << 14; // 16384 slots ≈ 1 MiB
static BASE: AtomicPtr<u8> = AtomicPtr::new(ptr::null_mut());
static PATH: AtomicPtr<u8> = AtomicPtr::new(ptr::null_mut());
static ENABLED: AtomicU8 = AtomicU8::new(0); // 0 untried, 1 off, 2 on
static LAPPED: AtomicU64 = AtomicU64::new(0);

/// Monotonic clock (ns). Shared with host probes so CPU/GPU/NPU share one timeline.
#[inline]
pub fn now_ns() -> u64 {
    crate::host::builtin()
        .clock()
        .monotonic_now_ns()
        .unwrap_or(0)
}

#[inline]
pub fn enabled() -> bool {
    if ENABLED.load(Ordering::Acquire) == 0 {
        init_from_env();
    }
    ENABLED.load(Ordering::Acquire) == 2
}

/// Map `GGML_NPU_TIMELINE` once and publish the writer. Idempotent; safe before any mark.
pub fn init_from_env() {
    if ENABLED.load(Ordering::Acquire) != 0 {
        return;
    }
    let cfg = crate::host::builtin().config();
    let Some(path) = cfg.value(c"GGML_NPU_TIMELINE") else {
        ENABLED.store(1, Ordering::Release);
        return;
    };
    if path.to_bytes().is_empty() {
        ENABLED.store(1, Ordering::Release);
        return;
    }
    let bytes = region_bytes(CAP);
    // SAFETY: process-owned anonymous mapping; lives for process lifetime.
    let base = unsafe {
        rustix::mm::mmap_anonymous(
            ptr::null_mut(),
            bytes,
            rustix::mm::ProtFlags::READ | rustix::mm::ProtFlags::WRITE,
            rustix::mm::MapFlags::PRIVATE,
        )
    };
    let Ok(base) = base else {
        ENABLED.store(1, Ordering::Release);
        return;
    };
    let base = base.cast::<u8>();
    unsafe { Writer::create(base, CAP) };
    // Keep the path CStr bytes alive (env values are static; store the pointer for dump).
    PATH.store(path.as_ptr() as *mut u8, Ordering::Release);
    BASE.store(base, Ordering::Release);
    if ENABLED
        .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        crate::host::builtin()
            .diagnostics()
            .register_process_exit(dump_at_exit, ptr::null_mut());
    } else {
        // Lost race: region still fine; another thread already registered exit.
        ENABLED.store(2, Ordering::Release);
    }
}

#[inline]
fn writer() -> Option<Writer> {
    let base = BASE.load(Ordering::Acquire);
    (!base.is_null()).then(|| unsafe { Writer::from_base(base, CAP) })
}

/// Record a complete span `[t0, t1)` on `domain`.
#[inline]
pub fn span(domain: Domain, kind: Kind, t0: u64, t1: u64, bytes: u64) {
    if !enabled() {
        return;
    }
    let Some(w) = writer() else { return };
    w.push(Event {
        ts_ns: t0,
        dur_ns: t1.saturating_sub(t0),
        bytes,
        domain: domain as u8,
        kind: kind as u8,
        flags: 0,
        _pad: 0,
        seq: 0,
    });
}

/// Instant marker (zero duration), typically a memory load.
#[inline]
pub fn instant(domain: Domain, kind: Kind, bytes: u64) {
    if !enabled() {
        return;
    }
    let Some(w) = writer() else { return };
    w.push(Event {
        ts_ns: now_ns(),
        dur_ns: 0,
        bytes,
        domain: domain as u8,
        kind: kind as u8,
        flags: 0,
        _pad: 0,
        seq: 0,
    });
}

fn domain_name(d: u8) -> &'static str {
    match d {
        x if x == Domain::Cpu as u8 => "cpu",
        x if x == Domain::Gpu as u8 => "gpu",
        x if x == Domain::Npu as u8 => "npu",
        x if x == Domain::Meta as u8 => "meta",
        _ => "unknown",
    }
}

/// Domains reported in the summary / chrome lanes (CPU · GPU · NPU · Meta).
const DOMAIN_COUNT: usize = 4;
const DOMAIN_NAMES: [&str; DOMAIN_COUNT] = ["cpu", "gpu", "npu", "meta"];

fn kind_name(k: u8) -> &'static str {
    match k {
        x if x == Kind::Forward as u8 => "forward",
        x if x == Kind::GraphCompute as u8 => "graph_compute",
        x if x == Kind::StageIn as u8 => "stage_in",
        x if x == Kind::Submit as u8 => "submit",
        x if x == Kind::Wait as u8 => "wait",
        x if x == Kind::CopyOut as u8 => "copy_out",
        x if x == Kind::SchedCopy as u8 => "sched_copy",
        x if x == Kind::Synchronize as u8 => "synchronize",
        x if x == Kind::MemLoad as u8 => "mem_load",
        x if x == Kind::NpuCompute as u8 => "npu_compute",
        x if x == Kind::NpuDma as u8 => "npu_dma",
        x if x == Kind::NpuSw as u8 => "npu_sw",
        _ => "other",
    }
}

/// Busy kinds for merged busy%/idle%. Host staging copies count on the domain that owns
/// them (CPU after host-prep attribution); `forward` is a window marker only and is excluded
/// so the outer sched span does not force CPU busy% ≈ 100.
fn is_busy_kind(k: u8) -> bool {
    matches!(
        k,
        x if x == Kind::GraphCompute as u8
            || x == Kind::Submit as u8
            || x == Kind::Wait as u8
            || x == Kind::Synchronize as u8
            || x == Kind::StageIn as u8
            || x == Kind::CopyOut as u8
            || x == Kind::SchedCopy as u8
    )
}

fn is_copy_kind(k: u8) -> bool {
    matches!(
        k,
        x if x == Kind::StageIn as u8
            || x == Kind::CopyOut as u8
            || x == Kind::SchedCopy as u8
    )
}

fn is_load_kind(k: u8) -> bool {
    k == Kind::MemLoad as u8 || is_copy_kind(k)
}

#[derive(Clone, Copy, Default)]
struct DomainStats {
    busy_ns: u64,
    copy_ns: u64,
    load_bytes: u64,
    // Device-reported per-engine busy times (Level Zero graph profiling). These overlap each
    // other and the host Wait window, so they are accumulated separately rather than merged into
    // busy%; they answer "DMA-bound vs DPU-bound", not a partition of wall-clock.
    compute_ns: u64,
    dma_ns: u64,
    sw_ns: u64,
}

fn merge_busy(intervals: &mut Vec<(u64, u64)>) -> u64 {
    if intervals.is_empty() {
        return 0;
    }
    intervals.sort_by_key(|i| i.0);
    let mut total = 0u64;
    let (mut cur_s, mut cur_e) = intervals[0];
    for &(s, e) in intervals.iter().skip(1) {
        if s <= cur_e {
            cur_e = cur_e.max(e);
        } else {
            total = total.saturating_add(cur_e.saturating_sub(cur_s));
            cur_s = s;
            cur_e = e;
        }
    }
    total.saturating_add(cur_e.saturating_sub(cur_s))
}

fn summarize(events: &[Event], lapped: u64) -> String {
    let mut window_lo = u64::MAX;
    let mut window_hi = 0u64;
    let mut busy_iv: [Vec<(u64, u64)>; DOMAIN_COUNT] = Default::default();
    let mut stats = [DomainStats::default(); DOMAIN_COUNT];

    for e in events {
        let end = e.ts_ns.saturating_add(e.dur_ns);
        if e.ts_ns < window_lo {
            window_lo = e.ts_ns;
        }
        if end > window_hi {
            window_hi = end;
        }
        let di = e.domain as usize;
        if di >= DOMAIN_COUNT {
            continue;
        }
        if is_busy_kind(e.kind) && e.dur_ns > 0 {
            busy_iv[di].push((e.ts_ns, end));
        }
        if is_copy_kind(e.kind) {
            stats[di].copy_ns = stats[di].copy_ns.saturating_add(e.dur_ns);
        }
        if is_load_kind(e.kind) {
            stats[di].load_bytes = stats[di].load_bytes.saturating_add(e.bytes);
        }
        if e.kind == Kind::NpuCompute as u8 {
            stats[di].compute_ns = stats[di].compute_ns.saturating_add(e.dur_ns);
        } else if e.kind == Kind::NpuDma as u8 {
            stats[di].dma_ns = stats[di].dma_ns.saturating_add(e.dur_ns);
        } else if e.kind == Kind::NpuSw as u8 {
            stats[di].sw_ns = stats[di].sw_ns.saturating_add(e.dur_ns);
        }
    }

    let window = if window_lo == u64::MAX {
        0
    } else {
        window_hi.saturating_sub(window_lo)
    };
    for (i, iv) in busy_iv.iter_mut().enumerate() {
        stats[i].busy_ns = merge_busy(iv);
    }

    let pct = |part: u64, whole: u64| -> f64 {
        if whole == 0 {
            0.0
        } else {
            (part as f64) * 100.0 / (whole as f64)
        }
    };
    let ms = |ns: u64| -> f64 { ns as f64 / 1_000_000.0 };

    let mut out = format!(
        "hetero timeline: window_ms={:.3} events={} lapped={}\n",
        ms(window),
        events.len(),
        lapped
    );
    for (i, name) in DOMAIN_NAMES.iter().enumerate() {
        let busy = stats[i].busy_ns.min(window);
        let idle = window.saturating_sub(busy);
        out.push_str(&format!(
            "  {name}: busy%={:.1} idle%={:.1} busy_ms={:.3} idle_ms={:.3} copy_ms={:.3} load_bytes={} compute_ms={:.3} dma_ms={:.3} sw_ms={:.3}\n",
            pct(busy, window),
            pct(idle, window),
            ms(busy),
            ms(idle),
            ms(stats[i].copy_ns),
            stats[i].load_bytes,
            ms(stats[i].compute_ns),
            ms(stats[i].dma_ns),
            ms(stats[i].sw_ns)
        ));
    }
    out
}

fn write_file(path: &CStr, contents: &[u8]) -> bool {
    use rustix::fs::{open, Mode, OFlags};
    use rustix::io::write;
    let Ok(fd) = open(
        path,
        OFlags::CREATE | OFlags::TRUNC | OFlags::WRONLY | OFlags::CLOEXEC,
        Mode::from_bits_truncate(0o644),
    ) else {
        return false;
    };
    let mut off = 0;
    while off < contents.len() {
        match write(&fd, &contents[off..]) {
            Ok(0) => return false,
            Ok(n) => off += n,
            Err(_) => return false,
        }
    }
    true
}

fn path_with_suffix(base: &CStr, suffix: &str) -> Option<Vec<u8>> {
    let mut v = Vec::with_capacity(base.to_bytes().len() + suffix.len() + 1);
    v.extend_from_slice(base.to_bytes());
    v.extend_from_slice(suffix.as_bytes());
    v.push(0);
    Some(v)
}

fn drain_all() -> (Vec<Event>, u64) {
    let base = BASE.load(Ordering::Acquire);
    if base.is_null() {
        return (Vec::new(), 0);
    }
    let mut reader = unsafe { Reader::from_base(base, CAP) };
    let mut events = Vec::new();
    let mut buf = [Event::default(); 256];
    let mut lapped = 0u64;
    loop {
        let stats = reader.drain(&mut buf);
        lapped = lapped.saturating_add(stats.lapped);
        if stats.drained == 0 {
            break;
        }
        events.extend_from_slice(&buf[..stats.drained as usize]);
    }
    LAPPED.fetch_add(lapped, Ordering::Relaxed);
    (events, LAPPED.load(Ordering::Relaxed))
}

fn dump_jsonl(events: &[Event]) -> String {
    let mut out = String::new();
    for e in events {
        out.push_str(&format!(
            "{{\"ts_ns\":{},\"dur_ns\":{},\"domain\":\"{}\",\"kind\":\"{}\",\"bytes\":{},\"seq\":{}}}\n",
            e.ts_ns,
            e.dur_ns,
            domain_name(e.domain),
            kind_name(e.kind),
            e.bytes,
            e.seq
        ));
    }
    out
}

fn dump_chrome(events: &[Event]) -> String {
    let mut out = String::from("{\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n");
    let mut first = true;
    // Normalize so the first event is near t=0 for chrome://tracing.
    let origin = events.first().map(|e| e.ts_ns).unwrap_or(0);
    for e in events {
        if !first {
            out.push_str(",\n");
        }
        first = false;
        let ts_us = (e.ts_ns.saturating_sub(origin)) as f64 / 1000.0;
        let dur_us = e.dur_ns as f64 / 1000.0;
        let tid = e.domain as u32 + 1;
        let ph = if e.dur_ns == 0 { "i" } else { "X" };
        out.push_str(&format!(
            "{{\"name\":\"{}\",\"cat\":\"{}\",\"ph\":\"{ph}\",\"ts\":{ts_us:.3},\"dur\":{dur_us:.3},\"pid\":1,\"tid\":{tid},\"args\":{{\"bytes\":{}}}}}",
            kind_name(e.kind),
            domain_name(e.domain),
            e.bytes
        ));
    }
    out.push_str("\n]}\n");
    out
}

extern "C" fn dump_at_exit(_arg: *mut c_void) {
    let path_ptr = PATH.load(Ordering::Acquire);
    if path_ptr.is_null() {
        return;
    }
    let path = unsafe { CStr::from_ptr(path_ptr as *const c_char) };
    let (events, lapped) = drain_all();
    let summary = summarize(&events, lapped);
    crate::host::builtin()
        .diagnostics()
        .write_stderr(summary.as_bytes());

    if let Some(p) = path_with_suffix(path, ".summary.txt") {
        let c = CStr::from_bytes_with_nul(&p).ok();
        if let Some(c) = c {
            let _ = write_file(c, summary.as_bytes());
        }
    }
    if let Some(p) = path_with_suffix(path, ".jsonl") {
        let c = CStr::from_bytes_with_nul(&p).ok();
        if let Some(c) = c {
            let body = dump_jsonl(&events);
            let _ = write_file(c, body.as_bytes());
        }
    }
    if let Some(p) = path_with_suffix(path, ".trace.json") {
        let c = CStr::from_bytes_with_nul(&p).ok();
        if let Some(c) = c {
            let body = dump_chrome(&events);
            let _ = write_file(c, body.as_bytes());
        }
    }
}

// ---------------------------------------------------------------------------
// C ABI (resolved via registry get_proc_address)
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn ggml_npu_timeline_enabled() -> c_int {
    init_from_env();
    if enabled() {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_npu_timeline_now_ns() -> u64 {
    now_ns()
}

fn domain_from_c(domain: c_int) -> Option<Domain> {
    match domain {
        0 => Some(Domain::Cpu),
        1 => Some(Domain::Gpu),
        2 => Some(Domain::Npu),
        3 => Some(Domain::Meta),
        _ => None,
    }
}

/// `domain`: 0=cpu 1=gpu 2=npu 3=meta · `kind`: see [`Kind`] · span `[t0_ns, t1_ns)`.
#[no_mangle]
pub unsafe extern "C" fn ggml_npu_timeline_span(
    domain: c_int,
    kind: c_int,
    t0_ns: u64,
    t1_ns: u64,
    bytes: u64,
) {
    if !enabled() {
        // Host may call before our init when probing; resolve env once.
        init_from_env();
        if !enabled() {
            return;
        }
    }
    let Some(d) = domain_from_c(domain) else {
        return;
    };
    let k = match kind {
        1 => Kind::Forward,
        2 => Kind::GraphCompute,
        3 => Kind::StageIn,
        4 => Kind::Submit,
        5 => Kind::Wait,
        6 => Kind::CopyOut,
        7 => Kind::SchedCopy,
        8 => Kind::Synchronize,
        9 => Kind::MemLoad,
        10 => Kind::NpuCompute,
        11 => Kind::NpuDma,
        12 => Kind::NpuSw,
        _ => return,
    };
    span(d, k, t0_ns, t1_ns, bytes);
}

/// Instant event (memory load / marker).
#[no_mangle]
pub unsafe extern "C" fn ggml_npu_timeline_instant(domain: c_int, kind: c_int, bytes: u64) {
    if !enabled() {
        init_from_env();
        if !enabled() {
            return;
        }
    }
    let Some(d) = domain_from_c(domain) else {
        return;
    };
    let k = match kind {
        1 => Kind::Forward,
        9 => Kind::MemLoad,
        7 => Kind::SchedCopy,
        _ => Kind::MemLoad,
    };
    instant(d, k, bytes);
}

pub(crate) unsafe fn proc_address(name: *const c_char) -> *mut c_void {
    if name.is_null() {
        return ptr::null_mut();
    }
    let bytes = unsafe { CStr::from_ptr(name) }.to_bytes();
    match bytes {
        b"ggml_npu_timeline_enabled" => ggml_npu_timeline_enabled as *mut c_void,
        b"ggml_npu_timeline_now_ns" => ggml_npu_timeline_now_ns as *mut c_void,
        b"ggml_npu_timeline_span" => ggml_npu_timeline_span as *mut c_void,
        b"ggml_npu_timeline_instant" => ggml_npu_timeline_instant as *mut c_void,
        _ => ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline_ring::{Domain, Event, Kind};

    #[test]
    fn summarize_busy_idle_and_copy() {
        let events = [
            Event {
                ts_ns: 1_000_000,
                dur_ns: 100_000_000, // 100 ms forward window marker (not busy)
                domain: Domain::Cpu as u8,
                kind: Kind::Forward as u8,
                ..Event::default()
            },
            Event {
                ts_ns: 10_000_000,
                dur_ns: 40_000_000, // GPU busy 40 ms
                domain: Domain::Gpu as u8,
                kind: Kind::GraphCompute as u8,
                ..Event::default()
            },
            Event {
                ts_ns: 20_000_000,
                dur_ns: 30_000_000, // NPU wait 30 ms
                domain: Domain::Npu as u8,
                kind: Kind::Wait as u8,
                ..Event::default()
            },
            Event {
                ts_ns: 15_000_000,
                dur_ns: 5_000_000, // host stage-in on CPU
                bytes: 4096,
                domain: Domain::Cpu as u8,
                kind: Kind::StageIn as u8,
                ..Event::default()
            },
            Event {
                ts_ns: 55_000_000,
                dur_ns: 20_000_000, // Meta synchronize (TP fence reclaim)
                domain: Domain::Meta as u8,
                kind: Kind::Synchronize as u8,
                ..Event::default()
            },
            Event {
                ts_ns: 50_000_000,
                dur_ns: 0,
                bytes: 8192,
                domain: Domain::Gpu as u8,
                kind: Kind::MemLoad as u8,
                ..Event::default()
            },
        ];
        let s = summarize(&events, 0);
        assert!(s.contains("gpu:"));
        assert!(s.contains("npu:"));
        assert!(s.contains("meta:"));
        assert!(s.contains("load_bytes=8192") || s.contains("load_bytes=12288"));
        assert!(s.contains("busy%="));
        assert!(s.contains("idle%="));
        // Host prep is CPU copy_/busy, not NPU; Meta fence wait is its own lane.
        let cpu_line = s.lines().find(|l| l.trim_start().starts_with("cpu:")).unwrap();
        let npu_line = s.lines().find(|l| l.trim_start().starts_with("npu:")).unwrap();
        let meta_line = s.lines().find(|l| l.trim_start().starts_with("meta:")).unwrap();
        assert!(cpu_line.contains("copy_ms=5.000"));
        assert!(npu_line.contains("busy_ms=30.000"));
        assert!(meta_line.contains("busy_ms=20.000"));
    }

    #[test]
    fn forward_is_not_busy_kind() {
        assert!(!is_busy_kind(Kind::Forward as u8));
        assert!(is_busy_kind(Kind::StageIn as u8));
        assert!(is_busy_kind(Kind::Wait as u8));
        assert!(is_busy_kind(Kind::Synchronize as u8));
    }

    #[test]
    fn device_engine_kinds_named_and_not_busy() {
        assert_eq!(kind_name(Kind::NpuCompute as u8), "npu_compute");
        assert_eq!(kind_name(Kind::NpuDma as u8), "npu_dma");
        assert_eq!(kind_name(Kind::NpuSw as u8), "npu_sw");
        // Per-engine device times overlap; they must not drive the merged busy% for the domain.
        assert!(!is_busy_kind(Kind::NpuCompute as u8));
        assert!(!is_busy_kind(Kind::NpuDma as u8));
        assert!(!is_busy_kind(Kind::NpuSw as u8));
    }

    #[test]
    fn summarize_reports_device_engine_breakdown() {
        let events = [
            Event {
                ts_ns: 10_000_000,
                dur_ns: 50_000_000, // host-observed NPU wait window: 50 ms busy
                domain: Domain::Npu as u8,
                kind: Kind::Wait as u8,
                ..Event::default()
            },
            Event {
                ts_ns: 10_000_000,
                dur_ns: 30_000_000, // DPU compute 30 ms (device-reported, nested in Wait)
                domain: Domain::Npu as u8,
                kind: Kind::NpuCompute as u8,
                ..Event::default()
            },
            Event {
                ts_ns: 10_000_000,
                dur_ns: 40_000_000, // DMA 40 ms (overlaps compute)
                domain: Domain::Npu as u8,
                kind: Kind::NpuDma as u8,
                ..Event::default()
            },
            Event {
                ts_ns: 10_000_000,
                dur_ns: 5_000_000, // SHAVE/software 5 ms
                domain: Domain::Npu as u8,
                kind: Kind::NpuSw as u8,
                ..Event::default()
            },
        ];
        let s = summarize(&events, 0);
        let npu_line = s.lines().find(|l| l.trim_start().starts_with("npu:")).unwrap();
        // Device engine columns present with the expected durations.
        assert!(npu_line.contains("compute_ms=30.000"), "{npu_line}");
        assert!(npu_line.contains("dma_ms=40.000"), "{npu_line}");
        assert!(npu_line.contains("sw_ms=5.000"), "{npu_line}");
        // Overlapping engine times must NOT inflate busy% past the host Wait window (50 ms).
        assert!(npu_line.contains("busy_ms=50.000"), "{npu_line}");
    }

    #[test]
    fn domain_abi_accepts_meta() {
        assert_eq!(domain_from_c(3), Some(Domain::Meta));
        assert_eq!(domain_from_c(4), None);
    }

    #[test]
    fn proc_address_resolves_abi() {
        let p = unsafe { proc_address(c"ggml_npu_timeline_span".as_ptr()) };
        assert!(!p.is_null());
        let p2 = unsafe { proc_address(c"ggml_npu_timeline_enabled".as_ptr()) };
        assert!(!p2.is_null());
        let miss = unsafe { proc_address(c"nope".as_ptr()) };
        assert!(miss.is_null());
    }

    #[test]
    fn merge_overlapping_busy_intervals() {
        let mut iv = alloc::vec![(0, 10), (5, 15), (20, 25)];
        assert_eq!(merge_busy(&mut iv), 20); // [0,15) + [20,25)
    }
}
