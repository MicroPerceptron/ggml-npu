//! os0 scheduler telemetry ring — the single source of truth for the wire format shared by the
//! `ggml-npu` producer (the `MetricsSink`) and any consumer (the wasmrt host's `host_get_telemetry`,
//! an `nputop`-style TUI, etc.).
//!
//! The ring core below is `core`-only (no `std`, no `alloc`, no locks — just `AtomicU64` + a
//! per-slot seqlock), so this file drops into the `ggml-npu` `no_std` cdylib unchanged. The
//! producer writes fixed 32-byte records into a host-visible shared region (a `/dev/shm` mmap
//! today; the same `zeMemAllocHost`/dmabuf buffer later); a separate process maps the same region
//! and reads lock-free. The producer never blocks on the reader — if the reader falls behind it is
//! lapped (bounded staleness), never back-pressured, so telemetry can never perturb the hot path.

use core::sync::atomic::{AtomicU64, Ordering::*};

/// One telemetry event. `#[repr(C)]`, fixed 32 bytes, POD — safe to memcpy across the process
/// boundary. Field meaning is keyed by `kind` (see [`Kind`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Record {
    pub kind: u16,   // Kind
    pub backend: u8, // Backend
    pub flags: u8,   // bit0 = unitary program
    pub a: u32,      // kind-specific (nodes / op / submit_ns / cache_id …)
    pub b: u32,      // kind-specific (reason / wait_ns / hit-miss …)
    pub c: u32,      // kind-specific (wire op / bytes …)
    pub t_ns: u64,   // duration or volume (ns / bytes)
    pub seq: u64,    // monotonic record id (producer-assigned)
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    NpuProgram = 1,
    PlacementRejection = 2,
    Dispatch = 3,
    Cache = 4,
    GraphPartition = 5,
    Memory = 6,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Npu = 0,
    Gpu = 1,
    Cpu = 2,
    Unknown = 255,
}

/// Placement/claim reasons, mapped from the string reasons the engine already emits
/// (`recognized_subgraph`, `stateful_whole_graph`, `per_node`, `glue_fallback`,
/// `unsupported_shape`, `not_offloadable`, `glue_below_threshold`). Stable small ints.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    RecognizedSubgraph = 0,
    StatefulWholeGraph = 1,
    PerNode = 2,
    GlueFallback = 3,
    UnsupportedShape = 4,
    NotOffloadable = 5,
    GlueBelowThreshold = 6,
    Other = 255,
}

const MAGIC: u64 = 0x30_73_30_54_454c_454d; // "0s0TELEM"-ish
const SLOT: usize = 64; // one cache line: 8-byte seqlock + 32-byte record + pad
const HEADER: usize = 64;

/// Shared-region header (first 64 bytes). `head` is the total records ever written (SPSC: only the
/// producer stores it). `cap` is a power of two.
#[repr(C)]
struct Header {
    magic: u64,
    version: u32,
    slot_size: u32,
    cap: u64,
    _pad: u64,
    head: AtomicU64,
}

/// Total bytes for a ring of `cap` slots (cap must be a power of two).
pub const fn region_bytes(cap: usize) -> usize {
    HEADER + cap * SLOT
}

#[inline]
unsafe fn header<'a>(base: *mut u8) -> &'a Header {
    &*(base as *const Header)
}
#[inline]
unsafe fn slot_seq<'a>(base: *mut u8, i: u64, mask: u64) -> &'a AtomicU64 {
    &*(base.add(HEADER + (i & mask) as usize * SLOT) as *const AtomicU64)
}
#[inline]
unsafe fn slot_rec(base: *mut u8, i: u64, mask: u64) -> *mut Record {
    base.add(HEADER + (i & mask) as usize * SLOT + 8) as *mut Record
}

/// Producer handle. One per process (SPSC). `base` points at a shared region of
/// [`region_bytes(cap)`] bytes.
pub struct Writer {
    base: *mut u8,
    mask: u64,
}
unsafe impl Send for Writer {}

impl Writer {
    /// Initialize a fresh region and return a writer. `cap` must be a power of two.
    ///
    /// # Safety
    /// `base` must point at `region_bytes(cap)` writable, shared bytes that outlive the writer.
    pub unsafe fn create(base: *mut u8, cap: usize) -> Self {
        assert!(cap.is_power_of_two());
        let h = &mut *(base as *mut Header);
        h.magic = MAGIC;
        h.version = 1;
        h.slot_size = SLOT as u32;
        h.cap = cap as u64;
        h._pad = 0;
        h.head = AtomicU64::new(0);
        Self {
            base,
            mask: cap as u64 - 1,
        }
    }

    /// Append one record. Wait-free: a couple of `Release` stores + a 32-byte copy. Never blocks
    /// on the reader. Measured cost is tens of ns — a small fraction of a single NPU dispatch.
    #[inline]
    pub fn push(&self, mut rec: Record) {
        unsafe {
            let h = header(self.base);
            let n = h.head.load(Relaxed);
            rec.seq = n;
            let seq = slot_seq(self.base, n, self.mask);
            let s = seq.load(Relaxed);
            seq.store(s + 1, Release); // odd => writing
            core::ptr::write_volatile(slot_rec(self.base, n, self.mask), rec);
            seq.store(s + 2, Release); // even => committed
            h.head.store(n + 1, Release);
        }
    }
}

/// Consumer handle. Holds only a read cursor; the shared region is borrowed per call so the same
/// `Reader` logic serves an in-process poll, a separate process, or the wasmrt `host_get_telemetry`
/// import. Returns records newest-work-preserving; reports how many were dropped to lapping.
pub struct Reader {
    cursor: u64,
}
impl Default for Reader {
    fn default() -> Self {
        Self { cursor: 0 }
    }
}

pub struct Drain {
    pub read: usize,
    pub lapped: u64,
    pub torn: u64,
}

impl Reader {
    pub fn new() -> Self {
        Self { cursor: 0 }
    }

    /// Drain newly-committed records into `out`, up to its length. `base` maps the same region the
    /// producer wrote. Skips (and counts) records the producer already overwrote.
    ///
    /// # Safety
    /// `base` must map a region initialized by [`Writer::create`] with the same `cap`.
    pub unsafe fn drain(&mut self, base: *mut u8, out: &mut [Record]) -> Drain {
        let h = header(base);
        debug_assert_eq!(h.magic, MAGIC, "not an os0-telemetry region");
        let cap = h.cap;
        let mask = cap - 1;
        let head = h.head.load(Acquire);
        let mut d = Drain {
            read: 0,
            lapped: 0,
            torn: 0,
        };
        if head.wrapping_sub(self.cursor) > cap {
            // producer lapped us
            let behind = head - self.cursor - cap;
            d.lapped = behind;
            self.cursor = head - cap; // resync to oldest still-live slot
        }
        while self.cursor < head && d.read < out.len() {
            let seq = slot_seq(base, self.cursor, mask);
            loop {
                let s1 = seq.load(Acquire);
                if s1 & 1 == 1 {
                    core::hint::spin_loop();
                    continue;
                } // mid-write
                let rec = core::ptr::read_volatile(slot_rec(base, self.cursor, mask));
                core::sync::atomic::fence(Acquire);
                if seq.load(Acquire) == s1 {
                    // stable snapshot
                    if rec.seq == self.cursor {
                        out[d.read] = rec;
                        d.read += 1;
                    } else {
                        d.torn += 1;
                    } // lapped mid-read; skip
                    break;
                }
            }
            self.cursor += 1;
        }
        d
    }
}

impl Writer {
    /// Attach a writer to an already-initialized region (does not touch the header).
    ///
    /// # Safety
    /// `base` must map a region previously set up by [`Writer::create`] with the same `cap`.
    pub unsafe fn from_base(base: *mut u8, cap: usize) -> Self {
        debug_assert!(cap.is_power_of_two());
        Self {
            base,
            mask: cap as u64 - 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicBool, Ordering};

    fn mk(a: u32, c: u32) -> Record {
        Record {
            kind: Kind::NpuProgram as u16,
            backend: 0,
            flags: 0,
            a,
            b: 0,
            c,
            t_ns: 0,
            seq: 0,
        }
    }

    #[test]
    fn roundtrip_in_order() {
        const CAP: usize = 64;
        let mut buf = [0u8; region_bytes(CAP)];
        let base = buf.as_mut_ptr();
        let w = unsafe { Writer::create(base, CAP) };
        for i in 0..40u32 {
            w.push(mk(i, 0));
        }
        let mut r = Reader::new();
        let mut out = [Record::default(); 128];
        let d = unsafe { r.drain(base, &mut out) };
        assert_eq!(d.read, 40);
        assert_eq!(d.lapped, 0);
        assert_eq!(d.torn, 0);
        for i in 0..40 {
            assert_eq!(out[i].a, i as u32);
            assert_eq!(out[i].seq, i as u64);
        }
    }

    #[test]
    fn lapping_reported_never_blocks() {
        const CAP: usize = 8;
        let mut buf = [0u8; region_bytes(CAP)];
        let base = buf.as_mut_ptr();
        let w = unsafe { Writer::create(base, CAP) };
        // Far more than cap, with no reader draining: must not block or panic.
        for i in 0..1000u32 {
            w.push(mk(i, 0));
        }
        let mut r = Reader::new();
        let mut out = [Record::default(); 16];
        let d = unsafe { r.drain(base, &mut out) };
        assert!(d.lapped >= 1000 - CAP as u64); // reader was lapped, and it's reported
        assert_eq!(d.torn, 0);
        assert!(d.read <= CAP); // only the live window survives
    }

    #[test]
    fn seqlock_no_tear_under_contention() {
        use std::sync::Arc;
        use std::thread;
        const CAP: usize = 256;
        const TOTAL: u64 = 150_000;
        let mut buf = [0u8; region_bytes(CAP)];
        let base = buf.as_mut_ptr();
        let _init = unsafe { Writer::create(base, CAP) }; // initialize the header
        let addr = base as usize; // share by address; buf outlives the joined threads
        let done = Arc::new(AtomicBool::new(false));

        let dw = done.clone();
        let producer = thread::spawn(move || {
            let w = unsafe { Writer::from_base(addr as *mut u8, CAP) };
            for i in 0..TOTAL {
                let a = (i & 0xffff) as u32;
                let c = (i.wrapping_mul(2_654_435_761) & 0xffff) as u32;
                let mut rec = mk(a, c);
                // t_ns carries a checksum of the fields keyed by the record's seq (== i), so a torn
                // (mixed) snapshot would fail the recomputed checksum on the reader side.
                rec.t_ns = a as u64 ^ ((c as u64) << 20) ^ i.rotate_left(7);
                w.push(rec);
            }
            dw.store(true, Ordering::Release);
        });

        let dr = done.clone();
        let consumer = thread::spawn(move || {
            let mut r = Reader::new();
            let mut out = [Record::default(); 512];
            let (mut accepted, mut tears) = (0u64, 0u64);
            loop {
                let d = unsafe { r.drain(addr as *mut u8, &mut out) };
                for rec in &out[..d.read] {
                    accepted += 1;
                    let expect = rec.a as u64 ^ ((rec.c as u64) << 20) ^ rec.seq.rotate_left(7);
                    if rec.t_ns != expect {
                        tears += 1;
                    }
                }
                if d.read == 0 && dr.load(Ordering::Acquire) {
                    break;
                }
            }
            (accepted, tears)
        });

        producer.join().unwrap();
        let (accepted, tears) = consumer.join().unwrap();
        assert_eq!(tears, 0, "seqlock let a torn snapshot through");
        assert!(accepted > 0, "consumer never observed any records");
    }
}
