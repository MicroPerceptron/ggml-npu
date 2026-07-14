//! Lock-free SPSC ring for hetero forward-pass timeline events.
//!
//! `core`-only (no `std` / `alloc` / locks). The producer pushes fixed POD records; a consumer
//! drains without back-pressuring the hot path. A lagging reader is lapped (bounded staleness).

use core::sync::atomic::{AtomicU64, Ordering};

/// Absolute monotonic timestamp + optional duration (Chrome `ph:"X"` complete event when
/// `dur_ns > 0`; `dur_ns == 0` is an instant / marker).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Event {
    pub ts_ns: u64,
    pub dur_ns: u64,
    pub bytes: u64,
    pub domain: u8,
    pub kind: u8,
    pub flags: u8,
    pub _pad: u8,
    pub seq: u32,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    Cpu = 0,
    Gpu = 1,
    Npu = 2,
    /// Tensor-parallel Meta scheduler facade (not host CPU math).
    Meta = 3,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Forward = 1,
    GraphCompute = 2,
    StageIn = 3,
    Submit = 4,
    Wait = 5,
    CopyOut = 6,
    SchedCopy = 7,
    Synchronize = 8,
    MemLoad = 9,
    /// Device-reported per-engine breakdown of an NPU dispatch (Level Zero graph profiling):
    /// DPU compute, DMA (memory streaming), and SHAVE/software time. Per-engine busy times that
    /// can overlap, so they are informational accumulators rather than merged busy intervals.
    NpuCompute = 10,
    NpuDma = 11,
    NpuSw = 12,
}

const MAGIC: u64 = 0x4845_5445_524F_544C; // "HETEROTL"
const SLOT: usize = 64;
const HEADER: usize = 64;

#[repr(C)]
struct Header {
    magic: u64,
    version: u32,
    slot_size: u32,
    cap: u64,
    _pad: u64,
    head: AtomicU64,
}

/// Bytes for a ring of `cap` slots (`cap` must be a power of two).
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
unsafe fn slot_rec(base: *mut u8, i: u64, mask: u64) -> *mut Event {
    base.add(HEADER + (i & mask) as usize * SLOT + 8) as *mut Event
}

/// Producer. One writer per region (SPSC).
pub struct Writer {
    base: *mut u8,
    mask: u64,
}

unsafe impl Send for Writer {}

impl Writer {
    /// Initialize a fresh region.
    ///
    /// # Safety
    /// `base` must point at `region_bytes(cap)` writable bytes that outlive the writer.
    pub unsafe fn create(base: *mut u8, cap: usize) -> Self {
        assert!(cap.is_power_of_two());
        let h = &mut *(base as *mut Header);
        h.magic = MAGIC;
        h.version = 1;
        h.slot_size = SLOT as u32;
        h.cap = cap as u64;
        h._pad = 0;
        h.head = AtomicU64::new(0);
        for i in 0..cap {
            slot_seq(base, i as u64, (cap as u64) - 1).store(0, Ordering::Relaxed);
        }
        Self {
            base,
            mask: (cap as u64) - 1,
        }
    }

    /// Reattach to an already-initialized region.
    ///
    /// # Safety
    /// `base` must be a region previously initialized with the same `cap`.
    pub unsafe fn from_base(base: *mut u8, cap: usize) -> Self {
        debug_assert!(cap.is_power_of_two());
        Self {
            base,
            mask: (cap as u64) - 1,
        }
    }

    #[inline]
    pub fn push(&self, mut event: Event) {
        let h = unsafe { header(self.base) };
        let i = h.head.fetch_add(1, Ordering::Relaxed);
        event.seq = i as u32;
        let seq = unsafe { slot_seq(self.base, i, self.mask) };
        // Odd = write in progress (seqlock).
        seq.store(i.wrapping_mul(2).wrapping_add(1), Ordering::Release);
        unsafe {
            core::ptr::write(slot_rec(self.base, i, self.mask), event);
        }
        seq.store(i.wrapping_mul(2).wrapping_add(2), Ordering::Release);
    }

    #[allow(dead_code)]
    pub fn head(&self) -> u64 {
        unsafe { header(self.base) }.head.load(Ordering::Acquire)
    }
}

/// Consumer snapshot of drained events.
#[derive(Clone, Copy, Debug, Default)]
pub struct DrainStats {
    pub lapped: u64,
    pub drained: u64,
}

/// Reader. Independent of the writer (same region).
pub struct Reader {
    base: *mut u8,
    mask: u64,
    next: u64,
}

unsafe impl Send for Reader {}

impl Reader {
    /// # Safety
    /// `base` must be a valid initialized region with the given `cap`.
    pub unsafe fn from_base(base: *mut u8, cap: usize) -> Self {
        debug_assert!(cap.is_power_of_two());
        Self {
            base,
            mask: (cap as u64) - 1,
            next: 0,
        }
    }

    /// Drain into `out`, advancing the reader cursor. Returns how many events were copied and
    /// whether the producer lapped this reader (lost events).
    pub fn drain(&mut self, out: &mut [Event]) -> DrainStats {
        let head = unsafe { header(self.base) }.head.load(Ordering::Acquire);
        let cap = self.mask + 1;
        let mut lapped = 0u64;
        if head.saturating_sub(self.next) > cap {
            lapped = head.saturating_sub(self.next + cap);
            self.next = head.saturating_sub(cap);
        }
        let mut drained = 0u64;
        while self.next < head && (drained as usize) < out.len() {
            let i = self.next;
            let seq = unsafe { slot_seq(self.base, i, self.mask) };
            let s1 = seq.load(Ordering::Acquire);
            if s1 == 0 || (s1 & 1) != 0 {
                // Incomplete / empty slot — stop; producer may still be writing.
                break;
            }
            let rec = unsafe { core::ptr::read(slot_rec(self.base, i, self.mask)) };
            let s2 = seq.load(Ordering::Acquire);
            if s1 != s2 || (s2 & 1) != 0 {
                break;
            }
            out[drained as usize] = rec;
            drained += 1;
            self.next = i + 1;
        }
        DrainStats { lapped, drained }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn region(cap: usize) -> Vec<u8> {
        vec![0u8; region_bytes(cap)]
    }

    #[test]
    fn push_drain_round_trip() {
        let mut buf = region(8);
        let w = unsafe { Writer::create(buf.as_mut_ptr(), 8) };
        w.push(Event {
            ts_ns: 100,
            dur_ns: 10,
            bytes: 64,
            domain: Domain::Npu as u8,
            kind: Kind::Wait as u8,
            ..Event::default()
        });
        w.push(Event {
            ts_ns: 200,
            dur_ns: 0,
            bytes: 128,
            domain: Domain::Gpu as u8,
            kind: Kind::MemLoad as u8,
            ..Event::default()
        });
        let mut r = unsafe { Reader::from_base(buf.as_mut_ptr(), 8) };
        let mut out = [Event::default(); 8];
        let stats = r.drain(&mut out);
        assert_eq!(stats.drained, 2);
        assert_eq!(stats.lapped, 0);
        assert_eq!(out[0].ts_ns, 100);
        assert_eq!(out[0].kind, Kind::Wait as u8);
        assert_eq!(out[1].bytes, 128);
        assert_eq!(out[1].domain, Domain::Gpu as u8);
    }

    #[test]
    fn reader_reports_lap_when_overrun() {
        let mut buf = region(4);
        let w = unsafe { Writer::create(buf.as_mut_ptr(), 4) };
        for i in 0..10u64 {
            w.push(Event {
                ts_ns: i,
                dur_ns: 1,
                bytes: 0,
                domain: Domain::Cpu as u8,
                kind: Kind::GraphCompute as u8,
                ..Event::default()
            });
        }
        let mut r = unsafe { Reader::from_base(buf.as_mut_ptr(), 4) };
        let mut out = [Event::default(); 8];
        let stats = r.drain(&mut out);
        assert!(stats.lapped > 0);
        assert!(stats.drained <= 4);
        // Surviving events are the newest ring window.
        assert!(out[0].ts_ns >= 6);
    }

    #[test]
    fn region_bytes_is_header_plus_slots() {
        assert_eq!(region_bytes(16), 64 + 16 * 64);
    }
}
